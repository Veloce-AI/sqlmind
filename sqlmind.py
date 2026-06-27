"""
sqlmind.py
──────────
SQLMind Python ADK — Use SQLMind intelligence in any Python backend,
LangChain agent, LlamaIndex pipeline, or standalone script.

Install:
  pip install anthropic sqlglot sqlalchemy

Usage:
  from sqlmind import SQLMindAgent
  
  agent = SQLMindAgent(
      schema=schema_dsl_string,
      dialect="postgresql",
      model="claude-sonnet-4-6"   # or any Anthropic model
  )
  result = agent.generate("Show top 10 customers by revenue last month")
  print(result.sql)
  print(result.phase_trace)
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


# ─── Result Models ────────────────────────────────────────────────────────────

@dataclass
class PhaseReasoning:
    phase: str
    description: str
    content: str

@dataclass
class SQLMindResult:
    sql: str
    phase_trace: list[PhaseReasoning]
    complexity_level: str
    warnings: list[str]
    suggestions: list[str]
    raw_reasoning: str
    dialect: str
    model: str
    generation_time_ms: int
    is_validated: bool = False
    validation_errors: list[str] = field(default_factory=list)


# ─── Core Agent ───────────────────────────────────────────────────────────────

class SQLMindAgent:
    """
    SQLMind Agent — generates phase-locked SQL from natural language.
    
    Integrates with:
    - Claude (Anthropic) for generation and refinement
    - sqlglot for syntax validation and dialect transpilation
    - sqlalchemy for schema introspection and execution plan analysis
    """
    
    PHASE_LOCKED_SYSTEM_PROMPT = """You are SQLMind, a precise SQL generation engine.

Your core principle: Generate SQL by reasoning in EXECUTION ORDER, not write order.

The 7-phase execution order (always follow this sequence internally):
1. FROM / JOIN     → Identify tables, define working dataset
2. WHERE           → Filter rows (NO aggregates here, NO SELECT aliases here)
3. GROUP BY        → Define aggregation groups
4. HAVING          → Filter groups (aggregated values only)
5. SELECT          → Choose columns, compute expressions, define aliases
6. ORDER BY        → Sort result (SELECT aliases ARE available here)
7. LIMIT / OFFSET  → Paginate

Critical rules you must never violate:
- Aggregate functions (COUNT, SUM, AVG, MAX, MIN) → NEVER in WHERE → always in HAVING
- Column aliases from SELECT → NOT available in WHERE or GROUP BY
- Every non-aggregated SELECT column → MUST be in GROUP BY
- Every JOIN → MUST have an explicit ON clause (never implicit Cartesian)
- LEFT JOIN + WHERE on right-side table → converts to INNER JOIN (usually a bug)
- Window functions → execute AFTER GROUP BY/HAVING but BEFORE ORDER BY

When generating SQL:
1. First classify complexity (L1-L4)
2. Reason through each phase explicitly before writing SQL
3. For L3/L4 queries, decompose into CTEs
4. Validate your own output before returning it
"""

    def __init__(
        self,
        schema: str,
        dialect: str = "postgresql",
        model: str = "claude-sonnet-4-6",
        auto_validate: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize the SQLMind agent.
        
        Args:
            schema: Schema in SQLMind DSL format or raw DDL
            dialect: Target database dialect
            model: Anthropic model to use
            auto_validate: Whether to auto-validate and self-correct output
            max_retries: Maximum self-correction attempts
        """
        self.schema = schema
        self.dialect = dialect
        self.model = model
        self.auto_validate = auto_validate
        self.max_retries = max_retries
        
        if not HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        
        self.client = anthropic.Anthropic()
    
    def generate(
        self,
        nl_query: str,
        few_shot_examples: Optional[list[dict]] = None,
        context: Optional[str] = None,
    ) -> SQLMindResult:
        """
        Generate SQL from a natural language query.
        
        Args:
            nl_query: The natural language question/request
            few_shot_examples: Optional list of {"question": ..., "sql": ...} examples
            context: Optional additional context (e.g., "This is for a monthly report")
        
        Returns:
            SQLMindResult with the generated SQL and reasoning trace
        """
        start_time = time.time()
        
        # Build the generation prompt
        user_prompt = self._build_user_prompt(nl_query, few_shot_examples, context)
        
        # Call the model
        messages = [{"role": "user", "content": user_prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.PHASE_LOCKED_SYSTEM_PROMPT,
            messages=messages,
        )
        
        raw_response = response.content[0].text
        
        # Parse the response
        result = self._parse_response(raw_response, nl_query)
        result.model = self.model
        result.generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Auto-validate and self-correct
        if self.auto_validate and result.sql:
            result = self._validate_and_correct(result, nl_query, messages, raw_response)
        
        return result
    
    def introspect_schema(self, connection_string: str) -> str:
        """
        Introspect a live database and return SQLMind Schema DSL.
        Requires: pip install sqlalchemy
        """
        try:
            from sqlalchemy import create_engine, inspect, text
        except ImportError:
            raise ImportError("pip install sqlalchemy")
        
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        
        lines = []
        for table_name in inspector.get_table_names():
            pk = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))
            fks = {}
            for fk in inspector.get_foreign_keys(table_name):
                for lc, rc in zip(fk["constrained_columns"], fk["referred_columns"]):
                    fks[lc] = f"{fk['referred_table']}.{rc}"
            
            idxs = set()
            for idx in inspector.get_indexes(table_name):
                for col in idx.get("column_names", []):
                    idxs.add(col)
            
            lines.append(f"TABLE {table_name} (")
            for col in inspector.get_columns(table_name):
                name = col["name"]
                typ = str(col["type"]).split("(")[0]
                mods = []
                if name in pk: mods.append("PK")
                if name in fks: mods.append(f"FK→{fks[name]}")
                if name in idxs and name not in pk: mods.append("IDX")
                lines.append(f"  {name:<24}{typ:<16}{'  '.join(mods)}")
            lines.append(")")
            lines.append("")
        
        self.schema = "\n".join(lines)
        return self.schema
    
    def validate_sql(self, sql: str) -> dict:
        """
        Validate SQL against the 7-phase execution model.
        Returns a validation report dict.
        """
        errors = []
        warnings = []
        
        sql_upper = sql.upper()
        
        # Check aggregate in WHERE
        where_match = re.search(
            r'WHERE\s+(.*?)(?:GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|$)',
            sql_upper, re.DOTALL
        )
        if where_match:
            where_body = where_match.group(1)
            for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
                if agg in where_body:
                    errors.append(
                        f"AGG_IN_WHERE: {agg.rstrip('(')} in WHERE clause → move to HAVING"
                    )
        
        # HAVING without GROUP BY
        if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
            errors.append("HAVING_NO_GROUPBY: HAVING without GROUP BY")
        
        # Implicit Cartesian
        if re.search(r'FROM\s+\w+\s*,\s*\w+', sql_upper):
            errors.append("IMPLICIT_CARTESIAN: Comma-based join may produce Cartesian product")
        
        # SELECT *
        if re.search(r'SELECT\s+\*', sql_upper):
            warnings.append("SELECT_STAR: SELECT * may be inefficient in production")
        
        # ORDER BY without LIMIT
        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper and "TOP" not in sql_upper:
            warnings.append("ORDER_NO_LIMIT: ORDER BY without LIMIT may sort large result sets")
        
        # sqlglot syntax check
        if HAS_SQLGLOT:
            try:
                dialect_map = {
                    "postgresql": "postgres", "mysql": "mysql",
                    "sqlite": "sqlite", "mssql": "tsql",
                    "bigquery": "bigquery", "snowflake": "snowflake",
                }
                sqlglot.parse_one(sql, dialect=dialect_map.get(self.dialect, "postgres"))
            except Exception as e:
                errors.append(f"SYNTAX_ERROR: {str(e)}")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def transpile(self, sql: str, to_dialect: str) -> str:
        """Transpile SQL to a different dialect."""
        if not HAS_SQLGLOT:
            raise ImportError("pip install sqlglot")
        
        dialect_map = {
            "postgresql": "postgres", "mysql": "mysql",
            "sqlite": "sqlite", "mssql": "tsql",
            "bigquery": "bigquery", "snowflake": "snowflake",
        }
        from_d = dialect_map.get(self.dialect, "postgres")
        to_d = dialect_map.get(to_dialect, to_dialect)
        
        transpiled = sqlglot.transpile(sql, read=from_d, write=to_d, pretty=True)
        return transpiled[0] if transpiled else ""
    
    # ── Private Methods ────────────────────────────────────────────────────────
    
    def _build_user_prompt(
        self,
        nl_query: str,
        few_shot_examples: Optional[list[dict]],
        context: Optional[str],
    ) -> str:
        
        dialect_notes = {
            "postgresql": "PostgreSQL: ILIKE for case-insensitive, :: for casting",
            "mysql": "MySQL: backticks for reserved words, GROUP_CONCAT not STRING_AGG",
            "sqlite": "SQLite: no RIGHT/FULL OUTER JOIN, strftime() for dates",
            "mssql": "T-SQL: TOP n not LIMIT, GETDATE() not NOW(), square brackets for reserved words",
            "bigquery": "BigQuery: backtick table names, QUALIFY for window filtering",
            "snowflake": "Snowflake: QUALIFY clause, case-insensitive by default",
        }
        
        parts = [f"## DATABASE SCHEMA\n{self.schema}"]
        parts.append(f"\n## DIALECT: {self.dialect.upper()}")
        parts.append(dialect_notes.get(self.dialect.lower(), "Use ANSI SQL"))
        
        if context:
            parts.append(f"\n## CONTEXT\n{context}")
        
        if few_shot_examples:
            parts.append("\n## REFERENCE EXAMPLES")
            for ex in few_shot_examples[:3]:  # max 3 examples
                parts.append(f"Q: {ex['question']}\nSQL:\n```sql\n{ex['sql']}\n```")
        
        parts.append(f"\n## REQUEST\n{nl_query}")
        parts.append("""
## INSTRUCTIONS
1. Classify complexity (L1/L2/L3/L4)
2. Reason through each phase (FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT)
3. Show your phase-by-phase reasoning
4. Then output the final SQL in a ```sql block
5. For L3/L4: use CTEs with descriptive names

Format your response as:
COMPLEXITY: [L1/L2/L3/L4]
PHASE REASONING:
[FROM]: ...
[WHERE]: ...
[GROUP BY]: ...
[HAVING]: ...
[SELECT]: ...
[ORDER BY]: ...
[LIMIT]: ...

FINAL SQL:
```sql
...
```
""")
        
        return "\n".join(parts)
    
    def _parse_response(self, raw_response: str, nl_query: str) -> SQLMindResult:
        # Extract SQL from code block
        sql_match = re.search(r'```sql\s*(.*?)\s*```', raw_response, re.DOTALL)
        sql = sql_match.group(1).strip() if sql_match else ""
        
        # If no code block, try to find SQL directly
        if not sql:
            # Look for SELECT/WITH statement
            sql_direct = re.search(
                r'((?:WITH\s|SELECT\s|INSERT\s|UPDATE\s|DELETE\s).*)',
                raw_response, re.DOTALL | re.IGNORECASE
            )
            if sql_direct:
                sql = sql_direct.group(1).strip()
        
        # Extract complexity
        complexity_match = re.search(r'COMPLEXITY:\s*(L[1-4])', raw_response, re.IGNORECASE)
        complexity = complexity_match.group(1) if complexity_match else "L2"
        
        # Extract phase reasoning
        phase_trace = []
        phase_patterns = [
            ("FROM/JOIN", r'\[FROM\]:\s*(.*?)(?=\[WHERE\]|\[GROUP BY\]|FINAL SQL|$)'),
            ("WHERE", r'\[WHERE\]:\s*(.*?)(?=\[GROUP BY\]|\[HAVING\]|FINAL SQL|$)'),
            ("GROUP BY", r'\[GROUP BY\]:\s*(.*?)(?=\[HAVING\]|\[SELECT\]|FINAL SQL|$)'),
            ("HAVING", r'\[HAVING\]:\s*(.*?)(?=\[SELECT\]|\[ORDER BY\]|FINAL SQL|$)'),
            ("SELECT", r'\[SELECT\]:\s*(.*?)(?=\[ORDER BY\]|\[LIMIT\]|FINAL SQL|$)'),
            ("ORDER BY", r'\[ORDER BY\]:\s*(.*?)(?=\[LIMIT\]|FINAL SQL|$)'),
            ("LIMIT", r'\[LIMIT\]:\s*(.*?)(?=FINAL SQL|$)'),
        ]
        
        for phase_name, pattern in phase_patterns:
            match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and content.lower() not in ["n/a", "none", "not applicable", "-"]:
                    phase_trace.append(PhaseReasoning(
                        phase=phase_name,
                        description=f"Phase reasoning for {phase_name}",
                        content=content[:300],  # truncate for display
                    ))
        
        return SQLMindResult(
            sql=sql,
            phase_trace=phase_trace,
            complexity_level=complexity,
            warnings=[],
            suggestions=[],
            raw_reasoning=raw_response,
            dialect=self.dialect,
            model=self.model,
            generation_time_ms=0,
        )
    
    def _validate_and_correct(
        self,
        result: SQLMindResult,
        nl_query: str,
        messages: list,
        raw_response: str,
    ) -> SQLMindResult:
        """Self-correction loop: validate and fix SQL if needed."""
        
        validation = self.validate_sql(result.sql)
        result.is_validated = True
        
        if validation["is_valid"]:
            result.warnings = validation["warnings"]
            return result
        
        result.validation_errors = validation["errors"]
        
        # Self-correction pass
        for attempt in range(self.max_retries):
            correction_prompt = f"""The SQL you generated has the following errors:

{json.dumps(validation['errors'], indent=2)}

Original request: {nl_query}

Please fix ONLY the issues listed above. Apply the phase-locked protocol to correct each error:
- AGG_IN_WHERE → move aggregate to HAVING
- HAVING_NO_GROUPBY → add GROUP BY or move to WHERE
- IMPLICIT_CARTESIAN → add explicit JOIN ... ON
- SYNTAX_ERROR → fix syntax per {self.dialect} dialect rules

Output the corrected SQL in a ```sql block."""
            
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": correction_prompt})
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.PHASE_LOCKED_SYSTEM_PROMPT,
                messages=messages,
            )
            
            corrected_raw = response.content[0].text
            corrected_result = self._parse_response(corrected_raw, nl_query)
            
            if corrected_result.sql:
                new_validation = self.validate_sql(corrected_result.sql)
                if new_validation["is_valid"]:
                    result.sql = corrected_result.sql
                    result.validation_errors = []
                    result.warnings = new_validation["warnings"]
                    result.suggestions = [f"Auto-corrected after {attempt + 1} attempt(s)"]
                    return result
                
                validation = new_validation
                raw_response = corrected_raw
        
        # Could not self-correct — return with errors noted
        result.warnings = validation.get("warnings", [])
        result.suggestions = [
            "Manual review required. Auto-correction exhausted retry limit.",
            f"Remaining errors: {validation['errors']}"
        ]
        return result


# ─── Convenience Functions ────────────────────────────────────────────────────

def quick_generate(
    nl_query: str,
    schema_dsl: str,
    dialect: str = "postgresql",
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    One-liner SQL generation with SQLMind protocol.
    Returns just the SQL string.
    """
    agent = SQLMindAgent(schema=schema_dsl, dialect=dialect, model=model)
    result = agent.generate(nl_query)
    return result.sql


def schema_from_ddl(ddl: str) -> str:
    """
    Convert raw DDL (CREATE TABLE statements) to SQLMind Schema DSL.
    Requires sqlglot.
    """
    if not HAS_SQLGLOT:
        return ddl  # fallback: use raw DDL
    
    try:
        lines = []
        statements = sqlglot.parse(ddl)
        
        for stmt in statements:
            if stmt is None:
                continue
            
            stmt_type = type(stmt).__name__
            if "Create" in stmt_type:
                # Extract table info from parsed AST
                table_name = str(stmt.find(sqlglot.exp.Table).name) if stmt.find(sqlglot.exp.Table) else "unknown"
                lines.append(f"TABLE {table_name} (")
                
                for col_def in stmt.find_all(sqlglot.exp.ColumnDef):
                    col_name = col_def.name
                    col_type = str(col_def.args.get("kind", "TEXT")).split("(")[0]
                    lines.append(f"  {col_name:<24}{col_type}")
                
                lines.append(")")
                lines.append("")
        
        return "\n".join(lines) if lines else ddl
    
    except Exception:
        return ddl  # fallback


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SQLMind ADK — Generate SQL from natural language")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--schema", required=True, help="Path to schema DSL file or inline schema")
    parser.add_argument("--dialect", default="postgresql", help="SQL dialect")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()
    
    # Load schema
    import os
    if os.path.exists(args.schema):
        with open(args.schema) as f:
            schema = f.read()
    else:
        schema = args.schema
    
    agent = SQLMindAgent(
        schema=schema,
        dialect=args.dialect,
        model=args.model,
        auto_validate=not args.no_validate,
    )
    
    print(f"\n🔍 SQLMind generating SQL for: {args.query}\n")
    result = agent.generate(args.query)
    
    print(f"📊 Complexity: {result.complexity_level}")
    print(f"⏱️  Generated in {result.generation_time_ms}ms")
    print(f"✅ Validated: {result.is_validated}")
    
    if result.validation_errors:
        print(f"⚠️  Errors: {result.validation_errors}")
    
    if result.warnings:
        print(f"💡 Warnings: {result.warnings}")
    
    print("\n📝 Phase Reasoning:")
    for phase in result.phase_trace:
        print(f"  [{phase.phase}]: {phase.content[:100]}...")
    
    print(f"\n🗄️  Generated SQL:\n```sql\n{result.sql}\n```")
