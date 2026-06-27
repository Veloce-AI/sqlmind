"""
sqlmind_mcp_server.py
─────────────────────
SQLMind MCP Server — gives any LLM agent (Claude Code, Cursor, ADK)
structured SQL intelligence tools.

Install:
  pip install fastmcp sqlglot anthropic sqlalchemy

Run (stdio mode for Claude Code / Cursor):
  python sqlmind_mcp_server.py

Run (HTTP mode for remote / backend use):
  python sqlmind_mcp_server.py --transport http --port 8765

Add to Claude Code:
  claude mcp add sqlmind --command python --args sqlmind_mcp_server.py
"""

import json
import argparse
import re
from typing import Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Make sqlmind_graph importable when server is run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from sqlmind_graph import SchemaGraph, DialectRegistry

try:
    import sqlglot
    import sqlglot.errors
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError("pip install fastmcp")

_DIALECTS_YAML = Path(__file__).parent / "dialects.yaml"

# ─── MCP Server Setup ────────────────────────────────────────────────────────

mcp = FastMCP(
    name="sqlmind",
    description=(
        "SQL Intelligence Tools for LLM agents. "
        "Provides phase-locked query generation, schema introspection, "
        "query validation, and execution-order reasoning."
    ),
)

# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class PhaseTrace:
    phase: str
    clause: str
    content: str
    warnings: list[str]

@dataclass
class ValidationReport:
    is_valid: bool
    phases: list[dict]
    errors: list[str]
    warnings: list[str]
    suggestions: list[str]

@dataclass
class SchemaColumn:
    name: str
    type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None  # "table.column"
    is_indexed: bool = False
    enum_values: Optional[list[str]] = None

@dataclass
class SchemaTable:
    name: str
    columns: list[SchemaColumn]
    row_count_estimate: Optional[int] = None

# ─── TOOL 1: Schema Introspection ────────────────────────────────────────────

@mcp.tool(
    description=(
        "Introspect a live database and return a SQLMind Schema DSL — "
        "a compact, token-efficient representation of all tables, columns, "
        "primary keys, foreign keys, and indexes. "
        "Supports: postgresql, mysql, sqlite, mssql. "
        "Pass the full connection string."
    )
)
def sqlmind_introspect(
    connection_string: str,
    include_row_counts: bool = False,
    tables_filter: Optional[str] = None,
) -> str:
    """
    Introspects a live database and returns its schema as SQLMind DSL.

    Args:
        connection_string: SQLAlchemy connection string
            e.g. "postgresql://user:pass@host:5432/db"
                 "sqlite:///mydb.db"
                 "mysql+pymysql://user:pass@host/db"
        include_row_counts: Whether to include estimated row counts per table
        tables_filter: Comma-separated table names to include (all if None)

    Returns:
        JSON with schema_dsl ready to inject into LLM context, plus graph stats
    """
    try:
        graph = SchemaGraph().load_from_db(
            connection_string,
            include_row_counts=include_row_counts,
        )
    except ImportError as e:
        return json.dumps({
            "error": str(e),
            "fallback": "Pass schema manually to sqlmind_validate or sqlmind_generate",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "hint": "Check connection string format. Example: postgresql://user:pass@localhost:5432/mydb",
        })

    if tables_filter:
        keep = {t.strip().lower() for t in tables_filter.split(",")}
        graph.tables = {k: v for k, v in graph.tables.items() if k in keep}
        graph._build_adjacency()

    stats = graph.stats()
    return json.dumps({
        "status": "success",
        "table_count": stats["tables"],
        "column_count": stats["total_columns"],
        "fk_edge_count": stats["fk_edges"],
        "schema_dsl": graph.to_dsl(),
        "usage": (
            "Pass schema_dsl to sqlmind_generate() or "
            "include it in the LLM prompt for schema-grounded SQL generation."
        ),
    }, indent=2)


# ─── TOOL 2: Schema Linking ───────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Given a natural language query and a schema (SQLMind DSL or JSON), "
        "performs schema linking: identifies which tables and columns are relevant, "
        "detects join paths, infers filters/groups/metrics, and classifies complexity. "
        "Run this BEFORE sqlmind_generate to improve accuracy."
    )
)
def sqlmind_link_schema(
    nl_query: str,
    schema_dsl: str,
    dialect: str = "postgresql",
) -> str:
    """
    Schema linking — maps a natural language question to schema elements.
    
    Returns a structured analysis that should be passed to sqlmind_generate
    as the schema_link_context argument.
    """
    # This tool uses the LLM itself (Anthropic API) to perform schema linking
    # In a real deployment this would call the model; here we provide the
    # structured prompt template that can be called externally.
    
    schema_link_prompt = f"""You are a SQL schema linking expert. 
Given the natural language query and database schema below, perform schema linking.

SCHEMA:
{schema_dsl}

NATURAL LANGUAGE QUERY:
{nl_query}

Respond with a JSON object containing:
{{
  "entities_detected": ["entity1", "entity2"],
  "tables_needed": [
    {{"table": "table_name", "reason": "why needed", "alias": "suggested_alias"}}
  ],
  "columns_needed": [
    {{"table": "table_name", "column": "col_name", "purpose": "filter|select|join|group|sort"}}
  ],
  "join_path": [
    {{"from": "table_a.col", "to": "table_b.col", "type": "INNER|LEFT|RIGHT"}}
  ],
  "filters": [
    {{"column": "table.col", "operator": "=|>|<|LIKE|IN|BETWEEN", "value": "inferred_value", "phase": "WHERE|HAVING"}}
  ],
  "aggregations": [
    {{"function": "COUNT|SUM|AVG|MAX|MIN", "column": "table.col", "alias": "output_name"}}
  ],
  "group_by": ["table.col"],
  "order_by": [{{"column": "col", "direction": "ASC|DESC"}}],
  "complexity": "L1|L2|L3|L4",
  "complexity_reason": "brief explanation",
  "recommended_strategy": "direct|phase-locked|decompose|cte-first"
}}"""
    
    return json.dumps({
        "status": "success",
        "schema_link_prompt": schema_link_prompt,
        "instructions": (
            "Pass this prompt to your LLM. "
            "The response JSON should be provided as schema_link_context "
            "to sqlmind_generate()."
        ),
        "tip": (
            "For Claude Code integration, you can ask Claude to run this "
            "schema linking step internally using the SKILL.md protocol."
        )
    }, indent=2)


# ─── TOOL 3: SQL Validation ───────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Validates a SQL query through the SQLMind 7-phase execution model. "
        "Checks for: wrong clause ordering, aggregate-in-WHERE errors, "
        "missing GROUP BY columns, alias-in-WHERE errors, Cartesian JOIN risks, "
        "LEFT JOIN nullification, and dialect-specific issues. "
        "Returns a structured validation report with errors, warnings, and fixes."
    )
)
def sqlmind_validate(
    sql: str,
    dialect: str = "postgresql",
    schema_dsl: Optional[str] = None,
) -> str:
    """
    Validates SQL against the SQLMind 7-phase execution model.
    
    Args:
        sql: The SQL query to validate
        dialect: Target database dialect
        schema_dsl: Optional schema for column-existence checks
    
    Returns:
        JSON validation report
    """
    errors = []
    warnings = []
    suggestions = []
    phases_found = []
    
    sql_upper = sql.upper()
    sql_lower = sql.lower()
    
    # ── Phase detection ──
    clause_order = [
        ("FROM", "Phase 1: FROM/JOIN"),
        ("WHERE", "Phase 2: WHERE"),
        ("GROUP BY", "Phase 3: GROUP BY"),
        ("HAVING", "Phase 4: HAVING"),
        ("SELECT", "Phase 5: SELECT"),
        ("ORDER BY", "Phase 6: ORDER BY"),
        ("LIMIT", "Phase 7: LIMIT/OFFSET"),
    ]
    
    detected_phases = []
    for clause, phase_name in clause_order:
        if clause in sql_upper:
            pos = sql_upper.find(clause)
            detected_phases.append((pos, phase_name, clause))
    
    detected_phases.sort(key=lambda x: x[0])
    phases_found = [p[1] for p in detected_phases]
    
    # ── Check 1: Aggregate in WHERE ──
    agg_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
    where_match = re.search(r'WHERE\s+(.*?)(?:GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|$)', 
                             sql_upper, re.DOTALL)
    if where_match:
        where_content = where_match.group(1)
        for agg in agg_functions:
            if agg in where_content:
                errors.append({
                    "code": "AGG_IN_WHERE",
                    "message": f"Aggregate function {agg.rstrip('(')} found in WHERE clause",
                    "fix": "Move the aggregate condition to HAVING clause. "
                           "WHERE filters rows; HAVING filters groups after aggregation.",
                    "example": "BAD: WHERE COUNT(*) > 5 | GOOD: HAVING COUNT(*) > 5"
                })
    
    # ── Check 2: HAVING without GROUP BY ──
    if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
        errors.append({
            "code": "HAVING_NO_GROUPBY",
            "message": "HAVING clause found without GROUP BY",
            "fix": "HAVING requires a GROUP BY clause. Add GROUP BY, or move condition to WHERE.",
        })
    
    # ── Check 3: ORDER BY without LIMIT on potentially large result ──
    if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper and "TOP" not in sql_upper:
        warnings.append({
            "code": "ORDER_NO_LIMIT",
            "message": "ORDER BY without LIMIT/TOP may sort large result sets inefficiently",
            "suggestion": "Add LIMIT if you only need top N results"
        })
    
    # ── Check 4: SELECT * usage ──
    if re.search(r'SELECT\s+\*', sql_upper):
        warnings.append({
            "code": "SELECT_STAR",
            "message": "SELECT * retrieves all columns — may be inefficient in production",
            "suggestion": "Enumerate only the columns you need"
        })
    
    # ── Check 5: Implicit JOIN (comma-based) ──
    # Detect "FROM table1, table2" pattern
    from_match = re.search(r'FROM\s+(\w+)\s*,\s*(\w+)', sql_upper)
    if from_match:
        errors.append({
            "code": "IMPLICIT_CARTESIAN_JOIN",
            "message": (
                f"Implicit JOIN detected between "
                f"{from_match.group(1)} and {from_match.group(2)}. "
                "This may produce a Cartesian product."
            ),
            "fix": "Use explicit JOIN ... ON syntax",
            "example": "GOOD: FROM table1 INNER JOIN table2 ON table1.id = table2.fk_id"
        })
    
    # ── Check 6: LEFT JOIN with WHERE on right-side column ──
    left_join_match = re.search(r'LEFT\s+(?:OUTER\s+)?JOIN\s+(\w+)', sql_upper)
    if left_join_match:
        right_table = left_join_match.group(1)
        # Check if WHERE references the right-table (simplified check)
        if where_match:
            where_body = where_match.group(1)
            # Look for right table alias or name followed by a dot
            if right_table.lower() in where_body.lower():
                warnings.append({
                    "code": "LEFT_JOIN_WHERE_NULLIFICATION",
                    "message": (
                        f"LEFT JOIN on {right_table} may be nullified by WHERE clause "
                        "that references the right-side table. "
                        "This converts the LEFT JOIN into an INNER JOIN."
                    ),
                    "fix": (
                        "Move the filter condition to the JOIN's ON clause, "
                        "or use a subquery to pre-filter."
                    ),
                    "example": (
                        "BAD: LEFT JOIN orders ON ... WHERE orders.status='active'\n"
                        "GOOD: LEFT JOIN orders ON ... AND orders.status='active'"
                    )
                })
    
    # ── Check 7: Subquery in WHERE that could use EXISTS ──
    if re.search(r'WHERE\s+\w+\s+IN\s*\(SELECT', sql_upper):
        suggestions.append({
            "code": "IN_TO_EXISTS",
            "message": "IN (SELECT ...) found — consider EXISTS for better performance",
            "suggestion": "EXISTS short-circuits after finding the first match; IN evaluates all rows"
        })
    
    # ── Check 8: sqlglot syntax validation ──
    syntax_valid = True
    syntax_error = None
    if HAS_SQLGLOT:
        try:
            dialect_map = {
                "postgresql": "postgres",
                "mysql": "mysql",
                "sqlite": "sqlite",
                "mssql": "tsql",
                "bigquery": "bigquery",
                "snowflake": "snowflake",
            }
            glot_dialect = dialect_map.get(dialect.lower(), "postgres")
            sqlglot.parse_one(sql, dialect=glot_dialect)
        except sqlglot.errors.ParseError as e:
            syntax_valid = False
            syntax_error = str(e)
            errors.append({
                "code": "SYNTAX_ERROR",
                "message": f"Syntax error: {syntax_error}",
                "fix": "Review the query structure against the target dialect's syntax rules"
            })
    
    # ── Check 9: Schema column validation via SchemaGraph ──
    if schema_dsl:
        try:
            sg = SchemaGraph().load_from_dsl(schema_dsl)
            col_errors = sg.validate_sql_columns(sql)
            for ce in col_errors:
                errors.append({
                    "code": "COLUMN_NOT_FOUND",
                    "message": f"Column '{ce['column']}' not found in table '{ce['table']}'",
                    "fix": f"Available columns: {', '.join(ce['available'])}",
                    "suggestion": ce.get("suggestion"),
                })
        except Exception:
            # DSL parse failure is non-fatal for validation
            pass
    
    # ── Build Report ──
    is_valid = len(errors) == 0
    
    report = {
        "is_valid": is_valid,
        "syntax_valid": syntax_valid,
        "phases_detected": phases_found,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "overall": "✅ VALID" if is_valid else f"❌ INVALID ({len(errors)} error(s))",
    }
    
    return json.dumps(report, indent=2)


# ─── TOOL 4: Phase-Locked SQL Generation Prompt Builder ──────────────────────

@mcp.tool(
    description=(
        "Builds an optimized, phase-locked prompt for SQL generation. "
        "Takes a natural language query, schema DSL, and optional schema-linking context. "
        "Returns a structured prompt that instructs the LLM to reason in SQL execution order "
        "(FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT) "
        "rather than write order, dramatically improving accuracy for complex queries."
    )
)
def sqlmind_build_prompt(
    nl_query: str,
    schema_dsl: str,
    dialect: str = "postgresql",
    schema_link_context: Optional[str] = None,
    complexity_hint: Optional[str] = None,
    few_shot_examples: Optional[str] = None,
) -> str:
    """
    Builds the phase-locked SQL generation prompt.
    
    Pass the returned prompt to your LLM (Claude, GPT, etc.) to get
    significantly better SQL for complex queries.
    """
    dialect_notes = {
        "postgresql": "Use standard PostgreSQL syntax. ILIKE for case-insensitive. :: for casting.",
        "mysql": "Use backticks for reserved words. GROUP_CONCAT instead of STRING_AGG. LIMIT x,y syntax.",
        "sqlite": "No RIGHT JOIN. No FULL OUTER JOIN. strftime() for dates. Very permissive GROUP BY.",
        "mssql": "Use TOP n instead of LIMIT. GETDATE() for current time. Square brackets for reserved words.",
        "bigquery": "Use backtick-quoted table names. QUALIFY for window filtering. ARRAY_AGG for arrays.",
        "snowflake": "Use QUALIFY for window filtering. Case-insensitive by default. FLATTEN for arrays.",
    }
    
    dialect_note = dialect_notes.get(dialect.lower(), "Use ANSI SQL. Note any dialect assumptions.")
    
    complexity = complexity_hint or "AUTO-DETECT"
    
    schema_link_section = ""
    if schema_link_context:
        schema_link_section = f"""
## SCHEMA LINKING ANALYSIS (Pre-computed)
{schema_link_context}

Use this analysis to guide your table/column selection.
"""
    
    few_shot_section = ""
    if few_shot_examples:
        few_shot_section = f"""
## EXAMPLE QUERIES (Reference)
{few_shot_examples}
"""
    
    prompt = f"""You are a precise SQL generation engine. Generate correct {dialect.upper()} SQL.

## DATABASE SCHEMA
{schema_dsl}

## DIALECT RULES
{dialect_note}
{schema_link_section}{few_shot_section}
## USER REQUEST
{nl_query}

## GENERATION PROTOCOL — FOLLOW EXACTLY

### Step 1: Classify Complexity
- L1 Simple: Single table, no aggregation → direct generation
- L2 Moderate: 2-3 tables, basic GROUP BY → phase-locked generation  
- L3 Complex: 4+ tables, nested subqueries, window functions → decompose into CTEs
- L4 Analytical: Self-joins, correlated subqueries → CTE-first scaffolding

Complexity: {complexity}

### Step 2: Reason in EXECUTION ORDER (not write order)

Think through each phase BEFORE writing SQL:

**[PHASE 1 — FROM/JOIN]**
- Tables needed: _____
- Join type (INNER/LEFT/RIGHT): _____  
- Join condition (table_a.col = table_b.col): _____
- Risk: Could this produce a Cartesian product? _____

**[PHASE 2 — WHERE]**
- Row-level filters: _____
- ⚠️ NO aggregates here. NO SELECT aliases here.
- Column names must be original names from schema.

**[PHASE 3 — GROUP BY]**
- Grouping columns: _____
- ⚠️ Every non-aggregated SELECT column must appear here.

**[PHASE 4 — HAVING]**
- Group-level filters (on aggregated values): _____
- ⚠️ Only here can you filter by COUNT(), SUM(), AVG(), etc.

**[PHASE 5 — SELECT]**
- Output columns: _____
- Aggregate functions: _____
- Column aliases defined here.

**[PHASE 6 — ORDER BY]**
- Sort columns and direction: _____
- ✅ SELECT aliases ARE available here.

**[PHASE 7 — LIMIT/OFFSET]**
- Row cap: _____
- Always pair with ORDER BY for deterministic results.

### Step 3: Anti-Hallucination Checks
Before writing the final SQL, verify:
- [ ] Every table name exists in the schema
- [ ] Every column name exists in its table
- [ ] No aggregate function is in WHERE (it belongs in HAVING)
- [ ] GROUP BY includes all non-aggregated SELECT columns
- [ ] No SELECT alias is referenced in WHERE or GROUP BY
- [ ] No Cartesian JOIN risk (every JOIN has an ON clause)
- [ ] LEFT JOINs are not nullified by WHERE on the right-side table

### Step 4: Output Format

First output your phase-by-phase reasoning, then the final SQL.

```sql
-- SQLMind Phase-Locked Output
-- Phase 1: FROM/JOIN — [brief description]
-- Phase 2: WHERE — [brief description]
-- Phase 3: GROUP BY — [brief description if applicable]
-- Phase 4: HAVING — [brief description if applicable]  
-- Phase 5: SELECT — [brief description]
-- Phase 6: ORDER BY — [brief description if applicable]
-- Phase 7: LIMIT — [brief description if applicable]

[FINAL SQL HERE]
```

If the query is L3/L4, use CTEs:
```sql
WITH
  step_1_name AS (
    -- Purpose: [describe what this answers]
    SELECT ...
  ),
  step_2_name AS (
    -- Purpose: [describe what this answers]
    SELECT ... FROM step_1_name ...
  )
SELECT * FROM step_2_name;
```
"""
    
    return json.dumps({
        "status": "success",
        "prompt": prompt,
        "dialect": dialect,
        "complexity_hint": complexity,
        "usage": (
            "Pass the 'prompt' value to your LLM as the user message. "
            "For best results with Claude, set temperature=0 and include "
            "thinking/extended reasoning if available."
        )
    }, indent=2)


# ─── TOOL 5: Execution Plan Analyzer ─────────────────────────────────────────

@mcp.tool(
    description=(
        "Runs EXPLAIN ANALYZE on a SQL query against a live database and "
        "returns a human-readable summary of the execution plan, "
        "highlighting sequential scans, missing indexes, and expensive operations. "
        "Supports PostgreSQL (EXPLAIN ANALYZE), MySQL (EXPLAIN FORMAT=JSON), "
        "SQLite (EXPLAIN QUERY PLAN)."
    )
)
def sqlmind_explain(
    sql: str,
    connection_string: str,
    dialect: str = "postgresql",
) -> str:
    """
    Runs the query through EXPLAIN and returns a structured analysis.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        return json.dumps({"error": "pip install sqlalchemy"})
    
    try:
        engine = create_engine(connection_string)
        
        if dialect.lower() == "postgresql":
            explain_sql = f"EXPLAIN (ANALYZE false, FORMAT JSON) {sql}"
        elif dialect.lower() == "mysql":
            explain_sql = f"EXPLAIN FORMAT=JSON {sql}"
        elif dialect.lower() == "sqlite":
            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        else:
            explain_sql = f"EXPLAIN {sql}"
        
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            plan_raw = result.fetchall()
        
        # Parse and summarize
        plan_text = str(plan_raw)
        
        warnings = []
        suggestions = []
        
        if "Seq Scan" in plan_text:
            warnings.append("⚠️ Sequential scan detected — consider adding an index")
        if "Hash Join" in plan_text:
            suggestions.append("ℹ️ Hash join used — generally efficient for large tables")
        if "Nested Loop" in plan_text and "rows=1" not in plan_text:
            warnings.append("⚠️ Nested Loop join on large tables — may be slow")
        if "Sort" in plan_text and "Index Scan" not in plan_text:
            warnings.append("⚠️ Sort operation without index — consider index on ORDER BY column")
        
        return json.dumps({
            "status": "success",
            "plan_raw": plan_text[:3000],  # truncate for token budget
            "warnings": warnings,
            "suggestions": suggestions,
            "tip": "Use EXPLAIN ANALYZE (without ANALYZE false) to see actual row counts"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


# ─── TOOL 6: SQL Dialect Transpiler ──────────────────────────────────────────

@mcp.tool(
    description=(
        "Transpiles SQL from one dialect to another. "
        "Supports: postgresql, mysql, sqlite, mssql, bigquery, snowflake. "
        "Handles syntax differences: TOP vs LIMIT, GETDATE vs NOW, "
        "backtick vs double-quote identifiers, dialect-specific functions."
    )
)
def sqlmind_transpile(
    sql: str,
    from_dialect: str,
    to_dialect: str,
) -> str:
    """
    Transpiles SQL between dialects using sqlglot.
    """
    if not HAS_SQLGLOT:
        return json.dumps({
            "error": "sqlglot not installed. Run: pip install sqlglot",
            "fallback": "Install sqlglot for dialect transpilation"
        })
    
    dialect_map = {
        "postgresql": "postgres",
        "mysql": "mysql",
        "sqlite": "sqlite",
        "mssql": "tsql",
        "bigquery": "bigquery",
        "snowflake": "snowflake",
    }
    
    from_d = dialect_map.get(from_dialect.lower(), from_dialect.lower())
    to_d = dialect_map.get(to_dialect.lower(), to_dialect.lower())
    
    try:
        transpiled = sqlglot.transpile(sql, read=from_d, write=to_d, pretty=True)
        
        return json.dumps({
            "status": "success",
            "original_dialect": from_dialect,
            "target_dialect": to_dialect,
            "transpiled_sql": transpiled[0] if transpiled else "",
            "warnings": (
                "Review the transpiled SQL. Automatic transpilation may miss "
                "dialect-specific functions without direct equivalents."
            )
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "hint": "Check that the source SQL is valid in the from_dialect"
        })


# ─── TOOL 7: Query Complexity Scorer ─────────────────────────────────────────

@mcp.tool(
    description=(
        "Scores a SQL query for complexity and generation difficulty. "
        "Returns L1-L4 classification, a risk score (0-100), "
        "and specific recommendations for LLM generation strategy. "
        "Use this to decide whether to use direct generation, phase-locked, "
        "CTE decomposition, or multi-agent approaches."
    )
)
def sqlmind_score_complexity(sql: str) -> str:
    """
    Scores SQL query complexity and recommends a generation strategy.
    """
    sql_upper = sql.upper()
    
    score = 0
    factors = []
    
    # Count JOINs
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    if join_count >= 4:
        score += 30
        factors.append(f"4+ JOINs ({join_count} detected)")
    elif join_count >= 2:
        score += 15
        factors.append(f"Multiple JOINs ({join_count} detected)")
    elif join_count == 1:
        score += 5
        factors.append("Single JOIN")
    
    # Subqueries
    subquery_count = sql_upper.count("SELECT") - 1
    if subquery_count >= 3:
        score += 25
        factors.append(f"Deep nesting ({subquery_count} subqueries)")
    elif subquery_count >= 1:
        score += 12
        factors.append(f"{subquery_count} subquery/ies")
    
    # CTEs
    cte_count = len(re.findall(r'\bWITH\b', sql_upper))
    if cte_count > 0:
        score += 10 * cte_count
        factors.append(f"{cte_count} CTE(s)")
    
    # Window functions
    if re.search(r'\bOVER\s*\(', sql_upper):
        score += 20
        factors.append("Window function(s) detected")
    
    # Aggregations
    agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper))
    if agg_count >= 3:
        score += 15
        factors.append(f"Multiple aggregations ({agg_count})")
    elif agg_count >= 1:
        score += 5
        factors.append(f"{agg_count} aggregation(s)")
    
    # HAVING
    if "HAVING" in sql_upper:
        score += 5
        factors.append("HAVING clause")
    
    # CASE expressions
    case_count = len(re.findall(r'\bCASE\b', sql_upper))
    if case_count >= 2:
        score += 10
        factors.append(f"{case_count} CASE expressions")
    elif case_count == 1:
        score += 5
        factors.append("CASE expression")
    
    # Determine level
    if score <= 10:
        level = "L1"
        level_desc = "Simple"
        strategy = "direct_generation"
        llm_guidance = "Direct generation. Single LLM call sufficient."
    elif score <= 30:
        level = "L2"
        level_desc = "Moderate"
        strategy = "phase_locked"
        llm_guidance = "Use phase-locked protocol. Reason FROM→WHERE→GROUP BY→SELECT."
    elif score <= 60:
        level = "L3"
        level_desc = "Complex"
        strategy = "cte_decomposition"
        llm_guidance = (
            "Decompose into CTEs. Break into sub-questions, "
            "generate one CTE per sub-question, then merge."
        )
    else:
        level = "L4"
        level_desc = "Analytical"
        strategy = "multi_step_agent"
        llm_guidance = (
            "Use multi-step agentic approach. Consider: "
            "(1) schema linking pass, (2) plan generation pass, "
            "(3) SQL generation pass, (4) self-correction pass."
        )
    
    return json.dumps({
        "complexity_level": level,
        "complexity_description": level_desc,
        "risk_score": min(score, 100),
        "complexity_factors": factors,
        "recommended_strategy": strategy,
        "llm_guidance": llm_guidance,
        "failure_probability": f"{min(score, 95)}% chance of failure with naive single-shot generation"
    }, indent=2)


# ─── TOOL 8: Phase-Locked SQL Generation (calls Anthropic) ───────────────────

@mcp.tool(
    description=(
        "Generate SQL from a natural language query using the SQLMind phase-locked "
        "protocol and the Anthropic API (claude-sonnet-4-6). "
        "Pass schema_dsl from sqlmind_introspect. "
        "Returns validated, ready-to-run SQL with phase annotations."
    )
)
def sqlmind_generate(
    nl_query: str,
    schema_dsl: str,
    dialect: str = "postgresql",
    anthropic_api_key: Optional[str] = None,
) -> str:
    """
    Generate SQL using the SQLMind 7-phase protocol via the Anthropic API.

    Args:
        nl_query: Natural language query (e.g. "top 10 customers by revenue last month")
        schema_dsl: SQLMind Schema DSL string from sqlmind_introspect
        dialect: Target SQL dialect (postgresql, mysql, sqlite, bigquery, snowflake, …)
        anthropic_api_key: API key (falls back to ANTHROPIC_API_KEY env var)

    Returns:
        JSON with generated SQL and the phase trace used to produce it
    """
    import os

    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return json.dumps({
            "status": "error",
            "error": "ANTHROPIC_API_KEY not set. Pass anthropic_api_key or set the env var.",
        })

    try:
        import anthropic
    except ImportError:
        return json.dumps({
            "status": "error",
            "error": "anthropic package not installed. Run: pip install anthropic",
        })

    # Build the phase-locked generation prompt via sqlmind_build_prompt
    prompt_result = json.loads(sqlmind_build_prompt(
        nl_query=nl_query,
        schema_dsl=schema_dsl,
        dialect=dialect,
    ))
    phase_locked_prompt = prompt_result["prompt"]

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": phase_locked_prompt}],
        )
        response_text = message.content[0].text
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

    # Extract the SQL block from the response
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    generated_sql = sql_match.group(1).strip() if sql_match else response_text.strip()

    # Auto-validate the generated SQL against the schema
    validation_raw = sqlmind_validate(sql=generated_sql, dialect=dialect, schema_dsl=schema_dsl)
    validation = json.loads(validation_raw)

    return json.dumps({
        "status": "success",
        "dialect": dialect,
        "generated_sql": generated_sql,
        "full_response": response_text,
        "validation": {
            "is_valid": validation["is_valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        },
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    }, indent=2)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQLMind MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    if args.transport == "http":
        print(f"SQLMind MCP Server running on http://{args.host}:{args.port}")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
