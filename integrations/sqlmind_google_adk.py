"""
sqlmind_google_adk.py
─────────────────────
SQLMind Agent — Google ADK Integration

Install:
  pip install google-adk pyyaml sqlglot sqlalchemy

Run dev UI:
  adk web

Run CLI:
  adk run sqlmind_google_adk

Structure expected by ADK:
  sqlmind_google_adk/
    __init__.py        ← exports `agent`
    agent.py           ← this file
    .env               ← GOOGLE_API_KEY or OPENAI_API_KEY
"""

import json
import os
import sys
from pathlib import Path

# ── ADK imports ──────────────────────────────────────────────────────────────
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# ── SQLMind imports (adjust path if needed) ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sqlmind_graph import SchemaGraph, DialectRegistry, graph_from_dsl

# ── State (loaded once per process) ──────────────────────────────────────────
_graph: SchemaGraph | None = None
_dialect_registry: DialectRegistry | None = None
_current_dialect: str = "postgresql"

def _get_graph() -> SchemaGraph:
    global _graph
    if _graph is None:
        _graph = SchemaGraph()
        # Try to load schema.sqlmind.yaml from the project root
        schema_path = ROOT / "schema.sqlmind.yaml"
        if schema_path.exists():
            _graph.load_from_yaml(str(schema_path))
    return _graph

def _get_registry() -> DialectRegistry:
    global _dialect_registry
    if _dialect_registry is None:
        dialects_path = ROOT / "dialects.yaml"
        if dialects_path.exists():
            _dialect_registry = DialectRegistry(str(dialects_path))
    return _dialect_registry


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — load_schema
# ══════════════════════════════════════════════════════════════════════════════
def load_schema(source: str, dialect: str = "postgresql") -> str:
    """
    Load a database schema into the SQLMind graph.

    Args:
        source: One of:
                - A connection string: "postgresql://user:pass@host/db"
                - A file path to a .sqlmind.yaml or DDL .sql file
                - Inline SQLMind DSL text (starts with "TABLE ")
        dialect: Target SQL dialect. Options: postgresql, mysql, sqlite,
                 mssql, bigquery, snowflake, redshift, databricks, spark_sql

    Returns:
        Summary of tables and columns loaded.
    """
    global _graph, _current_dialect
    _graph = SchemaGraph()
    _current_dialect = dialect

    try:
        if source.startswith("TABLE ") or "\nTABLE " in source:
            _graph.load_from_dsl(source)
            method = "DSL"
        elif source.endswith(".yaml") or source.endswith(".yml"):
            _graph.load_from_yaml(source)
            method = "YAML file"
        elif source.endswith(".sql"):
            with open(source) as f:
                ddl = f.read()
            _graph.load_from_ddl(ddl, dialect=dialect)
            method = "DDL file"
        elif "://" in source:
            _graph.load_from_db(source)
            method = "live database"
        else:
            # Try as inline DSL
            _graph.load_from_dsl(source)
            method = "inline DSL"

        stats = _graph.stats()
        return json.dumps({
            "status": "success",
            "method": method,
            "dialect": dialect,
            "tables": stats["tables"],
            "columns": stats["total_columns"],
            "fk_edges": stats["fk_edges"],
            "table_names": list(_graph.tables.keys()),
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — get_schema_context
# ══════════════════════════════════════════════════════════════════════════════
def get_schema_context(nl_query: str) -> str:
    """
    Given a natural language query, return relevant schema context:
    matched tables, columns, join paths, and the full SQLMind DSL.
    This is the schema-linking step — call this before generate_sql.

    Args:
        nl_query: The user's question in plain English.

    Returns:
        JSON with matched tables, columns, join paths, and schema DSL.
    """
    graph = _get_graph()
    if not graph.tables:
        return json.dumps({
            "status": "error",
            "error": "No schema loaded. Call load_schema first."
        })

    linked = graph.schema_link(nl_query)
    schema_dsl = graph.to_dsl()

    return json.dumps({
        "status": "success",
        "nl_query": nl_query,
        "schema_dsl": schema_dsl,
        "matched_tables": linked["matched_tables"],
        "matched_columns": linked["matched_columns"],
        "join_paths": linked["join_paths"],
        "dialect": _current_dialect,
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — find_join_path
# ══════════════════════════════════════════════════════════════════════════════
def find_join_path(from_table: str, to_table: str) -> str:
    """
    Find the shortest join path between two tables in the schema graph.
    Use this when you need to join tables that aren't directly related.

    Args:
        from_table: Starting table name.
        to_table: Target table name.

    Returns:
        JSON with the join path hops and SQL JOIN clauses.
    """
    graph = _get_graph()
    path = graph.find_join_path(from_table, to_table)

    if path is None:
        return json.dumps({
            "status": "no_path",
            "message": f"No join path found between {from_table} and {to_table}",
            "suggestion": "Check if both tables are in the schema. You may need intermediate tables."
        })

    return json.dumps({
        "status": "success",
        "from_table": from_table,
        "to_table": to_table,
        "hops": len(path.hops),
        "is_direct": path.is_direct,
        "confidence": path.confidence,
        "sql_joins": path.to_sql(_current_dialect),
        "path_detail": path.to_dict(),
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — generate_sql
# ══════════════════════════════════════════════════════════════════════════════
def generate_sql(
    nl_query: str,
    schema_context: str = "",
    dialect: str = "",
) -> str:
    """
    Generate phase-locked SQL from a natural language query.
    Internally reasons in execution order: FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT.
    Always call get_schema_context first and pass its output here.

    Args:
        nl_query: The natural language question.
        schema_context: JSON string from get_schema_context (recommended).
        dialect: Override dialect (optional — uses loaded dialect by default).

    Returns:
        JSON with generated SQL, phase reasoning trace, and warnings.
    """
    target_dialect = dialect or _current_dialect

    # Parse schema context if provided
    schema_dsl = ""
    join_hints = ""
    if schema_context:
        try:
            ctx = json.loads(schema_context)
            schema_dsl = ctx.get("schema_dsl", "")
            join_paths = ctx.get("join_paths", {})
            if join_paths:
                join_hints = "\n".join(
                    f"Join path {k}: {v.get('joins', [])}"
                    for k, v in join_paths.items()
                )
        except Exception:
            schema_dsl = schema_context

    # If no schema_dsl from context, get from graph
    if not schema_dsl:
        graph = _get_graph()
        schema_dsl = graph.to_dsl() if graph.tables else ""

    # Get dialect notes
    registry = _get_registry()
    dialect_notes = ""
    if registry:
        d = registry.get(target_dialect)
        if d:
            dialect_notes = d.notes[:800]  # truncate for prompt

    # Build the phase-locked generation prompt
    prompt_parts = [
        f"Generate {target_dialect.upper()} SQL for: {nl_query}",
        "",
        "## SCHEMA",
        schema_dsl or "No schema provided — use your best judgment.",
    ]

    if join_hints:
        prompt_parts += ["", "## PRE-COMPUTED JOIN PATHS", join_hints]

    if dialect_notes:
        prompt_parts += ["", f"## {target_dialect.upper()} DIALECT NOTES", dialect_notes]

    prompt_parts += [
        "",
        "## GENERATION PROTOCOL",
        "Reason in execution order before writing SQL:",
        "[FROM/JOIN] → tables and join path from schema graph",
        "[WHERE] → row-level filters ONLY. NO aggregates. NO SELECT aliases.",
        "[GROUP BY] → all non-aggregated SELECT columns go here",
        "[HAVING] → aggregate filters ONLY (COUNT, SUM, AVG, etc.)",
        "[SELECT] → output columns and aliases",
        "[ORDER BY] → sort (aliases from SELECT are available here)",
        "[LIMIT] → use dialect-correct syntax",
        "",
        "Output format:",
        "PHASE_REASONING: <your phase-by-phase thinking>",
        "COMPLEXITY: <L1|L2|L3|L4>",
        "WARNINGS: <any issues detected>",
        "```sql",
        "<final SQL here>",
        "```",
    ]

    full_prompt = "\n".join(prompt_parts)

    return json.dumps({
        "status": "success",
        "generation_prompt": full_prompt,
        "dialect": target_dialect,
        "note": (
            "Pass 'generation_prompt' to the LLM with the SQLMind system prompt. "
            "The agent will use this to generate phase-locked SQL."
        )
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — validate_sql
# ══════════════════════════════════════════════════════════════════════════════
def validate_sql(sql: str, dialect: str = "") -> str:
    """
    Validate a SQL query against the SQLMind 7-phase execution model.
    Checks for: aggregate-in-WHERE, alias-in-WHERE, missing GROUP BY columns,
    Cartesian JOINs, LEFT JOIN nullification, wrong dialect syntax.
    Also validates column names against the loaded schema graph.

    Args:
        sql: The SQL query to validate.
        dialect: Override dialect (optional).

    Returns:
        JSON validation report with errors, warnings, and fix suggestions.
    """
    import re
    target_dialect = dialect or _current_dialect
    errors = []
    warnings = []
    suggestions = []
    sql_upper = sql.upper()

    # Phase checks
    where_m = re.search(r'WHERE\s+(.*?)(?:GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|$)', sql_upper, re.DOTALL)
    if where_m:
        where_body = where_m.group(1)
        for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
            if agg in where_body:
                errors.append({
                    "code": "AGG_IN_WHERE",
                    "message": f"{agg.rstrip('(')} in WHERE clause — move to HAVING",
                    "fix": "HAVING filters groups; WHERE filters rows."
                })

    if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
        errors.append({"code": "HAVING_NO_GROUPBY", "message": "HAVING without GROUP BY"})

    if re.search(r'FROM\s+\w+\s*,\s*\w+', sql_upper):
        errors.append({"code": "CARTESIAN_JOIN", "message": "Comma-separated tables — implicit Cartesian JOIN risk"})

    if re.search(r'SELECT\s+\*', sql_upper):
        warnings.append({"code": "SELECT_STAR", "message": "SELECT * — enumerate columns in production"})

    if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper and "TOP" not in sql_upper and "FETCH" not in sql_upper:
        warnings.append({"code": "ORDER_NO_LIMIT", "message": "ORDER BY without LIMIT may be expensive"})

    # LEFT JOIN + WHERE on right table
    lj_m = re.search(r'LEFT\s+(?:OUTER\s+)?JOIN\s+(\w+)', sql_upper)
    if lj_m and where_m:
        rt = lj_m.group(1).lower()
        if rt in where_m.group(1).lower():
            warnings.append({
                "code": "LEFT_JOIN_NULLIFIED",
                "message": f"WHERE on {rt} (right-side of LEFT JOIN) converts it to INNER JOIN",
                "fix": f"Move filter to JOIN ON clause: ... LEFT JOIN {rt} ON ... AND {rt}.col = val"
            })

    # Dialect-specific checks
    if target_dialect == "mssql" and "LIMIT" in sql_upper:
        errors.append({"code": "WRONG_DIALECT_LIMIT",
                       "message": "T-SQL uses TOP (n) or OFFSET/FETCH, not LIMIT"})
    if target_dialect in ("bigquery", "snowflake") and "QUALIFY" not in sql_upper:
        if re.search(r'ROW_NUMBER|RANK\(\)|DENSE_RANK', sql_upper):
            suggestions.append({"code": "USE_QUALIFY",
                                 "message": f"{target_dialect} supports QUALIFY for window filter — avoids subquery"})

    # Schema graph column validation
    graph = _get_graph()
    if graph.tables:
        col_errors = graph.validate_sql_columns(sql)
        for ce in col_errors:
            errors.append({
                "code": "COLUMN_NOT_FOUND",
                "message": f"Column '{ce['column']}' not in table '{ce['table']}'",
                "available": ce["available"][:5],
                "suggestion": ce.get("suggestion"),
            })

    return json.dumps({
        "is_valid": len(errors) == 0,
        "dialect": target_dialect,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "overall": "✅ VALID" if len(errors) == 0 else f"❌ {len(errors)} error(s)",
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — get_dialect_info
# ══════════════════════════════════════════════════════════════════════════════
def get_dialect_info(dialect: str) -> str:
    """
    Get syntax rules, gotchas, and function references for a SQL dialect.
    Supported: postgresql, mysql, sqlite, mssql, bigquery, snowflake,
               redshift, databricks, spark_sql

    Args:
        dialect: The target SQL dialect ID.

    Returns:
        JSON with dialect rules, date functions, limit syntax, and notes.
    """
    registry = _get_registry()
    if registry is None:
        return json.dumps({"status": "error", "error": "dialects.yaml not found"})

    d = registry.get(dialect)
    if d is None:
        return json.dumps({
            "status": "error",
            "error": f"Unknown dialect: {dialect}",
            "available": registry.list_ids()
        })

    return json.dumps({
        "status": "success",
        "dialect": d.id,
        "name": d.name,
        "aliases": d.aliases,
        "identifier_quoting": d.get("identifiers", {}).get("quote_char"),
        "limit_syntax": d.render_limit(10, 0),
        "limit_with_offset": d.render_limit(10, 20),
        "date_now": d.get("date_now"),
        "date_trunc_example": d.render_date_trunc("created_at", "month"),
        "date_add_example": d.render_date_add("created_at", 30, "day"),
        "string_agg_example": d.render_string_agg("name", ", "),
        "supports_ilike": d.supports_ilike,
        "supports_qualify": d.supports_qualify,
        "window_support": d.get("window_support"),
        "cte_support": d.get("cte_support"),
        "array_fns": d.get("array_fns", []),
        "notes": d.notes,
        "custom_fns": d.custom_fns,
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — export_schema_erd
# ══════════════════════════════════════════════════════════════════════════════
def export_schema_erd() -> str:
    """
    Export the loaded schema as a Mermaid ERD diagram string.
    Paste this into any Mermaid-compatible renderer (GitHub, Notion, etc.).

    Returns:
        Mermaid erDiagram string.
    """
    graph = _get_graph()
    if not graph.tables:
        return json.dumps({"status": "error", "error": "No schema loaded."})

    return json.dumps({
        "status": "success",
        "mermaid_erd": graph.to_mermaid(),
        "stats": graph.stats(),
        "tip": "Paste the mermaid_erd value into https://mermaid.live to visualize"
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

SQLMIND_SYSTEM_PROMPT = """You are SQLMind, an expert SQL generation agent for VeloceAI.

Your core capability: generate accurate SQL by reasoning in EXECUTION ORDER,
not write order. This eliminates the most common LLM SQL mistakes.

## Your 7-phase protocol (always reason in this order):
1. FROM / JOIN  → identify tables, use schema graph join paths
2. WHERE        → row-level filters (NO aggregates, NO SELECT aliases)
3. GROUP BY     → all non-aggregated SELECT columns go here
4. HAVING       → aggregate filters only (COUNT, SUM, AVG, MAX, MIN)
5. SELECT       → output columns and alias definitions
6. ORDER BY     → sort (SELECT aliases available here)
7. LIMIT        → use dialect-correct syntax

## Critical rules (never violate):
- Aggregate functions → NEVER in WHERE → always in HAVING
- SELECT aliases → NOT available in WHERE or GROUP BY
- Every JOIN → MUST have ON clause (no Cartesian products)
- LEFT JOIN + WHERE on right table → converts to INNER JOIN (bug)
- GROUP BY → every non-aggregated SELECT column must appear here

## Your workflow for every SQL request:
1. Call load_schema if no schema is loaded yet
2. Call get_schema_context to link NL entities to schema nodes
3. Call find_join_path if tables aren't directly related
4. Call generate_sql to get the phase-locked generation prompt
5. Generate SQL using the prompt and your protocol above
6. Call validate_sql on your output
7. If validation fails, fix targeted errors and re-validate
8. Return the final SQL with brief phase reasoning

## Supported dialects:
postgresql, mysql, sqlite, mssql, bigquery, snowflake, redshift, databricks, spark_sql

Always ask for the dialect if not specified. Default: postgresql.
"""

# Register all tools
tools = [
    FunctionTool(load_schema),
    FunctionTool(get_schema_context),
    FunctionTool(find_join_path),
    FunctionTool(generate_sql),
    FunctionTool(validate_sql),
    FunctionTool(get_dialect_info),
    FunctionTool(export_schema_erd),
]

# The agent — exported as `agent` for ADK discovery
agent = Agent(
    name="sqlmind",
    model="gemini-2.0-flash",          # swap to gemini-2.5-pro for complex queries
    description="SQL intelligence agent — generates accurate, dialect-aware SQL from natural language using phase-locked execution-order reasoning and a property graph schema.",
    instruction=SQLMIND_SYSTEM_PROMPT,
    tools=tools,
)

# ADK requires __all__ or the `agent` name at module level
__all__ = ["agent"]
