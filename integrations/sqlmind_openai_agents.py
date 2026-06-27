"""
sqlmind_openai_agents.py
────────────────────────
SQLMind Agent — OpenAI Agents SDK Integration

Install:
  pip install openai-agents pyyaml sqlglot sqlalchemy

Run:
  python sqlmind_openai_agents.py

Supports any OpenAI-compatible model endpoint (Claude, Gemini, local Ollama)
via the SDK's provider-agnostic interface.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── OpenAI Agents SDK ─────────────────────────────────────────────────────────
from agents import Agent, Runner, function_tool, RunConfig
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

# ── SQLMind imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sqlmind_graph import SchemaGraph, DialectRegistry

# ── Shared state ──────────────────────────────────────────────────────────────
_graph = SchemaGraph()
_registry: Optional[DialectRegistry] = None
_dialect = "postgresql"

def _load_registry():
    global _registry
    if _registry is None:
        p = ROOT / "dialects.yaml"
        if p.exists():
            _registry = DialectRegistry(str(p))
    return _registry


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — decorated with @function_tool for automatic schema generation
# ══════════════════════════════════════════════════════════════════════════════

@function_tool
def load_schema(source: str, dialect: str = "postgresql") -> str:
    """
    Load a database schema into SQLMind's property graph.
    Call this first before any SQL generation.

    Args:
        source: A connection string (postgresql://...), file path (.yaml/.sql),
                or inline SQLMind DSL text starting with 'TABLE'.
        dialect: SQL dialect — postgresql, mysql, sqlite, mssql, bigquery,
                 snowflake, redshift, databricks, spark_sql
    """
    global _graph, _dialect
    _graph = SchemaGraph()
    _dialect = dialect

    try:
        if "://" in source:
            _graph.load_from_db(source)
            method = "live database"
        elif source.endswith(".yaml") or source.endswith(".yml"):
            _graph.load_from_yaml(source)
            method = "YAML"
        elif source.endswith(".sql"):
            _graph.load_from_ddl(open(source).read(), dialect=dialect)
            method = "DDL"
        else:
            _graph.load_from_dsl(source)
            method = "DSL"

        s = _graph.stats()
        return json.dumps({
            "loaded": True, "method": method, "dialect": dialect,
            "tables": s["tables"], "columns": s["total_columns"],
            "fk_edges": s["fk_edges"],
            "table_list": list(_graph.tables.keys()),
        })
    except Exception as e:
        return json.dumps({"loaded": False, "error": str(e)})


@function_tool
def schema_link(nl_query: str) -> str:
    """
    Map a natural language query to relevant schema nodes.
    Returns matched tables, columns, and join paths.
    Always call this before generate_sql.

    Args:
        nl_query: The user's question in natural language.
    """
    if not _graph.tables:
        return json.dumps({"error": "No schema loaded. Call load_schema first."})

    linked = _graph.schema_link(nl_query)
    return json.dumps({
        "schema_dsl": _graph.to_dsl(),
        "dialect": _dialect,
        "matched_tables": linked["matched_tables"],
        "matched_columns": linked["matched_columns"],
        "join_paths": linked["join_paths"],
    })


@function_tool
def find_join_path(from_table: str, to_table: str) -> str:
    """
    Find the shortest FK-based join path between two tables.
    Use when tables are not directly related via a single foreign key.

    Args:
        from_table: Starting table name.
        to_table: Destination table name.
    """
    path = _graph.find_join_path(from_table, to_table)
    if not path:
        return json.dumps({
            "found": False,
            "message": f"No path between {from_table} and {to_table}",
        })
    return json.dumps({
        "found": True,
        "hops": len(path.hops),
        "is_direct": path.is_direct,
        "sql_joins": path.to_sql(_dialect),
        "detail": path.to_dict(),
    })


@function_tool
def validate_sql(sql: str) -> str:
    """
    Validate SQL against SQLMind's 7-phase execution model.
    Catches: aggregate-in-WHERE, alias-in-WHERE, Cartesian JOINs,
    LEFT JOIN nullification, missing GROUP BY, dialect syntax errors,
    and column/table name errors against the schema graph.

    Args:
        sql: The SQL query to validate.
    """
    errors, warnings, suggestions = [], [], []
    sql_upper = sql.upper()

    # Aggregate in WHERE
    wm = re.search(r'WHERE\s+(.*?)(?:GROUP\s+BY|HAVING|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
    if wm:
        for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
            if agg in wm.group(1):
                errors.append(f"AGG_IN_WHERE: {agg[:-1]} in WHERE → move to HAVING")

    if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
        errors.append("HAVING_NO_GROUPBY: HAVING without GROUP BY")

    if re.search(r'FROM\s+\w+\s*,\s*\w+', sql_upper):
        errors.append("CARTESIAN_JOIN: implicit Cartesian product — use explicit JOIN ON")

    if re.search(r'SELECT\s+\*', sql_upper):
        warnings.append("SELECT_STAR: enumerate specific columns in production")

    if "ORDER BY" in sql_upper and not any(x in sql_upper for x in ["LIMIT", "TOP", "FETCH"]):
        warnings.append("ORDER_NO_LIMIT: ORDER BY without LIMIT sorts all rows")

    # LEFT JOIN nullification
    lj = re.search(r'LEFT\s+(?:OUTER\s+)?JOIN\s+(\w+)', sql_upper)
    if lj and wm and lj.group(1).lower() in wm.group(1).lower():
        errors.append(f"LEFT_JOIN_NULLIFIED: WHERE on right-table {lj.group(1)} converts LEFT JOIN to INNER JOIN")

    # Dialect-specific
    if _dialect == "mssql" and "LIMIT" in sql_upper:
        errors.append("WRONG_SYNTAX: T-SQL uses TOP(n) or OFFSET/FETCH NEXT n ROWS ONLY")

    # Schema column check
    if _graph.tables:
        for ce in _graph.validate_sql_columns(sql):
            errors.append(
                f"COLUMN_NOT_FOUND: {ce['table']}.{ce['column']} — "
                f"available: {', '.join(ce['available'][:4])}"
                + (f" (did you mean: {ce['suggestion']}?)" if ce.get("suggestion") else "")
            )

    return json.dumps({
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "summary": "✅ VALID" if not errors else f"❌ {len(errors)} error(s)",
    })


@function_tool
def get_dialect_rules(dialect: str) -> str:
    """
    Get syntax rules and gotchas for a specific SQL dialect.
    Use this when generating SQL for an unfamiliar dialect.

    Args:
        dialect: One of: postgresql, mysql, sqlite, mssql, bigquery,
                 snowflake, redshift, databricks, spark_sql
    """
    r = _load_registry()
    if not r:
        return json.dumps({"error": "dialects.yaml not found"})
    d = r.get(dialect)
    if not d:
        return json.dumps({"error": f"Unknown: {dialect}", "available": r.list_ids()})
    return json.dumps({
        "dialect": d.id,
        "limit_example": d.render_limit(10),
        "date_now": d.get("date_now"),
        "date_trunc_example": d.render_date_trunc("col", "month"),
        "string_agg_example": d.render_string_agg("name", ", "),
        "ilike": d.supports_ilike,
        "qualify": d.supports_qualify,
        "notes": d.notes[:600],
    })


@function_tool
def export_erd() -> str:
    """
    Export the loaded schema as a Mermaid ERD diagram.
    Paste output into mermaid.live or a GitHub README to visualize.
    """
    if not _graph.tables:
        return json.dumps({"error": "No schema loaded."})
    return json.dumps({
        "mermaid_erd": _graph.to_mermaid(),
        "stats": _graph.stats(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are SQLMind, VeloceAI's SQL intelligence agent.

You generate correct, dialect-aware SQL by reasoning in EXECUTION ORDER first,
then writing in standard SQL write order.

## Execution order (reason through these phases before writing SQL):
1. FROM/JOIN   → identify tables; use find_join_path for multi-hop routes
2. WHERE       → row filters ONLY — no aggregates, no SELECT aliases
3. GROUP BY    → every non-aggregated SELECT column must appear here
4. HAVING      → aggregate filters (COUNT/SUM/AVG/MAX/MIN) ONLY
5. SELECT      → output columns; aliases defined here
6. ORDER BY    → sort; SELECT aliases available here
7. LIMIT       → dialect-correct row cap

## Your workflow for every SQL request:
1. If schema unknown → call load_schema
2. Call schema_link to map the question to tables/columns/joins
3. Call find_join_path for any indirect table relationships
4. Generate SQL reasoning phase-by-phase (internal reasoning)
5. Call validate_sql on your output
6. Fix any errors, re-validate if needed
7. Return: phase reasoning summary + final SQL + any warnings

## Never violate:
- Aggregates in WHERE → move to HAVING
- SELECT alias in WHERE/GROUP BY → use original column name
- JOIN without ON → always explicit ON clause
- LEFT JOIN + WHERE on right table → move filter to ON clause
"""

sqlmind_agent = Agent(
    name="SQLMind",
    instructions=SYSTEM_PROMPT,
    tools=[
        load_schema,
        schema_link,
        find_join_path,
        validate_sql,
        get_dialect_rules,
        export_erd,
    ],
    # Model — swap to any OpenAI-compatible endpoint:
    # model="gpt-4o"                           # default OpenAI
    # model="claude-sonnet-4-6"                 # via OpenRouter or Anthropic
    # model="gemini-2.0-flash-exp"              # via OpenRouter
    model="gpt-4o",
)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER — interactive CLI + programmatic API
# ══════════════════════════════════════════════════════════════════════════════

async def run_query(user_message: str, stream: bool = False) -> str:
    """Run a single query through the SQLMind agent."""
    if stream:
        result_text = ""
        async with Runner.run_streamed(sqlmind_agent, user_message) as stream_ctx:
            async for event in stream_ctx.stream_events():
                if hasattr(event, "delta") and event.delta:
                    print(event.delta, end="", flush=True)
                    result_text += event.delta
        print()
        return result_text
    else:
        result = await Runner.run(sqlmind_agent, user_message)
        return result.final_output


async def interactive_cli():
    """Simple interactive CLI for testing the agent."""
    print("╔══════════════════════════════════════╗")
    print("║  SQLMind Agent — OpenAI Agents SDK   ║")
    print("║  VeloceAI | type 'exit' to quit      ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("SQLMind: Goodbye!")
            break

        print("SQLMind: ", end="")
        response = await run_query(user_input, stream=True)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES (run as __main__)
# ══════════════════════════════════════════════════════════════════════════════

async def demo():
    """Demo: load a schema and generate SQL."""

    DEMO_SCHEMA = """
TABLE orders (
  id          INT     PK
  customer_id INT     FK→customers.id  IDX
  amount      DECIMAL IDX
  status      VARCHAR [pending, confirmed, shipped, cancelled]
  created_at  TIMESTAMP IDX
)
TABLE customers (
  id     INT     PK
  name   VARCHAR
  region VARCHAR IDX
  tier   VARCHAR [bronze, silver, gold]
)
TABLE products (
  id       INT     PK
  sku      VARCHAR IDX
  name     VARCHAR
  price    DECIMAL
  category VARCHAR IDX
)
TABLE order_items (
  id         INT     PK
  order_id   INT     FK→orders.id
  product_id INT     FK→products.id
  qty        INT
  unit_price DECIMAL
)
"""
    print("Loading schema...")
    _graph.load_from_dsl(DEMO_SCHEMA)
    print(f"Loaded {_graph.stats()['tables']} tables\n")

    questions = [
        "Show top 5 regions by total revenue last 30 days",
        "Which gold-tier customers haven't ordered in 90 days?",
        "Revenue breakdown by product category this month vs last month",
    ]

    for q in questions:
        print(f"Q: {q}")
        print("A: ", end="")
        response = await run_query(q, stream=True)
        print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo())
    else:
        asyncio.run(interactive_cli())
