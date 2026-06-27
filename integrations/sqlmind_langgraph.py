"""
sqlmind_langgraph.py
─────────────────────
SQLMind Agent — LangGraph / LangChain Integration

Install:
  pip install langgraph langchain langchain-openai langchain-anthropic pyyaml sqlglot sqlalchemy

Run:
  python sqlmind_langgraph.py --chat
  python sqlmind_langgraph.py --demo
  python sqlmind_langgraph.py --server   # FastAPI streaming server
"""

import json
import os
import re
import sys
import asyncio
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

# ── LangGraph / LangChain ─────────────────────────────────────────────────────
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Model — swap to whichever you want
try:
    from langchain_openai import ChatOpenAI
    _llm_cls = ChatOpenAI
    _llm_kwargs = {"model": "gpt-4o", "temperature": 0, "streaming": True}
except ImportError:
    try:
        from langchain_anthropic import ChatAnthropic
        _llm_cls = ChatAnthropic
        _llm_kwargs = {"model": "claude-sonnet-4-6", "temperature": 0, "streaming": True}
    except ImportError:
        raise ImportError("pip install langchain-openai or langchain-anthropic")

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
# TOOLS — @tool decorator for LangChain/LangGraph
# ══════════════════════════════════════════════════════════════════════════════

@tool
def load_schema(source: str, dialect: str = "postgresql") -> str:
    """
    Load a database schema into the SQLMind property graph.
    Call this first. Accepts: connection string, .yaml file, .sql DDL, or inline DSL.

    Args:
        source: postgresql://... | /path/to/schema.yaml | /path/to/ddl.sql | inline TABLE ...
        dialect: postgresql, mysql, sqlite, mssql, bigquery, snowflake, redshift, databricks, spark_sql
    """
    global _graph, _dialect
    _graph = SchemaGraph()
    _dialect = dialect
    try:
        if "://" in source:
            _graph.load_from_db(source)
            m = "db"
        elif source.endswith((".yaml", ".yml")):
            _graph.load_from_yaml(source)
            m = "yaml"
        elif source.endswith(".sql"):
            _graph.load_from_ddl(open(source).read(), dialect=dialect)
            m = "ddl"
        else:
            _graph.load_from_dsl(source)
            m = "dsl"
        s = _graph.stats()
        return json.dumps({"ok": True, "method": m, "tables": s["tables"],
                           "columns": s["total_columns"], "fk_edges": s["fk_edges"],
                           "table_list": list(_graph.tables.keys())})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def schema_link(nl_query: str) -> str:
    """
    Map a natural language query to schema nodes (tables, columns, join paths).
    Always call before generate_sql. Returns schema DSL + matched elements.

    Args:
        nl_query: User's question in plain English.
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


@tool
def find_join_path(from_table: str, to_table: str) -> str:
    """
    Find the shortest FK join path between two tables.
    Use when tables aren't directly connected by a single FK.

    Args:
        from_table: Starting table.
        to_table: Target table.
    """
    path = _graph.find_join_path(from_table, to_table)
    if not path:
        return json.dumps({"found": False, "msg": f"No path: {from_table}→{to_table}"})
    return json.dumps({
        "found": True, "hops": len(path.hops),
        "direct": path.is_direct, "confidence": path.confidence,
        "sql": path.to_sql(_dialect), "detail": path.to_dict(),
    })


@tool
def validate_sql(sql: str) -> str:
    """
    Validate SQL against SQLMind's 7-phase execution model.
    Checks: aggregate-in-WHERE, LEFT JOIN nullification, Cartesian JOINs,
    missing GROUP BY, dialect syntax, and schema column existence.

    Args:
        sql: The SQL query to validate.
    """
    errors, warnings = [], []
    sql_upper = sql.upper()

    wm = re.search(r'WHERE\s+(.*?)(?:GROUP\s+BY|HAVING|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
    if wm:
        for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
            if agg in wm.group(1):
                errors.append(f"AGG_IN_WHERE: {agg[:-1]}() must be in HAVING, not WHERE")

    if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
        errors.append("HAVING without GROUP BY")

    if re.search(r'FROM\s+\w+\s*,\s*\w+', sql_upper):
        errors.append("Implicit Cartesian JOIN — use explicit JOIN ON")

    if re.search(r'SELECT\s+\*', sql_upper):
        warnings.append("SELECT * — enumerate columns in production queries")

    if "ORDER BY" in sql_upper and not any(x in sql_upper for x in ["LIMIT","TOP","FETCH"]):
        warnings.append("ORDER BY without LIMIT — will sort all rows")

    lj = re.search(r'LEFT\s+(?:OUTER\s+)?JOIN\s+(\w+)', sql_upper)
    if lj and wm and lj.group(1).lower() in wm.group(1).lower():
        errors.append(f"LEFT JOIN nullified: WHERE on {lj.group(1)} converts LEFT to INNER JOIN")

    if _dialect == "mssql" and "LIMIT" in sql_upper:
        errors.append("T-SQL: use TOP(n) or OFFSET/FETCH NEXT, not LIMIT")

    if _graph.tables:
        for ce in _graph.validate_sql_columns(sql):
            errors.append(
                f"Column not found: {ce['table']}.{ce['column']}"
                + (f" → did you mean: {ce['suggestion']}?" if ce.get("suggestion") else "")
            )

    return json.dumps({
        "valid": len(errors) == 0,
        "errors": errors, "warnings": warnings,
        "status": "✅ VALID" if not errors else f"❌ {len(errors)} error(s)"
    })


@tool
def get_dialect_rules(dialect: str) -> str:
    """
    Get syntax rules and functions for a SQL dialect.

    Args:
        dialect: postgresql, mysql, sqlite, mssql, bigquery, snowflake,
                 redshift, databricks, spark_sql
    """
    r = _load_registry()
    if not r:
        return json.dumps({"error": "dialects.yaml not found"})
    d = r.get(dialect)
    if not d:
        return json.dumps({"error": f"Unknown: {dialect}", "known": r.list_ids()})
    return json.dumps({
        "dialect": d.id, "limit": d.render_limit(10),
        "date_now": d.get("date_now"),
        "date_trunc": d.render_date_trunc("col", "month"),
        "string_agg": d.render_string_agg("name", ", "),
        "ilike": d.supports_ilike, "qualify": d.supports_qualify,
        "notes": d.notes[:600],
    })


@tool
def export_erd() -> str:
    """Export the loaded schema as a Mermaid ERD diagram string."""
    if not _graph.tables:
        return json.dumps({"error": "No schema loaded."})
    return json.dumps({"mermaid": _graph.to_mermaid(), "stats": _graph.stats()})


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH AGENT DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [load_schema, schema_link, find_join_path, validate_sql, get_dialect_rules, export_erd]

SYSTEM_PROMPT = """You are SQLMind, VeloceAI's SQL intelligence agent.

Generate correct, dialect-aware SQL by reasoning in EXECUTION ORDER:
1. FROM/JOIN  → tables + join paths (use find_join_path for indirect routes)
2. WHERE      → row filters ONLY — no aggregates, no SELECT aliases
3. GROUP BY   → every non-aggregated SELECT column must be here
4. HAVING     → aggregate filters only (COUNT/SUM/AVG/MAX/MIN)
5. SELECT     → output columns and alias definitions
6. ORDER BY   → sort; SELECT aliases available here
7. LIMIT      → dialect-correct row cap

Workflow for every SQL request:
1. Call load_schema if needed
2. Call schema_link to map question → schema
3. Call find_join_path for indirect table relationships
4. Generate SQL using phase-locked protocol above
5. Call validate_sql — fix any errors, re-validate
6. Return: brief phase reasoning + final SQL + warnings

Critical rules:
- Aggregates in WHERE → HAVING
- SELECT alias in WHERE/GROUP BY → use original column name
- Every JOIN → explicit ON clause
- LEFT JOIN + WHERE on right table → move filter to ON clause
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Build the LLM and bind tools
llm = _llm_cls(**_llm_kwargs)
llm_with_tools = llm.bind_tools(TOOLS)


def llm_node(state: AgentState) -> dict:
    """Main LLM reasoning node."""
    messages = state["messages"]
    # Inject system prompt if first message
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_llm(state: AgentState) -> Literal["tools", "__end__"]:
    """Route: if last message has tool calls → tools node, else → end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# Build the StateGraph
tool_node = ToolNode(TOOLS)

builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "llm")   # loop back after tool execution

sqlmind_graph = builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

async def run_query(question: str, stream: bool = True) -> str:
    """Run a query through the LangGraph agent."""
    inputs = {"messages": [HumanMessage(content=question)]}

    if stream:
        final = ""
        async for chunk in sqlmind_graph.astream(inputs, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "llm":
                    msgs = update.get("messages", [])
                    for m in msgs:
                        if isinstance(m, AIMessage) and m.content:
                            print(m.content, end="", flush=True)
                            final += m.content if isinstance(m.content, str) else ""
        print()
        return final
    else:
        result = await sqlmind_graph.ainvoke(inputs)
        last = result["messages"][-1]
        return last.content if isinstance(last.content, str) else str(last.content)


async def interactive_cli():
    """Interactive CLI."""
    print("╔══════════════════════════════════╗")
    print("║  SQLMind Agent — LangGraph       ║")
    print("║  VeloceAI | 'exit' to quit       ║")
    print("╚══════════════════════════════════╝\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not q: continue
        if q.lower() in ("exit", "quit"): print("SQLMind: Goodbye!"); break
        print("SQLMind: ", end="")
        await run_query(q)
        print()


# ── Optional: FastAPI streaming server ───────────────────────────────────────
def create_fastapi_app():
    """
    Create a FastAPI app with SSE streaming endpoint.
    Install: pip install fastapi uvicorn sse-starlette
    Run:     uvicorn sqlmind_langgraph:app --reload
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    app = FastAPI(title="SQLMind API", version="1.0")

    class QueryRequest(BaseModel):
        question: str
        dialect: str = "postgresql"

    @app.post("/generate")
    async def generate(req: QueryRequest):
        global _dialect
        _dialect = req.dialect

        async def event_stream():
            inputs = {"messages": [HumanMessage(content=req.question)]}
            async for chunk in sqlmind_graph.astream(inputs, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "llm":
                        for m in update.get("messages", []):
                            if isinstance(m, AIMessage) and m.content:
                                text = m.content if isinstance(m.content, str) else ""
                                if text:
                                    yield f"data: {json.dumps({'text': text, 'node': node})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no"})

    @app.get("/health")
    async def health():
        return {"status": "ok", "tables": _graph.stats()["tables"]}

    return app


# ══════════════════════════════════════════════════════════════════════════════
# DEMO + MAIN
# ══════════════════════════════════════════════════════════════════════════════

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
TABLE order_items (
  id         INT  PK
  order_id   INT  FK→orders.id
  product_id INT  FK→products.id
  qty        INT
  unit_price DECIMAL
)
TABLE products (
  id       INT     PK
  sku      VARCHAR IDX
  name     VARCHAR
  price    DECIMAL
  category VARCHAR IDX
)
"""


async def demo():
    """Demo run with a preloaded schema."""
    _graph.load_from_dsl(DEMO_SCHEMA)
    print(f"Schema loaded: {_graph.stats()['tables']} tables\n")

    for q in [
        "Top 5 regions by revenue in the last 30 days",
        "Which products have been ordered more than 100 times this year?",
    ]:
        print(f"Q: {q}")
        print("A: ", end="")
        await run_query(q)
        print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chat", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--server", action="store_true")
    args = p.parse_args()

    if args.server:
        import uvicorn
        app = create_fastapi_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    elif args.demo:
        asyncio.run(demo())
    else:
        asyncio.run(interactive_cli())

# Expose graph for LangGraph Studio / LangSmith
graph = sqlmind_graph
