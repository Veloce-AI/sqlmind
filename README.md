
<div align="center">

<img src="docs/logo.svg" width="80" height="80" alt="SQLMind logo" /><br/>

# SQLMind

**SQL intelligence layer for LLM agents**

*by [VeloceAI.in](https://veloceai.in/)*

[![Python](https://img.shields.io/badge/Python-3.10+-a371f7?style=flat-square&logo=python&logoColor=white)](https://python.org) [![Tests](https://img.shields.io/github/actions/workflow/status/Veloce-AI/sqlmind/ci.yml?label=tests&style=flat-square&color=39d353)](https://github.com/Veloce-AI/sqlmind/actions) [![License](https://img.shields.io/badge/License-MIT-8b949e?style=flat-square)](LICENSE) [![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-58a6ff?style=flat-square)](https://pypi.org/project/sqlmind) [![Dialects](https://img.shields.io/badge/SQL_dialects-10-f0883e?style=flat-square)](#dialects) [![MCP](https://img.shields.io/badge/MCP_server-8_tools-a371f7?style=flat-square)](#mcp-server)

Stop writing SQL prompts. Start reasoning in execution order.<br/>
SQLMind gives agents a **property graph schema**, a **7-phase generation protocol**,<br/>
and a **5-layer validation gate** — across 10 SQL engines, in any framework.

[Get Started](#quick-start) · [Docs](https://github.com/Veloce-AI/sqlmind/wiki) · [Discord](https://discord.gg/veloceai) · [Contributing](CONTRIBUTING.md)

</div>

<div align="center">
  <img src="docs/hero_banner.svg" width="100%" alt="SQLMind hero" />
</div>


---

## How it works

| Stage | What happens | Needs API key? |
|---|---|---|
| **01 Property Graph** | DSL / live DB / DDL → TableNodes, ColumnNodes, FK edges | No |
| **02 Schema Linking** | NL question → matched tables, columns, BFS join paths | No |
| **03 Phase-locked SQL** | LLM reasons FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT | Yes (LLM only) |
| **04 Validation Gate** | 5 layers: Phase · JOIN · Schema · Dialect · Syntax | No |

The API key only touches stage 3 — the LLM that decides which tool to call.
All tools, the graph engine, validation, and dialect rules are pure Python.

---

## Why SQLMind

| Feature | SQLMind | Raw LLM | LangChain SQL Agent |
|---|---|---|---|
| Phase-locked reasoning | ✅ Built-in 7-phase protocol | ❌ Write-order only | ❌ Single prompt |
| Schema as property graph | ✅ TableNodes + FK edges | ❌ Raw text only | ⚠️ DDL string injection |
| Join path discovery | ✅ BFS over FK graph | ❌ Guesses from text | ❌ Not available |
| Pre-execution validation | ✅ 5 independent layers | ❌ None | ⚠️ Basic syntax only |
| Self-correction loop | ✅ Targeted, max 2 retries | ❌ None | ⚠️ Retry on exec error |
| Multi-dialect support | ✅ 10 dialects, editable YAML | ❌ Ad-hoc per prompt | ⚠️ Dialect param only |
| MCP server | ✅ 8 tools, editor-ready | ❌ None | ❌ None |
| Tools need API key | ✅ Pure Python — no key | ❌ Always | ❌ Always |

---

## Quick Start

### Install

SQLMind is not yet on PyPI. Install directly from source:

```bash
# Clone the repo
git clone https://github.com/Veloce-AI/sqlmind.git
cd sqlmind

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

# Install with all extras
pip install -e ".[all]"

# Or just the core + database + graph
pip install -e ".[db,graph]"
```

> **PyPI release coming soon.** Once published, `pip install sqlmind` will work.

### Set your API key

```bash
cp .env.example .env
# Edit .env — add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 5 commands to get SQL generated

```bash
# 1. Inspect a schema file
python sqlmind_graph.py inspect examples/schema.sqlmind.yaml

# 2. Find a join path between two tables
python sqlmind_graph.py join-path examples/schema.sqlmind.yaml orders products

# 3. Schema-link a natural language question
python sqlmind_graph.py link examples/schema.sqlmind.yaml "revenue by customer region"

# 4. Export a Mermaid ERD
python sqlmind_graph.py erd examples/schema.sqlmind.yaml > schema.mmd

# 5. Generate SQL from natural language (calls Anthropic API)
python sqlmind_graph.py generate examples/schema.sqlmind.yaml \
  "top 10 customers by total revenue last month" \
  --dialect postgresql
```

---

## Repo Structure

```
sqlmind/
├── sqlmind_graph.py              ← Property graph engine (SchemaGraph, DialectRegistry, CLI)
├── sqlmind_mcp_server.py         ← MCP server (8 tools)
├── sqlmind.py                    ← Standalone Python ADK agent
├── dialects.yaml                 ← User-editable dialect config (10 engines)
├── SKILL.md                      ← Claude Code skill (phase-locked protocol)
├── CLAUDE.md                     ← Drop in project root to activate skill
├── docs/
│   ├── hero_banner.svg           ← Wide hero image (1360×680) for GitHub README top
│   └── logo.svg                  ← Standalone icon (200×200) for favicon / social preview
├── integrations/
│   ├── sqlmind_google_adk.py     ← Google ADK agent
│   ├── sqlmind_openai_agents.py  ← OpenAI Agents SDK agent
│   └── sqlmind_langgraph.py      ← LangGraph agent + FastAPI server
├── tests/
│   ├── test_graph.py             ← Graph engine tests (14 tests)
│   └── test_dialects.py          ← Dialect registry tests
├── examples/
│   └── schema.sqlmind.yaml       ← Example schema (orders/customers/products)
├── pyproject.toml
├── .env.example
├── CONTRIBUTING.md
└── .github/workflows/ci.yml      ← GitHub Actions CI (pytest on push)
```

---

## How the Property Graph Works

```python
from sqlmind_graph import SchemaGraph

# Load from .sqlmind.yaml
graph = SchemaGraph().load_from_yaml("examples/schema.sqlmind.yaml")

# Load from a live database (requires sqlalchemy)
graph = SchemaGraph().load_from_db("postgresql://user:pass@localhost/mydb")

# Load from DDL
graph = SchemaGraph().load_from_ddl("""
    CREATE TABLE orders (
        id          SERIAL PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        amount      NUMERIC(10,2)
    );
""")

# BFS join path discovery
path = graph.find_join_path("orders", "products")
print(path.to_sql())
# orders
# INNER JOIN order_items ON orders.id = order_items.order_id
# INNER JOIN products ON order_items.product_id = products.id

# Alias-aware column validation
errors = graph.validate_sql_columns("SELECT o.bad_col FROM orders o")
# → [{"type": "COLUMN_NOT_FOUND", "table": "orders", "column": "bad_col", ...}]
```

---

## Schema DSL Format

The `.sqlmind.yaml` file is the canonical schema format:

```yaml
tables:
  orders:
    name: orders
    description: "Customer purchase orders"
    row_count: 2500000
    columns:
      - { name: id,          type: INT,       pk: true }
      - { name: customer_id, type: INT,       fk: true, references: customers.id, indexed: true }
      - { name: amount,      type: DECIMAL }
      - { name: status,      type: VARCHAR,   enum: [pending, confirmed, shipped, cancelled] }
      - { name: created_at,  type: TIMESTAMP, indexed: true }
edges:
  - { from_table: orders, from_col: customer_id, to_table: customers, to_col: id }
```

You can also write schemas in compact DSL notation (supported by `load_from_dsl`):

```
TABLE orders (
  id          INT     PK
  customer_id INT     FK→customers.id  IDX
  amount      DECIMAL
  status      VARCHAR [pending, confirmed, shipped, cancelled]
  created_at  TIMESTAMP IDX
)
```

---

## Dialects

10 SQL engines supported out of the box — all rules live in `dialects.yaml`:

| ID | Engine | Key differences |
|---|---|---|
| `postgresql` | PostgreSQL | `ILIKE`, `::` cast, `STRING_AGG`, `RETURNING` |
| `mysql` | MySQL / MariaDB | Backticks, `GROUP_CONCAT`, no `DATE_TRUNC` |
| `sqlite` | SQLite | No `RIGHT JOIN`, `strftime()`, permissive `GROUP BY` |
| `mssql` | SQL Server (T-SQL) | `TOP`, `GETDATE()`, `OFFSET/FETCH`, `[]` identifiers |
| `bigquery` | Google BigQuery | Backtick tables, `QUALIFY`, `ARRAY_AGG`, `EXCEPT DISTINCT` |
| `snowflake` | Snowflake | `QUALIFY`, `ILIKE`, `LISTAGG`, `TRY_CAST` |
| `redshift` | Amazon Redshift | `LISTAGG`, `GETDATE()`, `ILIKE`, limited arrays |
| `databricks` | Databricks SQL | `COLLECT_LIST`, `LATERAL VIEW EXPLODE`, time travel |
| `spark_sql` | Apache Spark SQL | `RLIKE`, `LATERAL VIEW EXPLODE`, `COLLECT_LIST` |
| `oracle` | Oracle Database | `FETCH FIRST`, `SYSDATE`, `NVL`, `CONNECT BY`, `MINUS` |

Add your own — copy any block in `dialects.yaml` and override the fields that differ.

---

## MCP Server

### Claude Code

```bash
# Add to Claude Code (stdio mode — works in any directory)
claude mcp add sqlmind --command python --args /path/to/sqlmind_mcp_server.py

# Or drop CLAUDE.md in your project root — Claude Code auto-discovers it
```

Once connected, 8 tools are available:

| Tool | What it does |
|---|---|
| `sqlmind_introspect` | Connect to a live DB → returns SQLMind DSL schema |
| `sqlmind_link_schema` | Map NL entities to tables, columns, and join paths |
| `sqlmind_validate` | 7-phase + schema column check on any SQL string |
| `sqlmind_build_prompt` | Build the phase-locked generation prompt |
| `sqlmind_generate` | NL → SQL via Anthropic API, auto-validated |
| `sqlmind_explain` | Run `EXPLAIN ANALYZE` and summarize the plan |
| `sqlmind_transpile` | Convert SQL between dialects via sqlglot |
| `sqlmind_score_complexity` | L1–L4 complexity score + generation strategy |

Example prompt to Claude Code once connected:
```
Run sqlmind_introspect on postgresql://localhost/mydb, then use
sqlmind_generate to write SQL for "monthly revenue by customer tier".
```

### Cursor

1. Open **Cursor Settings → MCP → Add Server**
2. Name: `sqlmind`  |  Command: `python`  |  Args: `["/absolute/path/sqlmind_mcp_server.py"]`
3. Restart Cursor. Tools appear in the Agent panel (Cmd+I / Ctrl+I).

### Windsurf (Codeium)

Add to `.windsurf/config.json` in your project root:

```json
{
  "mcp": {
    "servers": {
      "sqlmind": {
        "command": "python",
        "args": ["sqlmind_mcp_server.py"],
        "cwd": "/path/to/sqlmind"
      }
    }
  }
}
```

Reload Windsurf. The Cascade agent can now call all 8 sqlmind tools.

### Codex / OpenAI Plugins / HTTP clients

```bash
# Start in HTTP mode
python sqlmind_mcp_server.py --transport http --port 8765

# Point any HTTP-based MCP client to:
# http://localhost:8765
```

### Antispace / Any MCP-compatible client

Any tool that supports the MCP protocol can connect via stdio or HTTP mode. Use the command line above.

---

## Agent Framework Integrations

### Google ADK

```python
from integrations.sqlmind_google_adk import create_sqlmind_agent

agent = create_sqlmind_agent(
    schema_path="examples/schema.sqlmind.yaml",
    dialect="bigquery",
)
result = await agent.run("top 5 products by revenue last quarter")
print(result.sql)
```

Deploy to Vertex AI Agent Engine:
```bash
gcloud ai agents deploy sqlmind-agent \
  --region=us-central1 \
  --source=integrations/sqlmind_google_adk.py
```

### OpenAI Agents SDK

```python
from integrations.sqlmind_openai_agents import build_sql_agent

agent = build_sql_agent(schema_yaml="examples/schema.sqlmind.yaml")
result = agent.run("monthly revenue by region")
print(result.final_output)
```

### LangGraph + FastAPI

```bash
# Start the streaming FastAPI server
uvicorn integrations.sqlmind_langgraph:app --reload --port 8000

# POST a query
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "orders placed last week", "dialect": "postgresql"}'
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[db,graph,dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sqlmind_graph --cov-report=term-missing
```

Current suite: **18 tests, all passing** (14 graph + 4 dialect).

---

## Deploying the MCP Server

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[db,graph]"
EXPOSE 8765
CMD ["python", "sqlmind_mcp_server.py", "--transport", "http", "--port", "8765"]
```

```bash
docker build -t sqlmind-mcp .
docker run -p 8765:8765 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY sqlmind-mcp
```

### Railway

```toml
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python sqlmind_mcp_server.py --transport http --port $PORT"
healthcheckPath = "/"
restartPolicyType = "ON_FAILURE"
```

---

## CLI Reference

```
python sqlmind_graph.py <command> [options]

Commands:
  inspect    <file>              Print schema stats + DSL
  join-path  <file> <t1> <t2>   Find shortest join path between two tables
  link       <file> <query>      Schema-link a natural language query
  erd        <file>              Export Mermaid ERD to stdout
  generate   <file> <query>      Generate SQL via Anthropic API
               --dialect         Target SQL dialect (default: postgresql)
               --model           Model ID (default: claude-sonnet-4-6)
```

---

## Contributors

Thanks to everyone who has contributed to SQLMind!

<a href="https://github.com/Veloce-AI/sqlmind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Veloce-AI/sqlmind" />
</a>

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to:
- Add a new SQL dialect to `dialects.yaml`
- Write a new validation rule
- Add agent framework integrations
- Run the test suite and open a PR

---

<div align="center">

**SQLMind** · SQL intelligence layer for LLM agents

Developed with ♥ by [VeloceAI.in](https://veloceai.in/) — open source for the community

[GitHub](https://github.com/Veloce-AI/sqlmind) · [Discord](https://discord.gg/veloceai) · [MIT License](LICENSE)

</div>
