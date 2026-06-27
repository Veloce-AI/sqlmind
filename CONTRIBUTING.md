# Contributing to SQLMind

SQLMind is MIT licensed and open to contributions from the community.
Whether you're fixing a bug, adding a dialect, improving validation, or writing docs — you're welcome here.

Developed with ♥ by [VeloceAI.in](https://veloceai.in/) — open source for the community.

---

## Table of contents

- [Quick start](#quick-start)
- [What to contribute](#what-to-contribute)
- [Adding a new SQL dialect](#adding-a-new-sql-dialect)
- [Adding a validation rule](#adding-a-validation-rule)
- [Writing tests](#writing-tests)
- [Running the MCP server locally](#running-the-mcp-server-locally)
- [Code style](#code-style)
- [Pull request checklist](#pull-request-checklist)
- [Commit message format](#commit-message-format)
- [Getting help](#getting-help)

---

## Quick start

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/sqlmind.git
cd sqlmind

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[db,graph,dev]"

# 4. Copy the env template
cp .env.example .env
# Add your API key if you want to test the LLM generation features

# 5. Run the full test suite — should be green before you start
pytest tests/ -v

# 6. Create your branch
git checkout -b feat/your-feature-name
```

---

## What to contribute

| Area | Examples |
|---|---|
| New dialect | Oracle, DuckDB, Trino, CockroachDB, PlanetScale |
| Validation rule | New anti-pattern, dialect-specific check |
| Schema graph | Improved inference, new export format |
| Agent integration | CrewAI, AutoGen, Semantic Kernel |
| Tests | More coverage, edge cases, large schema tests |
| Docs | Examples, tutorials, dialect notes |
| Bug fix | Anything in the issues list |

Start with an issue before writing code for large changes, so we can discuss the approach first.

---

## Adding a new SQL dialect

This is the most common contribution. All dialects live in `dialects.yaml`.

### Step 1 — Copy an existing entry

Open `dialects.yaml` and copy the block that's closest to your target dialect.
The `postgresql` entry is the most complete reference.

### Step 2 — Fill in the required fields

Every dialect entry needs these fields:

```yaml
- id: your_dialect          # snake_case, used in code and CLI
  name: "Your Dialect"      # display name
  aliases: [alias1, alias2] # other names users might type

  identifiers:
    quote_char: '"'          # how to quote reserved words: " or ` or []
    case_sensitive: true     # are identifiers case-sensitive by default?

  limit:
    syntax: "LIMIT %n OFFSET %o"   # %n = row count, %o = offset
    no_offset: "LIMIT %n"           # version without offset

  date_now: "NOW()"                         # current timestamp expression
  date_trunc: "DATE_TRUNC('%part', %col)"   # %part = month/day/year, %col = column
  date_add: "%col + INTERVAL '%n %unit'"    # %n = number, %unit = day/month/year
  date_diff: "DATEDIFF(%unit, %col1, %col2)"
  cast: "CAST(%col AS %type)"
  string_agg: "STRING_AGG(%col, '%sep')"

  ilike: false              # true if dialect has native ILIKE
  qualify_clause: false     # true for BigQuery/Snowflake QUALIFY
  window_support: true      # supports window functions
  cte_support: true         # supports WITH (CTE)
  except_syntax: EXCEPT     # or EXCEPT DISTINCT (BigQuery)

  schema_query: |
    SELECT table_name FROM information_schema.tables ...

  column_query: |
    SELECT column_name, data_type FROM information_schema.columns
    WHERE table_name = '%table' ...

  explain_prefix: "EXPLAIN"

  notes: |
    - Key gotchas the LLM needs to know
    - One per line, starting with a dash
    - Keep each note under 100 characters

  custom_fns: {}            # user-defined shorthand expansions
```

### Step 3 — Add a test

In `tests/test_dialects.py`, add your dialect to the `test_all_nine_dialects` list and add specific checks:

```python
def test_your_dialect_limit():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry("dialects.yaml")
    d = r.get("your_dialect")
    assert d is not None
    assert "YOUR_LIMIT_SYNTAX" in d.render_limit(10)

def test_your_dialect_notes():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry("dialects.yaml")
    d = r.get("your_dialect")
    assert len(d.notes) > 50   # notes should be substantive
```

### Step 4 — Verify it works

```bash
python sqlmind_graph.py inspect examples/schema.sqlmind.yaml --dialect your_dialect
pytest tests/test_dialects.py -v -k your_dialect
```

---

## Adding a validation rule

Validation rules live in two places:

- `sqlmind_graph.py` → `SchemaGraph.validate_sql_columns()` — schema-based checks
- `sqlmind_mcp_server.py` → `sqlmind_validate()` — phase and syntax checks
- `integrations/sqlmind_openai_agents.py` → `validate_sql()` — agent-side checks

All three should be kept in sync for any new rule.

### Rule structure

```python
# Each rule follows this pattern:
# 1. detect the pattern in sql_upper (the SQL in UPPERCASE)
# 2. append to errors[] or warnings[] with a clear code + message + fix
# 3. add a test that creates a SQL string triggering the rule

# Example: detect DISTINCT in GROUP BY (redundant)
if "GROUP BY" in sql_upper and re.search(r'SELECT\s+DISTINCT', sql_upper):
    warnings.append({
        "code": "DISTINCT_WITH_GROUPBY",
        "message": "SELECT DISTINCT with GROUP BY is redundant — GROUP BY already deduplicates",
        "fix": "Remove DISTINCT when using GROUP BY"
    })
```

### Error codes

Use SCREAMING_SNAKE_CASE. Existing codes to follow as a pattern:
`AGG_IN_WHERE`, `HAVING_NO_GROUPBY`, `CARTESIAN_JOIN`, `LEFT_JOIN_NULLIFIED`,
`SELECT_STAR`, `ORDER_NO_LIMIT`, `COLUMN_NOT_FOUND`, `WRONG_DIALECT_LIMIT`

---

## Writing tests

Tests live in `tests/`. We use `pytest` with no extra plugins required.

```bash
# run everything
pytest tests/ -v

# run a specific file
pytest tests/test_graph.py -v

# run a specific test
pytest tests/test_graph.py::test_direct_join_path -v

# run with coverage
pytest tests/ --cov=sqlmind_graph --cov-report=term-missing
```

### Test patterns

```python
# Graph test pattern
def test_your_feature():
    from sqlmind_graph import SchemaGraph
    g = SchemaGraph()
    g.load_from_dsl("""
    TABLE orders (
      id INT PK
      customer_id INT FK→customers.id
    )
    TABLE customers (
      id INT PK
    )
    """)
    # your assertion
    result = g.find_join_path("orders", "customers")
    assert result is not None
    assert result.is_direct is True

# Dialect test pattern
@pytest.mark.skipif(not Path("dialects.yaml").exists(), reason="dialects.yaml not found")
def test_dialect_feature():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry("dialects.yaml")
    d = r.get("postgresql")
    assert "DATE_TRUNC" in d.render_date_trunc("col", "month")
```

### What to test for new dialects

- `render_limit(n)` returns the correct syntax
- `render_limit(n, offset)` handles offset correctly
- `supports_ilike` is correct (True/False)
- `supports_qualify` is correct
- `notes` is non-empty and contains key gotchas
- `render_date_trunc` produces a valid expression

---

## Running the MCP server locally

```bash
# stdio mode (for Claude Code / Cursor)
python sqlmind_mcp_server.py

# HTTP mode (for testing in a browser or with curl)
python sqlmind_mcp_server.py --transport http --port 8765

# test a tool directly
curl -X POST http://localhost:8765/tools/sqlmind_validate \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT COUNT(*) FROM orders WHERE COUNT(*) > 5", "dialect": "postgresql"}'
```

To add the server to Claude Code during development:
```bash
claude mcp add sqlmind-dev --command python --args sqlmind_mcp_server.py
```

---

## Code style

We use `ruff` for linting and formatting.

```bash
# check
ruff check .

# fix auto-fixable issues
ruff check . --fix

# format
ruff format .
```

Rules:
- Type hints on all public functions
- Docstrings on all public functions and classes
- No `# TODO: implement` placeholders — either implement it or open an issue
- `snake_case` for functions and variables, `PascalCase` for classes
- Maximum line length: 100 characters
- Imports: stdlib → third-party → local, alphabetical within groups

---

## Pull request checklist

Before opening a PR, verify:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `ruff check .` — no lint errors
- [ ] New feature has tests covering the happy path and at least one edge case
- [ ] New dialect has a full entry in `dialects.yaml` and tests in `test_dialects.py`
- [ ] New validation rule is added to all three locations (graph, MCP server, agent)
- [ ] `CONTRIBUTING.md` updated if you're changing the contribution process
- [ ] PR description explains what changed and why

PR title format: `type: short description`
Examples: `feat: add DuckDB dialect`, `fix: LEFT JOIN nullification false positive`, `docs: add Oracle dialect notes`

---

## Commit message format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type: short description (max 72 chars)

Optional longer body explaining why, not what.

Closes #123
```

Types: `feat` · `fix` · `docs` · `test` · `refactor` · `chore`

---

## Getting help

- Open a [GitHub Discussion](https://github.com/Veloce-AI/sqlmind/discussions) for questions
- Open a [GitHub Issue](https://github.com/Veloce-AI/sqlmind/issues) for bugs
- Join the [Discord](https://discord.gg/veloceai) for real-time help

---

Developed with ♥ by [VeloceAI.in](https://veloceai.in/) — open source for the community
