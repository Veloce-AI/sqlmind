# SQLMind — SQL Intelligence Skill for LLM Agents
> Version 2.0 | Graph-backed schema · 9 dialects · Phase-locked generation

---

## 1. PURPOSE

SQLMind solves five root causes of LLM SQL failure:

| Failure Mode | Root Cause | SQLMind Fix |
|---|---|---|
| Wrong table/column names | No schema grounding | Schema Graph — verified node lookup |
| Bad JOIN logic | Can't infer FK paths | Graph path-finder — BFS over FK edges |
| WHERE vs HAVING confusion | Wrong execution order | Phase-locked 7-step protocol |
| Broken aggregate queries | Reasoning in write-order | Execution-order internal reasoning |
| Dialect syntax errors | One-size-fits-all output | Dialect Registry (9 engines, user-editable) |

---

## 2. SCHEMA GRAPH (Core Data Structure)

The schema is a **property graph** — not a flat list of tables.

```
Nodes:
  TableNode   → name, row_count, partitioning, description
  ColumnNode  → name, type, PK, FK, IDX, nullable, enum_values

Edges:
  FK Edge     → from_table.col → to_table.col (explicit foreign key)
  Inferred    → col ending in _id → likely FK even without constraint
```

### 2.1 Schema DSL Format (load into graph)

```
TABLE orders (
  id          INT         PK
  customer_id INT         FK→customers.id  IDX
  product_id  INT         FK→products.id
  amount      DECIMAL     IDX
  status      VARCHAR     [pending, confirmed, shipped, cancelled]
  created_at  TIMESTAMP   IDX
)

TABLE customers (
  id     INT     PK
  name   VARCHAR
  region VARCHAR IDX
  tier   VARCHAR [bronze, silver, gold]
)
```

Legend: `PK` primary key · `FK→table.col` foreign key · `IDX` indexed · `[vals]` enum

### 2.2 Graph Operations (what the LLM can reason about)

**Join path discovery**: Given any two tables, the graph finds the shortest FK path:
```
orders → customers   (1 hop: orders.customer_id = customers.id)
orders → products    (1 hop: orders.product_id = products.id)
invoices → products  (2 hops: invoices.order_id = orders.id → orders.product_id = products.id)
```

**Schema linking**: The graph maps NL words → matched table/column nodes.

**Column validation**: The graph checks every `table.column` reference in generated SQL exists.

---

## 3. DIALECT REGISTRY

SQLMind supports 9 dialects. Each is defined in `dialects.yaml` (user-editable).

| Dialect ID | Name | Key Differences |
|---|---|---|
| `postgresql` | PostgreSQL | `ILIKE`, `::` cast, `RETURNING`, `DISTINCT ON`, `generate_series` |
| `mysql` | MySQL / MariaDB | backticks, `GROUP_CONCAT`, no `ILIKE`, `AUTO_INCREMENT` |
| `sqlite` | SQLite | no `RIGHT JOIN`, no `FULL OUTER JOIN`, `strftime()`, permissive `GROUP BY` |
| `mssql` | SQL Server (T-SQL) | `TOP (n)`, `GETDATE()`, `CROSS APPLY`, `[brackets]`, `OFFSET/FETCH` |
| `bigquery` | Google BigQuery | backtick tables, `QUALIFY`, `ARRAY_AGG`, `EXCEPT DISTINCT`, `STRUCT` |
| `snowflake` | Snowflake | `QUALIFY`, native `ILIKE`, `LISTAGG`, `LATERAL FLATTEN`, `TRY_CAST` |
| `redshift` | Amazon Redshift | `ILIKE`, `LISTAGG`, `DISTKEY/SORTKEY`, PG-derived but limited |
| `databricks` | Databricks SQL | `COLLECT_LIST`, `LATERAL VIEW EXPLODE`, `RLIKE`, Unity Catalog 3-level namespace |
| `spark_sql` | Apache Spark SQL | `COLLECT_LIST`, `LATERAL VIEW EXPLODE`, `RLIKE`, Hive-compatible |

### 3.1 Key Dialect Rules (Always Apply)

**LIMIT / Pagination**
- PostgreSQL, MySQL, SQLite, BigQuery, Snowflake, Redshift, Databricks, Spark: `LIMIT n`
- SQL Server: `TOP (n)` (simple) or `OFFSET 0 ROWS FETCH NEXT n ROWS ONLY` (pagination)

**Identifier Quoting**
- PostgreSQL, Snowflake, Redshift: `"double_quotes"`
- MySQL, Databricks, Spark, BigQuery: `` `backticks` ``
- SQL Server: `[square brackets]`
- BigQuery full reference: `` `project.dataset.table` ``
- Databricks full reference: `` `catalog`.`schema`.`table` ``

**String Aggregation**
- PostgreSQL, BigQuery, SQL Server 2017+: `STRING_AGG(col, ',')`
- MySQL: `GROUP_CONCAT(col SEPARATOR ',')`
- Snowflake, Redshift: `LISTAGG(col, ',') WITHIN GROUP (ORDER BY col)`
- Databricks, Spark: `COLLECT_LIST(col)` → wrap with `ARRAY_JOIN(..., ',')`

**Case-Insensitive Search**
- Has native ILIKE: PostgreSQL, Snowflake, Redshift
- Use `LOWER(col) LIKE LOWER('%val%')` in: MySQL, SQLite, SQL Server, BigQuery, Databricks, Spark
- BigQuery alternative: `REGEXP_CONTAINS(col, r'(?i)pattern')`

**Window Function Post-Filter (QUALIFY)**
- Supported: BigQuery, Snowflake
- All others: wrap in a CTE or subquery with `WHERE rn = 1`

**Date / Time Functions**
- Current time: `NOW()` (PG, MySQL) · `CURRENT_TIMESTAMP()` (BQ, Snowflake, Databricks, Spark) · `GETDATE()` (SQL Server, Redshift) · `datetime('now')` (SQLite)
- Date trunc: `DATE_TRUNC('month', col)` (PG, BQ, Snowflake, Databricks, Spark) · `TRUNC(col, 'MM')` (Oracle) · `DATE_FORMAT(col, '%Y-%m-01')` (MySQL) · `strftime('%Y-%m-01', col)` (SQLite)

**Array Operations**
- Full arrays: BigQuery (`ARRAY_AGG`, `UNNEST`), Snowflake (`ARRAY_AGG`, `FLATTEN`), Databricks/Spark (`COLLECT_LIST`, `EXPLODE`)
- No native arrays: SQL Server, Redshift, MySQL, SQLite → use `STRING_AGG` / `GROUP_CONCAT`

**Regex**
- PostgreSQL: `col ~ 'pattern'` (POSIX) or `col ~* 'pattern'` (case-insensitive)
- MySQL, SQLite: `col REGEXP 'pattern'`
- BigQuery: `REGEXP_CONTAINS(col, r'pattern')` (RE2 syntax)
- Databricks, Spark: `col RLIKE 'pattern'` (Java regex)
- SQL Server: no native regex; use `LIKE` or CLR

---

## 4. PHASE-LOCKED GENERATION PROTOCOL

**ALWAYS reason in execution order. Write in write order.**

```
EXECUTION ORDER (reason in this sequence):
  1. FROM / JOIN  →  Identify working dataset from graph join path
  2. WHERE        →  Row-level filters (NO aggregates, NO SELECT aliases)
  3. GROUP BY     →  Aggregation groups
  4. HAVING       →  Group-level filters (aggregated values only)
  5. SELECT       →  Output columns, aggregations, alias definitions
  6. ORDER BY     →  Sort (SELECT aliases ARE available here)
  7. LIMIT        →  Row cap (use dialect-correct syntax)

WRITE ORDER (the SQL you output):
  SELECT ... FROM ... JOIN ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT ...
```

### 4.1 Phase Reasoning Template

Before writing any SQL, complete this internally:

```
[PHASE 1 — FROM/JOIN]
  Tables needed: ___
  Graph join path: [use find_join_path output]
  Join type: INNER | LEFT | RIGHT
  ⚠️ Every JOIN needs an ON clause. No Cartesian products.

[PHASE 2 — WHERE]
  Row-level filters: ___
  ⛔ No aggregate functions here (COUNT, SUM, AVG, MAX, MIN → HAVING)
  ⛔ No SELECT aliases here (they don't exist yet)
  Use original column names from the schema graph.

[PHASE 3 — GROUP BY]
  Grouping columns: ___
  ⚠️ Every non-aggregated SELECT column must appear here.

[PHASE 4 — HAVING]
  Group-level filters: ___
  ✅ Aggregate functions ARE allowed here.

[PHASE 5 — SELECT]
  Output columns: ___
  Aggregations: ___
  Aliases defined here (available in ORDER BY, not in WHERE/GROUP BY).

[PHASE 6 — ORDER BY]
  Sort: ___
  ✅ SELECT aliases are available here.

[PHASE 7 — LIMIT]
  Row cap: ___  [Use dialect: postgresql→LIMIT n, mssql→TOP(n) or OFFSET/FETCH]
  Always pair with ORDER BY for deterministic results.
```

---

## 5. COMPLEXITY CLASSIFICATION

| Level | Signal | Strategy |
|---|---|---|
| L1 Simple | Single table, no aggregation | Direct generation |
| L2 Moderate | 2-3 tables, simple GROUP BY | Phase-locked, direct |
| L3 Complex | 4+ tables, nested subqueries, window functions | CTE decomposition |
| L4 Analytical | Self-joins, correlated subqueries, QUALIFY | Multi-step CTE + schema graph path |

**L3/L4: Always decompose into CTEs**
```sql
WITH
  step_1 AS (
    -- Sub-question: [describe what this answers]
    SELECT ...
  ),
  step_2 AS (
    -- Sub-question: [describe what this answers]
    SELECT ... FROM step_1 ...
  )
SELECT * FROM step_2;
```

---

## 6. ANTI-HALLUCINATION GUARDS

Before finalizing any SQL, verify:

```
SCHEMA GUARDS (check against the graph)
  □ Every table name exists as a TableNode in the graph
  □ Every column name exists in its TableNode
  □ Every FK join follows a validated FK Edge or inferred edge

PHASE GUARDS
  □ No aggregate function in WHERE → move to HAVING
  □ No SELECT alias used in WHERE or GROUP BY → use original column name
  □ All non-aggregated SELECT columns appear in GROUP BY
  □ Every JOIN has an explicit ON clause
  □ LEFT JOIN + WHERE on right-side column → move filter to ON clause

DIALECT GUARDS
  □ LIMIT syntax correct for target dialect (not just LIMIT for mssql)
  □ Date functions use dialect-correct form (NOW vs GETDATE vs CURRENT_TIMESTAMP)
  □ String aggregation uses correct function for dialect
  □ Window filter: use QUALIFY only for BigQuery/Snowflake; else CTE

PERFORMANCE GUARDS
  □ SELECT * → enumerate specific columns in production
  □ ORDER BY without LIMIT → note it will sort all rows
  □ Function on indexed column in WHERE → use range instead
  □ Subquery with IN → consider EXISTS for short-circuit evaluation
```

---

## 7. COMMON ERROR PATTERNS & GRAPH-AWARE FIXES

### Error: Missing intermediate JOIN table (graph path needed)
```sql
-- WRONG — no direct FK between invoices and products
SELECT i.id, p.name FROM invoices i
INNER JOIN products p ON i.product_id = p.id;  -- product_id doesn't exist on invoices!

-- CORRECT — graph reveals: invoices → orders → order_items → products
SELECT i.id, p.name
FROM invoices i
INNER JOIN orders o ON i.order_id = o.id
INNER JOIN order_items oi ON o.id = oi.order_id
INNER JOIN products p ON oi.product_id = p.id;
```

### Error: Aggregate in WHERE
```sql
-- WRONG: WHERE AVG(salary) > 60000
-- CORRECT: GROUP BY department HAVING AVG(salary) > 60000
```

### Error: SELECT alias in WHERE
```sql
-- WRONG: SELECT salary * 1.1 AS adj FROM emp WHERE adj > 50000
-- CORRECT (CTE): WITH base AS (SELECT salary * 1.1 AS adj FROM emp) SELECT * FROM base WHERE adj > 50000
```

### Error: LEFT JOIN nullified by WHERE
```sql
-- WRONG: ... LEFT JOIN orders o ... WHERE o.status = 'active'
-- CORRECT: ... LEFT JOIN orders o ON ... AND o.status = 'active'
```

### Error: QUALIFY not available
```sql
-- WRONG for PostgreSQL/MySQL/SQL Server:
SELECT *, ROW_NUMBER() OVER(...) AS rn FROM t QUALIFY rn = 1;

-- CORRECT (all dialects except BigQuery/Snowflake):
WITH ranked AS (SELECT *, ROW_NUMBER() OVER(...) AS rn FROM t)
SELECT * FROM ranked WHERE rn = 1;
```

### Error: Wrong date function for dialect
```sql
-- WRONG for BigQuery: WHERE created_at >= NOW() - INTERVAL '30 days'
-- CORRECT for BigQuery: WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)

-- WRONG for SQL Server: WHERE created_at >= NOW() - INTERVAL '30 days'
-- CORRECT for SQL Server: WHERE created_at >= DATEADD(day, -30, GETDATE())
```

---

## 8. SELF-CORRECTION LOOP

If generated SQL fails execution:

```
ERROR RECEIVED: [paste error]

Step 1: Classify error type
  Syntax error → check dialect rules (Section 3.1)
  Column not found → re-run graph schema linking (Section 2.2)
  Aggregation error → re-check GROUP BY completeness (Phase 3)
  Type mismatch → check join key types in the graph nodes
  No join path → use graph find_join_path to discover intermediate tables
  Dialect error → check dialects.yaml for correct function/syntax

Step 2: Fix only the failing clause
  Do not rewrite the whole query.
  Re-validate the fix against Section 6 guards.

Step 3: Re-run the 7-phase mental trace from Section 4.1
```

---

## 9. INTEGRATION

### Claude Code / Cursor / Windsurf
- `SKILL.md` in `.claude/skills/sqlmind/` — auto-loaded by Claude Code
- `CLAUDE.md` in project root — activates SQL protocol for all sessions
- `schema.sqlmind.yaml` in project root — graph loads automatically

### MCP Tools (sqlmind_mcp_server.py)
- `sqlmind_introspect` → live DB → schema graph DSL
- `sqlmind_validate` → 7-phase + dialect validation
- `sqlmind_build_prompt` → phase-locked prompt builder
- `sqlmind_transpile` → dialect-to-dialect conversion
- `sqlmind_score_complexity` → L1–L4 + strategy

### Python ADK (sqlmind_graph.py + sqlmind.py)
```python
from sqlmind_graph import SchemaGraph, DialectRegistry

graph = SchemaGraph().load_from_db("postgresql://...")
path = graph.find_join_path("orders", "products")
print(path.to_sql())

registry = DialectRegistry("dialects.yaml")
dialect = registry.get("snowflake")
print(dialect.render_limit(50))  # → LIMIT 50
print(dialect.notes)              # → Snowflake-specific gotchas
```

---

## 10. USER-EDITABLE FILES

| File | What to edit |
|---|---|
| `dialects.yaml` | Add dialects, override functions, add custom house-style functions |
| `schema.sqlmind.yaml` | Your database schema graph (auto-generated or hand-written) |
| `CLAUDE.md` | Set default dialect, schema path, project-specific SQL rules |

---

*SQLMind Skill v2.0 | VeloceAI | Graph-backed · 9 dialects · Phase-locked*
