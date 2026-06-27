import pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlmind_graph import SchemaGraph

DSL = """
TABLE orders (
  id          INT     PK
  customer_id INT     FK→customers.id  IDX
  amount      DECIMAL
  status      VARCHAR [pending, confirmed, cancelled]
  created_at  TIMESTAMP IDX
)
TABLE customers (
  id     INT     PK
  name   VARCHAR
  region VARCHAR IDX
)
TABLE order_items (
  id         INT  PK
  order_id   INT  FK→orders.id
  product_id INT  FK→products.id
)
TABLE products (
  id    INT     PK
  name  VARCHAR
  price DECIMAL
)
"""

DDL = """
CREATE TABLE customers (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount      NUMERIC(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"""

@pytest.fixture
def graph():
    g = SchemaGraph()
    g.load_from_dsl(DSL)
    return g


# ── Original tests ────────────────────────────────────────────────────────────

def test_load(graph):           assert len(graph.tables) == 4
def test_pk(graph):             assert graph.get_table("orders").get_column("id").is_pk
def test_enum(graph):           assert "pending" in graph.get_table("orders").get_column("status").enum_values
def test_fk_edges(graph):       assert any(e.from_table == "orders" for e in graph.edges)
def test_direct_path(graph):    assert graph.find_join_path("orders", "customers").is_direct
def test_multihop_path(graph):  assert len(graph.find_join_path("orders", "products").hops) == 2
def test_schema_link(graph):    assert graph.schema_link("revenue by region")["matched_tables"]
def test_mermaid(graph):        assert "erDiagram" in graph.to_mermaid()
def test_stats(graph):          assert graph.stats()["tables"] == 4
def test_col_validation(graph): assert any(e["column"] == "bad_col" for e in graph.validate_sql_columns("SELECT o.bad_col FROM orders o"))


# ── Extended tests ────────────────────────────────────────────────────────────

def test_load_from_ddl():
    """SchemaGraph.load_from_ddl parses CREATE TABLE statements correctly."""
    g = SchemaGraph().load_from_ddl(DDL, dialect="postgresql")
    assert "customers" in g.tables
    assert "orders" in g.tables
    customers = g.get_table("customers")
    assert customers.get_column("id") is not None
    assert customers.get_column("name") is not None


def test_to_dict_from_dict_roundtrip(graph):
    """to_dict / load_from_dict roundtrip preserves tables, columns, and FK edges."""
    d = graph.to_dict()
    g2 = SchemaGraph().load_from_dict(d)

    assert set(g2.tables) == set(graph.tables)
    for t_name in graph.tables:
        original_cols = {c.name for c in graph.get_table(t_name).columns}
        roundtripped_cols = {c.name for c in g2.get_table(t_name).columns}
        assert original_cols == roundtripped_cols, f"Column mismatch on table {t_name}"

    assert len(g2.edges) == len(graph.edges)


def test_find_all_join_paths_three_tables(graph):
    """find_all_join_paths returns a path for every pair in a 3-table list."""
    tables = ["orders", "customers", "products"]
    paths = graph.find_all_join_paths(tables)

    assert ("orders", "customers") in paths
    assert ("orders", "products") in paths
    assert ("customers", "products") in paths

    # orders → customers is direct (1 hop)
    assert paths[("orders", "customers")].is_direct
    # orders → products requires passing through order_items (2 hops)
    assert len(paths[("orders", "products")].hops) == 2
    # customers → products has no direct FK; path exists via orders+order_items
    assert paths[("customers", "products")] is not None


def test_validate_sql_columns_no_false_positive_on_alias(graph):
    """validate_sql_columns must NOT flag a valid column accessed via a table alias."""
    # c.region is valid (customers.region exists); alias 'c' → customers
    errors = graph.validate_sql_columns(
        "SELECT c.region, COUNT(*) FROM customers c GROUP BY c.region"
    )
    assert not any(e["column"] == "region" for e in errors), (
        "False positive: 'region' is a real column on customers but was flagged"
    )
