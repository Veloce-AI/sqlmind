import pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

DIALECTS_PATH = Path(__file__).parent.parent / "dialects.yaml"

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_all_ten_dialects():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    for d in ["postgresql","mysql","sqlite","mssql","bigquery","snowflake",
              "redshift","databricks","spark_sql","oracle"]:
        assert r.get(d) is not None, f"missing dialect: {d}"

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_limit_syntax():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    assert "LIMIT"      in r.get("postgresql").render_limit(10)
    assert "TOP"        in r.get("mssql").render_limit(10, 0)
    assert "FETCH FIRST" in r.get("oracle").render_limit(10)

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_ilike():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    assert r.get("postgresql").supports_ilike is True
    assert r.get("snowflake").supports_ilike is True
    assert r.get("mysql").supports_ilike is False
    assert r.get("oracle").supports_ilike is False

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_qualify():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    assert r.get("bigquery").supports_qualify is True
    assert r.get("snowflake").supports_qualify is True
    assert r.get("postgresql").supports_qualify is False
    assert r.get("oracle").supports_qualify is False

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_oracle_dialect():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    oracle = r.get("oracle")
    assert oracle is not None
    assert "FETCH FIRST" in oracle.render_limit(10)
    assert "TRUNC" in oracle.render_date_trunc("created_at", "MM")
    assert oracle.get("except_syntax") == "MINUS"
    assert len(oracle.notes) > 50
