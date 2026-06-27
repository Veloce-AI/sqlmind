import pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

DIALECTS_PATH = Path(__file__).parent.parent / "dialects.yaml"

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_all_nine_dialects():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    for d in ["postgresql","mysql","sqlite","mssql","bigquery","snowflake","redshift","databricks","spark_sql"]:
        assert r.get(d) is not None, f"missing: {d}"

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_limit_syntax():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    assert "LIMIT" in r.get("postgresql").render_limit(10)
    assert "TOP"   in r.get("mssql").render_limit(10, 0)

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_ilike():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    assert r.get("postgresql").supports_ilike is True
    assert r.get("mysql").supports_ilike is False

@pytest.mark.skipif(not DIALECTS_PATH.exists(), reason="dialects.yaml not found")
def test_qualify():
    from sqlmind_graph import DialectRegistry
    r = DialectRegistry(str(DIALECTS_PATH))
    assert r.get("bigquery").supports_qualify is True
    assert r.get("postgresql").supports_qualify is False
