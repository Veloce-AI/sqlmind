"""
sqlmind_graph.py
────────────────
SQLMind Schema Graph Engine

Represents a database schema as a property graph:
  - TableNode   → table with metadata (row count estimate, partitioning, etc.)
  - ColumnNode  → column with type, constraints, index info, enum values
  - FK Edge     → foreign key relationship between column nodes
  - JoinPath    → inferred multi-hop join path between any two tables

Key capabilities:
  1. Load schema from: live DB, DDL string, or .sqlmind.yaml file
  2. Join path discovery (BFS shortest path between tables)
  3. Schema linking (map NL entities to graph nodes)
  4. Dialect-aware validation
  5. Export to SQLMind DSL, JSON, or Mermaid ERD

Install:
  pip install sqlglot pyyaml sqlalchemy networkx

Usage:
  from sqlmind_graph import SchemaGraph, DialectRegistry

  graph = SchemaGraph()
  graph.load_from_yaml("schema.sqlmind.yaml")

  path = graph.find_join_path("orders", "products")
  print(path.to_sql("postgresql"))
  # → orders INNER JOIN order_items ON orders.id = order_items.order_id
  #   INNER JOIN products ON order_items.product_id = products.id

  registry = DialectRegistry("dialects.yaml")
  dialect = registry.get("snowflake")
  print(dialect.render_limit(10, 0))
  # → LIMIT 10
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import sqlglot
    import sqlglot.errors
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


# ─── Node Types ───────────────────────────────────────────────────────────────

@dataclass
class ColumnNode:
    """A column in the schema graph."""
    table: str
    name: str
    dtype: str
    is_pk: bool = False
    is_fk: bool = False
    references: Optional[str] = None   # "other_table.other_col"
    is_indexed: bool = False
    is_nullable: bool = True
    enum_values: list[str] = field(default_factory=list)
    default_value: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.table}.{self.name}"

    def to_dsl_line(self) -> str:
        mods = []
        if self.is_pk: mods.append("PK")
        if self.is_fk and self.references: mods.append(f"FK→{self.references}")
        if self.is_indexed and not self.is_pk: mods.append("IDX")
        if not self.is_nullable: mods.append("NOT NULL")
        if self.enum_values: mods.append(f"[{', '.join(self.enum_values)}]")
        mod_str = "  " + "  ".join(mods) if mods else ""
        return f"  {self.name:<24}{self.dtype:<16}{mod_str}"


@dataclass
class TableNode:
    """A table in the schema graph."""
    name: str
    columns: list[ColumnNode] = field(default_factory=list)
    row_count: Optional[int] = None
    schema_name: Optional[str] = None
    catalog_name: Optional[str] = None
    partitioned_by: Optional[str] = None
    description: Optional[str] = None

    @property
    def primary_keys(self) -> list[ColumnNode]:
        return [c for c in self.columns if c.is_pk]

    @property
    def foreign_keys(self) -> list[ColumnNode]:
        return [c for c in self.columns if c.is_fk]

    def get_column(self, name: str) -> Optional[ColumnNode]:
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None

    def to_dsl(self) -> str:
        lines = [f"TABLE {self.name} ("]
        if self.description:
            lines.append(f"  -- {self.description}")
        for col in self.columns:
            lines.append(col.to_dsl_line())
        if self.row_count is not None:
            lines.append(f"  -- ~{self.row_count:,} rows")
        lines.append(")")
        return "\n".join(lines)


@dataclass
class FKEdge:
    """A foreign key relationship edge in the schema graph."""
    from_table: str
    from_col: str
    to_table: str
    to_col: str
    join_type: str = "INNER"  # recommended join type

    @property
    def from_full(self) -> str:
        return f"{self.from_table}.{self.from_col}"

    @property
    def to_full(self) -> str:
        return f"{self.to_table}.{self.to_col}"

    def to_sql(self, from_alias: str = "", to_alias: str = "") -> str:
        fa = from_alias or self.from_table
        ta = to_alias or self.to_table
        return (
            f"{self.join_type} JOIN {self.to_table}"
            + (f" {to_alias}" if to_alias and to_alias != self.to_table else "")
            + f" ON {fa}.{self.from_col} = {ta}.{self.to_col}"
        )


@dataclass
class JoinPath:
    """A computed join path between two tables (may span multiple hops)."""
    from_table: str
    to_table: str
    hops: list[FKEdge] = field(default_factory=list)
    is_direct: bool = True
    confidence: str = "high"  # high / medium / low

    def to_sql(self, dialect_id: str = "postgresql") -> str:
        """Render the join path as SQL JOIN clauses."""
        if not self.hops:
            return ""
        parts = [self.from_table]
        for hop in self.hops:
            parts.append(hop.to_sql())
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "from": self.from_table,
            "to": self.to_table,
            "hops": len(self.hops),
            "is_direct": self.is_direct,
            "confidence": self.confidence,
            "joins": [{"from": h.from_full, "to": h.to_full, "type": h.join_type}
                      for h in self.hops]
        }


# ─── Schema Graph ─────────────────────────────────────────────────────────────

class SchemaGraph:
    """
    Property graph representation of a database schema.

    Nodes:
      - TableNode (one per table)
      - ColumnNode (one per column, owned by a TableNode)

    Edges:
      - FKEdge (foreign key constraint between columns)
      - Inferred edges (when no explicit FK exists but pattern suggests a join)

    The graph is stored as:
      - self.tables: dict[str, TableNode]
      - self.edges: list[FKEdge]
      - self._nx_graph: networkx.Graph (for BFS pathfinding)
    """

    def __init__(self):
        self.tables: dict[str, TableNode] = {}
        self.edges: list[FKEdge] = []
        self._adjacency: dict[str, list[FKEdge]] = {}  # fallback if no networkx

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_from_yaml(self, path: str) -> "SchemaGraph":
        """Load schema from a .sqlmind.yaml file."""
        if not HAS_YAML:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._from_dict(data)

    def load_from_dict(self, data: dict) -> "SchemaGraph":
        """Load schema from a Python dict (schema dict format)."""
        return self._from_dict(data)

    def load_from_dsl(self, dsl: str) -> "SchemaGraph":
        """
        Parse a SQLMind Schema DSL string.

        Format:
          TABLE orders (
            id          INT     PK
            customer_id INT     FK→customers.id  IDX
            amount      DECIMAL
            status      VARCHAR [pending, confirmed, cancelled]
          )
        """
        current_table: Optional[TableNode] = None

        for raw_line in dsl.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("--"):
                continue

            table_m = re.match(r'TABLE\s+(\w+)\s*\(', line, re.IGNORECASE)
            if table_m:
                current_table = TableNode(name=table_m.group(1))
                self.tables[current_table.name.lower()] = current_table
                continue

            if line == ")" and current_table:
                current_table = None
                continue

            if current_table and line:
                parts = line.split()
                if not parts:
                    continue
                col_name = parts[0]
                col_type = parts[1] if len(parts) > 1 else "TEXT"

                is_pk = "PK" in parts
                is_fk = any(p.startswith("FK→") for p in parts)
                is_idx = "IDX" in parts
                is_not_null = "NOT" in parts and "NULL" in parts

                fk_ref = None
                for p in parts:
                    if p.startswith("FK→"):
                        fk_ref = p[3:]  # e.g. "customers.id"
                        break

                enum_vals = []
                enum_m = re.search(r'\[([^\]]+)\]', line)
                if enum_m:
                    enum_vals = [v.strip() for v in enum_m.group(1).split(",")]

                col = ColumnNode(
                    table=current_table.name,
                    name=col_name,
                    dtype=col_type,
                    is_pk=is_pk,
                    is_fk=is_fk,
                    references=fk_ref,
                    is_indexed=is_idx,
                    is_nullable=not is_not_null,
                    enum_values=enum_vals,
                )
                current_table.columns.append(col)

                if is_fk and fk_ref:
                    ref_parts = fk_ref.split(".")
                    if len(ref_parts) == 2:
                        edge = FKEdge(
                            from_table=current_table.name,
                            from_col=col_name,
                            to_table=ref_parts[0],
                            to_col=ref_parts[1],
                        )
                        self.edges.append(edge)

        self._build_adjacency()
        return self

    def load_from_ddl(self, ddl: str, dialect: str = "postgresql") -> "SchemaGraph":
        """Parse CREATE TABLE DDL into the schema graph via sqlglot."""
        if not HAS_SQLGLOT:
            raise ImportError("pip install sqlglot")

        dialect_map = {"postgresql": "postgres", "mysql": "mysql",
                       "mssql": "tsql", "bigquery": "bigquery",
                       "snowflake": "snowflake", "sqlite": "sqlite",
                       "spark_sql": "spark", "databricks": "databricks"}
        glot_dialect = dialect_map.get(dialect.lower(), "postgres")

        statements = sqlglot.parse(ddl, dialect=glot_dialect)
        for stmt in statements:
            if stmt is None:
                continue
            if not hasattr(stmt, "find"):
                continue

            table_exp = stmt.find(sqlglot.exp.Table)
            if not table_exp:
                continue
            table_name = table_exp.name
            table = TableNode(name=table_name)
            self.tables[table_name.lower()] = table

            for col_def in stmt.find_all(sqlglot.exp.ColumnDef):
                col_name = col_def.name
                col_type = str(col_def.args.get("kind", "TEXT")).split("(")[0]

                constraints = col_def.args.get("constraints", [])
                is_pk = any("PrimaryKey" in type(c).__name__ for c in constraints)
                is_not_null = any("NotNull" in type(c).__name__ for c in constraints)

                col = ColumnNode(
                    table=table_name,
                    name=col_name,
                    dtype=col_type,
                    is_pk=is_pk,
                    is_nullable=not is_not_null,
                )
                table.columns.append(col)

            # Extract FK constraints from the statement
            for fk in stmt.find_all(sqlglot.exp.ForeignKey):
                try:
                    from_cols = [str(c) for c in fk.args.get("expressions", [])]
                    ref = fk.args.get("reference")
                    if ref:
                        ref_table = str(ref.find(sqlglot.exp.Table).name)
                        ref_cols = [str(c) for c in ref.find_all(sqlglot.exp.Column)]
                        for fc, rc in zip(from_cols, ref_cols):
                            edge = FKEdge(from_table=table_name, from_col=fc,
                                          to_table=ref_table, to_col=rc)
                            self.edges.append(edge)
                            # Mark column as FK
                            col_obj = table.get_column(fc)
                            if col_obj:
                                col_obj.is_fk = True
                                col_obj.references = f"{ref_table}.{rc}"
                except Exception:
                    pass

        self._build_adjacency()
        self._infer_edges()
        return self

    def load_from_db(
        self,
        connection_string: str,
        include_row_counts: bool = False,
        schemas_filter: Optional[list[str]] = None,
    ) -> "SchemaGraph":
        """Introspect a live database into the schema graph."""
        try:
            from sqlalchemy import create_engine, inspect, text
        except ImportError:
            raise ImportError("pip install sqlalchemy")

        engine = create_engine(connection_string)
        inspector = inspect(engine)

        for table_name in inspector.get_table_names():
            pk = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))
            fks = {}
            for fk in inspector.get_foreign_keys(table_name):
                for lc, rc in zip(fk["constrained_columns"], fk["referred_columns"]):
                    fks[lc] = (fk["referred_table"], rc)

            indexed = set()
            for idx in inspector.get_indexes(table_name):
                for c in idx.get("column_names", []):
                    indexed.add(c)

            row_count = None
            if include_row_counts:
                try:
                    with engine.connect() as conn:
                        row_count = conn.execute(
                            text(f"SELECT COUNT(*) FROM {table_name}")
                        ).scalar()
                except Exception:
                    pass

            table = TableNode(name=table_name, row_count=row_count)
            for col_info in inspector.get_columns(table_name):
                cname = col_info["name"]
                ctype = str(col_info["type"]).split("(")[0]
                fk_ref = None
                is_fk = cname in fks
                if is_fk:
                    ref_table, ref_col = fks[cname]
                    fk_ref = f"{ref_table}.{ref_col}"
                    self.edges.append(FKEdge(
                        from_table=table_name, from_col=cname,
                        to_table=ref_table, to_col=ref_col,
                    ))

                col = ColumnNode(
                    table=table_name, name=cname, dtype=ctype,
                    is_pk=cname in pk, is_fk=is_fk, references=fk_ref,
                    is_indexed=cname in indexed,
                    is_nullable=col_info.get("nullable", True),
                )
                table.columns.append(col)
            self.tables[table_name.lower()] = table

        self._build_adjacency()
        self._infer_edges()
        return self

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_to_yaml(self, path: str) -> None:
        """Save schema graph to a .sqlmind.yaml file."""
        if not HAS_YAML:
            raise ImportError("pip install pyyaml")
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dsl(self) -> str:
        """Export schema as SQLMind DSL string."""
        blocks = []
        for table in self.tables.values():
            blocks.append(table.to_dsl())
        return "\n\n".join(blocks)

    def to_dict(self) -> dict:
        """Export schema as a serializable dict."""
        return {
            "tables": {
                name: {
                    "name": t.name,
                    "schema": t.schema_name,
                    "catalog": t.catalog_name,
                    "row_count": t.row_count,
                    "partitioned_by": t.partitioned_by,
                    "description": t.description,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.dtype,
                            "pk": c.is_pk,
                            "fk": c.is_fk,
                            "references": c.references,
                            "indexed": c.is_indexed,
                            "nullable": c.is_nullable,
                            "enum": c.enum_values or None,
                            "default": c.default_value,
                        }
                        for c in t.columns
                    ]
                }
                for name, t in self.tables.items()
            },
            "edges": [
                {
                    "from_table": e.from_table,
                    "from_col": e.from_col,
                    "to_table": e.to_table,
                    "to_col": e.to_col,
                    "join_type": e.join_type,
                }
                for e in self.edges
            ]
        }

    def to_mermaid(self) -> str:
        """Export as a Mermaid ERD diagram."""
        lines = ["erDiagram"]
        for t in self.tables.values():
            lines.append(f"  {t.name} {{")
            for c in t.columns:
                pk_tag = " PK" if c.is_pk else ""
                fk_tag = " FK" if c.is_fk else ""
                lines.append(f"    {c.dtype} {c.name}{pk_tag}{fk_tag}")
            lines.append("  }")
        for e in self.edges:
            lines.append(f"  {e.from_table} ||--o{{ {e.to_table} : \"{e.from_col}→{e.to_col}\"")
        return "\n".join(lines)

    # ── Query Operations ───────────────────────────────────────────────────────

    def get_table(self, name: str) -> Optional[TableNode]:
        return self.tables.get(name.lower())

    def find_join_path(
        self,
        from_table: str,
        to_table: str,
        max_hops: int = 4,
    ) -> Optional[JoinPath]:
        """
        Find the shortest join path between two tables using BFS.
        Returns None if no path exists within max_hops.
        """
        from_lower = from_table.lower()
        to_lower = to_table.lower()

        if from_lower == to_lower:
            return JoinPath(from_table=from_table, to_table=to_table, hops=[], is_direct=True)

        if HAS_NX:
            return self._find_path_nx(from_lower, to_lower, max_hops)
        else:
            return self._find_path_bfs(from_lower, to_lower, max_hops)

    def find_all_join_paths(
        self,
        tables: list[str],
        max_hops: int = 4,
    ) -> dict[tuple[str, str], Optional[JoinPath]]:
        """Find join paths between all pairs in a list of tables."""
        result = {}
        for i, t1 in enumerate(tables):
            for t2 in tables[i+1:]:
                path = self.find_join_path(t1, t2, max_hops)
                result[(t1, t2)] = path
        return result

    def schema_link(self, nl_query: str) -> dict:
        """
        Heuristic schema linking: map NL keywords to graph nodes.
        Returns a dict of likely tables, columns, and join paths.
        """
        nl_lower = nl_query.lower()
        words = set(re.findall(r'\b\w+\b', nl_lower))

        matched_tables = []
        matched_columns = []
        matched_table_set: set[str] = set()

        for t_name, table in self.tables.items():
            # Check table name match
            if t_name in words or t_name.rstrip("s") in words:
                matched_tables.append(t_name)
                matched_table_set.add(t_name)
            # Check column name matches; table is relevant if any column matches
            for col in table.columns:
                if col.name.lower() in words:
                    matched_columns.append(col.full_name)
                    if t_name not in matched_table_set:
                        matched_tables.append(t_name)
                        matched_table_set.add(t_name)

        # Find join paths between all matched tables
        join_paths = {}
        for i, t1 in enumerate(matched_tables):
            for t2 in matched_tables[i+1:]:
                path = self.find_join_path(t1, t2)
                if path:
                    join_paths[f"{t1}→{t2}"] = path.to_dict()

        return {
            "matched_tables": matched_tables,
            "matched_columns": matched_columns,
            "join_paths": join_paths,
            "unmatched_words": [
                w for w in words
                if len(w) > 3
                and w not in {t for t in self.tables}
                and w not in {"from", "where", "group", "order", "having",
                               "select", "join", "inner", "outer", "left", "right",
                               "count", "sum", "avg", "max", "min", "with", "distinct",
                               "show", "find", "get", "list", "give", "total", "number"}
            ]
        }

    def validate_sql_columns(self, sql: str) -> list[dict]:
        """
        Check that all table.column references in a SQL query exist in the graph.
        Resolves explicit table aliases (e.g. FROM orders o) before validation.
        Returns a list of validation errors.
        """
        # Build alias → table map from FROM/JOIN clauses
        alias_map: dict[str, str] = {}
        for tbl, alias in re.findall(
            r'(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)', sql, re.IGNORECASE
        ):
            if tbl.lower() in self.tables:
                alias_map[alias.lower()] = tbl.lower()

        errors = []
        dot_refs = re.findall(r'(\b\w+\b)\.(\b\w+\b)', sql)
        for table_ref, col_ref in dot_refs:
            t_lower = table_ref.lower()
            # Resolve alias → real table name
            resolved = alias_map.get(t_lower, t_lower)
            table = self.tables.get(resolved)
            if table is None:
                # Still unknown — skip (schema name prefix, CTE name, etc.)
                continue
            col = table.get_column(col_ref)
            if col is None:
                errors.append({
                    "type": "COLUMN_NOT_FOUND",
                    "table": table.name,
                    "column": col_ref,
                    "available": [c.name for c in table.columns],
                    "suggestion": self._suggest_column(col_ref, table),
                })
        return errors

    def stats(self) -> dict:
        """Return graph statistics."""
        return {
            "tables": len(self.tables),
            "total_columns": sum(len(t.columns) for t in self.tables.values()),
            "fk_edges": len(self.edges),
            "pk_columns": sum(
                sum(1 for c in t.columns if c.is_pk)
                for t in self.tables.values()
            ),
            "indexed_columns": sum(
                sum(1 for c in t.columns if c.is_indexed)
                for t in self.tables.values()
            ),
            "tables_with_enums": sum(
                1 for t in self.tables.values()
                if any(c.enum_values for c in t.columns)
            ),
        }

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _build_adjacency(self):
        """Build adjacency list from edges for BFS."""
        self._adjacency = {name: [] for name in self.tables}
        for edge in self.edges:
            ft = edge.from_table.lower()
            tt = edge.to_table.lower()
            if ft in self._adjacency:
                self._adjacency[ft].append(edge)
            # Add reverse edge for undirected traversal
            rev = FKEdge(from_table=tt, from_col=edge.to_col,
                         to_table=ft, to_col=edge.from_col,
                         join_type=edge.join_type)
            if tt in self._adjacency:
                self._adjacency[tt].append(rev)

    def _infer_edges(self):
        """
        Infer join edges where explicit FKs don't exist
        based on naming conventions (e.g. customer_id → customers.id).
        """
        for t_name, table in self.tables.items():
            for col in table.columns:
                if col.is_fk or not col.name.endswith("_id"):
                    continue
                # e.g. customer_id → look for table "customers" or "customer"
                candidate = col.name[:-3]  # strip "_id"
                for suffix in ["s", ""]:
                    ref_table = candidate + suffix
                    ref = self.tables.get(ref_table.lower())
                    if ref:
                        pk_cols = ref.primary_keys
                        if pk_cols:
                            inferred_edge = FKEdge(
                                from_table=t_name,
                                from_col=col.name,
                                to_table=ref.name,
                                to_col=pk_cols[0].name,
                                join_type="INNER",
                            )
                            # Only add if not already present
                            existing = {(e.from_table.lower(), e.from_col.lower())
                                        for e in self.edges}
                            if (t_name.lower(), col.name.lower()) not in existing:
                                self.edges.append(inferred_edge)
                                col.is_fk = True
                                col.references = f"{ref.name}.{pk_cols[0].name}"
                        break
        self._build_adjacency()

    def _find_path_bfs(
        self, from_table: str, to_table: str, max_hops: int
    ) -> Optional[JoinPath]:
        """BFS join path finder (fallback when networkx not available)."""
        if from_table not in self._adjacency:
            return None

        queue = deque([(from_table, [])])
        visited = {from_table}

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for edge in self._adjacency.get(current, []):
                neighbor = edge.to_table.lower()
                new_path = path + [edge]
                if neighbor == to_table:
                    return JoinPath(
                        from_table=from_table,
                        to_table=to_table,
                        hops=new_path,
                        is_direct=len(new_path) == 1,
                        confidence="high" if len(new_path) == 1 else "medium",
                    )
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None

    def _find_path_nx(
        self, from_table: str, to_table: str, max_hops: int
    ) -> Optional[JoinPath]:
        """NetworkX-based shortest path finder."""
        G = nx.Graph()
        for t in self.tables:
            G.add_node(t)
        for e in self.edges:
            ft = e.from_table.lower()
            tt = e.to_table.lower()
            G.add_edge(ft, tt, edge=e)

        try:
            path_nodes = nx.shortest_path(G, from_table, to_table)
            if len(path_nodes) - 1 > max_hops:
                return None
            hops = []
            for i in range(len(path_nodes) - 1):
                edge_data = G[path_nodes[i]][path_nodes[i+1]].get("edge")
                if edge_data:
                    hops.append(edge_data)
            return JoinPath(
                from_table=from_table,
                to_table=to_table,
                hops=hops,
                is_direct=len(hops) == 1,
                confidence="high" if len(hops) == 1 else "medium",
            )
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def _suggest_column(self, col_name: str, table: TableNode) -> Optional[str]:
        """Fuzzy-match a column name to available columns."""
        col_lower = col_name.lower()
        for c in table.columns:
            if col_lower in c.name.lower() or c.name.lower() in col_lower:
                return c.name
        return None

    def _from_dict(self, data: dict) -> "SchemaGraph":
        """Load schema graph from dict."""
        for t_name, t_data in data.get("tables", {}).items():
            table = TableNode(
                name=t_data.get("name", t_name),
                schema_name=t_data.get("schema"),
                catalog_name=t_data.get("catalog"),
                row_count=t_data.get("row_count"),
                partitioned_by=t_data.get("partitioned_by"),
                description=t_data.get("description"),
            )
            for c_data in t_data.get("columns", []):
                col = ColumnNode(
                    table=table.name,
                    name=c_data["name"],
                    dtype=c_data.get("type", "TEXT"),
                    is_pk=c_data.get("pk", False),
                    is_fk=c_data.get("fk", False),
                    references=c_data.get("references"),
                    is_indexed=c_data.get("indexed", False),
                    is_nullable=c_data.get("nullable", True),
                    enum_values=c_data.get("enum") or [],
                    default_value=c_data.get("default"),
                )
                table.columns.append(col)
            self.tables[table.name.lower()] = table

        for e_data in data.get("edges", []):
            self.edges.append(FKEdge(
                from_table=e_data["from_table"],
                from_col=e_data["from_col"],
                to_table=e_data["to_table"],
                to_col=e_data["to_col"],
                join_type=e_data.get("join_type", "INNER"),
            ))

        self._build_adjacency()
        return self


# ─── Dialect Registry ─────────────────────────────────────────────────────────

class DialectConfig:
    """A loaded dialect with rendering helpers."""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.name = data["name"]
        self.aliases = data.get("aliases", [])
        self._data = data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def render_limit(self, n: int, offset: int = 0) -> str:
        """Render the LIMIT/OFFSET/TOP/FETCH clause for this dialect."""
        if self.id == "mssql":
            top = self._data.get("top", {})
            if top and offset == 0:
                return f"TOP ({n})"
            limit = self._data.get("limit", {})
            syntax = limit.get("syntax", "OFFSET %o ROWS FETCH NEXT %n ROWS ONLY")
            return syntax.replace("%n", str(n)).replace("%o", str(offset))

        limit = self._data.get("limit", {})
        if offset == 0:
            syntax = limit.get("no_offset", "LIMIT %n")
        else:
            syntax = limit.get("syntax", "LIMIT %n OFFSET %o")
        return syntax.replace("%n", str(n)).replace("%o", str(offset))

    def render_date_trunc(self, column: str, part: str) -> str:
        template = self._data.get("date_trunc", "DATE_TRUNC('%part', %col)")
        return template.replace("%col", column).replace("%part", part)

    def render_date_add(self, column: str, n: int, unit: str) -> str:
        template = self._data.get("date_add", "%col + INTERVAL '%n %unit'")
        return (template.replace("%col", column)
                        .replace("%n", str(n))
                        .replace("%unit", unit))

    def render_cast(self, column: str, dtype: str) -> str:
        template = self._data.get("cast", "CAST(%col AS %type)")
        return template.replace("%col", column).replace("%type", dtype)

    def render_string_agg(self, column: str, sep: str) -> str:
        template = self._data.get("string_agg", "STRING_AGG(%col, '%sep')")
        return template.replace("%col", column).replace("%sep", sep)

    def quote_identifier(self, name: str) -> str:
        q = self._data.get("identifiers", {}).get("quote_char", '"')
        if q == "[]":
            return f"[{name}]"
        return f"{q}{name}{q}"

    @property
    def supports_qualify(self) -> bool:
        return bool(self._data.get("qualify_clause", False))

    @property
    def supports_ilike(self) -> bool:
        return bool(self._data.get("ilike", False))

    @property
    def notes(self) -> str:
        return self._data.get("notes", "")

    @property
    def custom_fns(self) -> dict:
        return self._data.get("custom_fns", {})


class DialectRegistry:
    """
    Loads and manages dialects from dialects.yaml.

    Usage:
      registry = DialectRegistry("dialects.yaml")
      pg = registry.get("postgresql")
      print(pg.render_limit(10))
      # → LIMIT 10
    """

    def __init__(self, yaml_path: str = "dialects.yaml"):
        if not HAS_YAML:
            raise ImportError("pip install pyyaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        self._dialects: dict[str, DialectConfig] = {}
        for d in data.get("dialects", []):
            cfg = DialectConfig(d)
            self._dialects[d["id"]] = cfg
            for alias in d.get("aliases", []):
                self._dialects[alias.lower()] = cfg

    def get(self, dialect_id: str) -> Optional[DialectConfig]:
        return self._dialects.get(dialect_id.lower())

    def list_ids(self) -> list[str]:
        return [k for k, v in self._dialects.items() if v.id == k]

    def all_notes_for(self, dialect_id: str) -> str:
        d = self.get(dialect_id)
        if not d:
            return f"Unknown dialect: {dialect_id}"
        return f"## {d.name} Notes\n{d.notes}"


# ─── Quick helpers ────────────────────────────────────────────────────────────

def graph_from_dsl(dsl: str) -> SchemaGraph:
    """One-liner: parse DSL string into a SchemaGraph."""
    return SchemaGraph().load_from_dsl(dsl)

def graph_from_db(connection_string: str, **kwargs) -> SchemaGraph:
    """One-liner: introspect a live DB into a SchemaGraph."""
    return SchemaGraph().load_from_db(connection_string, **kwargs)

def graph_from_ddl(ddl: str, dialect: str = "postgresql") -> SchemaGraph:
    """One-liner: parse DDL into a SchemaGraph."""
    return SchemaGraph().load_from_ddl(ddl, dialect=dialect)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="SQLMind Graph Engine")
    sub = parser.add_subparsers(dest="cmd")

    p_load = sub.add_parser("inspect", help="Inspect a schema file")
    p_load.add_argument("file", help=".sqlmind.yaml or .sql DDL file")
    p_load.add_argument("--dialect", default="postgresql")

    p_path = sub.add_parser("join-path", help="Find join path between two tables")
    p_path.add_argument("file")
    p_path.add_argument("from_table")
    p_path.add_argument("to_table")

    p_link = sub.add_parser("link", help="Schema-link a natural language query")
    p_link.add_argument("file")
    p_link.add_argument("query")

    p_erd = sub.add_parser("erd", help="Export Mermaid ERD")
    p_erd.add_argument("file")

    p_gen = sub.add_parser("generate", help="Generate SQL from a natural language query")
    p_gen.add_argument("file", help=".sqlmind.yaml or DDL file")
    p_gen.add_argument("query", help='Natural language query, e.g. "top 10 customers by revenue"')
    p_gen.add_argument("--dialect", default="postgresql", help="Target SQL dialect")
    p_gen.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model ID")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    graph = SchemaGraph()
    if args.file.endswith(".yaml") or args.file.endswith(".yml"):
        graph.load_from_yaml(args.file)
    else:
        with open(args.file) as f:
            content = f.read()
        if "TABLE " in content.upper() and "(" in content:
            graph.load_from_dsl(content)
        else:
            graph.load_from_ddl(content, getattr(args, "dialect", "postgresql"))

    if args.cmd == "inspect":
        stats = graph.stats()
        print(f"Tables: {stats['tables']}  Columns: {stats['total_columns']}  FK edges: {stats['fk_edges']}")
        print()
        print(graph.to_dsl())

    elif args.cmd == "join-path":
        path = graph.find_join_path(args.from_table, args.to_table)
        if path:
            print(f"Join path ({len(path.hops)} hop(s), confidence: {path.confidence}):")
            print(path.to_sql())
        else:
            print(f"No join path found between {args.from_table} and {args.to_table}")

    elif args.cmd == "link":
        result = graph.schema_link(args.query)
        print(json.dumps(result, indent=2))

    elif args.cmd == "erd":
        print(graph.to_mermaid())

    elif args.cmd == "generate":
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
            sys.exit(1)
        try:
            import anthropic
        except ImportError:
            print("Error: pip install anthropic", file=sys.stderr)
            sys.exit(1)

        dialect_notes = {
            "postgresql": "Use standard PostgreSQL syntax. ILIKE for case-insensitive. :: for casting.",
            "mysql": "Use backticks for reserved words. GROUP_CONCAT instead of STRING_AGG.",
            "sqlite": "No RIGHT JOIN. strftime() for dates. Permissive GROUP BY.",
            "mssql": "Use TOP n instead of LIMIT. GETDATE() for current time. Square brackets for reserved words.",
            "bigquery": "Use backtick-quoted table names. QUALIFY for window filtering.",
            "snowflake": "Use QUALIFY for window filtering. TRY_CAST. ILIKE supported.",
            "oracle": "FETCH FIRST n ROWS ONLY. SYSDATE. NVL. CONNECT BY. MINUS not EXCEPT.",
        }
        dialect_note = dialect_notes.get(args.dialect.lower(), "Use ANSI SQL.")

        schema_dsl = graph.to_dsl()
        prompt = f"""You are a precise SQL generation engine. Generate correct {args.dialect.upper()} SQL.

## DATABASE SCHEMA
{schema_dsl}

## DIALECT RULES
{dialect_note}

## USER REQUEST
{args.query}

## GENERATION PROTOCOL

Reason in EXECUTION ORDER: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
Write in standard SQL order. Check every column name against the schema above.

Anti-hallucination rules:
- Never use an aggregate function in WHERE (use HAVING)
- Never reference a SELECT alias in WHERE or GROUP BY
- Every non-aggregated SELECT column must appear in GROUP BY
- Every JOIN must have an ON clause

Output the final SQL in a ```sql block.
"""
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Generating SQL for: {args.query!r} (dialect={args.dialect})", file=sys.stderr)
        message = client.messages.create(
            model=args.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        response = message.content[0].text
        import re as _re
        sql_match = _re.search(r'```sql\s*(.*?)\s*```', response, _re.DOTALL | _re.IGNORECASE)
        if sql_match:
            print(sql_match.group(1).strip())
        else:
            print(response)
