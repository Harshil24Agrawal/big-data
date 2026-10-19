"""
hypercube_unified.py
====================
Unified single-pass 2D hypercube variant (benchmarks only, not part of the package).

This is the map-literal approach from pokec_benchmark.run_hypercube_unified,
repackaged so it is comparable with the other variants:
  - hubs and (r, s) grids come from the SAME planner as the split hypercube
    (hypergraph.topology.profile_topology), so both variants route identical
    hubs to identical grids;
  - output rows are shaped by salting_1d.shape_2hop_output, so the schema and
    struct layout match native GraphFrames find() exactly.

Every edge goes through one join on (dst == src, r_coord, s_coord). Non-hub
vertices get r = s = 1 (no replication), so there are no anti-joins or unions.
Limitation: the hub table is inlined as a SQL map literal, so it is only used
when the hub count is at most MAX_HUBS_FOR_MAP_LITERAL.
"""

from typing import Any, Dict, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr

from hypergraph.planner import parse_2hop_motif
from hypergraph.topology import profile_topology
from salting_1d import shape_2hop_output

MAX_HUBS_FOR_MAP_LITERAL = 2000


def _hub_map_literal(hub_rows, key_type: str) -> str:
    if not hub_rows:
        return f"map(cast(-1 as {key_type}), named_struct('r', 1, 's', 1))"
    entries = [
        f"cast({h['id']} as {key_type}), named_struct('r', {int(h['r_final'])}, 's', {int(h['s_final'])})"
        for h in hub_rows
    ]
    return f"map({', '.join(entries)})"


def run_hypercube_unified_2hop(
    edges: DataFrame,
    vertices: Optional[DataFrame],
    hub_rows,
    motif_spec: Dict[str, Any],
    skip_id_only_vertex_join: bool = False,
) -> DataFrame:
    e_cols = edges.columns
    pair_cols = [f"__e1_{c}" for c in e_cols] + [f"__e2_{c}" for c in e_cols]
    key_type = edges.schema["dst"].dataType.simpleString()
    hub_map = _hub_map_literal(hub_rows, key_type)

    e1 = (
        edges.select(*[col(c).alias(f"__e1_{c}") for c in e_cols])
        .withColumn("__hub", expr(f"element_at({hub_map}, __e1_dst)"))
        .withColumn("__r", expr("coalesce(__hub.r, 1)"))
        .withColumn("__s", expr("coalesce(__hub.s, 1)"))
        .withColumn("__r_coord", expr("pmod(hash(__e1_src), __r)"))
        .withColumn("__s_coord", expr("explode(sequence(0, __s - 1))"))
        .select(*[f"__e1_{c}" for c in e_cols], "__r_coord", "__s_coord")
        .alias("e1u")
    )
    e2 = (
        edges.select(*[col(c).alias(f"__e2_{c}") for c in e_cols])
        .withColumn("__hub", expr(f"element_at({hub_map}, __e2_src)"))
        .withColumn("__r", expr("coalesce(__hub.r, 1)"))
        .withColumn("__s", expr("coalesce(__hub.s, 1)"))
        .withColumn("__s_coord", expr("pmod(hash(__e2_dst), __s)"))
        .withColumn("__r_coord", expr("explode(sequence(0, __r - 1))"))
        .select(*[f"__e2_{c}" for c in e_cols], "__r_coord", "__s_coord")
        .alias("e2u")
    )
    cond = (
        (col("e1u.__e1_dst") == col("e2u.__e2_src"))
        & (col("e1u.__r_coord") == col("e2u.__r_coord"))
        & (col("e1u.__s_coord") == col("e2u.__s_coord"))
    )
    if motif_spec["a"] == motif_spec["c"]:
        cond = cond & (col("e1u.__e1_src") == col("e2u.__e2_dst"))

    matched = e1.join(e2, cond).select(
        *[col(f"e1u.{c}").alias(c) if c.startswith("__e1_") else col(f"e2u.{c}").alias(c) for c in pair_cols]
    )
    return shape_2hop_output(matched, edges, vertices, motif_spec, skip_id_only_vertex_join)


class UnifiedHypercubeGraphFrame:
    """GraphFrame wrapper with the same interface as HyperGraphFrame (profile / find / uncache_topology)."""

    def __init__(self, graph_frame: Any, work_threshold: int = 1_000_000, p_cells: int = 100,
                 skip_id_only_vertex_join: bool = False):
        self._gf = graph_frame
        self.work_threshold = work_threshold
        self.p_cells = p_cells
        self.skip_id_only_vertex_join = skip_id_only_vertex_join
        self._hub_meta: Optional[DataFrame] = None
        self._hub_ids: Optional[DataFrame] = None
        self._hub_rows = None

    @property
    def hub_count(self) -> Optional[int]:
        return None if self._hub_rows is None else len(self._hub_rows)

    def profile(self) -> "UnifiedHypercubeGraphFrame":
        if self._hub_rows is None:
            self._gf.edges.cache()
            self._hub_meta, self._hub_ids = profile_topology(self._gf.edges, self.work_threshold, self.p_cells)
            # The map literal needs the hub table on the driver.
            self._hub_rows = [r.asDict() for r in self._hub_meta.collect()]
        return self

    def uncache_topology(self) -> "UnifiedHypercubeGraphFrame":
        for df in (self._hub_meta, self._hub_ids):
            if df is not None:
                try:
                    df.unpersist()
                except Exception:
                    pass
        self._hub_meta = self._hub_ids = self._hub_rows = None
        return self

    def find(self, pattern: str) -> DataFrame:
        spec = parse_2hop_motif(pattern)
        if spec is None:
            return self._gf.find(pattern)
        self.profile()
        if len(self._hub_rows) > MAX_HUBS_FOR_MAP_LITERAL:
            raise ValueError(
                f"hypercube_unified inlines hubs as a SQL map literal; {len(self._hub_rows)} hubs exceeds "
                f"MAX_HUBS_FOR_MAP_LITERAL={MAX_HUBS_FOR_MAP_LITERAL}"
            )
        return run_hypercube_unified_2hop(self._gf.edges, self._gf.vertices, self._hub_rows, spec,
                                          self.skip_id_only_vertex_join)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gf, name)
