"""
salting_1d.py
=============
1D-salting baseline for 2-hop motif queries (benchmarks only, not part of the published package).

Uses the same hub definition W(h) = d_in * d_out > work_threshold and the same
partition budget S = p_cells salt buckets as HyperGraphFrames:
  - Heavy hubs: hash e1 rows into S salt buckets on e1.src and replicate e2 rows across all S buckets;
    join on (dst == src, salt == salt).
  - Non-hub vertices: standard equi-join on dst == src.
  - Union both parts and rebuild the a, e1, b, e2, c structs with the exact same schema as native find().
"""

from typing import Optional, Dict, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, expr, explode, sequence, broadcast, struct

from hypergraph.planner import parse_2hop_motif


def profile_topology_1d(
    edges: DataFrame,
    work_threshold: int = 1_000_000,
) -> Tuple[DataFrame, DataFrame]:
    """
    Profiles graph edge topology to identify heavy hubs where W(v) = d_in(v) * d_out(v) > work_threshold.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        - hub_meta: Cached DataFrame of schema (id, in_deg, out_deg, work)
        - hub_ids: Cached DataFrame containing only 'id' of identified hubs
    """
    out_deg = (
        edges.groupBy("src")
        .count()
        .withColumnRenamed("src", "id")
        .withColumnRenamed("count", "out_deg")
    )
    in_deg = (
        edges.groupBy("dst")
        .count()
        .withColumnRenamed("dst", "id")
        .withColumnRenamed("count", "in_deg")
    )

    degrees = in_deg.join(out_deg, on="id", how="full").fillna(0)
    degrees = degrees.withColumn("work", col("in_deg") * col("out_deg"))

    hub_meta = (
        degrees.filter(col("work") > work_threshold)
        .select("id", "in_deg", "out_deg", "work")
        .cache()
    )
    hub_ids = hub_meta.select("id").cache()
    return hub_meta, hub_ids


def shape_2hop_output(
    matched_edges: DataFrame,
    edges: DataFrame,
    vertices: Optional[DataFrame],
    motif_spec: Dict[str, Any],
    skip_id_only_vertex_join: bool = False,
) -> DataFrame:
    """
    Rebuilds the a, e1, b, e2, c structs identically to HyperGraphFrame.run_hypercube_2hop.
    """
    e_cols = edges.columns
    v_cols = vertices.columns if vertices is not None else []

    if vertices is not None:
        if skip_id_only_vertex_join and set(v_cols) == {"id"}:
            struct_map: Dict[str, Any] = {
                motif_spec["a"]: struct(col("__e1_src").alias("id")).alias(motif_spec["a"]),
                motif_spec["b"]: struct(col("__e1_dst").alias("id")).alias(motif_spec["b"]),
            }
            if motif_spec["a"] != motif_spec["c"]:
                struct_map[motif_spec["c"]] = struct(col("__e2_dst").alias("id")).alias(motif_spec["c"])

            if motif_spec["e1"]:
                struct_map[motif_spec["e1"]] = struct(*[col(f"__e1_{c}").alias(c) for c in e_cols]).alias(motif_spec["e1"])
            if motif_spec["e2"]:
                struct_map[motif_spec["e2"]] = struct(*[col(f"__e2_{c}").alias(c) for c in e_cols]).alias(motif_spec["e2"])

            final_cols = [struct_map[tok] for tok in motif_spec["ordered_tokens"]]
            return matched_edges.select(*final_cols)
        else:
            va = vertices.select(*[col(c).alias(f"__va_{c}") for c in v_cols])
            vb = vertices.select(*[col(c).alias(f"__vb_{c}") for c in v_cols])
            vc = vertices.select(*[col(c).alias(f"__vc_{c}") for c in v_cols])

            joined = (
                matched_edges
                .join(va, col("__va_id") == col("__e1_src"))
                .join(vb, col("__vb_id") == col("__e1_dst"))
            )
            if motif_spec["a"] != motif_spec["c"]:
                joined = joined.join(vc, col("__vc_id") == col("__e2_dst"))

            struct_map = {
                motif_spec["a"]: struct(*[col(f"__va_{c}").alias(c) for c in v_cols]).alias(motif_spec["a"]),
                motif_spec["b"]: struct(*[col(f"__vb_{c}").alias(c) for c in v_cols]).alias(motif_spec["b"]),
            }
            if motif_spec["a"] != motif_spec["c"]:
                struct_map[motif_spec["c"]] = struct(*[col(f"__vc_{c}").alias(c) for c in v_cols]).alias(motif_spec["c"])

            if motif_spec["e1"]:
                struct_map[motif_spec["e1"]] = struct(*[col(f"__e1_{c}").alias(c) for c in e_cols]).alias(motif_spec["e1"])
            if motif_spec["e2"]:
                struct_map[motif_spec["e2"]] = struct(*[col(f"__e2_{c}").alias(c) for c in e_cols]).alias(motif_spec["e2"])

            final_cols = [struct_map[tok] for tok in motif_spec["ordered_tokens"]]
            return joined.select(*final_cols)
    else:
        return matched_edges


def run_edgejoin_2hop(
    edges: DataFrame,
    vertices: Optional[DataFrame],
    motif_spec: Dict[str, Any],
    skip_id_only_vertex_join: bool = False,
) -> DataFrame:
    """
    Executes a plain unsalted e1 ⋈ e2 2-hop motif join with the same output struct
    construction as HyperGraphFrame and Salted1DGraphFrame.
    """
    e_cols = edges.columns
    pair_cols = [f"__e1_{c}" for c in e_cols] + [f"__e2_{c}" for c in e_cols]

    e1_proj = [col(c).alias(f"__e1_{c}") for c in e_cols]
    e2_proj = [col(c).alias(f"__e2_{c}") for c in e_cols]
    e1 = edges.select(*e1_proj).alias("e1")
    e2 = edges.select(*e2_proj).alias("e2")

    cond = col("e1.__e1_dst") == col("e2.__e2_src")
    if motif_spec["a"] == motif_spec["c"]:
        cond = cond & (col("e1.__e1_src") == col("e2.__e2_dst"))

    matched_edges = e1.join(e2, cond).select(
        *[col(f"e1.{c}").alias(c) if c.startswith("__e1_") else col(f"e2.{c}").alias(c) for c in pair_cols]
    )
    return shape_2hop_output(
        matched_edges=matched_edges,
        edges=edges,
        vertices=vertices,
        motif_spec=motif_spec,
        skip_id_only_vertex_join=skip_id_only_vertex_join,
    )


def run_salting_1d_2hop(
    edges: DataFrame,
    vertices: Optional[DataFrame],
    hub_ids: DataFrame,
    motif_spec: Dict[str, Any],
    hub_count: int,
    salt_buckets: int = 100,
    skip_id_only_vertex_join: bool = False,
) -> DataFrame:
    """
    Executes a 2-hop motif query using 1D Salting with budget S = salt_buckets.
    """
    s_buckets = max(1, int(salt_buckets))
    e_cols = edges.columns
    pair_cols = [f"__e1_{c}" for c in e_cols] + [f"__e2_{c}" for c in e_cols]

    if hub_count == 0:
        return run_edgejoin_2hop(
            edges=edges,
            vertices=vertices,
            motif_spec=motif_spec,
            skip_id_only_vertex_join=skip_id_only_vertex_join,
        )
    else:
        # ------------------ BRANCH 1: HUB 1D SALTED JOIN ------------------
        e1_hub = (
            edges.join(broadcast(hub_ids), edges["dst"] == hub_ids["id"])
            .select(*[edges[c].alias(f"__e1_{c}") for c in e_cols])
            .withColumn("__salt", expr(f"pmod(hash(__e1_src), {s_buckets})"))
            .select(*[f"__e1_{c}" for c in e_cols], "__salt")
            .alias("e1h")
        )

        e2_hub = (
            edges.join(broadcast(hub_ids), edges["src"] == hub_ids["id"])
            .select(*[edges[c].alias(f"__e2_{c}") for c in e_cols])
            .withColumn("__salt", explode(sequence(lit(0), lit(s_buckets - 1))))
            .select(*[f"__e2_{c}" for c in e_cols], "__salt")
            .alias("e2h")
        )

        hub_cond = (
            (col("e1h.__e1_dst") == col("e2h.__e2_src"))
            & (col("e1h.__salt") == col("e2h.__salt"))
        )
        if motif_spec["a"] == motif_spec["c"]:
            hub_cond = hub_cond & (col("e1h.__e1_src") == col("e2h.__e2_dst"))

        hub_join = e1_hub.join(e2_hub, hub_cond).select(
            *[col(f"e1h.{c}").alias(c) if c.startswith("__e1_") else col(f"e2h.{c}").alias(c) for c in pair_cols]
        )

        # ------------------ BRANCH 2: NON-HUB HASH JOIN -------------------
        e1_proj = [col(c).alias(f"__e1_{c}") for c in e_cols]
        e2_proj = [col(c).alias(f"__e2_{c}") for c in e_cols]

        e1_nonhub = (
            edges.join(broadcast(hub_ids), edges["dst"] == hub_ids["id"], "left_anti")
            .select(*e1_proj)
            .alias("e1n")
        )
        e2_nonhub = (
            edges.join(broadcast(hub_ids), edges["src"] == hub_ids["id"], "left_anti")
            .select(*e2_proj)
            .alias("e2n")
        )

        nonhub_cond = col("e1n.__e1_dst") == col("e2n.__e2_src")
        if motif_spec["a"] == motif_spec["c"]:
            nonhub_cond = nonhub_cond & (col("e1n.__e1_src") == col("e2n.__e2_dst"))

        nonhub_join = e1_nonhub.join(e2_nonhub, nonhub_cond).select(
            *[col(f"e1n.{c}").alias(c) if c.startswith("__e1_") else col(f"e2n.{c}").alias(c) for c in pair_cols]
        )

        # ------------------ BRANCH 3: UNION RESULTS -----------------------
        matched_edges = hub_join.unionByName(nonhub_join)

    return shape_2hop_output(
        matched_edges=matched_edges,
        edges=edges,
        vertices=vertices,
        motif_spec=motif_spec,
        skip_id_only_vertex_join=skip_id_only_vertex_join,
    )


class EdgeJoinGraphFrame:
    """
    Plain unsalted e1 ⋈ e2 wrapper around GraphFrame with identical output construction
    to HyperGraphFrame and Salted1DGraphFrame (for controlled baseline comparison).
    """

    def __init__(
        self,
        graph_frame: Any,
        skip_id_only_vertex_join: bool = False,
    ):
        self._gf = graph_frame
        self.skip_id_only_vertex_join = skip_id_only_vertex_join

    @property
    def vertices(self) -> DataFrame:
        return self._gf.vertices

    @property
    def edges(self) -> DataFrame:
        return self._gf.edges

    def find(self, pattern: str) -> DataFrame:
        motif_spec = parse_2hop_motif(pattern)
        if motif_spec is not None:
            return run_edgejoin_2hop(
                edges=self._gf.edges,
                vertices=self._gf.vertices,
                motif_spec=motif_spec,
                skip_id_only_vertex_join=self.skip_id_only_vertex_join,
            )
        return self._gf.find(pattern)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gf, name)


class Salted1DGraphFrame:
    """
    1D-salting wrapper around GraphFrame with the same interface as HyperGraphFrame.
    """

    def __init__(
        self,
        graph_frame: Any,
        work_threshold: int = 1_000_000,
        p_cells: int = 100,
        skip_id_only_vertex_join: bool = False,
    ):
        self._gf = graph_frame
        self.work_threshold = work_threshold
        self.p_cells = p_cells
        self.skip_id_only_vertex_join = skip_id_only_vertex_join

        self._hub_meta: Optional[DataFrame] = None
        self._hub_ids: Optional[DataFrame] = None
        self._hub_count: Optional[int] = None

    @property
    def vertices(self) -> DataFrame:
        return self._gf.vertices

    @property
    def edges(self) -> DataFrame:
        return self._gf.edges

    @property
    def hub_meta(self) -> Optional[DataFrame]:
        return self._hub_meta

    @property
    def hub_ids(self) -> Optional[DataFrame]:
        return self._hub_ids

    @property
    def hub_count(self) -> Optional[int]:
        return self._hub_count

    def profile(self) -> "Salted1DGraphFrame":
        if self._hub_meta is None or self._hub_ids is None or self._hub_count is None:
            self._gf.edges.cache()
            hub_meta, hub_ids = profile_topology_1d(
                edges=self._gf.edges,
                work_threshold=self.work_threshold,
            )
            self._hub_meta = hub_meta.cache()
            self._hub_ids = hub_ids.cache()
            self._hub_count = self._hub_meta.count()
        return self

    def uncache_topology(self) -> "Salted1DGraphFrame":
        if self._hub_meta is not None:
            try:
                self._hub_meta.unpersist()
            except Exception:
                pass
            self._hub_meta = None
        if self._hub_ids is not None:
            try:
                self._hub_ids.unpersist()
            except Exception:
                pass
            self._hub_ids = None
        self._hub_count = None
        return self

    def find(self, pattern: str) -> DataFrame:
        motif_spec = parse_2hop_motif(pattern)
        if motif_spec is not None:
            if self._hub_meta is None or self._hub_ids is None or self._hub_count is None:
                self.profile()
            return run_salting_1d_2hop(
                edges=self._gf.edges,
                vertices=self._gf.vertices,
                hub_ids=self._hub_ids,
                motif_spec=motif_spec,
                hub_count=self._hub_count,
                salt_buckets=self.p_cells,
                skip_id_only_vertex_join=self.skip_id_only_vertex_join,
            )
        return self._gf.find(pattern)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gf, name)
