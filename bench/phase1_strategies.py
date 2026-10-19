"""
phase1_strategies.py
====================
Candidate execution strategies for Phase 1 = (existing bench variant) x (AQE on/off).

Variant names are the ones bench/bench.py already uses, plus hypercube_unified:
  native            GraphFrames g.find()                          (reference semantics)
  edgejoin          plain e1 JOIN e2, same output shaping          (salting_1d.EdgeJoinGraphFrame)
  salt1d            1D salting of hub keys, S = p_cells buckets    (salting_1d.Salted1DGraphFrame)
  hypercube         split 2D hypercube (hub branch + non-hub branch, Algorithm 1)  (hypergraph.HyperGraphFrame)
  hypercube_vjoin   hypercube with vertex joins kept (drop-in default)             (hypergraph.HyperGraphFrame)
  hypercube_unified single-pass map-literal 2D hypercube, same planner             (hypercube_unified.py)

A strategy id is "<variant>__aqe_<on|off>", e.g. "hypercube__aqe_off".
"""

from typing import Any, Dict, List

# Spark 3.5 defaults for the AQE skew/coalesce knobs, pinned explicitly so an
# environment-level override cannot silently change the "AQE on" condition.
AQE_ON_CONF = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": str(256 * 1024 * 1024),
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "5.0",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": str(64 * 1024 * 1024),
}
AQE_OFF_CONF = {"spark.sql.adaptive.enabled": "false"}

VARIANTS = ["native", "edgejoin", "salt1d", "hypercube", "hypercube_vjoin", "hypercube_unified"]
PROFILED_VARIANTS = {"salt1d", "hypercube", "hypercube_vjoin", "hypercube_unified"}

# Pilot candidate set. AQE on/off for the two unpartitioned baselines (native, edgejoin);
# the application-level partitioners run with AQE off, as in the existing Pokec harness.
PILOT_STRATEGIES = [
    "native__aqe_on",
    "native__aqe_off",
    "edgejoin__aqe_on",
    "edgejoin__aqe_off",
    "salt1d__aqe_off",
    "hypercube__aqe_off",
    "hypercube_unified__aqe_off",
]


def parse_strategy(sid: str) -> Dict[str, str]:
    variant, aqe = sid.split("__aqe_")
    if variant not in VARIANTS or aqe not in ("on", "off"):
        raise ValueError(f"Unknown strategy id {sid!r}")
    return {"strategy": sid, "variant": variant, "aqe": aqe}


def all_strategy_ids() -> List[str]:
    return [f"{v}__aqe_{a}" for v in VARIANTS for a in ("off", "on")]


def apply_aqe(spark, aqe: str) -> Dict[str, str]:
    """Set the AQE condition for the next query and return the effective values."""
    conf = AQE_ON_CONF if aqe == "on" else AQE_OFF_CONF
    for k, v in conf.items():
        spark.conf.set(k, v)
    keys = list(AQE_ON_CONF)
    return {k: spark.conf.get(k) for k in keys}


def build_runner(variant: str, g: Any, work_threshold: int, p_cells: int, regime: str = "id_only"):
    """
    Return an object with .find(motif) and, for partitioned variants, .profile()/.uncache_topology().
    Uses the repository's existing wrappers; skip_id_only_vertex_join follows bench.py's id_only regime.
    """
    from hypergraph import HyperGraphFrame
    from salting_1d import EdgeJoinGraphFrame, Salted1DGraphFrame
    from hypercube_unified import UnifiedHypercubeGraphFrame

    skip = regime == "id_only"
    if variant == "native":
        return g
    if variant == "edgejoin":
        return EdgeJoinGraphFrame(g, skip_id_only_vertex_join=skip)
    if variant == "salt1d":
        return Salted1DGraphFrame(g, work_threshold=work_threshold, p_cells=p_cells, skip_id_only_vertex_join=skip)
    if variant == "hypercube":
        return HyperGraphFrame(g, work_threshold=work_threshold, p_cells=p_cells, skip_id_only_vertex_join=skip)
    if variant == "hypercube_vjoin":
        return HyperGraphFrame(g, work_threshold=work_threshold, p_cells=p_cells, skip_id_only_vertex_join=False)
    if variant == "hypercube_unified":
        return UnifiedHypercubeGraphFrame(g, work_threshold=work_threshold, p_cells=p_cells,
                                          skip_id_only_vertex_join=skip)
    raise ValueError(variant)
