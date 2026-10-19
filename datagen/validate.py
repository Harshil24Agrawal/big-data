"""
datagen/validate.py
===================
Measured topology statistics and pre-declared family checks.

`degree_stats` works on numpy arrays (used right after generation and on Pokec).
`run_family_checks` compares those measurements with config.FAMILY_CHECKS.
`spark_load_check` re-reads a saved edge file through Spark and recomputes the
same invariants independently, so a graph only counts as valid if the file
Spark will benchmark matches what the generator measured.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from datagen import config


def _gini(x: np.ndarray) -> float:
    if x.sum() == 0:
        return 0.0
    s = np.sort(x.astype(np.float64))
    n = len(s)
    cum = np.cumsum(s)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def _side(deg: np.ndarray, prefix: str, hub_k: int) -> Dict[str, float]:
    mean = float(deg.mean())
    top = np.sort(deg)[::-1]
    out = {
        f"{prefix}_mean": round(mean, 4),
        f"{prefix}_std": round(float(deg.std()), 4),
        f"{prefix}_cv": round(float(deg.std() / mean), 4) if mean else None,
        f"{prefix}_max": int(top[0]),
        f"{prefix}_p50": float(np.percentile(deg, 50)),
        f"{prefix}_p99": float(np.percentile(deg, 99)),
        f"{prefix}_p999": float(np.percentile(deg, 99.9)),
        f"{prefix}_max_over_mean": round(float(top[0] / mean), 4) if mean else None,
        f"{prefix}_gini": round(_gini(deg), 4),
        f"{prefix}_top1pct_edge_share": round(float(top[: max(1, len(deg) // 100)].sum() / deg.sum()), 4),
        f"{prefix}_top_hub_edge_share": round(float(top[:hub_k].sum() / deg.sum()), 4),
        f"{prefix}_zero_degree_vertices": int((deg == 0).sum()),
    }
    return out


def degree_stats(
    src: np.ndarray,
    dst: np.ndarray,
    num_vertices: int,
    hub_k: int = 10,
    work_threshold: int = config.WORK_THRESHOLD,
) -> Dict[str, float]:
    """All measurements used for validation and as downstream graph features."""
    n = int(num_vertices)
    din = np.bincount(dst, minlength=n).astype(np.int64)
    dout = np.bincount(src, minlength=n).astype(np.int64)
    work = din * dout
    work_sum = int(work.sum())
    top_work = np.sort(work)[::-1][:hub_k]

    both = (din > 0) & (dout > 0)
    if both.sum() > 2:
        corr = float(np.corrcoef(np.log(din[both]), np.log(dout[both]))[0, 1])
    else:
        corr = None

    stats = {
        "num_vertices": n,
        "num_edges": int(len(src)),
        "num_active_vertices": int(((din > 0) | (dout > 0)).sum()),
        "hub_k": hub_k,
        "work_threshold": work_threshold,
        # work_sum = sum_v d_in(v) * d_out(v): exact row count of e1.dst == e2.src
        "work_sum": work_sum,
        "work_max": int(work.max()),
        "work_max_vertex": int(work.argmax()),
        "work_max_share": round(float(work.max() / work_sum), 6) if work_sum else 0.0,
        "num_work_hubs": int((work > work_threshold).sum()),
        "work_in_hubs_share": round(float(work[work > work_threshold].sum() / work_sum), 6) if work_sum else 0.0,
        "top_hubs_work_share": round(float(top_work.sum() / work_sum), 6) if work_sum else 0.0,
        "top_hubs_min_over_max_work": round(float(top_work[-1] / top_work[0]), 6) if top_work[0] else 0.0,
        "log_degree_corr": round(corr, 4) if corr is not None else None,
        "max_in_over_max_out": round(float(din.max() / max(1, dout.max())), 4),
    }
    stats.update(_side(din, "in", hub_k))
    stats.update(_side(dout, "out", hub_k))
    return stats


def structural_checks(src: np.ndarray, dst: np.ndarray, num_vertices: int) -> Dict[str, object]:
    """ID validity, self-loops, duplicates (must all be zero for synthetic graphs)."""
    keys = (src.astype(np.int64) << 32) | dst.astype(np.int64)
    return {
        "min_id": int(min(src.min(), dst.min())),
        "max_id": int(max(src.max(), dst.max())),
        "ids_in_range": bool(src.min() >= 0 and dst.min() >= 0
                             and src.max() < num_vertices and dst.max() < num_vertices),
        "self_loops": int((src == dst).sum()),
        "duplicate_edges": int(len(keys) - len(np.unique(keys))),
    }


def _compare(value, op, target) -> bool:
    if value is None:
        return False
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    if op == "between":
        lo, hi = target
        return lo <= value <= hi
    raise ValueError(op)


def run_family_checks(family: str, stats: Dict, structure: Dict) -> Tuple[str, List[Dict]]:
    results = []
    for metric, op, target in config.FAMILY_CHECKS[family]:
        val = stats.get(metric)
        results.append({"metric": metric, "op": op, "target": target, "measured": val,
                        "passed": _compare(val, op, target)})
    for metric, ok in [("ids_in_range", structure["ids_in_range"]),
                       ("self_loops==0", structure["self_loops"] == 0),
                       ("duplicate_edges==0", structure["duplicate_edges"] == 0)]:
        results.append({"metric": metric, "op": "==", "target": True, "measured": ok, "passed": bool(ok)})
    status = "PASS" if all(r["passed"] for r in results) else "FAIL"
    return status, results


# ---------------------------------------------------------------------------
# Spark re-read check
# ---------------------------------------------------------------------------

def spark_load_check(spark, path: str, expected: Optional[Dict] = None) -> Dict[str, object]:
    """
    Read an edge file with Spark and recompute counts and work_sum independently.
    `expected` (from graph.json) is compared field by field when given.
    """
    from pyspark.sql import functions as F

    df = spark.read.parquet(path)
    schema = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    agg = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("src").isNull() | F.col("dst").isNull(), 1).otherwise(0)).alias("nulls"),
        F.min(F.least("src", "dst")).alias("min_id"),
        F.max(F.greatest("src", "dst")).alias("max_id"),
        F.sum(F.when(F.col("src") == F.col("dst"), 1).otherwise(0)).alias("self_loops"),
    ).collect()[0]
    distinct_pairs = df.select("src", "dst").distinct().count()
    din = df.groupBy(F.col("dst").alias("id")).agg(F.count(F.lit(1)).alias("din"))
    dout = df.groupBy(F.col("src").alias("id")).agg(F.count(F.lit(1)).alias("dout"))
    w = din.join(dout, "id").agg(F.sum(F.col("din") * F.col("dout")).alias("ws"),
                                 F.max(F.col("din") * F.col("dout")).alias("wm")).collect()[0]
    active = (df.select(F.col("src").alias("id")).union(df.select(F.col("dst").alias("id")))
              .distinct().count())
    res = {
        "schema": schema,
        "rows": int(agg["rows"]),
        "null_endpoints": int(agg["nulls"] or 0),
        "min_id": int(agg["min_id"]) if agg["min_id"] is not None else None,
        "max_id": int(agg["max_id"]) if agg["max_id"] is not None else None,
        "self_loops": int(agg["self_loops"] or 0),
        "duplicate_edges": int(agg["rows"]) - int(distinct_pairs),
        "active_vertices": int(active),
        "work_sum": int(w["ws"] or 0),
        "work_max": int(w["wm"] or 0),
    }
    if expected is not None:
        mism = []
        for k_res, k_exp in [("rows", "num_edges"), ("work_sum", "work_sum"),
                             ("work_max", "work_max"), ("active_vertices", "num_active_vertices"),
                             ("self_loops", "self_loops"), ("duplicate_edges", "duplicate_edges")]:
            if k_exp in expected and expected[k_exp] != res[k_res]:
                mism.append({"field": k_res, "spark": res[k_res], "expected": expected[k_exp]})
        if res["null_endpoints"]:
            mism.append({"field": "null_endpoints", "spark": res["null_endpoints"], "expected": 0})
        res["mismatches"] = mism
        res["status"] = "PASS" if not mism else "FAIL"
    return res
