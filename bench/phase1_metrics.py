"""
phase1_metrics.py
=================
Extra per-run metrics on top of bench.collect_window_metrics:

  - dominant-stage task duration mean / std / coefficient of variation (from the task list;
    bench.py only reports quantiles),
  - dominant-stage per-task shuffle-read bytes and records (the actual reduce-partition
    sizes the join saw, after any AQE coalescing or skew splitting),
  - AQE decisions from the SQL REST API: for every AQEShuffleRead node in the run's SQL
    executions, the number of partitions, coalesced partitions, skewed partitions and
    skew splits Spark reports.

Everything degrades to None when the UI endpoint does not expose a value; nothing is guessed.
"""

import re
import statistics
from typing import Any, Dict, List, Optional

import requests


def _quantiles(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"min": None, "median": None, "p95": None, "max": None, "mean": None}
    s = sorted(vals)
    n = len(s)
    return {
        "min": s[0],
        "median": s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2,
        "p95": s[min(n - 1, int(round(0.95 * (n - 1))))],
        "max": s[-1],
        "mean": round(statistics.mean(s), 3),
    }


def stage_task_profile(ui_url: str, app_id: str, stage_id: int, attempt_id: int) -> Dict[str, Any]:
    """Task-level durations and shuffle-read sizes for one stage (all tasks, not just quantiles)."""
    url = (f"{ui_url}/api/v1/applications/{app_id}/stages/{stage_id}/{attempt_id}"
           f"/taskList?length=1000000")
    try:
        tasks = requests.get(url, timeout=30).json()
    except Exception as e:
        return {"error": f"taskList fetch failed: {e}"}
    if not isinstance(tasks, list):
        return {"error": "taskList not a list"}
    dur, rbytes, rrecs = [], [], []
    for t in tasks:
        if t.get("status") not in (None, "SUCCESS"):
            continue
        dur.append(float(t.get("duration", 0) or 0))
        m = t.get("taskMetrics") or {}
        sr = m.get("shuffleReadMetrics") or {}
        rbytes.append(float((sr.get("localBytesRead", 0) or 0) + (sr.get("remoteBytesRead", 0) or 0)))
        rrecs.append(float(sr.get("recordsRead", 0) or 0))
    mean = statistics.mean(dur) if dur else 0.0
    std = statistics.pstdev(dur) if len(dur) > 1 else 0.0
    return {
        "num_tasks": len(dur),
        "duration_ms": _quantiles(dur),
        "duration_std_ms": round(std, 3),
        "duration_cv": round(std / mean, 4) if mean > 0 else None,
        "shuffle_read_bytes": _quantiles(rbytes),
        "shuffle_read_records": _quantiles(rrecs),
    }


def max_sql_execution_id(ui_url: str, app_id: str) -> int:
    try:
        rows = requests.get(f"{ui_url}/api/v1/applications/{app_id}/sql?details=false&length=100000",
                            timeout=15).json()
        return max((int(r["id"]) for r in rows), default=-1)
    except Exception:
        return -1


_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _metric_number(value: str) -> Optional[float]:
    """Parse '12' or '1,234' or 'total (min, med, max ...)\\n12.3 MiB (...)' -> first number (unit-less)."""
    if value is None:
        return None
    m = _NUM.search(str(value).replace("\n", " "))
    return float(m.group(0).replace(",", "")) if m else None


def aqe_sql_metrics(ui_url: str, app_id: str, after_sql_id: int) -> Dict[str, Any]:
    """Aggregate AQEShuffleRead node metrics over SQL executions with id > after_sql_id."""
    try:
        rows = requests.get(f"{ui_url}/api/v1/applications/{app_id}/sql?details=true"
                            f"&planDescription=false&length=100000", timeout=30).json()
    except Exception as e:
        return {"error": f"sql endpoint failed: {e}"}
    nodes_out = []
    totals = {"aqe_shuffle_read_nodes": 0, "aqe_partitions": 0, "aqe_coalesced_partitions": 0,
              "aqe_skewed_partitions": 0, "aqe_skew_splits": 0}
    seen_metric = {k: False for k in totals}
    for ex in rows:
        if int(ex.get("id", -1)) <= after_sql_id:
            continue
        for node in ex.get("nodes", []):
            if "AQEShuffleRead" not in node.get("nodeName", ""):
                continue
            mets = {m["name"]: m["value"] for m in node.get("metrics", [])}
            totals["aqe_shuffle_read_nodes"] += 1
            seen_metric["aqe_shuffle_read_nodes"] = True
            for key, name in [("aqe_partitions", "number of partitions"),
                              ("aqe_coalesced_partitions", "number of coalesced partitions"),
                              ("aqe_skewed_partitions", "number of skewed partitions"),
                              ("aqe_skew_splits", "number of skewed partition splits")]:
                if name in mets:
                    v = _metric_number(mets[name])
                    if v is not None:
                        totals[key] += int(v)
                        seen_metric[key] = True
            nodes_out.append({"execution_id": ex.get("id"), "node": node.get("nodeName"), "metrics": mets})
    # A metric Spark never reported stays None rather than a misleading 0.
    for k, seen in seen_metric.items():
        if not seen and k != "aqe_shuffle_read_nodes":
            totals[k] = None
    totals["aqe_skew_split_applied"] = bool(totals["aqe_skewed_partitions"]) if totals["aqe_skewed_partitions"] is not None else None
    return {"totals": totals, "nodes": nodes_out}


def summarize_plan_analysis(pa: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of run_single_ablation.analyze_plan output for the flat record."""
    if not pa or "error" in pa:
        return {"plan_is_final": None, "plan_aqe_shuffle_reads": None, "plan_skew_split": None,
                "plan_partial_reducer_specs": None, "plan_coalesced_specs": None}
    partial = coalesced = 0
    for r in pa.get("aqe_shuffle_reads", []):
        for stype, cnt in r.get("spec_types", {}).items():
            if "Partial" in stype:
                partial += cnt
            if "Coalesced" in stype:
                coalesced += cnt
    return {
        "plan_is_final": pa.get("is_final_plan"),
        "plan_aqe_shuffle_reads": pa.get("num_aqe_shuffle_reads"),
        "plan_skew_split": pa.get("any_show_skew_splits"),
        "plan_partial_reducer_specs": partial,
        "plan_coalesced_specs": coalesced,
    }


def aqe_plan_metrics(df) -> Dict[str, Any]:
    """
    AQE decisions read from the executed (final) plan of `df` after it has run.

    The timed action (queryExecution().toRdd().count()) does not register a SQL execution, so the
    SQL REST endpoint never sees it; the AQEShuffleRead nodes' own SQLMetrics are read instead.
    Spark 3.5 metric keys: numPartitions, numCoalescedPartitions, numSkewedPartitions,
    numSkewedSplits, numEmptyPartitions, partitionDataSize.
    """
    from run_single_ablation import find_aqe_reads

    totals = {"aqe_shuffle_read_nodes": 0, "aqe_partitions": None, "aqe_coalesced_partitions": None,
              "aqe_skewed_partitions": None, "aqe_skew_splits": None, "aqe_empty_partitions": None,
              "aqe_partition_data_size_bytes": None}
    key_map = {"numPartitions": "aqe_partitions", "numCoalescedPartitions": "aqe_coalesced_partitions",
               "numSkewedPartitions": "aqe_skewed_partitions", "numSkewedSplits": "aqe_skew_splits",
               "numEmptyPartitions": "aqe_empty_partitions", "partitionDataSize": "aqe_partition_data_size_bytes"}
    nodes = []
    try:
        reads = find_aqe_reads(df._jdf.queryExecution().executedPlan())
    except Exception as e:
        return {"totals": totals, "nodes": [], "error": str(e)}
    for r in reads:
        totals["aqe_shuffle_read_nodes"] += 1
        node = {}
        try:
            m = r.metrics()
            it = m.keysIterator()
            while it.hasNext():
                k = it.next()
                v = int(m.apply(k).value())
                node[k] = v
                if k in key_map:
                    tk = key_map[k]
                    totals[tk] = (totals[tk] or 0) + v
        except Exception as e:
            node["error"] = str(e)
        nodes.append(node)
    if totals["aqe_skewed_partitions"] is not None:
        totals["aqe_skew_split_applied"] = totals["aqe_skewed_partitions"] > 0
    else:
        totals["aqe_skew_split_applied"] = None if totals["aqe_shuffle_read_nodes"] else False
    return {"totals": totals, "nodes": nodes}
