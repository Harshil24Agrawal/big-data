#!/usr/bin/env python3
"""
phase1_aggregate.py
===================
Derived summaries from immutable Phase 1 raw run records. Raw files are only read.

  python bench/phase1_aggregate.py <campaign_id> [<campaign_id> ...]

Writes datasets/derived/<campaign_id>/:
  strategy_summary.csv   one row per (graph_id, strategy): counts by status, correctness, and
                         median / p95 / min / max / IQR of the measured runs
  graph_summary.csv      one row per graph: fastest strategy by median total_s and by median
                         query_s, plus the margin over the runner-up (a convenience for review;
                         labelling for the ML model is Member 3's call)
  SOURCE.txt             which raw files and record counts these were computed from

Only records with run_kind == "measured" and status == "ok" and correctness_status == "match"
enter the statistics; everything else is counted, never averaged in.
"""

import csv
import datetime
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RAW_DIR = os.path.join(REPO_ROOT, "datasets", "raw_runs")
DERIVED_DIR = os.path.join(REPO_ROOT, "datasets", "derived")

METRICS = ["query_s", "total_s", "profiling_s", "executor_run_time_s", "shuffle_read_bytes", "shuffle_write_bytes",
           "memory_spill_bytes", "disk_spill_bytes", "dom_task_median_ms", "dom_task_p95_ms", "dom_task_max_ms",
           "dom_task_max_median_ratio", "dom_task_cv", "dom_shuffle_read_bytes_max", "dom_shuffle_read_bytes_median",
           "total_tasks", "aqe_partitions", "aqe_skewed_partitions", "aqe_skew_splits"]


def q(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {}
    return {"n": len(vals), "median": statistics.median(vals), "p95": q(vals, 0.95), "min": min(vals),
            "max": max(vals), "iqr": (q(vals, 0.75) - q(vals, 0.25)), "mean": statistics.mean(vals)}


def main(campaign_ids):
    for cid in campaign_ids:
        path = os.path.join(RAW_DIR, cid, "runs.jsonl")
        with open(path) as f:
            recs = [json.loads(line) for line in f]
        # If a graph was re-run (e.g. after --resume), only its latest Spark application counts;
        # earlier partial records stay in the raw file and are reported as superseded.
        app_order = {}
        for i, r in enumerate(recs):
            if r.get("spark_app_id"):
                app_order[(r["graph_id"], r["spark_app_id"])] = i
        latest_app = {}
        for (gid, app), i in app_order.items():
            if gid not in latest_app or i > app_order[(gid, latest_app[gid])]:
                latest_app[gid] = app
        superseded = defaultdict(int)
        groups = defaultdict(list)
        for r in recs:
            app = r.get("spark_app_id")
            if app and app != latest_app.get(r["graph_id"]):
                superseded[(r["graph_id"], r["strategy"])] += 1
                continue
            groups[(r["graph_id"], r["strategy"])].append(r)

        rows = []
        for (gid, sid), rs in sorted(groups.items()):
            good = [r for r in rs if r["run_kind"] == "measured" and r["status"] == "ok"
                    and r.get("correctness_status") == "match"]
            gate = [r for r in rs if r["run_kind"] == "correctness"]
            any_r = rs[0]
            row = {
                "campaign_id": cid, "graph_id": gid, "graph_source": any_r.get("graph_source"),
                "family": any_r.get("family"), "size_tier": any_r.get("size_tier"), "seed": any_r.get("seed"),
                "strategy": sid, "variant": any_r.get("variant"), "aqe": any_r.get("aqe"),
                "config_id": any_r.get("config_id"), "machine_id": any_r.get("machine_id"),
                "expected_path_count": any_r.get("expected_path_count"),
                "gate_status": gate[0]["correctness_status"] if gate else "",
                "n_measured_ok": len(good),
                "n_measured_failed": sum(r["status"] == "failed" and r["run_kind"] == "measured" for r in rs),
                "n_measured_timeout": sum(r["status"] == "timeout" and r["run_kind"] == "measured" for r in rs),
                "n_skipped": sum(r["status"] == "skipped" for r in rs),
                "skip_reason": next((r.get("skip_reason") for r in rs if r["status"] == "skipped"), ""),
                "n_superseded_records": superseded[(gid, sid)],
                "n_warmup_ok": sum(r["run_kind"] == "warmup" and r["status"] == "ok" for r in rs),
                "hub_count": next((r.get("hub_count") for r in good if r.get("hub_count") is not None), ""),
                "aqe_skew_split_any_run": any(bool(r.get("aqe_skew_split_applied")) for r in good) if good else "",
            }
            for m in METRICS:
                st = stats([r.get(m) for r in good])
                for k in ("median", "p95", "min", "max", "iqr"):
                    row[f"{m}_{k}"] = st.get(k, "")
            rows.append(row)

        out = os.path.join(DERIVED_DIR, cid)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "strategy_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        by_graph = defaultdict(list)
        for r in rows:
            if r["n_measured_ok"]:
                by_graph[r["graph_id"]].append(r)
        grows = []
        for gid, rs in sorted(by_graph.items()):
            g = {"campaign_id": cid, "graph_id": gid, "family": rs[0]["family"], "size_tier": rs[0]["size_tier"],
                 "seed": rs[0]["seed"], "strategies_with_results": len(rs)}
            for metric in ("total_s", "query_s"):
                ranked = sorted(rs, key=lambda r: r[f"{metric}_median"])
                g[f"fastest_by_median_{metric}"] = ranked[0]["strategy"]
                g[f"fastest_median_{metric}"] = ranked[0][f"{metric}_median"]
                if len(ranked) > 1:
                    g[f"runner_up_by_median_{metric}"] = ranked[1]["strategy"]
                    g[f"margin_pct_{metric}"] = round(100 * (ranked[1][f"{metric}_median"] /
                                                             ranked[0][f"{metric}_median"] - 1), 2)
            grows.append(g)
        if grows:
            cols = sorted({k for g in grows for k in g}, key=lambda k: list(grows[0]).index(k) if k in grows[0] else 99)
            with open(os.path.join(out, "graph_summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(grows)

        with open(path, "rb") as f:
            raw_hash = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(out, "SOURCE.txt"), "w") as f:
            f.write(f"derived_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
                    f"raw_file=datasets/raw_runs/{cid}/runs.jsonl\nraw_sha256={raw_hash}\n"
                    f"raw_records={len(recs)}\nfilter=run_kind==measured & status==ok & correctness_status==match\n")
        print(f"{cid}: {len(rows)} (graph, strategy) rows, {len(grows)} graphs -> {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
