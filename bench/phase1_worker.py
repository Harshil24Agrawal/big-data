#!/usr/bin/env python3
"""
phase1_worker.py
================
Benchmarks every candidate strategy on ONE graph inside one fresh Spark application.
Launched by bench/run_phase1.py (one subprocess per graph); not usually run by hand.

Sequence inside the JVM:
  1. Load edges exactly as bench/bench.py does (read Parquet, cast int, repartition, cache,
     materialize) and build the id_only vertex table (distinct endpoints, cached, broadcast-hinted).
  2. Correctness gate (untimed): every strategy is executed once and its row count plus two
     order-independent row checksums are compared with native GraphFrames find() and with the
     expected path count (work_sum). A strategy that fails the gate gets no timed runs.
  3. Timed rounds: 1 warm-up round + N measured rounds. Each round runs every gated strategy
     once, in an order shuffled with a seed recorded in the run records.

Every run produces exactly one append-only line in runs.jsonl (flat, for aggregation) and one
line in run_details.jsonl (stage-level metrics and AQE plan analysis), keyed by record_id.
Failures, timeouts and skipped strategies are recorded as such; nothing is filled in.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import threading
import time
import traceback
import uuid

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [THIS_DIR, os.path.join(REPO_ROOT, "hypergraphframes"), REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import bench  # noqa: E402
from phase1_metrics import (aqe_plan_metrics, stage_task_profile,  # noqa: E402
                            summarize_plan_analysis)
from phase1_strategies import PROFILED_VARIANTS, apply_aqe, build_runner, parse_strategy  # noqa: E402

SCHEMA_VERSION = "phase1-run-v1"
MOTIF = bench.MOTIF_QUERY


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class JsonlSink:
    """Append-only writer; each record is flushed and fsynced so a crash loses at most one run."""

    def __init__(self, path: str):
        self.path = path

    def write(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


def create_spark(app_name: str, core: dict):
    from pyspark.sql import SparkSession

    bench.setup_env()
    b = (
        SparkSession.builder.appName(app_name)
        .master(core["spark_master"])
        .config("spark.driver.memory", core["driver_memory"])
        .config("spark.driver.extraJavaOptions", bench.JAVA_ADD_OPENS)
        .config("spark.executor.extraJavaOptions", bench.JAVA_ADD_OPENS)
        .config(*bench.graphframes_jar_conf())
        .config("spark.sql.shuffle.partitions", str(core["shuffle_partitions"]))
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.retainedTasks", "1000000")
        .config("spark.ui.retainedStages", "10000")
        .config("spark.sql.ui.retainedExecutions", "10000")
    )
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def checksum_df(df):
    """Order-independent content fingerprint of (a.id, b.id, c.id) rows."""
    from pyspark.sql import functions as F

    mask = F.lit(0xFFFFFFFF).cast("long")
    row = df.select(
        F.xxhash64(F.col("a.id"), F.col("b.id"), F.col("c.id")).alias("h1"),
        F.hash(F.col("c.id"), F.col("b.id"), F.col("a.id")).cast("long").alias("h2"),
    ).agg(
        F.count(F.lit(1)).alias("n"),
        F.sum(F.col("h1").bitwiseAND(mask)).alias("c1"),
        F.sum(F.col("h2").bitwiseAND(mask)).alias("c2"),
    ).collect()[0]
    return int(row["n"]), int(row["c1"] or 0), int(row["c2"] or 0)


class RunTimeout:
    """Cancel the run's job group after `seconds`; .fired tells the caller it happened."""

    def __init__(self, sc, group: str, seconds: float):
        self.fired = False
        self._t = None
        if seconds and seconds > 0:
            def _cancel():
                self.fired = True
                sc.cancelJobGroup(group)
            self._t = threading.Timer(seconds, _cancel)
            self._t.daemon = True

    def __enter__(self):
        if self._t:
            self._t.start()
        return self

    def __exit__(self, *exc):
        if self._t:
            self._t.cancel()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--graph-json", required=True, help="JSON file describing the graph (written by run_phase1.py)")
    args = ap.parse_args()

    with open(os.path.join(args.campaign_dir, "campaign.json")) as f:
        campaign = json.load(f)
    with open(args.graph_json) as f:
        graph = json.load(f)

    core = campaign["core_config"]
    strategies = [parse_strategy(s) for s in campaign["strategies"]]
    runs_sink = JsonlSink(os.path.join(args.campaign_dir, "runs.jsonl"))
    details_sink = JsonlSink(os.path.join(args.campaign_dir, "run_details.jsonl"))
    gid = graph["graph_id"]
    expected = graph.get("work_sum")

    spark = create_spark(f"Phase1_{gid}", core)
    sc = spark.sparkContext
    app_id = sc.applicationId
    ui_url = sc.uiWebUrl or "http://localhost:4040"

    import pyspark
    from graphframes import GraphFrame
    from pyspark.sql.functions import broadcast, col
    import run_single_ablation

    env = bench.get_system_and_version_info(spark)
    base = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "config_id": campaign["config_id"],
        "graph_id": gid,
        "graph_source": graph["source"],
        "family": graph.get("family"),
        "size_tier": graph.get("size_tier"),
        "seed": graph.get("seed"),
        "num_vertices": graph.get("num_vertices"),
        "num_edges": graph.get("num_edges"),
        "edges_path": graph["edges_path"],
        "expected_path_count": expected,
        "motif": MOTIF,
        "action": "rdd_count",
        "regime": core["regime"],
        "work_threshold": core["work_threshold"],
        "p_cells": core["p_cells"],
        "shuffle_partitions": core["shuffle_partitions"],
        "input_partitions": core["input_partitions"],
        "spark_master": core["spark_master"],
        "driver_memory": core["driver_memory"],
        "spark_version": spark.version,
        "pyspark_version": pyspark.__version__,
        "java_version": env.get("java_version"),
        "scala_version": env.get("scala_version"),
        "graphframes_jar": os.path.basename(str(env.get("graphframes_jar"))),
        "hypergraphframes_version": env.get("hypergraphframes_version"),
        "git_commit": env.get("git_commit"),
        "git_dirty_tree": env.get("git_dirty_tree"),
        "machine_id": campaign["machine"]["machine_id"],
        "spark_app_id": app_id,
    }

    def emit(rec: dict, details: dict = None):
        rec["record_id"] = rec.get("record_id") or uuid.uuid4().hex
        runs_sink.write(rec)
        if details is not None:
            details["record_id"] = rec["record_id"]
            details_sink.write(details)

    # ------------------------------------------------------------------ load
    t_load = time.perf_counter()
    raw = spark.read.parquet(graph["edges_path"]).select(col("src").cast("int"), col("dst").cast("int"))
    edges = raw.repartition(core["input_partitions"]).cache()
    edge_count = int(edges.count())
    vertices_cached = (edges.select(col("src").alias("id")).union(edges.select(col("dst").alias("id")))
                       .distinct().cache())
    vertex_count = int(vertices_cached.count())
    g = GraphFrame(broadcast(vertices_cached), edges)
    load_s = round(time.perf_counter() - t_load, 3)
    print(f"[{gid}] loaded {edge_count:,} edges / {vertex_count:,} active vertices in {load_s}s", flush=True)
    base["loaded_edge_count"] = edge_count
    base["loaded_vertex_count"] = vertex_count
    if graph.get("num_edges") is not None and edge_count != graph["num_edges"]:
        print(f"[{gid}] WARNING loaded edge count {edge_count} != manifest {graph['num_edges']}", flush=True)

    def runner_for(s):
        return build_runner(s["variant"], g, core["work_threshold"], core["p_cells"], core["regime"])

    # ------------------------------------------------------------------ correctness gate
    gate_ok = {}
    reference = None
    gate_order = sorted(strategies, key=lambda s: (s["variant"] != "native", s["aqe"] != "off"))
    for s in gate_order:
        rec = dict(base, **s, run_kind="correctness", round_index=None, measured_index=None,
                   order_in_round=None, start_utc=utcnow())
        group = f"gate-{gid}-{s['strategy']}"
        to = None
        try:
            rec["aqe_conf"] = apply_aqe(spark, s["aqe"])
            sc.setJobGroup(group, group, interruptOnCancel=True)
            t0 = time.perf_counter()
            with RunTimeout(sc, group, core["run_timeout_s"]) as to:
                r = runner_for(s)
                n, c1, c2 = checksum_df(r.find(MOTIF))
            rec["gate_seconds"] = round(time.perf_counter() - t0, 3)
            rec.update(path_count=n, checksum_xxhash64=c1, checksum_hash=c2,
                       hub_count=getattr(r, "hub_count", None))
            if reference is None:
                reference = {"strategy": s["strategy"], "n": n, "c1": c1, "c2": c2}
            checks = {
                "count_matches_expected": (expected is None) or n == expected,
                "matches_reference": (n, c1, c2) == (reference["n"], reference["c1"], reference["c2"]),
            }
            ok = all(checks.values())
            rec.update(status="ok", correctness_status="match" if ok else "mismatch",
                       correctness_reference=reference["strategy"], **checks)
            gate_ok[s["strategy"]] = ok
            if hasattr(r, "uncache_topology"):
                r.uncache_topology()
        except Exception as e:
            rec.update(status="timeout" if (to is not None and to.fired) else "failed",
                       error_type=type(e).__name__, error_message=str(e)[:2000],
                       error_trace=traceback.format_exc()[-4000:], correctness_status="unchecked")
            gate_ok[s["strategy"]] = False
        finally:
            sc.setLocalProperty("spark.jobGroup.id", None)
        rec["end_utc"] = utcnow()
        emit(rec)
        print(f"[{gid}] gate {s['strategy']:<28} {rec['status']:<8} {rec.get('correctness_status')} "
              f"n={rec.get('path_count')}", flush=True)

    # ------------------------------------------------------------------ timed rounds
    runnable = [s for s in strategies if gate_ok.get(s["strategy"])]
    for s in strategies:
        if not gate_ok.get(s["strategy"]):
            emit(dict(base, **s, run_kind="measured", round_index=None, measured_index=None, order_in_round=None,
                      status="skipped", skip_reason="failed or did not complete the correctness gate",
                      start_utc=utcnow(), end_utc=utcnow()))

    n_warm, n_meas = core["warmup_runs"], core["measured_runs"]
    for round_index in range(n_warm + n_meas):
        order_seed = int(hashlib.sha256(f"{campaign['order_seed']}|{gid}|{round_index}".encode()).hexdigest()[:8], 16)
        order = list(runnable)
        random.Random(order_seed).shuffle(order)
        kind = "warmup" if round_index < n_warm else "measured"
        for pos, s in enumerate(order):
            rec = dict(base, **s, run_kind=kind, round_index=round_index,
                       measured_index=(round_index - n_warm + 1) if kind == "measured" else None,
                       order_in_round=pos, order_seed=order_seed)
            details = {"graph_id": gid, "strategy": s["strategy"], "run_kind": kind, "round_index": round_index}
            group = f"run-{gid}-{s['strategy']}-{round_index}"
            r = None
            to = None
            try:
                rec["aqe_conf"] = apply_aqe(spark, s["aqe"])
                r = runner_for(s)
                before_ids = bench.snapshot_stage_ids(ui_url, app_id)
                rec["start_utc"] = utcnow()
                sc.setJobGroup(group, group, interruptOnCancel=True)
                with RunTimeout(sc, group, core["run_timeout_s"]) as to:
                    profiling_s = 0.0
                    if s["variant"] in PROFILED_VARIANTS:
                        tp = time.perf_counter()
                        r.profile()
                        if getattr(r, "hub_ids", None) is not None:
                            r.hub_ids.count()  # materialize before the query timer, as bench.py does
                        profiling_s = time.perf_counter() - tp
                    after_prof = bench.poll_completed_stages(ui_url, app_id)
                    before_query = set(after_prof.keys())
                    tq = time.perf_counter()
                    df = r.find(MOTIF)
                    path_count = int(df._jdf.queryExecution().toRdd().count())
                    query_s = time.perf_counter() - tq
                rec["end_utc"] = utcnow()
                after = bench.poll_completed_stages(ui_url, app_id)
                qm = bench.collect_window_metrics(ui_url, app_id, before_query, after)
                tm = bench.collect_window_metrics(ui_url, app_id, before_ids, after)
                dom = qm.get("dominant_stage") or {}
                dom_prof = (stage_task_profile(ui_url, app_id, dom["stageId"], dom.get("attemptId", 0))
                            if dom else {})
                try:
                    pa = run_single_ablation.analyze_plan(df)
                except Exception as e:
                    pa = {"error": str(e)}
                aqe_m = aqe_plan_metrics(df)
                td = dom.get("taskDurationMs") or {}
                dd = dom_prof.get("duration_ms") or {}
                sb = dom_prof.get("shuffle_read_bytes") or {}
                rec.update(
                    status="ok",
                    profiling_s=round(profiling_s, 4),
                    query_s=round(query_s, 4),
                    total_s=round(profiling_s + query_s, 4),
                    path_count=path_count,
                    correctness_status=("match" if expected is None or path_count == expected else "mismatch"),
                    hub_count=getattr(r, "hub_count", None),
                    num_stages=qm["num_stages"],
                    total_tasks=qm["total_tasks"],
                    executor_run_time_s=qm["total_executor_run_time_s"],
                    executor_cpu_time_s=qm["total_executor_cpu_time_s"],
                    executor_run_time_incl_profiling_s=tm["total_executor_run_time_s"],
                    shuffle_read_bytes=qm["total_shuffle_read_bytes"],
                    shuffle_write_bytes=qm["total_shuffle_write_bytes"],
                    memory_spill_bytes=qm["total_memory_bytes_spilled"],
                    disk_spill_bytes=qm["total_disk_bytes_spilled"],
                    peak_execution_memory_max=max((st["peakExecutionMemory"] for st in qm["stages"]), default=None),
                    dom_stage_id=dom.get("stageId"),
                    dom_stage_name=dom.get("name"),
                    dom_num_tasks=dom.get("numTasks"),
                    dom_task_median_ms=td.get("median"),
                    dom_task_p95_ms=td.get("p95"),
                    dom_task_max_ms=td.get("max"),
                    dom_task_max_median_ratio=td.get("max_median_ratio"),
                    dom_task_mean_ms=dd.get("mean"),
                    dom_task_cv=dom_prof.get("duration_cv"),
                    dom_shuffle_read_bytes_median=sb.get("median"),
                    dom_shuffle_read_bytes_p95=sb.get("p95"),
                    dom_shuffle_read_bytes_max=sb.get("max"),
                    dom_stage_spill_memory_bytes=dom.get("memoryBytesSpilled"),
                    dom_stage_spill_disk_bytes=dom.get("diskBytesSpilled"),
                    **(aqe_m.get("totals") or {}),
                    **summarize_plan_analysis(pa),
                )
                details.update(query_window=qm, incl_profiling_window=tm, dominant_stage_tasks=dom_prof,
                               aqe_plan_metrics=aqe_m, plan_analysis=pa)
            except Exception as e:
                rec.setdefault("start_utc", utcnow())
                rec["end_utc"] = utcnow()
                timed_out = to is not None and to.fired
                rec.update(status="timeout" if timed_out else "failed", error_type=type(e).__name__,
                           error_message=str(e)[:2000], correctness_status="unchecked")
                details["error_trace"] = traceback.format_exc()[-8000:]
            finally:
                sc.setLocalProperty("spark.jobGroup.id", None)
                if r is not None and hasattr(r, "uncache_topology"):
                    try:
                        r.uncache_topology()
                    except Exception:
                        pass
            emit(rec, details)
            print(f"[{gid}] {kind:<8} r{round_index} {s['strategy']:<28} {rec['status']:<8} "
                  f"q={rec.get('query_s')}s total={rec.get('total_s')}s paths={rec.get('path_count')} "
                  f"dom max/med={rec.get('dom_task_max_median_ratio')} cv={rec.get('dom_task_cv')} "
                  f"skew={rec.get('aqe_skewed_partitions')}", flush=True)

    spark.stop()


if __name__ == "__main__":
    main()
