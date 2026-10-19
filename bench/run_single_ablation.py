#!/usr/bin/env python3
"""
bench/run_single_ablation.py
============================
Executes a single controlled benchmark run for the AQE skew join / coalesce ablation.
Supports:
  - variant: edgejoin, native
  - aqe: on, off
  - skew_join: true, false
  - coalesce: true, false
  - run_id: int
  - out: output JSON path
"""

import argparse
import datetime
import io
import json
import os
import platform
import sys
import time
from typing import Dict, Any, List, Set, Optional

import psutil
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PKG_ROOT = os.path.join(REPO_ROOT, "hypergraphframes")

for p in [THIS_DIR, PKG_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import bench
from salting_1d import EdgeJoinGraphFrame

DEFAULT_SHUFFLE_PARTITIONS = 200
MOTIF_QUERY = "(a)-[e1]->(b); (b)-[e2]->(c)"
AQE_SKEWED_THRESHOLD_BYTES = 268435456  # 256 MB (default)
AQE_SKEWED_FACTOR = 5.0  # default factor


def find_aqe_reads(node, seen=None):
    """Recursively discover all AQEShuffleReadExec nodes in the executed plan tree."""
    if seen is None:
        seen = set()
    h = node.hashCode()
    if h in seen:
        return []
    seen.add(h)
    res = []
    cls_name = node.getClass().getSimpleName()
    if "AQEShuffleRead" in cls_name:
        res.append(node)
    if "AdaptiveSparkPlan" in cls_name:
        try:
            res.extend(find_aqe_reads(node.executedPlan(), seen))
        except Exception:
            pass
    if "QueryStage" in cls_name:
        try:
            res.extend(find_aqe_reads(node.plan(), seen))
        except Exception:
            pass
    children = node.children()
    for i in range(children.length()):
        res.extend(find_aqe_reads(children.apply(i), seen))
    return res


def analyze_plan(df) -> Dict[str, Any]:
    """Inspect df._jdf.queryExecution().executedPlan() for AQE shuffle reads and skew splits."""
    plan = df._jdf.queryExecution().executedPlan()
    plan_str = plan.toString()
    is_final = "isFinalPlan=true" in plan_str

    reads = find_aqe_reads(plan)
    aqe_nodes_info = []
    any_skew_splits = False

    for r in reads:
        simple_str = r.simpleString(100)
        has_coalesced = bool(r.hasCoalescedPartition())
        has_skewed = bool(r.hasSkewedPartition())
        is_coalesced_read = bool(r.isCoalescedRead())
        specs_len = r.partitionSpecs().length()
        specs = [r.partitionSpecs().apply(i).toString() for i in range(specs_len)]

        spec_types = {}
        for s in specs:
            stype = s.split("(")[0]
            spec_types[stype] = spec_types.get(stype, 0) + 1

        is_skew_split = has_skewed or any(
            "Partial" in st for st in spec_types.keys()
        ) or "skewed" in simple_str.lower()
        if is_skew_split:
            any_skew_splits = True

        aqe_nodes_info.append({
            "simple_string": simple_str,
            "has_coalesced_partition": has_coalesced,
            "has_skewed_partition": has_skewed,
            "is_coalesced_read": is_coalesced_read,
            "num_partition_specs": specs_len,
            "spec_types": spec_types,
            "sample_specs": specs[:10],
            "is_skew_split": is_skew_split,
        })

    return {
        "is_final_plan": is_final,
        "num_aqe_shuffle_reads": len(reads),
        "any_show_skew_splits": any_skew_splits,
        "aqe_shuffle_reads": aqe_nodes_info,
        "plan_string_snippet": plan_str[:2000],
    }


def fetch_stage_task_details(ui_url: str, app_id: str, stage_id: int, attempt_id: int) -> Dict[str, Any]:
    """Fetch taskSummary quantiles and taskList from Spark REST API for detailed partition inspection."""
    ts_url = (
        f"{ui_url}/api/v1/applications/{app_id}/stages/{stage_id}/{attempt_id}"
        f"/taskSummary?quantiles=0.0,0.25,0.5,0.75,0.95,1.0"
    )
    task_summary = {}
    try:
        r = requests.get(ts_url, timeout=10)
        if r.status_code == 200:
            task_summary = r.json()
    except Exception as e:
        task_summary = {"error": str(e)}

    # Also fetch individual task list
    tl_url = f"{ui_url}/api/v1/applications/{app_id}/stages/{stage_id}/{attempt_id}/taskList?length=1000"
    task_list = []
    try:
        r2 = requests.get(tl_url, timeout=15)
        if r2.status_code == 200:
            task_list = r2.json()
    except Exception:
        task_list = []

    tasks_parsed = []
    for t in task_list:
        metrics = t.get("taskMetrics") or {}
        sread = metrics.get("shuffleReadMetrics") or {}
        read_bytes = sread.get("localBytesRead", 0) + sread.get("remoteBytesRead", 0)
        read_records = sread.get("recordsRead", 0)
        tasks_parsed.append({
            "task_id": t.get("taskId"),
            "index": t.get("index"),
            "duration_ms": t.get("duration", 0),
            "shuffle_read_bytes": read_bytes,
            "shuffle_read_records": read_records,
            "mem_spill_bytes": metrics.get("memoryBytesSpilled", 0),
            "disk_spill_bytes": metrics.get("diskBytesSpilled", 0),
        })

    tasks_by_bytes = sorted(tasks_parsed, key=lambda x: x["shuffle_read_bytes"], reverse=True)
    tasks_by_records = sorted(tasks_parsed, key=lambda x: x["shuffle_read_records"], reverse=True)
    tasks_by_duration = sorted(tasks_parsed, key=lambda x: x["duration_ms"], reverse=True)

    max_bytes_task = tasks_by_bytes[0] if tasks_by_bytes else None
    median_bytes_task = (
        tasks_by_bytes[len(tasks_by_bytes) // 2] if tasks_by_bytes else None
    )

    max_records_task = tasks_by_records[0] if tasks_by_records else None
    median_records_task = (
        tasks_by_records[len(tasks_by_records) // 2] if tasks_by_records else None
    )

    max_duration_task = tasks_by_duration[0] if tasks_by_duration else None
    median_duration_task = (
        tasks_by_duration[len(tasks_by_duration) // 2] if tasks_by_duration else None
    )

    return {
        "task_summary": task_summary,
        "num_tasks_fetched": len(tasks_parsed),
        "max_bytes_task": max_bytes_task,
        "median_bytes_task": median_bytes_task,
        "max_records_task": max_records_task,
        "median_records_task": median_records_task,
        "max_duration_task": max_duration_task,
        "median_duration_task": median_duration_task,
    }


def main():
    parser = argparse.ArgumentParser(description="AQE Skew Join Ablation Single Run")
    parser.add_argument("--variant", required=True, choices=["edgejoin", "native"])
    parser.add_argument("--aqe", required=True, choices=["on", "off"])
    parser.add_argument("--skew-join", required=True, choices=["true", "false"])
    parser.add_argument("--coalesce", required=True, choices=["true", "false"])
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    bench.setup_env()
    aqe_bool = "true" if args.aqe.lower() == "on" else "false"
    skew_bool = args.skew_join.lower()
    coalesce_bool = args.coalesce.lower()

    app_name = (
        f"Ablation_{args.variant}_aqe_{args.aqe}_skew_{skew_bool}_coal_{coalesce_bool}_r{args.run_id}"
    )

    gf_pkg = bench.resolve_graphframes_package()
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.extraJavaOptions", bench.JAVA_ADD_OPENS)
        .config("spark.executor.extraJavaOptions", bench.JAVA_ADD_OPENS)
        .config("spark.driver.memory", "6g")
        .config(*bench.graphframes_jar_conf())
        .config("spark.sql.shuffle.partitions", str(DEFAULT_SHUFFLE_PARTITIONS))
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.adaptive.enabled", aqe_bool)
        .config("spark.sql.adaptive.skewJoin.enabled", skew_bool)
        .config("spark.sql.adaptive.coalescePartitions.enabled", coalesce_bool)
        .config("spark.ui.enabled", "true")
        .config("spark.ui.retainedTasks", "1000000")
        .config("spark.ui.retainedStages", "10000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        app_id = spark.sparkContext.applicationId
        ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
        parquet_path = bench.resolve_parquet_path()

        print(f"[{app_name}] Spark UI: {ui_url} | App ID: {app_id}")

        # 1. Load, repartition, cache, and materialize edges BEFORE timer
        raw_edges = spark.read.parquet(parquet_path).select(
            col("src").cast("int"), col("dst").cast("int")
        )
        edges = raw_edges.repartition(DEFAULT_SHUFFLE_PARTITIONS).cache()
        edge_count = int(edges.count())

        # 2. Build, cache, and materialize vertices BEFORE timer
        base_vertices = (
            edges.select(col("src").alias("id"))
            .union(edges.select(col("dst").alias("id")))
            .distinct()
        )
        vertices_cached = base_vertices.cache()
        vertex_count = int(vertices_cached.count())
        vertices = broadcast(vertices_cached)

        print(f"[{app_name}] Materialized {edge_count:,} edges and {vertex_count:,} vertices.")

        from graphframes import GraphFrame
        g = GraphFrame(vertices, edges)

        if args.variant == "native":
            target_gf = g
        elif args.variant == "edgejoin":
            target_gf = EdgeJoinGraphFrame(g, skip_id_only_vertex_join=True)
        else:
            raise ValueError(f"Unknown variant: {args.variant}")

        # Pre-query stage snapshot
        before_query_ids = bench.snapshot_stage_ids(ui_url, app_id)

        # Timed query window
        t0 = time.perf_counter()
        df = target_gf.find(MOTIF_QUERY)
        path_count = int(df._jdf.queryExecution().toRdd().count())
        query_s = round(time.perf_counter() - t0, 4)

        # Wait for completed stages to settle
        after_completed = bench.poll_completed_stages(ui_url, app_id)
        query_metrics = bench.collect_window_metrics(ui_url, app_id, before_query_ids, after_completed)

        # Dominant stage detailed task analysis
        dom_stage = query_metrics.get("dominant_stage") or {}
        dom_sid = dom_stage.get("stageId")
        dom_attempt = dom_stage.get("attemptId", 0)

        dom_task_details = {}
        if dom_sid is not None:
            dom_task_details = fetch_stage_task_details(ui_url, app_id, dom_sid, dom_attempt)

        # Analyze final executed plan
        plan_analysis = analyze_plan(df)

        # Save formatted explain plan
        out_abs = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_abs)
        plans_dir = os.path.join(out_dir, "plans")
        os.makedirs(plans_dir, exist_ok=True)

        plan_filename = f"plan_{args.variant}_aqe_{args.aqe}_skew_{skew_bool}_coal_{coalesce_bool}_r{args.run_id}.txt"
        plan_path = os.path.join(plans_dir, plan_filename)
        with open(plan_path, "w", encoding="utf-8") as pf:
            pf.write(df._jdf.queryExecution().executedPlan().toString())

        # Compile output payload
        payload = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config": {
                "variant": args.variant,
                "regime": "id_only",
                "aqe": args.aqe,
                "skew_join": skew_bool == "true",
                "coalesce": coalesce_bool == "true",
                "run_id": args.run_id,
                "shuffle_partitions": DEFAULT_SHUFFLE_PARTITIONS,
                "skewed_partition_threshold_bytes": AQE_SKEWED_THRESHOLD_BYTES,
                "skewed_partition_factor": AQE_SKEWED_FACTOR,
            },
            "timings": {
                "query_s": query_s,
            },
            "dataset": {
                "edges_count": edge_count,
                "vertices_count": vertex_count,
                "path_count": path_count,
            },
            "metrics": query_metrics,
            "dominant_stage_task_details": dom_task_details,
            "plan_analysis": plan_analysis,
            "plan_file": os.path.relpath(plan_path, REPO_ROOT),
        }

        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        dom_td = dom_stage.get("taskDurationMs") or {}
        max_bytes_info = dom_task_details.get("max_bytes_task") or {}
        med_bytes_info = dom_task_details.get("median_bytes_task") or {}

        print(
            f"[{app_name}] DONE | query_s={query_s:.3f}s | paths={path_count:,} | "
            f"dom_stage={dom_sid} (tasks={dom_stage.get('numTasks')}, max/med={dom_td.get('max_median_ratio')}) | "
            f"mem_spill={query_metrics.get('total_memory_bytes_spilled', 0) / (1024**2):.1f}MB | "
            f"disk_spill={query_metrics.get('total_disk_bytes_spilled', 0) / (1024**2):.1f}MB | "
            f"aqe_reads={plan_analysis['num_aqe_shuffle_reads']} (skew_splits={plan_analysis['any_show_skew_splits']}) | "
            f"dom_shuffle_bytes: max={max_bytes_info.get('shuffle_read_bytes', 0)/(1024**2):.2f}MB, "
            f"med={med_bytes_info.get('shuffle_read_bytes', 0)/(1024**2):.2f}MB"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
