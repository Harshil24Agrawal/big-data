#!/usr/bin/env python3
"""
pokec_benchmark.py
==================
Benchmarks topology-aware hypercube partitioning against the standard
relational join on soc-pokec (1.6M vertices, 30.6M directed edges).

Query: 2-hop motif  (a)-[e1]->(b)-[e2]->(c)
       Standard relational join: e1.join(e2, e1.dst == e2.src)

Conditions:
  0. GraphFrames find() baseline
  1. Baseline AQE Off  - Standard join, no adaptive optimization
  2. Baseline AQE On (Default 256MB threshold)
  3. Baseline AQE On (1MB threshold)
  4. Baseline AQE On (1MB threshold, No Coalesce)
  5. 1D Salting
  6. Hypercube Planner  - Degree-balanced 2D grid partitioning (our approach)
"""

import os
import sys
import time
import json
import statistics
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, expr, explode, sequence, broadcast
)

POKEC_PARQUET = "pokec_edges_parquet"
P_CELLS = 100
WORK_THRESHOLD = 1_000_000
WARMUP_RUNS = 1
MEASURED_RUNS = 5

def setup_env():
    if os.name == "nt":
        hadoop_home = os.path.join(os.getcwd(), "hadoop")
        if os.path.exists(hadoop_home):
            os.environ["HADOOP_HOME"] = hadoop_home
            os.environ["hadoop.home.dir"] = hadoop_home
            bin_dir = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

def create_spark():
    spark = (
        SparkSession.builder.appName("PokecBenchmark")
        .master("local[*]")
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED")
        .config("spark.executor.extraJavaOptions", "--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED")
        .config("spark.driver.memory", "6g")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def snapshot_stages(app_id, ui_url):
    try:
        url = f"{ui_url}/api/v1/applications/{app_id}/stages"
        stages = requests.get(url, timeout=5).json()
        return {s["stageId"] for s in stages if s.get("status") == "COMPLETE"}
    except Exception:
        return set()

def collect_metrics(app_id, ui_url, before_ids):
    time.sleep(1.5)
    try:
        url = f"{ui_url}/api/v1/applications/{app_id}/stages"
        all_stages = requests.get(url, timeout=5).json()
        completed = {
            s["stageId"]: s
            for s in all_stages
            if s.get("status") == "COMPLETE"
        }
        new_ids = sorted(set(completed.keys()) - before_ids)
        if not new_ids:
            return None

        total_shuffle_read = 0
        total_shuffle_write = 0
        total_disk_spill = 0
        total_mem_spill = 0

        stage_metrics_list = []

        for sid in new_ids:
            st = completed[sid]
            total_shuffle_read += st.get("shuffleReadBytes", 0)
            total_shuffle_write += st.get("shuffleWriteBytes", 0)
            total_disk_spill += st.get("diskBytesSpilled", 0)
            total_mem_spill += st.get("memoryBytesSpilled", 0)

            attempt = st.get("attemptId", 0)
            tasks_url = (
                f"{ui_url}/api/v1/applications/{app_id}"
                f"/stages/{sid}/{attempt}/taskList?length=10000"
            )
            try:
                tasks = requests.get(tasks_url, timeout=10).json()
                if isinstance(tasks, list):
                    durations = [t.get("duration", 0) for t in tasks]
                    if durations:
                        durations.sort()
                        n = len(durations)
                        mean_d = statistics.mean(durations)
                        std_d = statistics.stdev(durations) if n > 1 else 0.0
                        stage_metrics_list.append({
                            "max_task_ms": durations[-1],
                            "p95_task_ms": durations[int(n * 0.95)],
                            "median_task_ms": durations[n // 2],
                            "task_cv": round(std_d / mean_d, 4) if mean_d > 0 else 0,
                            "num_tasks": n
                        })
            except Exception:
                pass

        if not stage_metrics_list:
            return None

        bottleneck = max(stage_metrics_list, key=lambda x: x["max_task_ms"])

        return {
            "num_tasks": bottleneck["num_tasks"],
            "max_task_ms": bottleneck["max_task_ms"],
            "p95_task_ms": bottleneck["p95_task_ms"],
            "median_task_ms": bottleneck["median_task_ms"],
            "task_cv": bottleneck["task_cv"],
            "shuffle_read_bytes": total_shuffle_read,
            "shuffle_write_bytes": total_shuffle_write,
            "disk_spill_bytes": total_disk_spill,
            "mem_spill_bytes": total_mem_spill,
        }
    except Exception as e:
        print(f"  [metrics error] {e}")
        return None

def analyze_topology(edges):
    print("\n" + "=" * 70)
    print("TOPOLOGY ANALYSIS")
    print("=" * 70)

    out_deg = edges.groupBy("src").count().withColumnRenamed("src", "id").withColumnRenamed("count", "out_deg")
    in_deg = edges.groupBy("dst").count().withColumnRenamed("dst", "id").withColumnRenamed("count", "in_deg")
    degrees = out_deg.join(in_deg, "id", "full").fillna(0)
    degrees = degrees.withColumn("work", col("in_deg") * col("out_deg")).cache()

    row = degrees.selectExpr(
        "count(*) as n_vertices",
        "max(in_deg) as max_in",
        "max(out_deg) as max_out",
        "max(work) as max_work",
        "avg(in_deg) as avg_in",
        "avg(out_deg) as avg_out",
        "percentile_approx(in_deg, 0.99) as p99_in",
        "percentile_approx(out_deg, 0.99) as p99_out",
        "sum(work) as total_paths"
    ).collect()[0]

    expected_paths = int(row["total_paths"]) if row["total_paths"] is not None else 0

    print(f"  Vertices:           {row['n_vertices']:>12,}")
    print(f"  Max work (d-*d+):   {row['max_work']:>12,}")
    print(f"  Total 2-hop paths:  {expected_paths:>12,}")

    hubs = degrees.filter(col("work") > WORK_THRESHOLD)

    # Correct Algorithm.md formula
    hub_meta = (
        hubs.withColumn("raw_r", expr(f"sqrt(in_deg * {P_CELLS}.0 / greatest(out_deg, 1))"))
        .withColumn("raw_s", expr(f"sqrt(out_deg * {P_CELLS}.0 / greatest(in_deg, 1))"))
        .withColumn("scale", expr(f"IF(raw_r * raw_s > {P_CELLS}, sqrt({P_CELLS}.0 / (raw_r * raw_s)), 1.0)"))
        .withColumn("r_final", expr("cast(greatest(1, floor(raw_r * scale)) as int)"))
        .withColumn("s_final", expr("cast(greatest(1, floor(raw_s * scale)) as int)"))
        .select("id", "in_deg", "out_deg", "work", "r_final", "s_final")
        .cache()
    )

    hub_count = hub_meta.count()
    print(f"\n  Hubs identified:    {hub_count:>12,}")

    top = hub_meta.orderBy(col("work").desc()).limit(10).collect()
    for h in top:
        print(f"  {h['id']:>10}  {h['work']:>14,}  {h['r_final']:>4} x {h['s_final']:<4}")

    hub_ids = hub_meta.select("id").cache()

    topo_info = {
        "vertices": row["n_vertices"],
        "max_work": row["max_work"],
        "total_paths": expected_paths,
        "num_hubs": hub_count,
        "P_cells": P_CELLS,
    }

    return hub_meta, hub_ids, topo_info, expected_paths

def run_graphframes(edges, spark):
    from graphframes import GraphFrame
    v = edges.select(col("src").alias("id")).union(edges.select(col("dst").alias("id"))).distinct()
    g = GraphFrame(v, edges)
    return g.find("(A)-[]->(B); (B)-[]->(C)").count()

def run_1d_salting(edges, hub_meta, hub_ids):
    if hub_ids.count() == 0:
        return run_baseline(edges)
    e1_hub = edges.join(broadcast(hub_meta), edges["dst"] == hub_meta["id"])
    e1_salted = e1_hub.withColumn("salt", expr(f"pmod(hash(src), {P_CELLS})")).select("src", "dst", "salt").alias("e1h")
    e2_hub = edges.join(broadcast(hub_meta), edges["src"] == hub_meta["id"])
    e2_salted = e2_hub.withColumn("salt", explode(sequence(lit(0), lit(P_CELLS - 1)))).select("src", "dst", "salt").alias("e2h")
    hub_count = e1_salted.join(e2_salted, (col("e1h.dst") == col("e2h.src")) & (col("e1h.salt") == col("e2h.salt"))).count()
    e1_nonhub = edges.join(broadcast(hub_ids), edges["dst"] == hub_ids["id"], "left_anti").alias("e1n")
    e2_nonhub = edges.join(broadcast(hub_ids), edges["src"] == hub_ids["id"], "left_anti").alias("e2n")
    nonhub_count = e1_nonhub.join(e2_nonhub, col("e1n.dst") == col("e2n.src")).count()
    return hub_count + nonhub_count

def run_baseline(edges):
    e1 = edges.alias("e1")
    e2 = edges.alias("e2")
    return e1.join(e2, col("e1.dst") == col("e2.src")).count()

def run_hypercube(edges, hub_meta, hub_ids):
    e1_hub = edges.join(broadcast(hub_meta), edges["dst"] == hub_meta["id"]).select(
        edges["src"], edges["dst"], hub_meta["r_final"].alias("r"), hub_meta["s_final"].alias("s")
    )
    e1_salted = e1_hub.withColumn("r_coord", expr("pmod(hash(src), r)")).withColumn("s_coord", explode(sequence(lit(0), col("s") - 1))).select("src", "dst", "r_coord", "s_coord").alias("e1h")

    e2_hub = edges.join(broadcast(hub_meta), edges["src"] == hub_meta["id"]).select(
        edges["src"], edges["dst"], hub_meta["r_final"].alias("r"), hub_meta["s_final"].alias("s")
    )
    e2_salted = e2_hub.withColumn("s_coord", expr("pmod(hash(dst), s)")).withColumn("r_coord", explode(sequence(lit(0), col("r") - 1))).select("src", "dst", "r_coord", "s_coord").alias("e2h")

    hub_count = e1_salted.join(e2_salted, (col("e1h.dst") == col("e2h.src")) & (col("e1h.r_coord") == col("e2h.r_coord")) & (col("e1h.s_coord") == col("e2h.s_coord"))).count()

    e1_nonhub = edges.join(broadcast(hub_ids), edges["dst"] == hub_ids["id"], "left_anti").alias("e1n")
    e2_nonhub = edges.join(broadcast(hub_ids), edges["src"] == hub_ids["id"], "left_anti").alias("e2n")
    nonhub_count = e1_nonhub.join(e2_nonhub, col("e1n.dst") == col("e2n.src")).count()

    return hub_count + nonhub_count

def run_hypercube_unified(edges, hub_meta, hub_ids=None):
    hub_meta_local = hub_meta.collect()
    if len(hub_meta_local) == 0:
        hub_map_str = "map(cast(-1 as bigint), struct(cast(1 as int) as r, cast(1 as int) as s))"
    else:
        map_entries = []
        for h in hub_meta_local:
            map_entries.append(f"cast({h['id']} as bigint), struct(cast({h['r_final']} as int) as r, cast({h['s_final']} as int) as s)")
        hub_map_str = f"map({', '.join(map_entries)})"

    e1 = (
        edges.withColumn("hub_info", expr(f"element_at({hub_map_str}, dst)"))
        .withColumn("r", expr("coalesce(hub_info.r, 1)"))
        .withColumn("s", expr("coalesce(hub_info.s, 1)"))
        .withColumn("r_coord", expr("pmod(hash(src), r)"))
        .withColumn("s_coord", expr("explode(sequence(0, s - 1))"))
        .select("src", "dst", "r_coord", "s_coord")
        .alias("e1")
    )
    e2 = (
        edges.withColumn("hub_info", expr(f"element_at({hub_map_str}, src)"))
        .withColumn("r", expr("coalesce(hub_info.r, 1)"))
        .withColumn("s", expr("coalesce(hub_info.s, 1)"))
        .withColumn("s_coord", expr("pmod(hash(dst), s)"))
        .withColumn("r_coord", expr("explode(sequence(0, r - 1))"))
        .select("src", "dst", "r_coord", "s_coord")
        .alias("e2")
    )
    return e1.join(
        e2,
        (col("e1.dst") == col("e2.src")) &
        (col("e1.r_coord") == col("e2.r_coord")) &
        (col("e1.s_coord") == col("e2.s_coord"))
    ).count()

def run_condition(name, spark, app_id, edges, expected_paths, hub_meta=None, hub_ids=None, runner_func=None):
    print(f"\n{'=' * 70}\nCONDITION: {name}\n{'=' * 70}")
    ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
    all_runs = []
    for i in range(WARMUP_RUNS + MEASURED_RUNS):
        label = "WARMUP" if i < WARMUP_RUNS else f"Run {i - WARMUP_RUNS + 1}"
        print(f"\n  [{label}]", end=" ", flush=True)
        before = snapshot_stages(app_id, ui_url)
        t0 = time.time()

        if runner_func:
            path_count = runner_func(edges, spark) if name == "GraphFrames" else runner_func(edges, hub_meta, hub_ids)
        elif hub_meta is not None:
            path_count = run_hypercube(edges, hub_meta, hub_ids)
        else:
            path_count = run_baseline(edges)

        wall_s = time.time() - t0
        metrics = collect_metrics(app_id, ui_url, before) or {}
        metrics["wall_clock_s"] = round(wall_s, 3)
        metrics["path_count"] = path_count

        assert path_count == expected_paths, f"Correctness error: got {path_count:,}, expected {expected_paths:,}"

        print(f"paths={path_count:,}  wall={wall_s:.1f}s  max_task={metrics.get('max_task_ms', '?')}ms  cv={metrics.get('task_cv', '?')}  spill={metrics.get('disk_spill_bytes', 0) / 1024 / 1024:.1f}MB")
        if i >= WARMUP_RUNS:
            all_runs.append(metrics)
    return all_runs

def aggregate_runs(runs):
    summary = {}
    keys = ["wall_clock_s", "max_task_ms", "p95_task_ms", "median_task_ms", "task_cv", "shuffle_read_bytes", "shuffle_write_bytes", "disk_spill_bytes", "mem_spill_bytes"]
    for k in keys:
        vals = [r[k] for r in runs if k in r and isinstance(r[k], (int, float))]
        if vals:
            summary[k] = {"mean": round(statistics.mean(vals), 3), "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0, "raw": vals}
    summary["path_count"] = runs[0].get("path_count", 0) if runs else 0
    return summary

def main():
    setup_env()
    if not os.path.exists(POKEC_PARQUET):
        print(f"Error: {POKEC_PARQUET} not found. Ensure dataset is available.")
        sys.exit(1)

    spark = create_spark()
    app_id = spark.sparkContext.applicationId
    print(f"Spark App ID: {app_id}")

    edges = spark.read.parquet(POKEC_PARQUET).select(col("src").cast("int"), col("dst").cast("int"))
    edges = edges.repartition(200).cache()
    edges.count()

    hub_meta, hub_ids, topo_info, expected_paths = analyze_topology(edges)
    results = {}

    # 0. GraphFrames Baseline
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    runs = run_condition("GraphFrames", spark, app_id, edges, expected_paths, runner_func=run_graphframes)
    results["graphframes"] = {"condition": "GraphFrames", "aqe": False, "runs": runs, "summary": aggregate_runs(runs)}

    # 1. Baseline AQE Off
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    runs = run_condition("Baseline (AQE Off)", spark, app_id, edges, expected_paths)
    results["baseline_aqe_off"] = {"condition": "Baseline (AQE Off)", "aqe": False, "runs": runs, "summary": aggregate_runs(runs)}

    # 2. Baseline AQE On (Default 256MB threshold)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", str(256 * 1024 * 1024))
    runs = run_condition("Baseline (AQE On - 256MB)", spark, app_id, edges, expected_paths)
    results["baseline_aqe_on_default"] = {"condition": "Baseline (AQE On - 256MB)", "aqe": True, "runs": runs, "summary": aggregate_runs(runs)}

    # 3. Baseline AQE On (1MB threshold)
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", str(1024 * 1024))
    runs = run_condition("Baseline (AQE On - 1MB)", spark, app_id, edges, expected_paths)
    results["baseline_aqe_on_1mb"] = {"condition": "Baseline (AQE On - 1MB)", "aqe": True, "runs": runs, "summary": aggregate_runs(runs)}

    # 4. Baseline AQE On (1MB threshold, No Coalesce)
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
    runs = run_condition("Baseline (AQE On - 1MB, No Coalesce)", spark, app_id, edges, expected_paths)
    results["baseline_aqe_on_1mb_nocoalesce"] = {"condition": "Baseline (AQE On - 1MB, No Coalesce)", "aqe": True, "runs": runs, "summary": aggregate_runs(runs)}

    # 5. 1D Salting
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    runs = run_condition("1D Salting", spark, app_id, edges, expected_paths, hub_meta, hub_ids, runner_func=run_1d_salting)
    results["1d_salting"] = {"condition": "1D Salting", "aqe": False, "runs": runs, "summary": aggregate_runs(runs)}

    # 6. Hypercube Planner (Split)
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    runs = run_condition("Hypercube Planner (Split)", spark, app_id, edges, expected_paths, hub_meta, hub_ids)
    results["hypercube"] = {"condition": "Hypercube Planner (Split)", "aqe": False, "runs": runs, "summary": aggregate_runs(runs)}

    # 7. Hypercube Planner (Unified)
    runs = run_condition("Hypercube Planner (Unified)", spark, app_id, edges, expected_paths, hub_meta, hub_ids, runner_func=run_hypercube_unified)
    results["hypercube_unified"] = {"condition": "Hypercube Planner (Unified)", "aqe": False, "runs": runs, "summary": aggregate_runs(runs)}

    output = {
        "dataset": "soc-pokec",
        "edges": edge_count,
        "topology": topo_info,
        "config": {
            "P_cells": P_CELLS,
            "work_threshold": WORK_THRESHOLD,
            "shuffle_partitions": 200,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
        },
        "results": results,
    }
    with open("pokec_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    spark.stop()

if __name__ == "__main__":
    main()