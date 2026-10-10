#!/usr/bin/env python3
"""
baseline_graphframes_metrics.py
===============================
Executes standard unoptimized GraphFrames 2-hop motif query on soc-pokec
and extracts stage/task metrics via Spark REST API for empirical reporting.
"""

import os
import sys
import time
import json
import statistics
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def setup_env():
    """Inject native Hadoop binaries on Windows."""
    if os.name == "nt":
        hadoop_home = os.path.join(os.getcwd(), "hadoop")
        if os.path.exists(hadoop_home):
            os.environ["HADOOP_HOME"] = hadoop_home
            os.environ["hadoop.home.dir"] = hadoop_home
            bin_dir = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def create_spark():
    setup_env()
    spark = (
        SparkSession.builder.appName("BaselineGraphFramesMetrics")
        .master("local[*]")
        .config(
            "spark.driver.extraJavaOptions",
            "--add-opens=java.base/java.lang=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
            "--add-opens=java.base/java.io=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/java.util=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
            "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "--add-opens=java.base/java.lang=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
            "--add-opens=java.base/java.io=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/java.util=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
            "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        )
        .config("spark.driver.memory", "6g")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def snapshot_stages(app_id, ui_url):
    """Retrieve all completed stage IDs prior to query execution."""
    try:
        url = f"{ui_url}/api/v1/applications/{app_id}/stages"
        stages = requests.get(url, timeout=5).json()
        return {s["stageId"] for s in stages if s.get("status") == "COMPLETE"}
    except Exception:
        return set()


def collect_metrics(app_id, ui_url, before_ids):
    """Poll Spark REST API and parse task straggler and spill metrics."""
    time.sleep(2.0)  # Allow UI listener bus to flush event log
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
            tasks_url = f"{ui_url}/api/v1/applications/{app_id}/stages/{sid}/{attempt}/taskList?length=10000"
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
                            "stage_id": sid,
                            "num_tasks": n,
                            "max_task_ms": durations[-1],
                            "p95_task_ms": durations[int(n * 0.95)],
                            "median_task_ms": durations[n // 2],
                            "mean_task_ms": round(mean_d, 2),
                            "task_cv": round(std_d / mean_d, 4) if mean_d > 0 else 0.0,
                        })
            except Exception:
                pass

        if not stage_metrics_list:
            return None

        # Identify bottleneck join stage by maximum task duration
        bottleneck = max(stage_metrics_list, key=lambda x: x["max_task_ms"])

        return {
            "bottleneck_stage_id": bottleneck["stage_id"],
            "num_tasks": bottleneck["num_tasks"],
            "max_task_ms": bottleneck["max_task_ms"],
            "p95_task_ms": bottleneck["p95_task_ms"],
            "median_task_ms": bottleneck["median_task_ms"],
            "mean_task_ms": bottleneck["mean_task_ms"],
            "task_cv": bottleneck["task_cv"],
            "shuffle_read_bytes": total_shuffle_read,
            "shuffle_write_bytes": total_shuffle_write,
            "disk_spill_bytes": total_disk_spill,
            "mem_spill_bytes": total_mem_spill,
        }
    except Exception as e:
        print(f"[metrics extraction failed]: {e}")
        return None


def main():
    parquet_path = "pokec_edges_parquet"
    if not os.path.exists(parquet_path):
        print(f"Error: dataset path '{parquet_path}' does not exist.")
        sys.exit(1)

    spark = create_spark()
    app_id = spark.sparkContext.applicationId
    ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
    print(f"Spark UI available at: {ui_url} (App ID: {app_id})")

    edges = (
        spark.read.parquet(parquet_path)
        .select(col("src").cast("int"), col("dst").cast("int"))
        .repartition(200)
        .cache()
    )
    edges.count()

    vertices = (
        edges.select(col("src").alias("id"))
        .union(edges.select(col("dst").alias("id")))
        .distinct()
    )

    from graphframes import GraphFrame
    g = GraphFrame(vertices, edges)

    motif_query = "(a)-[e1]->(b); (b)-[e2]->(c)"
    print(f"\nExecuting native GraphFrames motif query: {motif_query}")
    motifs = g.find(motif_query)

    before_ids = snapshot_stages(app_id, ui_url)
    t0 = time.time()
    total_paths = motifs.count()
    wall_clock_s = round(time.time() - t0, 3)

    metrics = collect_metrics(app_id, ui_url, before_ids) or {}
    metrics["wall_clock_s"] = wall_clock_s
    metrics["total_paths"] = total_paths

    # Format values for publication-ready display
    disk_spill_mb = round(metrics.get("disk_spill_bytes", 0) / (1024 * 1024), 2)
    mem_spill_mb = round(metrics.get("mem_spill_bytes", 0) / (1024 * 1024), 2)
    shuf_read_mb = round(metrics.get("shuffle_read_bytes", 0) / (1024 * 1024), 2)
    shuf_write_mb = round(metrics.get("shuffle_write_bytes", 0) / (1024 * 1024), 2)
    max_task_s = round(metrics.get("max_task_ms", 0) / 1000, 2)
    p95_task_s = round(metrics.get("p95_task_ms", 0) / 1000, 2)
    median_task_s = round(metrics.get("median_task_ms", 0) / 1000, 2)

    print("\n" + "=" * 78)
    print("EMPIRICAL BENCHMARK METRICS (Baseline Native GraphFrames)")
    print("=" * 78)
    print(f"Total 2-Hop Motif Paths Evaluated : {total_paths:,}")
    print(f"End-to-End Wall-Clock Latency     : {wall_clock_s} s")
    print("-" * 78)
    print("STRAGGLER & WORKLOAD VARIANCE:")
    print(f"  Max Task Duration (max_task_ms) : {metrics.get('max_task_ms', 0):,} ms ({max_task_s} s)")
    print(f"  P95 Task Duration (p95_task_ms) : {metrics.get('p95_task_ms', 0):,} ms ({p95_task_s} s)")
    print(f"  Median Task Duration            : {metrics.get('median_task_ms', 0):,} ms ({median_task_s} s)")
    print(f"  Task Duration Coeff. of Var (CV): {metrics.get('task_cv', 0)}")
    print("-" * 78)
    print("INTERNAL BUFFER SPILL & NETWORK PROFILE:")
    print(f"  Disk Spill (disk_spill_bytes)   : {disk_spill_mb} MB  ({metrics.get('disk_spill_bytes', 0):,} bytes)")
    print(f"  Memory Spill (mem_spill_bytes)  : {mem_spill_mb} MB  ({metrics.get('mem_spill_bytes', 0):,} bytes)")
    print(f"  Total Shuffle Read              : {shuf_read_mb} MB")
    print(f"  Total Shuffle Write             : {shuf_write_mb} MB")
    print("=" * 78)

    # Export structured metrics to JSON for LaTeX/paper compilation
    with open("baseline_graphframes_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics written to baseline_graphframes_metrics.json")

    spark.stop()


if __name__ == "__main__":
    main()
