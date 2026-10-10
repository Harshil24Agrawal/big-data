#!/usr/bin/env python3
"""
hypercube_2hop_runner.py
========================
Executes 2-hop motif discovery: (a)-[e1]->(b); (b)-[e2]->(c)
using the newly installed HyperGraphFrames middleware library (hypergraph).
"""

import os
import sys
import time
import requests
import statistics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame
from hypergraph import HyperGraphFrame, __version__

# ---------------------------------------------------------------------------
# Configuration Parameters
# ---------------------------------------------------------------------------
POKEC_PARQUET = "pokec_edges_parquet"
P_CELLS = 100            # Total grid cells per hub (e.g., 10x10, 5x20, etc.)
WORK_THRESHOLD = 1_000_000 # d_in * d_out threshold to classify vertex as hub
SHUFFLE_PARTITIONS = 200

def setup_env():
    if os.name == "nt":
        env_hadoop = os.environ.get("HADOOP_HOME")
        candidates = [
            *( [env_hadoop] if env_hadoop else [] ),
            os.path.join(os.getcwd(), "hadoop"),
            os.path.abspath(os.path.join(os.getcwd(), "..", "hadoop")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "hadoop")),
        ]
        for hadoop_home in candidates:
            if os.path.exists(hadoop_home):
                os.environ["HADOOP_HOME"] = hadoop_home
                os.environ["hadoop.home.dir"] = hadoop_home
                bin_dir = os.path.join(hadoop_home, "bin")
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                break

def init_spark():
    setup_env()
    spark = (
        SparkSession.builder.appName("HyperGraphFrames_Runner")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.ui.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# ---------------------------------------------------------------------------
# REST Metric Instrumentation
# ---------------------------------------------------------------------------
def snapshot_stage_ids(ui_url, app_id):
    try:
        url = f"{ui_url}/api/v1/applications/{app_id}/stages"
        stages = requests.get(url, timeout=5).json()
        return {s["stageId"] for s in stages if s.get("status") == "COMPLETE"}
    except Exception:
        return set()

def fetch_stage_metrics(ui_url, app_id, baseline_stages):
    time.sleep(1.5)
    try:
        url = f"{ui_url}/api/v1/applications/{app_id}/stages"
        all_stages = requests.get(url, timeout=5).json()
        completed = {s["stageId"]: s for s in all_stages if s.get("status") == "COMPLETE"}
        new_ids = sorted(set(completed.keys()) - baseline_stages)

        if not new_ids:
            return None

        total_read = sum(completed[sid].get("shuffleReadBytes", 0) for sid in new_ids)
        total_write = sum(completed[sid].get("shuffleWriteBytes", 0) for sid in new_ids)
        disk_spill = sum(completed[sid].get("diskBytesSpilled", 0) for sid in new_ids)
        mem_spill = sum(completed[sid].get("memoryBytesSpilled", 0) for sid in new_ids)

        stage_tasks = []
        for sid in new_ids:
            attempt = completed[sid].get("attemptId", 0)
            t_url = f"{ui_url}/api/v1/applications/{app_id}/stages/{sid}/{attempt}/taskList?length=10000"
            try:
                tasks = requests.get(t_url, timeout=5).json()
                if isinstance(tasks, list) and tasks:
                    durations = sorted([t.get("duration", 0) for t in tasks])
                    n = len(durations)
                    mean_d = statistics.mean(durations)
                    std_d = statistics.stdev(durations) if n > 1 else 0.0
                    stage_tasks.append({
                        "max_ms": durations[-1],
                        "p95_ms": durations[int(n * 0.95)],
                        "median_ms": durations[n // 2],
                        "cv": round(std_d / mean_d, 4) if mean_d > 0 else 0.0,
                    })
            except Exception:
                pass

        bottleneck = max(stage_tasks, key=lambda x: x["max_ms"]) if stage_tasks else {}
        return {
            "max_task_ms": bottleneck.get("max_ms", 0),
            "p95_task_ms": bottleneck.get("p95_ms", 0),
            "median_task_ms": bottleneck.get("median_ms", 0),
            "task_cv": bottleneck.get("cv", 0.0),
            "shuffle_read_bytes": total_read,
            "shuffle_write_bytes": total_write,
            "disk_spill_bytes": disk_spill,
            "mem_spill_bytes": mem_spill,
        }
    except Exception as e:
        print(f"Metrics collection warning: {e}")
        return None

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    # Resolve parquet path whether run from root or subdirectory
    parquet_path = POKEC_PARQUET
    if not os.path.exists(parquet_path):
        alt_path = os.path.join("..", POKEC_PARQUET)
        if os.path.exists(alt_path):
            parquet_path = alt_path
        else:
            print(f"Error: Dataset '{POKEC_PARQUET}' not found.")
            sys.exit(1)

    spark = init_spark()
    app_id = spark.sparkContext.applicationId
    ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
    print(f"Spark UI initialized at: {ui_url} (App ID: {app_id})\n")

    print("[*] Loading and caching dataset...")
    edges = spark.read.parquet(parquet_path).select(
        col("src").cast("int"), col("dst").cast("int")
    ).repartition(SHUFFLE_PARTITIONS).cache()

    total_edges = edges.count()
    print(f"    Loaded {total_edges:,} edges.")

    # Create distinct vertices DataFrame
    vertices = (
        edges.select(col("src").alias("id"))
        .union(edges.select(col("dst").alias("id")))
        .distinct()
    )

    # 1. Instantiate native GraphFrame
    g = GraphFrame(vertices, edges)

    # 2. Wrap with HyperGraphFrame middleware library
    print(f"\n[*] Wrapping GraphFrame with HyperGraphFrame (v{__version__})...")
    hg = HyperGraphFrame(g, work_threshold=WORK_THRESHOLD, p_cells=P_CELLS)

    print("\n" + "=" * 70)
    print("EXECUTING TOPOLOGY-AWARE 2D HYPERCUBE 2-HOP MOTIF VIA HYPERGRAPHFRAMES")
    print("=" * 70)

    # Snapshot UI before run to isolate this action's metrics
    baseline_stages = snapshot_stage_ids(ui_url, app_id)
    t0 = time.time()

    # Execute query via clean wrapper API
    motif_query = "(a)-[e1]->(b); (b)-[e2]->(c)"
    motifs_df = hg.find(motif_query)
    path_count = motifs_df.count()

    wall_clock = time.time() - t0
    metrics = fetch_stage_metrics(ui_url, app_id, baseline_stages) or {}

    # -----------------------------------------------------------------------
    # Print Benchmark Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EMPIRICAL BENCHMARK METRICS (HyperGraphFrames Middleware)")
    print("=" * 70)
    print(f"Total 2-Hop Motif Paths Evaluated : {path_count:,}")
    print(f"End-to-End Wall-Clock Latency     : {wall_clock:.3f} s")
    print("-" * 70)
    print("STRAGGLER & WORKLOAD VARIANCE:")
    print(f"  Max Task Duration (max_task_ms) : {metrics.get('max_task_ms', 0):,} ms")
    print(f"  P95 Task Duration (p95_task_ms) : {metrics.get('p95_task_ms', 0):,} ms")
    print(f"  Median Task Duration            : {metrics.get('median_task_ms', 0):,} ms")
    print(f"  Task Duration Coeff. of Var (CV): {metrics.get('task_cv', 0):.4f}")
    print("-" * 70)
    print("INTERNAL BUFFER SPILL & NETWORK PROFILE:")
    disk_mb = metrics.get('disk_spill_bytes', 0) / (1024 * 1024)
    mem_mb = metrics.get('mem_spill_bytes', 0) / (1024 * 1024)
    shuff_read_mb = metrics.get('shuffle_read_bytes', 0) / (1024 * 1024)
    shuff_write_mb = metrics.get('shuffle_write_bytes', 0) / (1024 * 1024)
    print(f"  Disk Spill (disk_spill_bytes)   : {disk_mb:.2f} MB")
    print(f"  Memory Spill (mem_spill_bytes)  : {mem_mb:.2f} MB")
    print(f"  Total Shuffle Read              : {shuff_read_mb:.2f} MB")
    print(f"  Total Shuffle Write             : {shuff_write_mb:.2f} MB")
    print("=" * 70)

    spark.stop()

if __name__ == "__main__":
    main()
