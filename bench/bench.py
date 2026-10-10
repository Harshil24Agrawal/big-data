#!/usr/bin/env python3
"""
bench.py
========
Controlled, single-run benchmark runner for HyperGraphFrames vs 1D Salting vs Native GraphFrames.

Usage:
  python bench/bench.py --variant {native,salt1d,hypercube} --aqe {on,off} \
      --action {noop,count} --run-id N --out bench/results/<name>.json [--edges-limit K]
"""

import argparse
import contextlib
import datetime
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, Any, Set, Optional

import psutil
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast

# Ensure local repo root, bench/, and hypergraphframes/ are importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PKG_ROOT = os.path.join(REPO_ROOT, "hypergraphframes")

for p in [THIS_DIR, PKG_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hypergraph import HyperGraphFrame, __version__ as hg_version
from salting_1d import Salted1DGraphFrame, EdgeJoinGraphFrame

GRAPHFRAMES_JAR_SCALA212 = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"
GRAPHFRAMES_JAR_SCALA213 = "graphframes:graphframes:0.8.3-spark3.5-s_2.13"
DEFAULT_SHUFFLE_PARTITIONS = 200
DEFAULT_WORK_THRESHOLD = 1_000_000
DEFAULT_P_CELLS = 100
MOTIF_QUERY = "(a)-[e1]->(b); (b)-[e2]->(c)"


def resolve_graphframes_package() -> str:
    """Select the matching GraphFrames jar coordinate for the active PySpark version (Scala 2.12 vs 2.13)."""
    import pyspark
    if pyspark.__version__.startswith("3."):
        return GRAPHFRAMES_JAR_SCALA212
    return GRAPHFRAMES_JAR_SCALA213


def get_git_state() -> Dict[str, Any]:
    """Return current git commit hash and dirty-working-tree flag."""
    commit = "unknown"
    dirty = None
    try:
        r1 = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r1.returncode == 0:
            commit = r1.stdout.strip()
        r2 = subprocess.run(
            ["git", "status", "--porcelain", "-uno"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r2.returncode == 0:
            dirty = bool(r2.stdout.strip())
    except Exception:
        pass
    return {"git_commit": commit, "git_dirty_tree": dirty}

JAVA_ADD_OPENS = (
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
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
)


def setup_env() -> Optional[str]:
    """Configure Windows Hadoop winutils path and local Temurin JDK 17 (for Spark 3.5) inside the process only."""
    import pyspark
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if os.name == "nt":
        # If running PySpark 3.x and local Temurin 17 is available, point JAVA_HOME to it in-process
        if pyspark.__version__.startswith("3.") and not os.environ.get("BENCH_FORCE_SYSTEM_JAVA"):
            temurin_root = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Temurin")
            if os.path.exists(temurin_root):
                for entry in sorted(os.listdir(temurin_root), reverse=True):
                    cand_java = os.path.join(temurin_root, entry, "bin", "java.exe")
                    if os.path.exists(cand_java):
                        jdk_home = os.path.join(temurin_root, entry)
                        os.environ["JAVA_HOME"] = jdk_home
                        os.environ["PATH"] = os.path.join(jdk_home, "bin") + os.pathsep + os.environ.get("PATH", "")
                        break

        env_hadoop = os.environ.get("HADOOP_HOME")
        candidates = [
            *([env_hadoop] if env_hadoop else []),
            os.path.join(os.getcwd(), "hadoop"),
            os.path.join(REPO_ROOT, "hadoop"),
        ]
        for hadoop_home in candidates:
            if hadoop_home and os.path.exists(hadoop_home):
                hadoop_abs = os.path.abspath(hadoop_home)
                os.environ["HADOOP_HOME"] = hadoop_abs
                os.environ["hadoop.home.dir"] = hadoop_abs
                bin_dir = os.path.join(hadoop_abs, "bin")
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return hadoop_abs
    return os.environ.get("HADOOP_HOME")


def create_spark(aqe: str, app_name: str = "HyperGraphControlledBench") -> SparkSession:
    """
    Create a fresh SparkSession with identical configuration across all variants,
    varying only spark.sql.adaptive.enabled according to --aqe {on,off}.
    """
    setup_env()
    aqe_bool = "true" if aqe.lower() == "on" else "false"
    gf_pkg = resolve_graphframes_package()
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.extraJavaOptions", JAVA_ADD_OPENS)
        .config("spark.executor.extraJavaOptions", JAVA_ADD_OPENS)
        .config("spark.driver.memory", "6g")
        .config("spark.jars.packages", gf_pkg)
        .config("spark.sql.shuffle.partitions", str(DEFAULT_SHUFFLE_PARTITIONS))
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.adaptive.enabled", aqe_bool)
        .config("spark.ui.enabled", "true")
        .config("spark.ui.retainedTasks", "1000000")
        .config("spark.ui.retainedStages", "10000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_system_and_version_info(spark: SparkSession) -> Dict[str, Any]:
    """Record exact software versions and hardware specs for reproducibility."""
    import pyspark

    java_ver = spark.sparkContext._gateway.jvm.java.lang.System.getProperty("java.version")
    java_vm = spark.sparkContext._gateway.jvm.java.lang.System.getProperty("java.vm.name")
    scala_ver = spark.sparkContext._gateway.jvm.scala.util.Properties.versionNumberString()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(REPO_ROOT)
    git_state = get_git_state()

    try:
        gf_py_ver = importlib.metadata.version("graphframes")
    except Exception:
        gf_py_ver = "unknown"

    try:
        hg_dist_ver = importlib.metadata.version("hypergraphframes")
    except Exception:
        hg_dist_ver = "unknown"

    return {
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "pyspark_version": pyspark.__version__,
        "scala_version": scala_ver,
        "java_version": java_ver,
        "java_vm": java_vm,
        "graphframes_jar": resolve_graphframes_package(),
        "graphframes_python_wrapper": gf_py_ver,
        "hypergraphframes_version": hg_version,
        "hypergraphframes_dist_version": hg_dist_ver,
        "git_commit": git_state["git_commit"],
        "git_dirty_tree": git_state["git_dirty_tree"],
        "os": platform.platform(),
        "cpu_model": "AMD Ryzen AI 7 350 w/ Radeon 860M",
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "disk_model": "MTFDKBA512QGN-1BN1AABGA (NVMe SSD)",
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
    }


def parse_spark_iso_time(ts_str: Optional[str]) -> Optional[float]:
    """Parse Spark REST API timestamp (e.g. '2026-10-10T12:38:32.707GMT') to epoch seconds."""
    if not ts_str:
        return None
    try:
        clean = ts_str.replace("GMT", "+00:00").replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean)
        return dt.timestamp()
    except Exception:
        return None


def poll_completed_stages(
    ui_url: str,
    app_id: str,
    poll_interval: float = 0.25,
    timeout: float = 30.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Poll the Spark REST API until no stages are ACTIVE/PENDING and the set of
    COMPLETE stages (and their task counts / runtimes) is identical for 2 consecutive polls.
    """
    url = f"{ui_url}/api/v1/applications/{app_id}/stages"
    prev_sig = None
    deadline = time.perf_counter() + timeout
    last_completed: Dict[int, Dict[str, Any]] = {}

    while time.perf_counter() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            all_stages = resp.json()

            active = [
                s for s in all_stages
                if s.get("status") in ("ACTIVE", "PENDING")
            ]
            completed = {
                s["stageId"]: s
                for s in all_stages
                if s.get("status") == "COMPLETE"
            }
            last_completed = completed

            if not active:
                sig = tuple(
                    sorted(
                        (
                            sid,
                            st.get("attemptId", 0),
                            st.get("numCompleteTasks", 0),
                            st.get("executorRunTime", 0),
                            st.get("shuffleReadBytes", 0),
                            st.get("shuffleWriteBytes", 0),
                        )
                        for sid, st in completed.items()
                    )
                )
                if prev_sig is not None and sig == prev_sig:
                    return completed
                prev_sig = sig
            else:
                prev_sig = None
        except Exception:
            prev_sig = None

        time.sleep(poll_interval)

    return last_completed


def snapshot_stage_ids(ui_url: str, app_id: str) -> Set[int]:
    """Return the set of completed stageIds once stable."""
    completed = poll_completed_stages(ui_url, app_id)
    return set(completed.keys())


def collect_window_metrics(
    ui_url: str,
    app_id: str,
    before_ids: Set[int],
    after_completed: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Collect stage-level aggregates and taskSummary quantiles for ALL stages completed
    after before_ids.
    """
    if after_completed is None:
        after_completed = poll_completed_stages(ui_url, app_id)

    new_ids = sorted(set(after_completed.keys()) - before_ids)
    stages_out = []

    for sid in new_ids:
        st = after_completed[sid]
        attempt = st.get("attemptId", 0)
        sub_ts = parse_spark_iso_time(st.get("submissionTime"))
        comp_ts = parse_spark_iso_time(st.get("completionTime"))
        first_ts = parse_spark_iso_time(st.get("firstTaskLaunchedTime"))

        wall_ms = None
        if sub_ts is not None and comp_ts is not None:
            wall_ms = round(max(0.0, (comp_ts - sub_ts) * 1000.0), 2)

        # Fetch taskSummary quantiles: min (0.0), p25 (0.25), median (0.5), p75 (0.75), p95 (0.95), max (1.0)
        ts_url = (
            f"{ui_url}/api/v1/applications/{app_id}/stages/{sid}/{attempt}"
            f"/taskSummary?quantiles=0.0,0.25,0.5,0.75,0.95,1.0"
        )
        min_ms = p25_ms = median_ms = p75_ms = p95_ms = max_ms = 0.0
        try:
            ts_resp = requests.get(ts_url, timeout=10)
            if ts_resp.status_code == 200:
                ts_data = ts_resp.json()
                durations = ts_data.get("duration", [])
                if len(durations) >= 6:
                    min_ms = round(float(durations[0]), 2)
                    p25_ms = round(float(durations[1]), 2)
                    median_ms = round(float(durations[2]), 2)
                    p75_ms = round(float(durations[3]), 2)
                    p95_ms = round(float(durations[4]), 2)
                    max_ms = round(float(durations[5]), 2)
        except Exception:
            pass

        if median_ms > 0:
            max_median_ratio = round(max_ms / median_ms, 4)
        elif max_ms == 0:
            max_median_ratio = 1.0
        else:
            max_median_ratio = None

        stages_out.append({
            "stageId": sid,
            "attemptId": attempt,
            "name": st.get("name", ""),
            "numTasks": st.get("numTasks", 0),
            "numCompleteTasks": st.get("numCompleteTasks", 0),
            "submissionTime": st.get("submissionTime"),
            "firstTaskLaunchedTime": st.get("firstTaskLaunchedTime"),
            "completionTime": st.get("completionTime"),
            "stageWallClockMs": wall_ms,
            "stageWallClockS": round(wall_ms / 1000.0, 3) if wall_ms is not None else None,
            "executorRunTimeMs": st.get("executorRunTime", 0),
            "executorCpuTimeNs": st.get("executorCpuTime", 0),
            "jvmGcTimeMs": st.get("jvmGcTime", 0),
            "shuffleReadBytes": st.get("shuffleReadBytes", 0),
            "shuffleReadRecords": st.get("shuffleReadRecords", 0),
            "shuffleWriteBytes": st.get("shuffleWriteBytes", 0),
            "shuffleWriteRecords": st.get("shuffleWriteRecords", 0),
            "diskBytesSpilled": st.get("diskBytesSpilled", 0),
            "memoryBytesSpilled": st.get("memoryBytesSpilled", 0),
            "peakExecutionMemory": st.get("peakExecutionMemory", 0),
            "taskDurationMs": {
                "min": min_ms,
                "p25": p25_ms,
                "median": median_ms,
                "p75": p75_ms,
                "p95": p95_ms,
                "max": max_ms,
                "max_median_ratio": max_median_ratio,
            },
        })

    total_tasks = sum(s["numTasks"] for s in stages_out)
    total_exec_run_ms = sum(s["executorRunTimeMs"] for s in stages_out)
    total_exec_cpu_ns = sum(s["executorCpuTimeNs"] for s in stages_out)
    total_shuffle_read = sum(s["shuffleReadBytes"] for s in stages_out)
    total_shuffle_write = sum(s["shuffleWriteBytes"] for s in stages_out)
    total_disk_spill = sum(s["diskBytesSpilled"] for s in stages_out)
    total_mem_spill = sum(s["memoryBytesSpilled"] for s in stages_out)

    dominant_stage = (
        max(stages_out, key=lambda s: s["executorRunTimeMs"])
        if stages_out
        else None
    )

    return {
        "num_stages": len(stages_out),
        "total_tasks": total_tasks,
        "total_executor_run_time_ms": total_exec_run_ms,
        "total_executor_run_time_s": round(total_exec_run_ms / 1000.0, 3),
        "total_executor_cpu_time_ns": total_exec_cpu_ns,
        "total_executor_cpu_time_s": round(total_exec_cpu_ns / 1e9, 3),
        "total_shuffle_read_bytes": total_shuffle_read,
        "total_shuffle_write_bytes": total_shuffle_write,
        "total_disk_bytes_spilled": total_disk_spill,
        "total_memory_bytes_spilled": total_mem_spill,
        "dominant_stage": dominant_stage,
        "stages": stages_out,
    }


def capture_explain_formatted(df) -> str:
    """Capture df.explain('formatted') output string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain("formatted")
    return buf.getvalue()


def resolve_parquet_path() -> str:
    candidates = [
        os.path.join(os.getcwd(), "pokec_edges_parquet"),
        os.path.join(REPO_ROOT, "pokec_edges_parquet"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not find 'pokec_edges_parquet' directory.")


def execute_action(df, action: str) -> Optional[int]:
    if action == "count":
        return int(df.count())
    elif action == "noop":
        df.write.format("noop").mode("overwrite").save()
        return None
    else:
        raise ValueError(f"Unsupported action: {action}")


def main():
    parser = argparse.ArgumentParser(description="Controlled benchmark runner for HyperGraphFrames")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["native", "edgejoin", "salt1d", "hypercube", "hypercube_vjoin"],
    )
    parser.add_argument("--regime", default="id_only", choices=["id_only", "with_attrs"],
                        help="id_only: vertices has only 'id' (edgejoin/salt1d/hypercube elide vertex joins; "
                             "hypercube_vjoin keeps default skip_id_only_vertex_join=False); "
                             "with_attrs: vertices has extra attribute column so vertex joins are mandatory for all variants")
    parser.add_argument("--aqe", required=True, choices=["on", "off"])
    parser.add_argument("--action", required=True, choices=["noop", "count"])
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--edges-limit", type=int, default=None, help="Optional limit on edges for smoke testing")
    parser.add_argument("--work-threshold", type=int, default=DEFAULT_WORK_THRESHOLD)
    parser.add_argument("--p-cells", type=int, default=DEFAULT_P_CELLS)
    args = parser.parse_args()

    parquet_path = resolve_parquet_path()
    app_name = f"Bench_{args.variant}_{args.regime}_aqe_{args.aqe}_{args.action}_run{args.run_id}"
    spark = create_spark(aqe=args.aqe, app_name=app_name)

    try:
        app_id = spark.sparkContext.applicationId
        ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
        sys_info = get_system_and_version_info(spark)

        print(f"[{app_name}] Spark UI: {ui_url} | App ID: {app_id}")

        # 1. Load, repartition, cache, and materialize edges BEFORE timer
        raw_edges = spark.read.parquet(parquet_path).select(
            col("src").cast("int"), col("dst").cast("int")
        )
        if args.edges_limit is not None:
            raw_edges = raw_edges.limit(args.edges_limit)

        edges = raw_edges.repartition(DEFAULT_SHUFFLE_PARTITIONS).cache()
        edge_count = int(edges.count())

        # 2. Build, cache, and materialize vertices BEFORE timer for ALL variants
        base_vertices = (
            edges.select(col("src").alias("id"))
            .union(edges.select(col("dst").alias("id")))
            .distinct()
        )
        if args.regime == "with_attrs":
            vertices_cached = base_vertices.withColumn("v_attr", (col("id") % 100).cast("int")).cache()
            skip_id_only_vjoin = False
        else:
            vertices_cached = base_vertices.cache()
            skip_id_only_vjoin = True

        vertex_count = int(vertices_cached.count())
        # Broadcast-hint the cached vertex lookup table uniformly across all variants
        # while keeping spark.sql.autoBroadcastJoinThreshold = -1 for the edge-to-edge join.
        vertices = broadcast(vertices_cached)

        print(f"[{app_name}] Materialized {edge_count:,} edges and {vertex_count:,} vertices (regime={args.regime}).")

        from graphframes import GraphFrame
        g = GraphFrame(vertices, edges)

        profiling_s = 0.0
        query_s = 0.0
        total_s = 0.0
        hub_count = None
        path_count = None
        profiling_metrics = None
        query_metrics = None
        total_metrics = None

        if args.variant == "native":
            before_query_ids = snapshot_stage_ids(ui_url, app_id)

            t0 = time.perf_counter()
            df = g.find(MOTIF_QUERY)
            path_count = execute_action(df, args.action)
            query_s = round(time.perf_counter() - t0, 4)
            total_s = query_s

            after_completed = poll_completed_stages(ui_url, app_id)
            query_metrics = collect_window_metrics(ui_url, app_id, before_query_ids, after_completed)
            total_metrics = query_metrics

        elif args.variant == "edgejoin":
            eg = EdgeJoinGraphFrame(g, skip_id_only_vertex_join=skip_id_only_vjoin)
            before_query_ids = snapshot_stage_ids(ui_url, app_id)

            t0 = time.perf_counter()
            df = eg.find(MOTIF_QUERY)
            path_count = execute_action(df, args.action)
            query_s = round(time.perf_counter() - t0, 4)
            total_s = query_s

            after_completed = poll_completed_stages(ui_url, app_id)
            query_metrics = collect_window_metrics(ui_url, app_id, before_query_ids, after_completed)
            total_metrics = query_metrics

        elif args.variant == "salt1d":
            sg = Salted1DGraphFrame(
                g,
                work_threshold=args.work_threshold,
                p_cells=args.p_cells,
                skip_id_only_vertex_join=skip_id_only_vjoin,
            )
            # Profile before timed query window (also record profiling time/metrics for completeness)
            before_prof_ids = snapshot_stage_ids(ui_url, app_id)
            tp0 = time.perf_counter()
            sg.profile()
            # Also materialize hub_ids so broadcast inside query_s does not pay profiling work
            sg.hub_ids.count()
            salt1d_prof_s = round(time.perf_counter() - tp0, 4)
            hub_count = sg.hub_count

            after_prof_completed = poll_completed_stages(ui_url, app_id)
            profiling_metrics = collect_window_metrics(ui_url, app_id, before_prof_ids, after_prof_completed)
            before_query_ids = set(after_prof_completed.keys())

            t0 = time.perf_counter()
            df = sg.find(MOTIF_QUERY)
            path_count = execute_action(df, args.action)
            query_s = round(time.perf_counter() - t0, 4)
            profiling_s = salt1d_prof_s
            total_s = round(profiling_s + query_s, 4)

            after_query_completed = poll_completed_stages(ui_url, app_id)
            query_metrics = collect_window_metrics(ui_url, app_id, before_query_ids, after_query_completed)
            total_metrics = collect_window_metrics(ui_url, app_id, before_prof_ids, after_query_completed)

        elif args.variant in ("hypercube", "hypercube_vjoin"):
            eff_skip_vjoin = False if args.variant == "hypercube_vjoin" else skip_id_only_vjoin
            hg = HyperGraphFrame(
                g,
                work_threshold=args.work_threshold,
                p_cells=args.p_cells,
                skip_id_only_vertex_join=eff_skip_vjoin,
            )
            before_prof_ids = snapshot_stage_ids(ui_url, app_id)

            # Timed region 1: explicit profiling
            tp0 = time.perf_counter()
            hg.profile()
            profiling_s = round(time.perf_counter() - tp0, 4)
            hub_count = hg.hub_count

            after_prof_completed = poll_completed_stages(ui_url, app_id)
            profiling_metrics = collect_window_metrics(ui_url, app_id, before_prof_ids, after_prof_completed)
            before_query_ids = set(after_prof_completed.keys())

            # Timed region 2: find + action
            t0 = time.perf_counter()
            df = hg.find(MOTIF_QUERY)
            path_count = execute_action(df, args.action)
            query_s = round(time.perf_counter() - t0, 4)
            total_s = round(profiling_s + query_s, 4)

            after_query_completed = poll_completed_stages(ui_url, app_id)
            query_metrics = collect_window_metrics(ui_url, app_id, before_query_ids, after_query_completed)
            total_metrics = collect_window_metrics(ui_url, app_id, before_prof_ids, after_query_completed)

        # Save formatted explain plan outside timed window
        explain_str = capture_explain_formatted(df)
        out_abs = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_abs)
        os.makedirs(out_dir, exist_ok=True)
        plans_dir = os.path.join(out_dir, "plans")
        os.makedirs(plans_dir, exist_ok=True)

        limit_tag = f"_limit{args.edges_limit}" if args.edges_limit is not None else ""
        plan_filename = f"plan_{args.variant}_{args.regime}_aqe_{args.aqe}_{args.action}{limit_tag}.txt"
        plan_path = os.path.join(plans_dir, plan_filename)
        with open(plan_path, "w", encoding="utf-8") as pf:
            pf.write(explain_str)

        eff_skip_recorded = (
            False if args.variant in ("native", "hypercube_vjoin") else skip_id_only_vjoin
        )
        result_payload = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config": {
                "variant": args.variant,
                "regime": args.regime,
                "aqe": args.aqe,
                "action": args.action,
                "run_id": args.run_id,
                "edges_limit": args.edges_limit,
                "motif_query": MOTIF_QUERY,
                "spark_master": "local[*]",
                "driver_memory": "6g",
                "shuffle_partitions": DEFAULT_SHUFFLE_PARTITIONS,
                "auto_broadcast_join_threshold": -1,
                "retained_tasks": 1000000,
            },
            "variant_parameters": {
                "work_threshold": args.work_threshold if args.variant in ("salt1d", "hypercube", "hypercube_vjoin") else None,
                "p_cells": args.p_cells if args.variant in ("hypercube", "hypercube_vjoin") else None,
                "salt_buckets": args.p_cells if args.variant == "salt1d" else None,
                "skip_id_only_vertex_join": eff_skip_recorded,
                "hubs_identified": hub_count,
            },
            "dataset": {
                "name": "soc-pokec",
                "parquet_path": parquet_path,
                "edges_count": edge_count,
                "vertices_count": vertex_count,
                "path_count": path_count,
            },
            "environment": sys_info,
            "timings": {
                "profiling_s": profiling_s,
                "query_s": query_s,
                "total_s": total_s,
            },
            "metrics_query_only": query_metrics,
            "metrics_incl_profiling": total_metrics,
            "metrics_profiling_only": profiling_metrics,
            "plan_file": os.path.relpath(plan_path, REPO_ROOT),
        }

        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        dom = (query_metrics or {}).get("dominant_stage") or {}
        dom_td = dom.get("taskDurationMs") or {}
        print(
            f"[{app_name}] DONE | profiling_s={profiling_s:.3f}s | query_s={query_s:.3f}s | "
            f"total_s={total_s:.3f}s | paths={path_count} | "
            f"dom_stage={dom.get('stageId')} (max/med={dom_td.get('max_median_ratio')}) | "
            f"mem_spill={query_metrics.get('total_memory_bytes_spilled', 0) / (1024**2):.1f}MB | "
            f"disk_spill={query_metrics.get('total_disk_bytes_spilled', 0) / (1024**2):.1f}MB"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
