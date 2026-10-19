#!/usr/bin/env python3
"""
datagen/pokec.py
================
Validate and prepare the SNAP soc-Pokec graph as the Phase 1 real-graph case.

Reuses the existing preprocessing.py transformation (tab-separated text -> int src, int dst ->
Parquet) and adds what Phase 1 needs on top:
  1. Raw input checks: file hash, row count, unparseable rows, nulls, self-loops, duplicate
     edges, id range, distinct vertices, compared with SNAP's published 1,632,803 nodes /
     30,622,564 edges.
  2. The same edge policy as the synthetic graphs (directed, no self-loops, no duplicates),
     with every dropped row counted. If the raw file needs no cleaning, the output is
     identical to what preprocessing.py produces.
  3. The same measured-statistics validator as the synthetic graphs, plus an independent
     Spark re-read of the output, plus the exact two-hop workload work_sum.
  4. An HDFS check that only reports HDFS as available after a real hdfs:// read succeeds.

If datasets/pokec/pokec_graph.json already records a PASS for the same raw-file hash and the
output files still match, nothing is recomputed (no re-download, no re-preprocessing).

  python -m datagen.pokec --raw datasets/pokec/raw/soc-pokec-relationships.txt.gz
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datagen.generate import file_sha256  # noqa: E402
from datagen.spark_validate import make_spark  # noqa: E402
from datagen.validate import degree_stats, spark_load_check, structural_checks  # noqa: E402

POKEC_DIR = os.path.join(REPO_ROOT, "datasets", "pokec")
OUT_DIR = os.path.join(POKEC_DIR, "pokec_edges_parquet")
META = os.path.join(POKEC_DIR, "pokec_graph.json")
RECEIPT = os.path.join(POKEC_DIR, "pokec_prep_receipt.json")   # raw-scan report + cleaning for OUT_DIR
SNAP_PUBLISHED = {"nodes": 1_632_803, "edges": 30_622_564,
                  "source_url": "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
                  "dataset_page": "https://snap.stanford.edu/data/soc-Pokec.html"}
# The repo's pre-existing Pokec result (pokec_benchmark_results.json, big-data-course repo) reports
# total_paths = 1,780,294,899 for this graph; recorded here only as a cross-check.
PRIOR_REPORTED_WORK_SUM = 1_780_294_899


def dir_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(path, "*.parquet"))):
        h.update(os.path.basename(f).encode())
        h.update(file_sha256(f).encode())
    return h.hexdigest()


def hdfs_check(spark) -> dict:
    """Report HDFS only if an hdfs:// URI can actually be listed/read. Env vars alone do not count."""
    res = {
        "hadoop_home": os.environ.get("HADOOP_HOME"),
        "hadoop_conf_dir": os.environ.get("HADOOP_CONF_DIR"),
        "hdfs_cli": shutil.which("hdfs"),
        "fs_default_fs": spark.sparkContext._jsc.hadoopConfiguration().get("fs.defaultFS"),
        "real_hdfs_read": False,
    }
    target = os.environ.get("PHASE1_HDFS_URI")
    if not target and str(res["fs_default_fs"]).startswith("hdfs://"):
        target = res["fs_default_fs"].rstrip("/") + "/"
    if not target:
        res["status"] = "NOT_CONFIGURED"
        res["note"] = ("No hdfs:// default filesystem and no PHASE1_HDFS_URI set; Parquet is stored on the "
                       "local filesystem. winutils/HADOOP_HOME on Windows only provide local-FS support.")
        return res
    res["uri"] = target
    try:
        jvm = spark.sparkContext._jvm
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        p = jvm.org.apache.hadoop.fs.Path(target)
        fs = p.getFileSystem(conf)
        listing = fs.listStatus(p)
        res["real_hdfs_read"] = True
        res["entries_listed"] = len(listing)
        res["status"] = "OK"
    except Exception as e:
        res["status"] = "FAILED"
        res["error"] = str(e)[:500]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(POKEC_DIR, "raw", "soc-pokec-relationships.txt.gz"))
    ap.add_argument("--expected-raw-sha256", default=None,
                    help="Optional: hash recorded when the file was downloaded")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        raise SystemExit(f"Raw Pokec file not found: {args.raw}")
    t_all = time.perf_counter()
    raw_sha = file_sha256(args.raw)
    if args.expected_raw_sha256 and raw_sha != args.expected_raw_sha256:
        raise SystemExit(f"Raw file hash {raw_sha} != expected {args.expected_raw_sha256}")

    if os.path.exists(META) and not args.force:
        with open(META) as f:
            old = json.load(f)
        if (old.get("validation_status") == "PASS" and old["input"]["sha256"] == raw_sha
                and os.path.isdir(OUT_DIR) and dir_fingerprint(OUT_DIR) == old["output"]["fingerprint"]):
            print("Valid local Pokec copy already prepared for this raw file; nothing to do.")
            return

    from pyspark.sql import functions as F

    receipt = None
    if os.path.exists(RECEIPT) and os.path.isdir(OUT_DIR) and not args.force:
        with open(RECEIPT) as f:
            receipt = json.load(f)
        if receipt["input"]["sha256"] != raw_sha:
            receipt = None

    spark = make_spark("Phase1PokecPrep")          # one session for the whole script
    try:
        if receipt is not None:
            # A prepared output for this exact raw file exists: reuse it, do not preprocess again.
            print(f"Reusing {os.path.relpath(OUT_DIR, REPO_ROOT)} (receipt for raw sha256 {raw_sha[:12]}...)")
            raw_report, cleaning, write_s = receipt["input"], receipt["cleaning"], receipt.get("write_seconds")
        else:
            # ---- 1. Raw input, read the same way as preprocessing.py (tab separated, no header).
            raw = spark.read.csv(args.raw, sep="\t")
            n_cols = len(raw.columns)
            parsed = raw.select(F.col("_c0").alias("src_raw"), F.col("_c1").alias("dst_raw"),
                                F.col("_c0").cast("int").alias("src"), F.col("_c1").cast("int").alias("dst")).cache()
            r = parsed.agg(
                F.count(F.lit(1)).alias("rows"),
                F.sum(F.when(F.col("src_raw").isNull() | F.col("dst_raw").isNull(), 1).otherwise(0)).alias("null_fields"),
                F.sum(F.when((F.col("src").isNull() & F.col("src_raw").isNotNull()) |
                             (F.col("dst").isNull() & F.col("dst_raw").isNotNull()), 1).otherwise(0)).alias("unparseable"),
                F.sum(F.when(F.col("src") == F.col("dst"), 1).otherwise(0)).alias("self_loops"),
                F.min(F.least("src", "dst")).alias("min_id"),
                F.max(F.greatest("src", "dst")).alias("max_id"),
            ).collect()[0]
            valid = parsed.where(F.col("src").isNotNull() & F.col("dst").isNotNull()).select("src", "dst")
            valid_rows = valid.count()
            distinct_pairs = valid.distinct().count()
            distinct_vertices = (valid.select(F.col("src").alias("id")).union(valid.select(F.col("dst").alias("id")))
                                 .distinct().count())
            raw_report = {
                "path": os.path.relpath(args.raw, REPO_ROOT).replace(os.sep, "/"),
                "sha256": raw_sha, "bytes": os.path.getsize(args.raw), "columns_detected": n_cols,
                "rows": int(r["rows"]), "null_fields": int(r["null_fields"] or 0),
                "unparseable_ids": int(r["unparseable"] or 0), "self_loops": int(r["self_loops"] or 0),
                "duplicate_edges": int(valid_rows - distinct_pairs), "min_id": r["min_id"], "max_id": r["max_id"],
                "distinct_vertices": int(distinct_vertices),
                "matches_snap_published": {"edges": int(r["rows"]) == SNAP_PUBLISHED["edges"],
                                           "nodes": int(distinct_vertices) == SNAP_PUBLISHED["nodes"]},
            }
            print("raw:", json.dumps(raw_report), flush=True)

            # ---- 2. Same edge policy as synthetic graphs.
            cleaned = (valid.where(F.col("src") != F.col("dst")).dropDuplicates(["src", "dst"])
                       .orderBy("src", "dst"))
            tmp = OUT_DIR + ".tmp"
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            t0 = time.perf_counter()
            cleaned.write.mode("error").parquet(tmp)
            write_s = round(time.perf_counter() - t0, 2)
            if os.path.exists(OUT_DIR):
                stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                os.rename(OUT_DIR, f"{OUT_DIR}.superseded-{stamp}")   # keep, never delete an earlier output
            os.rename(tmp, OUT_DIR)
            cleaning = {
                "policy": "directed; drop rows with null/unparseable ids; drop self-loops; drop duplicate (src,dst)",
                "rows_dropped_null_or_unparseable": int(r["rows"]) - int(valid_rows),
                "rows_dropped_self_loops": raw_report["self_loops"],
                "rows_dropped_duplicates": raw_report["duplicate_edges"],
                "identical_to_preprocessing_py_output": (int(r["rows"]) == int(valid_rows)
                                                        and raw_report["self_loops"] == 0
                                                        and raw_report["duplicate_edges"] == 0),
            }
            parsed.unpersist()
            with open(RECEIPT, "w") as f:
                json.dump({"input": raw_report, "cleaning": cleaning, "write_seconds": write_s,
                           "output_fingerprint": dir_fingerprint(OUT_DIR),
                           "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)

        # ---- 3. Same numpy validator as the synthetic graphs (on the written file).
        import pyarrow.parquet as pq
        t = pq.read_table(OUT_DIR, columns=["src", "dst"])
        src = t.column("src").to_numpy().astype("int32")
        dst = t.column("dst").to_numpy().astype("int32")
        del t
        n_universe = int(max(src.max(), dst.max())) + 1     # ids are used directly as array indices
        stats = degree_stats(src, dst, n_universe)
        structure = structural_checks(src, dst, n_universe)
        del src, dst

        # ---- Independent Spark re-read of the written file.
        expected = dict(stats, self_loops=0, duplicate_edges=0)
        reread = spark_load_check(spark, OUT_DIR, expected)

        # ---- 4. HDFS (real read or nothing).
        hdfs = hdfs_check(spark)
    finally:
        spark.stop()

    expected_rows = raw_report["rows"] - cleaning["rows_dropped_null_or_unparseable"] \
        - cleaning["rows_dropped_self_loops"] - cleaning["rows_dropped_duplicates"]
    checks = [
        ("raw edges == SNAP published 30,622,564", raw_report["matches_snap_published"]["edges"]),
        ("distinct vertices == SNAP published 1,632,803", raw_report["matches_snap_published"]["nodes"]),
        ("no null or unparseable ids", raw_report["null_fields"] == 0 and raw_report["unparseable_ids"] == 0),
        ("output: no self-loops", structure["self_loops"] == 0),
        ("output: no duplicate edges", structure["duplicate_edges"] == 0),
        ("output ids non-negative", structure["min_id"] >= 0),
        ("Spark re-read matches numpy validator", reread["status"] == "PASS"),
        ("output rows == raw rows minus recorded drops", stats["num_edges"] == expected_rows),
        ("active vertices == raw distinct vertices", stats["num_active_vertices"] == raw_report["distinct_vertices"]),
    ]
    status = "PASS" if all(ok for _, ok in checks) else "FAIL"
    record = {
        "graph_id": f"pokec-real-{raw_sha[:8]}",
        "source": SNAP_PUBLISHED,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": raw_report,
        "cleaning": cleaning,
        "output": {"path": os.path.relpath(OUT_DIR, REPO_ROOT).replace(os.sep, "/"), "format": "parquet (Spark directory)",
                   "schema": {"src": "int32", "dst": "int32"}, "fingerprint": dir_fingerprint(OUT_DIR),
                   "bytes": sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT_DIR, "*.parquet"))),
                   "write_seconds": write_s, "storage": "local filesystem",
                   "prep_receipt": os.path.relpath(RECEIPT, REPO_ROOT).replace(os.sep, "/")},
        "id_note": "SNAP ids are 1..1,632,803; id 0 is unused. num_vertices below is the id universe (max_id+1); "
                   "num_active_vertices is the real vertex count.",
        "measured": stats,
        "structure": structure,
        "spark_reread": reread,
        "workload": {"work_sum": stats["work_sum"],
                     "prior_reported_total_paths": PRIOR_REPORTED_WORK_SUM,
                     "matches_prior_report": stats["work_sum"] == PRIOR_REPORTED_WORK_SUM},
        "hdfs": hdfs,
        "checks": [{"check": c, "passed": bool(ok)} for c, ok in checks],
        "validation_status": status,
        "seconds_total": round(time.perf_counter() - t_all, 1),
    }
    with open(META, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(json.dumps({k: record[k] for k in ("graph_id", "validation_status", "checks", "workload", "cleaning")},
                     indent=2, default=str))
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
