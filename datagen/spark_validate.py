#!/usr/bin/env python3
"""
datagen/spark_validate.py
=========================
Re-read every generated edge file through Spark (the same reader the benchmark
uses) and check it against the generator's own measurements: row count, nulls,
id range, self-loops, duplicates, active vertices, work_sum and max work.

Results are written into each graph.json under "spark_validation" and the
manifest is rebuilt.

  python -m datagen.spark_validate              # all graphs in datasets/synthetic
  python -m datagen.spark_validate --graph-id uniform-small-s101-f3568c8f
"""

import argparse
import datetime
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "bench"), os.path.join(REPO_ROOT, "hypergraphframes")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from datagen.generate import SYNTH_DIR, rebuild_manifest  # noqa: E402
from datagen.validate import spark_load_check  # noqa: E402


def make_spark(app_name: str = "Phase1SparkValidate"):
    import bench  # reuse the harness's environment setup (winutils, JDK, PYSPARK_PYTHON)
    from pyspark.sql import SparkSession

    bench.setup_env()
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", os.environ.get("PHASE1_DRIVER_MEMORY", "4g"))
        .config("spark.driver.extraJavaOptions", bench.JAVA_ADD_OPENS)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-id", action="append", help="Validate only these graph ids")
    ap.add_argument("--force", action="store_true", help="Re-run even if a spark_validation already exists")
    args = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(SYNTH_DIR, "*", "*", "graph.json")))
    spark = make_spark()
    failures = 0
    try:
        for mp in metas:
            with open(mp) as f:
                rec = json.load(f)
            gid = rec["graph_id"]
            if args.graph_id and gid not in args.graph_id:
                continue
            if rec.get("spark_validation") and not args.force:
                print(f"  {gid:<48} {rec['spark_validation']['status']:<6} (already validated)")
                continue
            pq = next(x for x in rec["files"] if x["format"] == "parquet")["path"]
            expected = dict(rec["measured"])
            expected["self_loops"] = 0
            expected["duplicate_edges"] = 0
            t0 = time.perf_counter()
            res = spark_load_check(spark, os.path.join(REPO_ROOT, pq), expected)
            res["seconds"] = round(time.perf_counter() - t0, 2)
            res["validated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            res["spark_version"] = spark.version
            if res["schema"] != {"src": "int", "dst": "int"}:
                res["mismatches"].append({"field": "schema", "spark": res["schema"],
                                          "expected": {"src": "int", "dst": "int"}})
                res["status"] = "FAIL"
            rec["spark_validation"] = res
            with open(mp, "w") as f:
                json.dump(rec, f, indent=2)
            failures += res["status"] != "PASS"
            print(f"  {gid:<48} {res['status']:<6} rows={res['rows']:,} work_sum={res['work_sum']:,} "
                  f"({res['seconds']}s) {res['mismatches'] or ''}")
    finally:
        spark.stop()
    rebuild_manifest()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
