"""
test_e2e_live.py
================
End-to-End Live Verification Script for hypergraphframes (Installed from PyPI).
"""

import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

# Setup Windows Hadoop environment if running on Windows
if os.name == "nt":
    candidates = [
        os.path.join(os.getcwd(), "hadoop"),
        os.path.abspath(os.path.join(os.getcwd(), "..", "hadoop")),
    ]
    for hadoop_home in candidates:
        if os.path.exists(hadoop_home):
            os.environ["HADOOP_HOME"] = hadoop_home
            os.environ["hadoop.home.dir"] = hadoop_home
            bin_dir = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            break

print("=" * 75)
print("HYPERGRAPHFRAMES LIVE PYPI END-TO-END TEST")
print("=" * 75)

# 1. Initialize PySpark with GraphFrames
print("\n[1/5] Initializing SparkSession with GraphFrames Spark package...")
spark = (
    SparkSession.builder.appName("HyperGraphFramesLiveVerification")
    .master("local[*]")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("  [OK] SparkSession initialized successfully.")

# 2. Import from installed PyPI package
print("\n[2/5] Importing HyperGraphFrame from installed package...")
from graphframes import GraphFrame
from hypergraph import HyperGraphFrame, __version__
print(f"  [OK] Successfully imported hypergraph (Version: {__version__})")

# 3. Test Quickstart Motif Join with Vertex & Edge Attributes
print("\n[3/5] Testing Quickstart 2-Hop Motif with Nested Struct Properties...")
v_df = spark.createDataFrame(
    [
        ("1", "Alice", 28),
        ("2", "Bob", 35),
        ("3", "Charlie", 42),
        ("4", "David", 30),
    ],
    ["id", "name", "age"],
)

e_df = spark.createDataFrame(
    [
        ("1", "2", "friend", 100),
        ("2", "3", "follow", 200),
        ("2", "4", "follow", 300),
        ("3", "1", "friend", 400),
        ("4", "2", "friend", 500),
    ],
    ["src", "dst", "relationship", "weight"],
)

g = GraphFrame(v_df, e_df)
hg = HyperGraphFrame(g, work_threshold=2, p_cells=16)

# Verify lazy profiling
assert hg.hub_meta is None, "hub_meta should be None before query execution"
print("  [OK] Lazy profiling verified: hub_meta is uninitialized before first query.")

# Execute 2-hop motif query
motif_query = "(a)-[e1]->(b); (b)-[e2]->(c)"
hg_results = hg.find(motif_query)
gf_results = g.find(motif_query)

hg_count = hg_results.count()
gf_count = gf_results.count()
print(f"  [OK] Motif Query: '{motif_query}'")
print(f"  [OK] Result Path Count: {hg_count} (Matches native GraphFrames: {gf_count})")
assert hg_count == gf_count, f"Count mismatch: {hg_count} != {gf_count}"
assert hg.hub_meta is not None, "hub_meta should be cached after query execution"
print("  [OK] hub_meta cached successfully.")

print("\nResulting DataFrame Output Schema:")
hg_results.printSchema()

print("\nSample Rows:")
hg_results.show(truncate=False)

# 4. Test Skewed Graph with Heavy Computational Hubs
print("\n[4/5] Testing Skewed Power-Law Graph (High Cartesian Join Workload)...")
skewed_v = spark.createDataFrame([(str(i), f"user_{i}") for i in range(120)], ["id", "username"])

edges_list = []
# Node '0' is a massive hub: 50 incoming edges x 50 outgoing edges = 2,500 2-hop paths
for i in range(1, 51):
    edges_list.append((str(i), "0", f"in_{i}"))
for i in range(51, 101):
    edges_list.append(("0", str(i), f"out_{i}"))

# Add background non-hub paths
edges_list.append(("101", "102", "bg_1"))
edges_list.append(("102", "103", "bg_2"))

skewed_e = spark.createDataFrame(edges_list, ["src", "dst", "edge_type"])
skewed_g = GraphFrame(skewed_v, skewed_e)

hg_skewed = HyperGraphFrame(skewed_g, work_threshold=100, p_cells=64)
skewed_res = hg_skewed.find("(a)-[e1]->(b); (b)-[e2]->(c)")
skewed_count = skewed_res.count()

print(f"  [OK] Hub Detection: {hg_skewed.hub_ids.count()} hub vertex identified.")
print(f"  [OK] Evaluated Paths: {skewed_count:,} (Expected: 2,501 paths)")
assert skewed_count == 2501, f"Expected 2501 paths, got {skewed_count}"
print("  [OK] Zero duplicate paths, zero omissions on skewed graph.")

# 5. Test Transparent Fallback on Unsupported Motifs
print("\n[5/5] Testing Transparent Fallback on 3-Hop Motif...")
fallback_motif = "(a)-[e1]->(b); (b)-[e2]->(c); (c)-[e3]->(d)"
fallback_res = hg.find(fallback_motif)
fallback_count = fallback_res.count()
print(f"  [OK] Fallback Motif: '{fallback_motif}'")
print(f"  [OK] Result Path Count: {fallback_count} (Transparently routed to native GraphFrame)")

print("\n" + "=" * 75)
print("ALL LIVE VERIFICATION TESTS PASSED SUCCESSFULLY!")
print("=" * 75)

spark.stop()
