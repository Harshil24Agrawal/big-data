"""
test_equivalence.py
===================
Comprehensive row-set equivalence test suite comparing native GraphFrame.find()
against HyperGraphFrame.find() across power-law graphs and edge cases.

Runnable via:
    python -m unittest discover -s tests
"""

import os
import random
import sys
import unittest
from typing import List, Tuple, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    BooleanType,
)

# Ensure hypergraph package is in sys.path
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from hypergraph import HyperGraphFrame


def setup_test_env():
    """Configure Windows Hadoop winutils if present."""
    if os.name == "nt":
        env_hadoop = os.environ.get("HADOOP_HOME")
        candidates = [
            *([env_hadoop] if env_hadoop else []),
            os.path.join(os.getcwd(), "hadoop"),
            os.path.abspath(os.path.join(os.getcwd(), "..", "hadoop")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "hadoop")),
        ]
        for hadoop_home in candidates:
            if hadoop_home and os.path.exists(hadoop_home):
                os.environ["HADOOP_HOME"] = hadoop_home
                os.environ["hadoop.home.dir"] = hadoop_home
                bin_dir = os.path.join(hadoop_home, "bin")
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                break


def generate_seeded_powerlaw_graph(
    num_vertices: int = 400,
    planted_hubs: List[Tuple[int, int, int]] = None,
    background_edges: int = 800,
    seed: int = 42,
):
    """
    Generate a deterministic power-law-ish directed graph with planted hubs.
    planted_hubs: list of (hub_id, in_degree, out_degree)
    Returns (vertex_rows, edge_rows).
    """
    rng = random.Random(seed)
    if planted_hubs is None:
        planted_hubs = [(0, 45, 45), (1, 60, 15), (2, 15, 60)]

    v_rows = []
    for vid in range(num_vertices):
        name = f"v_{vid}" if vid % 7 != 0 else None
        score = round(rng.uniform(0.0, 100.0), 2) if vid % 5 != 0 else None
        v_rows.append((vid, name, score))

    e_rows = []
    edge_idx = 0
    for hub_id, d_in, d_out in planted_hubs:
        in_candidates = [v for v in range(num_vertices) if v != hub_id]
        out_candidates = [v for v in range(num_vertices) if v != hub_id]
        in_srcs = [rng.choice(in_candidates) for _ in range(d_in)]
        out_dsts = [rng.choice(out_candidates) for _ in range(d_out)]
        for u in in_srcs:
            e_rows.append((u, hub_id, f"hub_in_{edge_idx}", edge_idx % 10))
            edge_idx += 1
        for w in out_dsts:
            e_rows.append((hub_id, w, f"hub_out_{edge_idx}", edge_idx % 10))
            edge_idx += 1

    # Add Zipf-weighted background edges
    weights = [1.0 / ((i + 1) ** 0.8) for i in range(num_vertices)]
    nodes = list(range(num_vertices))
    for _ in range(background_edges):
        u = rng.choices(nodes, weights=weights, k=1)[0]
        v = rng.choices(nodes, weights=weights, k=1)[0]
        rel = f"bg_{edge_idx}" if edge_idx % 6 != 0 else None
        e_rows.append((u, v, rel, edge_idx % 20))
        edge_idx += 1

    return v_rows, e_rows


class TestRowSetEquivalence(unittest.TestCase):
    MOTIF_2HOP = "(a)-[e1]->(b); (b)-[e2]->(c)"

    @classmethod
    def setUpClass(cls):
        setup_test_env()
        cls.spark = (
            SparkSession.builder.appName("HyperGraphEquivalenceSuite")
            .master("local[2]")
            .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.13")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def assert_ExactEquivalence(self, expected_df: DataFrame, actual_df: DataFrame):
        """
        Verify exact StructType schema, column order, row count, and bidirectional
        multiset equality via exceptAll on all columns.
        """
        self.assertEqual(
            actual_df.columns,
            expected_df.columns,
            f"Column ordering mismatch: {actual_df.columns} != {expected_df.columns}",
        )
        self.assertEqual(
            actual_df.schema,
            expected_df.schema,
            f"Schema mismatch:\nActual:   {actual_df.schema}\nExpected: {expected_df.schema}",
        )

        expected_count = expected_df.count()
        actual_count = actual_df.count()
        self.assertEqual(
            actual_count,
            expected_count,
            f"Row count mismatch: actual={actual_count} vs expected={expected_count}",
        )

        missing_from_actual = expected_df.exceptAll(actual_df).count()
        extra_in_actual = actual_df.exceptAll(expected_df).count()
        self.assertEqual(
            missing_from_actual,
            0,
            f"Found {missing_from_actual} rows in native GraphFrame missing from HyperGraphFrame",
        )
        self.assertEqual(
            extra_in_actual,
            0,
            f"Found {extra_in_actual} extra rows in HyperGraphFrame not in native GraphFrame",
        )

    def test_01_seeded_powerlaw_with_planted_hubs(self):
        """Seeded 400-vertex power-law graph with planted hubs (both attributed and id-only vertices)."""
        from graphframes import GraphFrame

        v_rows, e_rows = generate_seeded_powerlaw_graph(
            num_vertices=400,
            planted_hubs=[(0, 35, 35), (1, 50, 12), (2, 12, 50)],
            background_edges=600,
            seed=42,
        )
        v_schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("score", DoubleType(), True),
        ])
        e_schema = StructType([
            StructField("src", IntegerType(), True),
            StructField("dst", IntegerType(), True),
            StructField("rel", StringType(), True),
            StructField("weight", IntegerType(), True),
        ])
        v_df = self.spark.createDataFrame(v_rows, schema=v_schema)
        e_df = self.spark.createDataFrame(e_rows, schema=e_schema)

        # 1a. With vertex and edge attributes
        g_attr = GraphFrame(v_df, e_df)
        hg_attr = HyperGraphFrame(g_attr, work_threshold=200, p_cells=12)
        actual_attr = hg_attr.find(self.MOTIF_2HOP)
        expected_attr = g_attr.find(self.MOTIF_2HOP)

        self.assertIsNotNone(hg_attr.hub_meta)
        self.assertGreaterEqual(hg_attr.hub_count, 3, "Planted hubs must be classified as heavy hubs")
        self.assert_ExactEquivalence(expected_attr, actual_attr)

        # 1b. With id-only vertices (exercises planner's id-only projection optimization)
        v_id_only = v_df.select("id")
        g_id = GraphFrame(v_id_only, e_df.select("src", "dst"))
        hg_id = HyperGraphFrame(g_id, work_threshold=200, p_cells=12)
        self.assert_ExactEquivalence(g_id.find(self.MOTIF_2HOP), hg_id.find(self.MOTIF_2HOP))
        self.assertGreaterEqual(hg_id.hub_count, 3)

    def test_02_heavy_hub_symmetric(self):
        """Symmetric heavy hub (d_in == d_out = 30, W = 900 > threshold=100, p_cells=16)."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"u{i}") for i in range(100)], ["id", "label"])
        e_rows = [(i, 0, f"in_{i}") for i in range(1, 31)] + [(0, i, f"out_{i}") for i in range(31, 61)]
        e_rows += [(70, 71, "bg1"), (71, 72, "bg2")]
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "tag"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=16)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 1)
        hub_row = hg.hub_meta.collect()[0]
        self.assertEqual(hub_row["id"], 0)
        self.assertEqual(hub_row["r_final"], 4)
        self.assertEqual(hub_row["s_final"], 4)
        self.assert_ExactEquivalence(expected, actual)

    def test_03_heavy_in_side_only(self):
        """Hub heavy on in-side (d_in=60, d_out=3, W=180 > 100) plus pure sink (d_in=80, d_out=0, W=0)."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"n{i}") for i in range(120)], ["id", "name"])
        # Hub 0: 60 in, 3 out -> W = 180
        e_rows = [(i, 0, i) for i in range(1, 61)] + [(0, 100 + j, j) for j in range(3)]
        # Sink 99: 80 in, 0 out -> W = 0 (not a hub despite high in-degree)
        e_rows += [(i, 99, i) for i in range(1, 81)]
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "w"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=9)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 1)
        hub_ids = [r["id"] for r in hg.hub_ids.collect()]
        self.assertEqual(hub_ids, [0])
        self.assert_ExactEquivalence(expected, actual)

    def test_04_heavy_out_side_only(self):
        """Hub heavy on out-side (d_in=3, d_out=60, W=180 > 100) plus pure source (d_in=0, d_out=80, W=0)."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"n{i}") for i in range(120)], ["id", "name"])
        # Hub 0: 3 in, 60 out -> W = 180
        e_rows = [(100 + j, 0, j) for j in range(3)] + [(0, i, i) for i in range(1, 61)]
        # Source 99: 0 in, 80 out -> W = 0
        e_rows += [(99, i, i) for i in range(1, 81)]
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "w"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=9)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 1)
        hub_ids = [r["id"] for r in hg.hub_ids.collect()]
        self.assertEqual(hub_ids, [0])
        self.assert_ExactEquivalence(expected, actual)

    def test_05_multiple_interconnected_hubs(self):
        """Multiple hubs (symmetric, in-heavy, out-heavy) with hub-to-hub edges and non-hub paths."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(150)], ["id", "label"])
        e_rows = []
        # Hub 0 (20x20), Hub 1 (40x5), Hub 2 (5x40)
        for i in range(10, 30):
            e_rows.append((i, 0, "to_h0"))
            e_rows.append((0, i + 20, "from_h0"))
        for i in range(30, 70):
            e_rows.append((i, 1, "to_h1"))
        for i in range(70, 75):
            e_rows.append((1, i, "from_h1"))
        for i in range(75, 80):
            e_rows.append((i, 2, "to_h2"))
        for i in range(80, 120):
            e_rows.append((2, i, "from_h2"))
        # Hub-to-hub links: 0 -> 1 -> 2 -> 0
        e_rows.extend([(0, 1, "h0_h1"), (1, 2, "h1_h2"), (2, 0, "h2_h0")])
        # Non-hub chain
        e_rows.extend([(130, 131, "nh1"), (131, 132, "nh2")])

        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "kind"])
        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=150, p_cells=12)

        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)
        self.assertEqual(hg.hub_count, 3)
        self.assert_ExactEquivalence(expected, actual)

    def test_06_no_hubs_pure_fallback_branch(self):
        """High threshold so hub_count == 0; verifies non-hub fast path equals native."""
        from graphframes import GraphFrame

        v_rows, e_rows = generate_seeded_powerlaw_graph(
            num_vertices=150,
            planted_hubs=[(0, 10, 10)],
            background_edges=200,
            seed=99,
        )
        v_df = self.spark.createDataFrame(v_rows, ["id", "name", "score"])
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel", "weight"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=10_000_000, p_cells=16)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 0)
        self.assert_ExactEquivalence(expected, actual)

    def test_07_duplicate_edges_multigraph(self):
        """Duplicate edges (identical and distinct attributes) on both hubs and non-hubs."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"u{i}") for i in range(40)], ["id", "name"])
        e_rows = []
        # Hub 0: duplicate incoming edges from node 1 and 2, duplicate outgoing edges to node 3 and 4
        for k in range(15):
            e_rows.append((1, 0, "dup_same", 1))
            e_rows.append((2, 0, f"dup_diff_{k}", k))
            e_rows.append((0, 3, "dup_out_same", 2))
            e_rows.append((0, 4, f"dup_out_diff_{k}", k * 2))
        # Non-hub duplicate edges through node 10
        e_rows.extend([
            (20, 10, "nh_dup", 5),
            (20, 10, "nh_dup", 5),
            (10, 21, "nh_out", 7),
            (10, 21, "nh_out_2", 8),
        ])
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel", "w"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=8)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 1)
        self.assert_ExactEquivalence(expected, actual)

    def test_08_self_loops_on_hub_and_nonhub(self):
        """Self-loops on hub (0->0) and non-hub (5->5), tested on 2-hop chain and 2-cycle."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(30)], ["id", "name"])
        e_rows = [(0, 0, "hub_self"), (0, 0, "hub_self_dup")]
        for i in range(1, 15):
            e_rows.append((i, 0, f"in_{i}"))
            e_rows.append((0, i, f"out_{i}"))
        # Non-hub self-loop and adjacent edges
        e_rows.extend([(25, 25, "nh_self"), (24, 25, "to_25"), (25, 26, "from_25")])
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=50, p_cells=9)

        # 2-hop chain
        self.assert_ExactEquivalence(g.find(self.MOTIF_2HOP), hg.find(self.MOTIF_2HOP))
        self.assertEqual(hg.hub_count, 1)

        # 2-cycle
        cycle_motif = "(a)-[e1]->(b); (b)-[e2]->(a)"
        self.assert_ExactEquivalence(g.find(cycle_motif), hg.find(cycle_motif))

    def test_09_isolated_and_one_sided_vertices(self):
        """Graph with isolated vertices, in-only sinks, and out-only sources."""
        from graphframes import GraphFrame

        # Vertices 0..99, many isolated (50..99 have no edges at all)
        v_df = self.spark.createDataFrame([(i, f"node_{i}") for i in range(100)], ["id", "name"])
        # Hub 0: sources 1..15 (only out edges), sinks 20..34 (only in edges)
        e_rows = [(i, 0, "in") for i in range(1, 16)] + [(0, j, "out") for j in range(20, 35)]
        # Non-hub 2-hop: 40 (only out) -> 41 -> 42 (only in)
        e_rows += [(40, 41, "nh_in"), (41, 42, "nh_out")]
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=6)
        self.assert_ExactEquivalence(g.find(self.MOTIF_2HOP), hg.find(self.MOTIF_2HOP))
        self.assertEqual(hg.hub_count, 1)

    def test_10_p_cells_equals_1(self):
        """Edge case p_cells=1 on both symmetric and asymmetric hubs."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(60)], ["id", "name"])
        # Hub 0: symmetric 12x12 (W=144); Hub 1: asymmetric 25x6 (W=150)
        e_rows = (
            [(i, 0, "h0_in") for i in range(2, 14)]
            + [(0, i, "h0_out") for i in range(14, 26)]
            + [(i, 1, "h1_in") for i in range(2, 27)]
            + [(1, i, "h1_out") for i in range(30, 36)]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=1)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 2)
        self.assert_ExactEquivalence(expected, actual)

    def test_11_threshold_boundary_exact(self):
        """Boundary test: W == threshold (not a hub) vs W == threshold + 1 (is a hub)."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(200)], ["id", "name"])
        # Node 0: d_in=10, d_out=10 -> W = 100 (exact tie with threshold=100 -> NOT hub)
        # Node 1: d_in=101, d_out=1 -> W = 101 (threshold + 1 -> IS hub)
        # Node 2: d_in=11, d_out=10 -> W = 110 (> threshold -> IS hub)
        e_rows = (
            [(10 + i, 0, "n0_in") for i in range(10)]
            + [(0, 30 + i, "n0_out") for i in range(10)]
            + [(50 + i, 1, "n1_in") for i in range(101)]
            + [(1, 160, "n1_out")]
            + [(170 + i, 2, "n2_in") for i in range(11)]
            + [(2, 185 + i, "n2_out") for i in range(10)]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=100, p_cells=8)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        hub_ids = sorted([r["id"] for r in hg.hub_ids.collect()])
        self.assertEqual(hub_ids, [1, 2], "W == threshold (node 0) must not be a hub; W == threshold+1 (node 1) must be a hub")
        self.assert_ExactEquivalence(expected, actual)

    def test_12_string_and_integer_vertex_ids(self):
        """Tests both non-numeric string IDs and signed integer IDs."""
        from graphframes import GraphFrame

        # String IDs
        str_v = [(f"node_{i}", i) for i in range(40)] + [("hub_alpha", 999)]
        str_e = (
            [(f"node_{i}", "hub_alpha", "in") for i in range(15)]
            + [("hub_alpha", f"node_{i}", "out") for i in range(15, 30)]
            + [("node_31", "node_32", "nh1"), ("node_32", "node_33", "nh2")]
        )
        g_str = GraphFrame(
            self.spark.createDataFrame(str_v, ["id", "idx"]),
            self.spark.createDataFrame(str_e, ["src", "dst", "rel"]),
        )
        hg_str = HyperGraphFrame(g_str, work_threshold=100, p_cells=9)
        self.assert_ExactEquivalence(g_str.find(self.MOTIF_2HOP), hg_str.find(self.MOTIF_2HOP))
        self.assertEqual(hg_str.hub_count, 1)

        # Signed Integer IDs (including negative IDs and 0)
        int_v = [(i - 20, f"v_{i}") for i in range(50)]
        int_e = (
            [(i - 20, -5, 1) for i in range(15)]
            + [(-5, i, 2) for i in range(15)]
            + [(20, 21, 3), (21, 22, 4)]
        )
        g_int = GraphFrame(
            self.spark.createDataFrame(int_v, ["id", "label"]),
            self.spark.createDataFrame(int_e, ["src", "dst", "code"]),
        )
        hg_int = HyperGraphFrame(g_int, work_threshold=100, p_cells=9)
        self.assert_ExactEquivalence(g_int.find(self.MOTIF_2HOP), hg_int.find(self.MOTIF_2HOP))
        self.assertEqual(hg_int.hub_count, 1)

    def test_13_extra_attribute_columns_with_nulls(self):
        """Vertices and edges with multiple nullable attribute columns containing NULLs."""
        from graphframes import GraphFrame

        v_schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("score", DoubleType(), True),
            StructField("active", BooleanType(), True),
        ])
        e_schema = StructType([
            StructField("src", StringType(), True),
            StructField("dst", StringType(), True),
            StructField("rel", StringType(), True),
            StructField("weight", LongType(), True),
            StructField("note", StringType(), True),
        ])

        v_rows = [("hub", None, None, True)]
        for i in range(30):
            v_rows.append((
                f"u{i}",
                None if i % 2 == 0 else f"Name_{i}",
                None if i % 3 == 0 else float(i) * 1.5,
                None if i % 4 == 0 else (i % 2 == 1),
            ))

        e_rows = []
        for i in range(12):
            e_rows.append((
                f"u{i}",
                "hub",
                None if i % 2 == 0 else "in_rel",
                None if i % 3 == 0 else i * 10,
                None,
            ))
        for i in range(12, 24):
            e_rows.append((
                "hub",
                f"u{i}",
                None if i % 2 == 1 else "out_rel",
                None if i % 4 == 0 else i * 10,
                "note_val" if i % 5 == 0 else None,
            ))
        e_rows.append(("u25", "u26", None, None, None))
        e_rows.append(("u26", "u27", "bg", 42, None))

        g = GraphFrame(
            self.spark.createDataFrame(v_rows, schema=v_schema),
            self.spark.createDataFrame(e_rows, schema=e_schema),
        )
        hg = HyperGraphFrame(g, work_threshold=50, p_cells=6)
        self.assert_ExactEquivalence(g.find(self.MOTIF_2HOP), hg.find(self.MOTIF_2HOP))
        self.assertEqual(hg.hub_count, 1)

    def test_14_empty_edge_dataframe(self):
        """Graph with non-empty vertices and empty edges DataFrame."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([("1", "Alice"), ("2", "Bob")], ["id", "name"])
        e_schema = StructType([
            StructField("src", StringType(), True),
            StructField("dst", StringType(), True),
            StructField("rel", StringType(), True),
        ])
        e_df = self.spark.createDataFrame([], schema=e_schema)

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=1, p_cells=4)
        actual = hg.find(self.MOTIF_2HOP)
        expected = g.find(self.MOTIF_2HOP)

        self.assertEqual(hg.hub_count, 0)
        self.assertEqual(actual.count(), 0)
        self.assert_ExactEquivalence(expected, actual)

    def test_15_non_2hop_and_variant_motifs(self):
        """3-hop chain, 1-hop, anonymous 2-hop, and reversed-clause 2-hop must equal native."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(25)], ["id", "name"])
        e_rows = (
            [(i, 0, "in") for i in range(1, 10)]
            + [(0, i, "out") for i in range(10, 19)]
            + [(10, 20, "hop3"), (11, 21, "hop3_b")]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=20, p_cells=6)

        for motif in [
            "(a)-[e1]->(b); (b)-[e2]->(c); (c)-[e3]->(d)",
            "(a)-[e]->(b)",
            "(a)-[]->(b); (b)-[]->(c)",
            "(b)-[e2]->(c); (a)-[e1]->(b)",
        ]:
            with self.subTest(motif=motif):
                self.assert_ExactEquivalence(g.find(motif), hg.find(motif))

    def test_16_dangling_edge_endpoints(self):
        """Edges referencing vertex IDs missing from vertices DataFrame (both attributed and id-only)."""
        from graphframes import GraphFrame

        # Vertex '999' (endpoint c) and '888' (endpoint a) and '777' (intermediate b) are missing from vertices
        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(30)], ["id", "name"])
        e_rows = (
            [(i, 0, "in_ok") for i in range(1, 15)]
            + [(0, i, "out_ok") for i in range(15, 29)]
            # Dangling edges on hub 0: src=888 (not in v_df) -> 0, and 0 -> dst=999 (not in v_df)
            + [(888, 0, "dangling_src"), (0, 999, "dangling_dst")]
            # Dangling intermediate hub 777 (not in v_df): 15 in, 15 out
            + [(i, 777, "dangling_hub_in") for i in range(1, 16)]
            + [(777, i, "dangling_hub_out") for i in range(1, 16)]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        # 16a. Attributed vertices
        g_attr = GraphFrame(v_df, e_df)
        hg_attr = HyperGraphFrame(g_attr, work_threshold=50, p_cells=9)
        self.assert_ExactEquivalence(g_attr.find(self.MOTIF_2HOP), hg_attr.find(self.MOTIF_2HOP))

        # 16b. ID-only vertices (default behavior must still filter out dangling endpoints like native)
        g_id = GraphFrame(v_df.select("id"), e_df.select("src", "dst"))
        hg_id = HyperGraphFrame(g_id, work_threshold=50, p_cells=9)
        self.assert_ExactEquivalence(g_id.find(self.MOTIF_2HOP), hg_id.find(self.MOTIF_2HOP))

    def test_17_duplicate_vertex_ids(self):
        """Vertices DataFrame containing duplicate rows for the same vertex ID (attributed and id-only)."""
        from graphframes import GraphFrame

        v_rows = [(i, f"v{i}") for i in range(25)] + [(0, "hub0_dup"), (5, "v5_dup")]
        v_df = self.spark.createDataFrame(v_rows, ["id", "name"])
        e_rows = (
            [(i, 0, "in") for i in range(1, 12)]
            + [(0, i, "out") for i in range(12, 23)]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        # 17a. Attributed vertices
        g_attr = GraphFrame(v_df, e_df)
        hg_attr = HyperGraphFrame(g_attr, work_threshold=20, p_cells=6)
        self.assert_ExactEquivalence(g_attr.find(self.MOTIF_2HOP), hg_attr.find(self.MOTIF_2HOP))

        # 17b. ID-only vertices with duplicate vertex IDs
        g_id = GraphFrame(v_df.select("id"), e_df.select("src", "dst"))
        hg_id = HyperGraphFrame(g_id, work_threshold=20, p_cells=6)
        self.assert_ExactEquivalence(g_id.find(self.MOTIF_2HOP), hg_id.find(self.MOTIF_2HOP))

    def test_18_degenerate_patterns(self):
        """Degenerate 2-clause motifs: a==b, b==c, a==b==c, and reused edge alias e1==e2."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(20)], ["id", "name"])
        e_rows = (
            [(0, 0, "self_0"), (1, 1, "self_1")]
            + [(i, 0, f"in_{i}") for i in range(1, 10)]
            + [(0, i, f"out_{i}") for i in range(1, 10)]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])
        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=10, p_cells=4)

        for pattern in [
            "(a)-[e1]->(a); (a)-[e2]->(c)",
            "(a)-[e1]->(b); (b)-[e2]->(b)",
            "(a)-[e1]->(a); (a)-[e2]->(a)",
        ]:
            with self.subTest(pattern=pattern):
                self.assert_ExactEquivalence(g.find(pattern), hg.find(pattern))

        # Reused edge alias e1 == e2: native GraphFrames raises an exception; HyperGraphFrame must match
        reused_edge_pattern = "(a)-[e]->(b); (b)-[e]->(c)"
        native_raised = False
        try:
            g.find(reused_edge_pattern).count()
        except Exception:
            native_raised = True

        if native_raised:
            with self.assertRaises(Exception):
                hg.find(reused_edge_pattern).count()
        else:
            self.assert_ExactEquivalence(g.find(reused_edge_pattern), hg.find(reused_edge_pattern))

    def test_19_grid_budget_invariant(self):
        """Verify r_final * s_final <= p_cells even for extreme aspect ratios and p_cells=1."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(150)], ["id", "name"])
        # Hub 0: d_in=100, d_out=2 (W=200, aspect ratio 50:1)
        # Hub 1: d_in=2, d_out=100 (W=200, aspect ratio 1:50)
        e_rows = (
            [(i, 0, "in") for i in range(2, 102)]
            + [(0, 102, "out1"), (0, 103, "out2")]
            + [(104, 1, "in1"), (105, 1, "in2")]
            + [(1, i, "out") for i in range(2, 102)]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])
        g = GraphFrame(v_df, e_df)

        for p_budget in [1, 4, 9]:
            with self.subTest(p_cells=p_budget):
                hg = HyperGraphFrame(g, work_threshold=50, p_cells=p_budget)
                actual = hg.find(self.MOTIF_2HOP)
                expected = g.find(self.MOTIF_2HOP)
                for row in hg.hub_meta.collect():
                    self.assertGreaterEqual(row["r_final"], 1)
                    self.assertGreaterEqual(row["s_final"], 1)
                    self.assertLessEqual(
                        row["r_final"] * row["s_final"],
                        p_budget,
                        f"Grid budget violated for hub {row['id']}: {row['r_final']}*{row['s_final']} > {p_budget}",
                    )
                self.assert_ExactEquivalence(expected, actual)

    def test_20_2cycle_dedicated(self):
        """Dedicated 2-cycle motif (a)-[e1]->(b); (b)-[e2]->(a) across hub-to-hub, hub-to-nonhub, and nonhub pairs."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(50)], ["id", "name"])
        e_rows = []
        # Hub 0 <-> nodes 1..15 bidirectional (creates 2-cycles through hub 0 and through 1..15)
        for i in range(1, 16):
            e_rows.append((i, 0, f"to0_{i}"))
            e_rows.append((0, i, f"from0_{i}"))
        # Hub 1 <-> Hub 0 multiple parallel edges
        e_rows.extend([(0, 1, "h0_h1_a"), (0, 1, "h0_h1_b"), (1, 0, "h1_h0_a")])
        for i in range(16, 30):
            e_rows.append((i, 1, f"to1_{i}"))
            e_rows.append((1, i, f"from1_{i}"))
        # Non-hub 2-cycle 40 <-> 41
        e_rows.extend([(40, 41, "nh_a"), (41, 40, "nh_b")])

        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])
        g = GraphFrame(v_df, e_df)
        hg = HyperGraphFrame(g, work_threshold=50, p_cells=9)

        motif = "(a)-[e1]->(b); (b)-[e2]->(a)"
        actual = hg.find(motif)
        expected = g.find(motif)
        self.assertEqual(hg.hub_count, 2)
        self.assert_ExactEquivalence(expected, actual)


if __name__ == "__main__":
    unittest.main()
