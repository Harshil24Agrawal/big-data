"""
test_bench_variants_equivalence.py
==================================
Verifies that the benchmark baselines Salted1DGraphFrame and EdgeJoinGraphFrame
produce identical StructType schemas, column orderings, row counts, and
bidirectional exceptAll multisets to native GraphFrame.find() across both
id-only and attributed regimes and edge cases.
"""

import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PKG_ROOT = os.path.join(REPO_ROOT, "hypergraphframes")
TESTS_DIR = os.path.join(PKG_ROOT, "tests")

for p in [THIS_DIR, PKG_ROOT, TESTS_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

from salting_1d import Salted1DGraphFrame, EdgeJoinGraphFrame
from test_equivalence import setup_test_env, get_graphframes_jar, generate_seeded_powerlaw_graph


class TestBenchVariantsEquivalence(unittest.TestCase):
    MOTIF = "(a)-[e1]->(b); (b)-[e2]->(c)"

    @classmethod
    def setUpClass(cls):
        setup_test_env()
        cls.spark = (
            SparkSession.builder.appName("BenchVariantsEquivalenceTest")
            .master("local[2]")
            .config("spark.jars.packages", get_graphframes_jar())
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def assert_ExactEquivalence(self, expected_df: DataFrame, actual_df: DataFrame):
        self.assertEqual(actual_df.columns, expected_df.columns)
        self.assertEqual(actual_df.schema, expected_df.schema)
        self.assertEqual(actual_df.count(), expected_df.count())
        self.assertEqual(expected_df.exceptAll(actual_df).count(), 0)
        self.assertEqual(actual_df.exceptAll(expected_df).count(), 0)

    def test_01_salt1d_and_edgejoin_powerlaw_both_regimes(self):
        """Verify salt1d and edgejoin equal native on seeded power-law graph in both regimes."""
        from graphframes import GraphFrame

        v_rows, e_rows = generate_seeded_powerlaw_graph(
            num_vertices=300,
            planted_hubs=[(0, 30, 30), (1, 45, 10), (2, 10, 45)],
            background_edges=450,
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

        # Regime B: with_attrs
        g_attr = GraphFrame(v_df, e_df)
        expected_attr = g_attr.find(self.MOTIF)

        sg_attr = Salted1DGraphFrame(g_attr, work_threshold=150, p_cells=10, skip_id_only_vertex_join=False)
        eg_attr = EdgeJoinGraphFrame(g_attr, skip_id_only_vertex_join=False)

        self.assert_ExactEquivalence(expected_attr, sg_attr.find(self.MOTIF))
        self.assertGreaterEqual(sg_attr.hub_count, 3)
        self.assert_ExactEquivalence(expected_attr, eg_attr.find(self.MOTIF))

        # Regime A: id_only (vertices = distinct endpoints)
        g_id = GraphFrame(v_df.select("id"), e_df.select("src", "dst"))
        expected_id = g_id.find(self.MOTIF)

        sg_id = Salted1DGraphFrame(g_id, work_threshold=150, p_cells=10, skip_id_only_vertex_join=True)
        eg_id = EdgeJoinGraphFrame(g_id, skip_id_only_vertex_join=True)

        self.assert_ExactEquivalence(expected_id, sg_id.find(self.MOTIF))
        self.assert_ExactEquivalence(expected_id, eg_id.find(self.MOTIF))

    def test_02_edge_cases_duplicates_selfloops_2cycle_dangling(self):
        """Verify salt1d and edgejoin on duplicates, self-loops, 2-cycles, and dangling endpoints."""
        from graphframes import GraphFrame

        v_df = self.spark.createDataFrame([(i, f"v{i}") for i in range(25)], ["id", "name"])
        e_rows = (
            [(0, 0, "self"), (1, 0, "dup"), (1, 0, "dup"), (0, 1, "back")]
            + [(i, 0, "in") for i in range(2, 12)]
            + [(0, i, "out") for i in range(2, 12)]
            + [(999, 0, "dangling_src"), (0, 888, "dangling_dst")]
        )
        e_df = self.spark.createDataFrame(e_rows, ["src", "dst", "rel"])

        g = GraphFrame(v_df, e_df)
        sg = Salted1DGraphFrame(g, work_threshold=20, p_cells=5, skip_id_only_vertex_join=False)
        eg = EdgeJoinGraphFrame(g, skip_id_only_vertex_join=False)

        for motif in [self.MOTIF, "(a)-[e1]->(b); (b)-[e2]->(a)", "(b)-[e2]->(c); (a)-[e1]->(b)"]:
            with self.subTest(motif=motif):
                expected = g.find(motif)
                self.assert_ExactEquivalence(expected, sg.find(motif))
                self.assert_ExactEquivalence(expected, eg.find(motif))


if __name__ == "__main__":
    unittest.main()
