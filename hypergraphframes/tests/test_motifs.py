"""
test_motifs.py
==============
Test suite for HyperGraphFrames middleware (compatible with pytest & unittest).
"""

import os
import sys
import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

# Ensure hypergraph package is in sys.path
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from hypergraph import HyperGraphFrame, profile_topology, run_hypercube_2hop, parse_2hop_motif


def setup_test_env():
    """Maps Windows Hadoop utilities and local Temurin 17 (on PySpark 3.x) if present."""
    import pyspark
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if os.name == "nt":
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
        candidates = [
            os.path.join(os.getcwd(), "hadoop"),
            os.path.abspath(os.path.join(os.getcwd(), "..", "hadoop")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "hadoop")),
        ]
        for hadoop_home in candidates:
            if os.path.exists(hadoop_home):
                os.environ["HADOOP_HOME"] = hadoop_home
                os.environ["hadoop.home.dir"] = hadoop_home
                bin_dir = os.path.join(hadoop_home, "bin")
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                break


def get_graphframes_jar() -> str:
    import pyspark
    if pyspark.__version__.startswith("3."):
        return "graphframes:graphframes:0.8.3-spark3.5-s_2.12"
    return "graphframes:graphframes:0.8.3-spark3.5-s_2.13"


def graphframes_jar_conf():
    """Return the (key, value) Spark config that loads GraphFrames.

    Default: resolve the Maven coordinate via spark.jars.packages (unchanged behaviour).
    If the GRAPHFRAMES_JAR environment variable points to a local JAR, load that file via
    spark.jars instead (for offline or firewalled machines) and put the JAR on sys.path,
    because the GraphFrames Python wrapper ships inside the JAR.
    """
    local_jar = os.environ.get("GRAPHFRAMES_JAR")
    if local_jar:
        jar = os.path.abspath(local_jar)
        if jar not in sys.path:
            sys.path.insert(0, jar)
        return ("spark.jars", jar)
    return ("spark.jars.packages", get_graphframes_jar())


class TestHyperGraphFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_env()
        cls.spark = (
            SparkSession.builder.appName("HyperGraphFramesTestSuite")
            .master("local[2]")
            .config(*graphframes_jar_conf())
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

        from graphframes import GraphFrame

        # Sample standard graph
        vertices = cls.spark.createDataFrame(
            [
                ("1", "Alice", 28),
                ("2", "Bob", 35),
                ("3", "Charlie", 42),
                ("4", "David", 30),
            ],
            ["id", "name", "age"],
        )
        edges = cls.spark.createDataFrame(
            [
                ("1", "2", "friend", 10),
                ("2", "3", "follow", 20),
                ("2", "4", "follow", 30),
                ("3", "1", "friend", 40),
                ("4", "2", "friend", 50),
            ],
            ["src", "dst", "rel", "weight"],
        )
        cls.sample_graph = GraphFrame(vertices, edges)

        # Skewed graph with heavy hub at node '0'
        v_data = [(str(i), f"user_{i}") for i in range(105)]
        skewed_v = cls.spark.createDataFrame(v_data, ["id", "username"])
        edges_list = []
        for i in range(1, 51):
            edges_list.append((str(i), "0", f"in_{i}"))
        for i in range(51, 101):
            edges_list.append(("0", str(i), f"out_{i}"))
        edges_list.append(("101", "102", "bg_1"))
        edges_list.append(("102", "103", "bg_2"))
        skewed_e = cls.spark.createDataFrame(edges_list, ["src", "dst", "edge_type"])
        cls.skewed_graph = GraphFrame(skewed_v, skewed_e)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_01_lazy_profiling(self):
        """Verifies that topological profiling is lazy and executed only upon .find()."""
        hg = HyperGraphFrame(self.sample_graph, work_threshold=2, p_cells=16)
        self.assertIsNone(hg.hub_meta)
        self.assertIsNone(hg.hub_ids)

        res = hg.find("(a)-[e1]->(b); (b)-[e2]->(c)")
        self.assertIsNotNone(hg.hub_meta)
        self.assertIsNotNone(hg.hub_ids)
        self.assertGreater(res.count(), 0)

    def test_02_2hop_chain_named_edges(self):
        """Tests 2-hop chain with named edge variables for exact StructType schema and data match."""
        hg = HyperGraphFrame(self.sample_graph, work_threshold=1, p_cells=16)
        motif = "(a)-[e1]->(b); (b)-[e2]->(c)"

        expected_df = self.sample_graph.find(motif)
        actual_df = hg.find(motif)

        self.assertEqual(actual_df.columns, ["a", "e1", "b", "e2", "c"])
        self.assertEqual(actual_df.columns, expected_df.columns)
        self.assertEqual(actual_df.count(), expected_df.count())

        # Verify StructType schema fields
        for field_name in ["a", "b", "c"]:
            self.assertIsInstance(actual_df.schema[field_name].dataType, StructType)
            self.assertEqual(
                [f.name for f in actual_df.schema[field_name].dataType.fields],
                ["id", "name", "age"],
            )

        for field_name in ["e1", "e2"]:
            self.assertIsInstance(actual_df.schema[field_name].dataType, StructType)
            self.assertEqual(
                [f.name for f in actual_df.schema[field_name].dataType.fields],
                ["src", "dst", "rel", "weight"],
            )

    def test_03_2hop_chain_anonymous_edges(self):
        """Tests 2-hop chain with anonymous edges '[]'."""
        hg = HyperGraphFrame(self.sample_graph, work_threshold=1, p_cells=16)
        motif = "(a)-[]->(b); (b)-[]->(c)"

        expected_df = self.sample_graph.find(motif)
        actual_df = hg.find(motif)

        self.assertEqual(actual_df.columns, ["a", "b", "c"])
        self.assertEqual(actual_df.count(), expected_df.count())

    def test_04_2hop_chain_reordered_clauses(self):
        """Tests 2-hop chain with reordered clauses (b->c before a->b)."""
        hg = HyperGraphFrame(self.sample_graph, work_threshold=1, p_cells=16)
        motif = "(b)-[e2]->(c); (a)-[e1]->(b)"

        expected_df = self.sample_graph.find(motif)
        actual_df = hg.find(motif)

        self.assertEqual(actual_df.columns, ["b", "e2", "c", "a", "e1"])
        self.assertEqual(actual_df.count(), expected_df.count())

    def test_05_2cycle_motif(self):
        """Tests 2-cycle motif (a)->(b)->(a)."""
        hg = HyperGraphFrame(self.sample_graph, work_threshold=1, p_cells=16)
        motif = "(a)-[e1]->(b); (b)-[e2]->(a)"

        expected_df = self.sample_graph.find(motif)
        actual_df = hg.find(motif)

        self.assertEqual(actual_df.columns, ["a", "e1", "b", "e2"])
        self.assertEqual(actual_df.count(), expected_df.count())

    def test_06_fallback_unsupported_motif(self):
        """Tests fallback to native GraphFrames for 3-hop motifs and complex patterns."""
        hg = HyperGraphFrame(self.sample_graph, work_threshold=1, p_cells=16)
        motif = "(a)-[e1]->(b); (b)-[e2]->(c); (c)-[e3]->(d)"

        expected_df = self.sample_graph.find(motif)
        actual_df = hg.find(motif)

        self.assertEqual(actual_df.columns, expected_df.columns)
        self.assertEqual(actual_df.count(), expected_df.count())

    def test_07_skewed_power_law_correctness(self):
        """Validates 2D hypercube partitioning on a skewed power-law graph with heavy hub."""
        # Hub 0 has d_in=50, d_out=50 -> 2,500 2-hop paths through hub 0
        # Background has 1 path: 101 -> 102 -> 103
        # Total paths: 2,501
        hg = HyperGraphFrame(self.skewed_graph, work_threshold=100, p_cells=64)
        motif = "(a)-[e1]->(b); (b)-[e2]->(c)"

        expected_df = self.skewed_graph.find(motif)
        actual_df = hg.find(motif)

        self.assertEqual(actual_df.count(), 2501)
        self.assertEqual(actual_df.count(), expected_df.count())
        self.assertEqual(hg.hub_ids.count(), 1)

    def test_08_delegation_properties(self):
        """Verifies that GraphFrames native properties and methods pass through seamlessly."""
        hg = HyperGraphFrame(self.sample_graph)

        self.assertEqual(hg.vertices.count(), self.sample_graph.vertices.count())
        self.assertEqual(hg.edges.count(), self.sample_graph.edges.count())
        self.assertEqual(hg.degrees.count(), self.sample_graph.degrees.count())
        self.assertEqual(hg.inDegrees.count(), self.sample_graph.inDegrees.count())
        self.assertEqual(hg.outDegrees.count(), self.sample_graph.outDegrees.count())


if __name__ == "__main__":
    unittest.main()
