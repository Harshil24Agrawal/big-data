"""
test_phase1.py
==============
Spark tests for the Phase 1 additions:
  - hypercube_unified returns exactly native GraphFrames find() (schema, rows, multiset) in both
    vertex regimes and on edge cases, and routes the same hubs as the split hypercube;
  - the worker's order-independent checksum agrees across all registered strategies;
  - raw run records written by bench/run_phase1.py follow the documented schema.

Run from the repo root:  python -m pytest bench/test_phase1.py -q
(set GRAPHFRAMES_JAR to a local GraphFrames JAR on machines without Maven access)
"""

import glob
import json
import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [THIS_DIR, os.path.join(REPO_ROOT, "hypergraphframes"), os.path.join(REPO_ROOT, "hypergraphframes", "tests"),
          REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pyspark.sql import SparkSession  # noqa: E402
from test_equivalence import generate_seeded_powerlaw_graph, graphframes_jar_conf, setup_test_env  # noqa: E402

from hypercube_unified import UnifiedHypercubeGraphFrame  # noqa: E402
from phase1_strategies import PILOT_STRATEGIES, all_strategy_ids, build_runner, parse_strategy  # noqa: E402

MOTIF = "(a)-[e1]->(b); (b)-[e2]->(c)"


class TestPhase1Spark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_env()
        cls.spark = (
            SparkSession.builder.appName("Phase1Tests").master("local[2]")
            .config(*graphframes_jar_conf())
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def assertSame(self, expected, actual):
        self.assertEqual(actual.columns, expected.columns)
        self.assertEqual(actual.schema, expected.schema)
        self.assertEqual(actual.count(), expected.count())
        self.assertEqual(expected.exceptAll(actual).count(), 0)
        self.assertEqual(actual.exceptAll(expected).count(), 0)

    def _graphs(self):
        from graphframes import GraphFrame
        v_rows, e_rows = generate_seeded_powerlaw_graph(
            num_vertices=300, planted_hubs=[(0, 30, 30), (1, 45, 10), (2, 10, 45)], background_edges=450, seed=42)
        v = self.spark.createDataFrame(v_rows, ["id", "name", "score"])
        e = self.spark.createDataFrame(e_rows, ["src", "dst", "rel", "weight"])
        return GraphFrame(v, e), GraphFrame(v.select("id"), e.select("src", "dst"))

    def test_unified_equals_native_both_regimes(self):
        g_attr, g_id = self._graphs()
        u_attr = UnifiedHypercubeGraphFrame(g_attr, work_threshold=150, p_cells=10, skip_id_only_vertex_join=False)
        self.assertSame(g_attr.find(MOTIF), u_attr.find(MOTIF))
        self.assertGreaterEqual(u_attr.hub_count, 3)
        u_id = UnifiedHypercubeGraphFrame(g_id, work_threshold=150, p_cells=10, skip_id_only_vertex_join=True)
        self.assertSame(g_id.find(MOTIF), u_id.find(MOTIF))

    def test_unified_routes_same_hubs_as_split_hypercube(self):
        from hypergraph import HyperGraphFrame
        _, g_id = self._graphs()
        split = HyperGraphFrame(g_id, work_threshold=150, p_cells=10).profile()
        uni = UnifiedHypercubeGraphFrame(g_id, work_threshold=150, p_cells=10).profile()
        split_rows = sorted((r["id"], r["r_final"], r["s_final"]) for r in split.hub_meta.collect())
        uni_rows = sorted((r["id"], r["r_final"], r["s_final"]) for r in uni._hub_rows)
        self.assertEqual(split_rows, uni_rows)

    def test_unified_edge_cases_and_variant_motifs(self):
        from graphframes import GraphFrame
        v = self.spark.createDataFrame([(i, f"v{i}") for i in range(25)], ["id", "name"])
        rows = ([(0, 0, "self"), (1, 0, "dup"), (1, 0, "dup"), (0, 1, "back")]
                + [(i, 0, "in") for i in range(2, 12)] + [(0, i, "out") for i in range(2, 12)]
                + [(999, 0, "dangling_src"), (0, 888, "dangling_dst")])
        g = GraphFrame(v, self.spark.createDataFrame(rows, ["src", "dst", "rel"]))
        u = UnifiedHypercubeGraphFrame(g, work_threshold=20, p_cells=5)
        for motif in [MOTIF, "(a)-[e1]->(b); (b)-[e2]->(a)", "(b)-[e2]->(c); (a)-[e1]->(b)"]:
            with self.subTest(motif=motif):
                self.assertSame(g.find(motif), u.find(motif))

    def test_every_strategy_checksum_matches_native(self):
        from phase1_worker import checksum_df
        _, g_id = self._graphs()
        ref = checksum_df(g_id.find(MOTIF))
        for sid in all_strategy_ids():
            s = parse_strategy(sid)
            with self.subTest(strategy=sid):
                r = build_runner(s["variant"], g_id, work_threshold=150, p_cells=10)
                self.assertEqual(checksum_df(r.find(MOTIF)), ref)

    def test_strategy_registry(self):
        self.assertEqual(len(set(PILOT_STRATEGIES)), len(PILOT_STRATEGIES))
        for sid in PILOT_STRATEGIES:
            self.assertIn(sid, all_strategy_ids())
        with self.assertRaises(ValueError):
            parse_strategy("bogus__aqe_on")


REQUIRED_ALL = ["schema_version", "record_id", "campaign_id", "config_id", "graph_id", "graph_source",
                "strategy", "variant", "aqe", "run_kind", "status", "start_utc", "end_utc", "machine_id"]
REQUIRED_OK_TIMED = ["query_s", "total_s", "profiling_s", "path_count", "expected_path_count", "correctness_status",
                     "spark_version", "shuffle_partitions", "aqe_conf", "shuffle_read_bytes", "shuffle_write_bytes",
                     "memory_spill_bytes", "disk_spill_bytes", "dom_task_median_ms", "dom_task_p95_ms",
                     "dom_task_max_ms", "dom_task_cv", "round_index", "order_in_round"]


class TestRawRecordSchema(unittest.TestCase):
    def test_raw_records_follow_schema(self):
        files = glob.glob(os.path.join(REPO_ROOT, "datasets", "raw_runs", "*", "runs.jsonl"))
        if not files:
            self.skipTest("no campaigns recorded yet")
        for fpath in files:
            ids = set()
            with open(fpath) as f:
                for line in f:
                    rec = json.loads(line)
                    missing = [k for k in REQUIRED_ALL if k not in rec]
                    self.assertFalse(missing, f"{fpath}: {missing}")
                    self.assertNotIn(rec["record_id"], ids)
                    ids.add(rec["record_id"])
                    self.assertIn(rec["status"], {"ok", "failed", "timeout", "skipped"})
                    if rec["status"] == "ok" and rec["run_kind"] in ("warmup", "measured"):
                        missing = [k for k in REQUIRED_OK_TIMED if k not in rec]
                        self.assertFalse(missing, f"{fpath}: {missing}")
                        self.assertEqual(rec["correctness_status"], "match", rec["record_id"])
                    if rec["status"] in ("failed", "timeout"):
                        self.assertIn("error_message", rec)
                    if rec["status"] == "skipped":
                        self.assertTrue(rec.get("skip_reason"))


if __name__ == "__main__":
    unittest.main()
