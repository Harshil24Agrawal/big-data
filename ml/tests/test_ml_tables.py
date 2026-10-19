import csv
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml import build_feature_table as bft, build_training_table as btt  # noqa: E402

M = {"num_vertices": 1000, "num_edges": 7000, "num_active_vertices": 990, "in_gini": 0.3, "out_gini": 0.2,
     "in_p999": 15.0, "out_p999": 16.0, "work_sum": 49000, "work_max_share": 0.01, "num_work_hubs": 3,
     "work_in_hubs_share": 0.2, "work_threshold": 1000000}


class TestFeatures(unittest.TestCase):
    def test_transforms(self):
        f = bft.features_from_measured(M)
        self.assertAlmostEqual(f["log_num_vertices"], math.log1p(1000))
        self.assertAlmostEqual(f["active_fraction"], 0.99)
        self.assertAlmostEqual(f["log_num_work_hubs"], math.log1p(3))
        self.assertEqual(f["work_threshold"], 1000000)
        self.assertEqual(set(bft.FEATURE_COLUMNS) - set(f), set())

    def test_missing_field_is_an_error_not_a_default(self):
        bad = dict(M)
        del bad["in_p999"]
        with self.assertRaises(KeyError):
            bft.features_from_measured(bad)

    def test_no_execution_columns_in_features(self):
        for c in bft.FEATURE_COLUMNS:
            for banned in ("total_s", "query_s", "runtime", "strategy", "campaign"):
                self.assertNotIn(banned, c)

    def test_feature_table_matches_manifest_if_generated(self):
        path = os.path.join(bft.OUT_DIR, "graph_features.csv")
        if not os.path.exists(path):
            self.skipTest("graph_features.csv not generated")
        feats = {r["graph_id"]: r for r in csv.DictReader(open(path))}
        for r in csv.DictReader(open(bft.MANIFEST)):
            self.assertEqual(int(feats[r["graph_id"]]["raw_work_sum"]), int(r["work_sum"]))
            self.assertEqual(int(feats[r["graph_id"]]["raw_num_edges"]), int(r["num_edges"]))
        self.assertEqual(sum(1 for r in feats.values() if r["role"] == "external"), 1)


class TestLabels(unittest.TestCase):
    def test_best_and_margin(self):
        best, second, margin, clear = btt.choose_label({"a": 10.0, "b": 15.0, "c": 30.0}, {"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertEqual((best, second, margin, clear), ("a", "b", 50.0, True))

    def test_noise_level_gap_is_not_clear(self):
        _, _, margin, clear = btt.choose_label({"a": 10.0, "b": 10.2}, {"a": 1.0, "b": 1.0})
        self.assertFalse(clear)
        self.assertEqual(margin, 2.0)

    def test_tie_is_deterministic(self):
        best, _, _, _ = btt.choose_label({"b": 5.0, "a": 5.0}, {})
        self.assertEqual(best, "a")

    def test_mixed_configs_refused(self):
        with self.assertRaises(SystemExit):
            btt.assert_single_config({"g1": {"config_id": "x", "machine_id": "m"}, "g2": {"config_id": "y", "machine_id": "m"}})
        btt.assert_single_config({"g1": {"config_id": "x", "machine_id": "m"}, "g2": {"config_id": "x", "machine_id": "m"}})


if __name__ == "__main__":
    unittest.main()
