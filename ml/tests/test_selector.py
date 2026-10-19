import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml import train_selector as ts  # noqa: E402

STRATS = ["edgejoin__aqe_on", "hypercube_unified__aqe_off", "native__aqe_off", "native__aqe_on"]


class TestChoose(unittest.TestCase):
    def test_argmin(self):
        pred = np.log([5.0, 3.0, 9.0, 8.0])
        self.assertEqual(ts.choose(pred, STRATS), "hypercube_unified__aqe_off")

    def test_native_fallback_when_gain_small(self):
        pred = np.log([7.9, 9.0, 9.0, 8.0])  # edgejoin only 1.25 % better than native
        self.assertEqual(ts.choose(pred, STRATS, 0.05), "native__aqe_on")
        self.assertEqual(ts.choose(pred, STRATS, None), "edgejoin__aqe_on")

    def test_no_fallback_when_gain_large(self):
        pred = np.log([4.0, 9.0, 9.0, 8.0])
        self.assertEqual(ts.choose(pred, STRATS, 0.05), "edgejoin__aqe_on")


class TestRule(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(ts.rule_policy({"raw_num_work_hubs": "3"}, STRATS), "hypercube_unified__aqe_off")
        self.assertEqual(ts.rule_policy({"raw_num_work_hubs": "0"}, STRATS), "native__aqe_on")


class TestEvaluate(unittest.TestCase):
    def test_heldout_never_in_training_and_regret_math(self):
        rng = np.random.default_rng(0)
        n = 8
        X = rng.normal(size=(n, len(ts.FEATURE_COLUMNS)))
        T = np.exp(rng.normal(size=(n, len(STRATS)))) + 1.0
        rows = [{"graph_id": f"g{i}", "family": "f" + str(i % 2), "raw_num_work_hubs": "1"} for i in range(n)]
        cv = ts.evaluate(rows, STRATS, X, T, [r["graph_id"] for r in rows], 0.05)
        self.assertEqual(len(cv), n * 7)
        for r in cv:
            if r["policy"] == "oracle":
                self.assertEqual(r["selected_median_total_s"], r["oracle_median_total_s"])
            self.assertGreaterEqual(r["selected_median_total_s"], r["oracle_median_total_s"] - 1e-12)
        summ = ts.summarise(cv)
        self.assertAlmostEqual(summ["oracle"]["mean_regret"], 0.0)
        self.assertAlmostEqual(summ["native_aqe_on"]["geomean_speedup_vs_native_aqe_on"], 1.0)
        self.assertEqual(summ["oracle"]["graphs_where_oracle_chosen"], n)


if __name__ == "__main__":
    unittest.main()
