#!/usr/bin/env python3
"""
train_selector.py  (Phase 3 prototype -- PRELIMINARY)
=====================================================
Random-Forest runtime regressor + strategy selector, evaluated with held-out graphs, against baselines.

  python ml/train_selector.py                      # evaluate (leave-one-graph-out and leave-one-family-out), save model + report
  python ml/train_selector.py --select <graph_id>  # explain the selection for one graph from the saved model

Inputs : datasets/derived/training/training_table_eligible.csv  (python ml/build_training_table.py)
Outputs: datasets/derived/selector/{cv_predictions.csv, summary.json, SUMMARY.md, regret_by_graph.png, model.joblib, model_meta.json}

What this is / is not
 * A runnable end-to-end pipeline and an honest first measurement on the graphs that have complete benchmark data.
 * NOT evidence of generalisation: few graphs, one machine, one configuration, few label classes. Everything it prints is
   labelled preliminary, and the selection is always compared against fixed-strategy and rule baselines and the oracle.
 * Outcomes are table lookups of measured medians (`median_total_s__<strategy>`), not new executions. "Speedup" and
   "regret" are therefore statements about those measured medians.
 * Feature-extraction time is NOT included (not measured); inference time is measured and reported.

Selection method: one multi-output RandomForestRegressor predicts log(median_total_s) for every strategy; the strategy
with the smallest prediction is chosen. A 'native fallback' variant keeps native__aqe_on unless the predicted gain over it
exceeds --fallback-margin (default 5 %), which is the safe-by-default policy asked for in the project plan.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
from ml.build_feature_table import FEATURE_COLUMNS  # noqa: E402

DERIVED = os.path.join(REPO_ROOT, "datasets", "derived")
TABLE = os.path.join(DERIVED, "training", "training_table_eligible.csv")
OUT_DIR = os.path.join(DERIVED, "selector")
NATIVE = "native__aqe_on"
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=1, max_features=1.0, random_state=0, n_jobs=1)


def load(path=TABLE):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no label-eligible graphs; nothing to train on")
    strategies = sorted(c.split("__", 1)[1] for c in rows[0] if c.startswith("median_total_s__"))
    X = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows])
    T = np.array([[float(r[f"median_total_s__{s}"]) for s in strategies] for r in rows])
    return rows, strategies, X, T


def rule_policy(row, strategies):
    """Design-prior rule, fixed before any model fit and NOT tuned on the results:
    graphs that contain work hubs use the hub-aware unified hypercube; others use native with AQE on."""
    hubs = float(row["raw_num_work_hubs"]) > 0
    pick = "hypercube_unified__aqe_off" if hubs else NATIVE
    return pick if pick in strategies else NATIVE


def fit(X, T):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(X, np.log(T))
    return m


def choose(pred_log, strategies, fallback_margin=None):
    i = int(np.argmin(pred_log))
    if fallback_margin is None:
        return strategies[i]
    nat = strategies.index(NATIVE)
    gain = 1.0 - math.exp(pred_log[i] - pred_log[nat])
    return strategies[i] if gain > fallback_margin else NATIVE


def evaluate(rows, strategies, X, T, groups, fallback_margin):
    """Held-out evaluation: for each group, train on all other groups, choose for the held-out graphs."""
    out = []
    idx = {s: i for i, s in enumerate(strategies)}
    for g in sorted(set(groups)):
        te = [i for i, x in enumerate(groups) if x == g]
        tr = [i for i, x in enumerate(groups) if x != g]
        if len(tr) < 2:
            continue
        model = fit(X[tr], T[tr])
        mean_by_strategy = np.exp(np.log(T[tr]).mean(axis=0))
        best_fixed = strategies[int(np.argmin(mean_by_strategy))]
        for i in te:
            t0 = time.perf_counter()
            pred = model.predict(X[i:i + 1])[0]
            infer_ms = (time.perf_counter() - t0) * 1000
            picks = {
                "native_aqe_on": NATIVE,
                "native_aqe_off": "native__aqe_off",
                "best_fixed_train": best_fixed,
                "rule_hub": rule_policy(rows[i], strategies),
                "ml": choose(pred, strategies),
                "ml_native_fallback": choose(pred, strategies, fallback_margin),
                "oracle": strategies[int(np.argmin(T[i]))],
            }
            for policy, s in picks.items():
                out.append({"heldout_group": g, "graph_id": rows[i]["graph_id"], "family": rows[i]["family"],
                            "policy": policy, "selected": s, "selected_median_total_s": T[i, idx[s]],
                            "oracle_median_total_s": T[i].min(), "native_median_total_s": T[i, idx[NATIVE]],
                            "inference_ms": infer_ms if policy.startswith("ml") else ""})
    return out


def summarise(pred_rows):
    pols = sorted({r["policy"] for r in pred_rows})
    res = {}
    for p in pols:
        rs = [r for r in pred_rows if r["policy"] == p]
        reg = [r["selected_median_total_s"] / r["oracle_median_total_s"] - 1 for r in rs]
        spd = [r["native_median_total_s"] / r["selected_median_total_s"] for r in rs]
        res[p] = {"graphs": len(rs), "mean_regret": float(np.mean(reg)), "median_regret": float(np.median(reg)),
                  "max_regret": float(np.max(reg)),
                  "geomean_speedup_vs_native_aqe_on": float(np.exp(np.mean(np.log(spd)))),
                  "selections": {s: sum(r["selected"] == s for r in rs) for s in sorted({r["selected"] for r in rs})}}
        res[p]["graphs_where_oracle_chosen"] = int(sum(abs(x) < 1e-12 for x in reg))
    return res


def write_report(cv, label_counts, n_graphs, summaries):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "cv_predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cv[0].keys()))
        w.writeheader()
        w.writerows(cv)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"status": "PRELIMINARY", "graphs": n_graphs, "label_counts": label_counts, "protocols": summaries}, f, indent=2)
    lines = ["# Selector prototype -- PRELIMINARY", "",
             f"Label-eligible graphs: {n_graphs}. Label counts (lowest median total_s): {label_counts}.", "",
             "Outcomes are measured medians from one machine/config; speedup and regret are table lookups of those medians.",
             "Feature-extraction time is not included. Not evidence of generalisation.", ""]
    for proto, summ in summaries.items():
        lines += [f"## Held-out protocol: {proto}", "",
                  "| policy | graphs | mean regret | median regret | max regret | geomean speedup vs native (AQE on) | oracle chosen |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for p, s in summ.items():
            lines.append(f"| {p} | {s['graphs']} | {100*s['mean_regret']:.1f} % | {100*s['median_regret']:.1f} % | "
                         f"{100*s['max_regret']:.1f} % | {s['geomean_speedup_vs_native_aqe_on']:.3f} | {s['graphs_where_oracle_chosen']} |")
        lines.append("")
    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))


def plot(cv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pols = ["native_aqe_on", "best_fixed_train", "rule_hub", "ml", "ml_native_fallback"]
    graphs = sorted({r["graph_id"] for r in cv if r["heldout_group"] == r["graph_id"]})
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(graphs)), 4.5))
    w = 0.8 / len(pols)
    for k, p in enumerate(pols):
        ys = []
        for g in graphs:
            r = next(r for r in cv if r["graph_id"] == g and r["policy"] == p and r["heldout_group"] == g)
            ys.append(100 * (r["selected_median_total_s"] / r["oracle_median_total_s"] - 1))
        ax.bar([i + k * w for i in range(len(graphs))], ys, w, label=p)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(graphs))])
    ax.set_xticklabels([g.split("-s")[0] + "\n" + g.split("-")[1] + "-s" + g.split("-s")[1][:3] for g in graphs],
                       rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("regret vs oracle (%)  [leave-one-graph-out, measured medians]")
    ax.set_title("PRELIMINARY: selection regret by held-out graph")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "regret_by_graph.png"), dpi=130)


def run(fallback_margin):
    rows, strategies, X, T = load()
    from collections import Counter
    label_counts = dict(Counter(strategies[int(np.argmin(T[i]))] for i in range(len(rows))))
    summaries, cv_all = {}, []
    for proto, groups in (("leave-one-graph-out", [r["graph_id"] for r in rows]),
                          ("leave-one-family-out", [r["family"] for r in rows])):
        cv = evaluate(rows, strategies, X, T, groups, fallback_margin)
        for r in cv:
            r["protocol"] = proto
        summaries[proto] = summarise(cv)
        cv_all += cv
    write_report(cv_all, label_counts, len(rows), summaries)
    plot([r for r in cv_all if r["protocol"] == "leave-one-graph-out"])
    model = fit(X, T)
    import joblib
    joblib.dump(model, os.path.join(OUT_DIR, "model.joblib"))
    with open(os.path.join(OUT_DIR, "model_meta.json"), "w") as f:
        json.dump({"status": "PRELIMINARY", "features": FEATURE_COLUMNS, "strategies": strategies, "target": "log(median_total_s)",
                   "rf_params": RF_PARAMS, "training_graphs": [r["graph_id"] for r in rows],
                   "config_ids": sorted({r["config_id"] for r in rows}), "machine_ids": sorted({r["machine_id"] for r in rows}),
                   "fallback_margin": fallback_margin, "native_fallback": NATIVE}, f, indent=2)
    print(open(os.path.join(OUT_DIR, "SUMMARY.md")).read())


def select(graph_id, fallback_margin):
    import joblib
    meta = json.load(open(os.path.join(OUT_DIR, "model_meta.json")))
    model = joblib.load(os.path.join(OUT_DIR, "model.joblib"))
    feats = {r["graph_id"]: r for r in csv.DictReader(open(os.path.join(DERIVED, "features", "graph_features.csv")))}
    if graph_id not in feats:
        raise SystemExit(f"unknown graph_id {graph_id}")
    x = np.array([[float(feats[graph_id][c]) for c in meta["features"]]])
    t0 = time.perf_counter()
    pred = model.predict(x)[0]
    ms = (time.perf_counter() - t0) * 1000
    strategies = meta["strategies"]
    pick = choose(pred, strategies, fallback_margin)
    print(f"graph: {graph_id}   (model status: {meta['status']})")
    for s, p in sorted(zip(strategies, np.exp(pred)), key=lambda z: z[1]):
        print(f"  predicted median total_s  {p:8.2f}  {s}{'   <== selected' if s == pick else ''}")
    imp = sorted(zip(meta["features"], model.feature_importances_), key=lambda z: -z[1])[:3]
    print("  top features by importance: " + ", ".join(f"{k} ({v:.2f})" for k, v in imp))
    print(f"  policy: lowest predicted runtime, native fallback unless predicted gain > {100*fallback_margin:.0f} %; inference {ms:.1f} ms")
    if feats[graph_id]["role"] == "external":
        print("  NOTE: this graph is outside the synthetic training range (external); treat the selection as extrapolation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", metavar="GRAPH_ID")
    ap.add_argument("--fallback-margin", type=float, default=0.05)
    a = ap.parse_args()
    select(a.select, a.fallback_margin) if a.select else run(a.fallback_margin)
