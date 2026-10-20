#!/usr/bin/env python3
"""
build_training_table.py  (Phase 2/3 interface)
==============================================
One row per graph: pre-execution features + measured outcomes per strategy + `best_strategy` label.

  python ml/build_training_table.py [campaign_id ...]     # default: every campaign that has derived summaries

Inputs (read-only): datasets/derived/features/graph_features.csv  (python ml/build_feature_table.py)
                    datasets/derived/<campaign>/{strategy_summary,graph_summary}.csv (python bench/phase1_aggregate.py <campaign>)
Outputs: datasets/derived/training/training_table.csv           all graphs; label columns blank unless label_eligible
         datasets/derived/training/training_table_eligible.csv  only label-eligible graphs (the training rows)
         datasets/derived/training/SOURCE.txt

Rules
 * A graph's outcomes come from a campaign only if the aggregator marked it label_eligible
   (every planned strategy has >= the campaign's required valid measured runs, correctness gate passed).
 * All rows used come from ONE measurement configuration and ONE machine_id; mixed configurations are refused, not averaged.
   "Measurement configuration" = the campaign's core_config without the safety-only limits (run_timeout_s, graph_timeout_s,
   max_work_sum); `config_id` as recorded in the raw files is kept as a column.
 * If a graph is label-eligible in several campaigns, the latest campaign id (lexicographic = chronological) wins.
 * `best_strategy` = lowest median total_s (profiling + execution).  `label_margin_pct` is the gap to the runner-up and
   `label_clear` is True only when that gap exceeds the larger of the two strategies' IQRs; use it to drop or down-weight
   noise-level labels.  Labels are PRELIMINARY until the benchmark matrix is complete.
 * Pokec has no benchmark data, so it never appears in the eligible table.
"""
import csv
import datetime
import hashlib
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DERIVED = os.path.join(REPO_ROOT, "datasets", "derived")
FEATURES = os.path.join(DERIVED, "features", "graph_features.csv")
OUT_DIR = os.path.join(DERIVED, "training")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _f(x):
    return float(x) if x not in ("", None) else None


def choose_label(medians, iqrs):
    """medians/iqrs: {strategy: float}. Returns (best, runner_up, margin_pct, clear). Pure function."""
    ranked = sorted(medians, key=lambda s: (medians[s], s))
    best, second = ranked[0], ranked[1]
    gap = medians[second] - medians[best]
    margin = 100.0 * gap / medians[best]
    clear = gap > max(iqrs.get(best) or 0.0, iqrs.get(second) or 0.0)
    return best, second, round(margin, 2), clear


# Settings that only bound how long a run may take (safety limits); they do not change what is measured.
# run_phase1.py already leaves max_work_sum and graph_timeout_s out of config_id; run_timeout_s is compared the same way
# here, so that a campaign with a longer per-run limit (for larger graphs) can be combined when no run timed out.
SAFETY_ONLY_KEYS = ("run_timeout_s", "graph_timeout_s", "max_work_sum")


def measurement_config_id(campaign_id):
    """sha256[:10] of the campaign's core_config without safety-only keys, plus machine id and AQE settings."""
    path = os.path.join(REPO_ROOT, "datasets", "raw_runs", campaign_id, "campaign.json")
    with open(path) as f:
        c = json.load(f)
    core = {k: v for k, v in c["core_config"].items() if k not in SAFETY_ONLY_KEYS}
    payload = {"core": core, "machine": (c.get("machine") or {}).get("machine_id"),
               "strategies": c.get("strategies"), "aqe_on_conf": c.get("aqe_on_conf")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def collect(campaign_ids):
    """Return {graph_id: {campaign, config_id, machine_id, strategies:{sid:{median,iqr,n}}}} for eligible graphs only."""
    chosen = {}
    for cid in sorted(campaign_ids):
        gpath = os.path.join(DERIVED, cid, "graph_summary.csv")
        spath = os.path.join(DERIVED, cid, "strategy_summary.csv")
        if not (os.path.exists(gpath) and os.path.exists(spath)):
            continue
        eligible = {g["graph_id"] for g in _read(gpath) if g["label_eligible"] == "True"}
        by_graph = {}
        for r in _read(spath):
            if r["graph_id"] in eligible and r["n_measured_ok"] and int(r["n_measured_ok"]) > 0:
                by_graph.setdefault(r["graph_id"], []).append(r)
        for gid, rs in by_graph.items():
            chosen[gid] = {
                "campaign_id": cid, "config_id": rs[0]["config_id"], "machine_id": rs[0]["machine_id"],
                "measurement_config_id": measurement_config_id(cid),
                "strategies": {r["strategy"]: {"median": _f(r["total_s_median"]), "iqr": _f(r["total_s_iqr"]),
                                               "n": int(r["n_measured_ok"]),
                                               "query_median": _f(r["query_s_median"]),
                                               "profiling_median": _f(r["profiling_s_median"])} for r in rs}}
    return chosen


def assert_single_config(chosen):
    cfgs = {(v.get("measurement_config_id", v["config_id"]), v["machine_id"]) for v in chosen.values()}
    if len(cfgs) > 1:
        raise SystemExit(f"refusing to mix configurations/machines in one training table: {sorted(cfgs)}")


def build(campaign_ids):
    feats = _read(FEATURES)
    chosen = collect(campaign_ids)
    assert_single_config(chosen)
    strategies = sorted({s for v in chosen.values() for s in v["strategies"]})
    rows = []
    for fr in feats:
        row = dict(fr)
        c = chosen.get(fr["graph_id"])
        row["label_eligible"] = bool(c)
        row["campaign_id"] = c["campaign_id"] if c else ""
        row["config_id"] = c["config_id"] if c else ""
        row["measurement_config_id"] = c["measurement_config_id"] if c else ""
        row["machine_id"] = c["machine_id"] if c else ""
        for s in strategies:
            d = (c or {}).get("strategies", {}).get(s)
            row[f"median_total_s__{s}"] = d["median"] if d else ""
            row[f"total_s_iqr__{s}"] = d["iqr"] if d else ""
            row[f"n_valid_runs__{s}"] = d["n"] if d else 0
            row[f"median_query_s__{s}"] = d["query_median"] if d else ""
            row[f"median_profiling_s__{s}"] = d["profiling_median"] if d else ""
        if c and len(c["strategies"]) == len(strategies):
            med = {s: d["median"] for s, d in c["strategies"].items()}
            iqr = {s: d["iqr"] for s, d in c["strategies"].items()}
            best, second, margin, clear = choose_label(med, iqr)
            row.update(best_strategy=best, runner_up_strategy=second, label_margin_pct=margin, label_clear=clear)
        else:
            row["label_eligible"] = False
            row.update(best_strategy="", runner_up_strategy="", label_margin_pct="", label_clear="")
        rows.append(row)
    return rows, strategies


def main(argv):
    cids = argv or [d for d in sorted(os.listdir(DERIVED)) if os.path.isdir(os.path.join(DERIVED, d))
                    and "-smoke-" not in d and os.path.exists(os.path.join(DERIVED, d, "graph_summary.csv"))]
    rows, strategies = build(cids)
    cols = list(rows[0].keys())
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in (("training_table.csv", rows), ("training_table_eligible.csv", [r for r in rows if r["label_eligible"]])):
        with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(OUT_DIR, "SOURCE.txt"), "w") as f:
        f.write(f"derived_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\ncampaigns_considered={cids}\n"
                f"features_sha256={_sha(FEATURES)}\n")
    elig = [r for r in rows if r["label_eligible"]]
    from collections import Counter
    print(f"{len(rows)} graphs ({len(elig)} label-eligible); strategies={len(strategies)}; "
          f"labels={dict(Counter(r['best_strategy'] for r in elig))}; clear labels={sum(1 for r in elig if r['label_clear'] is True)}")


if __name__ == "__main__":
    main(sys.argv[1:])
