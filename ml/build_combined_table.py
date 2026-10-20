#!/usr/bin/env python3
"""
build_combined_table.py  (Phase 3, combined dataset)
====================================================
Merges the 31 original benchmarked graphs (5 measured runs, 7 strategies) with the hub-sweep graphs
(3 measured runs, 4 strategies) into ONE table restricted to the 3 strategies measured everywhere.

  python ml/build_combined_table.py
  -> datasets/derived/training/combined_table.csv   (+ combined_SOURCE.txt)

Comparability (checked, refused otherwise)
 * same machine_id for every row;
 * same measurement configuration EXCEPT the number of measured runs per strategy (5 vs 3) and the safety-only limits.
   The `protocol` column records which rows used which; evaluation code reports results overall and per protocol.
Strategies (3): edgejoin__aqe_off, native__aqe_on, hypercube_unified__aqe_off (salt1d was dropped from sweeps B/C to save time: it
   was never fastest on 31 original + 12 sweep-A graphs; its medians are kept as diagnostic columns where measured).
   (AQE-on edge-join is not re-measured on the sweep graphs: with the 256 MB skew threshold AQE skew splitting never
    fires at these data sizes, and on the 31 original graphs edge-join on/off are a statistical tie.)
Classes: edgejoin = edgejoin__aqe_off; hubsplit = hypercube_unified; native = native__aqe_on.
Label = lowest median total_s (profiling + query). `label_clear` = gap to runner-up exceeds both IQRs.
Only graph features known BEFORE running (FEATURE_COLUMNS) are model inputs; runtimes are targets/diagnostics.
"""
import csv
import datetime
import hashlib
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
from ml.build_feature_table import FEATURE_COLUMNS, features_from_measured  # noqa: E402
from ml.build_training_table import choose_label, SAFETY_ONLY_KEYS  # noqa: E402

DERIVED = os.path.join(REPO_ROOT, "datasets", "derived")
RAW = os.path.join(REPO_ROOT, "datasets", "raw_runs")
STRATS = ["edgejoin__aqe_off", "native__aqe_on", "hypercube_unified__aqe_off"]
DIAG = "salt1d__aqe_off"   # measured on the 31 original graphs and sweep A only; diagnostic column, never a label or input
CLASS_OF = {"edgejoin__aqe_off": "edgejoin", "native__aqe_on": "native",
            "hypercube_unified__aqe_off": "hubsplit"}
OUT = os.path.join(DERIVED, "training", "combined_table.csv")


def _read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    return float(x) if x not in ("", None) else None


def comparison_config(cid):
    """Measurement config without safety limits and without the run count; plus machine + AQE conf."""
    with open(os.path.join(RAW, cid, "campaign.json")) as f:
        c = json.load(f)
    core = {k: v for k, v in c["core_config"].items() if k not in SAFETY_ONLY_KEYS + ("measured_runs",)}
    payload = {"core": core, "machine": (c.get("machine") or {}).get("machine_id"), "aqe_on_conf": c.get("aqe_on_conf")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10], payload["machine"], \
        c["core_config"].get("measured_runs")


def original_rows():
    rows = []
    for r in _read(os.path.join(DERIVED, "training", "training_table_eligible.csv")):
        cfg, mach, runs = comparison_config(r["campaign_id"])
        d = {"graph_id": r["graph_id"], "family": r["family"], "size_tier": r["size_tier"], "seed": r["seed"],
             "protocol": f"{runs}runs_original", "source_campaign": r["campaign_id"], "cmp_config": cfg, "machine_id": mach}
        for c in FEATURE_COLUMNS:
            d[c] = r[c]
        d["raw_work_sum"] = r["raw_work_sum"]
        for s in STRATS:
            d[f"median_total_s__{s}"] = r[f"median_total_s__{s}"]
            d[f"total_s_iqr__{s}"] = r[f"total_s_iqr__{s}"]
            d[f"median_query_s__{s}"] = r[f"median_query_s__{s}"]
            d[f"median_profiling_s__{s}"] = r[f"median_profiling_s__{s}"]
            d[f"n_valid_runs__{s}"] = r[f"n_valid_runs__{s}"]
        d[f"median_total_s__{DIAG}"] = r[f"median_total_s__{DIAG}"]
        rows.append(d)
    return rows


def sweep_rows():
    chosen = {}
    chosen_diag = {}
    for cid in sorted(os.listdir(DERIVED)):
        gp, sp = (os.path.join(DERIVED, cid, n) for n in ("graph_summary.csv", "strategy_summary.csv"))
        if "hubsweep-sweep" not in cid or not (os.path.exists(gp) and os.path.exists(sp)):
            continue
        ok = {g["graph_id"] for g in _read(gp) if g["label_eligible"] == "True"}
        by = {}
        for r in _read(sp):
            if r["graph_id"] in ok and r["strategy"] in STRATS and int(r["n_measured_ok"] or 0) > 0:
                by.setdefault(r["graph_id"], {})[r["strategy"]] = r
        for gid, rs in by.items():
            if all(s in rs for s in STRATS):
                chosen[gid] = (cid, rs)
                chosen_diag[gid] = [r for r in _read(sp) if r["graph_id"] == gid and r["strategy"] == DIAG]
    out = []
    for gid, (cid, rs) in sorted(chosen.items()):
        with open(os.path.join(REPO_ROOT, "datasets", "hubsweep", gid, "graph.json")) as f:
            gj = json.load(f)
        spec = gj["spec"]
        feats = features_from_measured(gj["measured"])
        sdiag = (chosen_diag.get(gid) or [None])[0]
        cfg, mach, runs = comparison_config(cid)
        d = {"graph_id": gid, "family": "hub_sweep", "size_tier": "n100k", "seed": spec["seed"],
             "protocol": f"{runs}runs_sweep", "source_campaign": cid, "cmp_config": cfg, "machine_id": mach}
        for c in FEATURE_COLUMNS:
            d[c] = feats[c]
        d["raw_work_sum"] = feats["raw_work_sum"]
        d["hub_count"] = spec["params"]["hub_count"]
        d["hub_work_share_planned"] = spec["params"]["hub_work_share"]
        for s in STRATS:
            r = rs[s]
            d[f"median_total_s__{s}"] = r["total_s_median"]
            d[f"total_s_iqr__{s}"] = r["total_s_iqr"]
            d[f"median_query_s__{s}"] = r["query_s_median"]
            d[f"median_profiling_s__{s}"] = r["profiling_s_median"]
            d[f"n_valid_runs__{s}"] = r["n_measured_ok"]
        d[f"median_total_s__{DIAG}"] = sdiag["total_s_median"] if sdiag else ""
        out.append(d)
    return out


def build():
    rows = original_rows() + sweep_rows()
    cfgs = {(r["cmp_config"], r["machine_id"]) for r in rows}
    if len(cfgs) > 1:
        raise SystemExit(f"refusing to mix configurations/machines: {sorted(cfgs)}")
    for r in rows:
        med = {s: float(r[f"median_total_s__{s}"]) for s in STRATS}
        iqr = {s: _f(r[f"total_s_iqr__{s}"]) for s in STRATS}
        best, second, margin, clear = choose_label(med, iqr)
        r.update(best_strategy=best, best_class=CLASS_OF[best], runner_up_strategy=second,
                 label_margin_pct=margin, label_clear=clear)
        r.setdefault("hub_count", "")
        r.setdefault("hub_work_share_planned", "")
    return rows


def main():
    rows = build()
    cols = ["graph_id", "family", "size_tier", "seed", "protocol", "source_campaign", "machine_id", "cmp_config",
            "hub_count", "hub_work_share_planned"] + FEATURE_COLUMNS + ["raw_work_sum"]
    for s in STRATS:
        cols += [f"median_total_s__{s}", f"total_s_iqr__{s}", f"median_query_s__{s}", f"median_profiling_s__{s}",
                 f"n_valid_runs__{s}"]
    cols += [f"median_total_s__{DIAG}"]
    cols += ["best_strategy", "best_class", "runner_up_strategy", "label_margin_pct", "label_clear"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])
    with open(os.path.join(os.path.dirname(OUT), "combined_SOURCE.txt"), "w") as f:
        f.write(f"derived_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\nrows={len(rows)}\n"
                f"campaigns={sorted({r['source_campaign'] for r in rows})}\n")
    from collections import Counter
    print(f"{len(rows)} rows x {len(cols)} cols; protocols={dict(Counter(r['protocol'] for r in rows))}; "
          f"classes={dict(Counter(r['best_class'] for r in rows))}; strategies={dict(Counter(r['best_strategy'] for r in rows))}; "
          f"clear={sum(1 for r in rows if r['label_clear'] is True)}")


if __name__ == "__main__":
    main()
