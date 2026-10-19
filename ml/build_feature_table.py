#!/usr/bin/env python3
"""
build_feature_table.py  (Phase 2)
=================================
Pre-execution topology features, one row per graph, regenerated from graph artifacts only.

  python ml/build_feature_table.py            # writes datasets/derived/features/graph_features.csv (+ SCHEMA.json, SOURCE.txt)

Inputs  (read-only): datasets/manifest/dataset_manifest.csv, each synthetic graph's graph.json, datasets/pokec/pokec_graph.json
Output  (derived):   datasets/derived/features/graph_features.csv

Rules
 * Features are computed from the graph structure only (degree distribution, work distribution).  No benchmark result,
   runtime, strategy or campaign information is read here, so there is no execution-time leakage.
 * Every graph joins to benchmark data on `graph_id`.
 * Hub columns depend on `work_threshold` (a vertex is a work hub if d_in*d_out > work_threshold); the threshold is
   written into every row so the feature stays reproducible.
 * Pokec is included with role `external` (outside the synthetic range); it is never a training row by default.
"""
import csv
import datetime
import hashlib
import json
import math
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
MANIFEST = os.path.join(REPO_ROOT, "datasets", "manifest", "dataset_manifest.csv")
POKEC_JSON = os.path.join(REPO_ROOT, "datasets", "pokec", "pokec_graph.json")
OUT_DIR = os.path.join(REPO_ROOT, "datasets", "derived", "features")

# (output column, source key in graph.json["measured"], transform)
# Raw counts keep their raw value in `raw_*`; modelling features use log1p.
FEATURE_COLUMNS = [
    "log_num_vertices", "log_num_edges", "active_fraction",
    "in_gini", "out_gini", "log_in_p999", "log_out_p999",
    "log_work_sum", "work_max_share", "log_num_work_hubs", "work_in_hubs_share",
]
RAW_COLUMNS = ["num_vertices", "num_edges", "num_active_vertices", "in_p999", "out_p999", "work_sum",
               "num_work_hubs", "work_threshold"]
ID_COLUMNS = ["graph_id", "graph_source", "role", "family", "size_tier", "seed"]


def _log1p(x):
    return math.log1p(float(x))


def features_from_measured(m):
    """Pure function: graph.json 'measured' dict -> feature dict. Raises KeyError if a field is missing."""
    nv, ne, na = float(m["num_vertices"]), float(m["num_edges"]), float(m["num_active_vertices"])
    return {
        "log_num_vertices": _log1p(nv), "log_num_edges": _log1p(ne), "active_fraction": na / nv,
        "in_gini": float(m["in_gini"]), "out_gini": float(m["out_gini"]),
        "log_in_p999": _log1p(m["in_p999"]), "log_out_p999": _log1p(m["out_p999"]),
        "log_work_sum": _log1p(m["work_sum"]), "work_max_share": float(m["work_max_share"]),
        "log_num_work_hubs": _log1p(m["num_work_hubs"]), "work_in_hubs_share": float(m["work_in_hubs_share"]),
        "raw_num_vertices": int(nv), "raw_num_edges": int(ne), "raw_num_active_vertices": int(na),
        "raw_in_p999": float(m["in_p999"]), "raw_out_p999": float(m["out_p999"]),
        "raw_work_sum": int(m["work_sum"]), "raw_num_work_hubs": int(m["num_work_hubs"]),
        "work_threshold": int(m["work_threshold"]),
    }


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build():
    rows, sources = [], []
    with open(MANIFEST) as f:
        man = list(csv.DictReader(f))
    sources.append(("datasets/manifest/dataset_manifest.csv", _sha(MANIFEST)))
    for r in man:
        gj = os.path.join(REPO_ROOT, os.path.dirname(r["path"]) if r["path"].endswith(".parquet") else r["path"], "graph.json")
        if not os.path.exists(gj):
            gj = os.path.join(REPO_ROOT, "datasets", "synthetic", r["family"], r["graph_id"], "graph.json")
        with open(gj) as f:
            m = json.load(f)["measured"]
        row = {"graph_id": r["graph_id"], "graph_source": "synthetic", "role": "train_candidate",
               "family": r["family"], "size_tier": r["size_tier"], "seed": r["seed"]}
        row.update(features_from_measured(m))
        rows.append(row)
        sources.append((os.path.relpath(gj, REPO_ROOT), _sha(gj)))
    if os.path.exists(POKEC_JSON):
        with open(POKEC_JSON) as f:
            pj = json.load(f)
        row = {"graph_id": pj["graph_id"], "graph_source": "pokec", "role": "external",
               "family": "real_social", "size_tier": "real", "seed": ""}
        row.update(features_from_measured(pj["measured"]))
        rows.append(row)
        sources.append(("datasets/pokec/pokec_graph.json", _sha(POKEC_JSON)))
    return rows, sources


def main():
    rows, sources = build()
    cols = ID_COLUMNS + FEATURE_COLUMNS + ["work_threshold"] + [
        "raw_" + c for c in ("num_vertices", "num_edges", "num_active_vertices", "in_p999", "out_p999", "work_sum",
                             "num_work_hubs")]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "graph_features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["graph_source"], r["graph_id"])):
            w.writerow({c: r[c] for c in cols})
    with open(os.path.join(OUT_DIR, "SCHEMA.json"), "w") as f:
        json.dump({"id_columns": ID_COLUMNS, "feature_columns": FEATURE_COLUMNS,
                   "transform": "log1p on counts and p999 percentiles; ratios unchanged",
                   "work_hub_definition": "vertex with d_in*d_out > work_threshold (column work_threshold)",
                   "leakage_note": "no execution results used; join to benchmark data on graph_id",
                   "roles": {"train_candidate": "synthetic graph", "external": "Pokec; outside the synthetic range, test-only"}},
                  f, indent=2)
    with open(os.path.join(OUT_DIR, "SOURCE.txt"), "w") as f:
        f.write(f"derived_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\nrows={len(rows)}\n")
        for p, h in sources:
            f.write(f"{p} sha256={h}\n")
    print(f"wrote {len(rows)} rows x {len(cols)} columns -> {os.path.relpath(OUT_DIR, REPO_ROOT)}")


if __name__ == "__main__":
    main()
