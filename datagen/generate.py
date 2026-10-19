#!/usr/bin/env python3
"""
datagen/generate.py
===================
Generate Phase 1 synthetic graphs, save them as Parquet, validate them, and
rebuild the dataset manifest.

Examples (run from the repo root):
  python -m datagen.generate --plan pilot
  python -m datagen.generate --plan full                 # 5 families x 3 tiers x 4 seeds
  python -m datagen.generate --family multi_hub --tier small --seed 101
  python -m datagen.generate --plan pilot --verify-existing   # regenerate in memory, compare hashes

Output layout:
  datasets/synthetic/<family>/<graph_id>/edges.parquet   (src int32, dst int32)
  datasets/synthetic/<family>/<graph_id>/graph.json      (params, measured stats, checks)
  datasets/manifest/dataset_manifest.csv                 (one row per graph, rebuilt from graph.json files)

Existing graph directories are never overwritten: a graph_id is a hash of its
parameters, so changed parameters give a new graph_id and a new directory.
"""

import argparse
import csv
import datetime
import glob
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datagen import config  # noqa: E402
from datagen.generators import generate, resolve_params  # noqa: E402
from datagen.validate import degree_stats, run_family_checks, structural_checks  # noqa: E402

DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
SYNTH_DIR = os.path.join(DATASETS_DIR, "synthetic")
MANIFEST_DIR = os.path.join(DATASETS_DIR, "manifest")
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "dataset_manifest.csv")


def spec_for(family: str, tier: str, seed: int) -> Dict:
    t = config.TIERS[tier]
    return {
        "family": family,
        "size_tier": tier,
        "seed": int(seed),
        "num_vertices": t["num_vertices"],
        "avg_degree": t["avg_degree"],
        "m_target": t["num_vertices"] * t["avg_degree"],
        "params": resolve_params(family, tier),
        "generator_version": config.GENERATOR_VERSION,
        "edge_policy": config.EDGE_POLICY,
    }


def graph_id_for(spec: Dict) -> str:
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{spec['family']}-{spec['size_tier']}-s{spec['seed']}-{digest}"


def edges_sha256(src: np.ndarray, dst: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(src, dtype="<i4").tobytes())
    h.update(np.ascontiguousarray(dst, dtype="<i4").tobytes())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet(path: str, src: np.ndarray, dst: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"src": pa.array(src, pa.int32()), "dst": pa.array(dst, pa.int32())})
    pq.write_table(table, path, compression="snappy", row_group_size=1_000_000)


def write_csv_gz(path: str, src: np.ndarray, dst: np.ndarray) -> None:
    with gzip.open(path, "wt", newline="") as f:
        f.write("src,dst\n")
        np.savetxt(f, np.column_stack([src, dst]), fmt="%d", delimiter=",")


def hub_k_for(family_meta: Dict) -> int:
    """Hub-share checks look at the top-H vertices, H = the family's planted hub count (default 10)."""
    return int(family_meta.get("hub_count", 10))


def measure_and_check(family: str, src, dst, num_vertices: int, family_meta: Dict):
    stats = degree_stats(src, dst, num_vertices, hub_k=hub_k_for(family_meta))
    structure = structural_checks(src, dst, num_vertices)
    status, checks = run_family_checks(family, stats, structure)
    return stats, structure, status, checks


def revalidate(meta_path: str, reason: str) -> Tuple[str, str]:
    """Recompute measurements and checks from the saved Parquet file; keep the previous result in history."""
    import pyarrow.parquet as pq

    with open(meta_path) as f:
        rec = json.load(f)
    pq_rel = next(x for x in rec["files"] if x["format"] == "parquet")["path"]
    pq_path = os.path.join(REPO_ROOT, pq_rel)
    if file_sha256(pq_path) != next(x for x in rec["files"] if x["format"] == "parquet")["sha256"]:
        return rec["graph_id"], "FILE-HASH-MISMATCH"
    t = pq.read_table(pq_path)
    src = t.column("src").to_numpy()
    dst = t.column("dst").to_numpy()
    stats, structure, status, checks = measure_and_check(
        rec["spec"]["family"], src, dst, rec["spec"]["num_vertices"], rec["family_meta"])
    rec.setdefault("validation_history", []).append({
        "replaced_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": reason,
        "validation_status": rec["validation_status"],
        "failed_checks": [c for c in rec["family_checks"] if not c["passed"]],
    })
    rec.update({"measured": stats, "structure": structure, "validation_status": status, "family_checks": checks})
    with open(meta_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec["graph_id"], status


def build_one(family: str, tier: str, seed: int, csv_copy: bool = False,
              verify_existing: bool = False) -> Tuple[str, str]:
    spec = spec_for(family, tier, seed)
    gid = graph_id_for(spec)
    gdir = os.path.join(SYNTH_DIR, family, gid)
    meta_path = os.path.join(gdir, "graph.json")

    if os.path.exists(meta_path) and not verify_existing:
        return gid, "exists"

    t0 = time.perf_counter()
    (src, dst), gen_info = generate(family, tier, seed)
    gen_s = time.perf_counter() - t0
    e_hash = edges_sha256(src, dst)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        same = old["edges_sha256"] == e_hash
        return gid, "verified-identical" if same else "VERIFY-MISMATCH"

    t1 = time.perf_counter()
    stats, structure, status, checks = measure_and_check(family, src, dst, spec["num_vertices"],
                                                         gen_info["family_meta"])
    val_s = time.perf_counter() - t1

    os.makedirs(gdir, exist_ok=False)
    pq_path = os.path.join(gdir, "edges.parquet")
    write_parquet(pq_path, src, dst)
    files = [{"path": os.path.relpath(pq_path, REPO_ROOT).replace(os.sep, "/"), "format": "parquet",
              "bytes": os.path.getsize(pq_path), "sha256": file_sha256(pq_path)}]
    if csv_copy:
        csv_path = os.path.join(gdir, "edges.csv.gz")
        write_csv_gz(csv_path, src, dst)
        files.append({"path": os.path.relpath(csv_path, REPO_ROOT).replace(os.sep, "/"), "format": "csv.gz",
                      "bytes": os.path.getsize(csv_path), "sha256": file_sha256(csv_path)})

    import pyarrow
    record = {
        "graph_id": gid,
        "spec": spec,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generation_seconds": round(gen_s, 3),
        "validation_seconds": round(val_s, 3),
        "edges_sha256": e_hash,
        "files": files,
        "schema": {"src": "int32", "dst": "int32"},
        "cleaning": gen_info["cleaning"],
        "family_meta": gen_info["family_meta"],
        "measured": stats,
        "structure": structure,
        "validation_status": status,
        "family_checks": checks,
        "spark_validation": None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "platform": platform.platform(),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2)
    return gid, status


MANIFEST_COLUMNS = [
    "graph_id", "family", "seed", "size_tier", "num_vertices", "num_edges", "m_target",
    "num_active_vertices", "generator_version", "generator_params", "edge_policy",
    "path", "format", "file_bytes", "file_sha256", "edges_sha256", "csv_path",
    "generation_seconds", "validation_status", "failed_checks", "spark_validation_status",
    "work_sum", "work_max", "work_max_share", "num_work_hubs", "work_in_hubs_share",
    "in_max", "out_max", "in_mean", "in_cv", "out_cv", "in_gini", "out_gini",
    "in_max_over_mean", "out_max_over_mean", "log_degree_corr",
    "self_loops_removed", "duplicates_removed", "raw_edges_sampled", "created_utc",
]


def manifest_row(rec: Dict) -> Dict:
    s, m = rec["spec"], rec["measured"]
    pq = next(f for f in rec["files"] if f["format"] == "parquet")
    csvf = next((f for f in rec["files"] if f["format"] == "csv.gz"), None)
    sv = rec.get("spark_validation") or {}
    return {
        "graph_id": rec["graph_id"], "family": s["family"], "seed": s["seed"], "size_tier": s["size_tier"],
        "num_vertices": s["num_vertices"], "num_edges": m["num_edges"], "m_target": s["m_target"],
        "num_active_vertices": m["num_active_vertices"], "generator_version": s["generator_version"],
        "generator_params": json.dumps(s["params"], sort_keys=True),
        "edge_policy": "directed;no_self_loops;no_duplicates;int32",
        "path": pq["path"], "format": "parquet", "file_bytes": pq["bytes"], "file_sha256": pq["sha256"],
        "edges_sha256": rec["edges_sha256"], "csv_path": csvf["path"] if csvf else "",
        "generation_seconds": rec["generation_seconds"], "validation_status": rec["validation_status"],
        "failed_checks": ";".join(c["metric"] for c in rec["family_checks"] if not c["passed"]),
        "spark_validation_status": sv.get("status", "NOT_RUN"),
        "work_sum": m["work_sum"], "work_max": m["work_max"], "work_max_share": m["work_max_share"],
        "num_work_hubs": m["num_work_hubs"], "work_in_hubs_share": m["work_in_hubs_share"],
        "in_max": m["in_max"], "out_max": m["out_max"], "in_mean": m["in_mean"],
        "in_cv": m["in_cv"], "out_cv": m["out_cv"], "in_gini": m["in_gini"], "out_gini": m["out_gini"],
        "in_max_over_mean": m["in_max_over_mean"], "out_max_over_mean": m["out_max_over_mean"],
        "log_degree_corr": m["log_degree_corr"],
        "self_loops_removed": rec["cleaning"]["self_loops_removed"],
        "duplicates_removed": rec["cleaning"]["duplicates_removed"],
        "raw_edges_sampled": rec["cleaning"]["raw_edges_sampled"],
        "created_utc": rec["created_utc"],
    }


def rebuild_manifest() -> int:
    """The manifest is a derived index: rebuild it from every graph.json (sorted, deterministic)."""
    recs = []
    for p in sorted(glob.glob(os.path.join(SYNTH_DIR, "*", "*", "graph.json"))):
        with open(p) as f:
            recs.append(json.load(f))
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in sorted(recs, key=lambda r: r["graph_id"]):
            w.writerow(manifest_row(r))
    return len(recs)


def plan_items(args) -> List[Tuple[str, str, int]]:
    if args.plan == "pilot":
        return list(config.PILOT)
    if args.plan == "full":
        return [(f, t, s) for f in config.FAMILIES for t in config.TIERS for s in config.SEEDS]
    if not (args.family and args.tier and args.seed is not None):
        raise SystemExit("Give --plan, or all of --family/--tier/--seed.")
    return [(args.family, args.tier, args.seed)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=["pilot", "full"])
    ap.add_argument("--family", choices=config.FAMILIES)
    ap.add_argument("--tier", choices=list(config.TIERS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--csv", action="store_true", help="Also write edges.csv.gz next to the Parquet file")
    ap.add_argument("--revalidate", metavar="REASON",
                    help="Recompute stats/checks of existing graphs from their Parquet files "
                         "(previous result kept in validation_history)")
    ap.add_argument("--verify-existing", action="store_true",
                    help="Regenerate existing graphs in memory and compare edge hashes (writes nothing)")
    args = ap.parse_args()

    items = plan_items(args)
    print(f"Generating {len(items)} graph(s) [{config.GENERATOR_VERSION}]")
    bad = 0
    for fam, tier, seed in items:
        t0 = time.perf_counter()
        if args.revalidate:
            spec = spec_for(fam, tier, seed)
            mp = os.path.join(SYNTH_DIR, fam, graph_id_for(spec), "graph.json")
            gid, status = revalidate(mp, args.revalidate) if os.path.exists(mp) else (graph_id_for(spec), "missing")
        else:
            gid, status = build_one(fam, tier, seed, csv_copy=args.csv, verify_existing=args.verify_existing)
        print(f"  {gid:<48} {status:<20} {time.perf_counter() - t0:7.1f}s")
        bad += status in ("FAIL", "VERIFY-MISMATCH", "FILE-HASH-MISMATCH")
    n = rebuild_manifest()
    print(f"Manifest: {os.path.relpath(MANIFEST_PATH, REPO_ROOT)} ({n} graphs)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
