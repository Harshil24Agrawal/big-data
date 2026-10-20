#!/usr/bin/env python3
"""
datagen/hub_sweep.py
====================
"Hub sweep" graphs: a uniform background plus H planted hubs whose share of the total two-hop work
(`work_sum = sum_v d_in(v) * d_out(v)`) is set by design, so the share can be swept from a few percent to ~95 %.

Why: in the 60-graph collection the largest single-vertex work share is 12 %, so a skew-handling strategy never had
a reason to win. This family exists to cover the regime where one or a few vertices carry most of the work.

  python -m datagen.hub_sweep --plan probe             # 3 graphs (H=1; hub work share 50/80/95 %)
  python -m datagen.hub_sweep --plan sweep             # the full sweep
  python -m datagen.hub_sweep --validate               # Spark re-read of every sweep graph, then rebuild the manifest

Outputs (separate from the 60-graph collection):
  datasets/hubsweep/<graph_id>/{edges.parquet,graph.json}
  datasets/hubsweep/manifest.csv   same columns as datasets/manifest/dataset_manifest.csv

Construction (deterministic from the spec and seed): background = n*avg_degree uniform random edges; each hub gets
d distinct in-neighbours and d distinct out-neighbours with d = round(sqrt(s/(1-s) * n*avg_degree^2 / H)), so that the
planned hub work H*d^2 is the fraction s of (background + hub) work. The *measured* share is written to graph.json.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datagen.generators import _decode, _encode, clean_edges, sample_to_target  # noqa: E402
from datagen.generate import (MANIFEST_COLUMNS, edges_sha256, file_sha256, manifest_row,  # noqa: E402
                              write_parquet)
from datagen.validate import degree_stats, structural_checks  # noqa: E402

SWEEP_DIR = os.path.join(REPO_ROOT, "datasets", "hubsweep")
MANIFEST_PATH = os.path.join(SWEEP_DIR, "manifest.csv")
GENERATOR_VERSION = "hubsweep-v1"
AVG_DEGREE = 7

PLANS = {
    # (num_vertices, hub_count, hub_work_share, seed)
    "probe": [(100_000, 1, 0.50, 101), (100_000, 1, 0.80, 101), (100_000, 1, 0.95, 101)],
    # sweep A: the informative middle (hub work ~20M-100M), 1/2/4/8 hubs
    "sweep_a": [(100_000, h, w, 101) for w in (0.80, 0.90, 0.95) for h in (1, 2, 4, 8)],
    # sweep B: the two ends (50 % = mild, 98 % = extreme, hub work ~240M)
    "sweep_b": [(100_000, h, w, 101) for w in (0.50, 0.98) for h in (1, 2, 4, 8)],
    # sweep C: second seed for replication, same grid points as A and B where time allows
    "sweep_c": [(100_000, h, w, 202) for w in (0.90, 0.95, 0.98, 0.50) for h in (1, 4)],
    # sweep big: ONE hub holding 99 % / 99.5 % of the work (hub work ~0.5 B / ~1 B paths) -- the regime where a straggler task
    # could outweigh the hub-splitting profiling cost on 2 cores (sweeps A/B showed edge-join winning up to ~240M).
    "sweep_big": [(100_000, 1, 0.99, 101), (100_000, 1, 0.995, 101)],
    "sweep_big202": [(100_000, 1, 0.99, 202)],
}


def spec_for(n, hubs, share, seed):
    return {"family": "hub_sweep", "size_tier": f"n{n // 1000}k", "seed": int(seed), "num_vertices": int(n),
            "avg_degree": AVG_DEGREE, "m_target": int(n * AVG_DEGREE) , "generator_version": GENERATOR_VERSION,
            "params": {"hub_count": int(hubs), "hub_work_share": float(share), "background": "uniform random"},
            "edge_policy": "directed;no_self_loops;no_duplicates;int32"}


def graph_id_for(spec):
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    p = spec["params"]
    pct = p["hub_work_share"] * 100
    tag = f"{int(round(pct)):02d}" if abs(pct - round(pct)) < 1e-9 else f"{pct:g}".replace(".", "p")
    return f"hubsweep-H{p['hub_count']}-w{tag}-{spec['size_tier']}-s{spec['seed']}-{h}"


def hub_degree(n, hubs, share):
    bg_work = n * AVG_DEGREE * AVG_DEGREE
    return int(round((share / (1.0 - share) * bg_work / hubs) ** 0.5))


def generate(spec):
    n, p = spec["num_vertices"], spec["params"]
    hubs_n, share = p["hub_count"], p["hub_work_share"]
    rng = np.random.default_rng(spec["seed"])
    d = min(hub_degree(n, hubs_n, share), n - 2)
    hubs = rng.choice(n, size=hubs_n, replace=False).astype(np.int32)
    hs, hd = [], []
    for h in hubs:
        ins = rng.choice(n - 1, size=d, replace=False)
        ins = np.where(ins >= h, ins + 1, ins).astype(np.int32)
        outs = rng.choice(n - 1, size=d, replace=False)
        outs = np.where(outs >= h, outs + 1, outs).astype(np.int32)
        hs += [ins, np.full(d, h, np.int32)]
        hd += [np.full(d, h, np.int32), outs]
    hub_keys, _ = clean_edges(np.concatenate(hs), np.concatenate(hd))

    def sampler(size):
        return (rng.integers(0, n, size=size, dtype=np.int32), rng.integers(0, n, size=size, dtype=np.int32))

    (bs, bd), info = sample_to_target(rng, n * AVG_DEGREE, sampler)
    keys = np.unique(np.concatenate([hub_keys, _encode(bs, bd)]))
    info["duplicates_removed"] += int(len(hub_keys) + len(bs) - len(keys))
    info["background_edges"] = int(len(bs))
    info["hub_edges"] = int(len(hub_keys))
    src, dst = _decode(keys)
    order = np.lexsort((dst, src))
    meta = {"hub_count": hubs_n, "hub_ids": hubs.tolist(), "planned_hub_degree_each_side": d,
            "planned_hub_work_share": share}
    return (src[order], dst[order]), {"cleaning": info, "family_meta": meta}


def build_one(n, hubs, share, seed):
    spec = spec_for(n, hubs, share, seed)
    gid = graph_id_for(spec)
    gdir = os.path.join(SWEEP_DIR, gid)
    meta_path = os.path.join(gdir, "graph.json")
    if os.path.exists(meta_path):
        return gid, "exists"
    t0 = time.perf_counter()
    (src, dst), gen = generate(spec)
    gen_s = time.perf_counter() - t0
    e_hash = edges_sha256(src, dst)
    stats = degree_stats(src, dst, n, hub_k=max(hubs, 1))
    structure = structural_checks(src, dst, n)
    ok = (structure["ids_in_range"] and structure["self_loops"] == 0 and structure["duplicate_edges"] == 0)
    os.makedirs(gdir, exist_ok=False)
    pq = os.path.join(gdir, "edges.parquet")
    write_parquet(pq, src, dst)
    import pyarrow
    rec = {"graph_id": gid, "spec": spec, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "generation_seconds": round(gen_s, 3), "validation_seconds": 0.0, "edges_sha256": e_hash,
           "files": [{"path": os.path.relpath(pq, REPO_ROOT).replace(os.sep, "/"), "format": "parquet",
                      "bytes": os.path.getsize(pq), "sha256": file_sha256(pq)}],
           "schema": {"src": "int32", "dst": "int32"}, "cleaning": gen["cleaning"], "family_meta": gen["family_meta"],
           "measured": stats, "structure": structure,
           "validation_status": "PASS" if ok else "FAIL",
           "family_checks": [{"metric": "structure_clean", "passed": bool(ok)}],
           "spark_validation": None,
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "pyarrow": pyarrow.__version__, "platform": platform.platform()}}
    with open(meta_path, "w") as f:
        json.dump(rec, f, indent=2)
    return gid, rec["validation_status"], stats


def rebuild_manifest():
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(SWEEP_DIR, "*", "graph.json")))]
    os.makedirs(SWEEP_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in sorted(recs, key=lambda r: r["graph_id"]):
            w.writerow(manifest_row(r))
    return len(recs)


def spark_validate():
    sys.path.insert(0, os.path.join(REPO_ROOT, "bench"))
    from datagen.spark_validate import make_spark
    from datagen.validate import spark_load_check
    spark = make_spark("HubSweepValidate")
    try:
        for p in sorted(glob.glob(os.path.join(SWEEP_DIR, "*", "graph.json"))):
            rec = json.load(open(p))
            if rec.get("spark_validation"):
                continue
            expected = dict(rec["measured"])
            expected["self_loops"] = rec["structure"]["self_loops"]
            expected["duplicate_edges"] = rec["structure"]["duplicate_edges"]
            t0 = time.perf_counter()
            res = spark_load_check(spark, os.path.join(os.path.dirname(p), "edges.parquet"), expected)
            res["seconds"] = round(time.perf_counter() - t0, 2)
            res["validated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            res["spark_version"] = spark.version
            rec["spark_validation"] = res
            json.dump(rec, open(p, "w"), indent=2)
            print(f"  {rec['graph_id']}: spark validation {res['status']} ({res['seconds']} s)")
    finally:
        spark.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.plan:
        for item in PLANS[args.plan]:
            out = build_one(*item)
            if len(out) == 3:
                s = out[2]
                print(f"{out[0]}: {out[1]}  edges={s['num_edges']:,} work_sum={s['work_sum']:,} "
                      f"work_max_share={s['work_max_share']:.3f} hubs>{s['work_threshold']:,}: {s['num_work_hubs']}")
            else:
                print(f"{out[0]}: {out[1]}")
    if args.validate:
        spark_validate()
    print("manifest rows:", rebuild_manifest())


if __name__ == "__main__":
    main()
