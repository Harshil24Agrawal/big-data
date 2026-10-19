#!/usr/bin/env python3
"""
run_phase1.py
=============
Phase 1 benchmark campaign: run every candidate strategy on every selected graph
and store immutable raw run records.

  python bench/run_phase1.py --name pilot                         # all graphs in the manifest
  python bench/run_phase1.py --name pilot --graph-id <id> ...     # a subset
  python bench/run_phase1.py --name pokec --pokec --max-work-sum 2000000000
  python bench/run_phase1.py --resume <campaign_id>               # continue an interrupted campaign
  python bench/run_phase1.py --name x --dry-run                   # show plan and safety decisions only

Output (never overwritten; every campaign gets a new directory):
  datasets/raw_runs/<campaign_id>/campaign.json      config, machine, plan, safety decisions
  datasets/raw_runs/<campaign_id>/runs.jsonl         one flat record per (graph, strategy, run)
  datasets/raw_runs/<campaign_id>/run_details.jsonl  stage-level metrics + AQE plan analysis per record_id
  datasets/raw_runs/<campaign_id>/logs/<graph_id>.log
  datasets/raw_runs/INDEX.csv                        one line per campaign

Workload safety gate: before a graph runs, its work_sum = sum_v d_in(v)*d_out(v) (the exact
number of 2-hop paths under the simple-digraph edge policy) is compared with --max-work-sum.
Graphs above the limit are recorded as skipped for every strategy and are not executed.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [THIS_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from phase1_strategies import PILOT_STRATEGIES, parse_strategy  # noqa: E402

RAW_DIR = os.path.join(REPO_ROOT, "datasets", "raw_runs")
MANIFEST = os.path.join(REPO_ROOT, "datasets", "manifest", "dataset_manifest.csv")
POKEC_META = os.path.join(REPO_ROOT, "datasets", "pokec", "pokec_graph.json")
WORKER = os.path.join(THIS_DIR, "phase1_worker.py")


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def machine_info() -> dict:
    import psutil
    try:
        import bench
        cpu = bench.detect_cpu_model()
    except Exception:
        cpu = platform.processor() or "unknown"
    info = {
        "os": platform.platform(),
        "cpu_model": cpu,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "python": platform.python_version(),
        "label": os.environ.get("PHASE1_MACHINE_LABEL", ""),
    }
    # Stable id without recording the hostname itself.
    raw = json.dumps({k: info[k] for k in ("os", "cpu_model", "cpu_logical_cores", "ram_total_gb")}, sort_keys=True)
    info["machine_id"] = hashlib.sha256((platform.node() + raw).encode()).hexdigest()[:12]
    return info


def load_graphs(args) -> list:
    graphs = []
    if not args.pokec_only:
        with open(MANIFEST, newline="") as f:
            for row in csv.DictReader(f):
                if args.graph_id and row["graph_id"] not in args.graph_id:
                    continue
                if row["validation_status"] != "PASS" or row["spark_validation_status"] != "PASS":
                    print(f"[plan] excluding {row['graph_id']}: validation={row['validation_status']} "
                          f"spark={row['spark_validation_status']}")
                    continue
                graphs.append({
                    "graph_id": row["graph_id"], "source": "synthetic", "family": row["family"],
                    "size_tier": row["size_tier"], "seed": int(row["seed"]),
                    "num_vertices": int(row["num_vertices"]), "num_edges": int(row["num_edges"]),
                    "work_sum": int(row["work_sum"]), "work_max": int(row["work_max"]),
                    "edges_path": os.path.join(REPO_ROOT, row["path"]),
                })
    if args.pokec or args.pokec_only:
        with open(POKEC_META) as f:
            pk = json.load(f)
        if pk.get("validation_status") != "PASS":
            raise SystemExit(f"Pokec validation status is {pk.get('validation_status')}; refusing to benchmark")
        graphs.append({
            "graph_id": pk["graph_id"], "source": "pokec", "family": "real_pokec", "size_tier": "real",
            "seed": None, "num_vertices": pk["measured"]["num_active_vertices"],
            "num_edges": pk["measured"]["num_edges"], "work_sum": pk["measured"]["work_sum"],
            "work_max": pk["measured"]["work_max"],
            "edges_path": os.path.join(REPO_ROOT, pk["output"]["path"]),
        })
    return graphs


def append_index(row: dict) -> None:
    path = os.path.join(RAW_DIR, "INDEX.csv")
    cols = ["campaign_id", "created_utc", "name", "config_id", "machine_id", "cpu_model", "logical_cores",
            "strategies", "warmup_runs", "measured_runs", "graphs_planned", "graphs_skipped_by_safety",
            "status", "finished_utc"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def write_skip_records(cdir: str, campaign: dict, g: dict, reason: str) -> None:
    with open(os.path.join(cdir, "runs.jsonl"), "a") as f:
        for sid in campaign["strategies"]:
            rec = {
                "schema_version": "phase1-run-v1", "record_id": uuid.uuid4().hex,
                "campaign_id": campaign["campaign_id"], "config_id": campaign["config_id"],
                "graph_id": g["graph_id"], "graph_source": g["source"], "family": g["family"],
                "size_tier": g["size_tier"], "seed": g["seed"], "num_vertices": g["num_vertices"],
                "num_edges": g["num_edges"], "expected_path_count": g["work_sum"],
                **parse_strategy(sid), "run_kind": "planned", "status": "skipped", "skip_reason": reason,
                "machine_id": campaign["machine"]["machine_id"],
                "start_utc": utcnow().isoformat(), "end_utc": utcnow().isoformat(),
            }
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    ap.add_argument("--resume", metavar="CAMPAIGN_ID")
    ap.add_argument("--graph-id", action="append")
    ap.add_argument("--pokec", action="store_true", help="Also benchmark the validated Pokec graph")
    ap.add_argument("--pokec-only", action="store_true")
    ap.add_argument("--strategies", nargs="+", default=PILOT_STRATEGIES)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-work-sum", type=int, default=600_000_000,
                    help="Safety limit on sum_v d_in*d_out; graphs above it are recorded as skipped")
    ap.add_argument("--run-timeout-s", type=float, default=900, help="Cancel a single run after this many seconds")
    ap.add_argument("--graph-timeout-s", type=float, default=4 * 3600, help="Kill a graph's worker after this")
    ap.add_argument("--driver-memory", default=os.environ.get("PHASE1_DRIVER_MEMORY", "6g"))
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--shuffle-partitions", type=int, default=200)
    ap.add_argument("--input-partitions", type=int, default=200)
    ap.add_argument("--work-threshold", type=int, default=1_000_000)
    ap.add_argument("--p-cells", type=int, default=100)
    ap.add_argument("--order-seed", type=int, default=20261019)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for s in args.strategies:
        parse_strategy(s)

    if args.resume:
        cdir = os.path.join(RAW_DIR, args.resume)
        with open(os.path.join(cdir, "campaign.json")) as f:
            campaign = json.load(f)
    else:
        core = {
            "spark_master": args.master, "driver_memory": args.driver_memory,
            "shuffle_partitions": args.shuffle_partitions, "input_partitions": args.input_partitions,
            "auto_broadcast_join_threshold": -1, "vertices": "distinct endpoints, cached, broadcast-hinted",
            "regime": "id_only", "action": "rdd_count (df._jdf.queryExecution().toRdd().count())",
            "work_threshold": args.work_threshold, "p_cells": args.p_cells,
            "warmup_runs": args.warmup, "measured_runs": args.runs,
            "run_timeout_s": args.run_timeout_s, "graph_timeout_s": args.graph_timeout_s,
            "max_work_sum": args.max_work_sum,
            "session_mode": "one fresh Spark application per graph; strategies interleaved, order reshuffled each round",
        }
        machine = machine_info()
        config_id = hashlib.sha256(json.dumps({"core": {k: v for k, v in core.items()
                                                        if k not in ("max_work_sum", "graph_timeout_s")},
                                               "machine": machine["machine_id"]},
                                              sort_keys=True).encode()).hexdigest()[:10]
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        campaign = {
            "campaign_id": f"{stamp}-{args.name}-{uuid.uuid4().hex[:6]}",
            "created_utc": utcnow().isoformat(),
            "name": args.name, "config_id": config_id, "core_config": core, "machine": machine,
            "strategies": args.strategies, "order_seed": args.order_seed,
            "aqe_on_conf": __import__("phase1_strategies").AQE_ON_CONF,
            "graphs": [], "safety_skipped": [],
        }
        cdir = os.path.join(RAW_DIR, campaign["campaign_id"])

    graphs = load_graphs(args) if not args.resume else campaign["graphs"]
    limit = campaign["core_config"]["max_work_sum"]
    plan, skipped = [], []
    for g in graphs:
        if g.get("work_sum") is None:
            skipped.append((g, "work_sum unknown; refusing to run without a workload estimate"))
        elif g["work_sum"] > limit:
            skipped.append((g, f"work_sum {g['work_sum']:,} exceeds safety limit {limit:,}"))
        else:
            plan.append(g)

    print(f"Campaign {campaign['campaign_id']}  config={campaign['config_id']}  machine={campaign['machine']['cpu_model']} "
          f"({campaign['machine']['cpu_logical_cores']} threads, {campaign['machine']['ram_total_gb']} GB)")
    print(f"Strategies: {campaign['strategies']}")
    for g in plan:
        print(f"  RUN   {g['graph_id']:<48} work_sum={g['work_sum']:>14,}")
    for g, why in skipped:
        print(f"  SKIP  {g['graph_id']:<48} {why}")
    if args.dry_run:
        return

    if not args.resume:
        os.makedirs(cdir, exist_ok=False)
        os.makedirs(os.path.join(cdir, "logs"))
        os.makedirs(os.path.join(cdir, "graphs"))
        campaign["graphs"] = graphs
        campaign["safety_skipped"] = [{"graph_id": g["graph_id"], "reason": why} for g, why in skipped]
        with open(os.path.join(cdir, "campaign.json"), "w") as f:
            json.dump(campaign, f, indent=2)
        for g, why in skipped:
            write_skip_records(cdir, campaign, g, why)
        append_index({"campaign_id": campaign["campaign_id"], "created_utc": campaign["created_utc"],
                      "name": campaign["name"], "config_id": campaign["config_id"],
                      "machine_id": campaign["machine"]["machine_id"], "cpu_model": campaign["machine"]["cpu_model"],
                      "logical_cores": campaign["machine"]["cpu_logical_cores"],
                      "strategies": " ".join(campaign["strategies"]),
                      "warmup_runs": campaign["core_config"]["warmup_runs"],
                      "measured_runs": campaign["core_config"]["measured_runs"],
                      "graphs_planned": len(plan), "graphs_skipped_by_safety": len(skipped), "status": "started"})

    progress_path = os.path.join(cdir, "progress.json")
    progress = json.load(open(progress_path)) if os.path.exists(progress_path) else {}
    env = dict(os.environ)
    env["PHASE1_DRIVER_MEMORY"] = campaign["core_config"]["driver_memory"]

    for i, g in enumerate(plan, 1):
        if progress.get(g["graph_id"], {}).get("status") == "done":
            print(f"[{i}/{len(plan)}] {g['graph_id']} already done, skipping")
            continue
        gpath = os.path.join(cdir, "graphs", f"{g['graph_id']}.json")
        with open(gpath, "w") as f:
            json.dump(g, f, indent=2)
        log_path = os.path.join(cdir, "logs", f"{g['graph_id']}.log")
        print(f"[{i}/{len(plan)}] {g['graph_id']} -> {os.path.relpath(log_path, REPO_ROOT)}", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "a") as log:
            try:
                rc = subprocess.run([sys.executable, WORKER, "--campaign-dir", cdir, "--graph-json", gpath],
                                    stdout=log, stderr=subprocess.STDOUT, env=env,
                                    timeout=campaign["core_config"]["graph_timeout_s"]).returncode
                status = "done" if rc == 0 else f"worker_exit_{rc}"
            except subprocess.TimeoutExpired:
                status = "graph_timeout"
        progress[g["graph_id"]] = {"status": status, "seconds": round(time.perf_counter() - t0, 1),
                                   "finished_utc": utcnow().isoformat()}
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"      {status} in {progress[g['graph_id']]['seconds']}s", flush=True)
        if status != "done":
            # Leave an explicit record; the worker's own per-run records up to the failure are kept.
            write_skip_records(cdir, campaign, g, f"worker ended with {status}; see logs/{g['graph_id']}.log")

    done = sum(1 for v in progress.values() if v["status"] == "done")
    append_index({"campaign_id": campaign["campaign_id"], "created_utc": campaign["created_utc"],
                  "name": campaign["name"], "config_id": campaign["config_id"],
                  "machine_id": campaign["machine"]["machine_id"], "cpu_model": campaign["machine"]["cpu_model"],
                  "logical_cores": campaign["machine"]["cpu_logical_cores"],
                  "strategies": " ".join(campaign["strategies"]),
                  "warmup_runs": campaign["core_config"]["warmup_runs"],
                  "measured_runs": campaign["core_config"]["measured_runs"],
                  "graphs_planned": len(plan), "graphs_skipped_by_safety": len(skipped),
                  "status": f"finished ({done}/{len(plan)} graphs done)", "finished_utc": utcnow().isoformat()})
    print(f"Campaign finished: {done}/{len(plan)} graphs done -> {os.path.relpath(cdir, REPO_ROOT)}")


if __name__ == "__main__":
    main()
