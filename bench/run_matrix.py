#!/usr/bin/env python3
"""
run_matrix.py
=============
Orchestrates the controlled benchmark matrix:
  variants {native, salt1d, hypercube} x aqe {on, off} x N runs (default 5)
Each run executes in a fresh, isolated Python subprocess.
The execution sequence is interleaved and randomized with a fixed seed to distribute
thermal and OS buffer-cache effects across variants.
Execution is fully RESUMABLE (skips existing result files).
"""

import argparse
import os
import random
import subprocess
import sys
import time
import psutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
BENCH_SCRIPT = os.path.join(THIS_DIR, "bench.py")


def check_disk_space(required_gb: float = 25.0) -> float:
    usage = psutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < required_gb:
        print(f"⚠️ WARNING: Free disk space is {free_gb:.1f} GB, which is below the recommended {required_gb} GB.")
        print("Large SortMergeJoin memory/disk spills could risk filling the drive.")
    else:
        print(f"✓ Free disk space check passed: {free_gb:.1f} GB available (threshold: {required_gb} GB).")
    return free_gb


def build_run_list(variants, regimes, aqe_modes, num_runs, seed=42):
    items = []
    for r in range(1, num_runs + 1):
        for reg in regimes:
            for v in variants:
                # hypercube_vjoin is only distinct from hypercube in the id_only regime
                if v == "hypercube_vjoin" and reg != "id_only":
                    continue
                for aqe in aqe_modes:
                    items.append({
                        "variant": v,
                        "regime": reg,
                        "aqe": aqe,
                        "run_id": r,
                    })
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def main():
    parser = argparse.ArgumentParser(description="Run controlled benchmark matrix")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["native", "edgejoin", "salt1d", "hypercube", "hypercube_vjoin"],
    )
    parser.add_argument("--regimes", nargs="+", default=["id_only", "with_attrs"])
    parser.add_argument("--aqe", nargs="+", default=["off", "on"])
    parser.add_argument("--action", default="noop", choices=["noop", "count"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edges-limit", type=int, default=None)
    parser.add_argument("--work-threshold", type=int, default=1_000_000)
    parser.add_argument("--p-cells", type=int, default=100)
    parser.add_argument("--out-dir", default=os.path.join(THIS_DIR, "results"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    check_disk_space()

    run_plan = build_run_list(args.variants, args.regimes, args.aqe, args.runs, seed=args.seed)
    total_runs = len(run_plan)
    print(f"\n=======================================================")
    print(f"BENCHMARK MATRIX PLAN: {total_runs} runs total ({args.runs} runs per condition)")
    print(f"Variants : {args.variants}")
    print(f"Regimes  : {args.regimes}")
    print(f"AQE      : {args.aqe}")
    print(f"Action   : {args.action}")
    print(f"Seed     : {args.seed}")
    print(f"Out Dir  : {args.out_dir}")
    print(f"=======================================================\n")

    completed = 0
    skipped = 0

    t_start = time.perf_counter()

    for i, item in enumerate(run_plan, start=1):
        v = item["variant"]
        reg = item["regime"]
        aqe = item["aqe"]
        r = item["run_id"]
        limit_tag = f"_limit{args.edges_limit}" if args.edges_limit else ""
        out_filename = f"{v}_{reg}_aqe_{aqe}_{args.action}_run_{r}{limit_tag}.json"
        out_path = os.path.join(args.out_dir, out_filename)

        if os.path.exists(out_path):
            print(f"[{i}/{total_runs}] SKIP (already exists): {out_filename}")
            skipped += 1
            continue

        print(f"\n[{i}/{total_runs}] LAUNCHING: variant={v} regime={reg} aqe={aqe} action={args.action} run={r}")
        cmd = [
            sys.executable,
            BENCH_SCRIPT,
            "--variant", v,
            "--regime", reg,
            "--aqe", aqe,
            "--action", args.action,
            "--run-id", str(r),
            "--out", out_path,
            "--work-threshold", str(args.work_threshold),
            "--p-cells", str(args.p_cells),
        ]
        if args.edges_limit is not None:
            cmd.extend(["--edges-limit", str(args.edges_limit)])

        t_run0 = time.perf_counter()
        proc = subprocess.run(cmd)
        run_elapsed = time.perf_counter() - t_run0

        if proc.returncode != 0:
            print(f"❌ ERROR: Run failed with exit code {proc.returncode} for {out_filename}")
            sys.exit(proc.returncode)

        completed += 1
        print(f"✓ Completed [{i}/{total_runs}] in {run_elapsed:.1f}s")

    t_total = time.perf_counter() - t_start
    print(f"\n=======================================================")
    print(f"MATRIX EXECUTION COMPLETE")
    print(f"Total: {total_runs} | Completed: {completed} | Skipped: {skipped} | Wall Time: {t_total:.1f}s")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
