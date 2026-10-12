#!/usr/bin/env python3
"""
bench/run_ablation_matrix.py
============================
Orchestrates the 2x2 AQE ablation (+ AQE off baseline) across edgejoin and native variants:
  variants: [edgejoin, native]
  conditions:
    1. aqe=on,  skew=false, coal=false
    2. aqe=on,  skew=false, coal=true
    3. aqe=on,  skew=true,  coal=false
    4. aqe=on,  skew=true,  coal=true
    5. aqe=off, skew=false, coal=false  (AQE off baseline)
  runs: 3 per cell
Total runs: 2 variants x 5 conditions x 3 runs = 30 runs.
Execution is fully RESUMABLE (skips existing valid JSON files).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RUNNER_SCRIPT = os.path.join(THIS_DIR, "run_single_ablation.py")
RESULTS_DIR = os.path.join(THIS_DIR, "results_ablation")


def build_plan() -> List[Dict[str, Any]]:
    variants = ["edgejoin", "native"]
    conditions = [
        {"aqe": "off", "skew": "false", "coal": "false", "label": "aqe_off"},
        {"aqe": "on",  "skew": "false", "coal": "false", "label": "aqe_on_skew_false_coal_false"},
        {"aqe": "on",  "skew": "false", "coal": "true",  "label": "aqe_on_skew_false_coal_true"},
        {"aqe": "on",  "skew": "true",  "coal": "false", "label": "aqe_on_skew_true_coal_false"},
        {"aqe": "on",  "skew": "true",  "coal": "true",  "label": "aqe_on_skew_true_coal_true"},
    ]
    runs = [1, 2, 3]

    plan = []
    # Interleave runs across variants and conditions
    for r in runs:
        for c in conditions:
            for v in variants:
                plan.append({
                    "variant": v,
                    "aqe": c["aqe"],
                    "skew": c["skew"],
                    "coal": c["coal"],
                    "label": c["label"],
                    "run_id": r,
                })
    return plan


def is_valid_result(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "metrics" in data and "plan_analysis" in data
    except Exception:
        return False


def run_matrix():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plan = build_plan()
    total = len(plan)

    print(f"=======================================================")
    print(f"STARTING AQE SKEW JOIN / COALESCE ABLATION MATRIX")
    print(f"Total planned runs: {total} (3 runs per cell)")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"=======================================================\n")

    completed = 0
    skipped = 0
    t0_matrix = time.perf_counter()

    for i, item in enumerate(plan, start=1):
        v = item["variant"]
        aqe = item["aqe"]
        skew = item["skew"]
        coal = item["coal"]
        r = item["run_id"]

        out_name = f"{v}_{item['label']}_run_{r}.json"
        out_path = os.path.join(RESULTS_DIR, out_name)

        if is_valid_result(out_path):
            print(f"[{i}/{total}] SKIP (already completed): {out_name}")
            skipped += 1
            continue

        print(f"\n>>> [{i}/{total}] LAUNCHING: {out_name} (variant={v}, aqe={aqe}, skew={skew}, coal={coal}, run={r})")
        cmd = [
            sys.executable,
            RUNNER_SCRIPT,
            "--variant", v,
            "--aqe", aqe,
            "--skew-join", skew,
            "--coalesce", coal,
            "--run-id", str(r),
            "--out", out_path,
        ]

        t_run0 = time.perf_counter()
        proc = subprocess.run(cmd)
        elapsed = time.perf_counter() - t_run0

        if proc.returncode != 0:
            print(f"[ERROR] Run failed with return code {proc.returncode} for {out_name}")
            sys.exit(proc.returncode)

        completed += 1
        print(f"[OK] Completed [{i}/{total}] in {elapsed:.1f}s")

    t_total = time.perf_counter() - t0_matrix
    print(f"\n=======================================================")
    print(f"ABLATION EXECUTION FINISHED")
    print(f"Total: {total} | Completed: {completed} | Skipped: {skipped} | Total Time: {t_total:.1f}s")
    print(f"=======================================================\n")


if __name__ == "__main__":
    run_matrix()
