#!/usr/bin/env python3
"""Print the saved benchmark report in the terminal without starting Spark."""

import argparse
import csv
import glob
import json
import os
import sys


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Print saved benchmark results; does not launch Spark or run jobs."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Directory containing summary.md and per-run JSON files.",
    )
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    csv_path = os.path.join(results_dir, "summary.csv")
    if not os.path.isfile(csv_path):
        print(f"Saved summary not found: {csv_path}", file=sys.stderr)
        print("Create it from existing JSON files with: python bench/summarize.py", file=sys.stderr)
        return 1

    print("SAVED BENCHMARK RESULTS — read-only; no Spark tasks are run\n")

    counts = set()
    for result_path in glob.glob(os.path.join(results_dir, "*_count_*.json")):
        try:
            with open(result_path, "r", encoding="utf-8") as result_file:
                data = json.load(result_file)
            count = data.get("dataset", {}).get("path_count")
            if count is not None:
                counts.add(int(count))
        except (OSError, ValueError, TypeError) as exc:
            print(f"Note: skipped unreadable result {os.path.basename(result_path)}: {exc}", file=sys.stderr)

    if counts:
        print("Saved count-mode path count(s): " + ", ".join(f"{count:,}" for count in sorted(counts)))
        if len(counts) > 1:
            print("WARNING: saved count-mode runs disagree; inspect the run JSONs.")
        print()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        print("The saved summary CSV contains no benchmark rows.")
        return 0

    current_group = None
    group_rows = []

    def print_group(group, items):
        action, regime, aqe = group
        print(f"{action.upper()} | {regime.replace('_', ' ').upper()} | AQE {aqe.upper()}")
        latency_header = f"{'Variant':<17} {'N':>3} {'Query median [range]':>25} {'Total incl. profile':>25} {'Profile':>9}"
        print(latency_header)
        print("-" * len(latency_header))
        for item in items:
            query = f"{float(item['query_s_median']):.2f} [{float(item['query_s_min']):.2f}-{float(item['query_s_max']):.2f}]"
            total = f"{float(item['total_s_median']):.2f} [{float(item['total_s_min']):.2f}-{float(item['total_s_max']):.2f}]"
            print(
                f"{item['variant']:<17} {item['n']:>3} {query:>25} {total:>25} "
                f"{float(item['profiling_s_median']):>8.2f}s"
            )

        resource_header = f"{'Variant':<17} {'Max/med':>9} {'Tasks':>7} {'Exec s':>9} {'Shuffle R/W MB':>21} {'Spill mem/disk MB':>22}"
        print("\n" + resource_header)
        print("-" * len(resource_header))
        for item in items:
            shuffle = f"{float(item['shuffle_read_mb_median']):.1f}/{float(item['shuffle_write_mb_median']):.1f}"
            spill = f"{float(item['mem_spill_mb_median']):.1f}/{float(item['disk_spill_mb_median']):.1f}"
            print(
                f"{item['variant']:<17} {float(item['dom_max_med_ratio_median']):>8.2f}x "
                f"{float(item['total_tasks_median']):>7.0f} {float(item['exec_run_time_s_median']):>9.1f} "
                f"{shuffle:>21} {spill:>22}"
            )
        print()

    for row in rows:
        group = (row["action"], row["regime"], row["aqe"])
        if current_group is not None and group != current_group:
            print_group(current_group, group_rows)
            group_rows = []
        current_group = group
        group_rows.append(row)
    if current_group is not None:
        print_group(current_group, group_rows)
    print("Timing: median [min-max], seconds. Resource and task metrics: medians from saved runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
