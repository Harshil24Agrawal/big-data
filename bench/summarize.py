#!/usr/bin/env python3
"""
summarize.py
============
Aggregates per-run benchmark JSONs from bench/results/ into:
  - bench/results/summary.md
  - bench/results/summary.csv

Computes robust summary statistics (n, median, min, max, IQR) for:
  - query_s (query execution only)
  - total_s (including topological profiling for hypercube / salt1d)
  - dominant-stage max/median task duration ratio
  - total executor run time
  - task counts
  - shuffle read and write bytes
  - memory and disk buffer spills

Also generates:
  - Speedup ratios based on medians with min-max ranges
  - Per-stage breakdown tables for a representative run per condition
  - Path count verification checks across variants
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from typing import Dict, Any, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "iqr": None, "mean": None, "std": None}
    s = sorted(values)
    n = len(s)
    med = statistics.median(s)
    v_min = s[0]
    v_max = s[-1]

    # Compute IQR using standard quartile interpolation
    if n >= 4:
        # 25th and 75th percentiles
        mid = n // 2
        lower = s[:mid]
        upper = s[mid:] if n % 2 == 0 else s[mid + 1:]
        q1 = statistics.median(lower)
        q3 = statistics.median(upper)
        iqr = round(q3 - q1, 4)
    elif n > 1:
        iqr = round(v_max - v_min, 4)
    else:
        iqr = 0.0

    mean_v = statistics.mean(s)
    std_v = statistics.stdev(s) if n > 1 else 0.0

    return {
        "n": n,
        "median": round(med, 4),
        "min": round(v_min, 4),
        "max": round(v_max, 4),
        "iqr": iqr,
        "mean": round(mean_v, 4),
        "std": round(std_v, 4),
    }


def format_stat(st: Dict[str, Any], unit: str = "s", prec: int = 2) -> str:
    if st["median"] is None:
        return "N/A"
    return f"{st['median']:.{prec}f} [{st['min']:.{prec}f}–{st['max']:.{prec}f}] (IQR {st['iqr']:.{prec}f})"


def format_bytes(b: Optional[float]) -> str:
    if b is None:
        return "0.0 MB"
    if b >= 1024 ** 3:
        return f"{b / (1024 ** 3):.2f} GB"
    return f"{b / (1024 ** 2):.1f} MB"


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--results-dir", default=os.path.join(THIS_DIR, "results"))
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_md = args.out_md or os.path.join(results_dir, "summary.md")
    out_csv = args.out_csv or os.path.join(results_dir, "summary.csv")

    pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No result JSON files found in {results_dir}")
        sys.exit(0)

    # Group results by (action, aqe, variant)
    runs_by_group: Dict[str, List[Dict[str, Any]]] = {}
    path_counts_by_condition: Dict[str, Dict[str, int]] = {}

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not parse {fpath}: {e}")
            continue

        cfg = data.get("config", {})
        variant = cfg.get("variant")
        regime = cfg.get("regime", "id_only")
        aqe = cfg.get("aqe")
        action = cfg.get("action")
        run_id = cfg.get("run_id")

        if not (variant and aqe and action):
            continue

        group_key = f"{action}|{regime}|aqe_{aqe}|{variant}"
        if group_key not in runs_by_group:
            runs_by_group[group_key] = []
        data["_fpath"] = fpath
        runs_by_group[group_key].append(data)

        # Track path count if present
        pcount = data.get("dataset", {}).get("path_count")
        if pcount is not None:
            cond_key = f"{regime}|{aqe}|{run_id}"
            if cond_key not in path_counts_by_condition:
                path_counts_by_condition[cond_key] = {}
            path_counts_by_condition[cond_key][variant] = pcount

    # Check path count equivalence
    discrepancies = []
    for cond, counts in path_counts_by_condition.items():
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            discrepancies.append((cond, counts))

    summary_rows = []
    variant_order = {"edgejoin": 0, "salt1d": 1, "hypercube": 2, "hypercube_vjoin": 3, "native": 4}

    for group_key, runs in sorted(
        runs_by_group.items(),
        key=lambda kv: (
            kv[0].split("|")[0],
            kv[0].split("|")[1],
            kv[0].split("|")[2],
            variant_order.get(kv[0].split("|")[3], 99),
        ),
    ):
        action, regime, aqe_str, variant = group_key.split("|")
        runs.sort(key=lambda r: r.get("config", {}).get("run_id", 0))

        query_times = [r["timings"]["query_s"] for r in runs if "timings" in r and "query_s" in r["timings"]]
        total_times = [r["timings"]["total_s"] for r in runs if "timings" in r and "total_s" in r["timings"]]
        prof_times = [r["timings"].get("profiling_s", 0.0) for r in runs if "timings" in r]

        # Stage & task metrics from query window
        q_metrics = [r.get("metrics_query_only") or {} for r in runs]
        total_tasks = [m.get("total_tasks", 0) for m in q_metrics if m]
        exec_run_times = [m.get("total_executor_run_time_s", 0.0) for m in q_metrics if m]
        shuf_read = [m.get("total_shuffle_read_bytes", 0) for m in q_metrics if m]
        shuf_write = [m.get("total_shuffle_write_bytes", 0) for m in q_metrics if m]
        mem_spill = [m.get("total_memory_bytes_spilled", 0) for m in q_metrics if m]
        disk_spill = [m.get("total_disk_bytes_spilled", 0) for m in q_metrics if m]

        # Dominant stage max/median ratios
        dom_ratios = []
        for m in q_metrics:
            dom = m.get("dominant_stage") or {}
            td = dom.get("taskDurationMs") or {}
            ratio = td.get("max_median_ratio")
            if ratio is not None:
                dom_ratios.append(ratio)

        summary_rows.append({
            "action": action,
            "regime": regime,
            "aqe": aqe_str.replace("aqe_", ""),
            "variant": variant,
            "n": len(runs),
            "query_s": compute_stats(query_times),
            "total_s": compute_stats(total_times),
            "profiling_s": compute_stats(prof_times),
            "dom_max_med_ratio": compute_stats(dom_ratios),
            "exec_run_time_s": compute_stats(exec_run_times),
            "total_tasks": compute_stats(total_tasks),
            "shuffle_read_bytes": compute_stats(shuf_read),
            "shuffle_write_bytes": compute_stats(shuf_write),
            "mem_spill_bytes": compute_stats(mem_spill),
            "disk_spill_bytes": compute_stats(disk_spill),
            "representative_run": runs[len(runs) // 2],  # Median-ish run
        })

    # Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "action", "regime", "aqe", "variant", "n",
            "query_s_median", "query_s_min", "query_s_max", "query_s_iqr",
            "total_s_median", "total_s_min", "total_s_max", "total_s_iqr",
            "profiling_s_median", "profiling_s_min", "profiling_s_max",
            "dom_max_med_ratio_median", "dom_max_med_ratio_max",
            "exec_run_time_s_median",
            "total_tasks_median",
            "shuffle_read_mb_median",
            "shuffle_write_mb_median",
            "mem_spill_mb_median",
            "disk_spill_mb_median",
        ])
        for row in summary_rows:
            writer.writerow([
                row["action"], row["regime"], row["aqe"], row["variant"], row["n"],
                row["query_s"]["median"], row["query_s"]["min"], row["query_s"]["max"], row["query_s"]["iqr"],
                row["total_s"]["median"], row["total_s"]["min"], row["total_s"]["max"], row["total_s"]["iqr"],
                row["profiling_s"]["median"], row["profiling_s"]["min"], row["profiling_s"]["max"],
                row["dom_max_med_ratio"]["median"], row["dom_max_med_ratio"]["max"],
                row["exec_run_time_s"]["median"],
                row["total_tasks"]["median"],
                round((row["shuffle_read_bytes"]["median"] or 0) / (1024 ** 2), 2),
                round((row["shuffle_write_bytes"]["median"] or 0) / (1024 ** 2), 2),
                round((row["mem_spill_bytes"]["median"] or 0) / (1024 ** 2), 2),
                round((row["disk_spill_bytes"]["median"] or 0) / (1024 ** 2), 2),
            ])

    # Write Markdown summary
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as mf:
        mf.write("# Controlled Benchmark Summary Report\n\n")
        mf.write("Dataset: **soc-pokec** (30,622,564 edges, 1,632,803 vertices)\n")
        mf.write("Motif Query: `(a)-[e1]->(b); (b)-[e2]->(c)`\n")
        mf.write("Spark Configuration: `local[*]`, driver memory 6g, shuffle.partitions=200, autoBroadcastJoinThreshold=-1\n\n")

        if discrepancies:
            mf.write("### ⚠️ PATH COUNT DISCREPANCIES DETECTED!\n")
            for cond, counts in discrepancies:
                mf.write(f"- Condition `{cond}`: {counts}\n")
            mf.write("\n")
        else:
            mf.write("### ✓ Path Count Verification\n")
            mf.write("All count-mode runs produced identical path counts across all evaluated variants.\n\n")

        # Split by action (noop first, then count) and regime
        actions = sorted(list({r["action"] for r in summary_rows}), reverse=True)
        for act in actions:
            regimes = sorted(list({r["regime"] for r in summary_rows if r["action"] == act}))
            for reg in regimes:
                mf.write(f"## Action: `{act}` | Regime: `{reg}`\n\n")
                mf.write("| Variant | Role | AQE | N | Query Latency (s)<br>median [min–max] (IQR) | Total Latency (incl. prof.) (s)<br>median [min–max] | Dominant Stage<br>Max/Median Ratio | Executor Run Time (s)<br>median | Shuffle Read | Shuffle Write | Memory Spill | Disk Spill |\n")
                mf.write("| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

                act_rows = [r for r in summary_rows if r["action"] == act and r["regime"] == reg]
                for r in act_rows:
                    if r["variant"] == "native":
                        role = "Reference (GraphFrames)"
                    elif r["variant"] == "hypercube_vjoin":
                        role = "Drop-in Default (vjoin=on)"
                    else:
                        role = "Headline Comparison"
                    mf.write(
                        f"| **{r['variant']}** | {role} | {r['aqe']} | {r['n']} | "
                        f"`{format_stat(r['query_s'])}` | "
                        f"`{format_stat(r['total_s'])}` | "
                        f"`{r['dom_max_med_ratio']['median']} [max {r['dom_max_med_ratio']['max']}]` | "
                        f"`{r['exec_run_time_s']['median']:.1f}` | "
                        f"{format_bytes(r['shuffle_read_bytes']['median'])} | "
                        f"{format_bytes(r['shuffle_write_bytes']['median'])} | "
                        f"{format_bytes(r['mem_spill_bytes']['median'])} | "
                        f"{format_bytes(r['disk_spill_bytes']['median'])} |\n"
                    )
                mf.write("\n")

                # Headline Speedup Table: hypercube vs edgejoin vs salt1d (plus native reference)
                mf.write(f"### Headline Speedup Comparisons (`hypercube` vs `edgejoin` vs `salt1d`, Action: `{act}`, Regime: `{reg}`)\n\n")
                mf.write("| Condition | Variant | Query-Only Speedup vs `edgejoin` | Total (incl. prof.) Speedup vs `edgejoin` | Query Speedup vs `salt1d` | Total Speedup vs `salt1d` | Reference: Total Speedup vs `native` |\n")
                mf.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

                for aqe in ["off", "on"]:
                    edge_row = next((r for r in act_rows if r["aqe"] == aqe and r["variant"] == "edgejoin"), None)
                    salt1d_row = next((r for r in act_rows if r["aqe"] == aqe and r["variant"] == "salt1d"), None)
                    hyper_row = next((r for r in act_rows if r["aqe"] == aqe and r["variant"] == "hypercube"), None)
                    hyper_vj_row = next((r for r in act_rows if r["aqe"] == aqe and r["variant"] == "hypercube_vjoin"), None)
                    native_row = next((r for r in act_rows if r["aqe"] == aqe and r["variant"] == "native"), None)

                    for cand_row in [edge_row, salt1d_row, hyper_row, hyper_vj_row]:
                        if not cand_row or not cand_row["query_s"]["median"]:
                            continue
                        cand_q_med = cand_row["query_s"]["median"]
                        cand_tot_med = cand_row["total_s"]["median"]

                        # vs edgejoin
                        if edge_row and edge_row["query_s"]["median"]:
                            ej_q = edge_row["query_s"]["median"]
                            ej_tot = edge_row["total_s"]["median"]
                            q_sp_ej = ej_q / cand_q_med
                            tot_sp_ej = ej_tot / cand_tot_med
                            q_min_ej = (edge_row["query_s"]["min"] or ej_q) / (cand_row["query_s"]["max"] or cand_q_med)
                            q_max_ej = (edge_row["query_s"]["max"] or ej_q) / (cand_row["query_s"]["min"] or cand_q_med)
                            vs_ej_q_str = f"**{q_sp_ej:.2f}×** [{q_min_ej:.2f}–{q_max_ej:.2f}×]"
                            vs_ej_tot_str = f"**{tot_sp_ej:.2f}×**"
                        else:
                            vs_ej_q_str = "—"
                            vs_ej_tot_str = "—"

                        # vs salt1d
                        if salt1d_row and salt1d_row["query_s"]["median"]:
                            s_q = salt1d_row["query_s"]["median"]
                            s_tot = salt1d_row["total_s"]["median"]
                            q_sp_s = s_q / cand_q_med
                            tot_sp_s = s_tot / cand_tot_med
                            q_min_s = (salt1d_row["query_s"]["min"] or s_q) / (cand_row["query_s"]["max"] or cand_q_med)
                            q_max_s = (salt1d_row["query_s"]["max"] or s_q) / (cand_row["query_s"]["min"] or cand_q_med)
                            vs_s_q_str = f"**{q_sp_s:.2f}×** [{q_min_s:.2f}–{q_max_s:.2f}×]"
                            vs_s_tot_str = f"**{tot_sp_s:.2f}×**"
                        else:
                            vs_s_q_str = "—"
                            vs_s_tot_str = "—"

                        # vs native reference
                        if native_row and native_row["total_s"]["median"]:
                            nat_tot = native_row["total_s"]["median"]
                            vs_nat_str = f"{nat_tot / cand_tot_med:.2f}×"
                        else:
                            vs_nat_str = "—"

                        mf.write(
                            f"| AQE {aqe} | **{cand_row['variant']}** | "
                            f"{vs_ej_q_str} | {vs_ej_tot_str} | "
                            f"{vs_s_q_str} | {vs_s_tot_str} | "
                            f"{vs_nat_str} |\n"
                        )
                mf.write("\n")

        # Per-stage breakdown for representative runs
        mf.write("## Representative Per-Stage Execution Profiles\n\n")
        for act in actions:
            regimes = sorted(list({r["regime"] for r in summary_rows if r["action"] == act}))
            for reg in regimes:
                mf.write(f"### Action: `{act}` | Regime: `{reg}` (AQE off)\n\n")
                for var in ["edgejoin", "salt1d", "hypercube", "hypercube_vjoin", "native"]:
                    match_row = next(
                        (r for r in summary_rows if r["action"] == act and r["regime"] == reg and r["aqe"] == "off" and r["variant"] == var),
                        None,
                    )
                    if not match_row:
                        continue
                    rep = match_row["representative_run"]
                    q_met = rep.get("metrics_query_only") or {}
                    stages = q_met.get("stages", [])

                    mf.write(f"#### Variant: `{var}` (Regime: `{reg}`, Run ID: {rep.get('config', {}).get('run_id')})\n\n")
                    mf.write(f"- Query Time: `{rep['timings']['query_s']:.2f}s` | Profiling Time: `{rep['timings'].get('profiling_s', 0):.2f}s` | Total: `{rep['timings']['total_s']:.2f}s`\n")
                    mf.write(f"- Total Tasks: `{q_met.get('total_tasks', 0)}` | Executor Run Time: `{q_met.get('total_executor_run_time_s', 0)}s`\n")
                    mf.write(f"- Memory Spill: `{format_bytes(q_met.get('total_memory_bytes_spilled', 0))}` | Disk Spill: `{format_bytes(q_met.get('total_disk_bytes_spilled', 0))}`\n\n")

                    if stages:
                        mf.write("| Stage ID | Name | Tasks | Wall Clock (s) | Exec Time (s) | Min Task (ms) | Median Task (ms) | P95 Task (ms) | Max Task (ms) | Max/Median | Shuffle Read | Shuffle Write | Memory Spill | Disk Spill |\n")
                        mf.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                        for s in stages:
                            td = s.get("taskDurationMs", {})
                            mf.write(
                                f"| {s['stageId']} | {s['name'][:35]} | {s['numTasks']} | "
                                f"{s.get('stageWallClockS', 'N/A')} | {s['executorRunTimeMs'] / 1000.0:.2f} | "
                                f"{td.get('min', 0):.0f} | {td.get('median', 0):.0f} | {td.get('p95', 0):.0f} | {td.get('max', 0):.0f} | "
                                f"`{td.get('max_median_ratio', 'N/A')}` | "
                                f"{format_bytes(s['shuffleReadBytes'])} | {format_bytes(s['shuffleWriteBytes'])} | "
                                f"{format_bytes(s['memoryBytesSpilled'])} | {format_bytes(s['diskBytesSpilled'])} |\n"
                            )
                        mf.write("\n")

    print(f"Summary written to:\n  - {out_md}\n  - {out_csv}")


if __name__ == "__main__":
    main()
