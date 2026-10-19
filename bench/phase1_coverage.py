#!/usr/bin/env python3
"""
phase1_coverage.py
==================
Benchmark coverage matrix across ALL Phase 1 campaigns, computed only from raw records.

  python bench/phase1_coverage.py            # writes datasets/derived/coverage/ and prints a summary

Outputs (derived, regenerable; raw files are only read):
  datasets/derived/coverage/coverage_matrix.csv   one row per (campaign, graph, strategy) with run counts
  datasets/derived/coverage/campaign_status.csv   one row per campaign: status, graphs, valid runs, notes
  datasets/derived/coverage/SOURCE.txt            sha256 of every raw file used

Counting rules (same acceptance rule as phase1_aggregate.py):
  valid measured = run_kind == "measured" and status == "ok" and correctness_status == "match",
                   restricted to the latest spark_app_id of that graph within the campaign.
  Records of earlier spark_app_ids of the same graph are counted as `superseded` (interrupted
  attempts that were resumed), never as valid runs.
Campaign status is derived from evidence on disk: ABORTED.txt / INTERRUPTIONS.md markers,
progress.json, and INDEX.csv rows. Nothing is inferred from campaign names.
"""

import csv
import datetime
import hashlib
import json
import os
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RAW_DIR = os.path.join(REPO_ROOT, "datasets", "raw_runs")
OUT_DIR = os.path.join(REPO_ROOT, "datasets", "derived", "coverage")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_campaign(cdir):
    recs = []
    path = os.path.join(cdir, "runs.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    camp = {}
    if os.path.exists(os.path.join(cdir, "campaign.json")):
        with open(os.path.join(cdir, "campaign.json")) as f:
            camp = json.load(f)
    progress = {}
    if os.path.exists(os.path.join(cdir, "progress.json")):
        with open(os.path.join(cdir, "progress.json")) as f:
            progress = json.load(f)
    return recs, camp, progress


def latest_apps(recs):
    """Latest spark_app_id per graph by order of appearance in the append-only file."""
    last = {}
    for r in recs:
        if r.get("spark_app_id"):
            last[r["graph_id"]] = r["spark_app_id"]
    return last


def coverage_rows(cid, recs, camp):
    last = latest_apps(recs)
    planned = camp.get("strategies") or sorted({r["strategy"] for r in recs})
    target = (camp.get("core_config") or {}).get("measured_runs")
    cells = defaultdict(lambda: defaultdict(int))
    graphs = {}
    for r in recs:
        gid, sid = r["graph_id"], r["strategy"]
        graphs.setdefault(gid, r)
        c = cells[(gid, sid)]
        app = r.get("spark_app_id")
        if app and app != last.get(gid):
            c["superseded"] += 1
            continue
        kind, status = r.get("run_kind"), r.get("status")
        if kind == "correctness":
            c["gate_" + (r.get("correctness_status") or status)] += 1
        elif status == "skipped":
            c["skipped"] += 1
        elif kind == "warmup":
            c["warmup_" + status] += 1
        elif kind == "measured":
            if status == "ok" and r.get("correctness_status") == "match":
                c["valid_measured"] += 1
            elif status == "ok":
                c["measured_correctness_" + str(r.get("correctness_status"))] += 1
            else:
                c["measured_" + status] += 1
    rows = []
    for gid, g in sorted(graphs.items()):
        for sid in planned:
            c = cells.get((gid, sid), {})
            valid = c.get("valid_measured", 0)
            rows.append({
                "campaign_id": cid, "graph_id": gid, "family": g.get("family"), "size_tier": g.get("size_tier"),
                "seed": g.get("seed"), "strategy": sid, "target_measured_runs": target,
                "gate": ("match" if c.get("gate_match") else "mismatch" if c.get("gate_mismatch")
                         else "failed" if c.get("gate_failed") or c.get("gate_timeout") else "not_run"),
                "valid_measured": valid,
                "warmup_ok": c.get("warmup_ok", 0),
                "measured_timeout": c.get("measured_timeout", 0) + c.get("warmup_timeout", 0),
                "measured_failed": c.get("measured_failed", 0) + c.get("warmup_failed", 0),
                "measured_correctness_mismatch": c.get("measured_correctness_mismatch", 0),
                "skipped": c.get("skipped", 0),
                "superseded": c.get("superseded", 0),
                "complete": bool(target) and valid >= target,
            })
    return rows


def campaign_status(cid, cdir, recs, camp, progress, rows):
    notes = []
    if os.path.exists(os.path.join(cdir, "ABORTED.txt")):
        notes.append("ABORTED.txt present")
    if os.path.exists(os.path.join(cdir, "INTERRUPTIONS.md")):
        notes.append("INTERRUPTIONS.md present (sandbox restarts)")
    graphs_planned = len(camp.get("graphs") or []) - len(camp.get("safety_skipped") or [])
    done = sum(1 for v in progress.values() if v.get("status") == "done")
    cut = sum(1 for v in progress.values() if v.get("status") not in ("done", None))
    if "ABORTED.txt present" in notes:
        status = "aborted"
    elif graphs_planned and done == graphs_planned:
        status = "complete"
    elif cut and done + cut == graphs_planned:
        status = "finished_with_truncated_graphs"
    else:
        status = "partial_or_running"
    if cid.split("-")[1:2] == ["smoke"]:
        notes.append("harness check; exclude from results")
    return {
        "campaign_id": cid, "status": status, "config_id": camp.get("config_id"),
        "warmup_runs": (camp.get("core_config") or {}).get("warmup_runs"),
        "measured_runs_target": (camp.get("core_config") or {}).get("measured_runs"),
        "graphs_planned": graphs_planned, "graphs_safety_skipped": len(camp.get("safety_skipped") or []),
        "graphs_done": done, "graphs_truncated": cut,
        "raw_records": len(recs),
        "valid_measured_runs": sum(r["valid_measured"] for r in rows),
        "complete_cells": sum(r["complete"] for r in rows), "total_cells": len(rows),
        "failed_or_timeout_runs": sum(r["measured_failed"] + r["measured_timeout"] for r in rows),
        "correctness_mismatches": sum(r["measured_correctness_mismatch"] for r in rows)
                                  + sum(r["gate"] == "mismatch" for r in rows),
        "notes": "; ".join(notes),
    }


def main():
    all_rows, statuses, sources = [], [], []
    for cid in sorted(os.listdir(RAW_DIR)):
        cdir = os.path.join(RAW_DIR, cid)
        if not os.path.isdir(cdir):
            continue
        recs, camp, progress = load_campaign(cdir)
        rows = coverage_rows(cid, recs, camp)
        all_rows += rows
        statuses.append(campaign_status(cid, cdir, recs, camp, progress, rows))
        if os.path.exists(os.path.join(cdir, "runs.jsonl")):
            sources.append((f"datasets/raw_runs/{cid}/runs.jsonl", _sha(os.path.join(cdir, "runs.jsonl")), len(recs)))
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in (("coverage_matrix.csv", all_rows), ("campaign_status.csv", statuses)):
        if data:
            with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    with open(os.path.join(OUT_DIR, "SOURCE.txt"), "w") as f:
        f.write(f"derived_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        for p, h, n in sources:
            f.write(f"{p} sha256={h} records={n}\n")
    for s in statuses:
        print(f"{s['campaign_id']:<42} {s['status']:<30} graphs done {s['graphs_done']}/{s['graphs_planned']} "
              f"valid runs {s['valid_measured_runs']:>4}  complete cells {s['complete_cells']}/{s['total_cells']}  "
              f"fail/timeout {s['failed_or_timeout_runs']}  mismatches {s['correctness_mismatches']}  {s['notes']}")


if __name__ == "__main__":
    main()
