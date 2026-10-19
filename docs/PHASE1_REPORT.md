# Phase 1 report — pilot status, failures, constraints, next steps

Branch: `phase1-data-collection`. Companion to `docs/PHASE1_DATA_COLLECTION.md` (how-to and data contract).
Written 2026-10-19 under a hard same-day deadline; everything below is what was actually executed.

## 1. What is complete

| Item | Status | Evidence |
|---|---|---|
| Repo/version audit | done | README section "Setup"; Spark 3.5.5, GraphFrames 0.8.3 (local JAR), Python 3.11, Java 21 (repo docs say JDK 17) |
| Generator, 5 directed families, seeds, saved params | done | `datagen/`, `datasets/synthetic/<family>/<graph_id>/graph.json` |
| Edge policy (self-loops/duplicates removed, ids in [0,n)) | done, documented + tested | `datagen/config.py::EDGE_POLICY` |
| 11-graph pilot Parquet (5 families, 3 tiers) | done, all PASS family checks, determinism verified, Spark re-read validated | `datasets/manifest/dataset_manifest.csv` |
| Pokec reuse + validation | done, PASS (30,622,564 edges, 1,632,803 vertices, 0 nulls/dupes/self-loops, matches SNAP counts) | `datasets/pokec/pokec_graph.json` |
| HDFS read | **not configured in this sandbox** — not claimed; `hdfs_check` reports NOT_CONFIGURED | `datagen/pokec.py` |
| Benchmark runner (existing strategies + unified hypercube) | done | `bench/run_phase1.py`, `phase1_worker.py`, `phase1_strategies.py`, `phase1_metrics.py` |
| Correctness gate (count + 2 order-independent checksums vs native) | done, **all 7 strategies matched on every graph that got a gate** | `run_kind == "correctness"` records |
| Work-sum safety gate | done (limit 600M); all pilot graphs under it | dry-run output |
| Immutable raw records + derived summaries kept separate | done | `datasets/raw_runs/`, `datasets/derived/` |
| Tests | 21 passed (`datagen/tests`, `bench/test_phase1.py`); existing repo suites passed earlier | |

## 2. Pilot benchmark — what was and was NOT run

**Aborted full-protocol campaign** `20261019T135854Z-pilot-8f41eb` (11 graphs, 1 warm-up + 5 measured, 7 strategies). It was
stopped by the operator after ~6 minutes because on this 2 vCPU / 7.8 GB sandbox the projected time was many hours.
Its 17 raw records are kept (see `ABORTED.txt` in the campaign directory) and are **not** used in any summary.

**Executed campaign** `20261019T140509Z-pilot-reduced-e1367c`: reduced protocol, config `e133b2f5a9`.

* 7 of the 11 pilot graphs (the 5 small-tier graphs, `in_heavy-medium`, `uniform-large`).
  **Not run:** `out_heavy-large`, `powerlaw_balanced-small-s202`, `multi_hub-medium`, `powerlaw_balanced-medium` (largest, work_sum 535M). Their absence is a time decision, not a failure.
* 1 warm-up + **up to 3** measured runs (target was 5), `--graph-timeout-s 540`, `--run-timeout-s 240`.
* **Every graph hit the 540 s cap**, so the number of measured runs per (graph, strategy) is 0–3 and uneven. The runner
  interleaves strategies per round, so counts differ by at most about one round within a graph. Exact counts are in
  `n_measured_ok` in `strategy_summary.csv`. No run failed or timed out individually (`n_measured_failed = n_measured_timeout = 0`).
* `uniform-large` (1M vertices, 5M edges): load + correctness gate consumed most of the cap; only `hypercube__aqe_off`
  produced one measured run. Treat this graph as **not benchmarked**.
* `in_heavy-medium`: 1 measured run per strategy (`hypercube__aqe_off` none).

### Interpretation limits (important for Members 2–4)

* With n = 1–3 per cell, medians/IQRs are **not statistically meaningful**. `graph_summary.csv` names a "fastest" strategy,
  but those are preliminary and must not be used as ML labels. They are also not comparable to the earlier Windows/JDK 17
  repo results (different machine, Java, core count).
* Observed qualitatively (small graphs, ~5–50M paths): `edgejoin` was fastest by median on most graphs; split/unified
  hypercube and salt1d paid a visible planning/hub-detection overhead in `total_s` (compare `total_s` vs `query_s`).
  Hubs were absent or few at these scales (work threshold 1,000,000), so hub-handling strategies had little to fix.
  The AQE skew-split path did not trigger on any measured run (`aqe_skew_split_any_run` = False on all 42 populated rows); AQE only coalesced
  partitions (`aqe_partitions` = 4–6). A skew-driven regime needs the larger/hub-heavy graphs that were not run.
* `aqe_plan_metrics` populates `aqe_partitions`; `aqe_skewed_partitions` / `aqe_skew_splits` stayed empty because no
  skewed read occurred. This is a known unverified path under real skew.

## 3. Failures and constraints encountered

* Maven / spark-packages / SNAP blocked by the sandbox proxy → JAR and Pokec fetched through the user's browser pane, SHA-256 verified; runner gained a `GRAPHFRAMES_JAR` switch (offline-capable).
* Pokec prep crashed once on a second SparkSession; restructured to one session, with a labelled "recovered" receipt from logged output.
* 3 pilot graphs initially failed family checks (hub-share check used a fixed hub count); check corrected to use the planted hub count, re-validated via `--revalidate` with prior result preserved in `validation_history`. Edges unchanged.
* Machine: Intel Xeon 2.1 GHz, 2 threads, 7.84 GB; `local[*]`, 200 shuffle partitions, driver 4g. One run takes ~6–30 s even on 5M-path graphs.
* Pokec was **not benchmarked**: work_sum 1.78B exceeds the 600M gate and is infeasible here.
* `hypercube_vjoin` excluded from the pilot (kept in the codebase).

## 4. Remaining work to scale to the 60-graph collection

1. Run on a bigger machine (≥16 cores, ≥32 GB) or raise per-graph time; keep config/`machine_id` consistent across the whole campaign.
2. `python -m datagen.generate --plan full` (60 graphs), `python -m datagen.spark_validate`, then `bench/run_phase1.py --name full60 --dry-run` to inspect the safety gate before running.
3. Use `--runs 5 --warmup 1` with a generous `--graph-timeout-s` (≥ 3600 for medium/large). Re-run the 4 skipped pilot graphs and `uniform-large` first (`--resume` is supported).
4. Raise `--max-work-sum` deliberately (documented) for Pokec and the large heavy-hub graphs, or resize and register them under new graph IDs.
5. Increase hub density in the in/out-heavy families at small/medium tiers — pilot graphs have only 0–2 work hubs, so they are weak tests of skew handling.
6. Confirm schemas in the README data contract with Member 2 (features), Member 3 (ML labels), Member 4 (Tableau). They were proposed, not yet confirmed.
7. Verify AQE skew-split capture on a graph that actually triggers it.

## 5. Key paths

* Raw: `datasets/raw_runs/20261019T140509Z-pilot-reduced-e1367c/{runs.jsonl,run_details.jsonl,logs/,campaign.json}`
* Derived: `datasets/derived/20261019T140509Z-pilot-reduced-e1367c/{strategy_summary.csv,graph_summary.csv,SOURCE.txt}`
* Manifest: `datasets/manifest/dataset_manifest.csv`
