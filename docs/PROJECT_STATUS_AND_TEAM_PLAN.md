# Project status and team plan

**ML-Guided Selection of Graph-Query Execution Strategies in Apache Spark**

Status as of 2026-10-19 (evening, IST). Branch `phase1-data-collection` in `Harshil24Agrawal/big-data`.
Every number below was read from files in the repository at the time of writing; anything that could not
be verified is marked **unknown** or **pending**. Labels used throughout:

| Label | Meaning |
|---|---|
| **Confirmed implementation** | code exists and is exercised by tests or by a run whose records are in the repo |
| **Confirmed measurement** | value read from raw run records that passed the correctness gate |
| **Preliminary** | real measurement, but too few repetitions / one shared machine / incomplete coverage to support a conclusion |
| **Planned** | not started |
| **Unknown** | not verified |

---

## 1. Objective and query

We study how the execution strategy affects a directed two-hop pattern query in Apache Spark:

```
(a)-[e1]->(b); (b)-[e2]->(c)
```

The same logical query can be run by several strategies (native GraphFrames, a plain edge join, 1D salting,
2D hypercube variants), with Spark Adaptive Query Execution (AQE) on or off. The goal is an ML selector that
looks only at graph properties available **before** running the query, predicts each strategy's runtime, and picks
the fastest eligible one. It will be compared against native GraphFrames, a fixed rule, and an offline oracle (the
best measured strategy per graph, an evaluation reference only). **No claim that the selector helps can be made
until Phases 1–4 produce evidence.**

## 2. Architecture and workflow

```
datagen/ (generate + validate graphs) ──► datasets/synthetic/<family>/<graph_id>/edges.parquet + graph.json
datagen/pokec.py (validate real data) ──► datasets/pokec/pokec_edges_parquet + pokec_graph.json
                      │                         │
                      ▼                         ▼
        datasets/manifest/dataset_manifest.csv  (one row per graph, joins on graph_id)
                      │
bench/run_phase1.py (campaign) ──► bench/phase1_worker.py (one Spark app per graph)
   correctness gate → 1 warm-up round → N measured rounds (strategy order shuffled per round)
                      │
                      ▼
datasets/raw_runs/<campaign>/runs.jsonl  (append-only, one line per run)  + run_details.jsonl
                      │
bench/phase1_aggregate.py / phase1_coverage.py ──► datasets/derived/...  (regenerable summaries)
                      │
   Member 2: features ─► Member 3: per-strategy runtime model + selector ─► Member 4: evaluation, Tableau, paper
```

## 3. What is implemented (verified in the repo)

| Component | Path | Status | How it works |
|---|---|---|---|
| Synthetic generator, 5 families | `datagen/generators.py`, `config.py`, `generate.py` | Confirmed (7 tests, 15 subtests) | Seeded NumPy (`default_rng`) per graph; directed, int32 ids in [0, n); self-loops and duplicate edges removed; canonical sort; `graph_id = <family>-<tier>-s<seed>-<8 hex of SHA-256 over all generator inputs>`; re-running produces byte-identical edges (checked via `edges_sha256`) |
| Validation | `datagen/validate.py`, `spark_validate.py` | Confirmed | Degree statistics, Gini, work concentration, family-specific checks declared in advance in `config.FAMILY_CHECKS`; independent Spark re-read of every Parquet file |
| Manifest | `datasets/manifest/dataset_manifest.csv` | Confirmed | 11 rows, 40 columns (ids, seeds, params, file hashes, validation status, measured topology incl. `work_sum`) |
| Pokec validation | `datagen/pokec.py` | Confirmed | SNAP file hash-checked, scanned, cleaned under the same edge policy, written to Parquet, re-read by Spark |
| Strategies | `bench/phase1_strategies.py` (+ existing `salting_1d.py`, `hypergraphframes/`) | Confirmed (Spark equivalence tests) | `native`, `edgejoin`, `salt1d`, `hypercube` (split), `hypercube_vjoin`, `hypercube_unified`, each × AQE on/off |
| Unified hypercube | `bench/hypercube_unified.py` | Confirmed (equals native in tests and in every campaign's correctness gate) | single join on `(dst=src, r, s)`; hubs from the same planner as the split hypercube; non-hubs r = s = 1 |
| Campaign runner | `bench/run_phase1.py` | Confirmed | safety gate on `work_sum`, cheapest graph first, one subprocess per graph, `--resume` skips graphs marked done |
| Worker + correctness gate | `bench/phase1_worker.py` | Confirmed | gate = count + two order-independent checksums vs native and count vs `work_sum`; failed gate ⇒ no timed runs; per-run timeout via Spark job-group cancel |
| Metrics | `bench/phase1_metrics.py` | Confirmed (fixed today, see §7) | Spark REST API stage/task metrics; AQE decisions read from the executed plan |
| Aggregation | `bench/phase1_aggregate.py` | Confirmed (acceptance rule now unit-tested) | see §5 acceptance rule |
| Coverage matrix | `bench/phase1_coverage.py` | Confirmed (new today) | per (campaign, graph, strategy) counts from raw records only |
| `preprocessing.py` (repo root) | | **Not feature engineering.** | 43-line original Pokec CSV→Parquet converter with no cleaning or checks; superseded by `datagen/pokec.py`. Kept unchanged (team code). Member 2 should **not** build on it |
| Feature extraction | — | **Planned (Member 2)** | Graph-level statistics already computed per graph live in `graph.json`/manifest (`datagen/validate.py::degree_stats`) and can be reused |
| ML model / selector | — | **Planned (Member 3)** | none in repo |
| Tableau / paper figures | — | **Planned (Member 4)** | none in repo |

## 4. What has been generated and validated

### Synthetic graphs: 11 of 60 planned (pilot set)

Planned: 5 families × 3 tiers (small 100k V / 700k E, medium 500k / 3.5M, large 1M / 7M; average degree 7) × 4 seeds
(101, 202, 303, 404) = 60. **Present on disk and in the manifest: 11. Missing: 49.**

| graph_id | |V| | |E| (target) | work_sum | family checks | Spark re-read |
|---|---:|---:|---:|---|---|
| uniform-small-s101-f3568c8f | 100,000 | 699,974 (700,000) | 4,899,644 | PASS | PASS |
| out_heavy-small-s101-88c36855 | 100,000 | 700,000 | 4,968,648 | PASS | PASS |
| in_heavy-small-s101-addf9e1e | 100,000 | 700,000 | 4,968,986 | PASS | PASS |
| in_heavy-medium-s101-9829e80e | 500,000 | 3,500,000 | 24,250,876 | PASS | PASS |
| multi_hub-small-s101-cea55efc | 100,000 | 699,995 (700,000) | 47,151,493 | PASS | PASS |
| uniform-large-s101-57ef5eb7 | 1,000,000 | 6,999,969 (7,000,000) | 48,999,576 | PASS | PASS |
| powerlaw_balanced-small-s101-aa30b13b | 100,000 | 700,000 | 49,928,247 | PASS | PASS |
| out_heavy-large-s101-2c9ecb7d | 1,000,000 | 7,000,000 | 50,176,337 | PASS | PASS |
| powerlaw_balanced-small-s202-206df433 | 100,000 | 700,000 | 50,437,159 | PASS | PASS |
| multi_hub-medium-s101-7241a078 | 500,000 | 3,500,000 | 105,015,224 | PASS | PASS |
| powerlaw_balanced-medium-s101-2b210362 | 500,000 | 3,500,000 | 535,117,762 | PASS | PASS |

`work_sum = Σ_v d_in(v)·d_out(v)` is the exact number of 2-hop paths under our simple-digraph policy (it includes
a = c 2-cycles). It is a workload indicator, not a runtime estimate; the benchmark's correctness gate confirmed
measured output count == `work_sum` on every graph benchmarked so far.

**Discrepancy found today:** the docs said every graph has exactly `m_target` edges. `uniform` (no top-up by
design) and `multi_hub` (hub/background overlap removed after top-up) fall short by < 0.01 %. Actual counts were
always recorded correctly; the documentation is now corrected. Feasibility for the missing 49 (estimated from the
pilot): ≈ 1.4 GB Parquet, ≈ 7–10 min generation CPU, peak RSS ≈ 0.7 GB — feasible on this machine.

### Pokec (real data) — validated, not benchmarked

Confirmed from `datasets/pokec/pokec_graph.json` and `pokec_prep_receipt.json`, raw file SHA-256 re-checked today:
30,622,564 edges, 1,632,803 active vertices (ids 1..1,632,803), 0 null/unparseable ids, 0 self-loops, 0 duplicates,
counts equal SNAP's published figures; Spark re-read PASS; `work_sum` = 1,780,294,899 (equals the earlier
repository report). Note: the prep receipt was *recovered* from the logged output of a run that crashed after
writing; the rerun re-checked the written data. **HDFS: not configured; no HDFS read has been performed.**

## 5. Benchmark campaigns

Environment for all campaigns: cloud sandbox, Intel Xeon 2.1 GHz, **2 logical cores, 7.84 GB RAM**, Spark 3.5.5,
GraphFrames 0.8.3 (local JAR), Java 21, `local[*]`, 200 shuffle partitions, broadcast joins disabled
(`autoBroadcastJoinThreshold=-1`), driver memory 4 GB. AQE-on condition pins skew join on, skewed-partition
threshold 256 MB, factor 5, coalescing on, advisory 64 MB. These results must not be mixed with the team's earlier
Windows/JDK 17 results.

Pilot strategies (7): `native__aqe_on`, `native__aqe_off`, `edgejoin__aqe_on`, `edgejoin__aqe_off`,
`salt1d__aqe_off`, `hypercube__aqe_off`, `hypercube_unified__aqe_off` (`hypercube_vjoin` excluded from the pilot).

**Acceptance rule (implemented in `phase1_aggregate.py`, verified by a unit test today):** a timing is valid only
if `run_kind == "measured"` and `status == "ok"` and `correctness_status == "match"`, and it belongs to the latest
Spark application of that graph in the campaign (earlier interrupted attempts are "superseded"). A graph's
"fastest strategy" is now flagged `label_eligible` only if *all* planned strategies have at least the campaign's
target number of valid runs.

Counts from `bench/phase1_coverage.py` (raw records only):

| Campaign | What it was | Status | Valid measured runs | Complete cells* | Failures / timeouts / mismatches |
|---|---|---|---:|---:|---|
| `20261019T132121Z-smoke-8f4873` | harness check, 2 tiny graphs | complete | 14 | 14/14 | 0 / 0 / 0 — exclude from results |
| `20261019T135854Z-pilot-8f41eb` | full protocol, 11 graphs | **aborted** after ~6 min (would take many hours) | 3 | 0/7 | 0 / 0 / 0 |
| `20261019T140509Z-pilot-reduced-e1367c` | 7 graphs, 1 warm-up + ≤3 measured, 540 s per-graph cap | finished; **every graph hit the cap** | 64 | 5/49 | 0 / 0 / 0 |
| `20261019T163326Z-pilot-full-5d1324` | 10 graphs (1 safety-skipped), 1 warm-up + 5 measured | **running** (resumed twice after sandbox restarts) | 52 at time of writing | 7/21 | 0 / 0 / 0 |

\*cell = (graph, strategy); complete = valid runs ≥ campaign target.

### 6. Results genuinely supported by artifacts

- **Correctness (confirmed):** every strategy that ran passed the correctness gate on every graph — output count
  equal to `work_sum` and identical checksums to native GraphFrames. No mismatch, failure or timeout has been
  recorded in any campaign.
- **One complete graph (preliminary):** `uniform-small-s101-f3568c8f` in `pilot-full`, 5 valid runs per strategy:

  | strategy | median total_s | IQR | median query_s | min–max total_s |
  |---|---:|---:|---:|---:|
  | edgejoin__aqe_on | 7.74 | 0.67 | 7.74 | 5.40–15.98 |
  | native__aqe_on | 9.65 | 1.80 | 9.65 | 8.31–11.19 |
  | native__aqe_off | 10.15 | 2.64 | 10.15 | 9.60–22.12 |
  | edgejoin__aqe_off | 14.07 | 7.04 | 14.07 | 7.17–17.98 |
  | hypercube__aqe_off | 19.96 | 8.39 | 9.86 | 13.71–39.73 |
  | salt1d__aqe_off | 27.60 | 12.91 | 12.59 | 20.13–37.68 |
  | hypercube_unified__aqe_off | 31.93 | 20.69 | 17.51 | 17.84–42.39 |

  Spreads are wide (up to ~3× between min and max for one strategy) because the shared 2-core sandbox is noisy.
  On this small, hub-free graph the partitioning strategies pay a planning/profiling cost (`total_s` vs `query_s`)
  with nothing to rebalance. This is one graph on one machine: **not a general finding and not an ML label.**
- **AQE (confirmed, narrow):** in every AQE-on run so far AQE only *coalesced* partitions (4–6 read partitions);
  the final adaptive plan never contained a skew split (`plan_skew_split = false`, and every stored AQEShuffleRead
  node lacks skew metrics). Skew handling is therefore **untested**, not shown ineffective — none of the graphs run
  so far produced a partition above the 256 MB threshold.

### 7. Failed, incomplete or inconclusive work, and faults fixed today

- Aborted full-protocol pilot; reduced pilot where every graph was truncated by the time cap (0–3 runs per cell).
- `uniform-large` effectively not benchmarked (1 of 7 strategies got 1 run).
- Sandbox VM restarted twice during `pilot-full` and killed the benchmark; interrupted attempts are preserved and
  documented in the campaign's `INTERRUPTIONS.md`; the campaign was resumed (`--resume`).
- `powerlaw_balanced-medium` (work_sum 535M) safety-skipped in `pilot-full` (limit 110M); Pokec (1.78B) not run.
- **Faults found and fixed today** (raw records untouched):
  1. AQE metric reader reported "unknown" (null) for skew partitions when Spark simply had none. Spark 3.5 only
     registers skew metrics on reads that have a skewed partition, so a readable node without them means 0. Fixed in
     `aggregate_aqe_nodes()` (unit-tested). Records written before the fix keep null; use `plan_skew_split` for them.
  2. Aggregator turned null into `False` in `aqe_skew_split_any_run`; now null stays unknown, and a new
     `plan_skew_split_any_run` column carries the real evidence.
  3. Aggregator named a "fastest strategy" even when some strategies had no runs (e.g. `uniform-large`). Added
     `label_eligible`, `strategies_expected`, `min_measured_runs_per_strategy`.
  4. Correctness gate did not release cached hypercube tables when a gate run failed. Fixed.
  5. Documentation claimed exact `m_target` edges for every family (see §4).
- Evidence-quality caveat: short audit scripts (a few seconds of Python) ran alongside the benchmark while it was
  measuring `out_heavy-small` in `pilot-full`; treat that graph's timings with extra caution.

## 8. Phase 1 status and exit criteria

**Phase 1 is partially complete. It is not done.**

| Exit criterion | Status |
|---|---|
| Required datasets generated or documented unavailable | 11/60 synthetic generated + validated; 49 pending (feasible); Pokec validated |
| Dataset integrity demonstrated | Yes for the 11 + Pokec |
| Query correctness demonstrated | Yes for every graph benchmarked (7 graphs + smoke) |
| Benchmark matrix and exclusions documented | Coverage matrix exists; matrix far from complete (1 graph fully measured with 5 runs) |
| Labels only from valid measured runs | Yes — enforced by aggregator, unit-tested, `label_eligible` flag |
| Raw results reproduce derived tables | Yes — `phase1_aggregate.py` / `phase1_coverage.py` regenerate everything; `SOURCE.txt` stores raw hashes |
| Remaining gaps explicit | This document + `docs/PHASE1_REPORT.md` |

## 9. Team responsibilities

| Member | Role | Owns |
|---|---|---|
| 1 — Harshil | datasets, correctness, benchmarking | generator, manifest, Pokec, campaigns, raw records, coverage, AQE diagnostics |
| 2 | preprocessing & feature engineering | feature definitions, preprocessing script, graph-grouped splits, model-ready table |
| 3 | ML training & validation | label schema (with 1, 2), per-strategy log-runtime models, selector + native fallback, regret vs oracle |
| 4 | Tableau & paper | coverage/correctness tables, plots from validated data, data dictionary, report structure |

## 10. What each member can start now

- **Member 2:** build feature extraction on the 11 pilot graphs (`edges.parquet` + `graph.json`). Pre-execution
  candidates already computed per graph in the manifest: |V|, |E|, degree means/maxima/CV/Gini, `work_sum`,
  `work_max_share`, `num_work_hubs`, `work_in_hubs_share`, `log_degree_corr`. Do **not** use `query_s`, shuffle bytes,
  task durations or any `runs.jsonl` field as an input feature. Record extraction time per graph. Use
  `docs/PHASE1_DATA_COLLECTION.md` §5 "Handoff reference for Member 2" for the schema and an example record.
- **Member 3:** build the training/selection pipeline on mock data or on `strategy_summary.csv` rows; filter labels
  by `label_eligible == True` (currently only `uniform-small` in `pilot-full`). Split by `graph_id` (and keep seeds of
  one family/tier together if testing generalisation). Do not train a final model yet.
- **Member 4:** report skeleton, plot scripts reading `datasets/derived/coverage/*.csv` and
  `datasets/derived/<campaign>/strategy_summary.csv`; mark every current figure **preliminary**.

## 11. Dependencies and timeline

| Workstream | Depends on | Can proceed now? |
|---|---|---|
| Phase 1 benchmarks | machine time | yes (slow here: ~20 min per small graph with 5 runs × 7 strategies) |
| Phase 1 remaining 49 graphs | none | yes (after current campaign, to avoid disturbing timings) |
| Phase 2 features | pilot graphs + agreed schema | yes |
| Phase 3 pipeline scaffolding | feature/label schema | yes (mock / eligible data only) |
| Phase 3 final training | label-eligible data for many graphs | no |
| Phase 4 evaluation scripts | metrics/baselines agreed | yes |
| Phase 4/5 final results | Phases 1–3 | later |

Realistic Phase 1 completion requires a larger machine: at the measured pace (~20 min per small graph, much more
for medium/large) the 60-graph × 7-strategy × 6-round matrix is several days on this 2-core sandbox.

## 12. Paths and commands

```bash
# environment (sandbox)
export GRAPHFRAMES_JAR=$PWD/jars/graphframes-0.8.3-spark3.5-s_2.12.jar PYSPARK_PYTHON=$(which python)

python -m datagen.generate --plan pilot          # or --plan full (generates only missing graphs; existing ids are never overwritten)
python -m datagen.generate --plan pilot --verify-existing   # regenerate in memory, compare edges_sha256 (no writes)
python -m datagen.spark_validate                 # Spark re-read validation, updates manifest
python -m datagen.pokec                          # Pokec validation (reuses prep receipt)

python bench/run_phase1.py --name full60 --dry-run            # inspect the safety gate first
python bench/run_phase1.py --name full60 --warmup 1 --runs 5 --graph-timeout-s 6000
python bench/run_phase1.py --resume <campaign_id>             # continue after an interruption
python bench/phase1_aggregate.py <campaign_id>                # per-campaign summaries
python bench/phase1_coverage.py                               # coverage matrix over all campaigns

python -m pytest datagen/tests bench/test_phase1.py -q        # Phase 1 tests
python -m pytest hypergraphframes/tests -q                    # existing strategy equivalence tests
```

Key files: `datasets/manifest/dataset_manifest.csv`, `datasets/raw_runs/<campaign>/runs.jsonl`,
`datasets/raw_runs/INDEX.csv`, `datasets/derived/coverage/coverage_matrix.csv`,
`datasets/derived/<campaign>/strategy_summary.csv`, `graph_summary.csv`, `docs/PHASE1_DATA_COLLECTION.md`,
`docs/PHASE1_REPORT.md`. Parquet, raw downloads and the JAR are gitignored (regenerate / re-download).

## 13. Risks, blockers, constraints

- **Compute:** 2 cores, 7.8 GB; noisy timings (wide IQR). Biggest blocker for Phase 1 completion.
- **Sandbox restarts** kill long campaigns (twice today). Mitigated by `--resume`, but each restart loses the graph
  in progress.
- **AQE skew handling untested** — needs graphs that create > 256 MB skewed partitions (or a documented lower
  threshold as a separate, labelled experiment).
- **Small label set:** even the full plan gives 60 graphs; graph-grouped evaluation will have wide uncertainty.
- **Schema confirmation** with Members 2–4 still pending.
- **Generated data lives in an ephemeral workspace**; Parquet is regenerable from seeds, raw run records are in git.

## 14. Next actions (priority order)

1. Member 1: let `pilot-full` finish the cheapest graphs (resume after any restart); stop before graphs that cannot
   finish, recording them as not run.
2. Member 1: generate the 49 missing graphs (`--plan full`) and Spark-validate them; update the manifest.
3. Member 1: secure a larger machine for the full matrix; run `full60` with one fixed config.
4. Member 1: design a skew experiment (graphs whose hub partition exceeds 256 MB at 200 partitions), alongside
   compute-skewed graphs that do not trigger AQE.
5. Members 1–3: confirm feature and label schemas (`graph_id` join; `label_eligible` filter).
6. Members 2–4: build pipelines on pilot data, clearly marked preliminary.

## 15. Definition of done

| Phase | Done when |
|---|---|
| 1 | 60 graphs generated + validated (or exclusions documented); every planned (graph, strategy) cell has ≥ 5 valid measured runs from one config/machine, or is documented as excluded with a reason; AQE skew case exercised; coverage matrix and raw→derived reproducibility verified |
| 2 | feature table regenerable from graphs, documented, no post-execution leakage, joins on `graph_id` |
| 3 | reproducible training on label-eligible data, graph-grouped held-out test, selector with native fallback, compared to native / rule / oracle with regret and overhead |
| 4 | frozen selector evaluated on untouched test graphs including feature-extraction and inference overhead |
| 5 | every number traceable to an artifact; figures reproducible; limitations and failed experiments reported |
