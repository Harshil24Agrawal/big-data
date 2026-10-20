# Project status and team plan

**ML-Guided Selection of Graph-Query Execution Strategies in Apache Spark**

Status as of 2026-10-20, ~11:00 IST. Final update of the overnight run: 60/60 graphs generated; **31 of 60 graphs benchmarked** (1,085 valid runs, 4 campaigns, 1 machine); Phase 2 feature table delivered; first selector prototype run (preliminary). Branch `phase1-data-collection` in `Harshil24Agrawal/big-data`.

> ## UPDATE 2026-10-20, ~23:45 IST — all phases complete (supersedes the status line and phase tables below where they differ)
>
> * **Phase 1 (data) finished at 62 benchmarked graphs:** 31 original + 31 hub-sweep graphs (one machine, same Spark settings; 5/3/2 timed runs per strategy, recorded in a `protocol` column). 306 timed sweep runs, 0 failures, 0 timeouts, 0 wrong results. 29 of the 60 original graphs remain un-benchmarked (documented exclusion).
> * **Key finding:** edge-join is the best default on this 2-core machine (faster than native on 56/62 graphs, median 1.38×). Hub-splitting (unified hypercube) wins only when **one** vertex carries ≳ 1e8 two-hop paths (6 graphs, up to 2× faster than edge-join, 3.3× faster than native). With several equal hubs it never wins.
> * **Phase 3 (models) done in `notebooks/selector.ipynb`** (executed, 300 trees): best classifier 85.5 % leave-one-graph-out vs 80.6 % majority baseline; runtime-regression selector mean regret 4.6 % vs 8.3 % (best fixed) vs 0 % (oracle); no gain over the fixed policy on unseen graph families (leave-one-group-out). Feature-extraction cost is not measured.
> * **Phases 4–5 done:** charts in `datasets/derived/selector_v2/`; paper-style report `docs/FINAL_REPORT.md` (method, data, results, limitations, related work, references); explanation guide `docs/PROJECT_BRIEFING.md`.
> * Nothing is left for other team members except review/presentation; do not re-run benchmarks on another machine into the same table (the builder refuses mixed machines).
> * Git: everything is on `mine` (Harshil24Agrawal/big-data, branch `phase1-data-collection`); nothing was pushed to `origin` (Rayirth556/hypergraph).

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

### Synthetic graphs: 60 of 60 planned — generated and validated

5 families × 3 tiers (small 100k V / ≈700k E, medium 500k / ≈3.5M, large 1M / ≈7M; average degree 7) × 4 seeds
(101, 202, 303, 404) = **60 graphs. All 60 are in `datasets/manifest/dataset_manifest.csv`, all 60 pass the
family-specific checks, and all 60 pass the independent Spark re-read validation** (`validation_status` and
`spark_validation_status` both PASS; ~1.3 GB of Parquet on disk). The original 11 pilot graphs were not regenerated
or modified (same `graph_id`, same hashes); the other 49 were generated afterwards with the same seeds and settings
(≈7 min generation, peak memory < 1 GB). Parquet is gitignored; `graph.json` and the manifest are tracked.

Workload range (`work_sum = Σ_v d_in(v)·d_out(v)`, exact two-hop path count under our simple-digraph policy, includes
a = c 2-cycles; a workload indicator, not a runtime estimate): 4.5 M to 1.48 B. By tier:

| family | small | medium | large |
|---|---|---|---|
| uniform | ≈4.9 M | ≈24.5 M | ≈49 M |
| in_heavy / out_heavy | ≈4.5–5.0 M | ≈24–25 M | ≈46–50 M |
| multi_hub | ≈42–47 M | ≈99–105 M | ≈203–207 M |
| powerlaw_balanced | ≈50 M | ≈535–538 M | **≈1.47–1.48 B** |

12 of 60 graphs exceed 110 M; 4 (powerlaw_balanced large, all seeds) exceed the 600 M default safety limit and would be
**skipped** by the runner unless the limit is raised deliberately. The 11 pilot graphs (benchmarked or planned) are:

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


**Discrepancy found today:** the docs said every graph has exactly `m_target` edges. `uniform` (no top-up by
design) and `multi_hub` (hub/background overlap removed after top-up) fall short by < 0.01 %. Actual counts were always
recorded correctly in the manifest; the documentation is now corrected.

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
| `20261019T163326Z-pilot-full-5d1324` | 10 graphs (1 safety-skipped), 1 warm-up + 5 measured | **complete: 10/10 graphs done**; interrupted twice by sandbox restarts and resumed | **350** (= 10 × 7 × 5) | **70/77** (the 7 open cells are the safety-skipped graph) | 0 / 0 / 0 |
| `20261019T205726Z-pilot-hubs-67a899` | 8 hub-heavy graphs (multi_hub small ×3, medium ×3; powerlaw small ×2), same config id `ab41b3ff27` as pilot-full | **complete: 8/8 graphs done** (no restart) | **280** (= 8 × 7 × 5) | **56/56** | 0 / 0 / 0 |
| `20261019T235430Z-pilot-ext1-7aa307` | 2 large multi_hub graphs (work_sum ≈205M), `max_work_sum` 250M, per-run limit 900 s | **complete: 2/2**; sandbox restarted once during the second graph, resumed (`INTERRUPTIONS.md`) | **70** (= 2 × 7 × 5) | **14/14** | 0 / 0 / 0 |
| `20261020T011920Z-pilot-ext2-777331` | 12 planned core graphs (small/medium/large of in_heavy, out_heavy, uniform) | **11/12 done**; stopped deliberately during the 12th (`in_heavy-large-s101`) at the hand-over deadline (`NOTES.md`) | **385** (= 11 × 7 × 5) | **77/77** done graphs | 0 / 0 / 0 |
| **Total used for labels** | 31 graphs from `pilot-full`, `pilot-hubs`, `pilot-ext1`, `pilot-ext2` | — | **1,085** (= 31 × 7 × 5) | — | 0 / 0 / 0 |

\*cell = (graph, strategy); complete = valid runs ≥ campaign target.

**Configuration identity.** `pilot-full`, `pilot-hubs` and `pilot-ext2` carry `config_id ab41b3ff27`. `pilot-ext1` carries `5e57fecdd6`
only because its per-run safety limit was 900 s instead of 600 s (the larger graphs needed it); no run in any campaign timed out. The measurement
settings (master, driver memory, 200 shuffle/input partitions, broadcast off, warm-up 1, measured 5, work threshold, P cells, AQE settings, strategies,
machine) are identical. `ml/build_training_table.py` therefore compares a *measurement configuration id* that leaves out the safety-only limits
(`run_timeout_s`, `graph_timeout_s`, `max_work_sum`), refuses to combine anything else, and keeps the recorded `config_id` as a column.

### 6. Results genuinely supported by artifacts

The first table below and the 10-graph paragraph come from `datasets/derived/20261019T163326Z-pilot-full-5d1324/`; the
hub-heavy paragraph from `…pilot-hubs-67a899/`; the combined view from `datasets/derived/training/training_table_eligible.csv`
(all four campaigns, 31 graphs). Everything is regenerable from the raw `runs.jsonl` files. Environment: one shared 2-core sandbox, one
measurement configuration, 5 valid measured runs per cell. **31 graphs × 7 strategies × 5 runs is real data, but 31 graphs are
too few, too homogeneous and too one-sided to train a selector that can be claimed to generalise.**

- **Correctness (confirmed):** across all four campaigns every strategy that ran passed the correctness gate on every
  graph (output count equal to `work_sum`, checksums identical to native GraphFrames). 0 mismatches, 0 failures,
  0 timeouts.
- **Valid runs (confirmed):** 350 measured runs in `pilot-full` = 10 graphs × 7 strategies × 5; 35 records from the two
  interrupted attempts are counted as superseded and excluded.
- **Which strategy has the lowest median `total_s` (preliminary, 10 graphs, 1 machine):** an edge-join strategy on
  **all 10 graphs** — `edgejoin__aqe_on` on 6, `edgejoin__aqe_off` on 4. By execution-only time (`query_s`) the lowest is
  also edge-join on 9 graphs and `hypercube__aqe_off` on 1 (`uniform-large`).

  | graph | lowest median total_s | margin over runner-up |
  |---|---|---:|
  | in_heavy-small-s101 | edgejoin__aqe_on | 51.1 % |
  | out_heavy-small-s101 | edgejoin__aqe_off | 34.6 % |
  | in_heavy-medium-s101 | edgejoin__aqe_on | 35.3 % |
  | multi_hub-small-s101 | edgejoin__aqe_on | 34.2 % |
  | out_heavy-large-s101 | edgejoin__aqe_off | 29.4 % |
  | uniform-small-s101 | edgejoin__aqe_on | 24.8 % |
  | powerlaw_balanced-small-s202 | edgejoin__aqe_on | 13.5 % |
  | multi_hub-medium-s101 | edgejoin__aqe_off | 9.9 % |
  | uniform-large-s101 | edgejoin__aqe_off | 3.6 % |
  | powerlaw_balanced-small-s101 | edgejoin__aqe_on | 1.2 % |

  Two consequences: (1) the choice between AQE on and off *within* edge-join is often inside the run-to-run noise
  (margins of 1–4 % on two graphs); (2) **a classifier trained on this label would see a single class.** The hub-heavy
  graphs where hypercube or salting could win are the ones not yet benchmarked at scale (large multi_hub, medium and
  large powerlaw). Do not conclude that the hypercube strategies are worse in general: here they pay a profiling cost
  and there is little skew to remove.
**Hub-heavy campaign (`pilot-hubs`, 8 graphs, from `datasets/derived/20261019T205726Z-pilot-hubs-67a899/`):**
edge-join again has the lowest median `total_s` on all 8 graphs (`edgejoin__aqe_on` on 5, `edgejoin__aqe_off` on 3), with
margins over the runner-up of 25.2 %, 28.5 %, 13.2 %, 13.7 %, 23.6 %, 9.4 %, 4.6 % and 3.9 %. Across the 18 graphs measured so far
(`pilot-full` + `pilot-hubs`), the best edge-join strategy is **1.25–2.11× faster than the best non-edge-join strategy on 17
graphs and 1.01× on one** (`powerlaw_balanced-small-s101`), and 1.14–1.57× faster than native GraphFrames with AQE on. So far the salting and
hypercube variants never win on `total_s`: they pay a profiling pass and, at this size on 2 cores with 200 shuffle partitions, shuffle volume does not
dominate. No final adaptive plan in either campaign contains a skew split (`plan_skew_split_any_run = False` for every AQE-on strategy).
Reading: hub-concentrated *work* alone did not make hypercube/salting pay off here. This does not show they are worse on larger graphs, other machines
or truly byte-skewed partitions, which these graphs never produced.
**Combined view, 31 graphs (`training_table_eligible.csv`):**

| Quantity | Value |
|---|---|
| graphs by family | multi_hub 10, out_heavy 6, uniform 6, in_heavy 5, powerlaw_balanced 4 (all small); no powerlaw medium/large |
| lowest median `total_s` | `edgejoin__aqe_on` 16, `edgejoin__aqe_off` 14, `native__aqe_on` 1 |
| best edge-join vs best non-edge-join | non-edge-join is slower on 30 graphs (median 1.39×, up to 2.11×); on 1 graph (`uniform-small-s202`) native with AQE on is 16 % faster, in a cell with wide IQRs (5–7 s) |
| best of salt1d / hypercube / hypercube_unified vs best edge-join | **never faster**: 1.52–2.63× slower (median 2.09×) |
| native (AQE on) vs best edge-join | edge-join faster by a median 1.45× (max 2.11×) |
| labels with a gap larger than both strategies' IQR (`label_clear`) | only 7 of 31; 6 graphs have a margin under 5 % |
| skew splits in final adaptive plans | none in any campaign |

Reading: on this machine and at these sizes, edge-join variants dominate and the salting/hypercube variants are consistently worse because they
pay a profiling pass and the data never produced a byte-skewed partition. This is a statement about 2 cores, 200 partitions, 5–207 M work and
these generators; it is not evidence about larger clusters or true byte skew.

- **AQE (confirmed, narrow):** AQE only *coalesced* partitions in the AQE-on runs; the final adaptive plan never
  contained a skew split (`plan_skew_split = false`). No graph benchmarked so far produced a reduce partition near the
  256 MB threshold. Skew handling is therefore **untested**, not shown ineffective.
- **Noise (confirmed):** within one (graph, strategy) cell the max/min of `total_s` over 5 runs has median 1.7× and reaches 3.0× (70 cells), e.g. on
  `uniform-small-s101` `salt1d__aqe_off` spans 20.1–37.7 s. Medians and IQR, not single runs, must be used.

### 7. Failed, incomplete or inconclusive work, and faults fixed today

- Aborted full-protocol pilot; reduced pilot where every graph was truncated by the time cap (0–3 runs per cell). Both are kept as raw records and are superseded by `pilot-full`; neither is usable for labels.
- Sandbox VM restarted twice during `pilot-full` and killed the benchmark; interrupted attempts are preserved and
  documented in the campaign's `INTERRUPTIONS.md`; the campaign was resumed (`--resume`).
- `powerlaw_balanced-medium` (work_sum 535M) safety-skipped in `pilot-full` (limit 110M): 7 of 77 cells have no runs. 4 `powerlaw_balanced-large` graphs (≈1.48B) exceed even the 600M default gate; Pokec (1.78B) not run. About 50 of the 60 graphs are not benchmarked at all.
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
- `pilot-ext1`: one sandbox restart during its second graph (resumed); `pilot-ext2`: stopped at 11 of 12 graphs by design (see §5). `in_heavy-large-s101` has partial, non-eligible records only.
- 29 graphs remain un-benchmarked: 4 `powerlaw_balanced-large` (≈1.48B work, above the 600M default safety gate), 4 `powerlaw_balanced-medium` (≈535M), 2 `multi_hub-large` (≈205M) and 19 core graphs (in/out_heavy and uniform, mostly seeds 303/404 and some 202); the full matrix needs a bigger machine.
- Evidence-quality caveat (`pilot-hubs`): unit tests ran on the same machine during the first graph, `multi_hub-small-s303`; see the campaign's `NOTES.md`.
- Evidence-quality caveat: short audit scripts (a few seconds of Python) ran alongside the benchmark while it was
  measuring `out_heavy-small` in `pilot-full`; treat that graph's timings with extra caution.

## 8. Phase 1 status and exit criteria

**Decision (2026-10-20, project owner): Phase 1 is closed at 31 benchmarked graphs.** The other 29 graphs are generated and validated but **not benchmarked**;
this is a documented exclusion, not an oversight. Reason: the remaining runs need roughly 8 h (21 graphs) to 12–14 h (including 4 powerlaw medium) on the
2-core sandbox, and the 4 powerlaw large graphs (≈1.48B work) probably do not fit it at all; a second machine of a different hardware class cannot be mixed
into the same labels. What was benchmarked: 31 graphs ≈ 93 M edges, ≈ 577 MB of Parquet, 1,085 valid timed runs. Consequences to state in any write-up: no
benchmark evidence above ≈205M work, no powerlaw medium/large, no Pokec runtimes, and no skew-triggering cases; conclusions apply to what was measured.
The 29 graphs can be run later with the commands in `docs/PHASE1_REPORT.md` §6 if a bigger machine becomes available.

Earlier assessment, kept for the record: Phase 1 data collection is complete; the benchmark part is partial.

| Exit criterion | Status |
|---|---|
| Required datasets generated or documented unavailable | **60/60 synthetic generated + validated** (family checks, Spark re-read); Pokec validated (HDFS not configured) |
| Dataset integrity demonstrated | Yes for all 60 + Pokec (manifest hashes, deterministic regeneration of the original 11 unchanged) |
| Query correctness demonstrated | Yes for every graph benchmarked (10 in `pilot-full` + smoke); 0 mismatches |
| Benchmark matrix and exclusions documented | Coverage matrix exists (`datasets/derived/coverage/`); **31 of 60 graphs fully measured (5 valid runs × 7 strategies = 1,085 runs)**; 29 graphs not run (compute-bound; 4 are above the 600M default safety gate) |
| Labels only from valid measured runs | Yes — enforced by aggregator, unit-tested, `label_eligible` flag |
| Raw results reproduce derived tables | Yes — `phase1_aggregate.py` / `phase1_coverage.py` regenerate everything; `SOURCE.txt` stores raw hashes |
| Remaining gaps explicit | This document + `docs/PHASE1_REPORT.md` |

Data collection (graphs, loading, manifest) is complete. Benchmark collection is not, and the labels it has produced so
far are essentially one class (§6), so downstream training on them is not justified yet.

## 8a. Review of the proposed ML table (one row per graph)

The proposed table (X features `log_num_vertices … work_in_hubs_share`; outcomes `median_total_s__<strategy>`,
`total_s_iqr__<strategy>`, `n_valid_runs__<strategy>`; target `best_strategy`; tracking columns) is sound in structure.
Checked against what the artifacts actually contain:

| Point | Finding | Suggested handling |
|---|---|---|
| Source of features | all but `log_in_p999` / `log_out_p999` are in `dataset_manifest.csv`; the p999 degree percentiles exist only in each `graph.json` | read p999 from `graph.json` (or ask for them to be added to the manifest, a backward-compatible change) |
| Logs | counts and work_sum span 3+ orders of magnitude | `log1p` is correct; keep the raw columns too, documented |
| `log_num_vertices` vs `log_num_edges` | perfectly collinear in the 60 generated graphs: edges/vertices ≈ 7.0 for every graph, corr(log V, log E) = 0.99999999995 | `log_num_edges` adds no information on this dataset (it would only differ on Pokec); keep one for modelling and document why; `active_fraction` is the non-redundant size-related feature |
| Hub columns | depend on the threshold (work > 1,000,000, P = 100 cells) | store the threshold as a column or in the schema doc so features stay reproducible |
| Target `best_strategy` | must be blank when `label_eligible` is false; must not be filled from fewer than 5 valid runs per strategy | filter on `label_eligible == True`; keep ties/near-ties visible via the margin column |
| Label quality | 30 of 31 labelled graphs choose an edge-join and 1 chooses native; 6 margins < 5 %; only 7 labels are `label_clear` | keep `margin_pct_total_s` / `label_clear`; prefer regret-based training on measured runtimes over class labels; consider dropping or down-weighting non-clear labels |
| Leakage | none of the X columns uses execution results | keep it that way; runtime columns are outcomes only |
| Pokec | work_sum 1.78B, 1.6M active vertices, outside the synthetic range | use as an external test point, not as training data, until strategies have been measured on it |
| Machine/config | `config_id` and `machine_id` are required columns | a model trained on one machine's labels must not be presented as machine-independent |

Conclusion: the schema is correct to build pipelines against; the limiting factor is label data, not columns.

## 8b. Phase 2 and Phase 3 prototype (delivered 2026-10-20)

**Phase 2 (features) — delivered, final schema.** `ml/build_feature_table.py` → `datasets/derived/features/graph_features.csv` (61 rows: 60 synthetic
+ Pokec as `external`). Documented in `docs/PHASE2_FEATURES.md`. No execution results are used; unit-tested for leakage and for agreement with the manifest.
`ml/build_training_table.py` joins features with the benchmark summaries on `graph_id` → `datasets/derived/training/training_table_eligible.csv`
(31 label-eligible graphs). 15 tests pass in `ml/tests`.

**Phase 3 prototype — pipeline delivered, performance not established.** `ml/train_selector.py` trains one multi-output Random Forest on
log(median total_s) per strategy, selects the lowest prediction, and is evaluated by leave-one-graph-out and leave-one-family-out against native
(AQE on/off), the best fixed strategy, a design-prior hub rule and the oracle. `--select <graph_id>` explains a choice. Outputs are in `datasets/derived/selector/`
(`SUMMARY.md`, `cv_predictions.csv`, `regret_by_graph.png`, `model.joblib`, `model_meta.json`). Results (31 graphs, leave-one-graph-out, regret vs oracle on measured medians):

| policy | mean regret | median regret | geomean speedup vs native (AQE on) | oracle chosen |
|---|---:|---:|---:|---:|
| oracle | 0.0 % | 0.0 % | 1.438 | 31 |
| best fixed strategy (always `edgejoin__aqe_on`) | 10.1 % | 0.0 % | 1.317 | 16 |
| **ML selector** | **14.8 %** | 3.6 % | 1.272 | 12 |
| native, AQE on | 45.3 % | 44.6 % | 1.000 | 1 |
| native, AQE off | 56.0 % | 51.1 % | 0.937 | 0 |
| hub rule (hypercube_unified if hubs else native) | 84.7 % | 95.3 % | 0.802 | 1 |

Honest reading: **the ML selector does not beat simply always choosing an edge-join** (mean regret 14.8 % vs 10.1 %); it picks between edge-join with AQE on and off,
which is mostly within run-to-run noise, and its worst miss (103 %) is `multi_hub-small-s101`. Both are far better than native GraphFrames, but that gain comes from
the edge-join strategy, not from the model. The hub rule is worse than native because the hypercube variants lose everywhere here. Inference takes a median of
17.7 ms per graph; feature-extraction time is **not** measured and not included. Leave-one-family-out gives the same picture (mean regret 15.2 %).
There is nothing for the model to learn about *which* family of strategy to pick, because one family always wins; the open question is whether
that changes on larger graphs, bigger machines and byte-skewed inputs (§14).

## 9. Team responsibilities

| Member | Role | Owns |
|---|---|---|
| 1 — Harshil | datasets, correctness, benchmarking | generator, manifest, Pokec, campaigns, raw records, coverage, AQE diagnostics |
| 2 | preprocessing & feature engineering | feature definitions, preprocessing script, graph-grouped splits, model-ready table |
| 3 | ML training & validation | label schema (with 1, 2), per-strategy log-runtime models, selector + native fallback, regret vs oracle |
| 4 | Tableau & paper | coverage/correctness tables, plots from validated data, data dictionary, report structure |

## 10. What each member can start now

- **Member 2 (features):** the feature table is already delivered (`datasets/derived/features/graph_features.csv`, schema in `docs/PHASE2_FEATURES.md`).
  Review and extend it; do not recompute the same columns differently. Open questions: add `in_p999`/`out_p999` to the manifest; decide whether to keep
  both `log_num_vertices` and `log_num_edges` (perfectly collinear on the synthetic graphs); measure feature-extraction time on Parquet (not done yet).
  Do **not** use `query_s`, shuffle bytes, task durations or any `runs.jsonl` field as an input feature.
- **Member 3 (ML):** start from `ml/train_selector.py` and `datasets/derived/training/training_table_eligible.csv` (31 label-eligible graphs, one
  measurement configuration). Split by `graph_id` or family, never by run. The prototype shows the pipeline works and does not beat "always edge-join";
  do not report a final model or a speedup from it.
- **Member 4 (evaluation):** use `datasets/derived/selector/cv_predictions.csv`, `datasets/derived/coverage/*.csv` and
  `datasets/derived/<campaign>/strategy_summary.csv` for tables and plots; mark every current figure **preliminary** and report graph counts next to run counts.

## 11. Dependencies and timeline

| Workstream | Depends on | Can proceed now? |
|---|---|---|
| Phase 1 benchmarks, remaining 29 graphs | machine time (≈16–37 min per graph here, 7 strategies × (1 warm-up + 5 measured)) | yes; roughly 9–14 h on this sandbox for the 21 that fit the gates (19 core + 2 large multi_hub), and the 8 powerlaw medium/large graphs need a larger machine and a raised `--max-work-sum` |
| Skew experiment (partitions > 256 MB) | new graph design + machine time | yes |
| Phase 2 features | — | **delivered**; review by Member 2 |
| Phase 3 pipeline | feature/label schema | prototype delivered; real training waits for more and more varied labels |
| Phase 4 evaluation scripts | metrics/baselines agreed | yes |
| Phase 4/5 final results | Phases 1–3 | later |

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

1. *Optional, not planned (Phase 1 closed at 31 graphs):* obtain a larger machine (or a dedicated multi-hour slot) and run the remaining 29 graphs with the same measurement configuration
   (`--resume` per campaign; powerlaw medium/large with a raised `--max-work-sum`). Keep one machine id.
2. Member 1: design the skew experiment: graphs whose hub reduce-partition exceeds 256 MB at 200 partitions (AQE skew splitting has never fired),
   alongside compute-skewed controls that do not trigger it. Only there can a non-edge-join strategy plausibly win.
3. Members 1–3: confirm the feature/label schema (`docs/PHASE2_FEATURES.md`; `graph_id` join; `label_eligible` filter; `measurement_config_id`).
4. Member 2: measure feature-extraction time; decide on the p999 manifest columns.
5. Member 3: replace the prototype by a regret-aware model once labels are less one-sided; keep native/edge-join fallbacks and graph-grouped splits.
6. Member 4: build tables/plots from the derived files, with the limitations in §6–§8b.

## 15. Definition of done

| Phase | Done when |
|---|---|
| 1 | 60 graphs generated + validated (or exclusions documented); every planned (graph, strategy) cell has ≥ 5 valid measured runs from one config/machine, or is documented as excluded with a reason; AQE skew case exercised; coverage matrix and raw→derived reproducibility verified |
| 2 | feature table regenerable from graphs, documented, no post-execution leakage, joins on `graph_id` |
| 3 | reproducible training on label-eligible data, graph-grouped held-out test, selector with native fallback, compared to native / rule / oracle with regret and overhead |
| 4 | frozen selector evaluated on untouched test graphs including feature-extraction and inference overhead |
| 5 | every number traceable to an artifact; figures reproducible; limitations and failed experiments reported |
