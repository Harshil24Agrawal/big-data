# Project walkthrough: the whole project as one flow

**Project:** ML-guided selection of graph-query execution strategies in Apache Spark
**Repo:** `Harshil24Agrawal/big-data`, branch `phase1-data-collection` (local copy: `Desktop\big-data`)
**Purpose of this file:** read it top to bottom and you will know **what** we did, **in which order**, **why** each choice was made, and **which file** does each part. It follows the same order you should present in.

> Companion documents: `docs/PROJECT_BRIEFING.md` (results tables and likely questions), `docs/FINAL_REPORT.md` (paper-style report), `docs/TABLEAU_GUIDE.md` (building the Tableau workbook).

---

## 0. The flow in one picture

Your suggested order was *topic → datasets → synthesis → load → preprocessing → feature engineering → train → evaluation*. That is close. The corrected order for this project has two extra steps that most ML projects don't have: **we had to measure our own labels** (by running Spark), and **we went round the loop once** (results showed a gap in the data, so we generated more data). The real order is:

```
 1. TOPIC          What problem? Which query? Why does it matter?
        |
 2. CANDIDATES     Which execution strategies can Spark use? (the "classes" we choose between)
        |
 3. DATA DESIGN    Why synthetic graphs? Which shapes (families), sizes, seeds?
        |
 4. SYNTHESIS      Generate the graphs with seeded generators, clean them, save as Parquet
        |
 5. VALIDATE+LOAD  Check every graph has the promised shape; re-read it through Spark
        |
 6. BENCHMARK      Run every strategy on every graph in Spark, check correctness, time it
   (= labelling)   -> raw run records (this is how we get the "y" values)
        |
 7. PREPROCESS     Turn thousands of raw runs into one median time per (graph, strategy)
   measurements
        |
 8. FEATURES       Describe each graph's shape with 11 numbers computed from the graph only
        |
 9. TRAINING TABLE Join features + measured times; label = fastest strategy
        |
10. FIRST RESULT   Edge-join wins almost everywhere -> data has no strongly skewed graphs
   and THE LOOP    -> go back to step 3: design + generate + benchmark the "hub sweep" (31 graphs)
        |
11. COMBINED TABLE 31 original + 31 sweep = 62 graphs, 3 strategies measured everywhere
        |
12. TRAIN          Classifiers (which class is fastest) + runtime regressor -> selector
        |
13. EVALUATE       Held-out graphs only (LOGO / LOSG / SEED202), against baselines
        |
14. PRESENT        Charts in the notebook + Tableau sources + dashboard
        |
15. REPRODUCE      Fixed seeds; notebook re-runs to the same 101 numbers
```

---

## 1. Topic: what problem we solve

### 1.1 The question in one sentence
Spark can run the same graph query in several different ways, and which way is fastest depends on the graph's shape. **Can a model look at the graph's shape before running anything and pick the fastest way?**

### 1.2 The query we study
The **two-hop path** (also called a two-hop motif): `(a)-[e1]->(b); (b)-[e2]->(c)`. In words: find every path a → b → c in a directed graph. It is the building block of friend-of-friend recommendations, fraud rings, citation chains and so on.

In Spark the graph is just a table of edges `(src, dst)`. Finding two-hop paths means **joining the edge table with itself** where `e1.dst = e2.src`.

### 1.3 Why it is hard: skew
- The number of paths that pass through a middle vertex `b` is `d_in(b) × d_out(b)`: every edge coming in can pair with every edge going out. We call this the vertex's **work**.
- Summing over all vertices gives `work_sum`, which is exactly the number of result rows.
- Spark splits a join by key (here, the vertex `b`). All of `b`'s work lands on **one task**. If one vertex (a **hub**) has huge work, its task runs far longer than all others. This slow task is a **straggler**, and the problem is called **skew**.
- Spark has a built-in fix (AQE skew-join splitting), but it only fires when a partition is bigger than 256 MB *in bytes*. A hub with 4,000 in-edges and 4,000 out-edges is under 1 MB of input but creates 16 million paths, so AQE never sees it.

### 1.4 Why a model at all
Strategies that fix skew cost extra setup (they must first profile the graph to find hubs). On a graph without skew that setup is wasted. So **there is no single best strategy**, and the choice depends on shape. A model that predicts the best choice from cheap shape statistics is a small "query optimizer", the same idea as research systems Bao and Lero (predict cost, then choose).

---

## 2. Candidates: the strategies we choose between

Every strategy returns exactly the same rows. Only the speed differs. Names are `<method>__aqe_<on|off>`, where AQE is Spark's Adaptive Query Execution.

| Strategy | What it does | Code |
|---|---|---|
| `native` | GraphFrames' built-in `g.find("(a)-[e1]->(b); (b)-[e2]->(c)")`. The reference answer. | GraphFrames library |
| `edgejoin` | A plain `e1.join(e2, e1.dst == e2.src)` with the same output columns as `find()`. No skew handling. | `bench/salting_1d.py` (`EdgeJoinGraphFrame`) |
| `salt1d` | **Salting.** Hub rows from one side are spread over 100 random buckets, and the other side is copied into every bucket, so a hub's work is split over 100 tasks. | `bench/salting_1d.py` (`Salted1DGraphFrame`) |
| `hypercube` | **2-D hypercube (Shares-style).** Hub edges go to an r × s grid of cells (incoming edges hashed to rows and copied across columns, outgoing edges the other way). Non-hubs go through an ordinary join. The two results are unioned. | `hypergraphframes/hypergraph/` (the team's library) |
| `hypercube_unified` | The same grid idea done in **one** join pass: non-hubs get a 1 × 1 grid, so there is no second branch. Same hub planner. | `bench/hypercube_unified.py` |

- **Hub** = a vertex with work `d_in × d_out` > **1,000,000** (the planner's default, `WORK_THRESHOLD`). **Grid budget** P = 100 cells (`p_cells`).
- **AQE settings** (pinned in every campaign): AQE on, skew join on, threshold 256 MB, factor 5×, coalesce on, advisory partition 64 MB.
- **Used in the first benchmark (7 strategies):** native and edgejoin each with AQE on and off; salt1d, hypercube and hypercube_unified with AQE off.
- **Used in the final model (3 strategies = 3 classes):** `edgejoin__aqe_off` → class **edgejoin**; `hypercube_unified__aqe_off` → class **hubsplit**; `native__aqe_on` → class **native**. Why only 3: salting was never fastest on 43 graphs; the split hypercube was always slower than the unified one; edge-join AQE on/off tie within noise because AQE splitting never fired.

Theory behind the hypercube (grid sizes `r ≈ √(d_in·P/d_out)`, `s ≈ √(d_out·P/d_in)`): `Algorithm.md`, `final_research_framework.md`.

---

## 3. Data design: why synthetic graphs, and which ones

### 3.1 Why synthetic
To learn "shape → best strategy" we need many graphs whose shape we **control and know**, at sizes this machine can benchmark many times. Real graphs with every skew level don't exist ready-made, and each one is a single example. Synthetic graphs are reproducible from a seed, and we can make the same shape at several sizes.

### 3.2 The 60-graph grid: 5 families × 3 sizes × 4 seeds
Defined in `datagen/config.py`, the single source of truth for all generation settings.

| Family | How it is built | What it tests |
|---|---|---|
| `uniform` | k random permutations, so every vertex has about 7 in-edges and 7 out-edges | no skew (the control group) |
| `powerlaw_balanced` | directed Chung-Lu, γ = 2.5. A few vertices are heavy on **both** sides; each vertex's edge share is capped at 0.2 % | realistic heavy-tailed networks |
| `in_heavy` | 25 % of edges point **into** 10 hubs per 100 k vertices (Zipf popularity); the rest are uniform | many arrivals at one place |
| `out_heavy` | mirror image: hubs with many **outgoing** edges | many departures from one place |
| `multi_hub` | 10 / 20 / 40 hubs (small / medium / large), each with about 2,000 in and 2,000 out edges (±25 %), on a uniform background | several comparable heavy vertices |

| Size tier | Vertices | Edges (≈7 per vertex) |
|---|---|---|
| small | 100,000 | ≈700,000 |
| medium | 500,000 | ≈3,500,000 |
| large | 1,000,000 | ≈7,000,000 |

**Seeds:** 101, 202, 303, 404. Four independent copies of every (family, size), so the model can be tested on new random graphs of a known kind.

### 3.3 The real graph
**soc-Pokec** (SNAP social network): 1,632,803 users, 30,622,564 edges, 1.78 billion two-hop paths. It was validated with the same rules (`datagen/pokec.py`), but it was **not benchmarked** (too heavy for 2 cores) and **never used for training**. It shows the pipeline handles real data and gives a feature row for comparison.

---

## 4. Synthesis: how the graphs are generated

**File:** `datagen/generators.py` (the generators) and `datagen/generate.py` (the driver).
**Command:** `python -m datagen.generate --plan full` (60 graphs) or `--plan pilot` (11 graphs).

Step by step for one graph:
1. **Seed everything.** One `numpy.random.Generator(seed)` produces all randomness, so the same (family, size, seed, parameters) gives **byte-identical edges** on any machine.
2. **Sample edges** with the family's process (permutations, Chung-Lu, hub-plus-uniform, …).
3. **Clean** (`clean_edges`): directed edges, **remove self-loops** (a→a), **remove duplicate pairs**, vertex ids are `int32` in `[0, n)`.
4. **Top up** (`sample_to_target`): if cleaning dropped edges, sample more from the same distribution until the target count exists, then keep a seeded random subset of exactly `m_target` edges. Every graph of a tier therefore has exactly the planned edge count.
5. **Measure** in the same pass: in/out degrees, work per vertex, `work_sum`, Gini coefficients, top-hub shares and so on (`datagen/validate.py: degree_stats`).
6. **Name it by content:** `graph_id = <family>-<tier>-s<seed>-<hash>`, where the hash covers all generation parameters. Change any parameter and you get a new id, so an old graph is never silently altered.
7. **Save:** `datasets/synthetic/<family>/<graph_id>/edges.parquet` (columns `src`, `dst`) plus `graph.json` (parameters, measured statistics, check results, file hashes).

**Why Parquet:** columnar, compressed, and what Spark reads fastest. The edge files (1.3 GB in total) are not in git (`.gitignore`) because they can be regenerated exactly. Their hashes are in git.

---

## 5. Validation and loading

A family name is a promise; we check that every graph keeps it.

| Check | Where | What it does |
|---|---|---|
| Family checks | `datagen/config.py: FAMILY_CHECKS`, run by `datagen/validate.py: run_family_checks` | Thresholds fixed **before** generation. Examples: uniform needs degree CV ≤ 0.2; powerlaw needs max/mean ≥ 50 and Gini ≥ 0.4; in_heavy needs the top hubs to receive ≥ 15 % of edges; multi_hub needs ≥ 5 hubs holding ≥ 50 % of work with comparable sizes. Measured values are stored next to the thresholds in `graph.json`. |
| Spark re-read | `datagen/spark_validate.py` | Reads every Parquet file back **through Spark** (the same reader the benchmark uses) and recomputes row count, nulls, id range, self-loops, duplicates, active vertices, `work_sum` and max work. A graph only counts if Spark sees exactly what the generator measured. |
| Manifest | `datasets/manifest/dataset_manifest.csv` | One row per graph (60 rows, 40 columns): parameters, measured statistics, hashes, validation status. This is the catalogue every later step reads. |
| Unit tests | `datagen/tests/test_datagen.py` | Determinism per seed, cleaning, family checks reject the wrong shape, `work_sum` equals a brute-force path count. |

**Loading in Spark (inside the benchmark):** read the Parquet file, cast to int, repartition to 200 partitions, cache, and materialise. The vertex table is built from the distinct endpoints, cached and broadcast-hinted. Every strategy then works from the same in-memory edges.

**Result:** all 60 graphs passed the family checks and the Spark re-read.

---

## 6. Benchmark = how we collected the labels

This is the most expensive step. A label ("which strategy is fastest on this graph?") does not exist anywhere; we **measure** it.

### 6.1 Files
| File | Role |
|---|---|
| `bench/run_phase1.py` | **Campaign runner.** Picks graphs from a manifest, starts one worker subprocess per graph, writes the campaign folder. Resumable (`--resume`), with a dry-run mode. |
| `bench/phase1_worker.py` | **Worker.** Runs inside one fresh Spark application per graph: load → correctness gate → timed rounds → append run records. |
| `bench/phase1_strategies.py` | Registry that maps a strategy name like `edgejoin__aqe_off` to the code that runs it and its AQE settings. |
| `bench/phase1_metrics.py` | Extra per-run metrics from Spark's REST API: task-duration spread, shuffle sizes, and whether AQE split or coalesced anything. |
| `bench/bench.py` | Original single-run benchmark from the team repo; the worker reuses its Spark setup and metric collection. |

### 6.2 What happens for each graph
1. **Fresh Spark application** (local mode, all cores of this 2-vCPU machine, 4 GB driver memory, 200 shuffle partitions, broadcast joins off so every strategy does a real shuffle join).
2. **Correctness gate (untimed).** Every strategy runs once. Its row count must equal `work_sum`, and two order-independent checksums must equal native `find()`'s. A strategy that fails gets no timed runs. (None failed.)
3. **1 warm-up round** (discarded: the first run pays for JVM and code loading).
4. **N measured rounds** (5 for the original graphs; 3, or 2 for the three largest, for the sweep). In each round every strategy runs once, **in a freshly shuffled order** (the seed is recorded), so no strategy is always first or last.
5. **Time measured:** `total_s` = **profiling time** (finding hubs; zero for native and edge-join) + **query time** (the join, forced with an RDD count so the whole result is computed).
6. Each run becomes **one appended line** in `runs.jsonl`, plus stage-level detail in `run_details.jsonl`. Nothing is overwritten and failures are written as failures.

### 6.3 Where the results live
`datasets/raw_runs/<campaign_id>/` with `campaign.json` (exact config + machine id), `runs.jsonl`, `run_details.jsonl`, `graphs/<graph_id>.json`, `logs/`. `datasets/raw_runs/INDEX.csv` lists all campaigns.

| Campaign | Used for | Graphs |
|---|---|---|
| `smoke`, `pilot`, `pilot-reduced` | early tests of the pipeline | — (not used) |
| `pilot-full`, `pilot-hubs`, `pilot-ext1`, `pilot-ext2` | **original benchmark**, 7 strategies, 5 runs | 31 |
| `hubsweep-probe` | first test of single-hub graphs | 3 (not used in training) |
| `hubsweep-sweep_a/_b/_c`, `sweep_big`, `sweep_big202` | **hub sweep**, 3–4 strategies, 3 or 2 runs | 31 |

**Totals:** 1,085 valid timed runs on the original 31 graphs and 306 on the sweep. **0 failures, 0 timeouts, 0 wrong results.** One empty record (`…-hubsweep-sweep_big-31949d`, 0 graphs) is kept and ignored.

**Why only 31 of the 60 graphs were benchmarked:** the other 29 would need 8–14 more hours on 2 cores, and the four largest powerlaw graphs (1.5 billion paths) probably do not fit. This is recorded as a decision, not hidden.

---

## 7. Preprocessing the measurements

Raw runs are noisy (on the same graph and strategy, the slowest of 5 runs is a median 1.7× the fastest, up to 3×), so we summarise.

| File | Output | What it does |
|---|---|---|
| `bench/phase1_aggregate.py <campaign>` | `datasets/derived/<campaign>/strategy_summary.csv`, `graph_summary.csv`, `SOURCE.txt` | Per (graph, strategy): counts, correctness, **median**, p95, min, max and **IQR** of the measured runs. Per graph: fastest strategy and margin. Only reads the raw files. |
| `bench/phase1_coverage.py` | `datasets/derived/coverage/` | Which (campaign, graph, strategy) has how many valid runs, and the status of each campaign. Records a hash of every raw file used. |

**Why the median:** one bad run (garbage collection, OS hiccup) does not move it. **Why the IQR:** it tells us whether a "winner" truly beats the runner-up or only by noise.

---

## 8. Feature engineering: describing a graph with numbers

**File:** `ml/build_feature_table.py` → `datasets/derived/features/graph_features.csv` (61 rows: 60 graphs + Pokec; `SCHEMA.json` documents every column).

**Golden rule:** features come **only from the graph structure** (degrees and work). No runtime, strategy or campaign column is ever read here. The selector must decide **before** running anything; using runtimes would be cheating (label leakage).

**Preprocessing of the features:** `log1p` on counts (they span 5 orders of magnitude, from 50 to 10⁹), ratios left as they are.

| # | Feature | Meaning in plain words | Why it can matter |
|---|---|---|---|
| 1 | `log_num_vertices` | graph size | more data, more time |
| 2 | `log_num_edges` | size in edges | (≈ 7 × vertices here; kept for real graphs) |
| 3 | `active_fraction` | share of vertices touching any edge | isolated vertices do no work |
| 4 | `in_gini` | how unequal in-degrees are (0 = equal, 1 = one vertex has all) | skew in receivers |
| 5 | `out_gini` | same for out-degrees | skew in senders |
| 6 | `log_in_p999` | in-degree of the top 0.1 % of vertices | how fat the tail is |
| 7 | `log_out_p999` | same for out-degree | |
| 8 | `log_work_sum` | total number of two-hop paths | the total workload |
| 9 | `work_max_share` | share of all paths through the single busiest vertex | **the main skew signal** |
| 10 | `log_num_work_hubs` | number of vertices with work > 1 M | how many hubs the planner would find |
| 11 | `work_in_hubs_share` | share of all paths through those hubs | how much of the work is skewed |

**Two derived features added in the notebook (13 inputs in total):**
- `log_hub_work = log1p(work_sum × work_in_hubs_share)`: absolute paths in hubs. Hub strategies pay a roughly fixed profiling cost, so absolute hub work decides whether that cost pays off.
- `log_max_work = log1p(work_sum × work_max_share)`: paths through the busiest vertex. The plain join's straggler task is exactly that vertex, so this sets the runtime floor. It became the **most important feature**.
- Honesty note: `log_max_work` was chosen after seeing the sweep, so the notebook also reports results with only the 11 base features.

---

## 9. The training table and the label

| File | Output |
|---|---|
| `ml/build_training_table.py` | `datasets/derived/training/training_table.csv` and `training_table_eligible.csv` (31 rows × 69 columns: features + 7 strategies × 5 measurements + label + provenance) |
| `ml/build_combined_table.py` | `datasets/derived/training/combined_table.csv` (**62 rows × 43 columns**: the final training data) |

**How the label is chosen (`choose_label`):**
- `best_strategy` = the strategy with the **lowest median `total_s`**.
- `label_margin_pct` = how much faster it is than the runner-up, in %.
- `label_clear` = True if that gap is bigger than both strategies' IQR (the run spread). **32 of 62 labels are clear**; the rest are near-ties.
- `best_class` = edgejoin / hubsplit / native.

**Safety rules in the builders:** they refuse to mix different machines or Spark configurations (all rows: machine `6c5d1cb517a4`, one config). The only allowed difference is the number of timed runs, which is recorded in the `protocol` column (`5runs_original` 31 rows, `3runs_sweep` 28, `2runs_sweep` 3).

**Combined-table columns, by group:**
- ids/provenance: `graph_id, family, size_tier, seed, protocol, source_campaign, machine_id, cmp_config, hub_count, hub_work_share_planned`
- inputs: the 11 features + `raw_work_sum`
- outcomes for each of the 3 strategies: `median_total_s, total_s_iqr, median_query_s, median_profiling_s, n_valid_runs` (+ salt1d median as a diagnostic)
- answer: `best_strategy, best_class, runner_up_strategy, label_margin_pct, label_clear`

---

## 10. The first result, and why we went round the loop

**Prototype** (`ml/train_selector.py`, outputs in `datasets/derived/selector/`) on the 31 original graphs:
- Edge-join was fastest on 30 of 31 graphs (AQE on 16, off 14); native won 1. Salting and hypercube were **never** fastest (1.5–2.6× slower).
- Reason: in those graphs the busiest vertex holds at most **12 %** of the work. The skew strategies had no real skew to fix, so their setup cost was pure loss.
- So the labels were almost one-sided, and a model cannot learn "when does hub-splitting win?" from data where it never wins.

**Decision:** design a new graph family that covers the missing regime: **one or a few vertices holding most of the work.**

### 10.1 The hub sweep
**File:** `datagen/hub_sweep.py`. **Data:** `datasets/hubsweep/<graph_id>/` (31 graphs; these small edge files *are* in git, 96 MB) + `datasets/hubsweep/manifest.csv`.
- Each graph is a uniform background (100 k vertices, 7 random out-edges each) plus **H planted hubs** (H = 1, 2, 4 or 8).
- Each hub gets `d` distinct in-neighbours and `d` out-neighbours with `d = round(√(share/(1−share) · n·7·7 / H))`, so the hubs together hold the planned share of all two-hop work.
- Planned hub shares: 50 %, 80 %, 90 %, 95 %, 98 %, plus 99 % and 99.5 % for the big single-hub graphs.

| Set | Hubs | Hub share | Seed | Graphs |
|---|---|---|---|---:|
| sweep A | 1, 2, 4, 8 | 80, 90, 95 % | 101 | 12 |
| sweep B | 1, 2, 4, 8 | 50, 98 % | 101 | 8 |
| sweep C (new seed, for testing) | 1, 4 | 50, 90, 95, 98 % | 202 | 8 |
| big single hub | 1 | 99 % (seeds 101, 202), 99.5 % (101) | | 3 |

Benchmarked with the same worker: `python bench/run_phase1.py --manifest datasets/hubsweep/manifest.csv --strategies … --runs 3`. Salting was dropped after sweep A to save time.

### 10.2 What the sweep showed
- **Hub-splitting wins only with a single hub carrying about 10⁸ or more paths** (6 graphs). There it is up to 2× faster than edge-join and 3× faster than native: at 99.5 %, 80.5 s vs 154.8 s (edge-join) vs 251.2 s (native).
- With 2, 4 or 8 hubs the work is split between them, no single task is that slow, and **edge-join wins at every share**.
- Why: hub-splitting removes the straggler but costs about 7–10 s of profiling. On 2 cores the straggler only outweighs that once one vertex carries about 10⁸ paths.

---

## 11. Training (the notebook)

**File:** `notebooks/selector.ipynb`, generated by `notebooks/_build_selector_notebook.py` (edit the generator, not the notebook, then run `python notebooks/_build_selector_notebook.py`).
**Input:** only `datasets/derived/training/combined_table.csv`. **Outputs:** `datasets/derived/selector_v2/` and `datasets/derived/tableau/`.

### 11.1 Notebook sections in order
| Section | What it does | Chart |
|---|---|---|
| Setup | imports, finds the repo folder, prints library versions, fixes the seed | — |
| 1. Data | loads 62 rows, adds `group` (family, or `hub_sweep_H<k>`), adds the 2 derived features, builds the runtime matrix T (62 × 3) | label balance |
| 2. Runtime vs hub share | raw times of the sweep graphs, total and query-only | runtime_vs_hub_share |
| 2b. Crossover | hypercube ÷ edge-join time against paths through the busiest vertex | crossover_max_vertex_work |
| 3. Classifier | 7 models × 3 test schemes; 11-vs-13-feature ablation; skewed-regime check | — |
| 3b. Classifier comparison | accuracy / balanced accuracy / macro-F1 per model and scheme; per-class recall | classifier_comparison, classifier_recall_heatmap |
| Confusion + importance | confusion matrix of the best model; random-forest feature importance | classifier_confusion_importance |
| 4. Regressor | multi-output random forest predicts log runtime of all 3 strategies | regressor_pred_vs_actual |
| 5. Selector | pick argmin of predicted runtime; compare policies | regret_and_total_time |
| 6. Overheads | inference time; sensitivity to feature-extraction cost | — |
| 6b. More charts | feature space, speed-up by group, regret CDF, label noise, correlations | 5 charts |
| 7. Summary | auto-written summary from the numbers; writes results.json + CSVs | — |
| 8. Tableau | exports Tableau sources, loads them back, draws the dashboard preview | dashboard_preview |
| 9. Reproducibility | compares 101 key numbers with `notebooks/expected_results.json` | — |

### 11.2 The two kinds of model and why
**A. Classifier: which class (edgejoin / hubsplit / native) is fastest?**

| Model | Why it is included |
|---|---|
| majority (always "edgejoin") | the baseline every accuracy must beat |
| rule (has hubs → hubsplit, else native) | what a hand-written rule would do |
| logistic regression (scaled, class-balanced) | simple linear model, good at rare classes |
| depth-2 tree (balanced) | tiny, explainable, works with 6 examples per class |
| stump on `log_max_work` (a single threshold) | the simplest possible "skew" rule, learned from data |
| random forest (300 trees, balanced) | standard strong model for small tabular data |
| extra-trees (300 trees) | a more randomised forest, often more stable on small data |

**B. Regressor + selector: predict each strategy's runtime, choose the smallest.**
- One random forest (300 trees) predicts `log(median total time)` of all 3 strategies from the 13 features.
- **Selector** = pick the strategy with the lowest predicted time. Variant: **native fallback** (stay with native unless the predicted gain is > 5 %).
- **Why regression is the main model:** labels are 81 % one class and many are near-ties. The regressor keeps *how much* faster a strategy is, so a wrong pick in a near-tie costs almost nothing, while a missed 2× win is avoided.

**Why random forests:** few rows (62), mixed feature scales, non-linear thresholds (the crossover), and they give feature importances. Deep learning would overfit 62 rows.

**Determinism:** `random_state=0`, `n_jobs=1`, no random sampling elsewhere, so every re-run gives the same models.

---

## 12. Evaluation: how we test honestly

### 12.1 Never test on a graph the model has seen
| Scheme | Train on | Test on | Question it answers |
|---|---|---|---|
| **LOGO** (leave one graph out) | 61 graphs | the 62nd, repeated 62 times | does it work on a new graph of a known kind? |
| **LOSG** (leave one structural group out) | all groups but one | a whole family, or all sweep graphs with H hubs | does it work on a **new kind** of graph? (hardest) |
| **SEED202** | everything except the 9 seed-202 sweep graphs | those 9 | does it work on new random draws of the sweep? |

Splitting by **graph** (never by run) matters: runs of the same graph are not independent, and splitting by run would leak the answer.

### 12.2 Metrics
- **accuracy**, always next to the **majority baseline** (80.6 %); **balanced accuracy** (average recall per class, so ignoring a class is punished); **macro-F1**.
- **regret** = chosen strategy's time ÷ best strategy's time − 1 (0 % = perfect).
- **speed-up vs native** (geometric mean), **total time** over all graphs, **R²** of predicted log runtime, **top-k** (is the true best within the model's k best guesses).
- **oracle** = best strategy per graph, known only after running everything (the upper bound).

### 12.3 Final numbers (62 graphs; all from the notebook)
**Classifier, LOGO:**

| Model | Accuracy | Balanced acc. | Macro-F1 |
|---|---:|---:|---:|
| majority | 80.6 % | 0.333 | 0.298 |
| rule | 14.5 % | 0.500 | 0.171 |
| logistic regression | 75.8 % | **0.753** | 0.656 |
| depth-2 tree | 51.6 % | 0.556 | 0.528 |
| stump on max work | **85.5 %** | 0.598 | 0.561 |
| random forest | 83.9 % | 0.689 | 0.678 |
| extra-trees | **85.5 %** | 0.696 | **0.703** |

- Best accuracy 85.5 % vs 80.6 % majority (about 3 more graphs right out of 62). Hubsplit is found 5 of 6 times; native wins (mostly near-ties) are hard.
- LOSG: no model beats the majority (best 80.6 %). The models interpolate; they do not extrapolate to unseen kinds of graph.
- SEED202: best 77.8 % vs 55.6 % majority.

**Selector, LOGO:**

| Policy | Mean regret | Speed-up vs native | Total time |
|---|---:|---:|---:|
| always native | 49.5 % | 1.00× | 1810 s |
| always edge-join (best fixed rule) | 8.3 % | 1.34× | 1226 s |
| hub rule | 56.9 % | 0.95× | 1490 s |
| **ML selector** | **4.6 %** | **1.38×** | **1070 s** |
| ML + native fallback | 6.1 % | 1.36× | 1079 s |
| oracle | 0 % | 1.43× | 1037 s |

- Regressor: R² 0.889, median error 10.1 %, true best strategy in its top 2 on 100 % of graphs.
- On the skewed graphs (busiest vertex ≥ 90 % of work, 7 graphs): selector regret **3.0 %** vs **46.9 %** for always-edge-join.
- Overhead: 14 ms per prediction. Feature extraction was not timed; if it cost as much as the hypercube's profiling (≈ 9 s), the selector would lose on short jobs.

### 12.4 How to read each chart (all in `datasets/derived/selector_v2/`, all inline in the notebook)
| Chart | One-line message |
|---|---|
| `label_balance.png` | 50 edgejoin / 6 hubsplit / 6 native: heavy class imbalance, and hubsplit wins all come from the hub sweep |
| `runtime_vs_hub_share.png` | times only explode past 98 % hub share; query-only time shows hypercube's join is fast but profiling costs extra |
| `crossover_max_vertex_work.png` | hypercube ÷ edge-join drops below 1 only when the busiest vertex carries ≳ 10⁸ paths |
| `classifier_comparison.png` | 7 models × 3 schemes: small gain over majority on LOGO, none on LOSG |
| `classifier_recall_heatmap.png` | which classes each model finds: hubsplit yes, native rarely |
| `classifier_confusion_importance.png` | where the best model errs, and which features it uses (`log_max_work` first) |
| `regressor_pred_vs_actual.png` | predicted vs measured runtime on held-out graphs: points hug the diagonal |
| `regret_and_total_time.png` | per graph: how much slower each policy was than the best, and total time per policy (ML ≈ oracle) |
| `feature_space.png` | hubsplit wins sit in one corner (high busiest-vertex work and share) |
| `speedup_by_group.png` | edge-join beats native in every group; hypercube only in single-hub groups |
| `regret_cdf_speedup.png` | share of graphs within x % of the best: ML selector closest to the oracle |
| `label_noise.png` | many winners lead by only a few %: a limit on any classifier's accuracy |
| `correlations_and_errors.png` | size/work features are strongly correlated; regressor errors centred on zero |

---

## 13. Presentation layer: Tableau

**Why Tableau:** the course asks for a dashboard. Tableau does no learning; it presents the results.

**Section 8 of the notebook does all of it and runs with the rest:**
1. **Exports** `datasets/derived/tableau/`:
   - `graph_strategy_long.csv`: 186 rows (62 graphs × 3 strategies) with times, regret, speed-up, `max_vertex_work`, family, hub count, label. The main source.
   - `policy_scores.csv`: 21 rows (3 schemes × 7 policies).
   - `classifier_scores.csv`: 21 rows (3 schemes × 7 models).
   - `cv_predictions_logo.csv`: 62 rows, the true class and every model's held-out prediction (for a confusion matrix).
   - `selector_results.hyper`: one Tableau extract containing all four tables (written when `tableauhyperapi` is installed; it is in `requirements.txt`).
2. **Loads them back from disk** (what Tableau reads) and asserts they reproduce the notebook's numbers.
3. **Draws the dashboard preview** (`dashboard_preview.png`): KPI row (62 graphs · edge-join faster than native on 56/62 · 6 hub-splitting wins · selector regret 4.6 % vs 8.3 %), then Sheet 1 crossover, Sheet 2 runtime vs hub share, Sheet 3 policy regret, Sheet 4 classifier vs majority, Sheet 5 label balance by family, and Sheet 6 the graph table. Set `SCHEME = "LOSG"` to switch schemes, like a Tableau parameter.

**In Tableau Desktop / Public:** *Connect → Text file → `graph_strategy_long.csv`* (add the other three as separate data sources; do not join them), or *Connect → Hyper → `selector_results.hyper`*. Then build the six sheets as in `docs/TABLEAU_GUIDE.md` (it lists the calculated fields, such as the hypercube ÷ edge-join ratio). An interactive web version of the same dashboard also exists as a Claude artifact ("Spark Strategy Selector Dashboard").

---

## 14. Reproducibility: re-run and get the same answer

**Anaconda + VS Code setup (Windows, one time):** from the repo folder in *Anaconda Prompt*:
```
conda env create -f environment.yml
conda activate bigdata-selector
python -m pip install -r notebooks/requirements.txt
```
(The pip step is separate on purpose: conda's built-in pip step fails when the Windows user name contains a space, e.g. `Harshil Agrawal`. If the env already exists, skip the first command.)
Then in VS Code: open the `big-data` folder, open `notebooks/selector.ipynb`, click **Select Kernel → Python Environments → bigdata-selector**, then **Run All**. (If the kernel is not listed: Ctrl+Shift+P → "Developer: Reload Window".) The notebook needs no Spark or Java; it only reads `datasets/derived/training/combined_table.csv`. Spark is only needed to re-run the benchmark itself.

```
pip install -r notebooks/requirements.txt          # Python 3.11; pinned numpy/pandas/sklearn/scipy/matplotlib
cd notebooks
jupyter notebook selector.ipynb                     # Run All
#   or headless:
python -m nbconvert --to notebook --execute --inplace selector.ipynb
```
- The last cell prints **`REPRODUCED: all 101 key numbers match expected_results.json`** (accuracies, balanced accuracies, regrets, speed-ups, R², top-2, Tableau row counts and KPIs).
- Only wall-clock values (inference milliseconds) change between runs.
- `NB_TREES=20` gives a quick run (the check is skipped); `NB_UPDATE_EXPECTED=1` rewrites the expected file after a deliberate change.
- Verified here: repeated executions gave identical results; the Tableau CSVs come out byte-identical every run. (The `.hyper` file is a small database whose bytes differ per run even though its contents are the same.)
- Re-running the **benchmark** itself (Spark) gives different absolute times on another machine. That is why the notebook starts from the committed measurement table, and why every campaign records its machine.

**Full pipeline from scratch (hours):**
```
python -m datagen.generate --plan full              # 60 graphs + manifest
python -m datagen.spark_validate                    # Spark re-read check
python bench/run_phase1.py --name pilot-full ...    # benchmark campaigns (needs the GraphFrames jar in jars/)
python bench/phase1_aggregate.py <campaign_id>      # medians per graph x strategy
python ml/build_feature_table.py                    # 11 features
python ml/build_training_table.py                   # 31-graph table
python -m datagen.hub_sweep --plan sweep_a          # hub-sweep graphs (also sweep_b, sweep_c, sweep_big, sweep_big202)
python ml/build_combined_table.py                   # 62-graph table
# then the notebook
```

---

## 15. Every file in the repo and what it is for

Legend: **★** = part of our ML pipeline; *legacy* = from the original team repo (the HyperGraphFrames library work that our project builds on).

### Root
| File | Role |
|---|---|
| `Algorithm.md` | *legacy* formal spec of the 2-D hypercube algorithm (hub detection, grid sizes, join) |
| `final_research_framework.md` | *legacy* the original research framing: computational vs byte skew, why AQE misses it |
| `preprocessing.py` | *legacy* converts the Pokec text file to Parquet (reused by `datagen/pokec.py`) |
| `graphframes_demo.py`, `hypergraphframes_demo.py` | *legacy* short demos of native GraphFrames and the library |
| `baseline_graphframes.py` (+ `_metrics.json`) | *legacy* native GraphFrames on Pokec with Spark stage metrics |
| `hypercube.py` | *legacy* runs the library's hypercube on Pokec and records stage metrics |
| `pokec_benchmark.py` | *legacy* 7-condition Pokec benchmark (AQE variants, salting, hypercube) |
| `test_e2e_live.py` | *legacy* end-to-end test of the published PyPI package |
| `hadoop/bin/` | `winutils.exe`, `hadoop.dll`: needed to run Spark on Windows |
| `LICENSE`, `.gitignore` | licence; ignore rules (large edge files and jars are kept out of git) |

### `hypergraphframes/`: the team's library (*legacy*, used by the `hypercube` strategy)
| File | Role |
|---|---|
| `hypergraph/topology.py` | profiling: degrees, work per vertex, hubs, grid size per hub |
| `hypergraph/planner.py` | parses the motif and runs the 2-D hypercube join |
| `hypergraph/core.py` | `HyperGraphFrame`: drop-in wrapper around GraphFrame with `.find()` |
| `tests/` | equivalence tests (same rows as native) and motif tests |
| `pyproject.toml`, `setup.py`, `README.md` | packaging (published on PyPI as `hypergraphframes`) |

### `datagen/` ★ stages 3–5 and 10
| File | Role |
|---|---|
| `config.py` | all generation settings: seeds, tiers, families, edge policy, family checks, pilot list |
| `generators.py` | the 5 seeded family generators + cleaning + top-up |
| `generate.py` | driver: generate, measure, check, save Parquet + `graph.json`, rebuild the manifest |
| `validate.py` | degree/work statistics, family checks, Spark load check |
| `spark_validate.py` | re-reads every graph through Spark and compares |
| `hub_sweep.py` | hub-sweep generator and plans (probe, sweep_a/b/c, sweep_big, sweep_big202) |
| `pokec.py` | validates and prepares the real Pokec graph |
| `tests/test_datagen.py` | fast unit tests for generation and validation |

### `bench/` ★ stages 6–7 (plus older benchmark tools)
| File | Role |
|---|---|
| `run_phase1.py` | ★ campaign runner (one subprocess per graph, resumable) |
| `phase1_worker.py` | ★ per-graph worker: load, correctness gate, timed rounds, raw records |
| `phase1_strategies.py` | ★ strategy registry (name → code + AQE settings) |
| `phase1_metrics.py` | ★ task-time spread, shuffle sizes, AQE decisions from Spark's REST API |
| `phase1_aggregate.py` | ★ raw runs → median/IQR per graph × strategy |
| `phase1_coverage.py` | ★ which runs exist for which campaign/graph/strategy |
| `hypercube_unified.py` | ★ the single-pass hypercube strategy (the "hubsplit" class) |
| `salting_1d.py` | ★ edge-join and 1-D salting strategies, and shared output shaping |
| `bench.py` | single-run benchmark (Spark setup and metric collection reused by the worker) |
| `run_matrix.py`, `run_ablation_matrix.py`, `run_single_ablation.py` | older controlled benchmarks (variant × AQE matrix, AQE skew/coalesce ablation) |
| `summarize.py`, `print_results.py` | summaries of the older benchmark results |
| `test_phase1.py`, `test_bench_variants_equivalence.py` | Spark tests: every strategy returns exactly native's rows; record schema |

### `ml/` ★ stages 8–9
| File | Role |
|---|---|
| `build_feature_table.py` | 11 shape features per graph (structure only) |
| `build_training_table.py` | features + measured outcomes + label (31 graphs); `choose_label` |
| `build_combined_table.py` | final 62-graph table, 3 strategies, refuses mixed configs |
| `train_selector.py` | first prototype selector on 31 graphs (superseded by the notebook) |
| `tests/` | tests for features, labels, config ids and the prototype selector |

### `notebooks/` ★ stages 11–15
| File | Role |
|---|---|
| `selector.ipynb` | **the main notebook**: training, evaluation, all charts, Tableau, reproducibility check |
| `_build_selector_notebook.py` | generates the notebook (single source of truth for its code) |
| `requirements.txt` | pinned library versions |
| `expected_results.json` | the 101 reference numbers the last cell checks against |

### `datasets/`
| Path | Content |
|---|---|
| `synthetic/<family>/<graph_id>/graph.json` | 60 graphs' parameters, statistics, checks (edge Parquet regenerable, not in git) |
| `hubsweep/<graph_id>/` + `manifest.csv` | 31 hub-sweep graphs (Parquet + `graph.json`) |
| `manifest/dataset_manifest.csv` | catalogue of the 60 synthetic graphs |
| `pokec/` | Pokec validation receipt and statistics |
| `raw_runs/<campaign>/` + `INDEX.csv` | immutable benchmark records (never edited) |
| `derived/<campaign>/` | per-campaign summaries (median, IQR, fastest) |
| `derived/coverage/` | coverage matrix and campaign status |
| `derived/features/` | `graph_features.csv`, `SCHEMA.json` |
| `derived/training/` | `training_table*.csv` (31 graphs), **`combined_table.csv`** (62 graphs) |
| `derived/selector/` | prototype selector outputs (31 graphs) |
| `derived/selector_v2/` | **final** results: `results.json`, score CSVs, all charts |
| `derived/tableau/` | Tableau sources, `.hyper` extract, `dashboard_preview.png` |

### `docs/`
| File | Role |
|---|---|
| `PROJECT_WALKTHROUGH.md` | this file: the flow and every file |
| `PROJECT_BRIEFING.md` | results tables, decisions, likely questions |
| `FINAL_REPORT.md` | paper-style report |
| `TABLEAU_GUIDE.md` | building the Tableau workbook |
| `PHASE1_DATA_COLLECTION.md`, `PHASE1_REPORT.md` | protocol and results of the first benchmark |
| `PHASE2_FEATURES.md` | feature definitions |
| `PROJECT_STATUS_AND_TEAM_PLAN.md` | status log and team plan |

---

## 16. Presentation script (one slide per step)

| # | Slide | What to say |
|---|---|---|
| 1 | Topic | Same query, several execution plans; the best depends on graph shape. Can ML choose before running? |
| 2 | The query and skew | Two-hop paths = self-join; work = d_in × d_out; one hub = one slow task; AQE misses it (bytes vs work). |
| 3 | Strategies | native, edge-join, salting, hypercube (2 versions); hub threshold 1 M, 100 cells. |
| 4 | Data design | Synthetic for control and reproducibility: 5 families × 3 sizes × 4 seeds; Pokec as a real check. |
| 5 | Synthesis + validation | Seeded generators, cleaning, content-hash ids, family checks, Spark re-read. All 60 pass. |
| 6 | Benchmark = labels | Fresh Spark per graph, correctness gate, warm-up, shuffled rounds, medians; 1,391 timed runs, 0 failures. |
| 7 | Features | 11 structure-only numbers, log-scaled; no leakage; + busiest-vertex work. |
| 8 | First finding | Edge-join wins 30/31; no skewed graphs in the data → the hub sweep (31 graphs). |
| 9 | Crossover | Hub-splitting wins only with one hub ≳ 10⁸ paths (show the crossover chart). |
| 10 | Models | Classifier family vs majority; regressor → selector; why forests; why regression. |
| 11 | Evaluation | LOGO / LOSG / SEED202; accuracy always next to the majority line (show the comparison chart). |
| 12 | Results | 85.5 % vs 80.6 %; selector regret 4.6 % vs 8.3 %, 3.0 % vs 46.9 % on skewed graphs; fails on unseen families. |
| 13 | Dashboard | Tableau sources + dashboard preview (show it). |
| 14 | Limitations & next steps | One 2-core machine, 62 graphs, 6 hub wins, feature cost not timed; next: clusters, more real graphs, cheap sampled features. |

---

## 17. Limitations (say them before you are asked)
- One machine, 2 cores, local mode; on a real cluster the crossover will move.
- 62 graphs, only 6 hubsplit wins, all single-hub and from one generator recipe.
- Unequal repetitions (5 / 3 / 2 runs); half the labels are near-ties.
- AQE skew splitting never fired at these sizes, so it is not really tested.
- Feature-extraction time not measured; the selector does not generalise to unseen graph families (LOSG).
- Pokec validated but not benchmarked; no HDFS test.

## 18. Glossary
**Two-hop motif:** a path a→b→c. **Work of a vertex:** d_in × d_out, the paths through it. **work_sum:** total paths = result size. **Hub:** vertex with work > 1 M. **Skew / straggler:** uneven work / the one slow task. **AQE:** Spark's adaptive query execution. **Salting:** spreading a hot key over random buckets. **Hypercube / Shares:** spreading a hot key over a 2-D grid. **Profiling time:** time spent finding hubs before the join. **Campaign:** one benchmark session with a fixed config. **Correctness gate:** checking every strategy returns the right rows before timing. **Median / IQR:** middle value / spread of the middle half. **Label:** the fastest strategy. **Clear label:** winner's lead exceeds the run spread. **Feature:** a number describing the graph. **Leakage:** using information unavailable at decision time. **LOGO / LOSG / SEED202:** held-out graph / held-out structural group / held-out seed. **Majority baseline:** accuracy of always guessing the most common class. **Balanced accuracy:** average recall over classes. **Regret:** how much slower the pick is than the best. **Oracle:** the best pick known in hindsight. **Selector:** regressor + choose the lowest predicted time.
