# Project briefing — how to explain every part of this project

*A living document. Each section says whether it is **DONE**, **IN PROGRESS** or **PENDING**. Numbers marked ✔ were read from files in the repo; anything else is labelled as a plan or an estimate. Last updated: 2026-10-20, ~23:35 IST (all stages complete; final numbers from the executed notebook).*

---

## 0. The 60-second story

**Question.** Spark can run the same graph query in several different ways. The best way depends on the shape of the graph. Can a model look at a graph's shape *before* running anything and pick the fastest way?

**Query studied.** Two-hop paths: `(a)-[e1]->(b); (b)-[e2]->(c)`. Find every path that goes a → b → c in a directed graph.

**What we did.**
1. Built 60 synthetic graphs (5 shapes × 3 sizes × 4 random seeds) plus one real graph (Pokec).
2. Ran 7 execution strategies on 31 of them, 5 timed runs each (1,085 valid timed runs), and checked every result was correct.
3. Turned each graph into 11 numbers that describe its shape (the *features*).
4. Trained a model to predict each strategy's runtime from those 11 numbers, and to pick the cheapest.
5. Found a surprise: **one strategy family (edge-join) wins almost everywhere**, so the "which is best?" question was nearly one-sided on this machine. We then built a new family of graphs (the *hub sweep*, 31 graphs) to test whether any graph shape makes another strategy win.
6. Result: **hub-splitting wins only when ONE vertex carries roughly 1e8+ of the two-hop paths** (6 graphs, up to 2× faster than edge-join and 3× faster than native). With 2, 4 or 8 equal hubs it never wins.
7. Final models on all 62 graphs (31 original + 31 sweep), tested only on unseen graphs.

**One-line honest result.** Edge-join is the right default (faster than native on 56 of 62 graphs, median 1.38×). The best classifier reaches **85.5 %** accuracy leave-one-graph-out, but "always say edge-join" already gets **80.6 %**, so the gain is small. The runtime-regression *selector* is the real win: mean regret **4.6 %** vs 8.3 % for always-edge-join (oracle 0 %), and 3.0 % vs 46.9 % on single-hub graphs. It does not generalise to graph families it has never seen, and it ignores the cost of computing its own features.

---

## 1. The problem in plain words  *(DONE)*

- A **graph** is a list of edges `src → dst`. A **two-hop path** is two edges chained: `a→b` then `b→c`.
- To find all two-hop paths, Spark joins the edge list with itself where `e1.dst = e2.src`.
- **The count of paths** through a vertex `v` is `d_in(v) × d_out(v)` (edges coming in × edges going out). Adding this over all vertices gives `work_sum`, **the exact number of paths**. We call a vertex with a very large product a **hub**.
- **Why hubs hurt.** Spark splits the join by vertex key. One hub with a huge product lands on **one** task, which can take far longer than all the others. This is called **skew**.
- **Existing fixes** (the strategies below) try to spread a hub's work over many tasks, but they cost extra setup. So the best choice depends on how skewed the graph is.

## 2. The strategies we compare  *(DONE)*

Each strategy is named `<method>__aqe_<on|off>`. AQE is Spark's adaptive query execution, which can split skewed partitions automatically when a partition is above a size threshold (we pin it to 256 MB and 5× the median).

| Strategy | Plain meaning |
|---|---|
| `native` | GraphFrames' built-in `find()`. The reference. |
| `edgejoin` | A plain join of the edge table with itself on `e1.dst = e2.src`, with the same output shape. |
| `salt1d` | **Salting.** Hub rows are split into 100 random buckets and the other side is copied to every bucket, so a hub's work spreads over 100 tasks. |
| `hypercube` | **Hypercube (Shares-style).** Hubs go to a 2-D grid of cells, other vertices go through an ordinary join, then results are merged. |
| `hypercube_unified` | The same grid idea in a single pass (hubs found by the same planner). |

*Not used:* `hypercube_vjoin` (excluded to keep the pilot small). Both AQE settings run for `native` and `edgejoin`; the other three run with AQE off. That gives **7 strategies**.

## 3. Datasets  *(DONE)*

### 3a. The 60 synthetic graphs ✔

| Fact | Value |
|---|---|
| Graphs | 60 = 5 families × 3 sizes × 4 seeds (101, 202, 303, 404) |
| Total size | 224 M edges, 1.3 GB (Parquet, columns `src`, `dst`) |
| Sizes (vertices / edges) | small 100 k / 0.7 M · medium 500 k / 3.5 M · large 1 M / 7 M (≈7 edges per vertex) |
| Two-hop work (`work_sum`) | from 4.5 M (small) up to 1.5 B (powerlaw large) |
| Cleaning rules | directed, **no self-loops, no duplicate edges**, vertex ids 0 … n−1 |
| Reproducible | fixed seed per graph; hashes stored; regenerating gives identical bytes |

**The 5 families (why each exists):**

| Family | How it is built | What it tests |
|---|---|---|
| `uniform` | every vertex has about 7 in and 7 out edges | no skew (the control) |
| `powerlaw_balanced` | a few vertices have many edges on **both** sides (Chung-Lu, γ = 2.5) | realistic heavy-tailed networks |
| `in_heavy` | 25 % of all edges point into a few hubs | many arrivals at one place |
| `out_heavy` | the mirror image | many departures from one place |
| `multi_hub` | 10 / 20 / 40 hubs, each with about 2,000 in and 2,000 out edges | several equal heavy vertices |

Each graph has a check that its shape really is what the family name says (e.g. uniform: degree spread small).

### 3b. Real graph ✔
**Pokec** (social network, SNAP): 1.63 M users, 30.6 M edges, 1.78 B two-hop paths. Validated, but **not benchmarked** (too heavy for this machine) and **no HDFS test** was done. It is used only as an outside example, never for training.

### 3c. What was actually benchmarked ✔
**31 of the 60 graphs**, 7 strategies each, 5 timed runs after 1 warm-up → **1,085 valid timed runs**, 0 failures, 0 timeouts, 0 wrong results. The other 29 are generated and validated but not run, because it would take 8–14 more hours on this machine (decision of the project owner; see status doc).

### 3d. Hub sweep — the new dataset  *(DONE: 31 graphs)*
- **Why.** In all 31 benchmarked graphs the biggest single vertex holds at most **12 %** of the work. The strategies built for skew had no real skew to fix, so edge-join always won.
- **What.** Graphs with a uniform background plus 1–8 planted hubs, with the hub share of total work swept from about 5 % to about 95 %. The graph builder sets the hub size so the planned share is hit; the **measured** share is stored in each graph file.
- **Probe (3 graphs, one hub holding 50 / 80 / 95 % of the work):** results in section 8 below.
- **What was built ✔ (`datasets/hubsweep/`, `datagen/hub_sweep.py`):** **31 graphs benchmarked** (28 in sweeps A–C plus 3 extra single-hub graphs; the 3 probe graphs are repeats of A/B points and are not used), all 100 k vertices and about 0.7 M edges (a uniform background of 7 random edges per vertex, then 1–8 planted hubs). Each hub gets `d` distinct in-neighbours and `d` out-neighbours with `d = round(sqrt(share/(1−share) · n·7·7 / hubs))`, so it holds the planned share of the total work. Same cleaning rules as the 60 graphs (no self-loops, no duplicates). Every graph passed the structure check and a re-read by Spark.

| Set | Hubs H | Planned hub share of work | Seed | Graphs |
|---|---|---|---|---:|
| sweep A | 1, 2, 4, 8 | 80 %, 90 %, 95 % | 101 | 12 |
| sweep B | 1, 2, 4, 8 | 50 %, 98 % | 101 | 8 |
| sweep C (held-out seed) | 1, 4 | 50 %, 90 %, 95 %, 98 % | 202 | 8 |
| big single hub | 1 | 99 % (seeds 101 and 202), 99.5 % (seed 101) | 101/202 | 3 |

  Total two-hop work per graph: 50 % ≈ 9.9 M · 80 % ≈ 24.6 M · 90 % ≈ 49 M · 95 % ≈ 98 M · 98 % ≈ 245 M (more hub share ⇒ more total paths, because the hub's product d² grows). The measured share through the biggest vertex ≈ planned share ÷ H.
- **Measurement protocol for the sweep (cheaper than the original, and recorded as such):** sweep A used 4 strategies (edge-join AQE off, native AQE on, salt1d, unified hypercube); salt1d was dropped from B/C/big because it was never competitive (always 1.5–3× slower, 43 graphs measured). **1 warm-up + 3 timed runs** (A, B, C) or **2 timed runs** for the three 0.5–1 B-path graphs (each strategy takes 1–4 minutes per run there); same machine and same Spark settings; about 6–10 minutes per graph (30 min for the biggest). The column `protocol` in the combined table says which rows used 5 runs (original 31) and which used 3 (sweep). A simulation on the existing data showed 3 runs per strategy gives about 7.6 % median error of the runtime estimate and 2.9 % regret versus 6.4 % / 1.7 % for 4 runs (reused-run caveat: optimistic).
- **Final size:** 31 original + 31 sweep = **62 graphs → 62 rows × 43 columns** in the combined table. Total timed sweep runs: 306, **0 failed, 0 timeouts, 0 wrong results**. Spark's AQE skew splitting never fired (105 strategy summaries).

## 4. Features and preprocessing  *(DONE)*

**Preprocessing:** (1) generator removes self-loops and duplicate edges; (2) one pass computes degrees and the work per vertex (`d_in × d_out`); (3) summary numbers are stored in each graph's `graph.json`; (4) the feature builder applies `log1p` to counts and leaves ratios unchanged. It reads **only** the graph, never any runtime.

**The 11 input features** (`datasets/derived/features/graph_features.csv`):

| # | Feature | Plain meaning | Why it matters |
|---|---|---|---|
| 1 | `log_num_vertices` | graph size (log) | bigger graph → more work |
| 2 | `log_num_edges` | same, in edges (log) | *identical to #1 on our graphs (edges ≈ 7 × vertices); kept so Pokec stays representable* |
| 3 | `active_fraction` | share of vertices that touch any edge | isolated vertices do no work |
| 4 | `in_gini` | how unevenly arrivals are spread (0 = even, 1 = one vertex) | skew in who receives |
| 5 | `out_gini` | same for departures | skew in who sends |
| 6 | `log_in_p999` | in-degree of the top 0.1 % vertices (log) | how fat the tail is |
| 7 | `log_out_p999` | same for out-degree | |
| 8 | `log_work_sum` | total number of two-hop paths (log) | total workload |
| 9 | `work_max_share` | share of all paths through the single busiest vertex | **the main skew signal** |
| 10 | `log_num_work_hubs` | how many vertices exceed 1 M paths each (log) | how many hubs the planner would find |
| 11 | `work_in_hubs_share` | share of all paths that run through those hubs | how much work is skewed |

**Two derived inputs added in the notebook (13 inputs total).** `log_hub_work` = log of the paths that run through hubs (`work_sum × work_in_hubs_share`); `log_max_work` = log of the paths through the single busiest vertex (`work_sum × work_max_share`). Why: an edge join's slowest task is the one holding the busiest vertex, so *its* work decides the runtime floor. `log_max_work` became the most important input (importance 0.19). Both come from the existing features only (no runtime). Caveat: `log_max_work` was chosen after seeing the sweep; the notebook reports 11-feature results beside the 13-feature ones.

**Rules we kept:** no runtime or strategy column is ever an input (it would be cheating — the selector must choose *before* running); the feature table has 61 rows (60 graphs + Pokec) and joins to results by `graph_id`.

## 5. How a strategy's runtime was measured  *(DONE)*

1. One fresh Spark session per graph. **Local mode, 2 CPU cores, 7.8 GB RAM, 200 shuffle partitions, broadcast joins off.**
2. **Correctness gate first:** every strategy must return the same path count as `work_sum` and the same checksums as native. A strategy that disagrees is excluded.
3. Then **1 warm-up** run (thrown away) and **5 measured runs**, with the strategy order reshuffled each round so nothing is always first.
4. The number kept is the **median** of the 5 runs of `total_s` = profiling time (finding hubs, for the hub-splitting methods) + query time. Medians resist one bad run; we also keep the spread (IQR).
5. Raw records are append-only; summaries are regenerated from them. Every campaign records its config and machine. Mixing machines or configs is refused by the table builder.
6. **Noise is large:** within one graph/strategy the slowest of 5 runs is a median 1.7× the fastest (up to 3×). That is why we use medians and why close calls are treated as ties.

## 6. The training tables  *(DONE)*

`datasets/derived/training/training_table_eligible.csv` ✔ — **31 rows × 69 columns, 21 KB** (one row per graph).

| Group | Columns | Role |
|---|---|---|
| ids | `graph_id, graph_source, role, family, size_tier, seed` | who the row is |
| **inputs** | the 11 features above | what the model sees |
| reference copies | `work_threshold`, `raw_*` (7 columns) | readability, simple rules |
| outcomes | per strategy (×7): `median_total_s`, `total_s_iqr`, `n_valid_runs`, `median_query_s`, `median_profiling_s` (35 columns) | what the model learns to predict |
| answer & tracking | `best_strategy, runner_up_strategy, label_margin_pct, label_clear, label_eligible, campaign_id, config_id, measurement_config_id, machine_id` | label and provenance |

**Combined table (new) `datasets/derived/training/combined_table.csv`** — built by `ml/build_combined_table.py`: the 31 original graphs plus all 31 sweep graphs, **only the 3 strategies measured everywhere** (edge-join AQE off, native AQE on, unified hypercube; salt1d kept in one diagnostic column), **62 rows × 43 columns**:

| Group | Columns |
|---|---|
| ids / provenance | `graph_id, family, size_tier, seed, protocol, source_campaign, machine_id, cmp_config, hub_count, hub_work_share_planned` |
| inputs | the same 11 features + `raw_work_sum` (the notebook adds 2 derived inputs → 13) |
| outcomes (×3 strategies) | `median_total_s, total_s_iqr, median_query_s, median_profiling_s, n_valid_runs` (15 columns) + `median_total_s__salt1d__aqe_off` (diagnostic) |
| answer | `best_strategy`, **`best_class`** (edgejoin / hubsplit / native), `runner_up_strategy, label_margin_pct, label_clear` |

The builder refuses to combine rows from different machines or different Spark settings; the only allowed difference is the number of timed runs.

## 7. The models  *(DONE; final run executed in `notebooks/selector.ipynb`)*

**Three models, three different jobs:**

| | Classifier | Runtime regressor | Selector |
|---|---|---|---|
| **Predicts** | the fastest *class*: edge-join / hub-splitting / native | log of the median runtime of each of the 3 strategies | picks the strategy with the lowest predicted runtime |
| **Inputs** | 13 numbers per graph (11 shape features + `log_hub_work` + `log_max_work`) | the same 13 | same as the regressor |
| **Models** | majority guess, fixed rule, logistic regression, depth-2 tree, one-threshold stump, random forest (300 trees), extra-trees (300) | one multi-output random forest (300 trees) | the regressor + optional "native fallback" (stay with native unless the predicted gain is > 5 %) |
| **Why this model** | classes are tiny and imbalanced (50/6/6), so we include very simple models (stump, tree) that can still work with 6 examples, plus forests as the standard strong baseline | keeps the *size* of the differences ("tied → pick the safe one"), which a hard label throws away; this is the Bao/Lero idea of "predict cost, then choose" | what a user actually needs: a choice |

- **What a class is.** `edgejoin` = edge-join AQE off. `hubsplit` = unified hypercube. `native` = GraphFrames `find`, AQE on. Salting was dropped (never fastest). The AQE-on/off edge-join pair was merged earlier because they tie within noise (in the 31-graph data).
- **How many inputs, how many rows.** 13 inputs × 62 graphs. 62 rows is small, which is why every score is next to its majority baseline and the folds are by *graph*, never by run.
- **How it is tested (never on seen graphs).** **LOGO** = leave one graph out. **LOSG** = leave one structural group out (a whole family, or all sweep graphs with the same number of hubs, is hidden) — the harshest test. **SEED202** = train on all but the nine seed-202 sweep graphs, test on those.
- **Baselines every result is compared with:** the majority guess (always edge-join), always-native, always-hypercube, a fixed hub rule, and the **oracle** (best strategy per graph, only known after running all three).
- **Metrics:** accuracy (+ balanced accuracy, macro-F1), *regret* = chosen runtime ÷ best runtime − 1, *speed-up* = native ÷ chosen, top-k hit rate, R² of predicted runtime, total time over all graphs.
- **Notebook:** `notebooks/selector.ipynb` (built by `notebooks/_build_selector_notebook.py`, executed with 300 trees in 2 minutes). Sections: data → label balance → runtime vs hub share → crossover chart → classifier (3 held-out schemes) → feature ablation → skewed-regime test → confusion matrix & importance → regressor → selector → policy comparison → overheads → summary. Outputs in `datasets/derived/selector_v2/`.

## 8. Results  *(DONE)*

### 8a. The 31-graph benchmark ✔
| Finding | Number |
|---|---|
| Fastest by median total time | edge-join AQE on 16 · AQE off 14 · native AQE on 1 |
| Edge-join vs native (AQE on) | median **1.45×** faster, up to 2.1× |
| Salting / hypercube vs best edge-join | **never faster**: 1.5–2.6× slower |
| Clearly separated winners (gap bigger than noise) | only **7 of 31** |
| Skew splitting by AQE | never happened (no partition reached 256 MB) |

**Prototype selector (leave-one-graph-out, 31 graphs):**

| Policy | Mean regret | Speedup vs native | Winner chosen |
|---|---:|---:|---:|
| oracle (best possible) | 0 % | 1.44× | 31/31 |
| always edge-join AQE on | 10.1 % | 1.32× | 16/31 |
| **ML selector** | **14.8 %** | 1.27× | 12/31 |
| native AQE on | 45.3 % | 1.00× | 1/31 |
| native AQE off | 56.0 % | 0.94× | 0/31 |
| hub rule | 84.7 % | 0.80× | 1/31 |

Runtime-prediction quality on unseen graphs: **R² 0.86**, median error 10 %, per-graph ranking correlation 0.87, true best strategy inside the model's **top 2: 90 %**, top 3: 100 %.

**Chart `datasets/derived/selector/regret_by_graph.png`:** one group of bars per held-out graph; each bar is how much slower a policy's choice was than the best choice (0 = perfect). Tall green bars = the hub rule is often bad; the blue bar (native) is always taller than the others; orange (always edge-join) and red (ML) are close.

### 8b. Probe — can any graph shape make a hub-splitting strategy win?  *(DONE)*
Single-hub graphs, 100 k vertices, 1 hub holding a planned 50 / 80 / 95 % of the work; 2 timed runs per strategy (probe only — fewer runs than the real protocol). Median `total_s` in seconds:

| Hub share of work | edge-join | native | salt1d | hypercube | hypercube_unified |
|---|---:|---:|---:|---:|---:|
| 50 % (work 9.9 M) | 15.2 | **10.8** | 21.2 | 30.4 | 16.5 |
| 80 % (work 24.6 M) | **10.4** | 17.9 | 27.1 | 34.8 | 22.3 |
| 95 % (work 98 M) | **20.8** | 34.5 | 42.8 | 30.2 | 21.5 |

*Reading:* even when one vertex holds 80 % of all work, salting and hypercube are 2–3× slower than a plain join on this 2-core machine. They pay a setup cost, and with 2 cores there are not enough parallel workers to win it back. At 95 % the unified hypercube's *query phase* alone is faster than edge-join (12.6 s vs 20.8 s), but its profiling step costs ~8.9 s, which cancels the gain; the crossover needs more total hub work (about 245 M at 98 %), which is why the sweep goes up to 98 %. 2-run medians are noisy (edge-join at 50 % is 7.5 s and 22.8 s), so differences under about 30 % in this table are not reliable.

### 8c. Hub-sweep results  *(DONE)*
Median total time (s) per graph, one hub (H = 1), seed 101:

| Hub share | paths through the hub | edge-join | native | hypercube | winner |
|---|---:|---:|---:|---:|---|
| 50 % | 4.9 M | 11.2 | 12.6 | 14.2 | edge-join (tie with native) |
| 80 % | 20 M | 9.1 | 18.7 | 15.6 | edge-join |
| 90 % | 44 M | 13.4 | 21.4 | 18.7 | edge-join |
| 95 % | 93 M | 21.4 | 33.2 | 21.5 | tie (edge-join by 0.1 s); seed 202: **hypercube 18.9 vs 22.8** |
| 98 % | 240 M | 43.4 | 71.3 | **38.5** | **hypercube** (seed 202: 32.9 vs 43.2) |
| 99 % | 485 M | 82.6 | 132.6 | **47.2** | **hypercube** (seed 202: 40.6 vs 79.7) |
| 99.5 % | 976 M | 154.8 | 251.2 | **80.5** | **hypercube** |

With 2, 4 or 8 hubs (work split between them) edge-join wins at every share, even 98 %. **Reading:** a plain join's slowest task holds the busiest vertex; hub-splitting removes that straggler but costs ~7–10 s of profiling. On 2 cores the straggler only outweighs that cost once one vertex carries about 1e8+ paths.

**Chart `crossover_max_vertex_work.png`:** one dot per graph; x = paths through the busiest vertex (log scale), y = hypercube time ÷ edge-join time (log). Below the dashed line y = 1 the hypercube is faster. All six wins sit on the right (≥ 93 M); every graph left of ~90 M is above the line. **Chart `runtime_vs_hub_share.png`:** raw times against hub share; the times explode only past 98 %. **Chart `label_balance.png`:** how many graphs each class wins (50 / 6 / 6) and in which family — shows the class imbalance that limits accuracy.

### 8d. Final classifier results (62 graphs, held-out)  *(DONE)*
| Model | LOGO accuracy | balanced acc. | macro-F1 |
|---|---:|---:|---:|
| **majority (always edge-join)** | **80.6 %** | 33 % | 0.30 |
| fixed rule (hubs → hypercube, else native) | 14.5 % | 50 % | 0.17 |
| logistic regression | 75.8 % | 75 % | 0.66 |
| depth-2 tree | 51.6 % | 56 % | 0.53 |
| stump on busiest-vertex work | **85.5 %** | 60 % | 0.56 |
| random forest | 83.9 % | 69 % | 0.68 |
| extra-trees | **85.5 %** | 70 % | 0.70 |

- **Honest reading.** 85.5 % beats the 80.6 % baseline by ~5 points (≈ 3 graphs of 62). The models find hub-splitting (5 of 6) but not "native" (its 6 wins are mostly ties with edge-join). On the 9 held-out-seed graphs: 77.8 % vs 55.6 % majority. LOSG (a whole structural group hidden): **no model beats the majority** (80.6 %) — the models interpolate, they do not extrapolate.
- **The 85 % target** is reached by two models on LOGO, but it is a ~5-point gain over a trivial guess; do not present it as "classifier solves the problem".
- **What the model actually uses** (random-forest importance): paths through the busiest vertex (0.19), share of work in the busiest vertex (0.17), total work (0.12), share in hubs (0.12). Chart `classifier_confusion_importance.png` shows the confusion matrix (stump: 48/2/0, 1/5/0, 6/0/0 for edge-join/hubsplit/native true classes) and this importance bar chart.

### 8e. Final selector results (62 graphs, leave-one-graph-out)  *(DONE)*
| Policy | Class acc. | Mean regret | Speed-up vs native | Sum of times |
|---|---:|---:|---:|---:|
| always native | 9.7 % | 49.5 % | 1.00× | 1810 s |
| always edge-join (best fixed) | 80.6 % | 8.3 % | 1.34× | 1226 s |
| hub rule | 14.5 % | 56.9 % | 0.95× | 1490 s |
| **ML selector** | 82.3 % | **4.6 %** | **1.38×** | **1070 s** |
| ML + native fallback | 79.0 % | 6.1 % | 1.36× | 1079 s |
| oracle | 100 % | 0 % | 1.43× | 1037 s |

- Runtime regressor on unseen graphs: R² 0.889 (log scale), median error 10.1 %, per-graph ranking correlation 0.855, best strategy in its top 2 on 100 % of graphs.
- The selector closes **83 %** of the total-time gap between the best fixed policy and the oracle. On single-hub graphs (busiest vertex ≥ 90 %): regret **3.0 % vs 46.9 %** for always-edge-join. Held-out seed: 7.3 % vs 18.5 %.
- Under LOSG the selector is **no better than always-edge-join** (8.3 % both). The native fallback did not help (native is rarely right).
- **Overhead:** inference 14 ms/graph. Feature extraction was not timed; if it cost as much as the hypercube's profiling pass (9.1 s), the speed-up vs native would drop from 1.36× to 0.84× — features must be cheap (cached/sampled degree counts) for the selector to pay off on 10–25 s jobs; on the 0.5–1 B-path graphs it pays easily (35–75 s saved vs edge-join).
- **Chart `regret_and_total_time.png`:** left, each graph (sorted by hub share) gets one bar per policy showing how much slower than the best choice it was; blue (native) and green (rule) are tall almost everywhere; orange (always edge-join) is tall only on the right where hub-splitting wins (up to ~96 %); red/purple (ML) are near zero except where it misses. Right: total time over all graphs per policy — ML ≈ oracle. **Chart `regressor_pred_vs_actual.png`:** predicted vs measured runtime of held-out graphs on log axes; dots near the diagonal = good prediction.

## 9. Decisions and the reasons behind them (likely questions)

| Question | Answer |
|---|---|
| Why synthetic graphs? | Real graphs with known skew levels do not exist ready-made; synthetic ones let us control and label the shape, and are reproducible from a seed. Pokec is kept as a real check. |
| Why 5 families? | They cover no skew, balanced heavy-tail, in-skew, out-skew and several equal hubs. |
| Why work = d_in × d_out? | It is the exact number of two-hop paths through a vertex, so it measures real join work, not just degree. |
| Why a hub threshold of 1 M? | It is the repo planner's own default; we keep it so features match what the planner does. |
| Why 5 runs and the median? | Timings are noisy (up to 3×); the median of 5 resists outliers. |
| Why a warm-up run? | The first run is slower (code loading, caching). |
| Why one machine? | Runtimes depend on hardware; mixing machines would teach the model the machine, not the strategy. |
| Why a correctness gate? | A fast but wrong strategy is worthless; every strategy must reproduce the same path count and checksums. |
| Why not train on single runs? | Repeated runs of one graph are not independent; that would inflate accuracy. |
| Why regression instead of classification? | Labels are 81 % one class and the other classes are tiny or noisy; the regressor keeps the size of the gaps. We still report the classifier — and show it only slightly beats the majority guess. |
| Why not just say "85 % accuracy"? | Because always saying edge-join gets 80.6 %. We report 85.5 % **and** 80.6 % together; the real gain is in regret (4.6 % vs 8.3 %), and on single-hub graphs (3 % vs 47 %). |
| Why not report 97 % accuracy? | "Edge-join vs native" is 97 % by always saying edge-join. Accuracy only means something next to its majority-class baseline. |
| Why drop salting later? | It was never fastest on 43 graphs (always 1.5–3× slower), and it was the slowest part of each campaign. Its times stay in the table as a diagnostic column. |
| Why 2 or 3 runs on the sweep instead of 5? | Time: the big graphs take 1–4 minutes per run. A simulation on the 31-graph data gave ~7.6 % median error for 3 runs vs ~6.4 % for 4; the big-graph margins (≈ 2×) are far bigger than the noise. The `protocol` column records it. |
| Why add `log_max_work`? | The edge join's slowest task is the busiest vertex's; its work sets the runtime floor. It is derived from existing features only. It was chosen after seeing the sweep, so we also report 11-feature results. |
| Why only 31 of 60 graphs? | 29 more need 8–14 hours on a 2-core machine; the 4 largest (1.5 B work) probably do not fit. Documented as an exclusion, not an oversight. |
| Why a hub sweep? | To test the one regime the first 31 graphs never covered: one vertex holding most of the work. |
| Why no GPU? | The machine has none, and Spark's GPU path would change what is being measured. Speed comes from fewer strategies/runs and smaller graphs instead. |

## 10. Limitations to say out loud

- One machine, 2 cores, local mode. Real clusters have many cores, where hub-splitting can behave differently.
- 62 graphs is small; only 6 are hub-splitting wins, all single-hub, all from one generator recipe; confidence intervals are wide.
- Graph sizes: up to 205 M paths in the original families and up to 0.98 B in the single-hub sweep; vertex counts only 100 k–1 M.
- Skew splitting by AQE never triggered, so AQE's skew handling is untested.
- Feature-extraction time was not measured; with a 9 s proxy the selector's speed-up vs native would vanish on short jobs (section 8e).
- The selector does not generalise to graph families it has never seen (LOSG), and the 85.5 % accuracy is only ~5 points above the majority guess.
- Unequal repetitions (5/3/2 runs) across sets; one empty campaign record (`hubsweep-sweep_big-31949d`, 0 graphs, skipped because Spark validation had not run yet) is kept in the raw folder and unused.
- No HDFS test (not configured here).

## 11. Status board

| Stage | Status |
|---|---|
| 60 graphs + validation | DONE |
| 31-graph benchmark, features, training table | DONE |
| Prototype selector + metrics (31 graphs) | DONE |
| Hub-sweep probe, generator, 31 sweep graphs benchmarked | DONE |
| Combined table (62 rows) + notebook executed (300 trees) | DONE |
| Evaluation, charts, final report (`docs/FINAL_REPORT.md`) | DONE |
| Pushed to `mine` (Harshil24Agrawal/big-data); `origin` (Rayirth556) not pushed without your go-ahead | DONE / WAITING |

## 12. Glossary

**Skew** — uneven work across tasks. **Hub** — a vertex with a very large `d_in × d_out`. **AQE** — Spark's adaptive query execution. **Median / IQR** — middle value / spread of the middle half of runs. **Oracle** — the best strategy per graph, known only after running all. **Regret** — how much slower the pick was than the best pick. **Leave-one-graph-out** — train on all graphs but one, test on that one, repeat. **Majority baseline** — accuracy of always guessing the most common answer. **Hub-splitting** — strategies that spread a hub's work (here: the unified hypercube). **Straggler** — the one task that finishes last and sets the runtime. **LOGO / LOSG / SEED202** — leave-one-graph-out / leave-one-structural-group-out / held-out generator seed. **Stump** — a decision tree with a single yes/no threshold.

---

## 13. Notebook charts added (v2.1) and how to re-run it

`notebooks/selector.ipynb` now embeds every chart inline and re-runs to identical numbers.

**Re-run:** `pip install -r notebooks/requirements.txt`, then open the notebook (or `cd notebooks; python -m nbconvert --to notebook --execute --inplace selector.ipynb`) and run all cells. It needs only `datasets/derived/training/combined_table.csv` (committed). Seeds are fixed, `n_jobs=1`, no sampling. The last cell compares 92 key numbers (accuracies, balanced accuracies, regrets, speed-ups, R², top-2) to `notebooks/expected_results.json` and prints `REPRODUCED`. Only wall-clock figures (inference ms) differ between runs. `NB_TREES=20` gives a quick run (the check is then skipped). `NB_UPDATE_EXPECTED=1` rewrites the expected file after deliberate changes.

**New charts and how to explain them**
- *Classifier comparison* (`classifier_comparison.png`): accuracy, balanced accuracy and macro-F1 for 7 models under LOGO, LOSG and SEED202, with the majority line. Message: on LOGO the stump (85.5%) and extra-trees (85.5%) beat the 80.6% majority, random forest 83.9%, logreg is lower on accuracy (75.8%) but best on balanced accuracy (0.753) because it finds the rare classes. Under LOSG (unseen families) nothing beats majority. Accuracy alone flatters "always edge-join", so quote balanced accuracy too.
- *Per-class recall heatmap* (`classifier_recall_heatmap.png`): hubsplit is found 83% of the time by the forest/stump (logreg 100%), native only 33% (stump 0%). Native wins are the hard class.
- *Feature-space map* (`feature_space.png`): hub-split wins sit at high busiest-vertex share / work; edge-join fills the rest; native wins overlap with edge-join (the noise-level labels).
- *Speed-up by group* (`speedup_by_group.png`): native/edge-join and native/hypercube ratio per structural group on a log axis. Edge-join beats native almost everywhere; hypercube wins only in the single-hub groups.
- *Regret CDF and speed-up over native* (`regret_cdf_speedup.png`): share of graphs within x% of the best choice per policy. The ML selector tracks the oracle on about 82% of graphs and its worst case is far below always-native/hypercube.
- *Label noise* (`label_noise.png`): many winners lead the runner-up by only a few percent; 32 of 62 labels are clear. This bounds attainable accuracy.
- *Correlations and regressor error* (`correlations_and_errors.png`): heavily correlated size/work features (why trees are used and importances are shared); regressor error is centred on 0.

Numbers in the report are unchanged by this upgrade (verified by the reproducibility cell).

**Update (v2.2):** the notebook now has a **Tableau section (8)**. It exports the four Tableau CSVs plus a `selector_results.hyper` extract, loads them back, and draws the dashboard preview (`datasets/derived/tableau/dashboard_preview.png`). The reproducibility check (section 9) now covers 101 numbers. For the full step-by-step flow and a file-by-file map, read **`docs/PROJECT_WALKTHROUGH.md`**.
