# Project briefing — how to explain every part of this project

*A living document. Each section says whether it is **DONE**, **IN PROGRESS** or **PENDING**. Numbers marked ✔ were read from files in the repo; anything else is labelled as a plan or an estimate. Last updated: 2026-10-20, ~18:10 IST (stage 3 of 5 started: sweep collection running).*

---

## 0. The 60-second story

**Question.** Spark can run the same graph query in several different ways. The best way depends on the shape of the graph. Can a model look at a graph's shape *before* running anything and pick the fastest way?

**Query studied.** Two-hop paths: `(a)-[e1]->(b); (b)-[e2]->(c)`. Find every path that goes a → b → c in a directed graph.

**What we did.**
1. Built 60 synthetic graphs (5 shapes × 3 sizes × 4 random seeds) plus one real graph (Pokec).
2. Ran 7 execution strategies on 31 of them, 5 timed runs each (1,085 valid timed runs), and checked every result was correct.
3. Turned each graph into 11 numbers that describe its shape (the *features*).
4. Trained a model to predict each strategy's runtime from those 11 numbers, and to pick the cheapest.
5. Found a surprise: **one strategy family (edge-join) wins almost everywhere**, so the "which is best?" question is nearly one-sided on this machine. We then built a new family of graphs (the *hub sweep*) to test whether any graph shape makes another strategy win.

**One-line honest result so far.** Edge-join is ~1.4× faster than native GraphFrames at the median. A learned model does not (yet) beat "always use edge-join".

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

## 3. Datasets  *(60-graph collection DONE; hub sweep IN PROGRESS)*

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

### 3d. Hub sweep — the new dataset  *(IN PROGRESS)*
- **Why.** In all 31 benchmarked graphs the biggest single vertex holds at most **12 %** of the work. The strategies built for skew had no real skew to fix, so edge-join always won.
- **What.** Graphs with a uniform background plus 1–8 planted hubs, with the hub share of total work swept from about 5 % to about 95 %. The graph builder sets the hub size so the planned share is hit; the **measured** share is stored in each graph file.
- **Probe (3 graphs, one hub holding 50 / 80 / 95 % of the work):** results in section 8 below.
- **Planned size / rows / columns:** *to be filled when collection finishes.*

## 4. Features and preprocessing  *(DONE for the 11 features; sweep rows PENDING)*

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

**Rules we kept:** no runtime or strategy column is ever an input (it would be cheating — the selector must choose *before* running); the feature table has 61 rows (60 graphs + Pokec) and joins to results by `graph_id`.

## 5. How a strategy's runtime was measured  *(DONE)*

1. One fresh Spark session per graph. **Local mode, 2 CPU cores, 7.8 GB RAM, 200 shuffle partitions, broadcast joins off.**
2. **Correctness gate first:** every strategy must return the same path count as `work_sum` and the same checksums as native. A strategy that disagrees is excluded.
3. Then **1 warm-up** run (thrown away) and **5 measured runs**, with the strategy order reshuffled each round so nothing is always first.
4. The number kept is the **median** of the 5 runs of `total_s` = profiling time (finding hubs, for the hub-splitting methods) + query time. Medians resist one bad run; we also keep the spread (IQR).
5. Raw records are append-only; summaries are regenerated from them. Every campaign records its config and machine. Mixing machines or configs is refused by the table builder.
6. **Noise is large:** within one graph/strategy the slowest of 5 runs is a median 1.7× the fastest (up to 3×). That is why we use medians and why close calls are treated as ties.

## 6. The training table  *(DONE for the 31 graphs; sweep rows PENDING)*

`datasets/derived/training/training_table_eligible.csv` ✔ — **31 rows × 69 columns, 21 KB** (one row per graph).

| Group | Columns | Role |
|---|---|---|
| ids | `graph_id, graph_source, role, family, size_tier, seed` | who the row is |
| **inputs** | the 11 features above | what the model sees |
| reference copies | `work_threshold`, `raw_*` (7 columns) | readability, simple rules |
| outcomes | per strategy (×7): `median_total_s`, `total_s_iqr`, `n_valid_runs`, `median_query_s`, `median_profiling_s` (35 columns) | what the model learns to predict |
| answer & tracking | `best_strategy, runner_up_strategy, label_margin_pct, label_clear, label_eligible, campaign_id, config_id, measurement_config_id, machine_id` | label and provenance |

## 7. The models  *(prototype DONE; final notebook PENDING)*

- **Selector (regression).** One Random Forest (300 trees) takes the 11 features and predicts **log(runtime) for every strategy at once**. The selector picks the lowest. If the predicted gain over native is under 5 % it falls back to native (the safe default).
- **Why regression, not classification.** The labels are nearly one class, and the two edge-join variants (AQE on/off) tie within noise. A classifier on `best_strategy` scores **29 %** (RF) / 36 % (logistic regression), *below* the 52 % you get by always guessing the most common class. Regression keeps the *size* of the gaps, so it can say "tied → pick the safe one".
- **How it is tested.** Always on graphs it has **not** seen: leave-one-graph-out and leave-one-family-out. Never split by run, because runs of the same graph are not independent.
- **Baselines every result is compared with:** native (AQE on), native (AQE off), best fixed strategy, a hub rule, and the **oracle** (the best strategy per graph, known only after running everything).
- **Metrics:** *regret* = chosen runtime ÷ best runtime − 1; *speedup* = native ÷ chosen; top-k hit rate; R² of predicted runtime.
- **Classifier (new, planned).** Three classes — edge-join / hub-splitting (salt, hypercube) / native — with edge-join AQE on and off merged, because they tie. *Results: PENDING (stage 4).*

## 8. Results so far  *(31 graphs DONE; probe IN PROGRESS; sweep PENDING)*

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

### 8c. Hub sweep results, classifier accuracy, final comparison  *(PENDING — filled in stages 3–4)*

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
| Why regression instead of classification? | Labels are nearly one class and two classes tie within noise; see section 7. |
| Why not report 97 % accuracy? | "Edge-join vs native" is 97 % by always saying edge-join. Accuracy only means something next to its majority-class baseline. |
| Why only 31 of 60 graphs? | 29 more need 8–14 hours on a 2-core machine; the 4 largest (1.5 B work) probably do not fit. Documented as an exclusion, not an oversight. |
| Why a hub sweep? | To test the one regime the first 31 graphs never covered: one vertex holding most of the work. |
| Why no GPU? | The machine has none, and Spark's GPU path would change what is being measured. Speed comes from fewer strategies/runs and smaller graphs instead. |

## 10. Limitations to say out loud

- One machine, 2 cores, local mode. Real clusters have many cores, where hub-splitting can behave differently.
- 31 graphs (plus the hub sweep) is small; confidence intervals are wide.
- Graph sizes up to about 205 M two-hop paths; nothing above that was timed.
- Skew splitting by AQE never triggered, so AQE's skew handling is untested.
- Feature-extraction time was not measured, so the selector's total overhead is incomplete.
- No HDFS test (not configured here).

## 11. Status board

| Stage | Status |
|---|---|
| 60 graphs + validation | DONE |
| 31-graph benchmark, features, training table | DONE |
| Prototype selector + metrics | DONE |
| Hub-sweep probe (3 graphs) | IN PROGRESS (2 of 3 graphs timed) |
| Hub-sweep generator | DONE (`datagen/hub_sweep.py`) |
| Hub-sweep collection (~50 graphs) | PENDING |
| Classifier + notebook | PENDING |
| Evaluation, charts, report | PENDING |

## 12. Glossary

**Skew** — uneven work across tasks. **Hub** — a vertex with a very large `d_in × d_out`. **AQE** — Spark's adaptive query execution. **Median / IQR** — middle value / spread of the middle half of runs. **Oracle** — the best strategy per graph, known only after running all. **Regret** — how much slower the pick was than the best pick. **Leave-one-graph-out** — train on all graphs but one, test on that one, repeat. **Majority baseline** — accuracy of always guessing the most common answer.
