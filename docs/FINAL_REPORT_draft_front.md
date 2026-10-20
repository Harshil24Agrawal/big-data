# ML-Guided Selection of Graph-Query Execution Strategies in Apache Spark

*Project report (draft front matter; results sections are appended by the final run).*

## Abstract

Spark can execute the same graph pattern query in several ways, and the best way depends on how skewed the graph is. We study the directed two-hop motif `(a)-[e1]->(b); (b)-[e2]->(c)` on Spark 3.5.5 with GraphFrames 0.8.3 and compare GraphFrames' native `find`, a plain edge join, one-dimensional salting, and a hub-aware hypercube (Shares-style) join, with Spark's adaptive query execution (AQE) on and off. We built 60 synthetic graphs plus a 31-graph "hub sweep" with one to eight planted hubs holding 50–99.5 % of all two-hop work, and benchmarked them on one 2-core machine with a correctness gate and median-of-repeated-runs timing. We then trained models that choose a strategy from graph-shape features known before running. The main empirical finding is negative and specific: on a 2-core machine a plain edge join is fastest on the large majority of graphs; hub-splitting becomes faster only when a *single* hub holds roughly 95 % or more of a large amount of work. Classifiers therefore barely beat the majority-class baseline overall; a learned runtime model gives the best regret among the policies we tried, and all scores are reported beside their baselines.

## 1. Introduction

Motif and pattern queries on property graphs reduce to multiway joins over edge tables. When a few vertices have very high degree, a join keyed on the vertex produces a straggler task that dominates runtime (data skew). Systems offer several mitigations: Spark's AQE can split oversized shuffle partitions at run time; salting spreads a hot key over artificial buckets; and hypercube (Shares) partitioning assigns tuples to a multi-dimensional grid of reducers. Each mitigation has set-up cost (profiling to find hubs, replicated data, extra shuffles), so none is best everywhere. The question we ask is practical: *can a model look only at cheap graph statistics and pick the strategy that is actually fastest, and how much does it gain over a fixed default?*

Contributions: (1) a reproducible benchmark protocol (fresh Spark application per graph, correctness gate against the exact path count, interleaved and reshuffled repetitions, medians with IQR, append-only raw records); (2) a synthetic hub-sweep generator that controls the share of work held by hubs; (3) an honest evaluation of classifiers, a runtime regressor and a selector under held-out-graph, held-out-group and held-out-seed splits, each beside its majority-class baseline and an oracle.

## 2. Related work

- **GraphFrames** (Dave et al., GRADES 2016) provides motif finding as relational queries over vertex and edge DataFrames in Spark; it is our `native` baseline. [GraphFrames paper](https://people.csail.mit.edu/matei/papers/2016/grades_graphframes.pdf)
- **Adaptive query execution** in Spark SQL re-optimises at run time, including splitting skewed partitions of a sort-merge join when they exceed configured thresholds. [Databricks AQE blog](https://www.databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html). In our runs AQE skew splitting never triggered because no partition reached the 256 MB threshold at these data sizes.
- **Salting** is the standard manual remedy for hot keys: split the heavy key into several sub-keys and replicate the other side.
- **Shares / hypercube partitioning** (Afrati and Ullman, 2011) assigns tuples to a grid of reducers to optimise multiway joins in MapReduce. [Optimizing multiway joins in a map-reduce environment](https://ieeexplore.ieee.org/abstract/document/5710932). Our hub-aware variant applies the grid only to hub keys.
- **Learned optimisers.** Bao steers an existing optimiser by learning which hint sets to use ([Bao, SIGMOD 2021](https://people.csail.mit.edu/tatbul/publications/bao_sigmod21.pdf)); Lero learns to rank candidate plans ([Lero, VLDB 2023](https://www.vldb.org/pvldb/vol16/p1466-zhu.pdf)). Our selector is in the same spirit — predict cost per option, choose the cheapest, fall back to a safe default — but chooses among whole execution strategies using pre-execution graph statistics rather than plan features.

## 3. Method

**Query and strategies.** The query counts directed two-hop paths. Strategies are `<variant>__aqe_<on|off>`: `native` (GraphFrames `find`), `edgejoin` (plain self-join on `e1.dst = e2.src`), `salt1d` (hub keys split into 100 buckets), `hypercube` (hub and non-hub branches), `hypercube_unified` (single pass, same hub planner). AQE-on pins skew join on with a 256 MB threshold and factor 5.

**Work and hubs.** For vertex `v`, the number of two-hop paths through it is `d_in(v)·d_out(v)`; `work_sum` is their sum, i.e. the exact result size. A *hub* is a vertex with work above 1 000 000 (the planner's default).

**Graphs.** (a) 60 synthetic graphs: five families (uniform, balanced power law, in-heavy, out-heavy, multi-hub) × three sizes (0.1/0.5/1 M vertices) × four seeds, 31 of them benchmarked; (b) a hub sweep with a uniform background (7 random edges per vertex, 100 k vertices) plus planted hubs sized so that the planned share of work sits in the hubs, at 1–8 hubs and shares 50 %–99.5 %, two seeds.

**Measurement.** One fresh Spark application per graph, local mode, 2 vCPU, 7.8 GB, 200 shuffle partitions, broadcast off. A correctness gate requires every strategy to reproduce the path count and checksums. Then one warm-up and 5 (original graphs) or 3 / 2 (sweep graphs) measured runs, with strategy order reshuffled each round. The recorded cost is the median of `total_s = profiling + query` time; IQR is kept. All rows come from one machine and one Spark configuration (the builder refuses to merge otherwise); only the number of repetitions differs and is recorded in a `protocol` column.

**Features (12 inputs).** Log vertex and edge counts, active-vertex fraction, in/out Gini, log in/out 99.9th-percentile degree, log total work, share of work through the busiest vertex, log number of work hubs, share of work in hubs, and log absolute work in hubs. No runtime or strategy information is ever an input.

**Models.** Classification of the best *class* (edge-join / hub-splitting / native): majority guess, a fixed rule, logistic regression, depth-2 decision tree, random forest, extra-trees. Regression: a multi-output random forest predicts log median runtime for every strategy; the selector takes the argmin, optionally falling back to native unless the predicted gain exceeds 5 %.

**Evaluation.** Only on unseen graphs: leave-one-graph-out (LOGO), leave-one-structural-group-out (LOSG) and a held-out generator seed (SEED202). Reported: accuracy beside the majority baseline, macro-F1, regret (chosen ÷ best − 1), geometric-mean speed-up over native, and an oracle. Outcomes are look-ups of measured medians, not new runs.
