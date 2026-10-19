# Phase 1 — Data Collection and Loading

> Pilot status and limitations: see [`PHASE1_REPORT.md`](PHASE1_REPORT.md).

Reproducible graph inputs and raw benchmark runs for ML-guided selection of
2-hop motif execution strategies, `(a)-[e1]->(b); (b)-[e2]->(c)`, in Spark.

Phase 1 builds on the existing code rather than replacing it:

| Need | Existing code reused | Phase 1 addition |
|---|---|---|
| Strategies | `hypergraph.HyperGraphFrame` (split 2D hypercube), `bench/salting_1d.py` (`EdgeJoinGraphFrame`, `Salted1DGraphFrame`), GraphFrames `find()` | `bench/hypercube_unified.py` (map-literal unified hypercube from `pokec_benchmark.py`, now returning the same output schema and using the package planner) |
| Harness | `bench/bench.py` (Spark setup, REST stage metrics, task quantiles, env info), `bench/run_single_ablation.py` (`analyze_plan`, `toRdd().count()` action) | `bench/phase1_worker.py`, `bench/run_phase1.py`, `bench/phase1_metrics.py` |
| Pokec | `preprocessing.py` (tab text → int `src,dst` → Parquet) | `datagen/pokec.py` (same transform + validation, cleaning counts, HDFS check) |
| Graphs | — (only inline test generators existed) | `datagen/` (5 families, manifest, validation) |

---

## 1. Setup

Spark 3.5.x (Scala 2.12), Python 3.11, Java 17 or 21, GraphFrames 0.8.3.

```bash
pip install "pyspark==3.5.5" pyarrow numpy pandas psutil requests pytest
```

GraphFrames is still resolved from Maven by default (`spark.jars.packages`). On a machine
without Maven access, point `GRAPHFRAMES_JAR` at a local copy and every harness and test uses it:

```bash
export GRAPHFRAMES_JAR=$PWD/jars/graphframes-0.8.3-spark3.5-s_2.12.jar   # Windows: set GRAPHFRAMES_JAR=...
export PYSPARK_PYTHON=$(which python)
```

Optional labels recorded with every campaign: `PHASE1_MACHINE_LABEL`, `BENCH_CPU_MODEL`, `BENCH_DISK_MODEL`.

All commands run from the repository root.

## 2. Regenerate the datasets

```bash
python -m datagen.generate --plan pilot          # the 11 pilot graphs
python -m datagen.generate --plan full           # 5 families x 3 tiers x 4 seeds = 60 graphs
python -m datagen.generate --family multi_hub --tier small --seed 101
python -m datagen.generate --plan pilot --verify-existing    # regenerate in memory; prove identical edges
python -m datagen.spark_validate                 # re-read every file through Spark and cross-check
```

Generation is pure NumPy from one seeded generator per graph, so output does not depend on
Spark partitioning or the machine. Existing graph directories are never overwritten.

### Edge policy (all synthetic graphs, and applied to Pokec)

* Directed `src -> dst`; schema `src int32, dst int32`; ids in `[0, num_vertices)`.
* Self-loops removed. Duplicate `(src, dst)` pairs removed (simple digraph).
* When cleaning drops edges, more are drawn from the same distribution, then a seeded random
  subset of exactly `m_target` is kept. Dropped counts are recorded per graph.
* Under this policy `work_sum = Σ_v d_in(v)·d_out(v)` is the **exact** number of 2-hop paths
  (rows of `e1 ⋈ e1.dst = e2.src e2`, including `a = c` 2-cycles), not an approximation.
  It is not a runtime estimate.

### Families (`datagen/config.py`)

| Family | Process | Intended property, checked by measurement |
|---|---|---|
| `uniform` | 7 rounds of `i → perm[i]`: every vertex has in = out = 7 before cleaning | CV ≤ 0.2, max/mean ≤ 2 for in and out |
| `powerlaw_balanced` | Directed Chung–Lu, one weight per vertex for in and out, `gamma=2.5`, per-vertex share capped at 0.2 % | max/mean ≥ 50 and Gini ≥ 0.4 on both sides; log(in)–log(out) corr ≥ 0.5; max_in/max_out in [1/3, 3] |
| `in_heavy` | 25 % of edges point into `10 per 100k` random hubs (Zipf 1.0); rest uniform | in max/mean ≥ 100; out max/mean ≤ 5; top-H hubs hold ≥ 15 % of in-edges; out CV ≤ 0.6 |
| `out_heavy` | mirror of `in_heavy` | mirror checks |
| `multi_hub` | 10 / 20 / 40 hubs (small/medium/large), each ~2000 in **and** out edges (±25 %), uniform background | ≥ 5 vertices with W > 1M; top-H hubs hold ≥ 50 % of work_sum; weakest/strongest hub work ≥ 0.3 |

Size tiers: small 100k vertices / 700k edges, medium 500k / 3.5M, large 1M / 7M (avg degree 7).

Thresholds are declared before generation; each `graph.json` stores the measured value next to
every threshold, so a family label is never the only evidence of a topology.

## 3. Pokec

```bash
python -m datagen.pokec --raw datasets/pokec/raw/soc-pokec-relationships.txt.gz
```

Reads the SNAP file exactly as `preprocessing.py` does, checks it against SNAP's published counts,
applies the edge policy (counting every dropped row), writes `datasets/pokec/pokec_edges_parquet/`,
validates it with the same validator as the synthetic graphs, re-reads it through Spark, and
computes `work_sum`. If a validated copy for the same raw-file hash exists, it does nothing.

HDFS is reported as available only after a real `hdfs://` listing succeeds (set `PHASE1_HDFS_URI`
to test one). `HADOOP_HOME`/`winutils.exe` give Spark local-filesystem support on Windows; they are
not HDFS.

## 4. Run benchmarks

```bash
python bench/run_phase1.py --name pilot --dry-run          # plan + safety decisions only
python bench/run_phase1.py --name pilot                    # every PASS graph in the manifest
python bench/run_phase1.py --name pokec --pokec-only --max-work-sum 2000000000
python bench/run_phase1.py --resume <campaign_id>          # continue after an interruption
python bench/phase1_aggregate.py <campaign_id>             # derived summaries
```

Per graph: one fresh Spark application → load + cache edges (`repartition(200)`), vertex table =
distinct endpoints (cached, broadcast-hinted) → **correctness gate** → 1 warm-up round →
5 measured rounds. Every round runs each strategy once, order reshuffled with a recorded seed.

* **Correctness gate (untimed).** Every strategy computes row count + two order-independent row
  checksums of `(a.id, b.id, c.id)`; it must equal native GraphFrames (AQE off) and `work_sum`.
  A strategy that fails gets no timed runs and is recorded as skipped.
* **Timed action.** `df._jdf.queryExecution().toRdd().count()` (from `run_single_ablation.py`):
  materializes every output column, returns the path count (checked against `work_sum` on every
  run) and leaves the final AQE plan inspectable.
* **Profiling.** For `salt1d`/`hypercube*`, the degree-profiling pass is re-run and timed every
  run (`profiling_s`); `total_s = profiling_s + query_s`.
* **Safety gate.** Graphs with `work_sum > --max-work-sum` are recorded as `skipped` for every
  strategy and never executed. A run exceeding `--run-timeout-s` is cancelled and recorded as
  `timeout`; a worker exceeding `--graph-timeout-s` is killed and recorded.

### Candidate strategies (`bench/phase1_strategies.py`)

`<variant>__aqe_<on|off>` with the repo's variant names: `native`, `edgejoin`, `salt1d`,
`hypercube` (split, Algorithm 1), `hypercube_vjoin`, `hypercube_unified`.
Pilot set: `native__aqe_on`, `native__aqe_off`, `edgejoin__aqe_on`, `edgejoin__aqe_off`,
`salt1d__aqe_off`, `hypercube__aqe_off`, `hypercube_unified__aqe_off`.

AQE on = Spark 3.5 defaults, pinned explicitly: skew join on, skewed-partition threshold
**256 MB**, factor 5, coalescing on, advisory 64 MB. Core config: `local[*]`, 200 shuffle partitions,
`autoBroadcastJoinThreshold=-1`, hub threshold θ = 1,000,000, `P = 100` grid cells.
Every value is written into `campaign.json` and every run record.

## 5. Data contract for downstream members

Paths are relative to the repo root. Raw files are append-only; nothing in `datasets/raw_runs/`
is ever rewritten. **Proposed schema — please confirm or request changes before the 60-graph run.**

### `datasets/manifest/dataset_manifest.csv` (Member 2: graph features)

One row per graph. Identity: `graph_id`, `family`, `seed`, `size_tier`, `generator_version`,
`generator_params` (JSON), `edges_sha256`. Data: `path`, `format`, `file_bytes`, `file_sha256`,
`num_vertices`, `num_edges`, `num_active_vertices`. Quality: `validation_status`, `failed_checks`,
`spark_validation_status`. Measured topology: `work_sum`, `work_max`, `work_max_share`,
`num_work_hubs`, `work_in_hubs_share`, `in_/out_max`, `in_mean`, `in_/out_cv`, `in_/out_gini`,
`in_/out_max_over_mean`, `log_degree_corr`. Cleaning: `self_loops_removed`, `duplicates_removed`.
Full measurements (percentiles, top-share, zero-degree counts) are in each `graph.json`.

`graph_id = <family>-<tier>-s<seed>-<8 hex of SHA-256 over all generator inputs>`.

### `datasets/raw_runs/<campaign_id>/runs.jsonl` (Members 2–4)

One JSON object per (graph, strategy, run). Join to the manifest on `graph_id`.

| Group | Fields |
|---|---|
| identity | `schema_version`, `record_id`, `campaign_id`, `config_id`, `machine_id`, `spark_app_id` |
| graph | `graph_id`, `graph_source` (synthetic/pokec), `family`, `size_tier`, `seed`, `num_vertices`, `num_edges`, `expected_path_count` |
| strategy | `strategy`, `variant`, `aqe`, `aqe_conf` (effective AQE settings), `work_threshold`, `p_cells`, `hub_count` |
| run | `run_kind` (`correctness` / `warmup` / `measured` / `planned`), `round_index`, `measured_index` (1–5), `order_in_round`, `order_seed`, `status` (`ok` / `failed` / `timeout` / `skipped`), `skip_reason`, `error_type`, `error_message`, `start_utc`, `end_utc` |
| timing | `profiling_s`, `query_s`, `total_s` |
| correctness | `path_count`, `correctness_status` (`match` / `mismatch` / `unchecked`); gate rows also `checksum_xxhash64`, `checksum_hash`, `matches_reference` |
| totals (query window) | `num_stages`, `total_tasks`, `executor_run_time_s`, `executor_cpu_time_s`, `shuffle_read_bytes`, `shuffle_write_bytes`, `memory_spill_bytes`, `disk_spill_bytes`, `peak_execution_memory_max` |
| dominant stage (max executor time) | `dom_stage_id`, `dom_num_tasks`, `dom_task_median_ms`, `dom_task_p95_ms`, `dom_task_max_ms`, `dom_task_max_median_ratio`, `dom_task_mean_ms`, `dom_task_cv`, `dom_shuffle_read_bytes_median/p95/max` (actual reduce-partition sizes) |
| AQE decisions | `aqe_shuffle_read_nodes`, `aqe_partitions`, `aqe_coalesced_partitions`, `aqe_skewed_partitions`, `aqe_skew_splits`, `aqe_skew_split_applied`, `plan_is_final`, `plan_skew_split`, `plan_partial_reducer_specs`, `plan_coalesced_specs` |
| environment | `spark_version`, `pyspark_version`, `java_version`, `scala_version`, `graphframes_jar`, `hypergraphframes_version`, `git_commit`, `git_dirty_tree`, `spark_master`, `driver_memory`, `shuffle_partitions`, `input_partitions`, `action`, `regime` |

`run_details.jsonl` holds per-stage metrics, task-level distributions and the AQE plan analysis,
keyed by `record_id`. `campaign.json` holds the machine, core config and safety decisions.
`datasets/raw_runs/INDEX.csv` lists every campaign.

**Aggregating:** filter `run_kind == "measured" and status == "ok" and correctness_status == "match"`,
group by `(graph_id, strategy)`. Never average across different `config_id`/`machine_id` without
saying so. Campaigns named `smoke` are harness checks; exclude them.

### `datasets/derived/<campaign_id>/` (convenience, regenerable)

`strategy_summary.csv` (median/p95/min/max/IQR per graph × strategy, plus status counts),
`graph_summary.csv` (fastest strategy by median `total_s` and `query_s`, runner-up, margin),
`SOURCE.txt` (raw file hash). Regenerate any time with `phase1_aggregate.py`.

## 6. Tests

```bash
python -m pytest datagen/tests -q                       # generators/validators (NumPy, ~30 s)
python -m pytest bench/test_phase1.py -q                # unified hypercube, checksums, record schema (Spark)
cd hypergraphframes && python -m pytest tests -q        # existing package suite
python -m pytest bench/test_bench_variants_equivalence.py -q
```

## 7. Scaling from the pilot to 60 graphs

```bash
python -m datagen.generate --plan full && python -m datagen.spark_validate
python bench/run_phase1.py --name full60 --dry-run      # check safety decisions first
python bench/run_phase1.py --name full60
```

See `docs/PHASE1_REPORT.md` for pilot results, measured costs and the recommended limits.
