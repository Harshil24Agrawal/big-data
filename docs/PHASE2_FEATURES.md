# Phase 2 — pre-execution graph features (final schema)

Owner of the code: Member 1 delivered the first complete version so Member 2 can review, extend or replace it.
Everything here is regenerable from graph artifacts; nothing depends on benchmark results.

## Commands

```bash
source /home/claude/venv311/bin/activate            # or your own environment (pandas, numpy)
python ml/build_feature_table.py                    # -> datasets/derived/features/{graph_features.csv,SCHEMA.json,SOURCE.txt}
python ml/build_training_table.py [campaign_id ...] # -> datasets/derived/training/{training_table.csv,training_table_eligible.csv}
python -m pytest ml/tests -q
```

## `graph_features.csv` (one row per graph, 60 synthetic + Pokec = 61 rows; join key `graph_id`)

| Column group | Columns | Notes |
|---|---|---|
| ids | `graph_id, graph_source, role, family, size_tier, seed` | `role` = `train_candidate` (synthetic) or `external` (Pokec, test-only) |
| model features (11) | `log_num_vertices, log_num_edges, active_fraction, in_gini, out_gini, log_in_p999, log_out_p999, log_work_sum, work_max_share, log_num_work_hubs, work_in_hubs_share` | `log` = `log1p`; ratios untouched |
| reproducibility | `work_threshold` | a vertex is a *work hub* if `d_in·d_out > work_threshold` (1,000,000); `log_num_work_hubs` and `work_in_hubs_share` depend on it |
| raw values | `raw_num_vertices, raw_num_edges, raw_num_active_vertices, raw_in_p999, raw_out_p999, raw_work_sum, raw_num_work_hubs` | for inspection and for rules such as "has hubs" |

Where the values come from: each graph's `graph.json["measured"]` (computed in Phase 1 from the edge list, cross-checked
against Spark). `in_p999`/`out_p999` exist only in `graph.json`, not in `dataset_manifest.csv`; the builder reads them
from there. A missing field is an error, never a silent default.

## Properties that were checked

* **No leakage:** no runtime, strategy, campaign or run-count column exists in the feature table (unit-tested by name).
* **Consistency:** `raw_work_sum` and `raw_num_edges` equal the manifest values for all 60 graphs (unit-tested).
* **Redundancy:** edges/vertices ≈ 7.0 in every synthetic graph, so `log_num_vertices` and `log_num_edges` carry the same
  information on the synthetic data (corr 0.99999999995). Keep one when modelling; both are provided so Pokec (edges/vertices ≈ 18.8) stays representable.
* **Pokec** is outside the synthetic range (1.78B work, Gini ≈ 0.64): use it as an external test point, not for training.
* `work_sum` is a workload indicator (exact two-hop path count under the simple-digraph policy), not a runtime estimate.

## Joining to benchmark data

`ml/build_training_table.py` joins `graph_features.csv` with the aggregated benchmark summaries on `graph_id`:

* outcome columns per strategy: `median_total_s__<strategy>`, `total_s_iqr__<strategy>`, `n_valid_runs__<strategy>`
  (+ `median_query_s__…`, `median_profiling_s__…` for diagnostics; never features);
* label: `best_strategy` = lowest median `total_s`; `runner_up_strategy`, `label_margin_pct`, `label_clear`
  (gap larger than both strategies' IQR);
* tracking: `campaign_id, config_id, machine_id, label_eligible`;
* a graph is `label_eligible` only if every planned strategy has the campaign's required number of valid measured runs;
* a table mixing configurations or machines is refused (`SystemExit`).

`training_table_eligible.csv` contains only label-eligible graphs and is the table to train on.

## Limits

The feature set describes degree and two-hop-work concentration only. It does not describe edge-order, locality or
partition layout, and no feature depends on the Spark configuration. The schema is a proposal until Members 2–4 confirm it;
changes should be backward compatible (add columns, do not rename).
