# PRELIMINARY — do not use as ML training labels

These summaries come from a reduced pilot (7 graphs, 1 warm-up, 0-3 measured runs per graph/strategy, 540 s graph cap,
2 vCPU sandbox). Medians over n=1-3 are not statistically meaningful, several cells have n=0, `uniform-large` was
effectively not benchmarked, and AQE skew splitting never triggered on any run.

* `graph_summary.csv` "fastest_*" columns are NOT validated labels. Do not train the strategy selector on them.
* Safe to use now: dataset manifest, graph Parquet files, `graph.json` metadata, correctness-gate results,
  and these files as a schema/format example for pipeline testing (feature extraction, Tableau wiring).
* Validated labels require the full campaign: `--runs 5 --warmup 1`, consistent config/machine, skipped graphs run,
  and skew-triggering cases included. See docs/PHASE1_REPORT.md section 4.
