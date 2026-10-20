# Phase 4 - Tableau dashboard guide

Data (all CSV, no Tableau file is included): `datasets/derived/tableau/`.
An interactive dashboard of the same results (built outside Tableau) is published as a Claude artifact: "Spark Strategy Selector Dashboard".

## Connect
Tableau Desktop / Public -> Connect -> **Text file** -> `graph_strategy_long.csv`. Add `policy_scores.csv`, `classifier_scores.csv`, `cv_predictions_logo.csv` as separate data sources (they have different grains, do not join them).

## Calculated fields (graph_strategy_long)
- `Is hub-splitting` = `[strategy] = "hypercube_unified__aqe_off"`
- `Time ratio vs edge-join` (use a table calc or LOD):
  `SUM(IF [strategy]="hypercube_unified__aqe_off" THEN [median_total_s] END) / SUM(IF [strategy]="edgejoin__aqe_off" THEN [median_total_s] END)` with `graph_id` on Detail.
- `Winner label` = `[best_class]` (edgejoin / hubsplit / native).

## Sheets
1. **Crossover** - Columns: `max_vertex_work` (continuous, **log axis**); Rows: `Time ratio vs edge-join` (log axis); Detail: `graph_id`; Color: `best_class`; reference line at 1. Shows hub-splitting wins only past ~1e8 paths through one vertex.
2. **Runtime vs hub share** - Filter `hub_count = 1`, `seed = 101`, `family = hub_sweep`; Columns: `hub_work_share_planned`; Rows: `median_total_s` (log); Color: `strategy`; line + points.
3. **Policy regret** - source `policy_scores.csv`; filter `scheme`; Rows: `policy` sorted by `mean_regret_pct`; Columns: `mean_regret_pct`; Color: highlight "ML selector".
4. **Classifier vs majority** - source `classifier_scores.csv`; filter `scheme`; Rows: `model`; Columns: `accuracy`; reference line at the `majority` row value.
5. **Label balance** - `best_class` count by `family` (stacked bars).
6. **Graph table** - `graph_id`, `family`, `hub_count`, `max_vertex_work`, `median_total_s` by `strategy`, `best_class`.

## Dashboard
Layout: KPI row (graphs, edge-join faster than native, hub-splitting wins, selector regret) -> Crossover -> Runtime vs hub share + Policy regret -> Classifier vs majority -> table. Add a `scheme` parameter control shared by sheets 3 and 4.

Rules: always show the majority-class bar next to accuracy; label the protocol (`protocol` column) because the largest graphs used 2 timed runs.
