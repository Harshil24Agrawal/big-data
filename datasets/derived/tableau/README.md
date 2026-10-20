# Tableau-ready data (no .twbx was built)

Connect Tableau to these CSVs (Text file connector). All values come from `training/combined_table.csv` and `selector_v2/`.

| File | Grain | Use |
|---|---|---|
| `graph_strategy_long.csv` | graph x strategy (62 x 3) | main sheet: runtime vs `max_vertex_work` (log axis, colour = strategy); crossover; regret by graph; filter on `protocol`, `family`, `hub_count` |
| `policy_scores.csv` | scheme x policy | bar chart of mean regret / speed-up / total time per policy; `scheme` = LOGO / LOSG / SEED202 |
| `classifier_scores.csv` | scheme x model | accuracy next to the `majority` row (always show it) |
| `cv_predictions_logo.csv` | graph | confusion matrix (true_class vs pred_class_logo_<model>) |

Suggested sheets: (1) hypercube/edge-join ratio vs max_vertex_work (log-log, reference line y=1); (2) strategy runtime by hub share; (3) policy regret bars; (4) classifier accuracy vs majority; (5) label balance by family.
