# Selector prototype -- PRELIMINARY

Label-eligible graphs: 31. Label counts (lowest median total_s): {'edgejoin__aqe_off': 14, 'edgejoin__aqe_on': 16, 'native__aqe_on': 1}.

Outcomes are measured medians from one machine/config; speedup and regret are table lookups of those medians.
Feature-extraction time is not included. Not evidence of generalisation.

## Held-out protocol: leave-one-graph-out

| policy | graphs | mean regret | median regret | max regret | geomean speedup vs native (AQE on) | oracle chosen |
|---|---:|---:|---:|---:|---:|---:|
| best_fixed_train | 31 | 10.1 % | 0.0 % | 53.7 % | 1.317 | 16 |
| ml | 31 | 14.8 % | 3.6 % | 103.4 % | 1.272 | 12 |
| ml_native_fallback | 31 | 14.8 % | 3.6 % | 103.4 % | 1.272 | 12 |
| native_aqe_off | 31 | 56.0 % | 51.1 % | 160.4 % | 0.937 | 0 |
| native_aqe_on | 31 | 45.3 % | 44.6 % | 111.4 % | 1.000 | 1 |
| oracle | 31 | 0.0 % | 0.0 % | 0.0 % | 1.438 | 31 |
| rule_hub | 31 | 84.7 % | 95.3 % | 159.9 % | 0.802 | 1 |

## Held-out protocol: leave-one-family-out

| policy | graphs | mean regret | median regret | max regret | geomean speedup vs native (AQE on) | oracle chosen |
|---|---:|---:|---:|---:|---:|---:|
| best_fixed_train | 31 | 10.1 % | 0.0 % | 53.7 % | 1.317 | 16 |
| ml | 31 | 15.2 % | 2.6 % | 103.4 % | 1.271 | 14 |
| ml_native_fallback | 31 | 15.2 % | 2.6 % | 103.4 % | 1.271 | 14 |
| native_aqe_off | 31 | 56.0 % | 51.1 % | 160.4 % | 0.937 | 0 |
| native_aqe_on | 31 | 45.3 % | 44.6 % | 111.4 % | 1.000 | 1 |
| oracle | 31 | 0.0 % | 0.0 % | 0.0 % | 1.438 | 31 |
| rule_hub | 31 | 84.7 % | 95.3 % | 159.9 % | 0.802 | 1 |
