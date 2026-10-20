# Interruptions in pilot-ext1

- 2026-10-20 ~00:44 UTC: the sandbox VM restarted (uptime 0 minutes afterwards) while measuring the second graph,
  `multi_hub-large-s202-301552c4`. The first graph (`multi_hub-large-s101-6eed4f52`) had finished (status done). The partial
  records of the second graph stay in `runs.jsonl` and are treated as *superseded* once the graph is re-run under a new
  Spark application (`--resume`); aggregation and coverage count only the latest application per graph.
