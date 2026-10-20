# Notes on pilot-ext2

- Planned 12 graphs; 11 finished (status done). The campaign was stopped deliberately at about 05:00 UTC on 2026-10-20
  (10:30 IST) during its 12th graph, `in_heavy-large-s101-d1765691`, because that graph could not finish before the
  hand-over deadline. Its partial records stay in `runs.jsonl` (append-only), are never counted as valid runs unless the graph
  is completed with `--resume`, and the graph is not label-eligible.
- Resume command: `python bench/run_phase1.py --resume 20261019T..-pilot-ext2-.. --driver-memory 4g` (same environment variables as the other campaigns).
