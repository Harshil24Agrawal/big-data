# Interruptions of campaign 20261019T163326Z-pilot-full-5d1324

The sandbox VM restarted during this campaign, killing the benchmark process each time (uptime reset; no run_phase1 / worker process afterwards).
Raw records in runs.jsonl are untouched and append-only. Attempts are separated by `spark_app_id`; `bench/phase1_aggregate.py`
counts only the latest application per graph and reports earlier ones as superseded.

| Attempt | Graph | Spark app suffix | Records written before the kill | Outcome |
|---|---|---|---|---|
| 1 | uniform-small-s101-f3568c8f | ...628677 | 7 correctness, 7 warm-up, 12 of 35 measured | killed by sandbox restart (~16:53 UTC) - incomplete, not a failure of any strategy |
| 2 (resume) | uniform-small-s101-f3568c8f | ...933105 | 7 correctness, 2 warm-up | killed by a second sandbox restart (~17:03 UTC) - incomplete |

No run in this file has a failed or timeout status. Records of incomplete attempts must not be used as complete cells.
