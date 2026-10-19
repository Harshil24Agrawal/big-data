# Notes on pilot-hubs

- Evidence-quality caveat: between about 21:00 and 21:07 UTC on 2026-10-19 the Spark unit tests
  (`bench/test_phase1.py`, 84 s of local Spark) and coverage/aggregation scripts ran on the same 2-core sandbox while this
  campaign was measuring its first graph (`multi_hub-small-s303-7e581b99`). Timings of that graph may be inflated and
  noisier than the rest. Raw records are not edited; treat that graph's medians with extra caution or re-run it.
- No other work is intentionally run on the machine while this campaign measures.
