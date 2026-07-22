# ROUND-4 MICRO-CONFIRM of court_c0_ingest_20260721 fix3 (read-only) — single question

Only question: is the R3 relocation failure (review_r3.json probe 2) genuinely closed?
Re-run YOUR probe: emit the fixture corpus, COPY it to a different directory, run the real C1
trainer code path from repo root with --real-root at the copy; require 60/60 rows resolving inside
the copy, zero paths at the original, one image materialized from the copy. Also spot-check the
previously-passing checks were not disturbed (geometry reject on one sliver config; shards key
accepted on the real manifest; alias map hash unchanged). VERDICT in final JSON: ACCEPT | REJECT.
