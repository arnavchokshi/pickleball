# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| p06-freshworlds | sonnet agent | (running) | SendMessage: 'continue bounded polls; report per-clip' | fresh runs/ dirs + a100_known_hosts + fleet ledger status | fleet1 (A100) | ~2-4h (4 clips serial) | 2026-07-06 |
| p02-hygiene | codex | pid 20178 | codex exec resume <session> | third_party pins, calib shim, gpu_cold_start.sh, ball_arc_chain events_selected | none | ~1h | 2026-07-06 |
| p07-flightsim | codex | pid 20179 | codex exec resume <session> | NEW flight_simulator.py + corpus CLI + tests | none | ~1-2h | 2026-07-06 |
| p09-registry | codex | pid 20180 | codex exec resume <session> | NEW profile_registry.py + schema + tests | none | ~1h | 2026-07-06 |
