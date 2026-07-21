# w3a_product_bugs_20260721 — the two fresh-clip-exposed product bugs (Wave 3)

Codex gpt-5.6-sol xhigh. The fresh-clip proof run exposed two REAL defects (runs/lanes/pooling_wire_20260720/PROOF_RESULT.md + the agent report):
1. EXCLUSIVE_PROCESS CUDA CONFLICT: scripts/fleet/lane_vm_startup.sh sets nvidia-smi -c EXCLUSIVE_PROCESS, which blocked the BODY subprocess's second CUDA context → 0 meshes computed (1,368 scheduled). Fix: default compute mode for pipeline VMs (parameterize: EXCLUSIVE only for single-context training lanes; pipeline lanes get DEFAULT). Document in the script header + gpu_fleet notes text in your report.
2. COACHING_FACTS FIELD BUG: ValueError missing required players[1].frames[0].track_world_xy — a real player flowing deep for the first time hit this. Find the producer/consumer mismatch (facts builder expects track_world_xy per frame; tracks.json frames carry world_xy) and fix the CONSUMER to read the actual exported field (or tolerate absence with a typed degraded reason) — do NOT change the tracks export schema (other lanes own it). Test with a fixture built from the pulled tracks.json (runs/lanes/pooling_wire_20260720/gpu_replay_pull/tracks.json).
## HARD RULES
- NO commits/branches/pushes (manager commits after ultra review). VERIFIED=0. Honest reporting; misses are misses.
- **NO JUDGE PEEKING (new standing rule after a judge-contamination catch): develop against fixtures/synthetic cases ONLY. You may NOT run any GT scorer / frozen gate / protected eval during development. ONE final scored run happens later, by the manager, on frozen code. A log showing peek-tweak-rescore = automatic rejection.**
- Focused tests + wide suite (MPLBACKEND=Agg), real exit codes, attribute failures. Artifacts under YOUR lane dir.
- CROSS-SIGNAL ROW required in your report (what you consume/feed — North Star §3.1).
- Concurrent file-disjoint lanes are live — touch ONLY your fenced files.
FENCE: scripts/fleet/lane_vm_startup.sh, the coaching-facts builder file (locate it; likely threed/racketsport/coaching_facts*.py or scripts/racketsport equivalent), its tests, lane dir. NOT: player_selection, event_head, court files, finetune.
