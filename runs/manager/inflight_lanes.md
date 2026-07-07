# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

WAVE 4 OPEN 2026-07-07 (manager session on branch `worktree-wave4-manager`; boot prompt
`runs/manager/wave4_boot_prompt.md`). All lanes dispatched ~2026-07-07 as background codex exec per
manual §10; resume = `cd /Users/arnavchokshi/Desktop/pickleball && codex exec resume -c
model_reasoning_effort=<effort> <SESSION_ID> -` (session id in report.json, or `grep "session id"
runs/lanes/<lane>/log.txt`); harness bg-task ids listed for TaskOutput monitoring.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| w4_cammotion_diag_20260707 | codex xhigh READ-ONLY | bg task broazt3p3; codex session in log | see header | runs/lanes/w4_cammotion_diag_20260707/ only | — | ~1-2h | 2026-07-07 |
| w4_footattr_diag_20260707 | codex xhigh READ-ONLY | bg task b329ofe32 | see header | runs/lanes/w4_footattr_diag_20260707/ only | — | ~2-3h | 2026-07-07 |
| w4_burlmesh_diag_20260707 | codex medium READ-ONLY | bg task b5e43s1au | see header | runs/lanes/w4_burlmesh_diag_20260707/ only | — | ~30-60m | 2026-07-07 |
| w4_fleethosts_20260707 | codex medium | bg task bsw3ivn10 | see header | remote_body_dispatch.py, test_remote_body_dispatch.py, scripts/fleet/refresh_remote_host.*, configs/ssh/ (add-only) | — | ~30-60m | 2026-07-07 |
| w4_bvp_20260707 | codex xhigh | bg task bk3kmgnv1 | see header | ball_arc_solver.py, test_ball_arc_solver.py | — | ~2-4h | 2026-07-07 |
| w4_ballcode_20260707 | codex xhigh | bg task b8k3t8n6d | see header | train_ball_stage2.py(new), ball_sst_dataset.py(new), export_sst_disagreements.py(new), test_ball_stage2_*(new), train_ball_pretrain.py(minimal) | — | ~2-4h | 2026-07-07 |
| w4_court_harvestcal_20260707 | codex xhigh | bg task bwlg1m5pg | see header | calibrate_harvest_courts.py(new)+test, data/online_harvest_20260706/court_calibrations/(new) | — | ~2-3h | 2026-07-07 |

QUEUED (not yet dispatched): w4_ballgpu (Sonnet GPU H100 spot, AFTER w4_ballcode rules PASS — WASB
prestage sha-check + seed fine-tune on owner labels + SST r1 (teacher = 40 local raw-WASB sidecars)
+ threshold sweep, ALL scoring through the SCORING BRIDGE on Burlington/Wolverine, INTERNAL-VAL
ONLY, self-provision→run→verify→DELETE); w4_cammotion_fix + w4_footattr_fix (AFTER diagnosis
rulings; each ships with one adversarial-verify round — gate-adjacent); w4_h100body (conditional,
queue #7); wave-close sequence: integration micro-lane (if needed) → ONE wide-suite adjudication
(minus test_court_finding_technology_benchmark.py, standalone) → fresh-GPU 4-clip proof
(snapshot→fan, A100 proven SKU, version-stamp first, browser-verify replay_viewer_manifest.json) →
docs reconciliation (W4-F) → [WAVE-4 COMPLETE] bullet + wave-5 boot prompt.

Fleet at open: fleet1 STOPPED disk-intact (only VM, list-confirmed; impersonated-call auth OK
2026-07-07). No VM running. GPU budget stated in boot prompt: ~$12-25 expected.

Concurrent-session fence: ios/** + BUILD_CHECKLIST bullet 710 + working-tree ios changes belong to
the wave-2 manager session (brand-v2 verify leg) — never touched by wave-4 lanes or commits.
