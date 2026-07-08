# Frame-Scheduling Watch Adjudication

## Finding

The `541 vs 276 frames same clip after body_compute_execution.json delete` watch item is distinct from the wave-5 close-proof item that was cleared as `243=243` byte-identical. The close proof shows the current schedule/output can reproduce byte-identically; it does not explain the earlier P2-2 gate-check discrepancy.

## Evidence

- `BUILD_CHECKLIST.md` carries the original watch text as `541 vs 276 frames same clip after body_compute_execution.json delete` and later records the wave-5 closeout as cleared with `243=243` byte-identical evidence.
- `runs/manager/wave6_boot_prompt.md` still queues the 541/276 issue under the instrument/docs debt lane, so the manager had not treated that specific discrepancy as fully explained.
- `runs/lanes/w5_p22latent_20260707/vm_evidence/gate_check_baseline_report_v2.json` reports `total_real_frame_sample_count=276` for `/home/arnavchokshi/pickleball_git/runs/process_video_body_dispatch/wolverine_mixed_0200_mid_steep_corner_20260708T055634Z/body_mesh.json`.
- `runs/lanes/w5_p22latent_20260707/vm_evidence/gate_check_patched_report.json` reports `total_real_frame_sample_count=541` for the same remote `body_mesh.json` path.
- `runs/lanes/w5_p22latent_20260707/scripts/gate_check.py` computes `total_real_frame_sample_count` from real persisted `body_mesh.json` player frames with pose/joint payloads. It does not count `body_compute_execution.json` scheduled frames directly.
- The local dispatch execution copies under `runs/lanes/w5_p22latent_20260707/wolverine_body_dispatch/body_compute_execution.json` and `runs/lanes/w5_closeproof_20260708/wolverine/body_compute_execution.json` have the same scheduling shape: 266 scheduled entries and 244 unique `frame_idx` values.

## Mechanism Assessment

The cheap read-only evidence points to a BODY mesh population / gate-check input discrepancy, not a scheduler nondeterminism proof. The 276/541 numbers are counts of real player-frame samples extracted from `body_mesh.json`; the 243/243 close proof is a byte-identical closeout for the later proof artifact. The local lane artifacts do not include the remote monolithic `body_mesh.json`, so this lane cannot reproduce or fully root-cause the 276/541 delta without re-running or fetching VM evidence.

## Recommendation

Treat the 541/276 observation as a distinct, unresolved diagnosis item for manager ruling if any future proof depends on P2-2 gate-check frame counts. Do not change scheduling code from this lane.
