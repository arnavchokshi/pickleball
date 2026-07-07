# w4_ballgpu_20260707 — FINAL REPORT (lane copy; the manager-readable copy is the final agent message)

## OBJECTIVE RESULT: PARTIAL (all four work products banked; lane-candidate scores are harness_v0 measurement-mode only — a preprocessing-contract mismatch blocks official-mode scoring of ANY harness-trained checkpoint until wave-5 aligns training preprocessing)

## ACCEPTANCE (bridge cards, post-fix code 67298e599, --visible-threshold 0.5, candidate=fusion_temporal_vball100, INTERNAL-VAL reference clips only; labels/clip: 24 visible + 6/5 hidden — small-N, ±~4%/label)
| candidate | mode | label_f1_at_20px burlington | label_f1_at_20px wolverine | mean_visible_hit_recall | mean_hidden_false_positive_rate | vs 0.6685 ref | vs stage-1 bridge |
|---|---|---|---|---|---|---|---|
| official_tennis_control | official | 0.7143 | 0.7826 | 0.6875 | 0.65 | above both | — (healthy-bridge control) |
| stage-1 (w3_p11 latest.pt) | harness_v0 | 0.8936 | 0.2000 | 0.5625 | 0.4667 | above burl only | baseline |
| seed fine-tune (owner 486 rows, 1830 steps) | harness_v0 | 0.7368 | 0.5946 | 0.6458 | 0.2000 | above burl only | burl −0.157, wolv +0.395, hidden-FP −0.267 |
| SST student @3k steps (stage-1 init) | harness_v0 | 0.7442 | 0.7273 | 0.7708 | 0.65 | **ABOVE BOTH** | burl −0.149, wolv +0.527, recall +0.208 |
Pre-fix official-mode scoring of stage-1/seed measured F1 0.0/0.0 with degenerate constant-point tracks (bug evidence, kept at vm_artifacts/bridge/) while the official ckpt scored 0.714/0.783 through the identical path — that contrast is the proof of the preprocessing-contract mismatch, not of bad checkpoints.

## THRESHOLD SWEEP (knob = run_wasb_ball --visible-threshold: BallTrack visibility cutoff over the WASB heatmap peak confidence; harness_v0; per-point: mean_visible_hit_recall / mean_hidden_false_positive_rate / f1 burl / f1 wolv)
| thr | SST-3k | stage-1 |
|---|---|---|
| 0.3 | 0.7708 / 0.65 / 0.7442 / 0.7273 | 0.5625 / 0.4667 / 0.8936 / 0.2000 |
| 0.4 | identical to 0.3 | identical to 0.3 |
| 0.5 | identical to 0.3 | identical to 0.3 |
| 0.6 | 0.7500 / 0.5667 / 0.7317 / 0.7273 | 0.5625 / 0.1000 / 0.9333 / 0.2143 |
| 0.7 | 0.7083 / 0.4000 / 0.7692 / 0.6667 | 0.3750 / 0.0000 / 0.7027 / 0.1538 |
Reading: sub-0.5 is a plateau (peak confidences mostly >0.5); 0.6–0.7 buys hidden-FP down at recall cost. Stage-1's wolverine failure is threshold-invariant. No re-tuning was performed off this table.

## SST-INIT RULING
ELSE branch fired (student init = stage-1 checkpoint). At decision time the bridge could not produce valid lane-candidate numbers (pre-fix), so the IF condition was unmeasurable → ELSE mechanically. Post-fix numbers retroactively validate it: seed (0.7368) < stage-1 (0.8936) on burlington, so IF would have failed anyway.

## PROOFS
- WASB anchor: sha256 9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb verified on VM AND on the Mac copy at models/checkpoints/wasb/wasb_tennis_best.pth.tar (LOCAL BLOCKER closed).
- Protected-hash assert: eval_hash_count=35, collision_count=0 (1823 corpus video dhashes) — run before any training step.
- Version stamps: initial sync 5b268aa6d via full-history git bundle (10-file committed-md5 table all-match + whole-tree git-status proof; version_stamp.json); delta sync to 67298e599 (bridge fix) with fixed-file md5s matching Mac committed blobs (wasb_adapter 1c546c75…, run_wasb_ball a75a865a…).
- VM lifecycle: pickleball-h100-w4ballgpu (H100-80GB spot, asia-southeast1-b, EXCLUSIVE_PROCESS + preemption watcher verified) created 2026-07-07T17:43:49Z, DELETED 2026-07-07T23:38:18Z, list-confirmed (fleet_list_after_delete.log). Uptime 5.90h. Cost: $3.4–25 using the ledger's honest H100-spot price span ($0.57–4.25/hr); ≈$12–24 at the $2–4/hr mid-band. Zero preemptions.

## ARTIFACTS (Mac, md5-verified vs VM before delete)
- Seed run: vm_artifacts/w4_ball_stage2_owner_20260707T202900Z/ (latest.pt 11371641e887…, summary.json: init key-diff EMPTY, 486 samples = 268 pos/218 neg, WBCE 1:463/2:21/3:2)
- SST student: vm_artifacts/sst_round1_student_train/student_train/checkpoints/checkpoint_step_003000.pt (bdde8cb5822f…) + 5 earlier ckpts
- SST manifest: vm_artifacts/sst_manifest.json (9cef0a356…; 40 clips, 58,353 samples)
- Disagreement queue: vm_artifacts/sst_disagreements.json (639f6ed5…; 16 clips, 12,075 rows: 4,307 teacher-only / 2,194 student-only / 5,574 large-offset, ranked)
- Bridge: vm_artifacts/bridge_v2/ (4 cards + 8 sweep runs, all with input_preprocessing stamps) and vm_artifacts/bridge/ (pre-fix bug evidence)
- Student predictions: vm_artifacts/student_predictions/ (16 clips, run_wasb_ball harness_v0, stamped)
- Logs + lane tools: seed/sst/bridge/sweep logs, sanitize_ckpt.py (interim workaround), student_infer.py (written, superseded by the repo fix, unused for final queue), run_bridge.sh, run_sweep_and_students.sh

## HONEST ISSUES (unsoftened — see final agent message for the full list)
Preprocessing-contract mismatch (harness resize+/255 vs official affine+ImageNet norm) makes every lane card NON-PROMOTABLE harness_v0; torch-2.9 weights_only load bug hit live (fixed same-day by w4_bridgefix; interim bitwise-verified sanitized copies); SST stopped at 3k/12k steps by manager re-cap (video-seek loader ~0.6-1.0 steps/s, not ~8/s); queue coverage 16/40 clips from 2/6 sources (alphabetical cut, manager pre-approved ≥15); small-N labels; seed loss curve flat while bridge behavior clearly changed; macOS→VM EMSGSIZE transfer flake reproduced hard (bwlimit+append-verify+chunking required); ~1h of uptime lost to two API stalls (nohup pattern saved all work).

## NEXT (for the manager)
Wave-5 preprocessing alignment (retrain stage-2 with official WASB transform, or land a permanent dual-convention story) then re-score SST-3k in official mode; optionally resume SST from checkpoint_step_003000.pt for the remaining 9k steps (predecode frames first to kill the IO bottleneck); extend student predictions to the remaining 24 clips (~45 GPU-min) for a balanced owner queue; owner labeling session on the 12,075-row queue; add the fleet-ledger row for this VM.
