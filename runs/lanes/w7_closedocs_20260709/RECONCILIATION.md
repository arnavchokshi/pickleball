# W7 Close Docs Best-Stack Reconciliation

Date: 2026-07-09

`configs/racketsport/best_stack.json` is now revision 9. Zero wave-7 gains are unaccounted.

| Wave-7 gain | Manifest state | Evidence |
|---|---|---|
| P3-1 fused paddle default wiring | `paddle.fused_estimator` WIRED_DEFAULT in rev4; `paddle.reflection_cone_factor` DORMANT until P1-4 real 3D velocities | BUILD_CHECKLIST `[W7 P3-1 PADDLEWIRE PASS 2026-07-09]`; `runs/lanes/paddlewire_p31_20260709/report.json` |
| A2 association-profile repair | `tracking.global_association_profile` is the no-flag default; `tracking.eval_only_association_profiles` DORMANT for clip-keyed eval/internal-val selections in rev5 | BUILD_CHECKLIST `[W7 BESTSTACK A2 REPAIR PASS 2026-07-09]`; `runs/lanes/beststack_core_20260708/report_r3.json` |
| Human-review ghost mesh emission | `mesh.human_review_ghost_emission` WIRED_DEFAULT in rev6; wider `mesh.tier_eligibility_raise` remains PENDING | BUILD_CHECKLIST `[W7 TIERPROV PASS 2026-07-09]`; `runs/lanes/w7_ghostviewer_20260709/report.json`; `runs/lanes/w7_tierprov_20260709/report_r2.json` |
| Input-quality preflight | `input_quality.preflight` WIRED_DEFAULT in rev7, advisory by default with strict fail-close knob | BUILD_CHECKLIST `[W7 PIPEPOLISH PASS 2026-07-09]`; `runs/lanes/w7_pipepolish_20260709/report.json` |
| BODY+COURT match stats v0 | `stats.match_stats_v0` WIRED_DEFAULT in rev7, default-on/fail-open post-stage | BUILD_CHECKLIST `[W7 P6-2 STATS V0 PASS 2026-07-09]`; `runs/lanes/w7_p62stats_20260709/report.json`; `runs/lanes/w7_pipepolish_20260709/report.json` |
| BODY base cadence stride-2 | `body.skeleton_stride` WIRED_DEFAULT in rev8 | BUILD_CHECKLIST `[W7 CADENCE PASS 2026-07-09]`; `runs/lanes/w7_cadence_20260709/report.json` |
| BALL full-rate cadence | `ball.detection_stride` WIRED_DEFAULT in rev8 | BUILD_CHECKLIST `[W7 CADENCE PASS 2026-07-09]`; `runs/lanes/w7_cadence_20260709/report.json` |
| Future cadence doctrine | `cadence.future_stage_pattern` WIRED_DEFAULT in rev8 | BUILD_CHECKLIST `[W7 CADENCE PASS 2026-07-09]`; `runs/lanes/w7_cadence_20260709/report.json` |
| A_seed_official_aug ball candidate | `ball.seed_official_checkpoint` PENDING in rev9 with md5 `cfda3c423e1f93c0db42f20e32bdae9e`; named gate is pre-registered heldout_eval_ledger row + owner go, interim bar beats 0.7248 F1@20 held-out, recall >= 0.70, hidden-FP <= 0.05 | BUILD_CHECKLIST `[W7 BALLRETRAIN 1K-CHECKPOINT LANDED 2026-07-09]`, `[W7 BALLSCORE VERIFY 2026-07-09]`, `[W7 BALLCOMPLETE PARTIAL-VALID 2026-07-09]`; `runs/lanes/w7_ballretrain_20260709/REPORT.md`; `runs/lanes/w7_ballcomplete_20260709/REPORT.md`; `runs/lanes/w7_ballretrain_20260709/md5_manifest.txt` |
| Court GT correction | No best-stack default flip; `court.court_unet_v2` and `court.e4_fusion_default` remain PENDING on the existing court PCK gate | BUILD_CHECKLIST `[W7 COURTKP R2 (task-13 owner ruling executed) 2026-07-09]`; `runs/lanes/w7_courtkpingest_20260709/report_r2.json` |
| P2-2 pred_cam_t production/harness fixes | No best-stack default flip; `body.p22_lambda_foot_smoother` and `instrument.gate_check_body_decode` remain DORMANT/instrument-only because P2-2 is NOT-WIRING-READY | BUILD_CHECKLIST `[W7 P22 GPU MEASUREMENT DECISIVE 2026-07-09]`, `[W7 GATECHECKFIX2 COMPLETE 2026-07-09]`; `runs/lanes/w7_p22checklist_20260709/report_r3.json`; `runs/lanes/w7_gatecheckfix2_20260709/report_r2.json`; `runs/lanes/w7_p22gate_20260709/gate1b_raw_arm_report.json` |
| P2-4 masklet-conditioning spike | No manifest default flip; NO-ATTEMPT due permission/HF access gate, future re-attempt requires owner/manager execution grant | BUILD_CHECKLIST `[W7 P2-4 MASKLET NO-ATTEMPT 2026-07-09]`; `runs/lanes/w7_masklet_20260709/REPORT.md` |
| Security gate P7-4c | No best-stack delta; launch gate items booked outside runtime defaults | BUILD_CHECKLIST `[W7 SECURITYREVIEW P7-4c DONE 2026-07-09]`; `runs/lanes/w7_securityreview_20260709/report.json` |
| Licensing gate P7-4d | No best-stack delta; monetization blocked until commercial-clean source ledger and restricted terms are resolved | BUILD_CHECKLIST `[W7 LICENSECHECK P7-4d DONE 2026-07-09]`; `runs/lanes/w7_licensecheck_20260709/report.json` |

Unaccounted gains: none.
