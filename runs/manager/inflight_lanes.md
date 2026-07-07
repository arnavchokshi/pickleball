# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

WAVE 3 (dispatched 2026-07-07 UTC; plan: runs/manager/wave3_plan.md; specs: runs/lanes/<lane>/spec.md):

| lane | kind | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| w3_codesync_20260707 | codex | bg bkkdh47i0 | codex exec resume <session_id from report.json> | remote_body_dispatch.py, scripts/fleet/*, its tests | — | ~1-2h | 2026-07-07 |
| ~~w3_slidediag_20260707~~ | codex (diagnosis) | DONE — ACCEPTED (PARTIAL-by-design: offline limit pre-authorized). Guard-fire mechanism REFUTED (11-14% overlap, wolverine 0%); root cause = weak bilateral unknown-foot contact phases; outdoor max localized P3 r 791-804 @56.002mm inside weak phase; burlington near-gate p06-identical. | | | | ruled 2026-07-07 | |
| ~~w3_groundref_diag_20260707~~ | codex (diagnosis) | DONE — ACCEPTED. Self-kill = honest before/after foot-plane predicate; inputs are the fault: 100% bilateral_from_player_stance phases, 0 confidence keys, exact-foot agreement 0.363-0.651. Fix = upstream per-foot confident phases. | | | | ruled 2026-07-07 | |
| w3_cammotion_conditional_20260707 | codex | bg b5dkewwcs | codex exec resume <session_id from report.json> | camera_motion.py, estimate_camera_motion.py, process_video.py, their tests | — | ~1-2h | 2026-07-07 |
| ~~w3_img1605_mesh_diag_20260707~~ | codex (diagnosis) | DONE — ACCEPTED. Root cause: all img1605 frames manual_review_required → eligibility starves even the existing uniform fallback (eligible=0, selected=0). Burlington HUD notice = distinct virtual_world issue (166 mesh frames exist). | | | | ruled 2026-07-07 | |
| w3_phasefix_20260707 | codex FIX | bg bo22oiy8d | codex exec resume <session_id from report.json> | foot_contact.py, placement.py (phase/lock paths), foot_lock_solver.py, footlock.py, foot_pin.py, body_grounding_refine.py, worldhmr.py+body_grounding_quality.py (instrumentation only), pose_temporal.py (phase paths only), tests | — | ~2-3h | 2026-07-07 |
| w3_meshfallback_20260707 | codex FIX | bg bxdvnb82y | codex exec resume <session_id from report.json> | frame_rating.py, body_mesh_readiness.py (if needed), tests | — | ~1h | 2026-07-07 |
| w3_p11_prep_20260707 | codex | bg bgke5jozb | codex exec resume <session_id from report.json> | roboflow_corpus.py, ball_tracknet_cvat_dataset.py, NEW train_ball_pretrain.py, their tests | — | ~2-3h | 2026-07-07 |
| ~~w3_labelfactory_20260707~~ | sonnet | DONE — ACCEPTED (PASS). P0-4 LAUNCHED: CVAT 2.69.0 @ localhost:8080 (project 2, tasks 7-12, ~81 frames each), WASB prelabels imported 6/6 w/ screenshots, round-trip schema-valid, held-out clean (defense-in-depth), owner guide in lane dir (merge to docs/ at checkpoint), RAFT-small prefetched sha-recorded (opt-in only). Gotchas booked: frame_step not frame_filter; absolute frame ids in import XML; visibility defaults flagged in guide; docker VM 7.65GB tight for 17 containers. | | | | ruled 2026-07-07 | |
| ~~w3_fleetseed_20260707~~ | sonnet | DONE — ACCEPTED (PASS). fleet1 synced to HEAD (whole-tree proof) + STOPPED, $0.30, new IP 35.240.183.195. STRUCTURAL: 3D chain CLIs hard-require court cal → harvest teacher = 2D chain only (P4 dependency booked); 2D gate cuts 71-74%→27% coverage (canary false-lock correctly killed) → tune before mass-seed (manager ruled option B). | | | | ruled 2026-07-07 | |
| w3_teachertune_20260707 | codex (local CPU) | bg bhftujpel (r2 running; r1 BLOCKED honestly: 34/40 raw sidecars pruned on-disk post-wave-2 — owner Finder cleanup evidence (.DS_Store trail 23:15-23:30 PDT, survivors == exactly the 6 CVAT review clips); resumed at 8-clip scope (6 survivors + 2 fleetseed raw, canary incl.); report lands at report_r2.json (lane-written)) | codex exec resume -c model_reasoning_effort=xhigh 019f3b31-fd93-70d0-95c8-a6a2f1575d1d - <<prompt (NOTE v0.142.5: exec-level flags --cd/--sandbox/--output-schema/-o are REJECTED after `resume`; only -c works; sandbox+cwd persist from the recorded session — manual §10 resume template needs this correction) | NONE (runs/ only; runs repo CLIs) | — | ~1-2h | 2026-07-07 |

Held for later this wave: slide-FIX lane (after slidediag ruling) → adversarial verify → fresh GPU
proof; P1-1 GPU training (H100-first per owner directive 2, after p11_prep lands); P1-2 mass seeding
(after teacher validation); wave-end composed fresh-worlds GPU run + wide suite + browser verify.
