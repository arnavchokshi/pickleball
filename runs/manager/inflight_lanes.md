# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

WAVE 4 OPEN 2026-07-07 (manager session on branch `worktree-wave4-manager`; boot prompt
`runs/manager/wave4_boot_prompt.md`). All lanes dispatched ~2026-07-07 as background codex exec per
manual §10; resume = `cd /Users/arnavchokshi/Desktop/pickleball && codex exec resume -c
model_reasoning_effort=<effort> <SESSION_ID> -` (session id in report.json, or `grep "session id"
runs/lanes/<lane>/log.txt`); harness bg-task ids listed for TaskOutput monitoring.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| ~~w4_cammotion_diag_20260707~~ RULED PASS: root cause = OpenCV decode-orientation context mismatch (portrait img1605 raw-landscape decode collapses features 217→14 → probe 0.33-0.42 vs 53.7 orientation-applied); deterministic repro + --assert-fixed defect proof banked; frame-window/fps/normalization hypotheses refuted | codex DONE | broazt3p3 | — | — | — | done | 2026-07-07 |
| ~~w4_footattr_diag_20260707~~ RULED PASS: defect proof 0/90 confident phases today; measured signal inventory; Design A (BODY-skeleton-direct per-foot producer) RULED IN — predicts 127/186/33/10 confident phases, refine kill=false 4/4, slide gate green; Design B measured non-viable, C too thin | codex DONE | b329ofe32 | — | — | — | done | 2026-07-07 |
| ~~w4_footattr_fix_20260707~~ PARTIAL-per-fence ACCEPTED (uncommitted — held for verify round): all 7 achievable acceptances PASS — defect proof green unmodified; counts EXACTLY match Design A (127/186/33/10) w/ refine kill=false 4/4; offline slide 20.25/22.50/17.98/16.66mm (= w3 GPU values); phaseverify green ×2; 268/0. Fenced process_video stage insertion = deferred patch for integration lane. VERIFY MUST CHASE: exact-match circularity + banked-w3-artifact mutation hygiene | codex DONE (r1) | bw7gwdhl2 | resume via log session id | foot_contact.py + test (working-tree, uncommitted) + deferred_patches/ | — | verify pending | 2026-07-07 |
| ~~w4_footattr_verify_20260707~~ VERDICT: DEFECTS-FOUND ×5: (1) circularity CONFIRMED — patch wires gate-stream transform not skeleton builder (exact-match solved); (2) fabricated confidence — contradictory row certifies @1.0, crossover dual-confident; (3) FORBIDDEN gate-threshold exclusion reintroduced (r2 class, standing-rule violation); (4) deferred patch invalid ("garbage at line 4"; hand-applied copy proved stage order works, refine engages); (5) w3 banked artifacts overwritten w/o backup (originals reconstructible — Wolverine orig 18 src phases vs regen 39 = new population) + new tests vacuous (mutant passes 7/7). Execution evidence on real clips held good | codex DONE | brxa8dtar | — | — | — | done | 2026-07-07 |
| ~~w4_footattr fix r2~~ ALL 5 DEFECTS REPAIRED + THE HONEST FINDING → **THREAD CLOSED, LANDED UNWIRED + PUSHED 75e438223** (queue #2 = measured negative): true skeleton-direct phases = 2-12/clip predicting gate breaches 3/4 (34.6/33.6/48.4mm vs 30mm frozen; wolverine 20.8 passes); dominant rejection phase_penetrates_ground 120-167/clip → RAW-SKELETON NOISE is the binding constraint; circular 127/186/33/10 was an artifact. Producer inert (0 production call sites); forbidden gate-ref exclusion removed from body_grounding_quality; w3 banked artifacts RESTORED; refine un-kill re-queued behind P2-2 latent smoothing (wave 5); stationarity-criterion alternative flagged for scorecard | codex DONE (r2) | bvgifqagd | — | — | — | done | 2026-07-07 |
| ~~w4_cammotion_fix_20260707~~ PROVISIONAL PASS (uncommitted — held for verify round): img1605 production probe 53.70515 AUTO ON (orientation_meta=90); 3 statics bit-exact at baseline scores + identical decode hashes; defect proof green on FRESH production summary; threshold untouched; 303/0 | codex DONE (r1) | bkqb2z298 | resume via log session id | camera_motion.py + test (working-tree, uncommitted) | — | verify pending | 2026-07-07 |
| ~~w4_cammotion_verify_20260707~~ VERDICT: DEFECTS-FOUND (1): set/readback mismatch NOT fail-closed — refusing build → silent normal-looking probe + sparse summary w/ no signal (executable proof banked). Held: reachability single-chokepoint PASS, statics independently bit-exact, tests non-vacuous (2/2 fail on mutated copy), img1605 summary regenerated exact (Δ0.0), telemetry compat 288/288 | codex DONE | b7v5esn1k | — | — | — | done | 2026-07-07 |
| ~~w4_cammotion_fix r2~~ REPAIR PASS: verifier's unmodified proof defect=false; consequential(rotated)→untrusted+OFF-w/-reason, inconsequential→proceed+flag; img1605 53.70515 / statics bit-exact preserved; 4 ruled regression tests; 292/0; fenced first-class-keys patch banked | codex DONE (r2) | bq48lw5dr | — | — | — | done | 2026-07-07 |
| ~~w4_cammotion verify r2~~ CODE CONFIRMED-GOOD on all behavioral attacks → **THREAD CLOSED, LANDED+PUSHED cd0b59390** (queue #1 done). Residual defect = malformed deferred-patch FILE only (bare @@ headers) → integration lane edits fenced process_video directly from verifier's key spec (decode_orientation_mismatch/_consequential_mismatch/_untrusted/_mismatch_reason). LESSON booked: codex deferred 'patches' = design notes, never applyable artifacts — integration lanes re-derive + git apply --check | codex DONE | b921p735z | — | — | — | done | 2026-07-07 |

~~INTEGRATION LANE~~ RULED PASS, LANDED+PUSHED 1588b110f: 4 decode_orientation_* keys persist
first-class in camera_motion_auto (present→verbatim, absent→absent, base keys byte-compatible);
item 2 judged NO-CHANGE-NEEDED w/ evidence; 226/0. VI.0 step 7 DONE.

WAVE-CLOSE: wide-suite adjudication DONE + RULED GREEN — 3041 passed / 15 skipped / 1 failed in
8m29s (log runs/manager/wave4_final_wide_suite.log); the 1 failure = scaffold-index 'unknown'
category caused by UNTRACKED foreign scripts/racketsport/doctor.py (a concurrent session's
in-flight uncommitted work, NOT wave-4-owned; owning session must register it same-lane —
cross-lane note for the closeout bullet). Court benchmark excluded: no court-solver code touched
this wave; wave-3 banked 41/41 standalone cited. OUTAGE RECOVERY: spend-limit stall killed
freshproof pre-VM (no orphans, verified) and idled the ballgpu driver ~2h (VM GPU idle 2h43m
total per its honest PROGRESS.md; ~$5.50-11 incl. idle); both agents resumed, SST capped at 12k
steps by manager ruling, 3 A100 fans (w4fan1/2/3) staging 20:33Z — fleet at exactly the 4-GPU cap;
w4_freshproof GPU lane (Sonnet bg agent — A100 snapshot fans, cap-aware ≤4 incl. sibling
w4ballgpu H100, syncs committed HEAD ≥ 1588b110f, gated keys + camera_motion decisive proof +
warning-string + browser verify ×4). Then: rule ballgpu → docs lane (W4-F) → scorecard + wave-5
boot prompt + memory + branch merge.

INTEGRATION-LANE QUEUE (wave close, ONE lane, fenced files): (1) DROPPED — footattr stage
insertion deferred to wave 5 behind P2-2 (predictor forbids: 3/4 clips would breach the frozen
gate; valid patch banked in the fix lane dir for when phases become gate-safe); (2) cammotion
camera_motion_auto first-class mismatch keys per verifier spec; (3) OPTIONAL --remote-host
parser-level required=True hardening (fleethosts follow-up). All via direct fenced-file edits + focused suites; deferred patch files
are design notes only.
| ~~w4_burlmesh_diag_20260707~~ RULED PASS: warning is NOT burlington-specific — `virtual_world.py:_warnings()` checks embedded vertices only, ignores healthy `body_mesh_index/` sidecar (all 4 clips affected); cosmetic/misleading telemetry, no gate impact | codex medium DONE | b5e43s1au | — | — | — | done | 2026-07-07 |
| ~~w4_fleethosts_20260707~~ RULED PASS, LANDED+PUSHED dcc4dae42: --remote-host required fail-loud; refresh_remote_host.sh idempotent (live proof = w4_h100body); fenced process_video untouched (inherits empty default — parser-level required=True banked as optional close follow-up); 225/0 | codex DONE | bsw3ivn10 | — | — | — | done | 2026-07-07 |
| ~~w4_h100body_20260707~~ PASS → queue #7 CLOSED: snapshot→a3 boot works (pd-balanced required); 27/27 BODY GPU tests; wolverine BODY real @ **2.37× A100** (479.6s vs 1134.4s); stamp 73/73 @ c8c4c0425; VM DELETED list-confirmed; $0.55-4.11. H100 = BODY-VALIDATED (SKU policy updated in gpu_fleet.md). Tool bugs found → w4_syncstamp micro-lane | SONNET DONE | — | — | — | — | done | 2026-07-07 |
| ~~w4_syncstamp_20260707~~ RULED PASS, LANDED+PUSHED 190dea09f: stamps hash committed blobs (hash_source=git_committed_blob, dirty files non-gating INFO; true drift still loud); ssh transport error family retryable-bounded in tar_batch path; 209/0 | codex DONE | b24sabii7 | — | — | — | done | 2026-07-07 |
| ~~w4_bvp_20260707~~ PARTIAL (uncommitted — adjudication pending): D.3(a) improved (bad_fit=0), (c)(d) PASS (Outdoor vf 0.0), (e) RUN F1 exactly unchanged (0.772727/0.875000), a1 killed properly (+207s CPU), LOO per-holdout refit landed, 86/0. D.3(b) 0/5 EXACT intervals — re-segmented not demoted; replacements claim lower endpoint error; Burlington [497,543] baseline-perfect SPLIT. Re-ruling (exact-identity vs span-equivalence) HELD for adversarial adjudication (unfailable-acceptance risk: missing_exact_interval could mask demotions) | codex DONE (r1) | bk3kmgnv1 | resume via log session id | ball_arc_solver.py + test (working-tree, uncommitted) | — | verify pending | 2026-07-07 |
| ~~w4_bvp_verify_20260707~~ VERDICT: DEFECTS-FOUND — span-equivalence REFUTED: endpoint_error improved while fit→fallback tier drops (conf 0.72/0.88→0.10, RMSE 2-5×) on 3 spans; [497,543] split = noise-chasing on unverified contact_prior w/ absurd junction (0.40m, 23m/s, NEGATIVE time gap); LOO refit a SHAM (5 holdouts → 1 unique param set). Vindicated: audit not unfailable (mutation detectable), anchor rule fires 29× non-vacuous, D.3(e)(a)(d) numbers independently confirmed. LESSON: endpoint_error alone is not an equivalence metric | codex DONE | bpe36vzhe | — | — | — | done | 2026-07-07 |
| ~~w4_bvp r2+r3~~ THREAD CLOSED as PARTIAL, FALLBACK LANDED+PUSHED 5633c4b48: r2 honest runtime-kill (in-loop protection 2×); r3 primary met runtime (184s) but failed span quality (1/4 recovered) → fallback per ruling: LOO sham FIXED (5 unique refits, verifier-confirmed) + anchor diagnostics + junction helpers (inactive); F1 Δ0.0 both clips; runtime baseline; 88/0. D.3(b) span protection BANKED FOR WAVE 5 (design: frozen-baseline arc params as protected-span priors + junction repair pre-validity); Magnus stays gated behind STEP 1 | codex DONE (3 rounds) | bt2mbad0u/bpow2ad1p | — | — | — | done | 2026-07-07 |
| ~~w4_ballcode_20260707~~ RULED PASS, LANDED+PUSHED 5b268aa6d: sparse-review semantics solved (reviewed-only 486 rows: 268 pos/218 reviewed-absent; aborts on ambiguous exports; dense helper bypassed as unsafe-for-sparse); occlusion-aug+WBCE paired; strict init key-diff abort; SST manifest+disagreement CLIs; CPU smoke on real ckpt strict-decrease; 44/0. Scaffold commit = filtered 'sst' hunk only (p63's reference_ranges hunks left for that lane) | codex DONE | b8k3t8n6d | — | — | — | done | 2026-07-07 |
| w4_ballgpu_20260707 | SONNET GPU (bg agent) | dispatched post-ballcode | SendMessage resume | runs/lanes/w4_ballgpu_20260707/ + models/checkpoints/wasb/ (verified copy-back) + configs/ssh known_hosts via helper; VM-side only otherwise | H100 a3 spot, self-provision→DELETE; syncs from committed HEAD >= 5b268aa6d (never dirty tree) | ~2-4h | 2026-07-07 |
| ~~w4_court_harvestcal_20260707~~ RULED PASS + KILL FIRED HONESTLY, LANDED+PUSHED 83e090168: 1/6 sources manual_bar (73VurrTKCZ8 med 2.93/p95 6.0px, covers 8/40 clips); HyUqT7zFiwk+zwCtH_i1_S4 full-15pt FAIL p95 36/32px (net/far-side residuals → owner relabel/2nd-frame ask QUEUED); physics-gated teacher stays deferred (<2 sources); run_ball_chain --court-calibration handoff banked | codex DONE | bwlg1m5pg | — | — | — | done | 2026-07-07 |
| ~~w4_burlmesh_fix_20260707~~ RULED PASS, LANDED+PUSHED 684d03380: missing_embedded_mesh_vertices vs true absence distinguished fail-closed; viewer copy; 354 tests + vitest green; no gate consumer weakened | codex DONE | br1brmo31 | — | — | — | done | 2026-07-07 |

FOREIGN LANE (dispatched by the Fable-final/succession session, noted uncommitted on the MAIN
tree's copy of this board — preserve at merge): p63_reference_ranges_20260707 | codex | owned:
ONLY NEW FILES docs/racketsport/reference_ranges_{schema,v0}.json,
scripts/racketsport/validate_reference_ranges.py, tests/racketsport/test_reference_ranges.py
(+scaffold-index line — shared touchpoint with w4_fleethosts/w4_ballcode/w4_court_harvestcal
registrations; check at adjudication) | ~1-2h from 2026-07-07.

CONCURRENT-SESSION UPDATE: brand-v2 (wave-2 manager) session COMPLETED its arc — ios work
committed+pushed by it; its final docs bullet (c657a25c0) rode along with my 684d03380 push.
ios/** stays untouched by wave-4 regardless.

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
