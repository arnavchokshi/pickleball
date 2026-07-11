# Tranche-2 plan — 2026-07-10 autonomous window

## Decision

Run correctness and provenance work before spending the next GPU-hour. Start the iOS real-manifest
route and an isolated TT3D anchor-search core now, while the already-dispatched lanes finish. After
the spine owner lands, close the `unfingerprinted_stale` reuse loophole. Land a parity-only typed-
coordinate adoption slice before final GPU attestation. Then run one bounded TRK recovery sweep and
one BALL E2E attestation, on at most two new H100s while the court session may occupy one slot.

This is a planning ruling, not a promotion. `VERIFIED=0` remains binding. Protected Outdoor and
Indoor labels are excluded from every command and input manifest. BEST-STACK DELTA: **(c) none**.

## Why this order

- North Star says the product-route defects outrank isolated model work, requires NS-01 before
  trustworthy user evidence, and puts coordinates/status before spine and physical proof
  (`NORTH_STAR_ROADMAP.md:148-158`, `:292-308`, `:395-418`).
- The current wrapper still lets `unfingerprinted_stale` artifacts pass stage-local reuse and become
  a first content-addressed generation (`scripts/racketsport/process_video.py:1057-1063`). That can
  make a correct-looking GPU report score stale pixels or stale outputs.
- TRK loss is upstream and measurable: the audit found the zero-apron filter, missing ReID asset,
  and top-four fragmentation as separate droppers; it specifically requires a positive-margin plus
  actual-OSNet candidate followed by the frozen scorer, not visual tuning
  (`runs/lanes/dr_pipeline_20260710/FINDINGS.md:35-53`).
- BALL has two different failure classes. UKF can only recover near accepted fits and has a 13.8%
  owner-clip ceiling, while TT3D anchor search changes the structural fit and has a pre-registered
  Wolverine kill of fallback `<5/11` (`runs/lanes/dr_pipeline_20260710/FINDINGS.md:95-101`).
- iOS is not merely missing device proof: the app unconditionally calls
  `WorldBundle.loadBundledSample()` for the selected capture (`ios/App/AppRootView.swift:1631-1703`).
  Upload state has the matching `jobId` but no persisted manifest URL
  (`ios/Upload/Sources/PickleballUpload/UploadQueue.swift:13-49,217-237`), so the code defect is
  independently fixable and testable without the owner's phone.

## Dispatch and synchronization rules

1. No new lane may edit a file owned by `ballcand_20260710`, `spine017_20260710`, the ReID restore
   lane, the labeling/import lane, or any court lane. The manager rules and commits tranche-1 before
   any serialized integration edit begins.
2. Every scored run pins: committed HEAD, source-video SHA-256, calibration SHA-256, scorer SHA-256,
   GT package SHA-256, model/checkpoint SHA-256, candidate config, CUDA/Torch versions, and exact
   clip allowlist. An input containing `outdoor` or `indoor` (case-insensitive) is a hard preflight
   failure, not a skipped row.
3. Do not sync an uncommitted working tree to a VM. Tranche-1 must be ruled and committed first.
4. Provision only after all inputs are staged. H100 spot is the default and must be `<= $5/hr`.
   While a court GPU exists, provision at most two tranche-2 GPUs. Without it, never exceed three
   manager-owned concurrent GPUs. Arm a 60-minute no-heartbeat self-stop and delete/list-confirm at
   lane end.
5. H100 stockout kill: stop after 30 minutes or six spaced create attempts, whichever comes first;
   report `no-attempt`. Do not silently change GPU SKU for a decisive comparison.
6. Baseline runs first and must reproduce the pinned scorer/artifact protocol before any candidate
   result is interpreted. Runtime is reported but is not the accuracy gate.
7. No lane changes selected defaults, `best_stack.json`, model manifests, or promotion state. All
   candidates remain explicit and default-off.

## Ranked tranche-2 lanes

| Rank | Lane | Objective | Acceptance and kill | Files fence | GPU | Estimate |
|---:|---|---|---|---|---|---|
| 1 | `ns013_stale_reuse_close_20260710` | Remove the legacy first-generation reuse loophole before any attestation. A stage with no trusted fingerprint must rebuild, or consume an explicit immutable migration/import attestation. | **Accept:** changed video/model/config/upstream hashes rebuild the exact closure; identical fingerprinted input reuses; an explicit migration is separately identified; cold/reuse/partial tests pass. **Kill/stop:** if compatibility requires silently adopting an unfingerprinted artifact, do not land that behavior; leave it blocked and do not launch GPU scoring. | After `spine017` lands: `scripts/racketsport/process_video.py`, `threed/racketsport/run_identity.py`, `tests/racketsport/test_run_identity.py`, exact matching process-video tests, lane dir. | No | 1–2 h |
| 2 | `ios_real_replay_20260710` | Wire selected capture -> persisted clip/job identity -> job status -> matching `manifest_url` -> real `WorldBundle`; include the deferred NS-01.5 native partial/trust fields and the missing `ball_arc_render_url` decoder field. | **Accept:** URLProtocol/temp-fixture tests prove matching row identity, `complete` and honest `partial` are inspectable, relative HTTP and file artifact URLs resolve, missing optional assets do not substitute another capture, and a row without ready output shows an explicit not-ready state. `swift test --package-path ios` and simulator `xcodebuild build-for-testing` pass. The production row never falls back to the bundled sample. **Kill/stop:** auth unavailable is a typed state; no physical/device claim; no fixture fallback on any error. | `ios/App/{AppRootView,DinkVisionModels,DinkVisionRuntimeConfiguration,DinkVisionUploadCoordinator}.swift`; `ios/Upload/Sources/PickleballUpload/{RenderGatewayClient,UploadQueue}.swift`; `ios/Replay/Sources/PickleballReplay/World/{WorldBundle,WorldViewerManifest}.swift`; exact Swift tests; lane dir. No capture implementation changes. | No | 2–4 h; start now |
| 3 | `tt3d_anchor_core_20260710` | Start the structural BALL centerpiece now without colliding with `ballcand`: implement a pure joint bounce/contact-anchor candidate search adapter around immutable observations, court/net planes, and the existing solver contract. | **Accept:** deterministic unit/fixture tests cover candidate enumeration, ray-plane constraints, robust bounds, provenance, and refusal on insufficient observations. The module emits candidates only; it cannot mark them measured or alter defaults. **Kill/stop:** if an honest core cannot be built without editing active ball-owned files, stop as `no-attempt` and wait for integration; never copy or fork the solver. | New `threed/racketsport/ball_joint_anchor_search.py`, new `tests/racketsport/test_tt3d_joint_anchor_search.py`, lane dir only. Explicitly forbidden now: `ball_arc_chain.py`, `ball_arc_solver.py`, `ball_physics_fill.py`, `ball_ransac_arc_gate.py`, `process_video.py`. | No | 1.5–3 h; start now |
| 4 | `tt3d_anchor_integrate_20260710` | After `ballcand` is ruled/committed, connect the isolated anchor search as an explicit default-off solver candidate and produce the exact candidate artifacts needed by E2E scoring. | **Accept:** on/off parity, raw observations byte-identical, chosen/rejected anchor hypotheses persisted, all resulting samples traverse the existing fail-closed gate, and the candidate is callable from the frozen benchmark. **Kill:** Wolverine remains `>=5/11` fallback after scoring -> reject TT3D and do not start whole-rally DP segmentation. Any measured/provenance promotion or fail-closed bypass is an immediate stop. | After ball owner release: `threed/racketsport/ball_joint_anchor_search.py`, `ball_arc_solver.py`, `ball_arc_chain.py`, exact tests, lane dir. `process_video.py` only through the serialized integration owner if a public candidate flag cannot otherwise reach E2E. | No for build; GPU only in rank 7 attestation | 1.5–3 h after rank 3 and `ballcand` |
| 5 | `ns014_coords_adopt_remainder_20260710` | Adopt typed coordinate metadata/adapters at the paddle and exact TRK/BALL projection seams without changing numerical behavior. This is contract work, not the hypothesized paddle-accuracy fix. | **Accept:** raw/undistorted/preview/world inputs are explicit; canonical world literal replaces undocumented `court_Z0` at new boundaries with backwards-compatible decoding; distorted synthetic and frozen real-fixture parity tests show no numerical delta beyond declared tolerance; raw values remain. **Kill/defer:** any behavior-changing transform or unexplained positional delta is stopped until marker/gold GT. Do not claim better paddle placement. | Start only after conflicting tranche-1 owners release files. `threed/racketsport/coordinates.py`, `paddle_pose_fused.py`, narrowly required projection adapters such as `court_calibration.py`/`person_fast.py`, exact tests, lane dir. No court model/trainer, labels, ball solver, or process spine edits. | No | 2–3 h |
| 6 | `trk_reid_apron_score_20260710` | Measure the restored OSNet path and fixed apron candidates against the loose-pool baseline with one frozen scorer and unchanged detections. This is the bounded ReID/off-court recovery test authorized by the audit, not another open association sweep. | **Accept:** pre-register `{loose_pool, association_margin_0.5m, 1.0m, 2.0m}` with every non-margin knob fixed; score every arm on the same existing reviewed HARVEST person-GT allowlist, plus Burlington/Wolverine as clearly labeled historical internal diagnostics; report per-clip IDF1, switches, spectator FP, far-off-court FP, four-player coverage, runtime, and worst clip. No Outdoor/Indoor read. **Kill:** baseline reproduction/provenance mismatch stops all arms; any forbidden clip/hash mismatch stops the lane; if no arm improves worst-clip IDF1 and coverage over loose-pool without a new switch/FP, stop association work and route the next TRK experiment to detector/domain leverage (RF-DETR), not more margins. Full-gate interpretation remains IDF1 `>=0.85`, 0 switches, 0 spectator/far-off-court FP, coverage `>=0.95`; no stack promotion from historical/internal clips. | Scoring is read-only against `scripts/racketsport/{benchmark_person_trackers,score_person_track_sources}.py` and `threed/racketsport/person_track_gt_scoring.py`; outputs only under its lane dir/VM. If a harness defect is found, stop and dispatch a separate fenced code lane. | Yes, one H100 spot | 2–4 h including staging; 4 h wall cap |
| 7 | `ball_e2e_attest_20260710` | Attest the landed BALL candidates on fresh E2E generations, not only recomposed old artifacts. Compare baseline, UKF/RANSAC, TT3D, and the combined survivor set with the same 2D scorer and fail-closed emission audit. | **Sequence:** baseline -> `UKF+RANSAC` -> TT3D -> combined only if both individual arms survive. Use Wolverine/Burlington reviewed internal cards for frozen 2D metrics and the owner-class zwcth clip for source-only fallback/emission generalization. **Accept:** baseline reproduces; F1@20/recall/hFP and p95/p99/teleport metrics use one scorer; every recovered frame is `physics_predicted` with covariance/horizon; zero court/net/contact-crossing hard violations; fail-closed suppression parity preserved. **Kill:** baseline mismatch/stale generation stops all; hFP regression or any fail-open/provenance error rejects the arm; TT3D fallback `>=5/11` rejects TT3D and kills DP follow-on; UKF cannot be credited beyond its measured eligible segments. No promotion without fresh source-disjoint gate data. | Committed tranche-1/TT3D code is read-only on the VM; outputs under lane dir. No code patches during attestation. Protected labels excluded. | Yes, one H100 spot | 2–3 h; 3 h wall cap |
| 8, conditional | `trk_rfdetector_probe_20260710` | Only if rank 6 decisively kills association recovery and at least two hours remain, run the next North-Star TRK step: one pinned RF-DETR det/seg baseline with detections scored before association. | **Accept:** checkpoint/license/provenance preflight, same reviewed allowlist and scorer, detector metrics separated from association, runtime recorded. **Kill/no-attempt:** absent commercial eligibility, absent frozen harness, <2 h remaining, or stockout. Reject if detector wall grows >20% without full-gate gain. No McByte/CAMELTrack in this window. | Separate new lane; no reuse of rank-6 output directory; only benchmark adapter files if pre-authorized, otherwise VM/output-only. | Conditional one H100, never a third new GPU while court holds one | 1.5–2.5 h |

## GPU allocation and sequence

1. Do **not** reserve a GPU for code development. Let ranks 1–5 and tranche-1 finish first.
2. Provision TRK H100-A as soon as the ReID artifact hash, content-reuse fix, coordinate-parity slice,
   committed code pin, frozen scorer, and allowed GT manifest exist. Run rank 6.
3. Provision BALL H100-B only after `ballcand`, TT3D integration, spine dependencies, and the stale-
   reuse fix are committed. Run rank 7. H100-A and H100-B may overlap only if the court slot plus
   both remains within the manager's cap.
4. Keep the remaining capacity unallocated. It is contingency for preemption/retry, not an excuse
   for an unplanned sweep. Rank 8 may consume it only after rank 6's pre-registered kill fires and
   its own prerequisites pass.

There is no higher-value GPU job in the current executable queue. The higher-value work is CPU-side
correctness: matching iOS replay routing and eliminating stale first-generation reuse.

## Direct answers to the six questions

### 1. Tranche-2 GPU allocation

Allocate one H100 to the bounded TRK ReID/apron comparison and one to BALL E2E attestation, in that
order of readiness; do not launch either before provenance gates. TRK may start first while BALL code
integration finishes. Keep a third slot empty unless the association sweep is killed and the pinned
RF-DETR probe is ready. Per-run and provisioning kills are in ranks 6–8 above.

### 2. iOS replay fixture hardcode

**Do it now.** It is the owner's active surface and a concrete P0-B/P0-E product-route defect, not a
physical-device-only task. Swift unit tests, URLProtocol fakes, local HTTP/file fixtures, full package
tests, and simulator build-for-testing can prove row/job/manifest identity and decode behavior. The
lane must explicitly report that NS-01.2b remains blocked: only the owner's signed device can prove
record/import -> upload -> GPU -> own replay with auth. No phone means no physical-E2E claim, not a
reason to leave known fixture routing in production.

### 3. NS-01.4 typed-coordinate adoption remainder

**Do a parity-only remainder this window, before final TRK/BALL attestation, after conflicting owners
release their files.** The roadmap orders coordinates ahead of NS-03 scoring. However, do not treat
the paddle's ugly position as proof of a transform error: the audit says that hypothesis is unproved
and the estimator has no corner/pose GT. Adopt types, declared raster conventions, canonical schema
literals, and validated adapters; stop any behavior-changing math until gold/marker GT.

### 4. TT3D joint-anchor search

**Start now in parallel, but split core from integration.** UKF and TT3D address different failure
classes, and the owner-class UKF ceiling is too low to justify waiting. The immediate lane may add
only a new pure candidate-search module and a non-overlapping test. Integration into the existing
solver/arc chain waits for `ballcand` to land. E2E scoring then enforces the pre-registered `<5/11`
Wolverine fallback kill. Do not start TT3D whole-rally DP if that kill fails.

### 5. Single biggest unattended-window risk

The biggest risk is **scoring the wrong generation while producing a plausible report**. The repo
currently permits unfingerprinted legacy reuse, the ReID asset is being restored concurrently, and
GPU snapshots can lag the committed code. That combination can make baseline/candidate deltas
meaningless without visibly crashing. Rank 1, commit-only VM sync, complete input hashes, exact clip
allowlists, and mandatory baseline reproduction are the stop rules. Stockout is cheaper and more
obvious than a scientifically invalid score.

### 6. Higher-value executable item from section 5 / defect ledger

Yes: close `unfingerprinted_stale` reuse first. It is executable without the owner, directly closes
the remaining P0-C loophole, and is higher value than a second model/association experiment because
every later score depends on artifact identity. The other high-value executable item is the iOS
real-manifest plus partial/trust propagation lane, which completes the deferred native half of
NS-01.5. The 1,200-frame representation-cap policy is important but needs a product-policy ruling;
RKT accuracy and physical proof remain owner/GT blocked.

## AGREE / DISAGREE with dispatched tranche-1

### AGREE

- **ReID asset restore:** correct prerequisite for measuring the intended association path. Require
  checkpoint SHA-256, provenance/license, manifest-path resolution, and a real embedding smoke. The
  restore itself is not an accuracy pass or best-stack change.
- **BALL candidate lane:** agree with UKF, RANSAC, blur-footprint inspection, default-off flags,
  frozen scoring, and fail-closed authority. Agree that missing persisted blur footprint should end
  as a documented stop instead of expanding into detector edits.
- **Spine lane:** agree that one owner must serialize `process_video.py`; the deferred runner honesty,
  post-BODY refined events, audio path, and dependency hashes are the right NS-01.5/01.7 work.
  Its structural-wall stop rule is essential because the scope is large.
- **Mac hardware FPS:** agree as a scoped hardware-WebGL measurement and as a correction to the
  SwiftShader numbers. It does not replace the owner-iPhone trace and must not become a device-wide
  performance claim.
- **Codex implementation model:** all new implementation lanes in this plan use the owner-directed
  `gpt-5.6-sol xhigh`. The already-dispatched Sonnet ReID restore is acceptable only as asset/ops
  restoration; any source-code fix discovered there must be handed to a Codex implementation lane.

### DISAGREE / tighten

- Do not treat `ballcand`'s offline artifact scoring as E2E attestation. Its module flags are
  default-off and it cannot own the process spine; rank 7 must regenerate pinned E2E artifacts after
  integration and prove the candidates actually ran.
- Do not gate TT3D research on UKF results. UKF's diagnosed owner-clip ceiling already proves it
  cannot answer the structural anchor problem. Only TT3D integration is serialized behind the
  ball-file owner.
- Do not launch the TRK GPU sweep immediately after the asset appears. First close stale reuse,
  freeze the candidate matrix, and use reviewed labels with one scorer. Otherwise the audit's
  0.5/1/2 m examples become threshold shopping.
- Do not interpret a restored ReID path, green Swift tests, Mac FPS, or internal BALL/TRK gains as a
  promotion. `VERIFIED=0`, physical NS-01.2b, fresh source-disjoint gates, and gold-dependent RKT/
  BALL-3D accuracy remain unchanged.

## Ten-hour timeline

- **T+0 to T+3:** tranche-1 continues. In parallel, start ranks 2 and 3. No GPU provision.
- **T+2 to T+5:** manager rules/commits tranche-1. Run rank 1 after spine release; run rank 4 after
  ball release; run rank 5 on released projection files. If any cannot obtain a clean fence, defer
  that lane rather than overlap.
- **T+4 to T+8:** provision H100-A for rank 6 when all preflights pass. Provision H100-B for rank 7
  when TT3D integration is committed. The two may overlap within the cap.
- **T+8 to T+10:** finish scoring, pull/hash artifacts, delete/list-confirm VMs, and write explicit
  adopt/reject/partial/no-attempt reports. Run rank 8 only if its trigger and time budget both pass.

## End-of-window required truth

- Every lane has a structured report, exact code/input hashes, and a teardown confirmation if it
  used a GPU.
- Protected label reads: zero.
- Default/best-stack changes: zero.
- Physical-device, gold-capture, iPhone-FPS, pb.vision-identity, and labeling-continuation gates are
  still reported as owner-blocked unless the owner independently returns.
- No scoped pass is relabeled `VERIFIED`; no candidate is promoted from visual plausibility,
  internal-val-only gain, or optimizer residual.

