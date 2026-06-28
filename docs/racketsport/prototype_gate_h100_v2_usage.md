# Prototype Gate H100 v2 Usage

Date: 2026-06-28

This is the current runnable prototype path for the accepted-four pickleball
clips. It is set up for qualitative review and held-out benchmark comparison.
It is not a production BALL gate and does not make BALL `VERIFIED`.

## Accepted Clips

Use only these clips for the current prototype gate:

- `burlington_gold_0300_low_steep_corner`
- `wolverine_mixed_0200_mid_steep_corner`
- `outdoor_webcam_iynbd_1500_long_high_baseline`
- `indoor_doubles_fwuks_0500_long_mid_baseline`

Burlington remains useful for BODY/player/ball/paddle smoke, but it is retired
for court calibration because fisheye curvature bends court lines. Do not use
it to prove no-tap or line-based court calibration.

Do not use `side_view_game5_0100_high_side_fence` for this gate. It is
`DEFERRED_REJECTED_SIDE_FISHEYE`.

## Current Usable Ball Artifact

For each accepted clip, the current strict no-click review track is:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json
```

Its watchable overlay is:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4
```

This track is built from:

- target-court-filtered TrackNetV3 output
- the motion-consistent TrackNet temporal path
- VballNet Fast and VballNet V1 verifier tracks
- a local trajectory-consistency filter that hides points far away from the
  surrounding before/after path

It does not read `ball_points.json`, click-corrected tracks, or any human-click
oracle output. The human clicks are held-out benchmark labels only.

## Current Benchmark Read

Latest benchmark summary:

```text
runs/eval0/prototype_gate_h100_v2/ball_tracker_benchmark/benchmark_summary_fusion_localtraj.md
```

Aggregate result on the four accepted clips:

| candidate | hit recall | p90 error px | hidden FP | teleports | score |
|---|---:|---:|---:|---:|---:|
| `fusion_temporal_vball100` | 0.563 | 46.784 | 0.425 | 8 | 0.235 |
| `fusion_temporal_vball100_localtraj` | 0.509 | 38.851 | 0.294 | 0 | 0.280 |
| `tracknet_court_temporal_path` | 0.415 | 35.043 | 0.325 | 5 | 0.169 |
| `tracknet_raw` | 0.694 | 431.866 | 0.950 | 71 | -0.243 |

Interpretation: `fusion_temporal_vball100_localtraj` is the current best strict
review artifact because it removes the remaining teleport-style jumps and lowers
hidden false positives. It hides more uncertain ball frames than the looser
fusion track, so use `fusion_temporal_vball100` as the recall comparison track.

## Human Review Packet

The easiest current review entrypoint is:

```text
runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review.html
```

Open that HTML file in a browser. It embeds the calibration and label overlay
videos, lists each `frame_compute_plan.json`, and summarizes each
`body_compute_execution.json` and `virtual_world.json`. Burlington, Wolverine,
and Indoor doubles now include pulled-back H100 BODY mesh outputs for the
scheduled world-mesh frames. Outdoor webcam remains an inspectable court-frame
assembly with player tracks and ball frames, but it has no BODY mesh because
its current frame plan schedules zero world-mesh BODY work. Paddle pose remains
preview-only until canonical RKT artifacts are generated.

For direct correction entry, the local review UI now reads the generated action
manifest. It shows global runtime/setup blockers once at the top, then shows
the per-clip blockers above each editable review section:

```bash
python scripts/racketsport/review_input_server.py --port 8765
```

Then open `http://127.0.0.1:8765`. Use it when you want to click top-net
points, add contact/rally notes, mark visible paddle examples, and save one
review JSON that can be exported into the corrections manifest.

The packet also lists a separate `virtual_world_paddle_preview.json` for each
accepted clip. These preview worlds include paddle meshes produced from the
current box-corner candidates, but they are marked `ambiguous_paddle_pose` and
do not replace canonical `racket_pose.json` or promote RKT. After the scheduled
H100 BODY run, Burlington, Wolverine, and Indoor doubles include real BODY mesh
samples in their preview worlds; Outdoor webcam still warns
`missing_mesh_vertices` because it has no scheduled BODY frames. Paddle preview
counts remain Burlington 42, Wolverine 36, Outdoor webcam 397, and Indoor
doubles 280.
The virtual-world summaries now also expose uncertainty counts used by the
packet and Three.js review HUD: approximate court-plane ball frames, mesh/joint
or track-only player frames, paddle players, and ambiguous paddle frames. In the
current accepted-four preview worlds, every paddle frame is still ambiguous:
Burlington 42, Wolverine 36, Outdoor webcam 397, and Indoor doubles 280.

Each accepted clip now also has `body_mesh_readiness.json`, which is the
explicit joints-vs-mesh audit. It now cross-checks
`frame_compute_plan.json` world-mesh demand and `body_compute_execution.json`
scheduled BODY work against actual `smpl_motion.json`/`skeleton3d.json` mesh
availability. Current local audits report Burlington
`world_mesh_required_available_unverified` with 9 mesh player-frames, Wolverine
`world_mesh_required_available_unverified` with 12 mesh player-frames, Indoor
doubles `world_mesh_required_available_unverified` with 6 mesh player-frames,
and Outdoor webcam `no_world_mesh_requested`. The mesh outputs carry 18,439
vertices per player frame. None of these promote BODY because the promoted
contact windows and schedule manifests are prototype artifacts, not
world-MPJPE/full-clip BODY gates.

The linked `racket_pose_readiness.json` files are the authoritative RKT gate
state. In the current accepted-four packet every candidate frame is
`box_derived`, while `true_corner_frame_count=0`,
`reference_gt_frame_count=0`, and `promoted_pose_frame_count=0`. That means the
preview meshes are useful for visual inspection, but paddle angles, contact
normal metrics, and RKT promotion remain blocked until true paddle
keypoint/mask or CAD pose evidence and reference/GT evaluation are present.
The Phase 6 evaluator now follows the same rule: `racket_pose.json` player,
frame, and contact counts are recorded only as `artifact_check.*` context. It
reports `not_measured` unless `labels/racket_pose.json` contains reviewed
reference/ArUco pose labels, and only those labels can drive the
`label_check.racket_face_angle_p90_error_deg <= 5` and contact-point gates.

Each accepted clip also has `racket_promotion_audit.json`. These audit files
prove the current preview paddle path has not leaked into canonical
`racket_pose.json`: all four accepted clips are `safe_preview_only` with
`canonical_racket_pose_present=false`, `promoted_pose_frame_count=0`, and
`unsafe_promoted_frame_count=0`.

Each accepted clip also has `paddle_true_corner_review.json` and
`racket_candidates/paddle_true_corner_crop_sheet.png`. These artifacts list the
specific box-candidate frames that need real paddle-face corner labels in
top-left, top-right, bottom-right, bottom-left order. They remain
`blocked_missing_true_corner_labels` until reviewed true-corner, CAD, or
reference/ArUco evidence is supplied; copying box corners into the true-corner
label file is not valid.

The packet has one global `racket_model_runtime_readiness.json` entry under
`__global__`. It is CPU-only and does not import model runtimes, use GPU,
download checkpoints, or claim any paddle model has run. Current status is
blocked: SAM 3, DINO-X/Grounded-SAM2, FoundationPose, GigaPose, FoundPose,
paddle CAD/reference assets, and reference/GT labels are not declared ready for
a paddle 6DoF GPU smoke.

The packet also lists `pipeline_readiness_e2e.json` for each accepted clip.
These are artifact-plus-semantic readiness reports, not accuracy gates. Current
reports show Burlington, Wolverine, and Indoor doubles have BODY mesh artifacts
but are still blocked from BODY promotion by missing world-MPJPE/full-clip gates
and from E2E by missing canonical RKT, metric, copy, drill, coach, and replay
outputs. Outdoor webcam additionally lacks `smpl_motion.json` and
`skeleton3d.json` because no BODY frames are scheduled. These reports are not
accuracy gates and can lag regenerated contact/BODY artifacts, so rebuild them
after changing `contact_windows.json` or BODY schedules. All four clips have
`court_line_evidence.json`, but the
automatic court evidence remains fail-closed unless trusted semantic line/top-net
evidence is present; Burlington should not be used for court calibration because
fisheye curves the court lines.

Each preview world now has a linked Three.js review page:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/virtual_world_paddle_preview.html
```

Serve from the repo root so the page can import the local Three.js module:

```bash
python -m http.server 8878 --bind 127.0.0.1
```

Then open
`http://127.0.0.1:8878/runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review.html`.
The packet links the four world-review pages.

The packet also indexes review-only `replay_scene.json` manifests for each
accepted clip. Those scenes point at static CPU-generated GLBs under:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/replay_review/
```

They include court/net lines, player paths, BODY joint/mesh point clouds when
present, ball paths, and preview paddle triangles when present. They are useful
for load/reference checks. Replay and e2e evaluators now also parse referenced
GLBs for GLB 2.0 header/chunk/glTF JSON structure, so placeholder bytes fail
the artifact gate. This is still not the final animated/compressed
OpenUSD/USDZ/GLB replay export and does not make RPL verified.

Each accepted clip also has a browser replay-viewer manifest:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/replay_viewer_manifest.json
```

Start the local viewer from `web/replay`:

```bash
npm run dev -- --port 5173
```

Then open:

```text
http://127.0.0.1:5173/?manifest=%2F%40fs%2FUsers%2Farnavchokshi%2FDesktop%2Fpickleball%2Fruns%2Feval0%2Fprototype_gate_h100_v2%2F<clip>%2Freplay_viewer_manifest.json
```

The viewer plays the local smoke video, overlays the current player boxes,
renders the court/net and players in Three.js, and shows floor placement and
physics status. The manifests also link the two imported
`racketsport_person_ground_truth` annotation files under
`runs/phase2/iphone_person_tracking_eval/labels/task_2376761/` and
`task_2376765/`; these are available as trusted annotation sources, while the
accepted-four `labels/players.json` overlays remain review-only when marked
`not_ground_truth`.

Each accepted clip now also has `physics_refinement.json`, produced from
`virtual_world.json` floor/contact constraints through the existing CPU
physics-refinement scaffold. Current artifacts are
`physics=cpu_fallback_scaffold`, `foot2_done=false`, and zero positive contact
frames because the current BODY output did not produce positive foot-contact
evidence. This is useful for replay inspection and downstream contract wiring,
but it is not MuJoCo/MJX, PhysPT, PHC/PULSE, FOOT-2, or physics verification.

The packet also includes H.264 player-track overlay videos for each accepted
clip under:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/player_tracks/player_track_overlay_h264.mp4
```

These are qualitative tracking review videos only; they do not make tracking
`VERIFIED` without the labeled IDF1/spectator/ID-switch gates.

The packet now also includes H.264 paddle-candidate overlay videos under:

```text
runs/eval0/prototype_gate_h100_v2/<clip>/racket_candidates/racket_candidate_overlay_h264.mp4
```

These overlays draw the current strict four-corner candidate boxes, scaled from
the 960x540 label-frame coordinate space to the 1920x1080 video space, and list
how many candidates were actually rendered. Burlington and Wolverine currently
use the existing 10-second TrackNet smoke source because their current
candidate frames are inside that window. Outdoor webcam and Indoor doubles use
recovered 30-second source clips under each `racket_candidates/` directory, so
all current candidate frames are now visible in the packet. If those recovered
sources are missing and the 10-second smoke video is used instead, the overlay
index will warn `candidate_frames_outside_video_window`.

When a valid `racket_pose.json` is present, `virtual_world.json` includes the
paddle SE(3) pose plus a rectangular paddle mesh (`mesh_vertices_world` and
`mesh_faces`) computed from the known paddle dimensions.

When only ambiguous box-derived candidates are present, use the preview-only
path below instead. It writes `racket_pose_preview.json` and
`virtual_world_paddle_preview.json`; both are human-review helpers, not RKT
accuracy artifacts.

Visible 2D ball detections without `world_xyz` are projected onto the court
plane with the calibration homography during virtual-world assembly. Those
fallback points are marked `approx=true`; they are usable for review pathing but
do not estimate true ball height.

Each `frame_compute_plan.json` now records a per-frame `target_representation`
(`track_only`, `joints_or_preview_mesh`, `world_mesh`, or
`manual_review_required`) plus per-player `player_targets`. The player targets
explain whether each tracked player should stay track-only, use a skeleton or
preview mesh, run deep mesh, or stop for human review, with the reasons that
drove that decision. Grouped `deep_mesh_windows` are built only from
`player_targets` that request `world_mesh`. When present, the BODY runner
consumes those windows and writes `body_compute_execution.json` before any Fast
SAM-3D-Body model invocation. The execution artifact is still a
schedule/coverage manifest only; it proves what should run, while
`pipeline_run.json`, `smpl_motion.json`, and `skeleton3d.json` prove whether the
H100 BODY runtime actually ran and produced mesh artifacts.

Current accepted-four BODY execution manifests are mixed. Canonical
`contact_windows.json` files now exist for all four clips from explicit human
review inputs, with `player_id` still untrusted/null. Burlington, Wolverine, and
Indoor doubles schedule limited deep-mesh BODY work; Outdoor webcam remains
fail-closed at zero because player coverage/uncertainty blocks world-mesh
targets. The current counts are:

| clip | scheduled BODY frames | skipped frames | skipped tiers |
|---|---:|---:|---|
| `burlington_gold_0300_low_steep_corner` | 3 / 9 player-frames | 597 | `human_review=589`, `skeleton_preview=8` |
| `wolverine_mixed_0200_mid_steep_corner` | 4 / 12 player-frames | 296 | `human_review=282`, `skeleton_preview=14` |
| `outdoor_webcam_iynbd_1500_long_high_baseline` | 0 | 1181 | `baseline=2`, `human_review=1166`, `skeleton_preview=13` |
| `indoor_doubles_fwuks_0500_long_mid_baseline` | 2 / 6 player-frames | 565 | `human_review=565` |

These schedules are review/audit evidence only. The latest H100 run did execute
Fast SAM-3D-Body on the scheduled Burlington, Wolverine, and Indoor frames and
wrote matching `smpl_motion.json`/`skeleton3d.json` artifacts, but the schedules
and smoke outputs still do not make BODY or BALL `VERIFIED`.

Each `body_compute_execution.json` summary now also includes
`skipped_by_target_representation` and `skipped_by_reason`, so the packet can
show whether frames stayed `track_only`, `joints_or_preview_mesh`, or
`manual_review_required`, and whether the main blocker was missing players,
missing ball frames, or uncertain ball frames.

Each `body_mesh_readiness.json` also carries the same representation boundary
into the mesh audit. Its `representation_decision` tells the review packet and
action checklist whether BODY output is missing because the frame plan actually
requested world meshes, or because the frame plan correctly stayed at
joints/preview/review-only. Current accepted-four values are Burlington,
Wolverine, and Indoor `world_mesh_required_available_unverified`, and Outdoor
`no_world_mesh_requested`.
The audit records requested/scheduled/available world-mesh counts separately so
smoke outputs cannot be confused with current BODY demand or BODY verification.

The packet also lists `contact_window_candidates.json` for each accepted clip.
Those files convert the uncertain prototype `labels/events.json` hints into
reviewable windows, but they keep `not_gate_verified=true` and
`trusted_for_body=false`. They are **not** read by adaptive BODY scheduling.
Deep-mesh BODY scheduling requires a root `contact_windows.json` written either
by reviewed human promotion or by BALL cue fusion. The current accepted-four
root contact windows were promoted from saved local review UI inputs at
`runs/review_inputs/pickleball_cv_review_latest.json`; they contain Burlington 9
contacts, Wolverine 6, Outdoor webcam 9, and Indoor doubles 10. Promotion emits
machine source scores at zero and `sources.human_review=1.0`; non-contact
candidate types (`bounce`, `net_cross`) are refused so they cannot accidentally
schedule BODY mesh work. Contact-review is no longer a user-gated action for
these clips unless the promoted files are regenerated or invalidated.

The BALL runner now also fails open only when all three non-human cue artifacts
are present in the clip input or run directory:

- `audio_onsets.json` with `onsets`
- `wrist_velocity_peaks.json` with `peaks`
- `ball_inflections.json` with `candidates`

When any cue family is missing, blocked, or empty, BALL writes an empty
schema-valid `contact_windows.json` and records the missing cue families in the
stage notes.
When all three cue families agree inside the temporal gate, BALL writes trusted
contact events with audio, wrist-velocity, and ball-inflection source scores;
the next `frame_compute_plan.json` can then create `world_mesh` player targets.
The same deterministic fusion logic is also available without running the full
BALL stage through `scripts/racketsport/build_contact_windows_from_cues.py`.
The current accepted-four root `contact_windows.json` files were not produced by
complete machine cue triplets; they were promoted from explicit human review
inputs. Treat them as prototype contact evidence that can schedule BODY review
work, not as BALL verification.

Each accepted clip now also has a review-only `ball_inflections.json` built from
the current `virtual_world.json` court-plane ball path. These artifacts reduce
one missing cue family for contact review, but they are explicitly
`not_gate_verified=true`: Burlington has 13 candidates, Wolverine 9, Outdoor
webcam 22, and Indoor doubles 12 after time suppression. Contact promotion still
requires audio-onset and wrist-velocity cue artifacts or explicit human
acceptance in `contact_window_review.json`.

Each accepted clip now also has packet-indexed audio/wrist cue artifacts.
Outdoor webcam has 4 review-only audio energy peaks and Indoor doubles has 1,
both generated from available 30-second `racket_candidates/source_0000_0030.mp4`
snippets clipped to the first 10 seconds. Burlington and Wolverine are blocked
by `no_audio_stream` in the local strict smoke videos. All four
`wrist_velocity_peaks.json` artifacts are blocked: Burlington has generic
SAM3D body joint names with no semantic wrist map, Wolverine and Indoor have
BODY skeletons but still lack a semantic wrist map, and Outdoor lacks
`skeleton3d.json`. These artifacts make the cue blockers explicit, but they do
not promote contacts.

When regenerating BALL through the fail-closed orchestrator, pass the strict
no-click ball track with `--ball-source` so root `ball_track.json` is rebuilt
from the same held-out no-click artifact and the post-run
`frame_compute_plan.json` includes ball uncertainty:

```bash
python -m threed.racketsport.orchestrator \
  --clip <clip> \
  --inputs runs/eval0/prototype_gate_h100_v2/<clip> \
  --out runs/eval0/prototype_gate_h100_v2/<clip> \
  --stage e2e \
  --tracking-mode precomputed \
  --ball-source runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json
```

For the fuller local input UI:

```bash
python scripts/racketsport/review_input_server.py --port 8765
```

Then open `http://127.0.0.1:8765`.

The corrections template starts empty so placeholder edits cannot be queued by
accident. After saving review inputs from that UI, export them into the
corrections queue format:

```bash
python scripts/racketsport/export_review_inputs_to_corrections.py \
  --review-input runs/review_inputs/pickleball_cv_review_latest.json \
  --out runs/review_packets/prototype_gate_h100_v2/corrections/prototype_gate_h100_v2_review_corrections.json \
  --manifest-id prototype_gate_h100_v2_review

python scripts/racketsport/validate_corrections.py \
  runs/review_packets/prototype_gate_h100_v2/corrections/prototype_gate_h100_v2_review_corrections.json

python scripts/racketsport/build_corrections_queue.py \
  --root runs/review_packets/prototype_gate_h100_v2/corrections \
  --out runs/corrections_queue/corrections_queue.json
```

If you used the UI's `Add contact at ball time` buttons, convert those explicit
contact marks into `contact_window_review.json` decisions before promoting
contacts. This updates only the review decision file and still fails closed if a
marked contact is not near an existing contact candidate:

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIP=burlington_gold_0300_low_steep_corner

python scripts/racketsport/apply_review_inputs_to_contact_review.py \
  --candidates "$RUN_ROOT/$CLIP/contact_window_candidates.json" \
  --review "$RUN_ROOT/$CLIP/contact_window_review.json" \
  --review-input runs/review_inputs/pickleball_cv_review_latest.json \
  --clip "$CLIP" \
  --out-review "$RUN_ROOT/$CLIP/contact_window_review.json"
```

To regenerate the packet after rebuilding artifacts:

```bash
python scripts/racketsport/build_review_packet.py \
  --run-root runs/eval0/prototype_gate_h100_v2 \
  --out-dir runs/review_packets/prototype_gate_h100_v2 \
  --packet-id prototype_gate_h100_v2_review \
  --corrections-root runs/review_packets/prototype_gate_h100_v2/corrections \
  --clip burlington_gold_0300_low_steep_corner \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --clip outdoor_webcam_iynbd_1500_long_high_baseline \
  --clip indoor_doubles_fwuks_0500_long_mid_baseline \
  --write-corrections-template
```

Then build the compact action checklist from that packet. It groups
court-evidence blockers, cue blockers, preview-only paddle pose blockers, BODY
schedule state, BODY mesh readiness blockers, E2E artifact-readiness blockers,
and virtual-world warnings into one browser page. When promoted
`contact_windows.json` files already exist, contact review should not be treated
as a user-gated next action; cue-related actions remain useful only for replacing
human-review contacts with a complete non-human fusion path later:

```bash
python scripts/racketsport/build_review_action_manifest.py \
  --packet runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review.json \
  --out-json runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review_actions.json \
  --out-html runs/review_packets/prototype_gate_h100_v2/prototype_gate_h100_v2_review_actions.html
```

## Rebuild The Current Review Tracks

Run from repo root. The same commands work locally for existing artifacts and on
the H100 under `/workspace/pickleball` after `git pull --ff-only`.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  base="$RUN_ROOT/$clip/tracknet_smoke_0000_0010"

  python scripts/racketsport/fuse_ball_tracks.py \
    --primary-ball-track "$base/ball_track_target_court_120px.json" \
    --stable-ball-track "$base/ball_track_target_court_temporal.json" \
    --verifier-ball-track "$base/vballnet_fast/ball_track.json" \
    --verifier-ball-track "$base/vballnet_v1/ball_track.json" \
    --outlier-distance-px 100 \
    --out "$base/ball_track_fusion_temporal_vball100.json" \
    --summary-out "$base/ball_track_fusion_temporal_vball100_summary.json"

  python scripts/racketsport/filter_ball_temporal.py \
    --mode local_trajectory \
    --ball-track "$base/ball_track_fusion_temporal_vball100.json" \
    --local-trajectory-window-frames 20 \
    --local-trajectory-max-error-px 80 \
    --local-trajectory-min-pair-predictions 4 \
    --max-iterations 3 \
    --out "$base/ball_track_fusion_temporal_vball100_localtraj.json" \
    --summary-out "$base/ball_track_fusion_temporal_vball100_localtraj_summary.json"

  python scripts/racketsport/render_ball_track_overlay.py \
    --video "$base/input_0000_0010.mp4" \
    --ball-track "$base/ball_track_fusion_temporal_vball100_localtraj.json" \
    --out "$base/ball_track_fusion_temporal_vball100_localtraj_overlay.mp4"

  ffmpeg -y \
    -i "$base/ball_track_fusion_temporal_vball100_localtraj_overlay.mp4" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "$base/ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4"
done
```

## Rebuild Preview Paddle Worlds

Run this after candidate artifacts change and you want paddle meshes in the
review world even though the canonical fail-closed racket stage still rejects
the box-derived candidates.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/build_racket_pose_preview.py \
    --court-calibration "$RUN_ROOT/$clip/court_calibration.json" \
    --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json" \
    --out "$RUN_ROOT/$clip/racket_pose_preview.json"

  readiness_args=(
    --clip "$clip"
    --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json"
    --racket-pose-preview "$RUN_ROOT/$clip/racket_pose_preview.json"
    --out "$RUN_ROOT/$clip/racket_pose_readiness.json"
  )

  if [[ -f "$RUN_ROOT/$clip/racket_pose.json" ]]; then
    readiness_args+=(--racket-pose "$RUN_ROOT/$clip/racket_pose.json")
  fi

  python scripts/racketsport/build_racket_pose_readiness.py "${readiness_args[@]}"

  audit_args=(
    --clip "$clip"
    --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json"
    --racket-pose-preview "$RUN_ROOT/$clip/racket_pose_preview.json"
    --out "$RUN_ROOT/$clip/racket_promotion_audit.json"
  )

  if [[ -f "$RUN_ROOT/$clip/racket_pose.json" ]]; then
    audit_args+=(--racket-pose "$RUN_ROOT/$clip/racket_pose.json")
  fi

  python scripts/racketsport/build_racket_promotion_audit.py "${audit_args[@]}"

  args=(
    --court-calibration "$RUN_ROOT/$clip/court_calibration.json"
    --tracks "$RUN_ROOT/$clip/tracks.json"
    --ball-track "$RUN_ROOT/$clip/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json"
    --racket-pose "$RUN_ROOT/$clip/racket_pose_preview.json"
    --out "$RUN_ROOT/$clip/virtual_world_paddle_preview.json"
  )

  if [[ -f "$RUN_ROOT/$clip/smpl_motion.json" ]]; then
    args+=(--smpl-motion "$RUN_ROOT/$clip/smpl_motion.json")
  fi

  if [[ -f "$RUN_ROOT/$clip/skeleton3d.json" ]]; then
    args+=(--skeleton3d "$RUN_ROOT/$clip/skeleton3d.json")
  fi

  python scripts/racketsport/build_virtual_world.py "${args[@]}"

  python scripts/racketsport/build_virtual_world_review.py \
    --virtual-world "$RUN_ROOT/$clip/virtual_world_paddle_preview.json" \
    --out-html "$RUN_ROOT/$clip/virtual_world_paddle_preview.html" \
    --index-out "$RUN_ROOT/$clip/virtual_world_review_index.json" \
    --title "$clip Paddle Preview World"
done
```

## Rebuild Paddle Runtime Readiness

Run this after editing `models/MANIFEST.json` or adding paddle CAD/reference
assets. It is CPU-only and writes a global report; use `--check-files` only in
an environment that can see the declared checkpoint paths.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2

python scripts/racketsport/build_racket_model_runtime_readiness.py \
  --manifest models/MANIFEST.json \
  --out "$RUN_ROOT/racket_model_runtime_readiness.json"
```

## Rebuild Ball-Inflection Cues

Run this after `virtual_world.json` changes. It is CPU-only and writes
review-only trajectory-turn cues; it does not create trusted contacts without
audio and wrist cues.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/build_ball_inflections.py \
    --virtual-world "$RUN_ROOT/$clip/virtual_world.json" \
    --out "$RUN_ROOT/$clip/ball_inflections.json"
done
```

## Rebuild Audio And Wrist Cues

Run this after replacing source media or BODY joint artifacts. Audio cues use an
audio-bearing 30-second source snippet when present, but analyze only the first
10 seconds to match the current strict ball/window artifacts. Wrist cues write a
blocked artifact when `skeleton3d.json` is absent so the review packet can show
the blocker explicitly.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  clip_dir="$RUN_ROOT/$clip"
  audio_input="$clip_dir/tracknet_smoke_0000_0010/input_0000_0010.mp4"
  if [ -f "$clip_dir/racket_candidates/source_0000_0030.mp4" ] &&
     ffprobe -v error -select_streams a:0 -show_entries stream=index -of csv=p=0 \
       "$clip_dir/racket_candidates/source_0000_0030.mp4" | grep -q .; then
    audio_input="$clip_dir/racket_candidates/source_0000_0030.mp4"
  fi

  python scripts/racketsport/build_audio_onsets.py \
    --input "$audio_input" \
    --out "$clip_dir/audio_onsets.json" \
    --clip "$clip" \
    --start-s 0 \
    --duration-s 10 \
    --analysis-sample-rate-hz 16000

  python scripts/racketsport/build_wrist_velocity_peaks.py \
    --skeleton3d "$clip_dir/skeleton3d.json" \
    --out "$clip_dir/wrist_velocity_peaks.json" \
    --allow-missing

  python scripts/racketsport/build_contact_windows_from_cues.py \
    --audio-onsets "$clip_dir/audio_onsets.json" \
    --wrist-velocity-peaks "$clip_dir/wrist_velocity_peaks.json" \
    --ball-inflections "$clip_dir/ball_inflections.json" \
    --tracks "$clip_dir/tracks.json" \
    --out "$clip_dir/contact_windows.json" || true
done
```

The cue-fusion command writes canonical `contact_windows.json` only when all
three cue artifacts parse and temporally agree. The `|| true` in the accepted-four
batch keeps the rebuild moving when a clip is still blocked by missing audio or
missing wrist samples; use the command output and action dashboard to see the
exact blocker. A successful empty output still does not promote BALL or BODY.

## Rebuild Contact-Window Candidates

Run this after `labels/events.json` changes. It is CPU-only, writes review
candidates only, and intentionally does **not** create trusted
`contact_windows.json`.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/build_contact_window_candidates.py \
    --events "$RUN_ROOT/$clip/labels/events.json" \
    --out "$RUN_ROOT/$clip/contact_window_candidates.json"
done
```

## Rebuild Contact-Window Review Templates

Run this after contact-window candidates change. The template is safe to commit
or hand-edit: pending/rejected decisions do not create scheduler inputs.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/promote_contact_windows.py \
    --candidates "$RUN_ROOT/$clip/contact_window_candidates.json" \
    --template-out "$RUN_ROOT/$clip/contact_window_review.json"

  python scripts/racketsport/render_contact_window_review.py \
    --candidates "$RUN_ROOT/$clip/contact_window_candidates.json" \
    --review "$RUN_ROOT/$clip/contact_window_review.json" \
    --out-html "$RUN_ROOT/$clip/contact_window_review.html"
done
```

Open `contact_window_review.html` from the packet to inspect the candidate row,
then edit the sibling `contact_window_review.json` or apply saved review-UI
contact marks into that file:

```bash
python scripts/racketsport/apply_review_inputs_to_contact_review.py \
  --candidates "$RUN_ROOT/$CLIP/contact_window_candidates.json" \
  --review "$RUN_ROOT/$CLIP/contact_window_review.json" \
  --review-input runs/review_inputs/pickleball_cv_review_latest.json \
  --clip "$CLIP" \
  --out-review "$RUN_ROOT/$CLIP/contact_window_review.json"
```

After reviewing the template, promote accepted decisions into the canonical root
`contact_windows.json`. This command fails closed unless accepted contacts
include `reviewer` and `reason`, and it refuses accepted `bounce`/`net_cross`
candidates. It also refuses to write an empty `contact_windows.json` when no
contacts are accepted unless `--allow-empty` is passed intentionally:

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIP=burlington_gold_0300_low_steep_corner

python scripts/racketsport/promote_contact_windows.py \
  --candidates "$RUN_ROOT/$CLIP/contact_window_candidates.json" \
  --review "$RUN_ROOT/$CLIP/contact_window_review.json" \
  --out-contact-windows "$RUN_ROOT/$CLIP/contact_windows.json"
```

## Rebuild BODY Execution Manifests

Run this after `tracks.json` or `frame_compute_plan.json` changes. It is
CPU-only and does not invoke Fast SAM-3D-Body.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/build_body_compute_execution.py \
    --tracks "$RUN_ROOT/$clip/tracks.json" \
    --frame-compute-plan "$RUN_ROOT/$clip/frame_compute_plan.json" \
    --out "$RUN_ROOT/$clip/body_compute_execution.json"
done
```

## Rebuild BODY Mesh Readiness Audits

Run this after `smpl_motion.json`, `skeleton3d.json`, `frame_compute_plan.json`,
or `body_compute_execution.json` changes. It is CPU-only, does not invoke Fast
SAM-3D-Body, refuses to treat joints-only previews as actual meshes, and reports
whether the frame plan actually requested world meshes.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/build_body_mesh_readiness.py \
    --clip "$clip" \
    --smpl-motion "$RUN_ROOT/$clip/smpl_motion.json" \
    --skeleton3d "$RUN_ROOT/$clip/skeleton3d.json" \
    --frame-compute-plan "$RUN_ROOT/$clip/frame_compute_plan.json" \
    --body-compute-execution "$RUN_ROOT/$clip/body_compute_execution.json" \
    --out "$RUN_ROOT/$clip/body_mesh_readiness.json"
done
```

## Rebuild E2E Readiness Reports

Run this after stage artifacts change. The report checks artifact presence plus
selected semantic blockers such as missing/empty `contact_windows.json`, zero
BODY scheduling where applicable, and BODY mesh readiness state; it does not
prove accuracy or promote any pipeline stage to `VERIFIED`.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/validate_pipeline_artifacts.py \
    --run-dir "$RUN_ROOT/$clip" \
    --stage e2e \
    --out "$RUN_ROOT/$clip/pipeline_readiness_e2e.json" || true
done
```

## Rebuild Virtual-World Artifacts

Run this after track, ball, body, or paddle artifacts change. The command is
safe to run locally for existing JSON artifacts; it does not invoke GPU models.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  args=(
    --court-calibration "$RUN_ROOT/$clip/court_calibration.json"
    --tracks "$RUN_ROOT/$clip/tracks.json"
    --ball-track "$RUN_ROOT/$clip/tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json"
    --out "$RUN_ROOT/$clip/virtual_world.json"
  )

  if [[ -f "$RUN_ROOT/$clip/smpl_motion.json" ]]; then
    args+=(--smpl-motion "$RUN_ROOT/$clip/smpl_motion.json")
  fi

  if [[ -f "$RUN_ROOT/$clip/skeleton3d.json" ]]; then
    args+=(--skeleton3d "$RUN_ROOT/$clip/skeleton3d.json")
  fi

  python scripts/racketsport/build_virtual_world.py "${args[@]}"
done
```

## Build Replay Review GLBs

Run this after `virtual_world_paddle_preview.json` changes and you want
`replay_scene.json` plus loadable review GLBs. This is CPU-only and writes static
review assets; it does not run the production replay renderer or export USDZ.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  python scripts/racketsport/build_replay_review_export.py \
    --virtual-world "$RUN_ROOT/$clip/virtual_world_paddle_preview.json" \
    --out-dir "$RUN_ROOT/$clip/replay_review" \
    --scene-out "$RUN_ROOT/$clip/replay_scene.json"
done
```

Current review GLB point sizes are Burlington 0.904516 MB, Wolverine 0.009264
MB, Outdoor webcam 0.051876 MB, and Indoor doubles 0.031712 MB. After generating
these, rerun `validate_pipeline_artifacts.py` and rebuild the review packet so
`replay_scene.json` is no longer reported as a missing artifact. The replay and
e2e evaluators additionally report `glb_files_valid` /
`referenced_glb_files_valid` after parsing the GLB header, chunks, and glTF JSON
asset metadata.

## Optional Explicit Paddle Candidates

The registered racket stage does not run a detector yet. It only consumes
schema-validated explicit four-corner paddle candidates and fails closed if the
artifact contract is malformed or if no frame passes PnP-IPPE reprojection and
ambiguity checks. Candidate corner order is the same as
`paddle_face_corners_object_cm`: top-left, top-right, bottom-right, bottom-left
when looking at the paddle face.

```json
{
  "schema_version": 1,
  "artifact_type": "racketsport_racket_candidates",
  "fps": 60.0,
  "players": [
    {
      "id": 7,
      "paddle_dims_in": {"length": 16.0, "width": 8.0},
      "frames": [
        {
          "t": 0.0,
          "corners_px": [[100.0, 100.0], [140.0, 102.0], [136.0, 180.0], [96.0, 178.0]],
          "conf": 0.9,
          "source": "manual_or_detector_candidate"
        }
      ]
    }
  ]
}
```

Place that file at `runs/eval0/prototype_gate_h100_v2/<clip>/racket_candidates.json`
or pass a custom runner path in code. A successful stage writes
`racket_pose.json`; a missing artifact, extra/unregistered fields, bad corner
shape, invalid confidence, ambiguous solve, or high-reprojection candidate does
not produce a pose artifact.

The current prototype setup can generate candidate-only paddle inputs from the
draft YOLO26m racket-box label artifacts. These boxes are not true paddle
corners, so the default `RacketStageRunner` now rejects any `label_bbox:*`
source before PnP and does not write `racket_pose.json`; it does write
`racket_stage_diagnostics.json` with box-source rejection counts. They are
useful because the review packet now exposes candidate counts, rejection counts,
and the exact missing-pose state for each clip. The separate preview CLI can
still solve and write ambiguous frames to `racket_pose_preview.json` so
`virtual_world_paddle_preview.json` can show where the paddle candidate would
sit in the court world for review.

For readiness accounting, sources beginning with `label_bbox:` count as
`box_derived` and cannot promote. Sources containing true detector keypoints or
mask-derived face corners count as `keypoint_or_mask`; sources containing
`synthetic`, `blenderproc`, or `cad` count as `synthetic_or_cad`; and sources
containing `aruco`, `april`, `tag`, `gt`, `ground_truth`, or `reference` count
as `reference_gt`. RKT still requires evaluation, so a promoted pose without
reference/GT remains `pose_present_needs_reference_and_eval`, and a promoted
pose with reference/GT remains `pose_present_needs_eval` until
`missing_racket_pose_evaluation` is cleared by reviewed reference labels and
passing Phase 6 `label_check.*` metrics. A syntactically valid `racket_pose.json`
does not make RKT pass by itself.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIPS=(
  burlington_gold_0300_low_steep_corner
  wolverine_mixed_0200_mid_steep_corner
  outdoor_webcam_iynbd_1500_long_high_baseline
  indoor_doubles_fwuks_0500_long_mid_baseline
)

for clip in "${CLIPS[@]}"; do
  video="$RUN_ROOT/$clip/tracknet_smoke_0000_0010/input_0000_0010.mp4"
  if [[ -f "$RUN_ROOT/$clip/racket_candidates/source_0000_0030.mp4" ]]; then
    video="$RUN_ROOT/$clip/racket_candidates/source_0000_0030.mp4"
  fi

  python scripts/racketsport/build_racket_candidates.py \
    --racket-labels "$RUN_ROOT/$clip/labels/racket_pose.json" \
    --manifest "$RUN_ROOT/$clip/labels/prototype_autolabel_manifest.json" \
    --min-confidence 0.25 \
    --out "$RUN_ROOT/$clip/racket_candidates.json"

  python scripts/racketsport/render_racket_candidate_overlay.py \
    --video "$video" \
    --racket-candidates "$RUN_ROOT/$clip/racket_candidates.json" \
    --out "$RUN_ROOT/$clip/racket_candidates/racket_candidate_overlay.mp4" \
    --h264-out "$RUN_ROOT/$clip/racket_candidates/racket_candidate_overlay_h264.mp4" \
    --candidate-coordinate-width 960 \
    --candidate-coordinate-height 540
done
```

At the current artifact state this produced candidate-only review inputs for all
accepted clips: Burlington 42 candidate frames, Wolverine 36, Outdoor webcam
397, and Indoor doubles 280. Running the default racket stage rejected all of
them before PnP because every source is `label_bbox:yolo26m_teacher`, which is
recorded under `racketsport_racket_stage_diagnostics` in the refreshed packet.
The preview-only paddle world path includes all current candidate frames but
marks all of them `ambiguous_paddle_pose`. The current overlay, preview, and
readiness coverage is:

| clip | candidate frames | rendered candidates | preview paddle frames | true/reference frames | reference/GT frames | court-plane ball points | preview warning |
|---|---:|---:|---:|---:|---:|---:|---|
| `burlington_gold_0300_low_steep_corner` | 42 | 42 | 42 | 0 | 0 | 288 | `ambiguous_paddle_pose` |
| `wolverine_mixed_0200_mid_steep_corner` | 36 | 36 | 36 | 0 | 0 | 119 | `missing_mesh_vertices`, `ambiguous_paddle_pose` |
| `outdoor_webcam_iynbd_1500_long_high_baseline` | 397 | 397 | 397 | 0 | 0 | 489 | `missing_mesh_vertices`, `ambiguous_paddle_pose` |
| `indoor_doubles_fwuks_0500_long_mid_baseline` | 280 | 280 | 280 | 0 | 0 | 117 | `missing_mesh_vertices`, `ambiguous_paddle_pose` |

## Benchmark Person Trackers With Adaptive BODY Audit

Use this when comparing person-tracking variants and you want the output to
show what adaptive frame rating would schedule from each candidate's own
`tracks.json`. This is CPU-side scheduling/review metadata only; it does not run
Fast SAM-3D-Body or make TRK/BODY verified.

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2
CLIP=burlington_gold_0300_low_steep_corner

python scripts/racketsport/benchmark_person_trackers.py \
  --clip "$CLIP" \
  --video "$RUN_ROOT/$CLIP/tracknet_smoke_0000_0010/input_0000_0010.mp4" \
  --calibration "$RUN_ROOT/$CLIP/court_calibration.json" \
  --ball-track "$RUN_ROOT/$CLIP/ball_track.json" \
  --contact-windows "$RUN_ROOT/$CLIP/contact_windows.json" \
  --expected-players 4 \
  --out-root runs/person_tracking_frame_rating_audit \
  --candidate "yolo26m_botsort_reid=models/yolo26m.pt,configs/racketsport/botsort_reid.yaml" \
  --device 0 \
  --max-frames 300 \
  --id-strategy role_lock
```

Each candidate directory now includes `tracks.json`, `frame_compute_plan.json`,
and `body_compute_execution.json`. The generated `REPORT.md` includes a
`BODY frames` column formatted as scheduled BODY frames / scheduled BODY
player-frames, so reviewers can compare tracker variants by both visible player
coverage and downstream mesh scheduling impact. With the current accepted-four
root artifacts, tracker variants should be evaluated against the promoted
human-review contact windows, while keeping the output as scheduling/review
metadata rather than TRK or BODY verification.

## Rerun The Held-Out Benchmark

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2

python scripts/racketsport/benchmark_ball_trackers.py \
  --run-root "$RUN_ROOT" \
  --review-root "$RUN_ROOT/ball_click_review_30" \
  --clip burlington_gold_0300_low_steep_corner \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --clip outdoor_webcam_iynbd_1500_long_high_baseline \
  --clip indoor_doubles_fwuks_0500_long_mid_baseline \
  --candidate "tracknet_raw=tracknet_smoke_0000_0010/ball_track_0000_0010.json" \
  --candidate "tracknet_court_temporal_path=tracknet_smoke_0000_0010/ball_track_target_court_temporal.json" \
  --candidate "fusion_temporal_vball100:generalizable_two_model_fusion=tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100.json" \
  --candidate "fusion_temporal_vball100_localtraj:generalizable_two_model_fusion=tracknet_smoke_0000_0010/ball_track_fusion_temporal_vball100_localtraj.json" \
  --out-json "$RUN_ROOT/ball_tracker_benchmark/benchmark_summary_fusion_localtraj.json" \
  --out-md "$RUN_ROOT/ball_tracker_benchmark/benchmark_summary_fusion_localtraj.md"
```

## Sync And Verify On H100

```bash
gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent bash -lc 'cd /workspace/pickleball && git pull --ff-only'"

gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent bash -lc 'cd /workspace/pickleball && /opt/conda/envs/fast_sam_3d_body/bin/python -m pytest -q -p no:cacheprovider tests/racketsport/test_ball_temporal_filter.py tests/racketsport/test_ball_benchmark.py tests/racketsport/test_ball_model_fusion.py'"

gcloud compute ssh body4d-gcp-prod --zone us-west1-b --command \
  "docker exec sam4dbody-pod-agent nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
```

Expected focused test result at the current commit: `9 passed`.

## If The Output Looks Wrong

- Background balls are selected: check the court calibration overlay first, then
  rerun the target-court filter before fusion.
- The ball jumps to a far-away false detection: use the local-trajectory track
  above, or lower `--local-trajectory-max-error-px` for a stricter pass.
- The real ball disappears too often: compare against
  `ball_track_fusion_temporal_vball100.json`; if that is better, raise
  `--local-trajectory-max-error-px` or keep both artifacts for review.
- Court lines or paddle flashes are selected as the ball: this needs better
  model-side candidate generation, not just post-processing.

## Explicit Next Steps

1. Promote the prototype `BallStageRunner` into a reachable real BALL path:
   run the best no-click track path through the orchestrator, generate non-empty
   contact windows, and write BALL contract artifacts fail-closed.
2. Improve candidate generation, not just filtering: obtain usable TrackNetV4
   weights or fine-tune TrackNetV3/V4/V5 on pickleball clips with neighboring
   courts, small balls, occlusions, and high-baseline viewpoints.
3. Extend from 10-second smoke windows to longer windows, then full clips, using
   the same benchmark command and keeping human clicks held out.
4. Add event/contact outputs only after the ball path is stable enough: audio
   pop, wrist-velocity peak, ball inflection, and doubles attribution.
5. Replace the prototype benchmark with the full BALL acceptance gate: ball F1
   at least 0.90, false positives below 5%, and contact timing within plus/minus
   2 frames on a representative labeled set.
6. Keep `BALL-1`, `BALL-3`, and `BALL-4` unverified until the real gate passes
   through the spine on real clips.
