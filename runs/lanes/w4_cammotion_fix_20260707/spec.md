# LANE w4_cammotion_fix_20260707 — fix the probe decode-orientation context (ruled diagnosis)

## OBJECTIVE
Implement the RULED fix from `runs/lanes/w4_cammotion_diag_20260707/REPORT.md`: the camera-motion
AUTO probe scored img1605 at 0.329 in-pipeline vs 53.7 offline because of an OpenCV
decode-orientation context mismatch (raw-landscape decode collapses features 217→14 on the
portrait iPhone clip). Make the probe's decode orientation EXPLICIT and deterministic, add the
missing decode telemetry, and prove: img1605 probe scores in the AUTO-ON regime (~53.7 > 2.5)
THROUGH the production probe entry path, while the three static clips' probe inputs and decisions
are bit-identical to today. GATE-ADJACENT: an independent adversarial verifier follows.

## EVIDENCE TO READ FIRST
- `runs/lanes/w4_cammotion_diag_20260707/REPORT.md` + `repro/cammotion_probe_context_repro.py` —
  the ruled root cause, the fix design (explicit CAP_PROP_ORIENTATION_AUTO policy + telemetry),
  expected scores, and the `--assert-fixed` defect proof (UNMODIFIABLE acceptance instrument).
- `threed/racketsport/camera_motion.py` — `estimate_camera_motion_probe`,
  `CAMERA_MOTION_AUTO_THRESHOLD = 2.5` (re-grep at HEAD), the VideoCapture/decode context.
- `runs/lanes/w3_cammotion_conditional_20260707/` — the landed AUTO wiring + measured baselines
  (wolverine 0.13 / burlington 0.52 / outdoor 0.57 AUTO OFF; static path bit-exact guarantee).

## DESIGN (pinned by the diagnosis — do not re-design)
1. In `camera_motion.py`'s probe/motion decode path: set the orientation policy EXPLICITLY on
   every VideoCapture it opens (per the diagnosis: orientation-applied is the correct context —
   the one the module's own offline measurements and the 3/4-correct decisions were made in).
   The policy must be deterministic across OpenCV builds: set the property AND record what the
   capture actually reports back (do not assume the set succeeded).
2. Telemetry: the probe result now records decoded frame dimensions, orientation-meta/rotation
   applied, feature count, and sampled frame indices (the diagnosis found current telemetry too
   sparse to catch this class). These are additive fields — do not break existing consumers
   (grep consumers of the probe result/summary fields; keep old keys byte-compatible).
3. If (and only if) the probe entry is invoked with pre-decoded frames from the pipeline (rather
   than opening the video itself), apply the equivalent explicit normalization at that seam INSIDE
   camera_motion.py; if the true seam lives in fenced `process_video.py`, deliver that part as a
   proposed diff + STOP that sub-item.

## ACCEPTANCE (exact; through the PRODUCTION entry path, not a replica)
1. `.venv/bin/python runs/lanes/w4_cammotion_diag_20260707/repro/cammotion_probe_context_repro.py`
   post-fix contexts agree; the diagnosis's `--assert-fixed` proof passes when fed a freshly
   computed probe summary from the fixed code, UNMODIFIED (if the proof needs a fresh pipeline
   summary file, generate one by invoking the PRODUCTION probe function on the local img1605 eval
   clip with production arguments and state exactly what you invoked).
2. img1605 (locate its eval clip via the w3 lane report): production-path probe score > 2.5
   (expected ≈53.7 — report the exact value) → AUTO ON decision.
3. Static clips (wolverine/burlington/outdoor, local eval clips): decoded probe INPUT provably
   unchanged (assert decoded dims + first-sampled-frame hash pre/post fix) AND probe scores equal
   to their banked baselines (0.13/0.52/0.57) → AUTO OFF unchanged. Bit-exactness of the static
   path is a FROZEN guarantee — if your fix changes any static clip's decode, that is a FAIL, not
   a tolerable delta.
4. Unit tests: orientation policy explicitly set + verified on a synthetic rotated fixture;
   telemetry fields present; threshold 2.5 untouched (grep-assert you did not modify it).
5. Full blast radius: `.venv/bin/python -m pytest tests/racketsport/test_camera_motion.py -q` plus
   every test file your grep shows importing `camera_motion` (list + run ALL).

## OWNED FILES (anti-collision fence)
`threed/racketsport/camera_motion.py`, `tests/racketsport/test_camera_motion.py`, your lane dir.
DO NOT TOUCH: `process_video.py`/`orchestrator.py` (fenced — proposed-diff only),
placement/grounding/phase files (another live lane owns them), any `ball_*` file,
`remote_body_dispatch.py`, `virtual_world.py`, `ios/**`, `runs/manager/**`, eval labels, ledger.

## KILL
If the orientation-applied context does NOT reproduce ≈53.7 on img1605 through the production
path after your fix (i.e. the diagnosis's root cause is incomplete), STOP: needs-validation with
your measurements — do not chase secondary causes past the ruled design.

## DISCIPLINE
`.venv/bin/python`; no git branch/commit/push; no network; no new root-level .md; pre-existing
failures proven at HEAD; the decisive in-pipeline GPU proof is the manager's wave-close job.

## STRUCTURED REPORT
Acceptance table (probe scores all 4 clips pre/post, static-input hashes, defect-proof result);
CHANGES file:line; full_suite; HONEST ISSUES; NEXT (what the adversarial verifier should attack).
