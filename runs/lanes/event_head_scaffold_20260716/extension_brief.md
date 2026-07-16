# EXTENSION BRIEF — event_head_scaffold_20260716 phase 2 (owner-directed acceleration, 2026-07-16 morning)

Your closure report landed and the manager ruled CPU smoke GREEN. The owner directive now requires
two bounded extensions so the GPU pretrain can run TODAY and its output can feed Track A's live
ball-arc anchor-fusion work. Same HARD RULES, protections, and file ownership as spec.md, plus:
you may now also create scripts/racketsport/build_event_head_anchor_candidates.py,
tests/racketsport/test_event_head_anchor_candidates.py, and register the new CLI in
list_scaffold_tools.py. Everything else unchanged. NO commits.

## Extension 1 — full pretrain mode on train_event_head.py (keep --smoke byte-compatible)
- New mutually-exclusive flag `--full` requiring: --manifest <builder dataset manifest JSON>,
  --device {cpu,cuda,mps}, --out, plus --steps, --image-size, --window-frames, --batch-size,
  --lr, --val-every, --seed. Trains on the manifest's train-split media-present samples
  (loss-masked union exactly as the loaders already do), evaluates on the val split every
  --val-every steps with the existing matcher (±2 frames), keeps best-by-val-F1 checkpoint +
  last checkpoint, writes train_manifest.json (provenance: git head, data manifest sha, seed,
  config, license_posture RD_ONLY, verified:false, steps/s measured).
- MUST print a steps/s line after the first 100 steps (the GPU manager sets wall caps from it)
  and support --max-wall-minutes N (hard stop with honest partial manifest, exit 0, state saved).
- Deterministic seeding; resume-from-checkpoint flag (--init-checkpoint) for preemption recovery.
- CPU proof: a bounded tiny --full run (small steps, tiny subset via --limit-clips flag) EXIT 0
  + test asserting the manifest fields exist and --smoke behavior unchanged (same defaults).

## Extension 2 — anchor-candidate inference CLI (Track A handoff artifact)
`scripts/racketsport/build_event_head_anchor_candidates.py`:
- Args: --checkpoint, --video <path>, --out <json>, --threshold, --nms-radius-frames,
  --device {cpu,cuda,mps}, --stride (window hop), optional --max-seconds (bounded smoke).
- Slides the model over the FULL video on-the-fly (no frame dumps), peak-picks typed events,
  emits EXACTLY this schema (coordinate contract with Track A — do not rename fields):
  {
    "artifact_type": "event_head_contact_anchor_candidates",
    "schema_version": 1,
    "source_video": {"path": str, "sha256": str},
    "video_provenance": str,          # caller-supplied via --video-provenance, default "unspecified"
    "never_training": true, "review_only": true, "verified": false,
    "model": {"checkpoint_path": str, "checkpoint_sha256": str, "license_posture": str,
              "pretrain_data": str},
    "config": {"threshold": float, "nms_radius_frames": int, "stride": int,
               "image_size": int, "window_frames": int, "fps": float,
               "pts_convention": "normalized_to_first_video_pts"},
    "events": [{"frame_idx": int, "pts_s": float, "class": "HIT"|"BOUNCE", "score": float}],
    "counts": {"HIT": int, "BOUNCE": int},
    "honest_limits": [str, ...]
  }
- PTS convention must match the repo's existing decode convention (same one the protected-seed
  eval uses). Document any deviation loudly.
- CPU proof: run on tests/racketsport/fixtures/event_head/tiny.avi (or a generated tiny video)
  EXIT 0, schema-valid output asserted by its direct-CLI reference test; scaffold index EXIT 0
  after registration.

## Report
Update/extend your structured report: per-extension acceptance w/ real exit codes, honest issues,
and re-run ONLY the affected test files + the scaffold index (the earlier full wide suite result
stands; do not rerun the whole suite unless you touched shared code).
