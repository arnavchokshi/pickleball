# Corrections Manifest

This directory contains the CPU-only linting contract for future manual corrections before they enter the corrections flywheel.

Validate a manifest with:

```bash
python scripts/racketsport/validate_corrections.py path/to/corrections.json
```

The validator is intentionally strict:

- `schema_version` must be `1`.
- Top-level and correction objects reject unknown fields.
- Targets use JSON Pointer paths such as `/players/0/frames/42/conf`.
- `set`, `replace`, and `append` operations require `value`.
- `delete` operations must not include `value`.
- Correction IDs must be unique within a manifest.
- Artifact paths must be relative and must not traverse outside the workspace.

Minimal manifest:

```json
{
  "schema_version": 1,
  "manifest_id": "eval3_manual_seed",
  "created_at": "2026-06-26T21:00:00Z",
  "corrections": [
    {
      "id": "corr_001",
      "target": {
        "artifact": "runs/eval3/clip_001/racket_pose.json",
        "clip_id": "clip_001",
        "frame_index": 42,
        "path": "/players/0/frames/42/conf"
      },
      "operation": "replace",
      "value": 0.91,
      "reason": "Manual review found a missed high-confidence paddle frame.",
      "annotator": "agent-k",
      "created_at": "2026-06-26T21:05:00Z"
    }
  ]
}
```
