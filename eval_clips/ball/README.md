# BALL local eval clips

This directory is a small committed fixture bundle for local BALL pipeline
smoke/eval runs. It is intentionally narrower than the full `data/testclips`
matrix and is not a promotion gate by itself.

Each clip has:

- `source.mp4`
- `labels/ball_points.json`
- `labels/events.json`
- `labels/foot_contact.json`
- `labels/court_corners.json`
- `clip_metadata.json`

`manifest.json` is the authoritative bundle index. It records the actual
OpenCV-readable frame count, resolution, duration, and SHA-256 for each copied
source video, plus label file SHA-256 values.
