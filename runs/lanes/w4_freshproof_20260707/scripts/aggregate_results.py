#!/usr/bin/env python3
"""Aggregate w4_freshproof per-clip checklists (extract_w4_checklist.extract) into checklists/*.json."""
import importlib.util
import json
import sys
from pathlib import Path

LANE = Path("/Users/arnavchokshi/Desktop/pickleball/runs/lanes/w4_freshproof_20260707")
spec = importlib.util.spec_from_file_location("xw4", LANE / "scripts/extract_w4_checklist.py")
xw4 = importlib.util.module_from_spec(spec)
sys.modules["xw4"] = xw4
spec.loader.exec_module(xw4)

CLIPS = {
    "outdoor": "outdoor_webcam_iynbd_1500_long_high_baseline",
    "burlington": "burlington_gold_0300_low_steep_corner",
    "wolverine": "wolverine_mixed_0200_mid_steep_corner",
    "img1605": "owner_IMG_1605_8a193402780b",
}

(LANE / "checklists").mkdir(exist_ok=True)
combined = {}
for short, clip in CLIPS.items():
    hits = sorted((LANE / short).rglob("PIPELINE_SUMMARY.json")) if (LANE / short).exists() else []
    if not hits:
        combined[short] = {"error": "no PIPELINE_SUMMARY.json found"}
        continue
    clip_dir = max(hits, key=lambda h: len(h.parts)).parent  # deepest: the real nested clip dir
    result = xw4.extract(clip_dir)
    (LANE / "checklists" / f"{short}.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    combined[short] = result
(LANE / "checklists" / "combined.json").write_text(json.dumps(combined, indent=2, sort_keys=True, default=str))
print(json.dumps({s: {"status": c.get("run", {}).get("pipeline_status"), "slide_max_m": c.get("slide_gate", {}).get("grounding_metrics.max_foot_lock_slide_m"), "cam_enabled": (c.get("camera_motion_auto") or {}).get("enabled"), "cam_score": (c.get("camera_motion_auto") or {}).get("score")} for s, c in combined.items()}, indent=2))
