#!/usr/bin/env python3
"""Salvage 2D-coverage slice: our WASB 2D ball track vs the pb.vision cv export.

Written 2026-07-15 by the Track A manager session after the full-stack MOVE-1
re-run stalled in ball_arc (segment-association scaling defect) and was torn
down. The frozen compare_vs_pbvision.py harness crashes on the full 11-minute
export inside its PB physics pillar (scipy least_squares x0-out-of-bounds), so
this standalone script computes ONLY the 2D coverage slice with explicit
definitions. No 3D claims. Competitor-reference diagnostic; pb.vision data is
R&D reference ONLY (never GT, never training data). VERIFIED=0 binding.

Definitions (matching runs/research_pbv11_20260713 forensics conventions):
- PB rally window r = global frames [frame_index, frame_index + len(frames)).
- in-rally frame set = union of all rally windows (42 cv rallies).
- PB 2D coverage = fraction of in-rally frames whose PB frame entry has a
  non-empty `balls` list.
- our 2D coverage = fraction of in-rally frames where our ball_track.json
  frame (same global index + offset) has visible == True.
- offset: ours = PB_global + offset; both tracks index the same 697.4s
  1280x720@30 source (sha 272a2132...); offsets {-3, 0, +3} reported to show
  window-scale insensitivity.

Sanity anchors from the frozen 07-13 forensics: PB in-rally coverage ~58.7%,
ours ~80.6% (computed there from the same export + our WASB outputs).
"""
import json
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[3]

PB_EXPORT = ROOT / "data/pbvision_11min_20260713/cv_export.json"
OURS = LANE / "vm_pull_partial/pbvision_11min_20260713/ball_track.json"


def main() -> None:
    pb = json.loads(PB_EXPORT.read_text())
    ours = json.loads(OURS.read_text())
    rallies = pb["sessions"][0]["rallies"]
    our_frames = ours["frames"]
    n_ours = len(our_frames)

    def our_visible(idx: int) -> bool:
        return 0 <= idx < n_ours and bool(our_frames[idx].get("visible"))

    per_rally = []
    for ri, rally in enumerate(rallies):
        start = int(rally["frame_index"])
        frames = rally["frames"]
        n = len(frames)
        pb_cov = sum(1 for fr in frames if fr.get("balls")) / n if n else 0.0
        row = {
            "cv_rally": ri,
            "start_frame": start,
            "n_frames": n,
            "pb_2d_coverage": round(pb_cov, 4),
        }
        for off in (-3, 0, 3):
            cov = sum(1 for k in range(n) if our_visible(start + k + off)) / n if n else 0.0
            row[f"ours_2d_coverage_off{off:+d}"] = round(cov, 4)
        per_rally.append(row)

    total_frames = sum(r["n_frames"] for r in per_rally)
    def total(key):
        return round(
            sum(r[key] * r["n_frames"] for r in per_rally) / total_frames, 4
        )

    summary = {
        "artifact_type": "pbv11_salvage_2d_coverage_scorecard",
        "schema_version": 1,
        "date": "2026-07-15",
        "status": "salvage_2d_only_no_3d_claims",
        "pb_export_sha_source_video": "272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383",
        "ours_source": str(OURS.relative_to(ROOT)),
        "n_cv_rallies": len(per_rally),
        "total_in_rally_frames": total_frames,
        "pb_2d_coverage_total": total("pb_2d_coverage"),
        "ours_2d_coverage_total_off0": total("ours_2d_coverage_off+0"),
        "ours_2d_coverage_total_off-3": total("ours_2d_coverage_off-3"),
        "ours_2d_coverage_total_off+3": total("ours_2d_coverage_off+3"),
        "forensics_anchors": {"pb": 0.587, "ours": 0.806},
        "per_rally": per_rally,
    }
    out = LANE / "scorecard_2d_salvage.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_rally"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
