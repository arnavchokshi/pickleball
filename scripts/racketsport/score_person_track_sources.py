#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration import calibration_image_size  # noqa: E402
from threed.racketsport.person_track_gt_scoring import (  # noqa: E402
    build_scoring_report,
    derive_track_source_id,
    render_scoring_report_markdown,
    score_tracks_against_person_ground_truth,
)
from threed.racketsport.schemas import CourtCalibration, PersonGroundTruth, Tracks, validate_artifact_file  # noqa: E402


JSON_REPORT = "person_track_gt_scoring_report.json"
MARKDOWN_REPORT = "PERSON_TRACK_GT_SCORING_REPORT.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score existing tracks.json artifacts against CVAT person GT.")
    parser.add_argument("--cvat-root", type=Path, default=Path("runs/cvat_imports/2026_06_30"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase2/person_track_gt_scoring_20260630"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--expected-players", type=int, default=None)
    args = parser.parse_args()

    try:
        ground_truth = _load_ground_truth(args.cvat_root)
        clip_resolutions = _load_clip_resolutions(args.cvat_root)
        clip_ids = sorted(ground_truth)
        track_paths = _discover_track_paths(args.runs_root, clip_ids=clip_ids, out_dir=args.out_dir)
        rows = []
        errors = []
        for path in track_paths:
            clip_id = _clip_id_for_path(path, clip_ids=clip_ids)
            if clip_id is None:
                continue
            try:
                parsed = validate_artifact_file("tracks", path)
                if not isinstance(parsed, Tracks):
                    raise ValueError("artifact did not parse as Tracks")
                source_id = derive_track_source_id(path, clip_ids=clip_ids)
                bbox_scale = _read_bbox_scale(path, clip_id=clip_id, clip_resolutions=clip_resolutions)
                # Gate v2's outside_image_false_positives bucket needs the image bounds in
                # the *same* pixel space `bbox_scale_x`/`bbox_scale_y` scales prediction boxes
                # into -- i.e. the clip's native manifest resolution, the same space ground
                # truth boxes are in (see review finding F4, 2026-07-02). Without this, a
                # prediction box centered outside the frame gets misclassified as
                # true_spectator_or_background instead of outside_image, sending debugging
                # toward the wrong failure mode.
                resolution = clip_resolutions.get(clip_id)
                image_width, image_height = resolution if resolution is not None else (None, None)
                row = score_tracks_against_person_ground_truth(
                    ground_truth=ground_truth[clip_id],
                    tracks=parsed,
                    candidate=_candidate_for_path(path, clip_ids=clip_ids),
                    tracks_path=path,
                    iou_threshold=args.iou_threshold,
                    expected_players=args.expected_players,
                    bbox_scale_x=bbox_scale["bbox_scale_x"],
                    bbox_scale_y=bbox_scale["bbox_scale_y"],
                    image_width=image_width,
                    image_height=image_height,
                )
                row["track_source_id"] = source_id
                row["bbox_scale_source"] = bbox_scale["source"]
                row["timing"] = _read_timing(path)
                rows.append(row)
            except Exception as exc:  # keep one bad artifact from hiding all usable scores
                errors.append({"tracks_path": str(path), "error": str(exc)})

        report = build_scoring_report(rows, required_clip_ids=clip_ids, iou_threshold=args.iou_threshold)
        report["score_errors"] = errors
        report["score_error_count"] = len(errors)
        report["input"] = {
            "cvat_root": str(args.cvat_root),
            "runs_root": str(args.runs_root),
            "track_paths_discovered": len(track_paths),
            "expected_players": args.expected_players,
        }
        _write_reports(report, args.out_dir)
    except Exception as exc:
        print(f"track scoring failed: {exc}", file=sys.stderr)
        return 1

    print(args.out_dir / JSON_REPORT)
    print(args.out_dir / MARKDOWN_REPORT)
    return 0


def _load_ground_truth(cvat_root: Path) -> dict[str, PersonGroundTruth]:
    ground_truth: dict[str, PersonGroundTruth] = {}
    for path in sorted(cvat_root.glob("*/person_ground_truth.json")):
        parsed = validate_artifact_file("person_ground_truth", path)
        if not isinstance(parsed, PersonGroundTruth):
            raise ValueError(f"{path} did not parse as PersonGroundTruth")
        ground_truth[parsed.clip_id] = parsed
    if not ground_truth:
        raise ValueError(f"no person_ground_truth.json files found under {cvat_root}")
    return ground_truth


def _load_clip_resolutions(cvat_root: Path) -> dict[str, tuple[int, int]]:
    manifest_path = cvat_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    resolutions = {}
    for clip in payload.get("clips", []):
        if not isinstance(clip, dict):
            continue
        resolution = clip.get("resolution")
        clip_id = clip.get("clip_id")
        if (
            isinstance(clip_id, str)
            and isinstance(resolution, list)
            and len(resolution) == 2
            and all(isinstance(value, int) and value > 0 for value in resolution)
        ):
            resolutions[clip_id] = (int(resolution[0]), int(resolution[1]))
    return resolutions


def _discover_track_paths(runs_root: Path, *, clip_ids: list[str], out_dir: Path) -> list[Path]:
    paths = []
    resolved_out = out_dir.resolve()
    for path in sorted(runs_root.rglob("tracks.json")):
        try:
            if path.resolve().is_relative_to(resolved_out):
                continue
        except OSError:
            pass
        if _clip_id_for_path(path, clip_ids=clip_ids) is not None:
            paths.append(path)
    return paths


def _clip_id_for_path(path: Path, *, clip_ids: list[str]) -> str | None:
    path_text = path.as_posix()
    for clip_id in clip_ids:
        if clip_id in path.parts or clip_id in path_text:
            return clip_id
    return None


def _candidate_for_path(path: Path, *, clip_ids: list[str]) -> str:
    parent = path.parent.name
    if parent in clip_ids:
        return "canonical_tracks"
    return parent


def _read_bbox_scale(
    tracks_path: Path,
    *,
    clip_id: str,
    clip_resolutions: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    metrics = _read_metrics_payload(tracks_path)
    counts = metrics.get("counts") if isinstance(metrics.get("counts"), dict) else {}
    source_width = _number(counts.get("source_width"))
    source_height = _number(counts.get("source_height"))
    calibration_width = _number(counts.get("calibration_width"))
    calibration_height = _number(counts.get("calibration_height"))
    if source_width and source_height and calibration_width and calibration_height:
        return {
            "bbox_scale_x": source_width / calibration_width,
            "bbox_scale_y": source_height / calibration_height,
            "source": "metrics_source_over_calibration",
        }

    bbox_scale_x = _number(counts.get("bbox_scale_x"))
    bbox_scale_y = _number(counts.get("bbox_scale_y"))
    if bbox_scale_x and bbox_scale_y:
        return {
            "bbox_scale_x": 1.0 / bbox_scale_x,
            "bbox_scale_y": 1.0 / bbox_scale_y,
            "source": "metrics_inverse_bbox_scale",
        }

    score_bbox_scale_x = _number(counts.get("score_bbox_scale_x"))
    score_bbox_scale_y = _number(counts.get("score_bbox_scale_y"))
    if score_bbox_scale_x and score_bbox_scale_y:
        return {
            "bbox_scale_x": score_bbox_scale_x,
            "bbox_scale_y": score_bbox_scale_y,
            "source": "metrics_score_bbox_scale",
        }

    resolution = clip_resolutions.get(clip_id)
    calibration_path = _find_nearby_calibration(tracks_path, clip_id=clip_id)
    if resolution and calibration_path is not None:
        parsed = validate_artifact_file("court_calibration", calibration_path)
        if isinstance(parsed, CourtCalibration):
            calibration_width, calibration_height = calibration_image_size(parsed)
            return {
                "bbox_scale_x": resolution[0] / calibration_width,
                "bbox_scale_y": resolution[1] / calibration_height,
                "source": f"manifest_resolution_over_calibration:{calibration_path}",
            }

    return {"bbox_scale_x": 1.0, "bbox_scale_y": 1.0, "source": "identity"}


def _find_nearby_calibration(tracks_path: Path, *, clip_id: str) -> Path | None:
    for parent in [tracks_path.parent, *tracks_path.parents]:
        candidate = parent / "court_calibration.json"
        if candidate.exists():
            return candidate
        if parent.name == clip_id:
            candidate = parent / "court_calibration.json"
            if candidate.exists():
                return candidate
            break
    return None


def _read_timing(tracks_path: Path) -> dict[str, Any]:
    metrics_path = tracks_path.with_name("metrics.json")
    payload = _read_metrics_payload(tracks_path)
    if not payload:
        return {}
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    return {
        "metrics_path": str(metrics_path),
        "wall_time_s": _number(payload.get("wall_time_s", timing.get("wall_time_s"))),
        "effective_fps": _number(payload.get("effective_fps", timing.get("effective_fps"))),
        "device": payload.get("device"),
        "model": payload.get("model"),
        "tracker_config": payload.get("tracker_config"),
    }


def _read_metrics_payload(tracks_path: Path) -> dict[str, Any]:
    metrics_path = tracks_path.with_name("metrics.json")
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / JSON_REPORT).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / MARKDOWN_REPORT).write_text(render_scoring_report_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
