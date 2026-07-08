#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THREE_CLIP_INPUT = ROOT / "runs/lanes/ball_f1_three_clip_runs_20260705"
IMG1605_INPUT = ROOT / "runs/lanes/w4_freshproof_20260707/img1605/owner_IMG_1605_8a193402780b/owner_IMG_1605_8a193402780b"
CLIPS = (
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
)
INTERNAL_VAL_CLIPS = CLIPS[:2]
F1_FLOORS = {
    "burlington_gold_0300_low_steep_corner": 0.7727272727,
    "wolverine_mixed_0200_mid_steep_corner": 0.875000,
}
FPS_BY_CLIP = {
    "burlington_gold_0300_low_steep_corner": 59.94005994005994,
    "wolverine_mixed_0200_mid_steep_corner": 30.0,
}
BASELINE_GOOD = (
    {"clip": "burlington_gold_0300_low_steep_corner", "name": "burlington_seg2_adjacent", "interval": [107, 132], "baseline_endpoint_error_m": 0.033695},
    {"clip": "burlington_gold_0300_low_steep_corner", "name": "burlington_seg4_adjacent", "interval": [139, 151], "baseline_endpoint_error_m": 0.019624},
    {"clip": "burlington_gold_0300_low_steep_corner", "name": "burlington_seg15_adjacent", "interval": [447, 497], "baseline_endpoint_error_m": 0.1741},
    {"clip": "burlington_gold_0300_low_steep_corner", "name": "burlington_seg16_adjacent", "interval": [497, 543], "baseline_endpoint_error_m": 0.0},
    {"clip": "wolverine_mixed_0200_mid_steep_corner", "name": "wolverine_seg4_region", "interval": [70, 104], "baseline_endpoint_error_m": 0.465495},
)
NAMED_VIOLATORS = (
    {"clip": "burlington_gold_0300_low_steep_corner", "name": "burlington_seg0", "interval": [19, 92]},
    {"clip": "burlington_gold_0300_low_steep_corner", "name": "burlington_seg13", "interval": [347, 423]},
    {"clip": "wolverine_mixed_0200_mid_steep_corner", "name": "wolverine_seg6", "interval": [156, 217]},
    {"clip": "outdoor_webcam_iynbd_1500_long_high_baseline", "name": "outdoor_seg1", "interval": [321, 426]},
    {"clip": "outdoor_webcam_iynbd_1500_long_high_baseline", "name": "outdoor_seg2", "interval": [426, 596]},
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def roundv(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, digits) if math.isfinite(number) else None


def _optional_path(path: Path) -> Path | None:
    return path if path.is_file() else None


def run_chain_variant(variant: str, *, fit_spin_scalar: bool) -> None:
    from threed.racketsport import ball_arc_chain
    from threed.racketsport.ball_arc_chain import run_default_ball_arc_chain

    original_config = ball_arc_chain.default_ball_arc_solver_config

    def configured() -> Any:
        return replace(original_config(), fit_spin_scalar=fit_spin_scalar)

    ball_arc_chain.default_ball_arc_solver_config = configured
    out_root = LANE / "verify" / "replay" / variant
    summary: dict[str, Any] = {"variant": variant, "fit_spin_scalar": fit_spin_scalar, "clips": {}}
    try:
        for clip in CLIPS:
            clip_dir = THREE_CLIP_INPUT / clip
            out_dir = out_root / clip
            if out_dir.exists():
                shutil.rmtree(out_dir)
            started = time.perf_counter()
            result = run_default_ball_arc_chain(
                clip=clip,
                ball_track_path=clip_dir / "ball_track.json",
                court_calibration_path=clip_dir / "court_calibration.json",
                out_dir=out_dir,
                contact_windows_path=_optional_path(clip_dir / "contact_windows.json"),
                skeleton3d_path=_optional_path(clip_dir / "skeleton3d.json"),
                net_plane_path=_optional_path(clip_dir / "net_plane.json"),
                rally_spans_path=_optional_path(clip_dir / "rally_spans.json"),
                frame_times_path=_optional_path(clip_dir / "frame_times.json"),
            )
            summary["clips"][clip] = {"wall_seconds": roundv(time.perf_counter() - started, 3), **result}
            print(json.dumps({"variant": variant, "clip": clip, "status": result["status"], "wall_seconds": summary["clips"][clip]["wall_seconds"]}, sort_keys=True))
    finally:
        ball_arc_chain.default_ball_arc_solver_config = original_config
    write_json(out_root / "run_summary.json", summary)


def run_img1605_variant(variant: str, *, fit_spin_scalar: bool) -> None:
    from threed.racketsport import ball_arc_chain
    from threed.racketsport.ball_arc_chain import run_default_ball_arc_chain

    original_config = ball_arc_chain.default_ball_arc_solver_config

    def configured() -> Any:
        return replace(original_config(), fit_spin_scalar=fit_spin_scalar)

    ball_arc_chain.default_ball_arc_solver_config = configured
    out_dir = LANE / "verify" / "img1605" / variant
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        result = run_default_ball_arc_chain(
            clip="owner_IMG_1605_8a193402780b",
            ball_track_path=IMG1605_INPUT / "ball_track.json",
            court_calibration_path=IMG1605_INPUT / "court_calibration.json",
            out_dir=out_dir,
            net_plane_path=_optional_path(IMG1605_INPUT / "net_plane.json"),
            frame_times_path=_optional_path(IMG1605_INPUT / "frame_times.json"),
        )
    finally:
        ball_arc_chain.default_ball_arc_solver_config = original_config
    write_json(out_dir / "run_summary.json", {"variant": variant, "fit_spin_scalar": fit_spin_scalar, **result})


def _span_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _overlaps(seg: Mapping[str, Any], interval: Sequence[int]) -> bool:
    return _span_overlap(int(interval[0]), int(interval[1]), int(seg["frame_start"]), int(seg["frame_end"])) > 0


def _segments(path_root: Path, clip: str) -> list[dict[str, Any]]:
    return list(read_json(path_root / clip / "ball_track_arc_solved.json").get("segments") or [])


def _finite_rmse_segments(segments: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        seg
        for seg in segments
        if str(seg.get("status") or "").startswith("fit") and _float_or_none(seg.get("reprojection_rmse_px")) is not None
    ]


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _match_segment(segments: Sequence[Mapping[str, Any]], interval: Sequence[int]) -> Mapping[str, Any] | None:
    exact = [
        seg
        for seg in segments
        if int(seg.get("frame_start", -1)) == int(interval[0]) and int(seg.get("frame_end", -1)) == int(interval[1])
    ]
    if exact:
        return exact[0]
    overlaps = [seg for seg in segments if _overlaps(seg, interval)]
    if not overlaps:
        return None
    return max(overlaps, key=lambda seg: (_span_overlap(int(interval[0]), int(interval[1]), int(seg["frame_start"]), int(seg["frame_end"])), -int(seg.get("segment_id", 0))))


def _segment_row(seg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if seg is None:
        return None
    return {
        "segment_id": seg.get("segment_id"),
        "frames": [int(seg["frame_start"]), int(seg["frame_end"])],
        "status": seg.get("status"),
        "reprojection_rmse_px": roundv(seg.get("reprojection_rmse_px")),
        "endpoint_error_m": roundv(seg.get("endpoint_error_m")),
        "spin_scalar": roundv(seg.get("spin_scalar"), 9),
        "inlier_count": seg.get("inlier_count"),
        "outlier_count": seg.get("outlier_count"),
        "physical_violations": ((seg.get("physical_sanity") or {}).get("violations") or []),
    }


def compare_reprojection() -> None:
    base_root = LANE / "verify/replay/baseline_default_off"
    spin_root = LANE / "verify/replay/spin_on"
    clip_rows = []
    segment_rows = []
    benefited = []
    for clip in CLIPS:
        before_segments = _segments(base_root, clip)
        after_segments = _segments(spin_root, clip)
        before_values = [float(seg["reprojection_rmse_px"]) for seg in _finite_rmse_segments(before_segments)]
        after_values = [float(seg["reprojection_rmse_px"]) for seg in _finite_rmse_segments(after_segments)]
        before_mean = _mean(before_values)
        after_mean = _mean(after_values)
        clip_rows.append(
            {
                "clip": clip,
                "before_mean_reprojection_rmse_px": roundv(before_mean),
                "after_mean_reprojection_rmse_px": roundv(after_mean),
                "delta": roundv((after_mean or 0.0) - (before_mean or 0.0)) if before_mean is not None and after_mean is not None else None,
                "verdict": "PASS" if before_mean is not None and after_mean is not None and after_mean <= before_mean + 1e-9 else "FAIL",
            }
        )
        by_interval = {(int(seg["frame_start"]), int(seg["frame_end"])): seg for seg in after_segments}
        for before in before_segments:
            after = by_interval.get((int(before["frame_start"]), int(before["frame_end"])))
            row = {"clip": clip, "before": _segment_row(before), "after": _segment_row(after)}
            segment_rows.append(row)
            b_rmse = _float_or_none(before.get("reprojection_rmse_px"))
            a_rmse = _float_or_none(after.get("reprojection_rmse_px")) if after else None
            if b_rmse is not None and a_rmse is not None and a_rmse < b_rmse - 1e-9:
                benefited.append(row)
    wolverine_interval = [156, 217]
    wolverine_before = _match_segment(_segments(base_root, "wolverine_mixed_0200_mid_steep_corner"), wolverine_interval)
    wolverine_after = _match_segment(_segments(spin_root, "wolverine_mixed_0200_mid_steep_corner"), wolverine_interval)
    before_rmse = _float_or_none(wolverine_before.get("reprojection_rmse_px")) if wolverine_before else None
    after_rmse = _float_or_none(wolverine_after.get("reprojection_rmse_px")) if wolverine_after else None
    wolverine_seg6 = {
        "interval": wolverine_interval,
        "before": _segment_row(wolverine_before),
        "after": _segment_row(wolverine_after),
        "delta": roundv((after_rmse or 0.0) - (before_rmse or 0.0)) if before_rmse is not None and after_rmse is not None else None,
        "verdict": "PASS" if before_rmse is not None and after_rmse is not None and after_rmse < before_rmse - 1e-9 else "FAIL",
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "w6_magnus_reprojection_compare",
        "clip_rows": clip_rows,
        "wolverine_seg6": wolverine_seg6,
        "benefited_segments": benefited,
        "segment_rows": segment_rows,
        "verdict": "PASS" if all(row["verdict"] == "PASS" for row in clip_rows) and wolverine_seg6["verdict"] == "PASS" else "FAIL",
    }
    write_json(LANE / "verify/reprojection_compare.json", payload)
    print(json.dumps({"verdict": payload["verdict"], "clip_rows": clip_rows, "wolverine_seg6": wolverine_seg6}, indent=2, sort_keys=True))


def audit_d3() -> None:
    rows: dict[str, Any] = {}
    for variant in ("baseline_default_off", "spin_on"):
        root = LANE / "verify/replay" / variant
        artifacts = {clip: read_json(root / clip / "ball_track_arc_solved.json") for clip in CLIPS}
        d3a_bad = []
        for clip, artifact in artifacts.items():
            for seg in artifact.get("segments") or []:
                if seg.get("status") == "fit" and bool((seg.get("physical_sanity") or {}).get("violation")):
                    d3a_bad.append({"clip": clip, "segment": _segment_row(seg)})
        named = []
        for item in NAMED_VIOLATORS:
            overlaps = [_segment_row(seg) for seg in artifacts[item["clip"]].get("segments") or [] if _overlaps(seg, item["interval"])]
            fit_with_violation = any(seg and seg["status"] == "fit" and seg["physical_violations"] for seg in overlaps)
            named.append({**item, "overlaps": overlaps, "verdict": "PASS" if not fit_with_violation else "FAIL"})
        d3b = []
        for item in BASELINE_GOOD:
            seg = _match_segment(artifacts[item["clip"]].get("segments") or [], item["interval"])
            endpoint = _float_or_none(seg.get("endpoint_error_m")) if seg else None
            status = str(seg.get("status") or "") if seg else "missing"
            d3b.append(
                {
                    **item,
                    "after": _segment_row(seg),
                    "verdict": "PASS" if status.startswith("fit") and endpoint is not None and endpoint <= float(item["baseline_endpoint_error_m"]) + 1e-9 else "FAIL",
                }
            )
        render_bad = []
        for clip, artifact in artifacts.items():
            render_path = root / clip / "ball_arc_render.json"
            if not render_path.is_file():
                continue
            bounds = None
            for seg in artifact.get("segments") or []:
                bounds = (((seg.get("physical_sanity") or {}).get("court_volume") or {}).get("bounds_m"))
                if bounds:
                    break
            if not bounds:
                continue
            for sample in (read_json(render_path).get("samples") or []):
                court_xy = sample.get("court_xy")
                world = sample.get("world_xyz")
                if court_xy is None or world is None:
                    continue
                x, y, z = float(court_xy[0]), float(court_xy[1]), float(world[2])
                if x < bounds["x_min"] or x > bounds["x_max"] or y < bounds["y_min"] or y > bounds["y_max"] or z < bounds["z_min"]:
                    render_bad.append({"clip": clip, "frame": sample.get("frame"), "world_xyz": world, "court_xy": court_xy})
        outdoor_phys = (artifacts["outdoor_webcam_iynbd_1500_long_high_baseline"].get("validation") or {}).get("physical_sanity") or {}
        rows[variant] = {
            "d3a": {"bad_fit_segments": d3a_bad, "named_violators": named, "verdict": "PASS" if not d3a_bad and all(row["verdict"] == "PASS" for row in named) else "FAIL"},
            "d3b": {"rows": d3b, "verdict": "PASS" if all(row["verdict"] == "PASS" for row in d3b) else "FAIL"},
            "d3c": {"render_bad_sample_count": len(render_bad), "examples": render_bad[:20], "verdict": "PASS" if not render_bad else "FAIL"},
            "d3d": {"outdoor_physical_sanity": {key: outdoor_phys.get(key) for key in ("segment_count", "fallback_excluded_segment_count", "violation_count", "violation_eligible_segment_count", "violation_fraction", "kill_threshold_fraction")}, "verdict": "PASS" if outdoor_phys.get("violation_fraction") is not None and outdoor_phys.get("violation_fraction") <= 0.142857 else "FAIL"},
        }
    payload = {
        "schema_version": 1,
        "artifact_type": "w6_magnus_d3_a_to_d_audit",
        "variants": rows,
        "spin_on_verdict": "PASS" if all(rows["spin_on"][key]["verdict"] == "PASS" for key in ("d3a", "d3b", "d3c", "d3d")) else "FAIL",
        "default_off_verdict": "PASS" if all(rows["baseline_default_off"][key]["verdict"] == "PASS" for key in ("d3a", "d3b", "d3c", "d3d")) else "FAIL",
    }
    write_json(LANE / "verify/d3_a_to_d_audit.json", payload)
    print(json.dumps({"spin_on_verdict": payload["spin_on_verdict"], "default_off_verdict": payload["default_off_verdict"]}, indent=2, sort_keys=True))


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(os.path.relpath(src, start=dst.parent))
    except OSError:
        shutil.copy2(src, dst)


def _product_view_from_arc(arc_path: Path, out_path: Path, clip: str) -> None:
    arc = read_json(arc_path)
    frames = []
    for frame in arc["frames"]:
        frames.append(
            {
                "t": frame.get("t"),
                "visible": bool(frame.get("visible")),
                "xy": frame.get("xy"),
                "conf": frame.get("conf"),
                "approx": bool(frame.get("approx", False)),
                "world_xyz": frame.get("world_xyz"),
                "speed_mps": frame.get("speed_mps"),
                "spin_rpm": frame.get("spin_rpm"),
            }
        )
    write_json(out_path, {"schema_version": 1, "source": "physics_filled", "fps": FPS_BY_CLIP[clip], "frames": frames, "bounces": arc.get("bounces", [])})


def run_d3e_eval() -> None:
    d3e_root = LANE / "verify/d3e_product_eval"
    review_root = d3e_root / "review_root"
    run_root = d3e_root / "run_root"
    product_root = d3e_root / "product_views"
    out_root = d3e_root / "eval_suite"
    for clip in INTERNAL_VAL_CLIPS:
        _symlink_or_copy(ROOT / "eval_clips/ball" / clip / "labels/ball_points.json", review_root / clip / "ball_points.json")
        _symlink_or_copy(ROOT / "eval_clips/ball" / clip / "source.mp4", run_root / clip / "tracknet_smoke_0000_0010/input_0000_0010.mp4")
        for variant in ("baseline_default_off", "spin_on"):
            _product_view_from_arc(LANE / "verify/replay" / variant / clip / "ball_track_arc_solved.json", product_root / clip / f"{variant}.json", clip)
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/racketsport/run_ball_tracking_eval_suite.py"),
        "--run-root",
        str(run_root),
        "--review-root",
        str(review_root),
        "--out-root",
        str(out_root),
        "--clip",
        "burlington_gold_0300_low_steep_corner",
        "--clip",
        "wolverine_mixed_0200_mid_steep_corner",
        "--no-pbmat-v0",
    ]
    for clip in INTERNAL_VAL_CLIPS:
        for variant in ("baseline_default_off", "spin_on"):
            cmd.extend(["--external-candidate", f"{clip}:{variant}:post_change={product_root / clip / f'{variant}.json'}"])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "w6_magnus_internal_val_d3e",
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    rows = []
    if proc.returncode == 0:
        benchmark = read_json(out_root / "benchmark.json")
        for row in benchmark["results"]:
            if row["candidate"] not in {"baseline_default_off", "spin_on"}:
                continue
            metrics = row["label_metrics"]
            rows.append(
                {
                    "clip": row["clip"],
                    "candidate": row["candidate"],
                    "label_f1_at_20px": metrics["label_f1_at_20px"],
                    "visible_recall_at_20px": metrics["visible_recall_at_20px"],
                    "visible_presence_recall": metrics["visible_presence_recall"],
                    "precision_at_20px": metrics["precision_at_20px"],
                    "hidden_false_positive_count": metrics["hidden_false_positive_count"],
                    "floor_label_f1_at_20px": F1_FLOORS[row["clip"]],
                    "delta_vs_floor": metrics["label_f1_at_20px"] - F1_FLOORS[row["clip"]],
                }
            )
    result["rows"] = rows
    spin_rows = [row for row in rows if row["candidate"] == "spin_on"]
    default_rows = [row for row in rows if row["candidate"] == "baseline_default_off"]
    result["spin_on_verdict"] = "PASS" if proc.returncode == 0 and all(row["delta_vs_floor"] >= -0.01 for row in spin_rows) else "FAIL"
    result["default_off_verdict"] = "PASS" if proc.returncode == 0 and all(row["delta_vs_floor"] >= -0.01 for row in default_rows) else "FAIL"
    write_json(d3e_root / "fresh_d3e_product_eval.json", result)
    print(json.dumps({"spin_on_verdict": result["spin_on_verdict"], "default_off_verdict": result["default_off_verdict"], "rows": rows}, indent=2, sort_keys=True))


def _reason_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        reason = str(event.get("selection_reason") or event.get("reason") or "null")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _img1605_counts(root: Path) -> dict[str, Any]:
    arc = read_json(root / "ball_track_arc_solved.json")
    bounce = read_json(root / "ball_bounce_candidates.json")
    selected = read_json(root / "events_selected.json")
    render = read_json(root / "ball_arc_render.json")
    return {
        "root": str(root.relative_to(ROOT)),
        "raw_bounce_candidate_count": (bounce.get("summary") or {}).get("raw_candidate_count"),
        "final_bounce_candidate_count": (bounce.get("summary") or {}).get("final_candidate_count", len(bounce.get("candidates") or [])),
        "selected_event_count": len(selected.get("selected") or []),
        "rejected_event_count": len(selected.get("rejected") or []),
        "rejected_reason_counts": _reason_counts(selected.get("rejected") or []),
        "segment_count": len(arc.get("segments") or []),
        "fit_segment_count": sum(1 for seg in arc.get("segments") or [] if str(seg.get("status") or "").startswith("fit")),
        "render_sample_count": len(render.get("samples") or []),
        "coverage_world_xyz_count": arc.get("summary", {}).get("coverage_world_xyz_count"),
    }


def img1605_census() -> None:
    before = _img1605_counts(LANE / "verify/img1605/baseline_default_off")
    after = _img1605_counts(LANE / "verify/img1605/spin_on")
    payload = {
        "schema_version": 1,
        "artifact_type": "w6_magnus_img1605_census",
        "before": before,
        "after": after,
        "tracked_key_deltas": {key: {"before": before.get(key), "after": after.get(key)} for key in sorted(before) if key != "root"},
    }
    write_json(LANE / "verify/img1605_census.json", payload)
    print(json.dumps(payload["tracked_key_deltas"], indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run-baseline")
    sub.add_parser("run-spin")
    sub.add_parser("compare")
    sub.add_parser("d3")
    sub.add_parser("d3e")
    sub.add_parser("img1605")
    sub.add_parser("all")
    args = parser.parse_args()
    if args.cmd in {"run-baseline", "all"}:
        run_chain_variant("baseline_default_off", fit_spin_scalar=False)
        run_img1605_variant("baseline_default_off", fit_spin_scalar=False)
    if args.cmd in {"run-spin", "all"}:
        run_chain_variant("spin_on", fit_spin_scalar=True)
        run_img1605_variant("spin_on", fit_spin_scalar=True)
    if args.cmd in {"compare", "all"}:
        compare_reprojection()
    if args.cmd in {"d3", "all"}:
        audit_d3()
    if args.cmd in {"d3e", "all"}:
        run_d3e_eval()
    if args.cmd in {"img1605", "all"}:
        img1605_census()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
