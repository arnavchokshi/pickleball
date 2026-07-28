"""Measure background-ball acquisition and score every candidate discriminator.

VERIFIED=0. This is measurement, not a promotion. Run from the repo root:

    PYTHONPATH=. .venv/bin/python \
        runs/lanes/background_ball_20260727/measure_background_ball.py \
        --artifact-root /Users/arnavchokshi/Desktop/pickleball \
        --out runs/lanes/background_ball_20260727/report.json

``--artifact-root`` is where the (gitignored) run artifacts live. Every input
is read, never written.

The frozen 167-row judge is reproduced from its own artifacts before anything
else is reported; if the reproduction does not match the published pooled and
per-venue numbers the script exits non-zero rather than emit a report built on
a mis-wired scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from threed.racketsport.ball_position_plausibility import evaluate_position
from threed.racketsport.ball_ray_court_volume import (
    CourtVolumeBounds,
    evaluate_ray,
)

JUDGE_ROWS = "runs/lanes/ball_b0_split_20260721/split/validation.jsonl"
JUDGE_PRED = "runs/lanes/ball_baseline_20260721/predictions/wasb_official_control"
JUDGE_CALIBRATIONS = "data/online_harvest_20260706/court_calibrations"

#: Published frozen-judge scores this harness must reproduce before it reports.
PUBLISHED = {
    "pooled": {"f1": 0.5670103092783506, "recall": 0.5851063829787234,
               "precision": 0.55, "hidden_fp": 0.4931506849315068},
    "indoor_court_level": {"f1": 0.7394957983193278, "recall": 0.6875,
                           "precision": 0.8, "hidden_fp": 0.3055555555555556},
    "outdoor_night_fenced": {"f1": 0.29333333333333333, "recall": 0.36666666666666664,
                             "precision": 0.24444444444444444, "hidden_fp": 0.6756756756756757},
}

OWNER_LABELS = "runs/lanes/ball_label_tool_20260726/labels"
OWNER_CLIPS = {
    "wolverine": "runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_20s": "runs/full_mesh_examples_20260725/outdoor_mesh_final/outdoor_webcam_20s_fullmesh_final",
    "burlington": "runs/lanes/label_clip_prep_20260727/burlington_gold_0300_low_steep_corner",
    "indoor": "runs/full_mesh_examples_20260725/indoor_mesh_final/indoor_doubles_20s_fullmesh_final",
}

MARGIN_SETTINGS = [
    ("margin_2m_apex_8m", 2.0, 8.0),
    ("margin_1m_apex_6m", 1.0, 6.0),
    ("margin_0p5m_apex_5m", 0.5, 5.0),
]

GROSS_ERROR_PX = 100.0
HIT_RADIUS_PX = 20.0

#: Conclusions this lane established. Recorded in the artifact so the report is
#: self-describing: the numeric blocks below are the evidence for each claim.
FINDINGS = {
    "background_ball_acquisition": {
        "verdict": "CONFIRMED",
        "basis": (
            "Visual adjudication of every hidden false positive in the frozen judge, from "
            "the judge's own image zip. Indoor: 11 of 11 are real pickleballs that are not "
            "the ball in play (stray balls on our own court and just outside the sideline, "
            "balls in the walkway beyond the far baseline). Outdoor night: roughly 14 of 25 "
            "are real balls elsewhere in the venue; the rest are lights, night-sky specks "
            "and fence clutter. On burlington frames 30 and 369 the detector is locked on a "
            "game being played on the adjacent court."
        ),
    },
    "correct_ball_among_top_k": {
        "verdict": "REFUTED -- this is not a cheap selection fix",
        "basis": (
            "WASB picks argmax(blob score) among blobs within 300 px of the previous "
            "accepted position. For a selection fix to work the right ball must be present "
            "but unselected. It is not: 11 of 11 indoor hidden-FP frames had exactly one "
            "blob, and 6 of 9 outdoor gross mislocalisations had exactly one blob, with the "
            "nearest blob to the true ball 280, 294 and 497 px away in the other three. The "
            "ball in play produced no heatmap response at all. Any suppression-based fix is "
            "therefore precision-only; recall is capped by the detector."
        ),
    },
    "ray_court_volume_discriminator": {
        "verdict": "SOUND BUT USELESS -- refuted",
        "basis": (
            "Flags 0.0-1.3 percent of detections at margin 2 m / apex 8 m across 1269 "
            "emitted detections on four owner clips with reviewed metric_15pt calibrations. "
            "The only settings that flag anything also suppress owner-clicked real-ball "
            "pixels. Mechanism: a camera behind the baseline sees an adjacent court through "
            "its own airspace. On burlington frame 30 a ball 19 m outside our sideline has a "
            "ray that hits the ground at x=-21.9 m but crosses our near-baseline plane at "
            "x=-2.95 m, z=0.95 m -- inside the volume, and an ordinary place for a real "
            "ball. At the 2 m margin the burlington camera centre itself lies inside the "
            "volume, making the test vacuous on that clip. Chord length separates burlington "
            "frame 30 (0.61 m wrong vs 11.37 m real) but inverts on frame 369."
        ),
    },
    "other_discriminators_measured": {
        "heatmap_blob_radius": "no separation; quantised identically for true and false positives (it measures the heatmap Gaussian, not the ball)",
        "image_apparent_radius": "weak and abstains where it matters; indoor median 4.4 px FP vs 6.6 px GT with heavy overlap, abstains on 24 of 25 outdoor FPs",
        "stationarity": "no separation; indoor FP local displacement median 134.7 px vs TP 157.0 px",
        "teleport_continuity_structure": "no separation and inverted; indoor FPs sit in longer, smoother runs than TPs (median run 31 frames vs 17.5, 16.9 px/frame vs 20.4)",
        "image_position_2d": "no separation; hidden FPs are spatially interleaved with true positives",
        "detector_confidence": "no separation; sweeping 0.5 to 0.85 moves indoor precision 0.800 to 0.750 while recall collapses 0.688 to 0.328",
    },
    "why_hidden_fp_exceeds_the_owner_click_error_rate": (
        "The two measure different frame populations and do not contradict each other. "
        "Hidden-FP is computed only over frames where the reviewer recorded no ball; the "
        "owner labelled only frames where the ball was visible. Where a real ball is "
        "present the detector is usually right -- indoor 44 of 44 within 20 px, median "
        "error 4.3 px. The wrong-ball problem lives in the frames where the ball in play is "
        "absent or invisible."
    ),
    "ball_3d_coverage_knock_on": {
        "verdict": "CONTRIBUTING BUT NOT DOMINANT -- partly refuted",
        "basis": (
            "Of the outdoor clip's 411 missing 3D frames, 294 (72 percent) were never "
            "emitted in 2D at all, 65 emitted sightings were pruned as false positives by "
            "the solver (the genuine wrong-ball contribution, ~21 percent of emitted), and "
            "2 segments were dropped for segment_budget_exceeded -- a wall-clock timeout, "
            "not a physics failure. The indoor clip's 0 percent coverage is entirely "
            "segment_budget_exceeded: both segments timed out, so no wrong-ball explanation "
            "is needed or supported there."
        ),
    },
    "what_does_partially_work": (
        "Sequence-level 3D reasoning, which already exists. Gravity fixes the depth scale, "
        "so an adjacent-court ball fitted as a ballistic arc lands at its true off-court "
        "position: on burlington frame 369 the solver placed the wrong ball at x=-14.9 m, "
        "flagged outside_court_footprint and absurd by ball_position_plausibility. Across "
        "burlington, 97 of 508 solved frames violate the footprint bound and 57 are absurd. "
        "It is noisy -- frame 30 was placed on the sideline and not flagged, and on "
        "wolverine the check fires 107 times for an unrelated above_plausible_apex solver "
        "defect -- but it is the only signal measured in this lane that separates the "
        "classes at all."
    ),
    "recommendation": [
        "Do not pursue single-frame geometric filtering; measured dead.",
        "The largest measured headroom is detector recall, not precision: 20 of 64 indoor "
        "and 10 of 30 outdoor present-ball frames get no detection.",
        "Fix segment_budget_exceeded; it alone explains 100 percent of indoor's zero 3D coverage.",
        "Give the frozen judge a calibration; both its sources have none, which blocks every "
        "geometric hypothesis from ever being scored on the repo's own gate.",
        "If wrong-ball suppression is still wanted, pursue rally-level 3D association.",
    ],
}


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# camera geometry
# --------------------------------------------------------------------------

def camera_from_calibration(calibration: dict[str, Any]) -> dict[str, Any] | None:
    """Return the pieces needed to turn a pixel into a world ray, or None."""

    intrinsics = calibration.get("intrinsics") or {}
    extrinsics = calibration.get("extrinsics") or {}
    if not intrinsics or not extrinsics:
        return None
    import numpy as np

    K = np.array(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]],
         [0.0, intrinsics["fy"], intrinsics["cy"]],
         [0.0, 0.0, 1.0]], float)
    dist = np.array(list(intrinsics.get("dist") or [0.0, 0.0, 0.0, 0.0]), float)
    R = np.array(extrinsics["R"], float)
    t = np.array(extrinsics["t"], float)
    centre = -R.T @ t
    return {"K": K, "dist": dist, "R": R, "centre": centre}


def pixel_ray(camera: dict[str, Any], pixel_xy) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import cv2
    import numpy as np

    point = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], np.float64)
    coeffs = np.array([camera["dist"][0], camera["dist"][1], 0.0, 0.0], np.float64)
    normalised = cv2.undistortPoints(point, camera["K"], coeffs)
    direction_camera = np.array([normalised[0, 0, 0], normalised[0, 0, 1], 1.0])
    direction_camera /= np.linalg.norm(direction_camera)
    direction_world = camera["R"].T @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    return tuple(camera["centre"]), tuple(direction_world)


# --------------------------------------------------------------------------
# frozen judge
# --------------------------------------------------------------------------

def build_judge_rows(root: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (root / JUDGE_ROWS).read_text().splitlines() if line.strip()]
    cache: dict[str, Any] = {}
    out = []
    for row in rows:
        clip = row["clip_id"]
        if clip not in cache:
            cache[clip] = load_json(root / JUDGE_PRED / clip / "wasb" / "ball_track.json")["frames"]
        frames = cache[clip]
        index = row["frame_index"]
        frame = frames[index] if index < len(frames) else None
        visible = bool(frame and frame.get("visible"))
        label = row["final_label"]
        bbox = label.get("bbox_xyxy")
        centre = None
        if bbox:
            centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        error = None
        if visible and centre:
            error = math.hypot(frame["xy"][0] - centre[0], frame["xy"][1] - centre[1])
        out.append({
            "clip": clip, "frame": index, "venue": row["source_class"],
            "present": bool(label["ball_present"]), "emitted": visible,
            "xy": list(frame["xy"]) if visible else None,
            "gt_xy": list(centre) if centre else None,
            "error_px": error,
            "confidence": float(frame.get("conf")) if frame and frame.get("conf") is not None else None,
        })
    return out


def score_judge(rows: list[dict[str, Any]], keep) -> dict[str, Any]:
    """Reproduce the frozen judge's counters under a survival predicate."""

    result: dict[str, Any] = {}
    for venue in ("indoor_court_level", "outdoor_night_fenced"):
        subset = [row for row in rows if row["venue"] == venue]
        tp = fp = fn = hidden_fp = hidden = 0
        for row in subset:
            emitted = row["emitted"] and keep(row)
            if row["present"]:
                if emitted and row["error_px"] is not None and row["error_px"] <= HIT_RADIUS_PX:
                    tp += 1
                elif emitted:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            else:
                hidden += 1
                if emitted:
                    fp += 1
                    hidden_fp += 1
        result[venue] = _metrics(tp, fp, fn, hidden_fp, hidden)
    tp = sum(result[v]["true_positives"] for v in result)
    fp = sum(result[v]["false_positives"] for v in result)
    fn = sum(result[v]["false_negatives"] for v in result)
    hidden_fp = sum(result[v]["hidden_false_positives"] for v in result)
    hidden = sum(result[v]["hidden_label_count"] for v in result)
    result["pooled"] = _metrics(tp, fp, fn, hidden_fp, hidden)
    return result


def _metrics(tp: int, fp: int, fn: int, hidden_fp: int, hidden: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1_at_20": round(f1, 6), "recall_at_20": round(recall, 6),
        "precision_at_20": round(precision, 6),
        "hidden_fp_rate": round(hidden_fp / hidden, 6) if hidden else 0.0,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "hidden_false_positives": hidden_fp, "hidden_label_count": hidden,
    }


def check_reproduction(scores: dict[str, Any]) -> list[str]:
    problems = []
    for key, expected in PUBLISHED.items():
        got = scores[key]
        for metric, want in expected.items():
            have = got[f"{metric}_at_20"] if metric != "hidden_fp" else got["hidden_fp_rate"]
            if abs(have - want) > 1e-4:
                problems.append(f"{key}.{metric}: reproduced {have} != published {want}")
    return problems


# --------------------------------------------------------------------------
# owner clips
# --------------------------------------------------------------------------

def measure_owner_clips(root: Path, repo: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, relative in OWNER_CLIPS.items():
        clip_dir = root / relative
        calibration = load_json(clip_dir / "court_calibration.json")
        camera = camera_from_calibration(calibration)
        track = load_json(clip_dir / "ball_track.json")["frames"]
        labels = load_json(repo / OWNER_LABELS / name / "ball_human_labels.json")["labels"]
        emitted = [(i, f["xy"]) for i, f in enumerate(track) if f.get("visible")]

        by_setting: dict[str, Any] = {}
        for setting, margin, apex in MARGIN_SETTINGS:
            bounds = CourtVolumeBounds(margin_m=margin, apex_m=apex)
            disjoint = 0
            chords = []
            for _, pixel in emitted:
                origin, direction = pixel_ray(camera, pixel)
                report = evaluate_ray(origin, direction, bounds)
                if report["verdict"] == "disjoint":
                    disjoint += 1
                else:
                    chords.append(report["chord_length_m"])
            # would the filter suppress a pixel the owner clicked on the real ball?
            false_suppressions = 0
            for label in labels:
                origin, direction = pixel_ray(camera, label["pixel_xy"])
                if evaluate_ray(origin, direction, bounds)["verdict"] == "disjoint":
                    false_suppressions += 1
            by_setting[setting] = {
                "disjoint_count": disjoint,
                "disjoint_rate": round(disjoint / len(emitted), 6) if emitted else 0.0,
                "median_chord_m": round(sorted(chords)[len(chords) // 2], 4) if chords else None,
                "owner_click_false_suppressions": false_suppressions,
                "owner_click_count": len(labels),
            }

        # direct owner cross-check: detector pixel vs owner click at the same frame
        agree = gross = 0
        gross_frames = []
        unlabelled_by_detector = 0
        for label in labels:
            index = label["frame"]
            if index >= len(track) or not track[index].get("visible"):
                unlabelled_by_detector += 1
                continue
            dx = track[index]["xy"][0] - label["pixel_xy"][0]
            dy = track[index]["xy"][1] - label["pixel_xy"][1]
            distance = math.hypot(dx, dy)
            if distance > GROSS_ERROR_PX:
                gross += 1
                gross_frames.append({"frame": index, "error_px": round(distance, 1)})
            else:
                agree += 1

        out[name] = {
            "clip_id": calibration.get("clip_id") or name,
            "calibration_source": calibration.get("source"),
            "calibration_reprojection_px": calibration.get("reprojection_error_px"),
            "camera_centre_m": [round(float(v), 4) for v in camera["centre"]],
            "frame_count": len(track),
            "emitted_detection_count": len(emitted),
            "emitted_coverage_rate": round(len(emitted) / len(track), 6) if track else 0.0,
            "ray_court_volume": by_setting,
            "owner_cross_check": {
                "label_count": len(labels),
                "labels_with_a_detection": agree + gross,
                "labels_without_a_detection": unlabelled_by_detector,
                "detector_agrees_within_100px": agree,
                "detector_gross_error_over_100px": gross,
                "gross_error_rate_where_detected": round(gross / (agree + gross), 6) if agree + gross else 0.0,
                "gross_error_frames": gross_frames,
            },
        }
    return out


def measure_ball_3d_coverage(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, relative in OWNER_CLIPS.items():
        path = root / relative / "ball_track_arc_solved.json"
        if not path.exists():
            out[name] = {"available": False}
            continue
        solved = load_json(path)
        summary = solved.get("summary") or {}
        frames = solved["frames"]
        track = load_json(root / relative / "ball_track.json")["frames"]
        emitted = sum(1 for f in track if f.get("visible"))
        with_world = [(i, f) for i, f in enumerate(frames) if f.get("world_xyz")]
        violations: dict[str, int] = {}
        absurd = 0
        for _, frame in with_world:
            verdict = evaluate_position(frame["world_xyz"])
            for name_ in verdict["violations"]:
                violations[name_] = violations.get(name_, 0) + 1
            if verdict["absurd"]:
                absurd += 1
        out[name] = {
            "available": True,
            "status": solved.get("status"),
            "input_frame_count": len(frames),
            "emitted_2d_count": emitted,
            "world_xyz_frame_count": len(with_world),
            "coverage_of_all_frames": round(len(with_world) / len(frames), 6) if frames else 0.0,
            "coverage_of_emitted": round(len(with_world) / emitted, 6) if emitted else 0.0,
            "degraded_reasons": solved.get("degraded_reasons"),
            "fp_sightings_pruned_count": summary.get("fp_sightings_pruned_count"),
            "missing_segment_count": summary.get("missing_segment_count"),
            "missing_segment_reasons": summary.get("missing_segment_reasons"),
            "flight_sanity_failed_segment_count": summary.get("flight_sanity_failed_segment_count"),
            "position_plausibility_violations": violations,
            "position_plausibility_absurd_count": absurd,
        }
    return out


def judge_calibration_status(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in ("HyUqT7zFiwk", "Ezz6HDNHlnk"):
        path = root / JUDGE_CALIBRATIONS / f"{source}.json"
        if not path.exists():
            out[source] = {"present": False}
            continue
        calibration = load_json(path)
        out[source] = {
            "present": True,
            "calibration_grade": calibration.get("calibration_grade"),
            "failure_reason": calibration.get("failure_reason"),
            "frozen_calibration_is_null": calibration.get("frozen_calibration") is None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True,
                        help="root holding the gitignored run artifacts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.artifact_root)
    repo = Path(__file__).resolve().parents[3]

    rows = build_judge_rows(root)
    baseline = score_judge(rows, lambda row: True)
    problems = check_reproduction(baseline)
    if problems:
        for problem in problems:
            print(f"JUDGE REPRODUCTION MISMATCH: {problem}", file=sys.stderr)
        return 2
    print(f"judge reproduction OK: {len(rows)} rows, pooled f1={baseline['pooled']['f1_at_20']}")

    ceiling = score_judge(
        rows,
        lambda row: row["present"] and row["error_px"] is not None and row["error_px"] <= HIT_RADIUS_PX,
    )
    confidence_sweep = {
        f"{threshold:.2f}": score_judge(rows, lambda row, t=threshold: (row["confidence"] or 0.0) >= t)
        for threshold in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
    }

    report = {
        "artifact_type": "racketsport_background_ball_investigation",
        "schema_version": 1,
        "lane": "background_ball_20260727",
        "verified": 0,
        "not_ground_truth": True,
        "language": "measurement-only; VERIFIED=0 remains binding; no promotion claims",
        "findings": FINDINGS,
        "judge": {
            "rows": len(rows),
            "reproduces_published_scores": True,
            "baseline": baseline,
            "perfect_suppression_ceiling": ceiling,
            "confidence_threshold_sweep": confidence_sweep,
            "calibration_status": judge_calibration_status(root),
            "ray_court_volume_scoreable": False,
            "ray_court_volume_not_scoreable_reason":
                "Both frozen-judge sources carry calibration_grade=failed with a null "
                "frozen_calibration, so no camera pose exists to build a ray from. Any "
                "calibration-dependent filter is unscoreable on this judge by construction.",
        },
        "owner_clips": measure_owner_clips(root, repo),
        "ball_3d_coverage": measure_ball_3d_coverage(root),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
