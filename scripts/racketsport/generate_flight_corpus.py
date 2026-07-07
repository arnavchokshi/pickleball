#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.flight_simulator import (
    DEFAULT_CALIBRATION_PATH,
    DetectorNoiseProfile,
    evaluate_simulated_flight_sanity,
    generate_corpus,
    generate_trajectory_pair,
    load_court_calibration,
    round_trip_fit_report,
    sample_shot_family,
)


MEASURED_ERROR_PROFILE = DetectorNoiseProfile()


@dataclass(frozen=True)
class BurstDropoutConfig:
    enabled: bool = True
    burst_dropout_share: float = 0.70
    min_frames: int = 3
    max_frames: int = 12

    def validate(self) -> None:
        if not 0.0 <= float(self.burst_dropout_share) <= 1.0:
            raise ValueError("burst_dropout_share must be in [0, 1]")
        if self.min_frames < 1:
            raise ValueError("burst-min-frames must be >= 1")
        if self.max_frames < self.min_frames:
            raise ValueError("burst-max-frames must be >= burst-min-frames")

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "burst_dropout_share": float(self.burst_dropout_share),
            "min_frames": int(self.min_frames),
            "max_frames": int(self.max_frames),
        }


class _NoiseStatsAccumulator:
    def __init__(self) -> None:
        self.errors_px: list[float] = []
        self.visible_clean_frames = 0
        self.true_positive_count = 0
        self.hidden_fp_count = 0
        self.frame_count = 0

    def add(self, clean_track: Sequence[Mapping[str, Any]], noisy: Mapping[str, Any]) -> None:
        self.frame_count += len(clean_track)
        clean_by_frame = {
            int(frame["frame"]): frame for frame in clean_track if bool(frame.get("visible"))
        }
        self.visible_clean_frames += len(clean_by_frame)
        for detection in noisy.get("detections") or []:
            kind = detection.get("kind")
            if kind == "hidden_false_positive":
                self.hidden_fp_count += 1
                continue
            if kind != "true_positive":
                continue
            matched = clean_by_frame.get(int(detection["matched_clean_frame"]))
            if matched is None:
                continue
            self.true_positive_count += 1
            clean_xy = [float(value) for value in matched["xy_px"]]
            det_xy = [float(value) for value in detection["xy_px"]]
            self.errors_px.append(math.hypot(det_xy[0] - clean_xy[0], det_xy[1] - clean_xy[1]))

    def to_stats(self, profile: DetectorNoiseProfile) -> dict[str, Any]:
        stats = {
            "visible_clean_frames": int(self.visible_clean_frames),
            "true_positive_count": int(self.true_positive_count),
            "hidden_fp_count": int(self.hidden_fp_count),
            "jitter_p95_px": _round(float(np.percentile(self.errors_px, 95)) if self.errors_px else 0.0, 6),
            "recall": _round(
                self.true_positive_count / self.visible_clean_frames if self.visible_clean_frames else 0.0,
                9,
            ),
            "hidden_fp_rate": _round(
                self.hidden_fp_count / self.frame_count if self.frame_count else 0.0,
                9,
            ),
            "target_profile": profile.to_json(),
        }
        stats["within_20_percent"] = _profile_all_within(stats, profile)
        return stats


class _BurstStatsAccumulator:
    def __init__(self) -> None:
        self.record_count = 0
        self.records_with_bursts = 0
        self.burst_lengths: list[int] = []
        self.iid_drop_count = 0
        self.dropped_frame_count = 0

    def add(self, noisy: Mapping[str, Any]) -> None:
        self.record_count += 1
        model = noisy.get("dropout_model") or {}
        lengths = [int(value) for value in model.get("burst_lengths") or []]
        if lengths:
            self.records_with_bursts += 1
            self.burst_lengths.extend(lengths)
        self.iid_drop_count += int(model.get("iid_drop_count") or 0)
        self.dropped_frame_count += int(model.get("dropped_frame_count") or 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "record_count": int(self.record_count),
            "records_with_bursts": int(self.records_with_bursts),
            "burst_count": len(self.burst_lengths),
            "burst_length_min": min(self.burst_lengths) if self.burst_lengths else None,
            "burst_length_p50": _percentile_int(self.burst_lengths, 50),
            "burst_length_p95": _percentile_int(self.burst_lengths, 95),
            "burst_length_max": max(self.burst_lengths) if self.burst_lengths else None,
            "iid_drop_count": int(self.iid_drop_count),
            "dropped_frame_count": int(self.dropped_frame_count),
        }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    profile = DetectorNoiseProfile(
        p95_jitter_px=float(args.jitter_p95_px),
        recall=float(args.recall),
        hidden_fp_rate=float(args.hidden_fp_rate),
    )
    _validate_profile(profile)
    burst_config = BurstDropoutConfig(
        enabled=not args.disable_burst_dropout,
        burst_dropout_share=float(args.burst_dropout_share),
        min_frames=int(args.burst_min_frames),
        max_frames=int(args.burst_max_frames),
    )
    burst_config.validate()
    calibration = load_court_calibration(args.calibration)
    report = generate_phase2_corpus(
        count=args.count,
        seed=args.seed,
        calibration=calibration,
        out=args.out,
        noise_profile=profile,
        burst_config=burst_config,
        roundtrip_samples=args.roundtrip_samples,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_flight_corpus_generation",
                "count": args.count,
                "seed": args.seed,
                "jsonl": str(args.out),
                "report": str(args.report) if args.report is not None else None,
                "acceptance": report["acceptance"],
                "error_profile_match": report["error_profile_match"],
                "round_trip": report["round_trip"]["position_error_m"],
                "performance": report["performance"],
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic pickleball 2D<->3D flight pairs as JSONL.")
    parser.add_argument("--count", type=int, required=True, help="Number of trajectories to emit.")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic numpy RNG seed.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="court_calibration_metric15pt.json used for camera projection.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--report", type=Path, help="Optional aggregate report JSON path.")
    parser.add_argument(
        "--roundtrip-samples",
        type=int,
        default=10,
        help="Number of clean trajectories to fit with ball_arc_solver for the round-trip report.",
    )
    parser.add_argument(
        "--jitter-p95-px",
        type=float,
        default=MEASURED_ERROR_PROFILE.p95_jitter_px,
        help="Detector true-positive radial jitter p95 in pixels.",
    )
    parser.add_argument(
        "--recall",
        type=float,
        default=MEASURED_ERROR_PROFILE.recall,
        help="Target detector recall on visible clean frames.",
    )
    parser.add_argument(
        "--hidden-fp-rate",
        type=float,
        default=MEASURED_ERROR_PROFILE.hidden_fp_rate,
        help="Target hidden false-positive detections per generated frame.",
    )
    parser.add_argument(
        "--disable-burst-dropout",
        action="store_true",
        help="Use iid missed detections only; default keeps contiguous occlusion bursts enabled.",
    )
    parser.add_argument(
        "--burst-dropout-share",
        type=float,
        default=0.70,
        help="Share of missed visible frames assigned to contiguous occlusion bursts.",
    )
    parser.add_argument(
        "--burst-min-frames",
        type=int,
        default=3,
        help="Minimum contiguous missed-frame burst length.",
    )
    parser.add_argument(
        "--burst-max-frames",
        type=int,
        default=12,
        help="Maximum contiguous missed-frame burst length.",
    )
    return parser


def generate_phase2_corpus(
    *,
    count: int,
    seed: int,
    calibration: Any,
    out: Path,
    noise_profile: DetectorNoiseProfile,
    burst_config: BurstDropoutConfig,
    roundtrip_samples: int,
) -> dict[str, Any]:
    if count < 0:
        raise ValueError("count must be non-negative")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    noise_stats = _NoiseStatsAccumulator()
    burst_stats = _BurstStatsAccumulator()
    roundtrip_reports: list[dict[str, Any]] = []
    failed_segments = 0
    demoted_frames = 0
    record_sanity_pass_count = 0
    shot_family_counts: dict[str, int] = {}
    bounced_records = 0
    calibration_payload = _embedded_calibration_payload(calibration)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for idx in range(count):
            record = generate_trajectory_pair(
                trajectory_id=f"sim_{idx:06d}",
                rng=rng,
                calibration=calibration,
                noise_profile=noise_profile,
                clean_only=True,
            )
            noisy = apply_burst_detector_noise(
                record["clean_2d_track"],
                rng=rng,
                image_size=tuple(record["projection"]["image_size"]),
                profile=noise_profile,
                burst_config=burst_config,
            )
            record["noisy_2d_detections"] = noisy
            _embed_calibration(record, calibration_payload)
            has_bounce = _tag_bounce_params(record)
            if has_bounce:
                bounced_records += 1

            family = str(record["truth_3d"]["shot"]["family"])
            shot_family_counts[family] = shot_family_counts.get(family, 0) + 1

            sanity = evaluate_simulated_flight_sanity(record)
            failed = int(sanity["summary"]["failed_segment_count"])
            demoted = int(sanity["summary"]["demoted_frame_count"])
            failed_segments += failed
            demoted_frames += demoted
            if failed == 0 and demoted == 0:
                record_sanity_pass_count += 1

            noise_stats.add(record["clean_2d_track"], noisy)
            burst_stats.add(noisy)
            if len(roundtrip_reports) < max(0, roundtrip_samples):
                roundtrip_reports.append(round_trip_fit_report(record, calibration))

            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    elapsed = time.perf_counter() - started
    noise_summary = noise_stats.to_stats(noise_profile)
    error_profile_match = _error_profile_match_from_stats(noise_summary, noise_profile)
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_flight_corpus_report",
        "deterministic_seed": int(seed),
        "trajectory_count": int(count),
        "calibration_path": str(calibration.path or ""),
        "output_jsonl": str(out),
        "acceptance": {
            "flight_sanity": {
                "failed_segments": int(failed_segments),
                "demoted_frames": int(demoted_frames),
                "record_pass_count": int(record_sanity_pass_count),
                "record_pass_fraction": _round(record_sanity_pass_count / count if count else 0.0, 9),
                "passed": failed_segments == 0 and demoted_frames == 0,
            },
            "noise_profile": noise_summary,
        },
        "error_profile_match": error_profile_match,
        "round_trip": {
            "samples_evaluated": len(roundtrip_reports),
            "reports": roundtrip_reports,
            "position_error_m": _combine_roundtrip_errors(roundtrip_reports),
        },
        "performance": {
            "trajectory_count": int(count),
            "wall_seconds": _round(elapsed, 6),
            "trajectories_per_second": _round(count / elapsed if elapsed > 0.0 else 0.0, 6),
        },
        "phase2": {
            "generator": "streaming_cli_wrapper_over_flight_simulator",
            "simulator_imports": {
                "DetectorNoiseProfile": True,
                "sample_shot_family": sample_shot_family is not None,
                "generate_trajectory_pair": True,
                "generate_corpus": generate_corpus is not None,
            },
            "detector_noise_profile": noise_profile.to_json(),
            "burst_dropout": {
                "config": burst_config.to_json(),
                "summary": burst_stats.to_json(),
            },
            "calibration_embedded": True,
            "bounce_params_measured_false_records": int(bounced_records),
            "shot_family_coverage": {
                "counts": dict(sorted(shot_family_counts.items())),
                "required_families": ["serve", "drive", "dink", "lob"],
                "all_required_present": all(shot_family_counts.get(family, 0) > 0 for family in ("serve", "drive", "dink", "lob")),
                "band_status": "plausible_unmeasured_prior",
            },
        },
    }
    return report


def apply_burst_detector_noise(
    clean_track: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    profile: DetectorNoiseProfile,
    burst_config: BurstDropoutConfig,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    local_rng = rng if rng is not None else np.random.default_rng(seed)
    burst_config.validate()
    visible_indices = [idx for idx, frame in enumerate(clean_track) if bool(frame.get("visible"))]
    target_tp = int(round(float(profile.recall) * len(visible_indices)))
    target_tp = max(0, min(len(visible_indices), target_tp))
    target_drop = len(visible_indices) - target_tp
    dropped_indices, burst_runs, iid_drop_count = _select_dropout_indices(
        visible_indices=visible_indices,
        target_drop=target_drop,
        rng=local_rng,
        clean_track=clean_track,
        burst_config=burst_config,
    )

    selected_indices = [idx for idx in visible_indices if idx not in dropped_indices]
    raw_jitter = (
        local_rng.normal(0.0, 1.0, size=(len(selected_indices), 2))
        if selected_indices
        else np.zeros((0, 2))
    )
    radial = np.linalg.norm(raw_jitter, axis=1) if len(selected_indices) else np.asarray([])
    p95 = float(np.percentile(radial, 95)) if radial.size else 1.0
    scale = float(profile.p95_jitter_px) / p95 if p95 > 1e-12 else 0.0
    jitter = raw_jitter * scale

    detections: list[dict[str, Any]] = []
    for jitter_index, clean_index in enumerate(selected_indices):
        frame = clean_track[clean_index]
        xy = [float(value) for value in frame["xy_px"]]
        offset = jitter[jitter_index]
        detections.append(
            {
                "frame": int(frame["frame"]),
                "t": float(frame["t"]),
                "xy_px": [_round(xy[0] + float(offset[0]), 6), _round(xy[1] + float(offset[1]), 6)],
                "confidence": _round(float(local_rng.uniform(0.35, 0.99)), 6),
                "kind": "true_positive",
                "matched_clean_frame": int(frame["frame"]),
            }
        )

    width, height = image_size
    fp_count = max(0, min(len(clean_track), int(round(float(profile.hidden_fp_rate) * len(clean_track)))))
    fp_indices = (
        local_rng.choice(len(clean_track), size=fp_count, replace=False)
        if fp_count and len(clean_track)
        else []
    )
    spurious: list[dict[str, Any]] = []
    for raw_index in sorted(int(value) for value in fp_indices):
        frame = clean_track[raw_index]
        detection = {
            "frame": int(frame["frame"]),
            "t": float(frame["t"]),
            "xy_px": [
                _round(float(local_rng.uniform(0.0, width)), 6),
                _round(float(local_rng.uniform(0.0, height)), 6),
            ],
            "confidence": _round(float(local_rng.uniform(0.20, 0.80)), 6),
            "kind": "hidden_false_positive",
            "matched_clean_frame": None,
        }
        spurious.append(detection)
        detections.append(detection)

    dropped_frames = sorted(int(clean_track[idx]["frame"]) for idx in dropped_indices)
    detections.sort(key=lambda item: (int(item["frame"]), str(item["kind"])))
    model = {
        "type": "occlusion_burst_plus_iid_fill" if burst_config.enabled else "iid_only",
        "config": burst_config.to_json(),
        "burst_runs": burst_runs,
        "burst_lengths": [int(run["length"]) for run in burst_runs],
        "iid_drop_count": int(iid_drop_count),
        "visible_clean_frames": len(visible_indices),
        "true_positive_count": len(selected_indices),
        "dropped_frame_count": len(dropped_frames),
        "recall": _round(len(selected_indices) / len(visible_indices) if visible_indices else 0.0, 9),
        "hidden_fp_count": len(spurious),
        "hidden_fp_rate": _round(len(spurious) / len(clean_track) if clean_track else 0.0, 9),
        "jitter_p95_px": _round(float(np.percentile(np.linalg.norm(jitter, axis=1), 95)) if len(jitter) else 0.0, 6),
    }
    return {
        "profile": profile.to_json(),
        "detections": detections,
        "dropped_frames": dropped_frames,
        "spurious_detections": spurious,
        "dropout_model": model,
    }


def build_error_profile_match(
    clean_tracks: Sequence[Sequence[Mapping[str, Any]]],
    noisy_tracks: Sequence[Mapping[str, Any]],
    profile: DetectorNoiseProfile,
) -> dict[str, Any]:
    accumulator = _NoiseStatsAccumulator()
    for clean, noisy in zip(clean_tracks, noisy_tracks, strict=True):
        accumulator.add(clean, noisy)
    return _error_profile_match_from_stats(accumulator.to_stats(profile), profile)


def _select_dropout_indices(
    *,
    visible_indices: Sequence[int],
    target_drop: int,
    rng: np.random.Generator,
    clean_track: Sequence[Mapping[str, Any]],
    burst_config: BurstDropoutConfig,
) -> tuple[set[int], list[dict[str, Any]], int]:
    if target_drop <= 0:
        return set(), [], 0
    dropped_positions: set[int] = set()
    burst_runs: list[dict[str, Any]] = []
    burst_budget = 0
    if burst_config.enabled and target_drop >= burst_config.min_frames:
        burst_budget = min(target_drop, int(round(target_drop * burst_config.burst_dropout_share)))
        if 0 < burst_budget < burst_config.min_frames:
            burst_budget = min(target_drop, burst_config.min_frames)

    attempts = 0
    max_attempts = max(50, target_drop * 20)
    while len(dropped_positions) < burst_budget and attempts < max_attempts:
        attempts += 1
        remaining = burst_budget - len(dropped_positions)
        if remaining < burst_config.min_frames:
            break
        max_len = min(burst_config.max_frames, remaining, len(visible_indices))
        if max_len < burst_config.min_frames:
            break
        length = int(rng.integers(burst_config.min_frames, max_len + 1))
        starts = _candidate_burst_starts(len(visible_indices), length, dropped_positions)
        if not starts:
            break
        start_position = starts[int(rng.integers(0, len(starts)))]
        positions = list(range(start_position, start_position + length))
        dropped_positions.update(positions)
        frames = [int(clean_track[visible_indices[position]]["frame"]) for position in positions]
        burst_runs.append(
            {
                "start_frame": min(frames),
                "end_frame": max(frames),
                "length": len(frames),
            }
        )

    remaining_positions = [
        position for position in range(len(visible_indices)) if position not in dropped_positions
    ]
    remaining_drop = target_drop - len(dropped_positions)
    iid_drop_count = 0
    if remaining_drop > 0 and remaining_positions:
        fill_positions = _non_adjacent_positions(remaining_positions, dropped_positions)
        if len(fill_positions) < remaining_drop:
            fill_positions = remaining_positions
        selected = rng.choice(fill_positions, size=min(remaining_drop, len(fill_positions)), replace=False)
        for position in selected:
            dropped_positions.add(int(position))
            iid_drop_count += 1

    dropped_indices = {int(visible_indices[position]) for position in dropped_positions}
    return dropped_indices, sorted(burst_runs, key=lambda item: (item["start_frame"], item["end_frame"])), iid_drop_count


def _candidate_burst_starts(total_positions: int, length: int, dropped_positions: set[int]) -> list[int]:
    starts: list[int] = []
    for start in range(0, total_positions - length + 1):
        padded_start = max(0, start - 1)
        padded_end = min(total_positions, start + length + 1)
        if any(position in dropped_positions for position in range(padded_start, padded_end)):
            continue
        starts.append(start)
    return starts


def _non_adjacent_positions(positions: Sequence[int], dropped_positions: set[int]) -> list[int]:
    return [
        int(position)
        for position in positions
        if int(position) - 1 not in dropped_positions and int(position) + 1 not in dropped_positions
    ]


def _embedded_calibration_payload(calibration: Any) -> dict[str, Any]:
    payload = dict(calibration.payload)
    return {
        "schema": "CourtCalibration",
        "schema_version": payload.get("schema_version"),
        "image_size": list(payload.get("image_size") or _image_size_from_model(calibration.model)),
        "intrinsics": payload.get("intrinsics"),
        "extrinsics": payload.get("extrinsics"),
        "coordinate_frame": payload.get("coordinate_frame"),
        "T_world_court": payload.get("T_world_court"),
        "sport": payload.get("sport"),
    }


def _embed_calibration(record: dict[str, Any], calibration_payload: Mapping[str, Any]) -> None:
    projection = dict(record.get("projection") or {})
    projection["calibration"] = dict(calibration_payload)
    projection["calibration_embedded"] = True
    record["projection"] = projection


def _tag_bounce_params(record: dict[str, Any]) -> bool:
    bounces = list(record["truth_3d"].get("bounces") or [])
    has_bounce = bool(bounces)
    if has_bounce:
        tagged = []
        for bounce in bounces:
            item = dict(bounce)
            item["bounce_params_measured"] = False
            tagged.append(item)
        record["truth_3d"]["bounces"] = tagged
        record["bounce_params_measured"] = False
    else:
        record["bounce_params_measured"] = None
    return has_bounce


def _error_profile_match_from_stats(stats: Mapping[str, Any], profile: DetectorNoiseProfile) -> dict[str, Any]:
    measured = {
        "jitter_p95_px": float(profile.p95_jitter_px),
        "recall": float(profile.recall),
        "hidden_fp_rate": float(profile.hidden_fp_rate),
    }
    metrics: dict[str, dict[str, Any]] = {}
    for key, target in (
        ("jitter_p95_px", profile.p95_jitter_px),
        ("recall", profile.recall),
        ("hidden_fp_rate", profile.hidden_fp_rate),
    ):
        generated = float(stats[key])
        relative_error = None if target == 0.0 else abs(generated - target) / abs(target)
        metrics[key] = {
            "measured": float(measured[key]),
            "generated": _round(generated, 9),
            "relative_error": _round(relative_error, 9) if relative_error is not None else None,
            "within_20_percent": _within_20_percent(generated, target),
        }
    return {
        "measured": measured,
        "generated": {
            "jitter_p95_px": float(stats["jitter_p95_px"]),
            "recall": float(stats["recall"]),
            "hidden_fp_rate": float(stats["hidden_fp_rate"]),
        },
        "metrics": metrics,
        "all_within_20_percent": all(item["within_20_percent"] for item in metrics.values()),
    }


def _combine_roundtrip_errors(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p95s = [
        float(report["position_error_m"]["p95"])
        for report in reports
        if isinstance(report.get("position_error_m"), Mapping)
        and report["position_error_m"].get("p95") is not None
    ]
    return _error_summary(p95s)


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": _round(float(np.mean(finite)), 6),
        "p50": _round(float(np.percentile(finite, 50)), 6),
        "p95": _round(float(np.percentile(finite, 95)), 6),
        "max": _round(max(finite), 6),
    }


def _validate_profile(profile: DetectorNoiseProfile) -> None:
    if profile.p95_jitter_px < 0.0:
        raise ValueError("jitter p95 must be non-negative")
    if not 0.0 <= profile.recall <= 1.0:
        raise ValueError("recall must be in [0, 1]")
    if profile.hidden_fp_rate < 0.0:
        raise ValueError("hidden FP rate must be non-negative")


def _profile_all_within(stats: Mapping[str, Any], profile: DetectorNoiseProfile) -> bool:
    return all(
        (
            _within_20_percent(float(stats["jitter_p95_px"]), profile.p95_jitter_px),
            _within_20_percent(float(stats["recall"]), profile.recall),
            _within_20_percent(float(stats["hidden_fp_rate"]), profile.hidden_fp_rate),
        )
    )


def _within_20_percent(value: float, target: float) -> bool:
    if target == 0.0:
        return abs(value) <= 0.20
    return abs(value - target) <= abs(target) * 0.20


def _image_size_from_model(model: Any) -> list[int]:
    if model.image_size is not None:
        return [int(model.image_size[0]), int(model.image_size[1])]
    return [int(round(model.intrinsics.cx * 2.0)), int(round(model.intrinsics.cy * 2.0))]


def _percentile_int(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    return int(round(float(np.percentile([int(value) for value in values], percentile))))


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return float(value)
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 10 ** (-(digits + 1)) else rounded


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
