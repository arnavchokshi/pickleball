#!/usr/bin/env python3
"""Build the blind, source-disjoint owner event-label review session.

This is a data-channel tool. It does not promote a model or consume protected
evaluation video. The protected 50-row seed is read only as an exclusion list
and as a timing-normalization regression check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GENERATOR_VERSION = "event_review_session_v1_20260715"
PAGE_GENERATOR_VERSION = "event_review_page_v1_20260715"
SESSION_ID = "event_labels_20260715"
CREATED_AT = "2026-07-15T00:00:00Z"  # Fixed so same inputs + seed are byte-identical.
SOURCES = (
    "73VurrTKCZ8",
    "Ezz6HDNHlnk",
    "HyUqT7zFiwk",
    "_L0HVmAlCQI",
    "wBu8bC4OfUY",
    "zwCtH_i1_S4",
)
TARGETS = {"audio_onset": 120, "track_discontinuity": 75, "uniform_random": 105}
TRAIN_SOURCES = {"73VurrTKCZ8", "Ezz6HDNHlnk", "_L0HVmAlCQI", "wBu8bC4OfUY"}
VALIDATION_SOURCES = {"HyUqT7zFiwk", "zwCtH_i1_S4"}
ANCHOR_MARGIN_S = 0.7
PROTECTED_SEED_RADIUS_S = 0.75
SIGNAL_MIN_SEPARATION_S = 0.8
UNIFORM_MIN_SEPARATION_S = 1.3
CROSS_STRATUM_RADIUS_S = 0.3
TRACK_CONFIDENCE_MIN = 0.5
TRACK_DIRECTION_CHANGE_DEG = 60.0
TRACK_SPEED_RATIO = 1.8
TRACK_MIN_SPEED_PX_S = 30.0
TRACK_GAP_MIN_FRAMES = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_int(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _rank(seed: int, stratum: str, source: str, clip_id: str, identity: object) -> int:
    return _stable_int(seed, stratum, source, clip_id, identity)


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _probe_video(path: Path) -> dict[str, Any]:
    payload = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = payload["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"invalid frame rate for {path}: {stream['avg_frame_rate']}")
    return {
        "duration_s": float(payload["format"]["duration"]),
        "fps": numerator / denominator,
        "fps_rational": stream["avg_frame_rate"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def _discover_universe(root: Path) -> list[dict[str, Any]]:
    rally_root = (root / "data/online_harvest_20260706/rallies").resolve()
    videos = sorted(rally_root.glob("*/*.mp4"))
    if len(videos) != 40:
        raise ValueError(f"expected exactly 40 rally clips, found {len(videos)}")
    actual_sources = {path.parent.name for path in videos}
    if actual_sources != set(SOURCES):
        raise ValueError(f"unexpected source universe: {sorted(actual_sources)}")
    frames_only_rallies = list((root / "data/online_harvest_20260712").glob("**/rallies/*.mp4"))
    if frames_only_rallies:
        raise ValueError("20260712 frames-only harvest unexpectedly contains rally videos")

    clips: list[dict[str, Any]] = []
    for path in videos:
        resolved = path.resolve()
        if not resolved.is_relative_to(rally_root):
            raise ValueError(f"E3 violation: {resolved} is outside the rally universe")
        lowered = resolved.as_posix().lower()
        if "/eval_clips/" in lowered or "/data/testclips/" in lowered:
            raise ValueError(f"E3 violation: protected/test video entered universe: {resolved}")
        probe = _probe_video(resolved)
        clips.append(
            {
                "clip_id": path.stem,
                "source": path.parent.name,
                "path": resolved,
                "video_path": path.relative_to(root).as_posix(),
                "video_sha256": _sha256(resolved),
                **probe,
            }
        )
    return clips


def _protected_seed(root: Path) -> tuple[dict[str, list[float]], dict[str, Any]]:
    path = root / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    if len(labels) != 50:
        raise ValueError(f"protected seed must contain 50 labels, found {len(labels)}")
    exclusions: dict[str, list[float]] = defaultdict(list)
    deltas: list[float] = []
    within_one_frame = 0
    for row in labels:
        clip_id = str(row["source"]["clip_id"])
        anchor = float(row["anchor"]["pts_s"])
        corrected = float(row["evidence"]["audio"]["corrected_time_s"])
        exclusions[clip_id].append(anchor)
        video = root / row["source"]["video_path"]
        fps = _probe_video(video)["fps"]
        snapped = round(corrected * fps) / fps
        delta = abs(snapped - anchor)
        deltas.append(delta)
        if delta <= (1.0 / fps) + 1e-6:
            within_one_frame += 1
    median = sorted(deltas)[len(deltas) // 2]
    if within_one_frame < 45 or median > 0.25:
        raise ValueError(
            "audio anchor normalization sanity check failed: "
            f"{within_one_frame}/50 within one frame, median disagreement {median:.6f}s"
        )
    return dict(exclusions), {
        "rows": 50,
        "within_one_frame": within_one_frame,
        "required_within_one_frame": 45,
        "max_abs_error_s": round(max(deltas), 9),
        "median_abs_error_s": round(median, 9),
        "status": "pass",
        "method": "round(corrected_time_s * probed_avg_fps) / probed_avg_fps; normalized first video PTS is zero",
    }


def _excluded(clip_id: str, anchor: float, protected: dict[str, list[float]]) -> bool:
    return any(abs(anchor - value) <= PROTECTED_SEED_RADIUS_S for value in protected.get(clip_id, []))


def _bounded_allocation(target: int, duration_by_source: dict[str, float]) -> dict[str, int]:
    floor = 8
    cap = math.floor(target * 0.30 + 1e-12)
    if floor * len(SOURCES) > target or cap * len(SOURCES) < target:
        raise ValueError(f"infeasible allocation target={target}, floor={floor}, cap={cap}")
    allocation = {source: floor for source in SOURCES}
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [source for source in SOURCES if allocation[source] < cap]
        weights = sum(duration_by_source[source] for source in eligible)
        ideals = {
            source: remaining * duration_by_source[source] / weights
            for source in eligible
        }
        additions = {
            source: min(cap - allocation[source], math.floor(ideals[source]))
            for source in eligible
        }
        placed = sum(additions.values())
        for source, count in additions.items():
            allocation[source] += count
        remaining -= placed
        if remaining == 0:
            break
        # Largest remainder, deterministic source-name tie break. The largest
        # source receives a true deficit only when all fractional remainders tie.
        ordered = sorted(
            (source for source in eligible if allocation[source] < cap),
            key=lambda source: (-(ideals[source] - math.floor(ideals[source])), -duration_by_source[source], source),
        )
        if not ordered:
            raise ValueError("allocation cap exhausted before target")
        for source in ordered:
            if remaining == 0:
                break
            if allocation[source] < cap:
                allocation[source] += 1
                remaining -= 1
    if sum(allocation.values()) != target:
        raise AssertionError("allocation arithmetic failed")
    return allocation


def _assign_bands(candidates: list[dict[str, Any]], score_key: str) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate["source"]].append(candidate)
    for source_candidates in by_source.values():
        ordered = sorted(
            source_candidates,
            key=lambda row: (float(row[score_key]), row["clip_id"], float(row["anchor_pts_s"])),
        )
        total = len(ordered)
        for index, candidate in enumerate(ordered):
            third = min(2, (index * 3) // max(1, total))
            candidate["score_band"] = ("low", "mid", "high")[third]


def _audio_candidates(
    root: Path,
    clips: list[dict[str, Any]],
    protected: dict[str, list[float]],
    exclusion_counts: Counter[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for clip in clips:
        onset_path = root / "data/event_bootstrap_20260713/audio_onsets_v0" / f"{clip['clip_id']}.json"
        if "pbvision" in onset_path.name.lower():
            raise AssertionError("E2: pbvision onset path constructed for rally clip")
        payload = json.loads(onset_path.read_text(encoding="utf-8"))
        for index, onset in enumerate(payload.get("onsets", [])):
            corrected = float(onset["corrected_time_s"])
            frame = int(round(corrected * clip["fps"]))
            anchor = frame / clip["fps"]
            if not (ANCHOR_MARGIN_S <= anchor <= clip["duration_s"] - ANCHOR_MARGIN_S):
                exclusion_counts["E4_audio_bounds_avoided"] += 1
                continue
            if _excluded(clip["clip_id"], anchor, protected):
                exclusion_counts["E1_audio_hits_avoided"] += 1
                continue
            candidates.append(
                {
                    "stratum": "audio_onset",
                    "source": clip["source"],
                    "clip_id": clip["clip_id"],
                    "anchor_pts_s": anchor,
                    "anchor_frame": frame,
                    "score": float(onset["score"]),
                    "signal_features": {
                        "audio_score": float(onset["score"]),
                        "onset_strength": float(onset.get("onset_strength", 0.0)),
                        "corrected_time_s": corrected,
                        "candidate_index": index,
                    },
                }
            )
    _assign_bands(candidates, "score")
    return candidates


def _angle_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def _track_candidates(
    root: Path,
    clips: list[dict[str, Any]],
    protected: dict[str, list[float]],
    exclusion_counts: Counter[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for clip in clips:
        payload = json.loads(
            (root / "data/online_harvest_20260706/prelabels" / clip["clip_id"] / "ball_track.json").read_text(
                encoding="utf-8"
            )
        )
        frames = payload["frames"]
        valid = [bool(row.get("visible")) and float(row.get("conf", 0.0)) >= TRACK_CONFIDENCE_MIN for row in frames]
        by_index: dict[int, dict[str, Any]] = {}

        def add(index: int, features: dict[str, Any], strength: float) -> None:
            row = frames[index]
            anchor = float(row["t"])
            if not (ANCHOR_MARGIN_S <= anchor <= clip["duration_s"] - ANCHOR_MARGIN_S):
                exclusion_counts["E4_track_bounds_avoided"] += 1
                return
            if _excluded(clip["clip_id"], anchor, protected):
                exclusion_counts["E1_track_hits_avoided"] += 1
                return
            current = by_index.get(index)
            if current is None or strength > current["strength"]:
                by_index[index] = {
                    "stratum": "track_discontinuity",
                    "source": clip["source"],
                    "clip_id": clip["clip_id"],
                    "anchor_pts_s": anchor,
                    "anchor_frame": index,
                    "strength": strength,
                    "signal_features": features,
                }

        for index in range(1, len(frames) - 1):
            if not (valid[index - 1] and valid[index] and valid[index + 1]):
                continue
            before_dt = float(frames[index]["t"]) - float(frames[index - 1]["t"])
            after_dt = float(frames[index + 1]["t"]) - float(frames[index]["t"])
            if before_dt <= 0 or after_dt <= 0:
                continue
            before = (
                (float(frames[index]["xy"][0]) - float(frames[index - 1]["xy"][0])) / before_dt,
                (float(frames[index]["xy"][1]) - float(frames[index - 1]["xy"][1])) / before_dt,
            )
            after = (
                (float(frames[index + 1]["xy"][0]) - float(frames[index]["xy"][0])) / after_dt,
                (float(frames[index + 1]["xy"][1]) - float(frames[index]["xy"][1])) / after_dt,
            )
            speed_before, speed_after = math.hypot(*before), math.hypot(*after)
            angle = _angle_degrees(before, after)
            ratio = speed_after / max(speed_before, 1e-9)
            direction_flag = angle >= TRACK_DIRECTION_CHANGE_DEG and min(speed_before, speed_after) >= TRACK_MIN_SPEED_PX_S
            ratio_flag = (
                min(speed_before, speed_after) >= TRACK_MIN_SPEED_PX_S
                and (ratio >= TRACK_SPEED_RATIO or ratio <= 1.0 / TRACK_SPEED_RATIO)
            )
            if not (direction_flag or ratio_flag):
                continue
            direction_strength = angle / TRACK_DIRECTION_CHANGE_DEG if direction_flag else 0.0
            ratio_strength = abs(math.log(max(ratio, 1e-9))) / math.log(TRACK_SPEED_RATIO) if ratio_flag else 0.0
            add(
                index,
                {
                    "direction_change_deg": round(angle, 6),
                    "speed_before_px_s": round(speed_before, 6),
                    "speed_after_px_s": round(speed_after, 6),
                    "speed_ratio": round(ratio, 6),
                    "direction_flag": direction_flag,
                    "speed_ratio_flag": ratio_flag,
                    "visibility_gap_boundary": False,
                    "confidence": float(frames[index]["conf"]),
                },
                max(direction_strength, ratio_strength),
            )

        index = 0
        while index < len(frames):
            if valid[index]:
                index += 1
                continue
            start = index
            while index < len(frames) and not valid[index]:
                index += 1
            gap = index - start
            if gap < TRACK_GAP_MIN_FRAMES:
                continue
            for boundary in (start - 1, index):
                if 0 <= boundary < len(frames) and valid[boundary]:
                    add(
                        boundary,
                        {
                            "direction_change_deg": None,
                            "speed_before_px_s": None,
                            "speed_after_px_s": None,
                            "speed_ratio": None,
                            "direction_flag": False,
                            "speed_ratio_flag": False,
                            "visibility_gap_boundary": True,
                            "visibility_gap_frames": gap,
                            "confidence": float(frames[boundary]["conf"]),
                        },
                        gap / TRACK_GAP_MIN_FRAMES,
                    )
        candidates.extend(by_index.values())
    _assign_bands(candidates, "strength")
    return candidates


def _band_targets(quota: int) -> dict[str, int]:
    base, remainder = divmod(quota, 3)
    return {band: base + (1 if index < remainder else 0) for index, band in enumerate(("low", "mid", "high"))}


def _select_signal(
    candidates: list[dict[str, Any]],
    allocation: dict[str, int],
    *,
    seed: int,
    stratum: str,
    protected: dict[str, list[float]],
    audio_selected: list[dict[str, Any]] | None,
    exclusion_counts: Counter[str],
    backfills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_times: dict[str, list[float]] = defaultdict(list)
    audio_times: dict[str, list[float]] = defaultdict(list)
    for row in audio_selected or []:
        audio_times[row["clip_id"]].append(float(row["anchor_pts_s"]))

    available_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        available_by_source[candidate["source"]].append(candidate)
    used: set[tuple[str, float]] = set()

    def eligible(candidate: dict[str, Any]) -> bool:
        key = (candidate["clip_id"], float(candidate["anchor_pts_s"]))
        if key in used:
            return False
        if any(abs(float(candidate["anchor_pts_s"]) - value) < SIGNAL_MIN_SEPARATION_S for value in selected_times[candidate["clip_id"]]):
            return False
        if stratum == "track_discontinuity" and any(
            abs(float(candidate["anchor_pts_s"]) - value) <= CROSS_STRATUM_RADIUS_S
            for value in audio_times[candidate["clip_id"]]
        ):
            exclusion_counts["audio_track_collisions_avoided"] += 1
            used.add(key)
            return False
        if _excluded(candidate["clip_id"], float(candidate["anchor_pts_s"]), protected):
            raise AssertionError("E1 candidate reached selector")
        return True

    deficits: dict[str, int] = {}
    for source in SOURCES:
        quota = allocation[source]
        band_targets = _band_targets(quota)
        source_rows = available_by_source[source]
        source_selected = 0
        for band in ("low", "mid", "high"):
            ordered = sorted(
                (row for row in source_rows if row["score_band"] == band),
                key=lambda row: _rank(seed, stratum, source, row["clip_id"], row["anchor_pts_s"]),
            )
            taken = 0
            for row in ordered:
                if taken >= band_targets[band]:
                    break
                if eligible(row):
                    selected.append(row)
                    used.add((row["clip_id"], float(row["anchor_pts_s"])))
                    selected_times[row["clip_id"]].append(float(row["anchor_pts_s"]))
                    taken += 1
                    source_selected += 1
            if taken < band_targets[band]:
                backfills.append(
                    {
                        "stratum": stratum,
                        "source": source,
                        "kind": "band_shortfall",
                        "score_band": band,
                        "requested": band_targets[band],
                        "selected": taken,
                    }
                )
        if source_selected < quota:
            ordered = sorted(
                source_rows,
                key=lambda row: _rank(seed, stratum, source, row["clip_id"], f"backfill:{row['anchor_pts_s']}"),
            )
            before = source_selected
            for row in ordered:
                if source_selected >= quota:
                    break
                if eligible(row):
                    selected.append(row)
                    used.add((row["clip_id"], float(row["anchor_pts_s"])))
                    selected_times[row["clip_id"]].append(float(row["anchor_pts_s"]))
                    source_selected += 1
            if source_selected > before:
                backfills.append(
                    {
                        "stratum": stratum,
                        "source": source,
                        "kind": "same_source_other_band_or_clip",
                        "count": source_selected - before,
                    }
                )
        deficits[source] = quota - source_selected

    total_deficit = sum(deficits.values())
    if total_deficit:
        ordered_sources = sorted(SOURCES, key=lambda source: (-len(available_by_source[source]), source))
        redistributed: Counter[str] = Counter()
        for source in ordered_sources:
            ordered = sorted(
                available_by_source[source],
                key=lambda row: _rank(seed, stratum, source, row["clip_id"], f"redistribute:{row['anchor_pts_s']}"),
            )
            for row in ordered:
                if total_deficit == 0:
                    break
                if eligible(row):
                    selected.append(row)
                    used.add((row["clip_id"], float(row["anchor_pts_s"])))
                    selected_times[row["clip_id"]].append(float(row["anchor_pts_s"]))
                    redistributed[source] += 1
                    total_deficit -= 1
            if total_deficit == 0:
                break
        for source, count in sorted(redistributed.items()):
            backfills.append({"stratum": stratum, "source": source, "kind": "cross_source_redistribution", "count": count})
    if total_deficit:
        backfills.append({"stratum": stratum, "kind": "unfilled_final_shortfall", "count": total_deficit})
    return selected


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    weight_total = sum(weights.values())
    ideals = {key: total * value / weight_total for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in ideals.items()}
    remainder = total - sum(result.values())
    for key in sorted(weights, key=lambda item: (-(ideals[item] - result[item]), item))[:remainder]:
        result[key] += 1
    return result


def _uniform_candidates(
    clips: list[dict[str, Any]],
    allocation: dict[str, int],
    *,
    seed: int,
    protected: dict[str, list[float]],
    exclusion_counts: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        source_clips = [clip for clip in clips if clip["source"] == source]
        per_clip = _largest_remainder(allocation[source], {clip["clip_id"]: clip["duration_s"] for clip in source_clips})
        for clip in source_clips:
            need = per_clip[clip["clip_id"]]
            rng = random.Random(_stable_int(seed, "uniform_random", source, clip["clip_id"]))
            anchors: list[float] = []
            attempts = 0
            while len(anchors) < need and attempts < max(1000, need * 1000):
                attempts += 1
                raw = rng.uniform(ANCHOR_MARGIN_S, clip["duration_s"] - ANCHOR_MARGIN_S)
                frame = int(round(raw * clip["fps"]))
                anchor = frame / clip["fps"]
                if not (ANCHOR_MARGIN_S <= anchor <= clip["duration_s"] - ANCHOR_MARGIN_S):
                    exclusion_counts["E4_uniform_bounds_avoided"] += 1
                    continue
                if _excluded(clip["clip_id"], anchor, protected):
                    exclusion_counts["E1_uniform_hits_avoided"] += 1
                    continue
                if any(abs(anchor - other) < UNIFORM_MIN_SEPARATION_S for other in anchors):
                    exclusion_counts["uniform_separation_collisions_avoided"] += 1
                    continue
                anchors.append(anchor)
                selected.append(
                    {
                        "stratum": "uniform_random",
                        "source": source,
                        "clip_id": clip["clip_id"],
                        "anchor_pts_s": anchor,
                        "anchor_frame": frame,
                        "score_band": None,
                        "signal_features": {"sampling": "uniform_over_valid_anchor_interval", "draw_attempt": attempts},
                    }
                )
            if len(anchors) != need:
                raise ValueError(f"unable to draw {need} separated uniform anchors for {clip['clip_id']}")
    return selected


def _validate_rows(rows: list[dict[str, Any]], clips_by_id: dict[str, dict[str, Any]], protected: dict[str, list[float]]) -> None:
    rally_prefix = "data/online_harvest_20260706/rallies/"
    for row in rows:
        clip = clips_by_id[row["clip_id"]]
        if _excluded(row["clip_id"], float(row["anchor_pts_s"]), protected):
            raise AssertionError(f"E1 violation: {row['clip_id']} at {row['anchor_pts_s']}")
        serialized = json.dumps(row, sort_keys=True).lower()
        if "pbvision_11min_20260713" in serialized or "pbvision" in row["clip_id"].lower():
            raise AssertionError("E2 violation")
        if not row["video_path"].startswith(rally_prefix) or "eval_clips/" in row["video_path"] or "data/testclips/" in row["video_path"]:
            raise AssertionError("E3 violation")
        if not (ANCHOR_MARGIN_S <= float(row["anchor_pts_s"]) <= clip["duration_s"] - ANCHOR_MARGIN_S):
            raise AssertionError("E4 violation")


def build_session(root: Path, *, seed: int = 20260715) -> dict[str, Any]:
    root = root.resolve()
    clips = _discover_universe(root)
    clips_by_id = {clip["clip_id"]: clip for clip in clips}
    protected, sanity = _protected_seed(root)
    exclusion_counts: Counter[str] = Counter()
    backfills: list[dict[str, Any]] = []
    durations = {source: sum(clip["duration_s"] for clip in clips if clip["source"] == source) for source in SOURCES}
    allocation = {stratum: _bounded_allocation(target, durations) for stratum, target in TARGETS.items()}

    audio_pool = _audio_candidates(root, clips, protected, exclusion_counts)
    audio = _select_signal(
        audio_pool,
        allocation["audio_onset"],
        seed=seed,
        stratum="audio_onset",
        protected=protected,
        audio_selected=None,
        exclusion_counts=exclusion_counts,
        backfills=backfills,
    )
    track_pool = _track_candidates(root, clips, protected, exclusion_counts)
    track = _select_signal(
        track_pool,
        allocation["track_discontinuity"],
        seed=seed,
        stratum="track_discontinuity",
        protected=protected,
        audio_selected=audio,
        exclusion_counts=exclusion_counts,
        backfills=backfills,
    )
    uniform = _uniform_candidates(
        clips,
        allocation["uniform_random"],
        seed=seed,
        protected=protected,
        exclusion_counts=exclusion_counts,
    )
    selected = audio + track + uniform
    if len(audio) != TARGETS["audio_onset"] or len(track) != TARGETS["track_discontinuity"] or len(uniform) != TARGETS["uniform_random"]:
        raise ValueError(f"stratum shortfall: audio={len(audio)}, track={len(track)}, uniform={len(uniform)}")

    presentation_rng = random.Random(_stable_int(seed, "presentation_shuffle"))
    presentation_rng.shuffle(selected)
    rows: list[dict[str, Any]] = []
    for row_number, candidate in enumerate(selected, start=1):
        clip = clips_by_id[candidate["clip_id"]]
        rows.append(
            {
                "label_id": f"els20260715_{row_number:03d}",
                "row": row_number,
                "stratum": candidate["stratum"],
                "score_band": candidate.get("score_band"),
                "signal_features": candidate["signal_features"],
                "clip_id": candidate["clip_id"],
                "source_group": candidate["source"],
                "video_path": clip["video_path"],
                "video_sha256": clip["video_sha256"],
                "source_fps": round(clip["fps"], 9),
                "anchor_pts_s": round(float(candidate["anchor_pts_s"]), 9),
                "anchor_frame": int(candidate["anchor_frame"]),
                "suggested_split": "train" if candidate["source"] in TRAIN_SOURCES else "validation",
            }
        )
    _validate_rows(rows, clips_by_id, protected)

    generator_path = Path(__file__).resolve()
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    actual_counts = {
        stratum: dict(sorted(Counter(row["source_group"] for row in rows if row["stratum"] == stratum).items()))
        for stratum in TARGETS
    }
    return {
        "session_id": SESSION_ID,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": _sha256(generator_path),
        "git_head": git_head,
        "created_at": CREATED_AT,
        "universe": {
            "description": "Exactly 40 rally MP4s under data/online_harvest_20260706/rallies across six declared source videos; protected eval, testclips, pbvision R&D, and the frames-only 20260712 harvest are outside the universe.",
            "video_root": "data/online_harvest_20260706/rallies",
            "clip_count": len(clips),
            "sources": list(SOURCES),
            "duration_s_by_source": {key: round(value, 6) for key, value in durations.items()},
            "online_harvest_20260712_rally_count": 0,
            "fps_by_clip": {clip["clip_id"]: clip["fps_rational"] for clip in clips},
        },
        "allocation_table": {
            stratum: {
                "target": TARGETS[stratum],
                "planned_by_source": allocation[stratum],
                "actual_by_source": actual_counts[stratum],
            }
            for stratum in TARGETS
        },
        "exclusion_counts": {
            **dict(sorted(exclusion_counts.items())),
            "protected_seed_rows": 50,
            "shortfalls": sum(int(item.get("count", 0)) for item in backfills if item["kind"] == "unfilled_final_shortfall"),
        },
        "backfills": backfills,
        "sampler_constants": {
            "anchor_margin_s": ANCHOR_MARGIN_S,
            "protected_seed_radius_s": PROTECTED_SEED_RADIUS_S,
            "signal_same_stratum_min_separation_s": SIGNAL_MIN_SEPARATION_S,
            "uniform_same_clip_min_separation_s": UNIFORM_MIN_SEPARATION_S,
            "audio_track_collision_radius_s": CROSS_STRATUM_RADIUS_S,
            "per_source_floor": 8,
            "per_source_cap_fraction": 0.30,
            "track_confidence_min": TRACK_CONFIDENCE_MIN,
            "track_direction_change_deg": TRACK_DIRECTION_CHANGE_DEG,
            "track_speed_ratio": TRACK_SPEED_RATIO,
            "track_min_speed_px_s": TRACK_MIN_SPEED_PX_S,
            "track_visibility_gap_min_frames": TRACK_GAP_MIN_FRAMES,
            "track_strength": "max(direction_change/60 when speed-qualified, abs(log(speed_ratio))/log(1.8), gap_frames/3)",
            "score_bands": "within-source rank terciles; draws balanced low/mid/high before documented backfill",
            "rng_substreams": "sha256(seed,stratum,source,clip,candidate); per-clip uniform PRNG; one separate presentation shuffle",
        },
        "anchor_sanity_check": sanity,
        "expected_owner_minutes": {
            "range": [75, 120],
            "basis": "300 rows; measured rich-review baseline about 24 seconds/row gives 120 minutes; 15 seconds/row is a faster floor allowing decision-only rows, giving 75 minutes.",
        },
        "source_split": {
            "unit": "original online video id, never extracted rally clip",
            "train_groups": sorted(TRAIN_SOURCES),
            "validation_groups": sorted(VALIDATION_SOURCES),
        },
        "verified": False,
        "honest_limits": [
            "VERIFIED=0 remains binding; this is an owner-review supply channel, not a model or dataset promotion.",
            "Signal strata are deliberately score-tercile balanced, not top-score selected; uniform_random remains signal-agnostic.",
            "The protected 50-row seed is used only for exclusion and timing regression, never as sampled training supply.",
        ],
        "rows": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _filename(row: dict[str, Any]) -> str:
    return f"{int(row['row']):03d}_pts{float(row['anchor_pts_s']):.3f}_{row['clip_id']}.mp4"


def _render_html(manifest: dict[str, Any]) -> str:
    items = [
        {
            "row": row["row"],
            "file": _filename(row),
            "label_id": row["label_id"],
            "anchor_pts_s": row["anchor_pts_s"],
            "source_fps": row["source_fps"],
        }
        for row in manifest["rows"]
    ]
    items_json = json.dumps(items, separators=(",", ":"))
    session_json = json.dumps(manifest["session_id"])
    page_version_json = json.dumps(PAGE_GENERATOR_VERSION)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contact review — {len(items)} clips</title>
<style>
:root{{--ink:#f6f2e8;--muted:#a9aaa3;--panel:#171915;--edge:#363a31;--green:#86b83e;--yellow:#f4d53f;--red:#d85c4a;--blue:#4aa0bb}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;background:radial-gradient(circle at 20% -10%,#2f3825 0,transparent 38%),#0d0f0c;color:var(--ink);font-family:Georgia,"Times New Roman",serif}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.18;background-image:repeating-linear-gradient(105deg,transparent 0 23px,#fff1 24px,transparent 25px)}}
.shell{{width:min(1120px,94vw);margin:auto;padding:18px 0 42px}}#progress{{font:700 14px ui-monospace,SFMono-Regular,monospace;letter-spacing:.13em;text-transform:uppercase;color:var(--yellow);margin:0 0 10px}}
.card{{background:#171915e8;border:1px solid var(--edge);border-radius:18px;box-shadow:0 24px 70px #0008;padding:clamp(16px,3vw,30px)}}h1{{font-size:clamp(32px,5vw,62px);line-height:.95;margin:8px 0 22px;font-weight:500}}p{{font-size:18px;line-height:1.55;color:var(--muted)}}.kicker{{font:700 12px ui-monospace,SFMono-Regular,monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--green)}}
#reviewScreen{{display:grid;gap:14px}}#vidwrap{{position:relative;background:#050605;border:1px solid var(--edge);border-radius:15px;overflow:hidden}}video{{width:100%;max-height:60vh;display:block;background:#000}}#marker{{position:absolute;width:28px;height:28px;margin:-14px 0 0 -14px;border:3px solid var(--yellow);border-radius:50%;box-shadow:0 0 0 2px #000,0 0 22px var(--yellow);display:none;pointer-events:none}}
#question{{font-size:clamp(22px,3vw,34px);text-align:center}}#hint{{font:15px ui-monospace,SFMono-Regular,monospace;text-align:center;color:var(--muted)}}.btns{{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}}button{{appearance:none;border:1px solid #ffffff24;border-radius:11px;background:#272b24;color:#fff;padding:13px 18px;font:700 15px ui-monospace,SFMono-Regular,monospace;cursor:pointer;box-shadow:0 5px 0 #070807;transition:transform .12s,filter .12s}}button:hover{{filter:brightness(1.16);transform:translateY(-1px)}}button:active{{transform:translateY(3px);box-shadow:0 2px 0 #070807}}button:disabled{{opacity:.35;cursor:default}}#paddle,#confirm{{background:#407f35}}#ground{{background:#276f82}}#otherhit{{background:#72523b}}#none{{background:#9b4035}}#unclear{{background:#51534f}}#start,#save{{background:var(--green);color:#10130e;font-size:20px;padding:17px 30px}}.small{{font-weight:600;color:#d4d5cd}}.hidden{{display:none!important}}.shortcut{{display:inline-block;border:1px solid #ffffff35;border-bottom-width:3px;border-radius:5px;padding:1px 6px;color:#fff;background:#242720}}
@media(max-width:640px){{.shell{{width:96vw;padding-top:8px}}.card{{border-radius:12px}}button{{padding:12px 10px;font-size:13px}}}}
</style></head><body><main class="shell"><div id="progress"></div>
<section id="startScreen" class="card"><div class="kicker">Owner review · blind session</div><h1>Find the exact touch.</h1><p>Each slow-motion clip opens paused at its center. Watch with sound, then choose what happened: paddle, ground, other (net/body), nothing, or unclear. For a contact, nudge to the exact source frame and click the ball.</p><p><span class="shortcut">1</span>–<span class="shortcut">5</span> choose in button order · <span class="shortcut">←</span>/<span class="shortcut">→</span> nudge one source frame · progress resumes automatically on this browser.</p><button id="start">Begin {len(items)} clips</button></section>
<section id="reviewScreen" class="card hidden"><div id="vidwrap"><video id="player" playsinline controls></video><div id="marker"></div></div><div id="question"></div><div id="hint"></div>
<div class="btns" id="phase1btns"><button id="paddle">1 · HIT PADDLE</button><button id="ground">2 · HIT GROUND</button><button id="otherhit">3 · HIT OTHER</button><button id="none">4 · NOTHING HIT</button><button id="unclear">5 · CAN'T TELL</button></div>
<div class="btns hidden" id="phase2btns"><button id="stepback" class="small">← source frame</button><button id="stepfwd" class="small">source frame →</button><button id="confirm" disabled>Confirm &amp; next</button><button id="rewatch" class="small">Watch again</button></div>
<div class="btns"><button id="back" class="small">Go back one clip</button></div></section>
<section id="doneScreen" class="card hidden"><div class="kicker">Session complete</div><h1>Answers are ready.</h1><div id="summary"></div><button id="save">Export JSON</button><p>Send the downloaded <b>event_labels_20260715_results.json</b> back for validated ingest.</p></section></main>
<script>
const ITEMS={items_json};const SESSION_ID={session_json};const PAGE_GENERATOR_VERSION={page_version_json};const KEY="event_labels_20260715_answers_v2";
let answers=JSON.parse(localStorage.getItem(KEY)||"{{}}");let i=0,pending=null,click=null;const $=id=>document.getElementById(id),player=$("player"),marker=$("marker");
function firstUnanswered(){{const k=ITEMS.findIndex(it=>!(it.row in answers));return k<0?ITEMS.length:k}}function show(id){{["startScreen","reviewScreen","doneScreen"].forEach(x=>$(x).classList.add("hidden"));$(id).classList.remove("hidden")}}function persist(){{localStorage.setItem(KEY,JSON.stringify(answers))}}
function centerPaused(){{player.loop=false;player.pause();if(player.duration)player.currentTime=player.duration/2}}
function phase1(){{pending=null;click=null;marker.style.display="none";$("phase1btns").classList.remove("hidden");$("phase2btns").classList.add("hidden");$("question").textContent="What happened at the center candidate?";$("hint").textContent="Press play to watch with sound. Keys 1–5 select a decision.";centerPaused()}}
function phase2(type){{pending=type;click=null;marker.style.display="none";$("confirm").disabled=true;$("phase1btns").classList.add("hidden");$("phase2btns").classList.remove("hidden");$("question").textContent="Click the ball at the contact point";$("hint").textContent="Use ←/→ for exactly one source frame, then click the ball.";centerPaused()}}
function render(){{if(i>=ITEMS.length){{finish();return}}show("reviewScreen");$("progress").textContent=(i+1)+" of "+ITEMS.length;player.src=ITEMS[i].file;$("back").style.visibility=i?"visible":"hidden";player.onloadedmetadata=()=>centerPaused();phase1()}}
function commit(decision,withClick){{const item=ITEMS[i],rec={{label_id:item.label_id,decision}};if(withClick){{if(!click)return;rec.x=click.x;rec.y=click.y;rec.dt=Math.round(((player.currentTime-player.duration/2)/2)*10000)/10000}}answers[item.row]=rec;persist();i++;render()}}
player.addEventListener("click",e=>{{if(!pending)return;const r=player.getBoundingClientRect();click={{x:Math.round((e.clientX-r.left)/r.width*10000)/10000,y:Math.round((e.clientY-r.top)/r.height*10000)/10000}};marker.style.left=(e.clientX-r.left)+"px";marker.style.top=(e.clientY-r.top)+"px";marker.style.display="block";$("confirm").disabled=false}});
function nudge(direction){{if(!pending)return;player.pause();const step=2/ITEMS[i].source_fps;player.currentTime=Math.max(0,Math.min(player.duration||2.4,player.currentTime+direction*step))}}
$("start").onclick=()=>{{i=firstUnanswered();render()}};$("paddle").onclick=()=>phase2("paddle");$("ground").onclick=()=>phase2("ground");$("otherhit").onclick=()=>phase2("other");$("none").onclick=()=>commit("none",false);$("unclear").onclick=()=>commit("unclear",false);$("confirm").onclick=()=>commit(pending,true);$("rewatch").onclick=()=>{{player.currentTime=0;player.loop=true;player.play().catch(()=>{{}})}};$("stepback").onclick=()=>nudge(-1);$("stepfwd").onclick=()=>nudge(1);$("back").onclick=()=>{{if(i>0){{i--;delete answers[ITEMS[i].row];persist();render()}}}};
document.addEventListener("keydown",e=>{{if($("reviewScreen").classList.contains("hidden"))return;if(e.key==="ArrowLeft"||e.key==="ArrowRight"){{e.preventDefault();nudge(e.key==="ArrowLeft"?-1:1);return}}if(!pending){{const actions={{"1":()=>phase2("paddle"),"2":()=>phase2("ground"),"3":()=>phase2("other"),"4":()=>commit("none",false),"5":()=>commit("unclear",false)}};if(actions[e.key])actions[e.key]()}}}});
function finish(){{show("doneScreen");$("progress").textContent="";const vals=Object.values(answers).map(a=>a.decision),c=v=>vals.filter(x=>x===v).length;$("summary").textContent=`Paddle ${{c("paddle")}} · Ground ${{c("ground")}} · Other ${{c("other")}} · Nothing ${{c("none")}} · Unclear ${{c("unclear")}}`}}
$("save").onclick=()=>{{const payload={{results_schema_version:2,session_id:SESSION_ID,page_generator_version:PAGE_GENERATOR_VERSION,coords:"normalized to displayed video, origin top-left",dt:"seconds in SOURCE time relative to the labeled anchor PTS",answers}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:"application/json"}}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="event_labels_20260715_results.json";a.click()}};
</script></body></html>'''


def _probe_render(path: Path) -> dict[str, Any]:
    payload = _run_json(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=codec_type",
            "-of", "json", str(path),
        ]
    )
    duration = float(payload["format"]["duration"])
    audio = any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))
    video = any(stream.get("codec_type") == "video" for stream in payload.get("streams", []))
    return {"duration_s": duration, "has_audio": audio, "has_video": video}


def render_session(root: Path, manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        source = (root / row["video_path"]).resolve()
        rally_root = (root / "data/online_harvest_20260706/rallies").resolve()
        if not source.is_relative_to(rally_root):
            raise ValueError(f"refusing to render non-universe path: {source}")
        output = out_dir / _filename(row)
        start = float(row["anchor_pts_s"]) - 0.6
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.9f}", "-t", "1.2", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", "setpts=2.0*PTS", "-af", "atempo=0.5",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            failures.append({"row": row["row"], "error": completed.stderr.strip()})
            continue
        validation = _probe_render(output)
        decode = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(output), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        validation.update({"row": row["row"], "file": output.name, "decode_exit_code": decode.returncode})
        validations.append(validation)
        if not (2.25 <= validation["duration_s"] <= 2.55 and validation["has_audio"] and validation["has_video"] and decode.returncode == 0):
            failures.append({"row": row["row"], "validation": validation, "decode_error": decode.stderr.strip()})
    if failures:
        raise RuntimeError(f"render/validation failures ({len(failures)}); deterministic substitution unavailable without changing sampled provenance: {failures[:3]}")
    html = _render_html(manifest)
    if "stratum" in html.lower() or "audio_onset" in html or "track_discontinuity" in html or "uniform_random" in html:
        raise AssertionError("blind page leaks sampling information")
    (out_dir / "START_HERE.html").write_text(html, encoding="utf-8")
    report = {
        "session_id": manifest["session_id"],
        "requested": len(manifest["rows"]),
        "rendered": len(validations),
        "validated": len(validations),
        "failures": failures,
        "substitutions": [],
        "duration_range_s": [round(min(row["duration_s"] for row in validations), 6), round(max(row["duration_s"] for row in validations), 6)],
        "all_have_audio": all(row["has_audio"] for row in validations),
        "all_decode_exit_zero": all(row["decode_exit_code"] == 0 for row in validations),
        "page_blind": True,
        "pack_bytes": sum(path.stat().st_size for path in out_dir.iterdir() if path.is_file()),
    }
    _write_json(out_dir / "render_report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample", help="Create the deterministic 300-row session manifest")
    sample.add_argument("--root", type=Path, default=Path("."))
    sample.add_argument("--seed", type=int, default=20260715)
    sample.add_argument("--out", type=Path, required=True)
    render = subparsers.add_parser("render", help="Render half-speed clips and the blind review page")
    render.add_argument("--root", type=Path, default=Path("."))
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "sample":
        payload = build_session(args.root, seed=args.seed)
        _write_json(args.out, payload)
        print(json.dumps({"session_id": payload["session_id"], "rows": len(payload["rows"]), "out": str(args.out)}, sort_keys=True))
    else:
        payload = render_session(args.root, args.manifest, args.out_dir)
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
