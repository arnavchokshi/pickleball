#!/usr/bin/env python3
"""Build Wave-6 owner CVAT labeling packages from Phase-B SST disagreements.

Writes only under:
  - cvat_upload/w6_labelpack_20260708/
  - cvat_upload/OWNER_SESSION_W6_20260708.md
  - runs/lanes/w6_labelpack_20260708/
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import cv2

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from threed.racketsport.ball_sst_dataset import build_sst_disagreement_queue  # noqa: E402

LANE = REPO / "runs/lanes/w6_labelpack_20260708"
OUT = REPO / "cvat_upload/w6_labelpack_20260708"
PACKAGES = OUT / "packages"
OWNER_RUNBOOK = REPO / "cvat_upload/OWNER_SESSION_W6_20260708.md"

TEACHER_ROOT = REPO / "data/online_harvest_20260706/prelabels"
TEACHER_SUBSET = LANE / "teacher_predictions_subset"
STUDENT_ROOT = REPO / "runs/lanes/w5_closeproof_20260708/phase_b_predictions"
RALLY_ROOT = REPO / "data/online_harvest_20260706/rallies"
HARVEST_MANIFEST = REPO / "data/online_harvest_20260706/manifest.json"
W5_MANIFEST = REPO / "cvat_upload/w5_labelpack_20260708/package_manifest.json"

DISAGREEMENT_QUEUE = LANE / "sst_disagreements.json"
SESSION_SIZE = 640
TYPE_ORDER = ["large-offset", "teacher-only", "student-only"]
TYPE_FRACTIONS = {"large-offset": 0.50, "teacher-only": 0.25, "student-only": 0.25}
NEW_SOURCE_ORDER = ["HyUqT7zFiwk", "wBu8bC4OfUY", "_L0HVmAlCQI", "zwCtH_i1_S4"]
W5_SOURCE_ORDER = ["73VurrTKCZ8", "Ezz6HDNHlnk"]
BOX_HALF_SIZE_PX = 8.0
JPEG_QUALITY = 92
LARGE_OFFSET_PX = 25.0

PROTECTED_PATTERNS = [
    "pwxNwFfYQlQ",
    "vQhtz8l6VqU",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
    "03_outdoor_webcam_iynbd",
    "04_indoor_doubles_fwuks",
]

SOURCE_CLASSES = {
    "73VurrTKCZ8": {
        "source_class": "outdoor_day_multicam",
        "venue_summary": "outdoor daylight, fixed-court rec source; multi-camera title/source, 1080p",
        "priority_note": "w5 source; already in owner queue",
    },
    "Ezz6HDNHlnk": {
        "source_class": "outdoor_night_fenced",
        "venue_summary": "outdoor night, lights/fence, low court-level rec source, 1080p",
        "priority_note": "w5 source; already in owner queue",
    },
    "HyUqT7zFiwk": {
        "source_class": "indoor_court_level",
        "venue_summary": "indoor court-level advanced doubles, 1080p",
        "priority_note": "new w6 Phase-B source; first session introduces it",
    },
    "wBu8bC4OfUY": {
        "source_class": "outdoor_night_tennis_overlay",
        "venue_summary": "outdoor night Rich Pickleball source with tennis-overlay court markings, 1080p",
        "priority_note": "new w6 Phase-B source; first session introduces it",
    },
    "_L0HVmAlCQI": {
        "source_class": "outdoor_night_tennis_overlay",
        "venue_summary": "outdoor night Rich Pickleball source with tennis-overlay court markings, 1080p",
        "priority_note": "new w6 Phase-B source; first session introduces it",
    },
    "zwCtH_i1_S4": {
        "source_class": "outdoor_day_broadcast_overlay",
        "venue_summary": "outdoor daytime fixed wide source with lower-third/broadcast overlay, 1080p",
        "priority_note": "new w6 Phase-B source; first session introduces it",
    },
}


def main() -> int:
    reset_owned_outputs()
    phase_b_clips = phase_b_clip_ids()
    queue_payload = build_phase_b_queue(phase_b_clips)
    clip_info = build_clip_info(phase_b_clips)
    source_meta = build_source_meta()

    selected_sessions, selection_summary = select_sessions(queue_payload["queue"])
    materialize_ball_frames(selected_sessions, clip_info)
    ball_inventory = package_ball_sessions(selected_sessions, clip_info)

    package_manifest = write_package_manifest(
        ball_inventory=ball_inventory,
        selection_summary=selection_summary,
        source_meta=source_meta,
        disagreement_summary=queue_payload["summary"],
        phase_b_clips=phase_b_clips,
    )
    validation = validate_packages(package_manifest)
    write_owner_runbook(package_manifest, validation)
    write_report(package_manifest, validation)
    print(
        json.dumps(
            {
                "status": "built",
                "sessions": len(ball_inventory),
                "frames": sum(int(item["frame_count"]) for item in ball_inventory),
                "package_manifest": rel(package_manifest),
                "validation_report": rel(LANE / "validation_report.json"),
                "report": rel(LANE / "report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def reset_owned_outputs() -> None:
    for path in [
        OUT,
        LANE / "frame_staging",
        LANE / "selection_manifest.json",
        LANE / "review_manifest.json",
        LANE / "package_manifest.json",
        LANE / "validation_report.json",
        LANE / "report.json",
        LANE / "REPORT.md",
        LANE / "protected_programmatic_check.txt",
        OWNER_RUNBOOK,
    ]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    PACKAGES.mkdir(parents=True, exist_ok=True)
    (LANE / "frame_staging/ball").mkdir(parents=True, exist_ok=True)


def phase_b_clip_ids() -> list[str]:
    if not STUDENT_ROOT.is_dir():
        raise FileNotFoundError(f"missing Phase-B prediction root: {STUDENT_ROOT}")
    clips = sorted(path.name for path in STUDENT_ROOT.iterdir() if (path / "ball_track.json").is_file())
    if len(clips) != 24:
        raise RuntimeError(f"expected 24 Phase-B clips, found {len(clips)} under {STUDENT_ROOT}")
    return clips


def build_phase_b_queue(clip_ids: list[str]) -> dict[str, Any]:
    if TEACHER_SUBSET.exists():
        if TEACHER_SUBSET.is_dir() and not TEACHER_SUBSET.is_symlink():
            shutil.rmtree(TEACHER_SUBSET)
        else:
            TEACHER_SUBSET.unlink()
    TEACHER_SUBSET.mkdir(parents=True, exist_ok=True)
    for clip_id in clip_ids:
        source = TEACHER_ROOT / clip_id
        if not (source / "ball_track.json").is_file():
            raise FileNotFoundError(f"missing teacher ball_track.json for {clip_id}: {source}")
        target = TEACHER_SUBSET / clip_id
        target.symlink_to(source, target_is_directory=True)
    queue = build_sst_disagreement_queue(
        teacher_predictions=TEACHER_SUBSET,
        student_predictions=STUDENT_ROOT,
        out_path=DISAGREEMENT_QUEUE,
        large_offset_px=LARGE_OFFSET_PX,
    )
    if int(queue["summary"]["clip_count"]) != 24:
        raise RuntimeError(f"unexpected disagreement clip_count: {queue['summary']}")
    return queue


def build_clip_info(clip_ids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for clip_id in clip_ids:
        source_id = source_id_from_clip(clip_id)
        video_path = RALLY_ROOT / source_id / f"{clip_id}.mp4"
        teacher_metadata = TEACHER_ROOT / clip_id / "ball_track_metadata.json"
        teacher_ball_track = TEACHER_ROOT / clip_id / "ball_track.json"
        student_ball_track = STUDENT_ROOT / clip_id / "ball_track.json"
        if not video_path.is_file():
            raise FileNotFoundError(f"missing rally video for {clip_id}: {video_path}")
        metadata = read_json(teacher_metadata)
        student = read_json(student_ball_track)
        runtime = metadata.get("runtime", {})
        width, height = runtime.get("source_video_size") or [None, None]
        if not width or not height:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {video_path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        frames = student.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"student ball_track frames must be a list: {student_ball_track}")
        result[clip_id] = {
            "clip_id": clip_id,
            "source_id": source_id,
            "video": video_path,
            "teacher_ball_track": teacher_ball_track,
            "student_ball_track": student_ball_track,
            "teacher_metadata": teacher_metadata,
            "fps": float(student.get("fps") or metadata.get("fps") or runtime.get("source_video_fps")),
            "frame_count": len(frames),
            "width": int(width),
            "height": int(height),
        }
    return result


def build_source_meta() -> dict[str, dict[str, Any]]:
    manifest = json.loads(HARVEST_MANIFEST.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in manifest if isinstance(item, dict) and item.get("id")}
    result: dict[str, dict[str, Any]] = {}
    for source_id, klass in SOURCE_CLASSES.items():
        meta = by_id.get(source_id, {})
        result[source_id] = {
            **klass,
            "source_id": source_id,
            "title": meta.get("title"),
            "channel": meta.get("channel"),
            "fps": meta.get("fps"),
            "resolution": [meta.get("width"), meta.get("height")],
            "status": meta.get("status"),
        }
    return result


def select_sessions(queue: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_keys: set[tuple[str, int]] = set()
    by_source_type: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    source_counts = Counter()
    type_counts = Counter()
    clip_counts = Counter()

    for index, raw in enumerate(queue):
        row = dict(raw)
        source_id = source_id_from_clip(row["clip_id"])
        if source_id not in NEW_SOURCE_ORDER:
            continue
        row["queue_index"] = index
        row["source_id"] = source_id
        by_source_type[(source_id, row["disagreement_type"])].append(row)
        source_counts[source_id] += 1
        type_counts[row["disagreement_type"]] += 1
        clip_counts[row["clip_id"]] += 1

    if sum(source_counts.values()) != len(queue):
        raise RuntimeError("Phase-B queue contained rows outside the four expected w6 sources")

    sessions: list[dict[str, Any]] = []
    session_index = 0
    while remaining_count(by_source_type, selected_keys) > 0:
        session_index += 1
        session_id = f"ball_session_{session_index:02d}"
        chosen = choose_session_rows(by_source_type, selected_keys)
        if not chosen:
            raise RuntimeError("session selector made no progress")
        ordered = order_session_rows(chosen)
        for ordinal, row in enumerate(ordered, start=1):
            row["session_id"] = session_id
            row["session_ordinal"] = ordinal
        sessions.append(
            {
                "session_id": session_id,
                "target_frame_count": SESSION_SIZE,
                "frames": ordered,
            }
        )
        print(
            f"selected {session_id}: frames={len(ordered)} "
            f"sources={dict(Counter(row['source_id'] for row in ordered))} "
            f"types={dict(Counter(row['disagreement_type'] for row in ordered))}",
            file=sys.stderr,
        )

    selected = [row for session in sessions for row in session["frames"]]
    selected_counts = Counter((row["clip_id"], int(row["frame_index"])) for row in selected)
    duplicates = [key for key, count in selected_counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"duplicate selected rows: {duplicates[:5]}")
    summary = {
        "strategy": (
            "Package every Phase-B disagreement row into 640-frame sessions. Sessions greedily keep all "
            "remaining new sources represented first, allocate frames evenly across active sources, use the "
            "w5 class mix scaled per active-source allocation when available, then fall back to the highest "
            "remaining queue rank within that source. Once a source exhausts, later sessions use the remaining "
            "active sources only."
        ),
        "ranking_assumption": (
            "Used queue order/rank from build_sst_disagreement_queue: large-offset rank is offset_px; "
            "teacher-only/student-only rank is visible model confidence. Class quotas are used before "
            "rank fallback because raw rank values are not directly comparable across disagreement semantics."
        ),
        "session_size": SESSION_SIZE,
        "source_order": NEW_SOURCE_ORDER,
        "queue_counts_phase_b": dict(type_counts),
        "queue_source_counts": dict(source_counts),
        "queue_clip_counts": dict(sorted(clip_counts.items())),
        "selected_count": len(selected),
        "selected_by_source": dict(Counter(row["source_id"] for row in selected)),
        "selected_by_type": dict(Counter(row["disagreement_type"] for row in selected)),
        "selected_by_source_type": {
            f"{source}/{typ}": count
            for (source, typ), count in sorted(Counter((row["source_id"], row["disagreement_type"]) for row in selected).items())
        },
        "sessions": [
            {
                "session_id": session["session_id"],
                "frame_count": len(session["frames"]),
                "source_counts": dict(Counter(row["source_id"] for row in session["frames"])),
                "type_counts": dict(Counter(row["disagreement_type"] for row in session["frames"])),
                "clip_count": len(set(row["clip_id"] for row in session["frames"])),
            }
            for session in sessions
        ],
    }
    if len(selected) != len(queue):
        raise RuntimeError(f"selected {len(selected)} rows but queue has {len(queue)} rows")
    write_json(
        LANE / "selection_manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "w6_labelpack_selection_manifest",
            "created_at_utc": utc_now(),
            "summary": summary,
            "sessions": sessions,
        },
    )
    return sessions, summary


def choose_session_rows(
    by_source_type: dict[tuple[str, str], deque[dict[str, Any]]],
    selected_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    active_sources = [source for source in NEW_SOURCE_ORDER if source_remaining(by_source_type, source, selected_keys) > 0]
    if not active_sources:
        return []
    allocation = even_allocation(active_sources, SESSION_SIZE, by_source_type, selected_keys)
    chosen: list[dict[str, Any]] = []
    for source in active_sources:
        count = allocation[source]
        source_rows: list[dict[str, Any]] = []
        quotas = type_quotas(count)
        for typ in TYPE_ORDER:
            source_rows.extend(pop_rows(by_source_type[(source, typ)], quotas[typ], selected_keys))
        if len(source_rows) < count:
            source_rows.extend(pop_any_source_rows(by_source_type, source, count - len(source_rows), selected_keys))
        chosen.extend(source_rows[:count])
    if len(chosen) < SESSION_SIZE:
        chosen.extend(pop_any_rows(by_source_type, SESSION_SIZE - len(chosen), selected_keys))
    return chosen


def even_allocation(
    active_sources: list[str],
    target: int,
    by_source_type: dict[tuple[str, str], deque[dict[str, Any]]],
    selected_keys: set[tuple[str, int]],
) -> dict[str, int]:
    remaining = {source: source_remaining(by_source_type, source, selected_keys) for source in active_sources}
    allocation = {source: 0 for source in active_sources}
    slots = min(target, sum(remaining.values()))
    while slots > 0:
        progressed = False
        for source in active_sources:
            if slots <= 0:
                break
            if allocation[source] >= remaining[source]:
                continue
            allocation[source] += 1
            slots -= 1
            progressed = True
        if not progressed:
            break
    return allocation


def type_quotas(count: int) -> dict[str, int]:
    large = int(round(count * TYPE_FRACTIONS["large-offset"]))
    teacher = int(round(count * TYPE_FRACTIONS["teacher-only"]))
    student = count - large - teacher
    return {"large-offset": large, "teacher-only": teacher, "student-only": student}


def remaining_count(by_source_type: dict[tuple[str, str], deque[dict[str, Any]]], selected_keys: set[tuple[str, int]]) -> int:
    return sum(source_remaining(by_source_type, source, selected_keys) for source in NEW_SOURCE_ORDER)


def source_remaining(
    by_source_type: dict[tuple[str, str], deque[dict[str, Any]]],
    source: str,
    selected_keys: set[tuple[str, int]],
) -> int:
    total = 0
    for typ in TYPE_ORDER:
        bucket = by_source_type[(source, typ)]
        while bucket and (bucket[0]["clip_id"], int(bucket[0]["frame_index"])) in selected_keys:
            bucket.popleft()
        total += sum(1 for row in bucket if (row["clip_id"], int(row["frame_index"])) not in selected_keys)
    return total


def pop_rows(bucket: deque[dict[str, Any]], count: int, selected_keys: set[tuple[str, int]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    while bucket and len(result) < count:
        row = bucket.popleft()
        key = (row["clip_id"], int(row["frame_index"]))
        if key in selected_keys:
            continue
        selected_keys.add(key)
        result.append(row)
    return result


def pop_any_source_rows(
    by_source_type: dict[tuple[str, str], deque[dict[str, Any]]],
    source: str,
    count: int,
    selected_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    while len(result) < count:
        best_key: tuple[str, str] | None = None
        best_row: dict[str, Any] | None = None
        for typ in TYPE_ORDER:
            bucket = by_source_type[(source, typ)]
            while bucket and (bucket[0]["clip_id"], int(bucket[0]["frame_index"])) in selected_keys:
                bucket.popleft()
            if bucket and (best_row is None or int(bucket[0]["queue_index"]) < int(best_row["queue_index"])):
                best_key = (source, typ)
                best_row = bucket[0]
        if best_key is None:
            break
        result.extend(pop_rows(by_source_type[best_key], 1, selected_keys))
    return result


def pop_any_rows(
    by_source_type: dict[tuple[str, str], deque[dict[str, Any]]],
    count: int,
    selected_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    while len(result) < count:
        best_key: tuple[str, str] | None = None
        best_row: dict[str, Any] | None = None
        for key, bucket in by_source_type.items():
            while bucket and (bucket[0]["clip_id"], int(bucket[0]["frame_index"])) in selected_keys:
                bucket.popleft()
            if bucket and (best_row is None or int(bucket[0]["queue_index"]) < int(best_row["queue_index"])):
                best_key = key
                best_row = bucket[0]
        if best_key is None:
            break
        result.extend(pop_rows(by_source_type[best_key], 1, selected_keys))
    return result


def order_session_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: int(item["queue_index"])):
        buckets[(row["source_id"], row["disagreement_type"])].append(row)
    bucket_order = [(source, typ) for source in NEW_SOURCE_ORDER for typ in TYPE_ORDER]
    ordered: list[dict[str, Any]] = []
    while len(ordered) < len(rows):
        moved = False
        for key in bucket_order:
            if buckets[key]:
                ordered.append(buckets[key].popleft())
                moved = True
        if not moved:
            break
    return ordered


def materialize_ball_frames(sessions: list[dict[str, Any]], clip_info: dict[str, dict[str, Any]]) -> None:
    staging = LANE / "frame_staging/ball"
    review_clips: list[dict[str, Any]] = []
    rows_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        for row in session["frames"]:
            info = clip_info[row["clip_id"]]
            row["prelabel"] = choose_prelabel(row, info)
            row["image_file_name"] = ball_image_name(row)
            row["image_path"] = str(staging / row["session_id"] / row["image_file_name"])
            rows_by_clip[row["clip_id"]].append(row)

    extraction_errors: list[str] = []
    for clip_id, rows in sorted(rows_by_clip.items()):
        info = clip_info[clip_id]
        cap = cv2.VideoCapture(str(info["video"]))
        if not cap.isOpened():
            extraction_errors.append(f"cannot open {info['video']}")
            continue
        next_frame_to_read: int | None = None
        for row in sorted(rows, key=lambda item: int(item["frame_index"])):
            frame_index = int(row["frame_index"])
            out_path = Path(row["image_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if next_frame_to_read is None or frame_index < next_frame_to_read or frame_index - next_frame_to_read > 120:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                next_frame_to_read = frame_index
            while next_frame_to_read < frame_index:
                ok, _ = cap.read()
                if not ok:
                    extraction_errors.append(f"cannot advance {clip_id} to frame {frame_index}")
                    break
                next_frame_to_read += 1
            if next_frame_to_read != frame_index:
                continue
            ok, frame = cap.read()
            next_frame_to_read += 1
            if not ok or frame is None:
                extraction_errors.append(f"cannot read {clip_id} frame {frame_index}")
                continue
            if frame.shape[1] != info["width"] or frame.shape[0] != info["height"]:
                extraction_errors.append(f"dimension mismatch {clip_id} frame {frame_index}: got {frame.shape[1]}x{frame.shape[0]}")
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        cap.release()
        print(f"extracted {clip_id}: {len(rows)} frames", file=sys.stderr)
    if extraction_errors:
        raise RuntimeError("frame extraction failed: " + "; ".join(extraction_errors[:20]))

    for session in sessions:
        review_items = []
        for row in session["frames"]:
            review_items.append(
                {
                    "clip": session["session_id"],
                    "frame": row["image_file_name"],
                    "frame_index": int(row["frame_index"]),
                    "image_path": row["image_path"],
                    "review_id": f"{session['session_id']}__{row['session_ordinal']:04d}",
                    "source_image_exists": True,
                    "target_file": "ball_track.json",
                }
            )
        review_clips.append({"clip": session["session_id"], "review_items": review_items})

    write_json(
        LANE / "review_manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "w6_labelpack_review_manifest_for_cvat_packages",
            "status": "candidate_prediction",
            "created_at_utc": utc_now(),
            "clips": review_clips,
            "not_ground_truth": True,
        },
    )


def choose_prelabel(row: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher") if isinstance(row.get("teacher"), dict) else None
    student = row.get("student") if isinstance(row.get("student"), dict) else None
    candidates: list[tuple[str, dict[str, Any]]] = []
    if teacher and teacher.get("visible"):
        candidates.append(("teacher", teacher))
    if student and student.get("visible"):
        candidates.append(("student", student))
    if not candidates:
        raise ValueError(f"disagreement row lacks visible prediction: {row}")
    source, pred = max(candidates, key=lambda item: float(item[1].get("score", 0.0)))
    x, y = [float(v) for v in pred["xy"]]
    width = float(info["width"])
    height = float(info["height"])
    return {
        "source": source,
        "score": float(pred.get("score", 0.0)),
        "xy": [x, y],
        "box": [
            max(0.0, x - BOX_HALF_SIZE_PX),
            max(0.0, y - BOX_HALF_SIZE_PX),
            min(width, x + BOX_HALF_SIZE_PX),
            min(height, y + BOX_HALF_SIZE_PX),
        ],
    }


def ball_image_name(row: dict[str, Any]) -> str:
    typ = row["disagreement_type"].replace("-", "_")
    return (
        f"{int(row['session_ordinal']):04d}__{row['source_id']}__{row['clip_id']}"
        f"__f{int(row['frame_index']):06d}__{typ}.jpg"
    )


def package_ball_sessions(sessions: list[dict[str, Any]], clip_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for session in sessions:
        session_id = session["session_id"]
        frame_count = len(session["frames"])
        image_dir = LANE / "frame_staging/ball" / session_id
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        data_zip = PACKAGES / f"{session_id}_{frame_count:03d}f_w6_images.zip"
        prelabel_zip = PACKAGES / f"{session_id}_{frame_count:03d}f_w6_prelabels_cvat1_1.zip"
        image_files = sorted(image_dir.glob("*.jpg"))
        if len(image_files) != frame_count:
            raise RuntimeError(f"{session_id}: expected {frame_count} images, found {len(image_files)}")
        zip_files(data_zip, image_files, image_dir)
        xml = build_ball_images_xml(session, clip_info)
        write_zip_text(prelabel_zip, "annotations.xml", xml)
        source_ids = [source for source in NEW_SOURCE_ORDER if any(row["source_id"] == source for row in session["frames"])]
        source_classes = {source: SOURCE_CLASSES[source]["source_class"] for source in source_ids}
        type_counts = Counter(row["disagreement_type"] for row in session["frames"])
        source_counts = Counter(row["source_id"] for row in session["frames"])
        clip_counts = Counter(row["clip_id"] for row in session["frames"])
        inventory.append(
            {
                "kind": "ball_session",
                "session_id": session_id,
                "task_name": f"w6_ball_sst_{session_id}_20260708",
                "image_zip": rel(data_zip),
                "prelabel_zip": rel(prelabel_zip),
                "frame_count": frame_count,
                "clip_count": len(clip_counts),
                "source_ids": source_ids,
                "source_classes": source_classes,
                "source_counts": dict(source_counts),
                "disagreement_type_counts": dict(type_counts),
                "clip_counts": dict(sorted(clip_counts.items())),
                "estimated_label_hours_at_240_fph": round(frame_count / 240.0, 2),
                "priority_rationale": "New-source coverage first while active sources remain; then ranked class/source fallback over all remaining Phase-B rows.",
            }
        )
        print(f"packaged {session_id}: {frame_count} frames", file=sys.stderr)
    return inventory


def build_ball_images_xml(session: dict[str, Any], clip_info: dict[str, dict[str, Any]]) -> str:
    labels_xml = ball_labels_xml()
    images_xml: list[str] = []
    for image_id, row in enumerate(session["frames"]):
        info = clip_info[row["clip_id"]]
        box = row["prelabel"]["box"]
        attrs = [
            ("visibility", "true"),
            ("visibility_level", "clear"),
            ("center_convention", "review_to_blur_streak_center"),
            ("blur_angle_deg", "0"),
            ("blur_length_px", "0"),
            ("blur_width_px", "0"),
            ("blur_label_quality", ""),
        ]
        attr_xml = "\n".join(f'      <attribute name="{escape(name)}">{escape(value)}</attribute>' for name, value in attrs)
        images_xml.append(
            f'  <image id="{image_id}" name="{escape(row["image_file_name"])}" width="{info["width"]}" height="{info["height"]}">\n'
            f'    <box label="ball" source="auto" occluded="0" xtl="{box[0]:.2f}" ytl="{box[1]:.2f}" '
            f'xbr="{box[2]:.2f}" ybr="{box[3]:.2f}" z_order="0">\n'
            f"{attr_xml}\n"
            f"    </box>\n"
            f"  </image>"
        )
    return cvat_images_xml(
        task_name=f"w6_ball_sst_{session['session_id']}_20260708",
        size=len(session["frames"]),
        labels_xml=labels_xml,
        images_xml="\n".join(images_xml),
        dumped="w6_labelpack_20260708 Phase-B ball SST prelabels; not ground truth",
    )


def ball_labels_xml() -> str:
    return """        <label>
          <name>ball</name>
          <type>rectangle</type>
          <attributes>
            <attribute><name>visibility</name><mutable>True</mutable><input_type>checkbox</input_type><default_value>true</default_value><values>true
false</values></attribute>
            <attribute><name>visibility_level</name><mutable>True</mutable><input_type>select</input_type><default_value>clear</default_value><values>clear
partial
full
out_of_frame</values></attribute>
            <attribute><name>center_convention</name><mutable>True</mutable><input_type>text</input_type><default_value></default_value><values></values></attribute>
            <attribute><name>blur_angle_deg</name><mutable>True</mutable><input_type>number</input_type><default_value>0</default_value><values>0
360
1</values></attribute>
            <attribute><name>blur_length_px</name><mutable>True</mutable><input_type>number</input_type><default_value>0</default_value><values>0
5000
1</values></attribute>
            <attribute><name>blur_width_px</name><mutable>True</mutable><input_type>number</input_type><default_value>0</default_value><values>0
5000
1</values></attribute>
            <attribute><name>blur_label_quality</name><mutable>True</mutable><input_type>text</input_type><default_value></default_value><values></values></attribute>
          </attributes>
        </label>"""


def cvat_images_xml(*, task_name: str, size: int, labels_xml: str, images_xml: str, dumped: str) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>0</id>
      <name>{escape(task_name)}</name>
      <size>{size}</size>
      <mode>annotation</mode>
      <overlap>0</overlap>
      <start_frame>0</start_frame>
      <stop_frame>{max(size - 1, 0)}</stop_frame>
      <labels>
{labels_xml}
      </labels>
    </task>
    <dumped>{escape(dumped)}</dumped>
  </meta>
{images_xml}
</annotations>
"""


def write_package_manifest(
    *,
    ball_inventory: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    source_meta: dict[str, dict[str, Any]],
    disagreement_summary: dict[str, Any],
    phase_b_clips: list[str],
) -> Path:
    w5_counts = w5_source_counts()
    w6_counts = Counter()
    for item in ball_inventory:
        w6_counts.update(item["source_counts"])
    union_counts = {source: int(w5_counts.get(source, 0) + w6_counts.get(source, 0)) for source in [*W5_SOURCE_ORDER, *NEW_SOURCE_ORDER]}
    represented = sorted({clip for item in ball_inventory for clip in item["clip_counts"]})
    missing = sorted(set(phase_b_clips) - set(represented))
    manifest = {
        "schema_version": 1,
        "artifact_type": "w6_labelpack_20260708_package_manifest",
        "created_at_utc": utc_now(),
        "objective": "Owner CVAT packages from Wave-6 Phase-B SST disagreement queue over 24 new clips.",
        "protected_patterns": PROTECTED_PATTERNS,
        "ball_sessions": ball_inventory,
        "selection_summary": selection_summary,
        "source_metadata": source_meta,
        "source_balance_note": {
            "w5_sources_already_packaged": W5_SOURCE_ORDER,
            "w6_new_sources_packaged": NEW_SOURCE_ORDER,
            "union_nonheldout_source_counts": union_counts,
            "union_nonheldout_source_count": len([source for source, count in union_counts.items() if count > 0]),
            "phase_b_clip_count": len(phase_b_clips),
            "phase_b_clips_represented": represented,
            "phase_b_clip_exclusions": {clip: "zero disagreement rows" for clip in missing},
        },
        "disagreement_queue": {
            "path": rel(DISAGREEMENT_QUEUE),
            "large_offset_px": LARGE_OFFSET_PX,
            "summary": disagreement_summary,
        },
        "export_cvat_tasks": {
            "status": "not_run",
            "reason": "w6 packages were written directly to the same CVAT image-zip + CVAT 1.1 prelabel-zip layout validated by the w5 packager to avoid duplicating 43,230 staged images.",
        },
        "owner_import_script": rel(LANE / "import_w6_labelpack_tasks.py"),
        "owner_runbook": rel(OWNER_RUNBOOK),
    }
    write_json(OUT / "package_manifest.json", manifest)
    write_json(LANE / "package_manifest.json", manifest)
    return LANE / "package_manifest.json"


def validate_packages(package_manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(package_manifest_path)
    package_items = [(item["image_zip"], item["prelabel_zip"]) for item in manifest["ball_sessions"]]

    errors: list[str] = []
    package_checks: list[dict[str, Any]] = []
    protected_hits: list[dict[str, Any]] = []
    total_images = 0
    total_boxes = 0
    for image_zip_rel, xml_zip_rel in package_items:
        image_zip = REPO / image_zip_rel
        xml_zip = REPO / xml_zip_rel
        image_names = zip_names(image_zip)
        xml_names = zip_names(xml_zip)
        if xml_names != ["annotations.xml"]:
            errors.append(f"{xml_zip_rel}: expected only annotations.xml, got {xml_names}")
            raw_xml = ""
            root = ET.Element("annotations")
        else:
            raw_xml = read_zip_text(xml_zip, "annotations.xml")
            root = ET.fromstring(raw_xml)
        if root.findtext("version") != "1.1":
            errors.append(f"{xml_zip_rel}: not CVAT version 1.1")
        xml_image_names = [image.attrib["name"] for image in root.findall("image")]
        missing_images = sorted(set(xml_image_names) - set(image_names))
        extra_images = sorted(set(image_names) - set(xml_image_names))
        if missing_images:
            errors.append(f"{xml_zip_rel}: XML references missing images {missing_images[:5]}")
        if extra_images:
            errors.append(f"{image_zip_rel}: image zip has unreferenced images {extra_images[:5]}")
        box_count = 0
        bad_box_images: list[str] = []
        for image in root.findall("image"):
            boxes = image.findall("box")
            box_count += len(boxes)
            if len(boxes) != 1:
                bad_box_images.append(image.attrib.get("name", "<missing-name>"))
        if bad_box_images:
            errors.append(f"{xml_zip_rel}: expected one ball box per image; bad images {bad_box_images[:5]}")
        for pattern in PROTECTED_PATTERNS:
            for name in image_names + xml_names:
                if pattern in name:
                    protected_hits.append({"package": image_zip_rel, "pattern": pattern, "where": "zip_entry", "value": name})
            if pattern in raw_xml:
                protected_hits.append({"package": xml_zip_rel, "pattern": pattern, "where": "annotations_xml"})
        total_images += len(image_names)
        total_boxes += box_count
        package_checks.append(
            {
                "image_zip": image_zip_rel,
                "prelabel_zip": xml_zip_rel,
                "image_count": len(image_names),
                "xml_image_count": len(xml_image_names),
                "box_count": box_count,
                "xml_entries": xml_names,
                "byte_layout_matches_known_good_pattern": len(xml_names) == 1 and xml_names[0] == "annotations.xml",
            }
        )
    if protected_hits:
        errors.append(f"protected package hits: {protected_hits}")

    protected_output = protected_programmatic_check(package_items)
    if any(line.startswith("MATCH ") for line in protected_output.splitlines()):
        errors.append("programmatic protected check found a MATCH")

    validation = {
        "schema_version": 1,
        "artifact_type": "w6_labelpack_validation_report",
        "created_at_utc": utc_now(),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "package_checks": package_checks,
        "summary": {
            "package_pair_count": len(package_items),
            "zip_count": len(package_items) * 2,
            "total_images": total_images,
            "total_boxes": total_boxes,
        },
        "protected_programmatic_check": {
            "status": "passed" if not protected_hits and not any(line.startswith("MATCH ") for line in protected_output.splitlines()) else "failed",
            "patterns": PROTECTED_PATTERNS,
            "hits": protected_hits,
        },
        "protected_grep_check_output": protected_output,
        "structural_roundtrip_note": "Validates against known-good CVAT image-package layout: image zip entries plus prelabel zip containing one annotations.xml, CVAT 1.1 root, XML image refs exactly match image zip entries, and one editable ball box per image.",
    }
    write_json(LANE / "validation_report.json", validation)
    write_json(OUT / "validation_report.json", validation)
    (LANE / "protected_programmatic_check.txt").write_text(protected_output + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("validation failed: " + "; ".join(errors))
    return validation


def protected_programmatic_check(package_items: list[tuple[str, str]]) -> str:
    lines = []
    package_paths = sorted(REPO / rel for pair in package_items for rel in pair)
    for pattern in PROTECTED_PATTERNS:
        matched = False
        for package in package_paths:
            if pattern in rel(package):
                lines.append(f"MATCH pattern={pattern} package={rel(package)} where=path")
                matched = True
            with zipfile.ZipFile(package) as zf:
                for name in zf.namelist():
                    if pattern in name:
                        lines.append(f"MATCH pattern={pattern} package={rel(package)} entry={name}")
                        matched = True
                    if name.endswith((".xml", ".json", ".txt", ".md")):
                        data = zf.read(name).decode("utf-8", errors="ignore")
                        if pattern in data:
                            lines.append(f"MATCH pattern={pattern} package={rel(package)} member={name}")
                            matched = True
        if not matched:
            lines.append(f"NO_MATCH pattern={pattern} packages={len(package_paths)}")
    return "\n".join(lines)


def write_owner_runbook(package_manifest_path: Path, validation: dict[str, Any]) -> None:
    manifest = read_json(package_manifest_path)
    lines = [
        "# Owner Session W6 2026-07-08",
        "",
        "Use these tasks in order. Full sessions are 640 frames, about 2.7 hours at the measured 240 frames/hour. The packages already carry one editable ball prelabel per frame.",
        "",
        "## First Command",
        "",
        "```bash",
        "cd /Users/arnavchokshi/Desktop/pickleball",
        "open -a Docker",
        "cd /Users/arnavchokshi/cvat_labelfactory/cvat_src && docker compose up -d",
        "cd /Users/arnavchokshi/Desktop/pickleball",
        "runs/lanes/w3_labelfactory_20260707/venv/bin/python runs/lanes/w6_labelpack_20260708/import_w6_labelpack_tasks.py --dry-run",
        "# The separate import lane removes --dry-run when it is ready to create tasks.",
        "```",
        "",
        "Then open http://localhost:8080 and label the tasks named below after the import lane creates them.",
        "",
        "## Ball Convention",
        "",
        "- Label the ball box around the visible ball or visible blur streak.",
        "- BlurBall convention: for motion-blurred balls, put the box center on the blur-streak center, not the leading edge.",
        "- `clear`: ball is visible and localizable without material occlusion.",
        "- `partial`: ball is localizable but partly occluded or blurred.",
        "- `full`: ball is expected in-frame but fully hidden.",
        "- `out_of_frame`: ball is outside the image bounds.",
        "- Keep one ball object per frame. Drag/correct the prelabel if it is close; delete/recreate only when it is misleading.",
        "- **Prelabel on a NON-ball while the real ball IS visible**: drag the box onto the real ball (or delete + redraw). Never leave clear/partial on a non-ball.",
        "- **Ball NOT in frame but a box exists**: either DELETE the box or set visibility_level=out_of_frame. Use the visibility_level ATTRIBUTE dropdown.",
        "- **Background ball**: ALWAYS delete that box. One game-ball box max per frame.",
        "- **When in doubt, delete.** False positives poison training far more than missed positives; reviewed-empty frames are useful negatives.",
        "",
        "## Session Order",
        "",
    ]
    for item in manifest["ball_sessions"]:
        source_bits = ", ".join(f"{source}={klass}" for source, klass in item["source_classes"].items())
        if len(item["source_ids"]) == 4:
            unlock = "Introduces all four Phase-B sources not covered by w5; label these first for fastest 6-source union coverage."
        elif len(item["source_ids"]) > 1:
            unlock = "Keeps the still-active Phase-B sources mixed before any single-source tail work."
        else:
            unlock = "Tail session after the smaller Phase-B sources are exhausted; still packages ranked disagreement rows from the remaining source."
        lines.extend(
            [
                f"### {item['session_id']} - {item['frame_count']} frames",
                "",
                f"- CVAT task: `{item['task_name']}`",
                f"- Images: `{item['image_zip']}`",
                f"- Prelabels: `{item['prelabel_zip']}`",
                f"- Source classes: {source_bits}",
                f"- Source counts: `{json.dumps(item['source_counts'], sort_keys=True)}`",
                f"- Error mix: `{json.dumps(item['disagreement_type_counts'], sort_keys=True)}`",
                f"- Unlocks: {unlock}",
                "",
            ]
        )
    lines.extend(
        [
            "## Export",
            "",
            "For each finished task: Actions -> Export task dataset -> `CVAT for images 1.1`. Save the zip under:",
            "",
            "```text",
            "cvat_upload/exports/w6_labelpack_20260708/",
            "```",
            "",
            "Recommended filenames: `<task_name>_annotations.zip`.",
            "",
            "## Package Check",
            "",
            "Protected material check passed before handoff:",
            "",
            "```text",
            validation["protected_grep_check_output"],
            "```",
        ]
    )
    OWNER_RUNBOOK.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(package_manifest_path: Path, validation: dict[str, Any]) -> None:
    manifest = read_json(package_manifest_path)
    inventory = [
        {
            "path": item["image_zip"],
            "prelabels": item["prelabel_zip"],
            "clip_count": item["clip_count"],
            "frame_count": item["frame_count"],
            "source_counts": item["source_counts"],
            "error_mix": item["disagreement_type_counts"],
            "estimated_hours": item["estimated_label_hours_at_240_fph"],
        }
        for item in manifest["ball_sessions"]
    ]
    session_table = [
        {
            "name": item["session_id"],
            "task_name": item["task_name"],
            "frames": item["frame_count"],
            "sources": item["source_counts"],
            "error_mix": item["disagreement_type_counts"],
        }
        for item in manifest["ball_sessions"]
    ]
    report = {
        "objective_result": "PASS",
        "created_at_utc": utc_now(),
        "deliverable_inventory": inventory,
        "session_table": session_table,
        "per_source_coverage": manifest["source_balance_note"]["union_nonheldout_source_counts"],
        "protected_clip_check_output": {
            "programmatic": validation["protected_grep_check_output"],
        },
        "acceptance": {
            "phase_b_clips_represented": manifest["source_balance_note"]["phase_b_clips_represented"],
            "phase_b_clip_exclusions": manifest["source_balance_note"]["phase_b_clip_exclusions"],
            "union_nonheldout_source_counts": manifest["source_balance_note"]["union_nonheldout_source_counts"],
            "validation_status": validation["status"],
            "package_pair_count": validation["summary"]["package_pair_count"],
            "total_packaged_frames": validation["summary"]["total_images"],
            "task_name_scheme": "w6_ball_sst_<ball_session_NN>_20260708",
        },
        "priority_rationale": [
            "w5 already covers 73VurrTKCZ8 and Ezz6HDNHlnk.",
            "w6 front-loads HyUqT7zFiwk, wBu8bC4OfUY, _L0HVmAlCQI, and zwCtH_i1_S4 in the first sessions.",
            "All 43,230 Phase-B disagreement rows are packaged; later sessions shrink to the remaining active sources as smaller sources exhaust.",
        ],
        "ranking_assumption": manifest["selection_summary"]["ranking_assumption"],
        "honest_issues": [
            "This lane did not import into CVAT; it only provides a dry-run import script proof, because CVAT/network import is assigned to a separate local lane.",
            "The all-row Phase-B queue is 43,230 frames, so the owner queue is 68 w6 sessions rather than the rough several-session expectation.",
            "W6 is package/labeling fuel only. It does not change BALL promotion status or VERIFIED=0.",
        ],
        "next": "Import lane command: runs/lanes/w3_labelfactory_20260707/venv/bin/python runs/lanes/w6_labelpack_20260708/import_w6_labelpack_tasks.py --dry-run first, then remove --dry-run only in the import lane. Owner doc: cvat_upload/OWNER_SESSION_W6_20260708.md.",
        "draft_BUILD_CHECKLIST_bullet_TEXT": "[W6 LABELPACK 2026-07-08] Phase-B owner CVAT packages generated for 24/24 new clips: 43,230 SST disagreement frames across 68 ball sessions under cvat_upload/w6_labelpack_20260708/packages/, with HyUqT7zFiwk/wBu8bC4OfUY/_L0HVmAlCQI/zwCtH_i1_S4 front-loaded so union with w5 covers all 6 non-heldout sources. Protected-material programmatic check NO_MATCH for pwxNwFfYQlQ/vQhtz8l6VqU/Outdoor/Indoor patterns across all packages; structural validation passed with one CVAT 1.1 ball prelabel per image. Import not executed in this lane; use runs/lanes/w6_labelpack_20260708/import_w6_labelpack_tasks.py and cvat_upload/OWNER_SESSION_W6_20260708.md.",
        "validation": validation,
    }
    write_json(LANE / "report.json", report)
    write_markdown_report(report)


def write_markdown_report(report: dict[str, Any]) -> None:
    lines = [
        "# w6_labelpack_20260708 Report",
        "",
        f"Objective result: {report['objective_result']}",
        "",
        "## Session Table",
        "",
        "| name | frames | sources | error mix |",
        "|---|---:|---|---|",
    ]
    for item in report["session_table"]:
        lines.append(
            f"| {item['name']} | {item['frames']} | `{json.dumps(item['sources'], sort_keys=True)}` | "
            f"`{json.dumps(item['error_mix'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Per-Source Coverage",
            "",
            "```json",
            json.dumps(report["per_source_coverage"], indent=2, sort_keys=True),
            "```",
            "",
            "## Protected Check",
            "",
            "```text",
            report["protected_clip_check_output"]["programmatic"],
            "```",
            "",
            "## Honest Issues",
            "",
        ]
    )
    for issue in report["honest_issues"]:
        lines.append(f"- {issue}")
    lines.extend(["", "## Next", "", report["next"], ""])
    (LANE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_id_from_clip(clip_id: str) -> str:
    marker = "_rally_"
    if marker not in clip_id:
        raise ValueError(f"bad clip id: {clip_id}")
    return clip_id.split(marker, 1)[0]


def zip_files(zip_path: Path, files: list[Path], arc_root: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(arc_root).as_posix())


def write_zip_text(zip_path: Path, name: str, text: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, text)


def zip_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return sorted(zf.namelist())


def read_zip_text(path: Path, member: str) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(member).decode("utf-8")


def w5_source_counts() -> Counter:
    counts: Counter = Counter()
    if not W5_MANIFEST.is_file():
        return counts
    manifest = read_json(W5_MANIFEST)
    for item in manifest.get("ball_sessions", []):
        if isinstance(item, dict):
            source_counts = item.get("source_counts")
            if isinstance(source_counts, dict):
                counts.update({str(k): int(v) for k, v in source_counts.items()})
            else:
                per_source = int(item.get("frame_count", 0)) // max(1, len(item.get("source_ids", [])))
                for source in item.get("source_ids", []):
                    counts[str(source)] += per_source
    return counts


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
