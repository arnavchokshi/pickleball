"""Render RTMW3D/BODY skeleton inference overlays on source video pixels."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .court_auto_evidence import calibration_for_image_size
from .court_calibration import project_planar_points, project_world_points
from .schemas import CourtCalibration, validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_skeleton_video_overlay"
INDEX_FILENAME = "skeleton_video_overlay_index.json"
LOW_CONFIDENCE_THRESHOLD = 0.25
CAPTION = "RTMW3D lane-A inference, unverified — review copy"
TEXT_COLOR = (255, 255, 255)
LOW_CONFIDENCE_COLOR = (80, 80, 255)
SMOOTHING_COLOR = (0, 220, 255)
MISSING_COLOR = (120, 120, 120)
WARNING_COLOR = (60, 180, 255)

BONE_NAME_PAIRS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
)


@dataclass(frozen=True)
class ProjectedJoint:
    index: int
    xy: list[float] | None
    confidence: float | None
    low_confidence: bool
    smoothing_flag: str | None
    source: str


def load_skeleton_packet(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if payload.get("artifact_type") != "racketsport_skeleton3d":
        raise ValueError(f"{path} is not a racketsport_skeleton3d artifact")
    players = payload.get("players")
    if not isinstance(players, list):
        raise ValueError("skeleton3d.json must contain a players list")
    return payload


def load_court_calibration(path: str | Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", Path(path))
    if not isinstance(parsed, CourtCalibration):
        raise ValueError("court calibration artifact did not parse as CourtCalibration")
    return parsed


def color_for_player(player_id: int) -> tuple[int, int, int]:
    palette = [
        (60, 220, 255),
        (80, 200, 80),
        (255, 180, 80),
        (220, 120, 255),
        (255, 255, 80),
        (80, 120, 255),
        (120, 255, 200),
        (255, 120, 120),
    ]
    return palette[(int(player_id) - 1) % len(palette)]


def caption_extra_from_skeleton(skeleton: Mapping[str, Any]) -> str | None:
    provenance = skeleton.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    repair = provenance.get("skeleton_upright_repair")
    if not isinstance(repair, Mapping):
        return None
    caption = repair.get("overlay_caption_extra")
    if not isinstance(caption, str) or not caption.strip():
        return None
    return caption.strip()


def project_skeleton_joints(
    frame_payload: Mapping[str, Any],
    *,
    calibration: CourtCalibration,
    native_frame_payload: Mapping[str, Any] | None = None,
) -> list[ProjectedJoint]:
    """Project a skeleton frame to image points, preferring native 2D joints."""

    native_joints = _native_2d_joints(native_frame_payload) or _native_2d_joints(frame_payload)
    if native_joints is not None:
        return _project_native_2d_joints(frame_payload, native_joints)
    return _project_world_joints(frame_payload, calibration)


def render_skeleton_overlay(
    *,
    run_dir: str | Path,
    video_path: str | Path,
    out_dir: str | Path | None = None,
    max_frames: int | None = None,
    contact_sheet_frame_count: int = 24,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    cv2 = cv2_module or _cv2()
    run = Path(run_dir)
    video = Path(video_path)
    out = Path(out_dir) if out_dir is not None else run / "skeleton_video_overlay"
    out.mkdir(parents=True, exist_ok=True)

    skeleton_path = run / "skeleton3d.json"
    calibration_path = run / "court_calibration.json"
    skeleton = load_skeleton_packet(skeleton_path)
    calibration = load_court_calibration(calibration_path)
    native_2d_path, native_2d_packet = _load_native_2d_pose_packet(run, skeleton)
    native_2d_index = _frames_by_index(native_2d_packet, fps=float(skeleton.get("fps") or 30.0)) if native_2d_packet else {}
    skeleton_fps = float(skeleton.get("fps") or 30.0)
    frames_by_index = _frames_by_index(skeleton, fps=skeleton_fps)
    joint_names = _joint_names(skeleton)
    bone_pairs = bone_pairs_for_joint_names(joint_names)
    caption_extra = caption_extra_from_skeleton(skeleton)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or skeleton_fps or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and total_frames > 0:
        total_frames = min(total_frames, int(max_frames))

    overlay_path = out / "skeleton_overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open skeleton overlay writer: {overlay_path}")

    scaled_calibration = calibration_for_image_size(calibration, width=width, height=height)
    frame_index = 0
    drawn_player_frame_count = 0
    projected_joint_count = 0
    native_2d_joint_count = 0
    missing_joint_count = 0
    low_confidence_joint_count = 0
    smoothing_frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_stats = _draw_overlay_frame(
                cv2=cv2,
                frame=frame,
                frame_index=frame_index,
                fps=video_fps,
                frame_items=frames_by_index.get(frame_index, []),
                native_items=native_2d_index.get(frame_index, []),
                calibration=scaled_calibration,
                joint_names=joint_names,
                bone_pairs=bone_pairs,
                caption_extra=caption_extra,
            )
            drawn_player_frame_count += draw_stats["player_frame_count"]
            projected_joint_count += draw_stats["projected_joint_count"]
            native_2d_joint_count += draw_stats["native_2d_joint_count"]
            missing_joint_count += draw_stats["missing_joint_count"]
            low_confidence_joint_count += draw_stats["low_confidence_joint_count"]
            smoothing_frame_count += int(draw_stats["smoothing_frame"])
            writer.write(frame)
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    sheet_frames = select_contact_sheet_frames(
        run_dir=run,
        total_frames=frame_index if frame_index else total_frames,
        fps=video_fps,
        target_count=contact_sheet_frame_count,
    )
    contact_sheet_path = out / "skeleton_overlay_contact_sheet.jpg"
    sheet_summary = render_contact_sheet(
        cv2=cv2,
        run_dir=run,
        video_path=video,
        output_path=contact_sheet_path,
        frame_indices=sheet_frames,
        skeleton=skeleton,
        calibration=calibration,
        native_2d_packet=native_2d_packet,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "rendered",
        "run_dir": str(run),
        "video_path": str(video),
        "skeleton_path": str(skeleton_path),
        "calibration_path": str(calibration_path),
        "overlay_path": str(overlay_path),
        "contact_sheet_path": str(contact_sheet_path),
        "index_path": str(out / INDEX_FILENAME),
        "frame_count": frame_index,
        "video_fps": video_fps,
        "video_size": [width, height],
        "player_count": len(_players(skeleton)),
        "player_frame_count": drawn_player_frame_count,
        "projected_joint_count": projected_joint_count,
        "native_2d_joint_count": native_2d_joint_count,
        "missing_joint_count": missing_joint_count,
        "low_confidence_joint_count": low_confidence_joint_count,
        "smoothing_frame_count": smoothing_frame_count,
        "joint_count": len(joint_names),
        "bone_count": len(bone_pairs),
        "skeleton_source_model": str(skeleton.get("source_model", "")),
        "skeleton_world_frame": str(skeleton.get("world_frame", "")),
        "skeleton_provenance_lane": _provenance_lane(skeleton),
        "native_2d_pose_path": str(native_2d_path) if native_2d_path is not None else None,
        "contact_sheet_frame_indices": sheet_frames,
        "contact_sheet": sheet_summary,
        "caption": CAPTION,
        "caption_extra": caption_extra,
        "qualitative_status": "review_copy_not_gate_verified",
        "reads_cvat_labels": False,
        "not_ground_truth": True,
    }
    _write_json(out / INDEX_FILENAME, summary)
    return summary


def render_contact_sheet(
    *,
    cv2: Any,
    run_dir: Path,
    video_path: Path,
    output_path: Path,
    frame_indices: Sequence[int],
    skeleton: Mapping[str, Any],
    calibration: CourtCalibration,
    native_2d_packet: Mapping[str, Any] | None = None,
    thumbnail_width: int = 320,
) -> dict[str, Any]:
    import numpy as np

    skeleton_fps = float(skeleton.get("fps") or 30.0)
    frames_by_index = _frames_by_index(skeleton, fps=skeleton_fps)
    native_2d_index = _frames_by_index(native_2d_packet, fps=skeleton_fps) if native_2d_packet else {}
    joint_names = _joint_names(skeleton)
    bone_pairs = bone_pairs_for_joint_names(joint_names)
    caption_extra = caption_extra_from_skeleton(skeleton)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video for contact sheet: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or skeleton_fps or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    scaled_calibration = calibration_for_image_size(calibration, width=width, height=height)
    thumbnails: list[Any] = []
    used_indices: list[int] = []
    try:
        for frame_index in frame_indices:
            if frame_index < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            _draw_overlay_frame(
                cv2=cv2,
                frame=frame,
                frame_index=int(frame_index),
                fps=fps,
                frame_items=frames_by_index.get(int(frame_index), []),
                native_items=native_2d_index.get(int(frame_index), []),
                calibration=scaled_calibration,
                joint_names=joint_names,
                bone_pairs=bone_pairs,
                caption_extra=caption_extra,
            )
            cv2.putText(frame, f"sheet frame {int(frame_index)}", (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)
            thumb_height = max(1, int(round(height * (thumbnail_width / max(width, 1)))))
            thumbnails.append(cv2.resize(frame, (thumbnail_width, thumb_height)))
            used_indices.append(int(frame_index))
    finally:
        cap.release()

    if not thumbnails:
        raise RuntimeError("no contact-sheet frames could be rendered")

    cols = min(6, max(1, math.ceil(math.sqrt(len(thumbnails)))))
    rows = math.ceil(len(thumbnails) / cols)
    thumb_h, thumb_w = thumbnails[0].shape[:2]
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbnails):
        row = index // cols
        col = index % cols
        sheet[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"cannot write skeleton contact sheet: {output_path}")
    return {
        "path": str(output_path),
        "frame_indices": used_indices,
        "thumbnail_count": len(thumbnails),
        "grid": [cols, rows],
    }


def select_contact_sheet_frames(
    *,
    run_dir: str | Path,
    total_frames: int,
    fps: float,
    target_count: int = 24,
) -> list[int]:
    run = Path(run_dir)
    total = max(0, int(total_frames))
    if total == 0 or target_count <= 0:
        return []
    contact_frames = _contact_frame_indices(run / "contact_windows.json", fps=fps, total_frames=total)
    jitter_frames = _jitter_frame_indices(run, total_frames=total)
    even_frames = _evenly_sampled_indices(total, min(target_count, total))

    contact_quota = min(max(1, target_count // 3), target_count)
    jitter_quota = min(max(0, target_count // 3), target_count - contact_quota)
    selected: list[int] = []
    selected.extend(_select_evenly_from_sorted(contact_frames, contact_quota))
    selected.extend(_select_evenly_from_sorted(jitter_frames, jitter_quota))
    selected.extend(even_frames)
    return _dedupe_bounded(selected, total_frames=total, target_count=target_count)


def bone_pairs_for_joint_names(joint_names: Sequence[str]) -> list[tuple[int, int]]:
    by_name = {_normalize_joint_name(name): index for index, name in enumerate(joint_names)}
    pairs: list[tuple[int, int]] = []
    for left, right in BONE_NAME_PAIRS:
        left_index = by_name.get(left)
        right_index = by_name.get(right)
        if left_index is not None and right_index is not None:
            pairs.append((left_index, right_index))
    return pairs


def _draw_overlay_frame(
    *,
    cv2: Any,
    frame: Any,
    frame_index: int,
    fps: float,
    frame_items: Sequence[Mapping[str, Any]],
    native_items: Sequence[Mapping[str, Any]],
    calibration: CourtCalibration,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[int, int]],
    caption_extra: str | None = None,
) -> dict[str, Any]:
    native_by_player = {_player_id(item): item for item in native_items}
    projected_joint_count = 0
    native_2d_joint_count = 0
    missing_joint_count = 0
    low_confidence_joint_count = 0
    smoothing_frame = False
    for item in frame_items:
        player_id = _player_id(item)
        player_color = color_for_player(player_id)
        native_item = native_by_player.get(player_id)
        projected = project_skeleton_joints(item["frame"], calibration=calibration, native_frame_payload=native_item.get("frame") if native_item else None)
        projected_joint_count += sum(1 for joint in projected if joint.xy is not None)
        native_2d_joint_count += sum(1 for joint in projected if joint.xy is not None and joint.source == "native_2d")
        missing_joint_count += sum(1 for joint in projected if joint.xy is None)
        low_confidence_joint_count += sum(1 for joint in projected if joint.low_confidence)
        smoothing_frame = smoothing_frame or any(_is_smoothing_flag(joint.smoothing_flag) for joint in projected)
        _draw_player_skeleton(
            cv2=cv2,
            frame=frame,
            player_id=player_id,
            projected=projected,
            color=player_color,
            bone_pairs=bone_pairs,
        )
    _draw_frame_caption(cv2, frame, frame_index=frame_index, fps=fps, smoothing_frame=smoothing_frame, caption_extra=caption_extra)
    return {
        "player_frame_count": len(frame_items),
        "projected_joint_count": projected_joint_count,
        "native_2d_joint_count": native_2d_joint_count,
        "missing_joint_count": missing_joint_count,
        "low_confidence_joint_count": low_confidence_joint_count,
        "smoothing_frame": smoothing_frame,
    }


def _draw_player_skeleton(
    *,
    cv2: Any,
    frame: Any,
    player_id: int,
    projected: Sequence[ProjectedJoint],
    color: tuple[int, int, int],
    bone_pairs: Sequence[tuple[int, int]],
) -> None:
    for left, right in bone_pairs:
        if left >= len(projected) or right >= len(projected):
            continue
        left_xy = projected[left].xy
        right_xy = projected[right].xy
        if left_xy is None or right_xy is None:
            continue
        thickness = 1 if projected[left].low_confidence or projected[right].low_confidence else 2
        cv2.line(frame, _point(left_xy), _point(right_xy), color, thickness, cv2.LINE_AA)

    label_xy: tuple[int, int] | None = None
    for joint in projected:
        if joint.xy is None:
            continue
        confidence = joint.confidence if joint.confidence is not None else 0.5
        radius = max(2, min(7, int(round(2.5 + confidence * 4.5))))
        joint_color = _joint_color(color, confidence)
        thickness = 2 if joint.low_confidence else -1
        cv2.circle(frame, _point(joint.xy), radius, LOW_CONFIDENCE_COLOR if joint.low_confidence else joint_color, thickness, cv2.LINE_AA)
        if _is_smoothing_flag(joint.smoothing_flag):
            cv2.circle(frame, _point(joint.xy), radius + 3, SMOOTHING_COLOR, 1, cv2.LINE_AA)
        if label_xy is None:
            label_xy = _point(joint.xy)
    if label_xy is not None:
        cv2.putText(
            frame,
            f"P{player_id}",
            (label_xy[0] + 6, max(18, label_xy[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def _draw_frame_caption(
    cv2: Any,
    frame: Any,
    *,
    frame_index: int,
    fps: float,
    smoothing_frame: bool,
    caption_extra: str | None,
) -> None:
    t = float(frame_index) / fps if fps > 0 else 0.0
    cv2.putText(frame, CAPTION, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)
    cv2.putText(frame, f"frame {frame_index}  t={t:.3f}s", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)
    next_y = 72
    if caption_extra:
        cv2.putText(frame, caption_extra, (16, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WARNING_COLOR, 2)
        next_y += 24
    if smoothing_frame:
        cv2.putText(frame, "smoothing flags present", (16, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SMOOTHING_COLOR, 2)


def _project_native_2d_joints(frame_payload: Mapping[str, Any], native_joints: Sequence[Any]) -> list[ProjectedJoint]:
    projected: list[ProjectedJoint] = []
    conf = _sequence_or_empty(frame_payload.get("joint_conf"))
    smoothing = _sequence_or_empty(frame_payload.get("smoothing_flag") or frame_payload.get("smoothing_flags"))
    for index, joint in enumerate(native_joints):
        xy = _xy(joint)
        confidence = _confidence_at(conf, index)
        projected.append(
            ProjectedJoint(
                index=index,
                xy=xy,
                confidence=confidence,
                low_confidence=_is_low_confidence(confidence),
                smoothing_flag=_flag_at(smoothing, index),
                source="native_2d" if xy is not None else "missing",
            )
        )
    return projected


def _project_world_joints(frame_payload: Mapping[str, Any], calibration: CourtCalibration) -> list[ProjectedJoint]:
    joints = frame_payload.get("joints_world")
    if not isinstance(joints, list):
        return []
    conf = _sequence_or_empty(frame_payload.get("joint_conf"))
    smoothing = _sequence_or_empty(frame_payload.get("smoothing_flag") or frame_payload.get("smoothing_flags"))
    projected: list[ProjectedJoint] = []
    for index, joint in enumerate(joints):
        xy = _project_world_joint(calibration, joint)
        confidence = _confidence_at(conf, index)
        projected.append(
            ProjectedJoint(
                index=index,
                xy=xy,
                confidence=confidence,
                low_confidence=_is_low_confidence(confidence),
                smoothing_flag=_flag_at(smoothing, index),
                source="world_projection" if xy is not None else "missing",
            )
        )
    return projected


def _project_world_joint(calibration: CourtCalibration, joint: Any) -> list[float] | None:
    vector = _xyz(joint)
    if vector is None:
        return None
    x, y, z = vector
    try:
        ground = project_planar_points(calibration.homography, [[x, y]])[0]
        if math.isclose(z, 0.0, abs_tol=1e-9):
            return [float(ground[0]), float(ground[1])]
        pnp_ground, pnp_joint = project_world_points(
            calibration.extrinsics,
            calibration.intrinsics,
            [[x, y, 0.0], [x, y, z]],
        )
        return [
            float(ground[0]) + (float(pnp_joint[0]) - float(pnp_ground[0])),
            float(ground[1]) + (float(pnp_joint[1]) - float(pnp_ground[1])),
        ]
    except (ValueError, OverflowError):
        return None


def _frames_by_index(packet: Mapping[str, Any] | None, *, fps: float) -> dict[int, list[dict[str, Any]]]:
    if packet is None:
        return {}
    by_index: dict[int, list[dict[str, Any]]] = {}
    for player in _players(packet):
        player_id = _player_id(player)
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for ordinal, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                continue
            frame_index = _frame_index(frame, fps=fps, ordinal=ordinal)
            if frame_index is None:
                continue
            by_index.setdefault(frame_index, []).append({"player_id": player_id, "frame": frame})
    return by_index


def _load_native_2d_pose_packet(run_dir: Path, skeleton: Mapping[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate in _native_2d_candidates(run_dir, skeleton):
        if not candidate.is_file():
            continue
        try:
            payload = _read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if _packet_has_native_2d(payload):
            return candidate, payload
    return None, None


def _native_2d_candidates(run_dir: Path, skeleton: Mapping[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for value in _walk_values(skeleton.get("provenance", {})):
        if isinstance(value, str) and "2d" in value.lower() and value.lower().endswith(".json"):
            path = Path(value)
            candidates.append(path if path.is_absolute() else run_dir / path)
    for name in (
        "skeleton2d.json",
        "pose2d.json",
        "native_pose2d.json",
        "rtmw3d_pose2d.json",
        "keypoints2d.json",
    ):
        candidates.append(run_dir / name)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _packet_has_native_2d(packet: Mapping[str, Any]) -> bool:
    for player in _players(packet):
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if isinstance(frame, Mapping) and _native_2d_joints(frame) is not None:
                return True
    return False


def _native_2d_joints(payload: Mapping[str, Any] | None) -> Sequence[Any] | None:
    if payload is None:
        return None
    for key in ("joints_2d", "keypoints_2d", "joints_image", "keypoints_image", "pose2d"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _contact_frame_indices(path: Path, *, fps: float, total_frames: int) -> list[int]:
    if not path.is_file():
        return []
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    frames: list[int] = []
    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            for value in _event_frame_candidates(event, fps=fps):
                if 0 <= value < total_frames:
                    frames.append(value)
    return sorted(set(frames))


def _event_frame_candidates(event: Mapping[str, Any], *, fps: float) -> list[int]:
    candidates: list[int] = []
    frame = _int_value(event.get("frame") or event.get("frame_idx") or event.get("frame_index"))
    if frame is not None:
        candidates.append(frame)
    t = _float_value(event.get("t"))
    if t is not None:
        candidates.append(int(round(t * fps)))
    window = event.get("window")
    if isinstance(window, Mapping):
        t0 = _float_value(window.get("t0"))
        t1 = _float_value(window.get("t1"))
        if t0 is not None and t1 is not None:
            center_t = (t0 + t1) / 2.0
            candidates.append(int(round(center_t * fps)))
    return candidates


def _jitter_frame_indices(run_dir: Path, *, total_frames: int) -> list[int]:
    frames: list[int] = []
    for path in _pose_audit_candidates(run_dir):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        frames.extend(_extract_jitter_frames(payload, total_frames=total_frames))
        if frames:
            break
    return sorted(set(frames))


def _pose_audit_candidates(run_dir: Path) -> list[Path]:
    direct_names = [
        "pose_jitter_audit.json",
        "pose_temporal_jitter_audit.json",
        "skeleton_pose_audit.json",
        "skeleton_jitter_audit.json",
        "body_pose_audit.json",
    ]
    candidates = [run_dir / name for name in direct_names]
    candidates.extend(sorted(run_dir.glob("*jitter*.json")))
    candidates.extend(sorted(run_dir.glob("*pose*audit*.json")))
    return candidates


def _extract_jitter_frames(payload: Mapping[str, Any], *, total_frames: int) -> list[int]:
    for key in ("worst_jitter_frames", "worst_frames", "highest_jitter_frames", "frames"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        frames: list[int] = []
        for item in value:
            if isinstance(item, int):
                frame = item
            elif isinstance(item, Mapping):
                frame = _int_value(item.get("frame") or item.get("frame_idx") or item.get("frame_index"))
                if frame is None:
                    continue
            else:
                continue
            if 0 <= frame < total_frames:
                frames.append(frame)
        if frames:
            return frames
    return []


def _select_evenly_from_sorted(values: Sequence[int], target_count: int) -> list[int]:
    unique = sorted(set(int(value) for value in values))
    if target_count <= 0 or not unique:
        return []
    if len(unique) <= target_count:
        return unique
    positions = [round(index * (len(unique) - 1) / max(target_count - 1, 1)) for index in range(target_count)]
    return [unique[int(position)] for position in positions]


def _evenly_sampled_indices(total_frames: int, target_count: int) -> list[int]:
    if total_frames <= 0 or target_count <= 0:
        return []
    if total_frames <= target_count:
        return list(range(total_frames))
    return [
        int(round(index * (total_frames - 1) / max(target_count - 1, 1)))
        for index in range(target_count)
    ]


def _dedupe_bounded(values: Sequence[int], *, total_frames: int, target_count: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for value in values:
        frame = int(value)
        if 0 <= frame < total_frames and frame not in seen:
            selected.append(frame)
            seen.add(frame)
        if len(selected) >= target_count:
            break
    if len(selected) < min(target_count, total_frames):
        for frame in _evenly_sampled_indices(total_frames, min(target_count, total_frames)):
            if frame not in seen:
                selected.append(frame)
                seen.add(frame)
            if len(selected) >= target_count:
                break
    return sorted(selected)


def _joint_color(player_color: tuple[int, int, int], confidence: float) -> tuple[int, int, int]:
    conf = max(0.0, min(1.0, float(confidence)))
    return tuple(int(round(channel * (0.55 + 0.45 * conf) + 255 * (0.10 * conf))) for channel in player_color)


def _point(xy: Sequence[float]) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def _xy(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    x = _float_value(value[0])
    y = _float_value(value[1])
    if x is None or y is None:
        return None
    return [x, y]


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    parsed = [_float_value(value[index]) for index in range(3)]
    if any(component is None for component in parsed):
        return None
    return [float(component) for component in parsed]


def _confidence_at(values: Sequence[Any], index: int) -> float | None:
    if index >= len(values):
        return None
    return _float_value(values[index])


def _flag_at(values: Sequence[Any], index: int) -> str | None:
    if index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    return str(value)


def _is_low_confidence(value: float | None) -> bool:
    return value is not None and value < LOW_CONFIDENCE_THRESHOLD


def _is_smoothing_flag(value: str | None) -> bool:
    return value is not None and value not in {"", "none", "None", "false", "False"}


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list | tuple) else []


def _players(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = packet.get("players")
    if not isinstance(players, list):
        return []
    return [player for player in players if isinstance(player, Mapping)]


def _player_id(payload: Mapping[str, Any]) -> int:
    value = payload.get("player_id", payload.get("id", 0))
    parsed = _int_value(value)
    return parsed if parsed is not None else 0


def _frame_index(frame: Mapping[str, Any], *, fps: float, ordinal: int) -> int | None:
    for key in ("frame_idx", "frame_index", "frame"):
        value = _int_value(frame.get(key))
        if value is not None:
            return value
    t = _float_value(frame.get("t"))
    if t is not None:
        return int(round(t * fps))
    return ordinal


def _joint_names(packet: Mapping[str, Any]) -> list[str]:
    value = packet.get("joint_names")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _normalize_joint_name(name: str) -> str:
    text = str(name).strip().lower().replace("-", "_")
    return re.sub(r"_+", "_", text)


def _provenance_lane(skeleton: Mapping[str, Any]) -> str | None:
    provenance = skeleton.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    lane = provenance.get("lane")
    return None if lane is None else str(lane)


def _walk_values(value: Any) -> list[Any]:
    values: list[Any] = [value]
    if isinstance(value, Mapping):
        for child in value.values():
            values.extend(_walk_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_values(child))
    return values


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for skeleton video overlay rendering") from exc
    return cv2


__all__ = [
    "ARTIFACT_TYPE",
    "CAPTION",
    "caption_extra_from_skeleton",
    "INDEX_FILENAME",
    "LOW_CONFIDENCE_THRESHOLD",
    "ProjectedJoint",
    "bone_pairs_for_joint_names",
    "color_for_player",
    "load_court_calibration",
    "load_skeleton_packet",
    "project_skeleton_joints",
    "render_contact_sheet",
    "render_skeleton_overlay",
    "select_contact_sheet_frames",
]
