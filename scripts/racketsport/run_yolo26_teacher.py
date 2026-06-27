#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.autolabel import PROTOTYPE_GATE_CLIPS


PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32
TENNIS_RACKET_CLASS = 38
TARGET_CLASSES = {PERSON_CLASS, SPORTS_BALL_CLASS, TENNIS_RACKET_CLASS}

DEFAULT_PERSON_MIN_CONF = 0.45
DEFAULT_BALL_MIN_CONF = 0.25
DEFAULT_RACKET_MIN_CONF = 0.30
DEFAULT_MAX_PLAYERS_PER_FRAME = 4


@dataclass(frozen=True)
class DetectionCandidate:
    cls: int
    score: float
    xyxy: tuple[float, float, float, float]
    box_index: int


@dataclass(frozen=True)
class DetectionFilter:
    min_conf: float
    min_width_px: float
    min_height_px: float
    min_area_ratio: float
    max_area_ratio: float
    min_aspect: float
    max_aspect: float


def build_detection_filters(
    *,
    person_min_conf: float = DEFAULT_PERSON_MIN_CONF,
    ball_min_conf: float = DEFAULT_BALL_MIN_CONF,
    racket_min_conf: float = DEFAULT_RACKET_MIN_CONF,
    person_min_width_px: float = 8.0,
    person_min_height_px: float = 24.0,
    person_min_area_ratio: float = 0.001,
    person_max_area_ratio: float = 0.60,
    person_min_aspect: float = 0.12,
    person_max_aspect: float = 1.40,
    ball_min_width_px: float = 2.0,
    ball_min_height_px: float = 2.0,
    ball_min_area_ratio: float = 0.0,
    ball_max_area_ratio: float = 0.01,
    ball_min_aspect: float = 0.35,
    ball_max_aspect: float = 3.0,
    racket_min_width_px: float = 3.0,
    racket_min_height_px: float = 2.0,
    racket_min_area_ratio: float = 0.0,
    racket_max_area_ratio: float = 0.08,
    racket_min_aspect: float = 0.08,
    racket_max_aspect: float = 12.0,
) -> dict[int, DetectionFilter]:
    return {
        PERSON_CLASS: DetectionFilter(
            min_conf=person_min_conf,
            min_width_px=person_min_width_px,
            min_height_px=person_min_height_px,
            min_area_ratio=person_min_area_ratio,
            max_area_ratio=person_max_area_ratio,
            min_aspect=person_min_aspect,
            max_aspect=person_max_aspect,
        ),
        SPORTS_BALL_CLASS: DetectionFilter(
            min_conf=ball_min_conf,
            min_width_px=ball_min_width_px,
            min_height_px=ball_min_height_px,
            min_area_ratio=ball_min_area_ratio,
            max_area_ratio=ball_max_area_ratio,
            min_aspect=ball_min_aspect,
            max_aspect=ball_max_aspect,
        ),
        TENNIS_RACKET_CLASS: DetectionFilter(
            min_conf=racket_min_conf,
            min_width_px=racket_min_width_px,
            min_height_px=racket_min_height_px,
            min_area_ratio=racket_min_area_ratio,
            max_area_ratio=racket_max_area_ratio,
            min_aspect=racket_min_aspect,
            max_aspect=racket_max_aspect,
        ),
    }


def filter_frame_detections(
    candidates: list[DetectionCandidate],
    *,
    frame_width: int,
    frame_height: int,
    filters: dict[int, DetectionFilter],
    max_players_per_frame: int | None = DEFAULT_MAX_PLAYERS_PER_FRAME,
) -> list[DetectionCandidate]:
    players: list[DetectionCandidate] = []
    others: list[DetectionCandidate] = []
    for candidate in candidates:
        if not _passes_detection_filter(candidate, frame_width=frame_width, frame_height=frame_height, filters=filters):
            continue
        if candidate.cls == PERSON_CLASS:
            players.append(candidate)
        else:
            others.append(candidate)

    players.sort(key=lambda candidate: candidate.score, reverse=True)
    if max_players_per_frame is not None:
        players = players[: max(0, max_players_per_frame)]
    return players + others


def _passes_detection_filter(
    candidate: DetectionCandidate,
    *,
    frame_width: int,
    frame_height: int,
    filters: dict[int, DetectionFilter],
) -> bool:
    filter_config = filters.get(candidate.cls)
    if filter_config is None or candidate.score < filter_config.min_conf:
        return False
    x1, y1, x2, y2 = candidate.xyxy
    width = x2 - x1
    height = y2 - y1
    if width < filter_config.min_width_px or height < filter_config.min_height_px:
        return False
    aspect = width / height
    if aspect < filter_config.min_aspect or aspect > filter_config.max_aspect:
        return False
    frame_area = frame_width * frame_height
    area_ratio = (width * height / frame_area) if frame_area > 0 else 0.0
    return filter_config.min_area_ratio <= area_ratio <= filter_config.max_area_ratio


def run_yolo26_teacher(
    *,
    frames_root: Path,
    out: Path,
    checkpoint: Path,
    clips: list[str],
    imgsz: int = 960,
    conf: float = 0.18,
    iou: float = 0.6,
    device: str = "0",
    max_frames: int | None = None,
    person_min_conf: float = DEFAULT_PERSON_MIN_CONF,
    ball_min_conf: float = DEFAULT_BALL_MIN_CONF,
    racket_min_conf: float = DEFAULT_RACKET_MIN_CONF,
    max_players_per_frame: int = DEFAULT_MAX_PLAYERS_PER_FRAME,
    person_min_width_px: float = 8.0,
    person_min_height_px: float = 24.0,
    person_min_area_ratio: float = 0.001,
    person_max_area_ratio: float = 0.60,
    person_min_aspect: float = 0.12,
    person_max_aspect: float = 1.40,
    ball_min_width_px: float = 2.0,
    ball_min_height_px: float = 2.0,
    ball_min_area_ratio: float = 0.0,
    ball_max_area_ratio: float = 0.01,
    ball_min_aspect: float = 0.35,
    ball_max_aspect: float = 3.0,
    racket_min_width_px: float = 3.0,
    racket_min_height_px: float = 2.0,
    racket_min_area_ratio: float = 0.0,
    racket_max_area_ratio: float = 0.08,
    racket_min_aspect: float = 0.08,
    racket_max_aspect: float = 12.0,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for the YOLO26 teacher pass") from exc

    model = YOLO(str(checkpoint))
    filters = build_detection_filters(
        person_min_conf=person_min_conf,
        ball_min_conf=ball_min_conf,
        racket_min_conf=racket_min_conf,
        person_min_width_px=person_min_width_px,
        person_min_height_px=person_min_height_px,
        person_min_area_ratio=person_min_area_ratio,
        person_max_area_ratio=person_max_area_ratio,
        person_min_aspect=person_min_aspect,
        person_max_aspect=person_max_aspect,
        ball_min_width_px=ball_min_width_px,
        ball_min_height_px=ball_min_height_px,
        ball_min_area_ratio=ball_min_area_ratio,
        ball_max_area_ratio=ball_max_area_ratio,
        ball_min_aspect=ball_min_aspect,
        ball_max_aspect=ball_max_aspect,
        racket_min_width_px=racket_min_width_px,
        racket_min_height_px=racket_min_height_px,
        racket_min_area_ratio=racket_min_area_ratio,
        racket_max_area_ratio=racket_max_area_ratio,
        racket_min_aspect=racket_min_aspect,
        racket_max_aspect=racket_max_aspect,
    )
    out.mkdir(parents=True, exist_ok=True)
    clip_summaries: list[dict[str, Any]] = []
    for clip in clips:
        frame_paths = sorted((frames_root / clip).glob("frame_*.jpg"))
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        labels_dir = out / clip / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        players: list[dict[str, Any]] = []
        balls: list[dict[str, Any]] = []
        rackets: list[dict[str, Any]] = []
        for frame_path in frame_paths:
            result = model.predict(str(frame_path), imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
            names = result.names
            candidates: list[DetectionCandidate] = []
            for box_index, box in enumerate(result.boxes):
                cls = int(box.cls.item())
                if cls not in TARGET_CLASSES:
                    continue
                score = float(box.conf.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
                candidates.append(DetectionCandidate(cls=cls, score=score, xyxy=(x1, y1, x2, y2), box_index=box_index))
            frame_width, frame_height = _frame_size_from_result(result=result, candidates=candidates)
            for candidate in filter_frame_detections(
                candidates,
                frame_width=frame_width,
                frame_height=frame_height,
                filters=filters,
                max_players_per_frame=max_players_per_frame,
            ):
                cls = candidate.cls
                score = candidate.score
                x1, y1, x2, y2 = candidate.xyxy
                base = _box_item(
                    frame=frame_path.name,
                    box_index=candidate.box_index,
                    cls=cls,
                    class_name=str(names.get(cls, cls)),
                    confidence=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                if cls == PERSON_CLASS:
                    players.append({**base, "review_id": f"person_{frame_path.stem}_{candidate.box_index}", "id": f"p{candidate.box_index + 1}"})
                elif cls == SPORTS_BALL_CLASS:
                    balls.append({**base, "review_id": f"ball_{frame_path.stem}_{candidate.box_index}", "xy_px": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)]})
                elif cls == TENNIS_RACKET_CLASS:
                    rackets.append(
                        {
                            **base,
                            "review_id": f"racket_{frame_path.stem}_{candidate.box_index}",
                            "player_id": None,
                            "keypoints_px": [[round(x1, 2), round((y1 + y2) / 2.0, 2)], [round(x2, 2), round((y1 + y2) / 2.0, 2)]],
                            "label": "racket",
                        }
                    )
        written = []
        if _write_teacher_payload(labels_dir / "players.json", "players.json", players, checkpoint=checkpoint, teacher="YOLO26m/person"):
            written.append("players.json")
        if _write_teacher_payload(labels_dir / "ball.json", "ball.json", balls, checkpoint=checkpoint, teacher="YOLO26m/sports_ball"):
            written.append("ball.json")
        if _write_teacher_payload(labels_dir / "racket_pose.json", "racket_pose.json", rackets, checkpoint=checkpoint, teacher="YOLO26m/tennis_racket"):
            written.append("racket_pose.json")
        clip_summaries.append(
            {
                "clip": clip,
                "frames": len(frame_paths),
                "players": len(players),
                "balls": len(balls),
                "rackets": len(rackets),
                "written": written,
            }
        )
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_yolo26_teacher_run",
        "frames_root": str(frames_root),
        "out": str(out),
        "checkpoint": str(checkpoint),
        "not_ground_truth": True,
        "clips": clip_summaries,
    }
    (out / "yolo26_teacher_run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _frame_size_from_result(*, result: Any, candidates: list[DetectionCandidate]) -> tuple[int, int]:
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape and len(orig_shape) >= 2:
        return int(orig_shape[1]), int(orig_shape[0])
    if not candidates:
        return 0, 0
    max_x = max(candidate.xyxy[2] for candidate in candidates)
    max_y = max(candidate.xyxy[3] for candidate in candidates)
    return int(max_x), int(max_y)


def _box_item(
    *,
    frame: str,
    box_index: int,
    cls: int,
    class_name: str,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> dict[str, Any]:
    return {
        "frame": frame,
        "source": "yolo26m_teacher",
        "class_id": cls,
        "class_name": class_name,
        "confidence": round(confidence, 6),
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
        "status": "accepted" if confidence >= 0.85 else "uncertain",
        "teacher_model": "YOLO26m",
    }


def _write_teacher_payload(path: Path, target_file: str, items: list[dict[str, Any]], *, checkpoint: Path, teacher: str) -> bool:
    if not items:
        return False
    payload = {
        "schema_version": 1,
        "status": "teacher_draft_unverified",
        "source": {"teacher": teacher, "checkpoint": str(checkpoint)},
        "annotation": {"target_file": target_file, "items": items},
        "not_ground_truth": True,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO26m teacher detections over prototype frame packs.")
    parser.add_argument("--frames-root", type=Path, default=Path("runs/label_frames"))
    parser.add_argument("--out", type=Path, default=Path("runs/teachers/prototype_gate"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/checkpoints/body4d/yolo26/yolo26m.pt"))
    parser.add_argument("--clip", action="append", dest="clips")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--person-min-conf", type=float, default=DEFAULT_PERSON_MIN_CONF)
    parser.add_argument("--ball-min-conf", type=float, default=DEFAULT_BALL_MIN_CONF)
    parser.add_argument("--racket-min-conf", type=float, default=DEFAULT_RACKET_MIN_CONF)
    parser.add_argument("--max-players-per-frame", type=int, default=DEFAULT_MAX_PLAYERS_PER_FRAME)
    parser.add_argument("--person-min-width-px", type=float, default=8.0)
    parser.add_argument("--person-min-height-px", type=float, default=24.0)
    parser.add_argument("--person-min-area-ratio", type=float, default=0.001)
    parser.add_argument("--person-max-area-ratio", type=float, default=0.60)
    parser.add_argument("--person-min-aspect", type=float, default=0.12)
    parser.add_argument("--person-max-aspect", type=float, default=1.40)
    parser.add_argument("--ball-min-width-px", type=float, default=2.0)
    parser.add_argument("--ball-min-height-px", type=float, default=2.0)
    parser.add_argument("--ball-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--ball-max-area-ratio", type=float, default=0.01)
    parser.add_argument("--ball-min-aspect", type=float, default=0.35)
    parser.add_argument("--ball-max-aspect", type=float, default=3.0)
    parser.add_argument("--racket-min-width-px", type=float, default=3.0)
    parser.add_argument("--racket-min-height-px", type=float, default=2.0)
    parser.add_argument("--racket-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--racket-max-area-ratio", type=float, default=0.08)
    parser.add_argument("--racket-min-aspect", type=float, default=0.08)
    parser.add_argument("--racket-max-aspect", type=float, default=12.0)
    args = parser.parse_args()
    summary = run_yolo26_teacher(
        frames_root=args.frames_root,
        out=args.out,
        checkpoint=args.checkpoint,
        clips=args.clips or list(PROTOTYPE_GATE_CLIPS),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_frames=args.max_frames,
        person_min_conf=args.person_min_conf,
        ball_min_conf=args.ball_min_conf,
        racket_min_conf=args.racket_min_conf,
        max_players_per_frame=args.max_players_per_frame,
        person_min_width_px=args.person_min_width_px,
        person_min_height_px=args.person_min_height_px,
        person_min_area_ratio=args.person_min_area_ratio,
        person_max_area_ratio=args.person_max_area_ratio,
        person_min_aspect=args.person_min_aspect,
        person_max_aspect=args.person_max_aspect,
        ball_min_width_px=args.ball_min_width_px,
        ball_min_height_px=args.ball_min_height_px,
        ball_min_area_ratio=args.ball_min_area_ratio,
        ball_max_area_ratio=args.ball_max_area_ratio,
        ball_min_aspect=args.ball_min_aspect,
        ball_max_aspect=args.ball_max_aspect,
        racket_min_width_px=args.racket_min_width_px,
        racket_min_height_px=args.racket_min_height_px,
        racket_min_area_ratio=args.racket_min_area_ratio,
        racket_max_area_ratio=args.racket_max_area_ratio,
        racket_min_aspect=args.racket_min_aspect,
        racket_max_aspect=args.racket_max_aspect,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
