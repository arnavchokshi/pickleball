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

from threed.racketsport.autolabel import PROTOTYPE_GATE_CLIPS


PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32
TENNIS_RACKET_CLASS = 38


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
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for the YOLO26 teacher pass") from exc

    model = YOLO(str(checkpoint))
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
            for box_index, box in enumerate(result.boxes):
                cls = int(box.cls.item())
                score = float(box.conf.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
                base = _box_item(
                    frame=frame_path.name,
                    box_index=box_index,
                    cls=cls,
                    class_name=str(names.get(cls, cls)),
                    confidence=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                if cls == PERSON_CLASS:
                    players.append({**base, "review_id": f"person_{frame_path.stem}_{box_index}", "id": f"p{box_index + 1}"})
                elif cls == SPORTS_BALL_CLASS:
                    balls.append({**base, "review_id": f"ball_{frame_path.stem}_{box_index}", "xy_px": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)]})
                elif cls == TENNIS_RACKET_CLASS:
                    rackets.append(
                        {
                            **base,
                            "review_id": f"racket_{frame_path.stem}_{box_index}",
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
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
