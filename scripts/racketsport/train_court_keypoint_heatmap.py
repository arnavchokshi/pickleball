from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


COURT_CORNER_TAXONOMY = {
    "near_left": "near_left_corner",
    "near_right": "near_right_corner",
    "far_right": "far_right_corner",
    "far_left": "far_left_corner",
}


def court_corner_keypoint_labels(payload: dict[str, Any]) -> dict[str, Any]:
    items = _items(payload)
    item = items[0]
    if not isinstance(item, dict):
        raise ValueError("court corner item must be an object")

    frame_name = item.get("frame")
    if not isinstance(frame_name, str) or not frame_name:
        raise ValueError("court corner item requires frame")

    corners = item.get("court_corners")
    if not isinstance(corners, dict):
        raise ValueError("court corner item requires court_corners")

    frame_dir = _frame_dir(payload)
    keypoints = {
        output_name: _corner_xy(corners, input_name)
        for input_name, output_name in COURT_CORNER_TAXONOMY.items()
    }
    return {"image_path": str(frame_dir / frame_name), "keypoints": keypoints}


def _items(payload: dict[str, Any]) -> list[Any]:
    annotation = payload.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError("court corner item annotation missing")
    items = annotation.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("court corner item missing")
    return items


def _frame_dir(payload: dict[str, Any]) -> Path:
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("frames block missing")
    frame_dir = frames.get("frame_dir")
    if not isinstance(frame_dir, str) or not frame_dir:
        raise ValueError("frames.frame_dir missing")
    return Path(frame_dir)


def _corner_xy(corners: dict[str, Any], key: str) -> list[float]:
    value = corners.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"court corner item missing {key}")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError(f"court corner item has non-numeric {key}")
    return [float(x), float(y)]


def load_real_corner_labels(root: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/labels/court_corners.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = court_corner_keypoint_labels(payload)
        row["clip"] = path.parent.parent.name
        labels.append(row)
    return labels


def heatmaps_for_points(
    points: dict[str, list[float] | tuple[float, float]],
    keypoint_names: list[str],
    width: int,
    height: int,
    *,
    sigma: float,
) -> tuple[Any, Any]:
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    heatmaps = np.zeros((len(keypoint_names), height, width), dtype=np.float32)
    masks = np.zeros_like(heatmaps, dtype=np.float32)
    for idx, name in enumerate(keypoint_names):
        if name not in points:
            continue
        x, y = float(points[name][0]), float(points[name][1])
        heatmaps[idx] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        masks[idx] = 1.0
    return heatmaps, masks


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _world_bounds() -> tuple[float, float, float, float]:
    xs = [point.world_xyz_m[0] for point in PICKLEBALL_KEYPOINTS]
    ys = [point.world_xyz_m[1] for point in PICKLEBALL_KEYPOINTS]
    return min(xs), max(xs), min(ys), max(ys)


def _bilinear(point: tuple[float, float], quad: list[tuple[float, float]]) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = _world_bounds()
    u = (point[0] - min_x) / (max_x - min_x)
    v = (point[1] - min_y) / (max_y - min_y)
    near_left, near_right, far_right, far_left = quad
    x = (1 - u) * (1 - v) * near_left[0] + u * (1 - v) * near_right[0] + u * v * far_right[0] + (1 - u) * v * far_left[0]
    y = (1 - u) * (1 - v) * near_left[1] + u * (1 - v) * near_right[1] + u * v * far_right[1] + (1 - u) * v * far_left[1]
    return x, y


def _random_quad(width: int, height: int, rng: random.Random) -> list[tuple[float, float]]:
    margin_x = width * rng.uniform(0.04, 0.18)
    near_y = height * rng.uniform(0.70, 0.93)
    far_y = height * rng.uniform(0.12, 0.45)
    near_left = (margin_x + rng.uniform(-6, 6), near_y + rng.uniform(-6, 6))
    near_right = (width - margin_x + rng.uniform(-6, 6), near_y + rng.uniform(-6, 6))
    far_width = width * rng.uniform(0.25, 0.70)
    far_center = width * rng.uniform(0.40, 0.60)
    far_left = (far_center - far_width / 2 + rng.uniform(-5, 5), far_y + rng.uniform(-5, 5))
    far_right = (far_center + far_width / 2 + rng.uniform(-5, 5), far_y + rng.uniform(-5, 5))
    return [near_left, near_right, far_right, far_left]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    keypoint_names = [point.name for point in PICKLEBALL_KEYPOINTS]
    width, height = args.image_width, args.image_height
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    real_labels = load_real_corner_labels(args.real_root) if args.real_root else []
    holdout = set(args.holdout_clip)
    train_real = [row for row in real_labels if row.get("clip") not in holdout]
    holdout_real = [row for row in real_labels if row.get("clip") in holdout] or real_labels[-1:]

    def make_model() -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, len(keypoint_names), 1),
        )

    def synthetic_batch(batch_size: int) -> tuple[Any, Any, Any]:
        images, targets, masks = [], [], []
        for _ in range(batch_size):
            quad = _random_quad(width, height, rng)
            image = Image.new("RGB", (width, height), tuple(rng.randint(35, 90) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            points = {
                point.name: _bilinear((point.world_xyz_m[0], point.world_xyz_m[1]), quad)
                for point in PICKLEBALL_KEYPOINTS
            }
            line_color = tuple(rng.randint(170, 255) for _ in range(3))
            for a, b in (
                ("near_left_corner", "near_right_corner"),
                ("near_right_corner", "far_right_corner"),
                ("far_right_corner", "far_left_corner"),
                ("far_left_corner", "near_left_corner"),
                ("near_nvz_left", "near_nvz_right"),
                ("far_nvz_left", "far_nvz_right"),
                ("net_left_sideline", "net_right_sideline"),
                ("near_baseline_center", "far_baseline_center"),
            ):
                draw.line([points[a], points[b]], fill=line_color, width=rng.randint(1, 3))
            arr = np.asarray(image, dtype=np.float32) / 255.0
            target, mask = heatmaps_for_points(points, keypoint_names, width, height, sigma=args.sigma)
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
            targets.append(torch.from_numpy(target))
            masks.append(torch.from_numpy(mask))
        return torch.stack(images), torch.stack(targets), torch.stack(masks)

    def real_batch(rows: list[dict[str, Any]]) -> tuple[Any, Any, Any] | None:
        if not rows:
            return None
        images, targets, masks = [], [], []
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            original_w, original_h = image.size
            image = image.resize((width, height))
            scaled = {
                name: [xy[0] * width / original_w, xy[1] * height / original_h]
                for name, xy in row["keypoints"].items()
            }
            arr = np.asarray(image, dtype=np.float32) / 255.0
            target, mask = heatmaps_for_points(scaled, keypoint_names, width, height, sigma=args.sigma)
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
            targets.append(torch.from_numpy(target))
            masks.append(torch.from_numpy(mask))
        return torch.stack(images), torch.stack(targets), torch.stack(masks)

    def evaluate(model: nn.Module, rows: list[dict[str, Any]], synthetic_batches: int = 4) -> dict[str, Any]:
        model.eval()
        real_errors: list[float] = []
        synthetic_errors: list[float] = []
        with torch.no_grad():
            for row in rows:
                batch = real_batch([row])
                if batch is None:
                    continue
                x, _, _ = [part.to(device) for part in batch]
                pred = model(x).detach().cpu()[0]
                image = Image.open(row["image_path"])
                sx, sy = width / image.size[0], height / image.size[1]
                for name, xy in row["keypoints"].items():
                    idx = keypoint_names.index(name)
                    flat = int(pred[idx].argmax())
                    py, px = divmod(flat, width)
                    real_errors.append(math.hypot(px - xy[0] * sx, py - xy[1] * sy))
            for _ in range(synthetic_batches):
                x, target, _ = synthetic_batch(args.batch_size)
                pred = model(x.to(device)).detach().cpu()
                for batch_i in range(pred.shape[0]):
                    for idx in range(pred.shape[1]):
                        pred_flat = int(pred[batch_i, idx].argmax())
                        target_flat = int(target[batch_i, idx].argmax())
                        py, px = divmod(pred_flat, width)
                        ty, tx = divmod(target_flat, width)
                        synthetic_errors.append(math.hypot(px - tx, py - ty))
        return {
            "real_corner_mean_px": mean(real_errors),
            "real_corner_count": len(real_errors),
            "synthetic_mean_px": mean(synthetic_errors),
            "synthetic_count": len(synthetic_errors),
        }

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    before = evaluate(model, holdout_real)
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        x, y, mask = synthetic_batch(args.batch_size)
        if train_real and epoch >= args.real_finetune_start_epoch:
            real = real_batch(train_real)
            if real is not None:
                rx, ry, rm = real
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)
                mask = torch.cat([mask, rm], dim=0)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(x) * mask, y * mask)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            row = evaluate(model, holdout_real)
            row.update({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    after = evaluate(model, holdout_real, synthetic_batches=8)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "court_keypoint_heatmap.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "keypoint_names": keypoint_names,
            "image_size": [width, height],
            "args": vars(args),
        },
        checkpoint,
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "court_keypoint_pretraining_run",
        "status": "trained_not_phase_verified",
        "checkpoint": str(checkpoint),
        "before": before,
        "after": after,
        "history": history,
        "real_train_count": len(train_real),
        "real_holdout_count": len(holdout_real),
        "note": "Synthetic pretraining plus limited real corner fine-tune; not a verified CAL-3 no-tap solver.",
    }
    (args.out / "court_keypoint_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight pickleball court-keypoint heatmap model.")
    parser.add_argument("--real-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-clip", action="append", default=["wolverine_mixed_0200_mid_steep_corner"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--real-finetune-start-epoch", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        summary = run_training(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"checkpoint": summary["checkpoint"], "before": summary["before"], "after": summary["after"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
