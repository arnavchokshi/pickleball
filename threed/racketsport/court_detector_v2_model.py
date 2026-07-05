"""Lightweight multi-task model scaffold for court detector v2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MOBILENET_V3_COURT_KEYPOINT_REGRESSOR_ARCHITECTURE = "mobilenet_v3_small_regressor"


def make_court_detector_v2_model(*, keypoint_count: int, line_count: int, net_count: int) -> Any:
    if keypoint_count <= 0 or line_count <= 0 or net_count <= 0:
        raise ValueError("head counts must be positive")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CourtDetectorV2Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 96, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(96, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.keypoint_head = nn.Conv2d(64, keypoint_count, 1)
            self.line_head = nn.Conv2d(64, line_count, 1)
            self.net_head = nn.Conv2d(64, net_count, 1)
            self.visibility_head = nn.Linear(128, keypoint_count)

        def forward(self, x: Any) -> dict[str, Any]:
            features = self.encoder(x)
            decoded = self.decoder(features)
            if decoded.shape[-2:] != x.shape[-2:]:
                decoded = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
            pooled = torch.mean(features, dim=(-2, -1))
            return {
                "keypoint_heatmaps": self.keypoint_head(decoded),
                "line_masks": self.line_head(decoded),
                "net_masks": self.net_head(decoded),
                "visibility_logits": self.visibility_head(pooled),
            }

    return CourtDetectorV2Model()


def make_resnet50_court_keypoint_regressor(*, keypoint_count: int = 15, weights: Any = None) -> Any:
    """Build a ResNet50 keypoint-regression baseline.

    Output layout is `[x, y, visibility]` repeated per keypoint. Coordinates are
    intentionally raw regression outputs; training code is responsible for target
    normalization and loss scaling.
    """

    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be positive")

    import torch.nn as nn
    from torchvision.models import resnet50

    model = resnet50(weights=weights)
    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, keypoint_count * 3)
    model.court_keypoint_count = int(keypoint_count)
    model.court_keypoint_output_layout = "x_y_visibility_per_keypoint"
    return model


def make_mobilenet_v3_court_keypoint_regressor(*, keypoint_count: int = 15, weights: Any = None) -> Any:
    """Build a lightweight MobileNetV3-small keypoint-regression baseline."""

    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be positive")

    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=weights)
    final = model.classifier[-1]
    if not hasattr(final, "in_features"):
        raise ValueError("unexpected MobileNetV3 classifier layout")
    model.classifier[-1] = nn.Linear(int(final.in_features), keypoint_count * 3)
    model.court_keypoint_count = int(keypoint_count)
    model.court_keypoint_output_layout = "x_y_visibility_per_keypoint"
    return model


def train_mobilenet_v3_court_keypoint_regressor(
    *,
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    holdout_clip_names: set[str] | frozenset[str] | None = None,
    input_size: tuple[int, int] = (224, 224),
    epochs: int = 1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    pck_threshold_px: float = 5.0,
    gate_threshold: float = 0.95,
    freeze_backbone: bool = True,
) -> dict[str, Any]:
    """Train a lightweight MobileNetV3 direct court-keypoint regressor.

    This is an experiment artifact writer, not a CAL promotion path. The saved
    checkpoint is intentionally named for the CAL diagnostic scanner.
    """

    output_dir = Path(out_dir)
    checkpoint_path = output_dir / "mobilenet_v3_court_keypoint_regressor.pt"
    metrics_path = output_dir / "mobilenet_v3_court_keypoint_metrics.json"
    base = {
        "schema_version": 1,
        "artifact_type": "mobilenet_v3_court_keypoint_regressor_training_report",
        "status": "unavailable",
        "verified": False,
        "not_cal3_verified": True,
        "diagnostic_only": True,
        "promotes_calibration": False,
        "architecture": MOBILENET_V3_COURT_KEYPOINT_REGRESSOR_ARCHITECTURE,
        "coordinate_mode": "sigmoid_normalized_xy",
        "checkpoint": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "input_size": [int(input_size[0]), int(input_size[1])],
        "device": device,
        "pck_threshold_px": float(pck_threshold_px),
        "gate_threshold": float(gate_threshold),
    }
    if epochs <= 0:
        report = dict(base)
        report.update({"reason": "epochs_must_be_positive", "train_row_count": 0, "holdout_row_count": 0})
        return report
    if learning_rate <= 0.0:
        report = dict(base)
        report.update({"reason": "learning_rate_must_be_positive", "train_row_count": 0, "holdout_row_count": 0})
        return report

    usable_rows = [row for row in rows if _mobilenet_v3_row_is_trainable(row)]
    train_rows, holdout_rows = _mobilenet_v3_train_holdout_split(usable_rows, holdout_clip_names=holdout_clip_names)
    if not train_rows or not holdout_rows:
        report = dict(base)
        report.update(
            {
                "reason": "missing_train_or_holdout_rows",
                "usable_row_count": len(usable_rows),
                "train_row_count": len(train_rows),
                "holdout_row_count": len(holdout_rows),
            }
        )
        return report

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        report = dict(base)
        report.update(
            {
                "reason": "torch_unavailable",
                "error": str(exc),
                "train_row_count": len(train_rows),
                "holdout_row_count": len(holdout_rows),
            }
        )
        return report

    torch.manual_seed(int(seed))
    torch_device = torch.device(device)
    keypoint_names = _default_court_keypoint_names()
    model = make_mobilenet_v3_court_keypoint_regressor(keypoint_count=len(keypoint_names), weights=None)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("classifier."):
                parameter.requires_grad_(False)
    model.to(torch_device)
    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(learning_rate))

    epoch_losses: list[float] = []
    skipped_train_rows: list[dict[str, Any]] = []
    for _epoch in range(int(epochs)):
        row_losses: list[float] = []
        for row_index, row in enumerate(train_rows):
            try:
                tensor, source_size = _mobilenet_v3_input_tensor_from_row(
                    torch,
                    row=row,
                    image_path=Path(str(row["image_path"])),
                    input_size=input_size,
                    device=torch_device,
                )
                target_xy = _mobilenet_v3_target_xy_tensor(
                    torch,
                    row=row,
                    keypoint_names=keypoint_names,
                    source_size=source_size,
                    device=torch_device,
                )
            except Exception as exc:
                skipped_train_rows.append({"row_index": row_index, "reason": "row_load_failed", "error": str(exc)})
                continue
            optimizer.zero_grad(set_to_none=True)
            raw = model(tensor).reshape(1, len(keypoint_names), 3)
            xy_loss = F.mse_loss(raw[:, :, 0:2].sigmoid(), target_xy.unsqueeze(0))
            visibility_target = torch.ones((1, len(keypoint_names)), dtype=torch.float32, device=torch_device)
            visibility_loss = F.mse_loss(raw[:, :, 2].sigmoid(), visibility_target)
            loss = xy_loss + 0.05 * visibility_loss
            loss.backward()
            optimizer.step()
            row_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(round(sum(row_losses) / len(row_losses), 6) if row_losses else math.nan)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": MOBILENET_V3_COURT_KEYPOINT_REGRESSOR_ARCHITECTURE,
            "model_state_dict": model.state_dict(),
            "keypoint_names": keypoint_names,
            "input_size": [int(input_size[0]), int(input_size[1])],
            "coordinate_mode": "sigmoid_normalized_xy",
            "training": {
                "epoch_count": int(epochs),
                "freeze_backbone": bool(freeze_backbone),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
            },
        },
        checkpoint_path,
    )
    evaluation = evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint(
        checkpoint_path=checkpoint_path,
        rows=holdout_rows,
        device=device,
        pck_threshold_px=pck_threshold_px,
        gate_threshold=gate_threshold,
    )
    report = dict(base)
    report.update(
        {
            "status": "trained_not_cal3_verified",
            "reason": None,
            "usable_row_count": len(usable_rows),
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "train_clip_names": sorted({str(row.get("clip")) for row in train_rows}),
            "holdout_clip_names": sorted({str(row.get("clip")) for row in holdout_rows}),
            "training": {
                "epoch_count": int(epochs),
                "learning_rate": float(learning_rate),
                "freeze_backbone": bool(freeze_backbone),
                "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable_parameters)),
                "epoch_losses": epoch_losses,
                "final_loss": None if not epoch_losses or math.isnan(epoch_losses[-1]) else epoch_losses[-1],
                "skipped_train_rows": skipped_train_rows,
            },
            "evaluation": evaluation,
            "gate_passed": evaluation.get("gate_passed") is True,
            "promotion_blockers": _mobilenet_v3_training_promotion_blockers(evaluation),
            "notes": [
                "Direct MobileNetV3 regression is diagnostic evidence only.",
                "Training and holdout rows are split by clip name to avoid frame-copy leakage in this small trial.",
            ],
        }
    )
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint(
    *,
    checkpoint_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    device: str = "cpu",
    pck_threshold_px: float = 5.0,
    gate_threshold: float = 0.95,
) -> dict[str, Any]:
    """Score an explicit MobileNetV3 court-keypoint regressor checkpoint.

    The result is diagnostic-only. Even a passing PCK gate is not a CAL promotion
    until a separate no-tap calibration gate consumes it.
    """

    path = Path(checkpoint_path)
    base = _mobilenet_v3_eval_base_report(
        checkpoint_path=path,
        device=device,
        pck_threshold_px=pck_threshold_px,
        gate_threshold=gate_threshold,
    )
    if not path.is_file():
        return _mobilenet_v3_unavailable_report(base, reason="missing_checkpoint")
    if not rows:
        return _mobilenet_v3_unavailable_report(base, reason="missing_reviewed_rows")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        report = _mobilenet_v3_unavailable_report(base, reason="torch_unavailable")
        report["error"] = str(exc)
        return report

    try:
        payload = _torch_load_checkpoint(torch, path, map_location=device)
    except Exception as exc:
        report = _mobilenet_v3_unavailable_report(base, reason="checkpoint_load_failed")
        report["error"] = str(exc)
        return report

    if isinstance(payload, Mapping):
        state_dict = _mobilenet_v3_state_dict_from_payload(payload)
        keypoint_names = _mobilenet_v3_keypoint_names_from_payload(payload)
        input_size = _mobilenet_v3_input_size_from_payload(payload)
        architecture = str(payload.get("architecture") or MOBILENET_V3_COURT_KEYPOINT_REGRESSOR_ARCHITECTURE)
        coordinate_mode = str(payload.get("coordinate_mode") or "sigmoid_normalized_xy")
    else:
        state_dict = payload
        keypoint_names = _default_court_keypoint_names()
        input_size = (224, 224)
        architecture = MOBILENET_V3_COURT_KEYPOINT_REGRESSOR_ARCHITECTURE
        coordinate_mode = "sigmoid_normalized_xy"

    if architecture != MOBILENET_V3_COURT_KEYPOINT_REGRESSOR_ARCHITECTURE:
        report = _mobilenet_v3_unavailable_report(base, reason="unsupported_architecture")
        report["architecture"] = architecture
        return report
    if coordinate_mode != "sigmoid_normalized_xy":
        report = _mobilenet_v3_unavailable_report(base, reason="unsupported_coordinate_mode")
        report["architecture"] = architecture
        report["coordinate_mode"] = coordinate_mode
        return report
    if not isinstance(state_dict, Mapping):
        report = _mobilenet_v3_unavailable_report(base, reason="missing_model_state_dict")
        report["architecture"] = architecture
        return report

    try:
        torch_device = torch.device(device)
        model = make_mobilenet_v3_court_keypoint_regressor(keypoint_count=len(keypoint_names), weights=None)
        model.load_state_dict(_strip_state_dict_module_prefix(state_dict))
        model.to(torch_device)
        model.eval()
    except Exception as exc:
        report = _mobilenet_v3_unavailable_report(base, reason="model_load_failed")
        report["architecture"] = architecture
        report["coordinate_mode"] = coordinate_mode
        report["error"] = str(exc)
        return report

    errors_px: list[float] = []
    per_row: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    reviewed_row_count = 0
    with torch.no_grad():
        for index, row in enumerate(rows):
            if row.get("label_status", "reviewed") == "reviewed":
                reviewed_row_count += 1
            image_path = row.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                skipped_rows.append({"row_index": index, "reason": "missing_image_path"})
                continue
            labels = row.get("keypoints")
            if not isinstance(labels, Mapping):
                skipped_rows.append({"row_index": index, "reason": "missing_keypoints"})
                continue
            try:
                tensor, source_size = _mobilenet_v3_input_tensor_from_row(
                    torch,
                    row=row,
                    image_path=Path(image_path),
                    input_size=input_size,
                    device=torch_device,
                )
            except Exception as exc:
                skipped_rows.append({"row_index": index, "reason": "image_load_failed", "error": str(exc)})
                continue
            try:
                raw = model(tensor)
                decoded = _decode_sigmoid_normalized_xy(raw, keypoint_names=keypoint_names, source_size=source_size)
            except Exception as exc:
                skipped_rows.append({"row_index": index, "reason": "inference_failed", "error": str(exc)})
                continue
            row_errors = _court_keypoint_prediction_errors(decoded, labels=labels)
            if not row_errors:
                skipped_rows.append({"row_index": index, "reason": "no_matching_keypoint_labels"})
                continue
            errors_px.extend(row_errors)
            per_row.append(
                {
                    "row_index": index,
                    "clip": row.get("clip"),
                    "image_path": image_path,
                    "source_size": [int(source_size[0]), int(source_size[1])],
                    "keypoint_count": len(row_errors),
                    "mean_error_px": round(sum(row_errors) / len(row_errors), 6),
                    "median_error_px": round(_median(row_errors), 6),
                    "p95_error_px": round(_percentile(row_errors, 95.0), 6),
                }
            )

    if not errors_px:
        report = _mobilenet_v3_unavailable_report(base, reason="no_valid_evaluations")
        report.update(
            {
                "architecture": architecture,
                "coordinate_mode": coordinate_mode,
                "input_size": [int(input_size[0]), int(input_size[1])],
                "reviewed_row_count": reviewed_row_count,
                "skipped_rows": skipped_rows,
            }
        )
        return report

    pck = sum(1 for error in errors_px if error <= pck_threshold_px) / float(len(errors_px))
    gate_passed = pck >= gate_threshold
    promotion_blockers = ["diagnostic_only", "not_cal3_verified"]
    if not gate_passed:
        promotion_blockers.append("gate_failed")
    report = dict(base)
    report.update(
        {
            "status": "scored",
            "reason": None,
            "architecture": architecture,
            "coordinate_mode": coordinate_mode,
            "input_size": [int(input_size[0]), int(input_size[1])],
            "reviewed_row_count": int(reviewed_row_count),
            "evaluated_row_count": len(per_row),
            "evaluated_keypoint_count": len(errors_px),
            "mean_error_px": round(sum(errors_px) / len(errors_px), 6),
            "median_error_px": round(_median(errors_px), 6),
            "p95_error_px": round(_percentile(errors_px, 95.0), 6),
            f"pck_at_{_format_threshold_for_key(pck_threshold_px)}px": round(pck, 6),
            "gate_metric": f"pck_at_{_format_threshold_for_key(pck_threshold_px)}px",
            "gate_value": round(pck, 6),
            "gate_passed": gate_passed,
            "promotion_blockers": promotion_blockers,
            "per_row": per_row,
            "skipped_rows": skipped_rows,
        }
    )
    return report


def _mobilenet_v3_eval_base_report(
    *,
    checkpoint_path: Path,
    device: str,
    pck_threshold_px: float,
    gate_threshold: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "mobilenet_v3_court_keypoint_regressor_eval",
        "status": "unavailable",
        "reason": None,
        "checkpoint": str(checkpoint_path),
        "device": device,
        "verified": False,
        "not_cal3_verified": True,
        "diagnostic_only": True,
        "promotes_calibration": False,
        "pck_threshold_px": float(pck_threshold_px),
        "gate_threshold": float(gate_threshold),
        "gate_passed": False,
        "promotion_blockers": ["diagnostic_only", "not_cal3_verified", "gate_failed"],
    }


def _mobilenet_v3_row_is_trainable(row: Mapping[str, Any]) -> bool:
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path or not Path(image_path).is_file():
        return False
    labels = row.get("keypoints")
    if not isinstance(labels, Mapping):
        return False
    for name in _default_court_keypoint_names():
        value = labels.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
            return False
        if not isinstance(value[0], (int, float)) or not isinstance(value[1], (int, float)):
            return False
    return True


def _mobilenet_v3_train_holdout_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_clip_names: set[str] | frozenset[str] | None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if holdout_clip_names:
        holdout_names = {str(name) for name in holdout_clip_names}
        holdout = [row for row in rows if str(row.get("clip")) in holdout_names]
        train = [row for row in rows if str(row.get("clip")) not in holdout_names]
        return train, holdout
    clip_names = sorted({str(row.get("clip")) for row in rows})
    if len(clip_names) >= 2:
        holdout_name = clip_names[-1]
        holdout = [row for row in rows if str(row.get("clip")) == holdout_name]
        train = [row for row in rows if str(row.get("clip")) != holdout_name]
        return train, holdout
    if len(rows) >= 2:
        return list(rows[:-1]), [rows[-1]]
    return list(rows), []


def _mobilenet_v3_target_xy_tensor(
    torch: Any,
    *,
    row: Mapping[str, Any],
    keypoint_names: Sequence[str],
    source_size: tuple[int, int],
    device: Any,
) -> Any:
    labels = row.get("keypoints")
    if not isinstance(labels, Mapping):
        raise ValueError("row missing keypoints")
    width = max(float(source_size[0]), 1.0)
    height = max(float(source_size[1]), 1.0)
    values: list[list[float]] = []
    for name in keypoint_names:
        label = labels.get(name)
        if not isinstance(label, Sequence) or isinstance(label, (str, bytes)) or len(label) < 2:
            raise ValueError(f"row missing keypoint {name}")
        x = max(0.0, min(1.0, float(label[0]) / width))
        y = max(0.0, min(1.0, float(label[1]) / height))
        values.append([x, y])
    return torch.tensor(values, dtype=torch.float32, device=device)


def _mobilenet_v3_training_promotion_blockers(evaluation: Mapping[str, Any]) -> list[str]:
    blockers = ["diagnostic_only", "not_cal3_verified"]
    if evaluation.get("gate_passed") is not True:
        blockers.append("gate_failed")
    if evaluation.get("status") != "scored":
        blockers.append("missing_scored_holdout")
    return blockers


def _mobilenet_v3_unavailable_report(base: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    report = dict(base)
    report["status"] = "unavailable"
    report["reason"] = reason
    report["gate_passed"] = False
    blockers = list(report.get("promotion_blockers", []))
    if reason not in blockers:
        blockers.append(reason)
    report["promotion_blockers"] = blockers
    return report


def _torch_load_checkpoint(torch: Any, path: Path, *, map_location: str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _mobilenet_v3_state_dict_from_payload(payload: Mapping[str, Any]) -> Any:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if all(isinstance(key, str) for key in payload.keys()):
        return payload
    return None


def _mobilenet_v3_keypoint_names_from_payload(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("keypoint_names")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        names = [str(item) for item in raw if isinstance(item, str) and item]
        if names:
            return names
    return _default_court_keypoint_names()


def _default_court_keypoint_names() -> list[str]:
    from .court_keypoint_net import PICKLEBALL_KEYPOINTS

    return [point.name for point in PICKLEBALL_KEYPOINTS]


def _mobilenet_v3_input_size_from_payload(payload: Mapping[str, Any]) -> tuple[int, int]:
    raw = payload.get("input_size") or payload.get("image_size")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        width, height = raw
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return (int(width), int(height))
    return (224, 224)


def _strip_state_dict_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if not any(str(key).startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }


def _mobilenet_v3_input_tensor_from_row(
    torch: Any,
    *,
    row: Mapping[str, Any],
    image_path: Path,
    input_size: tuple[int, int],
    device: Any,
) -> tuple[Any, tuple[int, int]]:
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    source_size = _source_size_from_row(row, fallback=image.size)
    resized = image.resize(input_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, source_size


def _source_size_from_row(row: Mapping[str, Any], *, fallback: tuple[int, int]) -> tuple[int, int]:
    raw = row.get("source_video_size")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        width, height = raw
        if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
            return (int(round(width)), int(round(height)))
    return (int(fallback[0]), int(fallback[1]))


def _decode_sigmoid_normalized_xy(
    raw: Any,
    *,
    keypoint_names: Sequence[str],
    source_size: tuple[int, int],
) -> dict[str, list[float]]:
    if raw.ndim == 2:
        raw = raw[0]
    flat = raw.detach().float().cpu().reshape(-1)
    expected = len(keypoint_names) * 3
    if int(flat.numel()) != expected:
        raise ValueError(f"expected {expected} outputs for {len(keypoint_names)} keypoints; got {int(flat.numel())}")
    width, height = float(source_size[0]), float(source_size[1])
    decoded: dict[str, list[float]] = {}
    for index, name in enumerate(keypoint_names):
        x = float(flat[index * 3].sigmoid().item()) * width
        y = float(flat[index * 3 + 1].sigmoid().item()) * height
        visibility = float(flat[index * 3 + 2].sigmoid().item())
        decoded[str(name)] = [x, y, visibility]
    return decoded


def _court_keypoint_prediction_errors(
    predictions: Mapping[str, Sequence[float]],
    *,
    labels: Mapping[str, Any],
) -> list[float]:
    errors: list[float] = []
    for name, predicted in predictions.items():
        label = labels.get(name)
        if not isinstance(label, Sequence) or isinstance(label, (str, bytes)) or len(label) < 2:
            continue
        x, y = label[0], label[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        errors.append(math.hypot(float(predicted[0]) - float(x), float(predicted[1]) - float(y)))
    return errors


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _format_threshold_for_key(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "_")
