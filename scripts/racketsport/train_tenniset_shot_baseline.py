#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class TenniSetSample:
    sample_id: str
    label: str
    joints_path: Path
    pos_path: Path
    shuttle_path: Path


def build_dataset_index(root: str | Path, split: str) -> tuple[list[TenniSetSample], list[str]]:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise ValueError(f"missing split directory: {split_dir}")
    labels = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    samples: list[TenniSetSample] = []
    for label in labels:
        label_dir = split_dir / label
        for joints_path in sorted(label_dir.glob("*_joints.npy")):
            sample_id = joints_path.name.removesuffix("_joints.npy")
            pos_path = label_dir / f"{sample_id}_pos.npy"
            shuttle_path = label_dir / f"{sample_id}_shuttle.npy"
            if not pos_path.is_file() or not shuttle_path.is_file():
                continue
            samples.append(
                TenniSetSample(
                    sample_id=sample_id,
                    label=label,
                    joints_path=joints_path,
                    pos_path=pos_path,
                    shuttle_path=shuttle_path,
                )
            )
    return samples, labels


def load_feature_vector(sample: TenniSetSample, *, seq_len: int) -> np.ndarray:
    np = _numpy()
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    joints = np.load(sample.joints_path).astype(np.float32, copy=False)
    pos = np.load(sample.pos_path).astype(np.float32, copy=False)
    shuttle = np.load(sample.shuttle_path).astype(np.float32, copy=False)
    if joints.ndim != 4 or pos.ndim != 3 or shuttle.ndim != 2:
        raise ValueError(f"unexpected feature dimensions for {sample.sample_id}")
    if not (joints.shape[0] == pos.shape[0] == shuttle.shape[0]):
        raise ValueError(f"modality length mismatch for {sample.sample_id}")
    sequence = np.concatenate(
        [
            joints.reshape(joints.shape[0], -1),
            pos.reshape(pos.shape[0], -1),
            shuttle.reshape(shuttle.shape[0], -1),
        ],
        axis=1,
    )
    return _resample_sequence(sequence, seq_len).reshape(-1).astype(np.float32, copy=False)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], *, class_count: int) -> float:
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    scores = []
    for label in range(class_count):
        tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred != label)
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return round(float(sum(scores) / len(scores)), 6)


def _resample_sequence(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    np = _numpy()
    if sequence.shape[0] == seq_len:
        return sequence
    if sequence.shape[0] == 1:
        return np.repeat(sequence, seq_len, axis=0)
    source_x = np.linspace(0.0, 1.0, num=sequence.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=seq_len, dtype=np.float32)
    output = np.empty((seq_len, sequence.shape[1]), dtype=np.float32)
    for index in range(sequence.shape[1]):
        output[:, index] = np.interp(target_x, source_x, sequence[:, index])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a small TenniSet external shot-class baseline.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing Train/Val/Test folders.")
    parser.add_argument("--out", type=Path, required=True, help="Output metrics JSON.")
    parser.add_argument("--checkpoint", type=Path, help="Optional PyTorch checkpoint output.")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    try:
        payload = train_external_baseline(
            data_root=args.data_root,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device_name=args.device,
            checkpoint_path=args.checkpoint,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: TenniSet training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def train_external_baseline(
    *,
    data_root: Path,
    seq_len: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    np = _numpy()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_samples, labels = build_dataset_index(data_root, "Train")
    val_samples, val_labels = build_dataset_index(data_root, "Val")
    test_samples, test_labels = build_dataset_index(data_root, "Test")
    if labels != val_labels or labels != test_labels:
        raise ValueError("Train/Val/Test labels do not match")

    label_to_id = {label: index for index, label in enumerate(labels)}
    x_train, y_train = _load_matrix(train_samples, label_to_id, seq_len=seq_len)
    x_val, y_val = _load_matrix(val_samples, label_to_id, seq_len=seq_len)
    x_test, y_test = _load_matrix(test_samples, label_to_id, seq_len=seq_len)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, len(labels)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_pred = _predict(model, x_val, device=device, batch_size=batch_size)
        val_acc = _accuracy(y_val.tolist(), val_pred)
        val_f1 = macro_f1(y_val.tolist(), val_pred, class_count=len(labels))
        history.append({"epoch": epoch, "val_accuracy": val_acc, "val_macro_f1": val_f1})
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred = _predict(model, x_test, device=device, batch_size=batch_size)
    test_acc = _accuracy(y_test.tolist(), test_pred)
    test_f1 = macro_f1(y_test.tolist(), test_pred, class_count=len(labels))

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "labels": labels,
                "seq_len": seq_len,
                "feature_mean": mean,
                "feature_std": std,
            },
            checkpoint_path,
        )

    return {
        "schema_version": 1,
        "dataset": "TenniSet external transfer baseline",
        "status": "external_source_domain_only",
        "warning": "These metrics are tennis-source-domain metrics, not pickleball accuracy.",
        "labels": labels,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "model": {
            "name": "tenniset_mlp_pose_ball_baseline",
            "seq_len": seq_len,
            "epochs": epochs,
            "best_val_macro_f1": best_val,
        },
        "test": {
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "predicted_counts": _counts(test_pred, labels),
            "truth_counts": _counts(y_test.tolist(), labels),
        },
        "history_tail": history[-5:],
    }


def _load_matrix(
    samples: Sequence[TenniSetSample],
    label_to_id: dict[str, int],
    *,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    np = _numpy()
    x = np.stack([load_feature_vector(sample, seq_len=seq_len) for sample in samples])
    y = np.array([label_to_id[sample.label] for sample in samples], dtype=np.int64)
    return x.astype(np.float32, copy=False), y


def _predict(model: object, x: np.ndarray, *, device: object, batch_size: int) -> list[int]:
    import torch

    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred = torch.argmax(model(xb), dim=1).cpu().tolist()
            predictions.extend(int(item) for item in pred)
    return predictions


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for TenniSet shot baseline training") from exc
    return np


def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return round(sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == pred) / len(y_true), 6)


def _counts(values: Sequence[int], labels: Sequence[str]) -> dict[str, int]:
    return {label: sum(1 for value in values if value == index) for index, label in enumerate(labels)}


if __name__ == "__main__":
    raise SystemExit(main())
