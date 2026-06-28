"""CPU-only trainable baseline for DATA-5 shot-window features."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.shot_classifier import ALLOWED_SHOT_LABELS


PHASE_LABELS = {
    "serve": "serve",
    "overhead": "overhead_candidate",
    "third_shot_drop": "drop_candidate",
    "dink": "soft_game",
    "reset_block": "soft_game",
    "lob": "lob_candidate",
}


@dataclass(frozen=True)
class ShotFeatureSchema:
    feature_names: tuple[str, ...]

    @classmethod
    def from_windows(cls, windows: Sequence[Mapping[str, Any]]) -> "ShotFeatureSchema":
        names: set[str] = set()
        for window in windows:
            features = window.get("features")
            if isinstance(features, Mapping):
                _collect_numeric_paths(features, prefix="features", names=names)
        return cls(feature_names=tuple(sorted(names)))

    @property
    def presence_offset(self) -> int:
        return len(self.feature_names)

    @property
    def vector_size(self) -> int:
        return len(self.feature_names) * 2

    def vectorize(self, window: Mapping[str, Any]) -> np.ndarray:
        vector = np.zeros(self.vector_size, dtype=float)
        for index, name in enumerate(self.feature_names):
            value = _path_value(window, name)
            if _is_number(value):
                vector[index] = float(value)
                vector[self.presence_offset + index] = 1.0
        return vector

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "presence_offset": self.presence_offset,
            "vector_size": self.vector_size,
        }


def phase_label_for_shot(label: str) -> str:
    return PHASE_LABELS.get(label, "normal_hit")


def abstract_shot_label(label: str) -> str:
    if label.startswith("fh_"):
        return "fh_shot"
    if label.startswith("bh_"):
        return "bh_shot"
    return label


def abstract_prediction(
    top2: Sequence[tuple[str, float]],
    *,
    exact_min_confidence: float = 0.65,
    family_min_confidence: float = 0.75,
) -> dict[str, Any]:
    if not top2:
        return {
            "type": "unknown",
            "type_conf": 0.0,
            "gated": True,
            "abstraction_level": "none",
        }
    ranked = [(str(label), float(score)) for label, score in top2]
    label, score = ranked[0]
    if score >= exact_min_confidence:
        return {
            "type": label,
            "type_conf": score,
            "gated": False,
            "abstraction_level": "specific",
        }

    family_scores: dict[str, float] = {}
    for candidate, candidate_score in ranked:
        family = abstract_shot_label(candidate)
        family_scores[family] = family_scores.get(family, 0.0) + candidate_score
    family_label, family_score = max(family_scores.items(), key=lambda item: item[1])
    if family_score >= family_min_confidence:
        return {
            "type": family_label,
            "type_conf": family_score,
            "specific_type_candidate": label,
            "gated": False,
            "abstraction_level": "family",
        }
    return {
        "type": "unknown",
        "type_conf": score,
        "specific_type_candidate": label,
        "gated": True,
        "abstraction_level": "none",
    }


def train_shot_window_baseline(*, manifest_path: str | Path) -> dict[str, Any]:
    manifest = _read_json_object(Path(manifest_path), "shot dataset manifest")
    manifest_dir = Path(manifest_path).parent
    samples = _load_samples(manifest, manifest_dir)
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    if not train_samples:
        raise ValueError("shot baseline requires at least one train sample")

    schema = ShotFeatureSchema.from_windows([sample["window"] for sample in train_samples])
    if schema.vector_size == 0:
        raise ValueError("shot baseline requires numeric feature values")

    centroids = _centroids(train_samples, schema)
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_samples = [sample for sample in samples if sample["split"] == split]
        splits[split] = _score_split(split_samples, schema, centroids)

    return {
        "schema_version": 1,
        "status": "trainable_baseline_not_poseconv3d_or_bst",
        "dataset_id": manifest.get("dataset_id", ""),
        "model": {
            "name": "shot_window_centroid_baseline",
            "feature_schema": schema.to_payload(),
            "trained_labels": sorted(centroids),
            "notes": [
                "CPU centroid baseline for DATA-5 sanity checks only",
                "Does not train or replace PoseConv3D, PoseC3D, BST, or H100 fusion gates",
            ],
        },
        "splits": splits,
    }


def _collect_numeric_paths(value: Mapping[str, Any], *, prefix: str, names: set[str]) -> None:
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, Mapping):
            _collect_numeric_paths(child, prefix=path, names=names)
        elif _is_number(child):
            names.add(path)


def _load_samples(manifest: Mapping[str, Any], manifest_dir: Path) -> list[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("shot dataset manifest entries must be an array")
    samples: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"entries/{index} must be an object")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"entries/{index}/path is required")
        if Path(path_value).is_absolute() or ".." in Path(path_value).parts:
            raise ValueError(f"entries/{index}/path must be safe and relative")
        feature_path = manifest_dir / path_value
        window = _read_json_object(feature_path, f"feature window {path_value}")
        label = str(entry.get("shot_label", window.get("truth", {}).get("shot_label", "")))
        if label not in ALLOWED_SHOT_LABELS:
            raise ValueError(f"entries/{index}/shot_label is not a known pickleball shot label")
        split = str(entry.get("split", ""))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"entries/{index}/split must be train, val, or test")
        samples.append(
            {
                "id": str(entry.get("id", f"sample_{index:03d}")),
                "split": split,
                "label": label,
                "window": window,
            }
        )
    return samples


def _centroids(samples: Sequence[Mapping[str, Any]], schema: ShotFeatureSchema) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["label"]), []).append(schema.vectorize(sample["window"]))
    return {label: np.mean(np.vstack(vectors), axis=0) for label, vectors in grouped.items()}


def _score_split(
    samples: Sequence[Mapping[str, Any]],
    schema: ShotFeatureSchema,
    centroids: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    for sample in samples:
        ranked = _rank_labels(schema.vectorize(sample["window"]), centroids)
        prediction = ranked[0][0] if ranked else "unknown"
        y_true.append(str(sample["label"]))
        y_pred.append(prediction)
        predictions.append(
            {
                "id": sample["id"],
                "truth": str(sample["label"]),
                "type": prediction,
                "top2": [{"type": label, "confidence": score} for label, score in ranked[:2]],
                "phase": {"type": phase_label_for_shot(prediction)},
                "abstract": abstract_prediction(ranked[:2]),
            }
        )
    return {
        "sample_count": len(samples),
        "accuracy": _accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred),
        "predictions": predictions,
    }


def _rank_labels(vector: np.ndarray, centroids: Mapping[str, np.ndarray]) -> list[tuple[str, float]]:
    if not centroids:
        return []
    distances = [(label, float(np.linalg.norm(vector - centroid))) for label, centroid in centroids.items()]
    max_distance = max((distance for _label, distance in distances), default=0.0)
    if max_distance <= 0:
        return [(label, 1.0 if index == 0 else 0.0) for index, (label, _distance) in enumerate(sorted(distances))]
    scores = [(label, max(0.0, 1.0 - distance / max_distance)) for label, distance in distances]
    total = sum(score for _label, score in scores)
    if total > 0:
        scores = [(label, score / total) for label, score in scores]
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    if not y_true:
        return None
    return round(sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true), 6)


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    if not y_true:
        return None
    labels = sorted(set(y_true) | set(y_pred))
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return round(sum(f1s) / len(f1s), 6)


def _path_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = [
    "ShotFeatureSchema",
    "abstract_prediction",
    "abstract_shot_label",
    "phase_label_for_shot",
    "train_shot_window_baseline",
]
