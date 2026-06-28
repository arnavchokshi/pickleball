from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from threed.racketsport.eval.metrics import NumericGate, evaluate_numeric_gates, metric
from threed.racketsport.eval.shot_event_eval import score_shot_events
from threed.racketsport.schemas import BallTrack, EvalMetric, RacketPose, RacketSportMetrics, Tracks


TRACK_PLAYER_LABEL_GATES = {
    "player_bbox_recall_iou50": NumericGate(
        name="label_check.track_player_bbox_recall_iou50",
        op=">=",
        threshold=0.9,
    ),
    "player_bbox_precision_iou50": NumericGate(
        name="label_check.track_player_bbox_precision_iou50",
        op=">=",
        threshold=0.9,
    ),
}

BALL_LABEL_GATES = {
    "ball_f1_at_10px": NumericGate(
        name="label_check.ball_f1_at_10px",
        op=">=",
        threshold=0.9,
    ),
    "ball_hidden_false_positive_rate": NumericGate(
        name="label_check.ball_hidden_false_positive_rate",
        op="<",
        threshold=0.05,
    ),
}
RACKET_LABEL_GATES = {
    "racket_face_angle_p90_error_deg": NumericGate(
        name="label_check.racket_face_angle_p90_error_deg",
        op="<=",
        threshold=5.0,
        unit="deg",
    ),
    "racket_contact_point_p90_error_cm": NumericGate(
        name="label_check.racket_contact_point_p90_error_cm",
        op="<=",
        threshold=3.0,
        unit="cm",
    ),
}
SHOT_LABEL_GATES = {
    "shot_label_macro_f1": NumericGate(
        name="label_check.shot_macro_f1",
        op=">=",
        threshold=0.65,
    ),
    "shot_label_top2_accuracy": NumericGate(
        name="label_check.shot_top2_accuracy",
        op=">=",
        threshold=0.85,
    ),
}


def score_player_bbox_labels(*, labels_dir: Path, tracks: Tracks) -> tuple[dict[str, EvalMetric], list[str]]:
    payload, note = _load_authoritative_payload(labels_dir / "players.json")
    if payload is None:
        return _not_measured_metrics(TRACK_PLAYER_LABEL_GATES), [note]

    labels = [
        {"frame_index": frame_index, "bbox": bbox}
        for item in _payload_items(payload)
        if _is_accepted_item(item)
        for frame_index, bbox in [(_frame_index(item), _bbox_xyxy(item))]
        if frame_index is not None and bbox is not None
    ]
    if not labels:
        return _not_measured_metrics(TRACK_PLAYER_LABEL_GATES), ["players.json has no reviewed player bbox labels"]

    predictions_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            predictions_by_frame.setdefault(frame_index, []).append((int(player.id), _tuple4(frame.bbox)))

    matched_predictions: set[tuple[int, int]] = set()
    true_positive = 0
    ious: list[float] = []
    for label in labels:
        candidates = predictions_by_frame.get(label["frame_index"], [])
        best_index: int | None = None
        best_iou = 0.0
        for candidate_index, (_player_id, bbox) in enumerate(candidates):
            if (label["frame_index"], candidate_index) in matched_predictions:
                continue
            candidate_iou = _iou(label["bbox"], bbox)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_index = candidate_index
        if best_index is not None and best_iou >= 0.5:
            matched_predictions.add((label["frame_index"], best_index))
            true_positive += 1
            ious.append(best_iou)

    evaluated_prediction_count = sum(len(predictions_by_frame.get(label["frame_index"], [])) for label in labels)
    recall = _ratio(true_positive, len(labels))
    precision = _ratio(true_positive, evaluated_prediction_count)
    metrics = evaluate_numeric_gates(
        {
            "player_bbox_recall_iou50": recall,
            "player_bbox_precision_iou50": precision,
        },
        TRACK_PLAYER_LABEL_GATES,
    )
    metrics["player_bbox_label_count"] = metric(
        value=len(labels),
        unit="labels",
        gate="label_check.track_player_bbox_labels_present",
        passed=None,
    )
    metrics["player_bbox_match_count"] = metric(
        value=true_positive,
        unit="matches",
        gate="label_check.track_player_bbox_matches_recorded",
        passed=None,
    )
    metrics["player_bbox_mean_iou"] = metric(
        value=_mean(ious),
        unit=None,
        gate="label_check.track_player_bbox_mean_iou_recorded",
        passed=None,
        status="measured" if ious else "not_measured",
    )
    return metrics, []


def score_ball_labels(*, labels_dir: Path, ball_track: BallTrack) -> tuple[dict[str, EvalMetric], list[str]]:
    payload, note = _load_authoritative_payload(labels_dir / "ball.json")
    if payload is None:
        return _not_measured_metrics(BALL_LABEL_GATES), [note]

    visible_labels: list[dict[str, Any]] = []
    hidden_labels: list[dict[str, Any]] = []
    for item in _payload_items(payload):
        if not _is_accepted_item(item):
            continue
        frame_index = _frame_index(item)
        if frame_index is None:
            continue
        xy = _xy_px(item)
        visible = _is_visible_ball_label(item, xy)
        if visible is True and xy is not None:
            visible_labels.append({"frame_index": frame_index, "xy": xy})
        elif visible is False:
            hidden_labels.append({"frame_index": frame_index})

    if not visible_labels and not hidden_labels:
        return _not_measured_metrics(BALL_LABEL_GATES), ["ball.json has no reviewed ball visibility labels"]

    samples_by_frame = {int(round(float(frame.t) * float(ball_track.fps))): frame for frame in ball_track.frames}
    true_positive = 0
    false_positive = 0
    false_negative = 0
    distances: list[float] = []
    for label in visible_labels:
        frame = samples_by_frame.get(label["frame_index"])
        if frame is None or not frame.visible:
            false_negative += 1
            continue
        distance = _distance(_tuple2(frame.xy), label["xy"])
        distances.append(distance)
        if distance <= 10.0:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1

    hidden_false_positive = 0
    for label in hidden_labels:
        frame = samples_by_frame.get(label["frame_index"])
        if frame is not None and frame.visible:
            hidden_false_positive += 1
            false_positive += 1

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _f1(precision, recall)
    hidden_fp_rate = _ratio(hidden_false_positive, len(hidden_labels)) if hidden_labels else None
    metrics = evaluate_numeric_gates(
        {
            "ball_f1_at_10px": f1,
            "ball_hidden_false_positive_rate": hidden_fp_rate,
        },
        BALL_LABEL_GATES,
    )
    metrics["ball_visible_label_count"] = metric(
        value=len(visible_labels),
        unit="labels",
        gate="label_check.ball_visible_labels_recorded",
        passed=None,
    )
    metrics["ball_hidden_label_count"] = metric(
        value=len(hidden_labels),
        unit="labels",
        gate="label_check.ball_hidden_labels_recorded",
        passed=None,
    )
    metrics["ball_precision_at_10px"] = metric(
        value=precision,
        unit=None,
        gate="label_check.ball_precision_at_10px_recorded",
        passed=None,
        status="measured" if precision is not None else "not_measured",
    )
    metrics["ball_recall_at_10px"] = metric(
        value=recall,
        unit=None,
        gate="label_check.ball_recall_at_10px_recorded",
        passed=None,
        status="measured" if recall is not None else "not_measured",
    )
    metrics["ball_p90_error_px"] = metric(
        value=_percentile(distances, 90.0),
        unit="px",
        gate="label_check.ball_p90_error_px_recorded",
        passed=None,
        status="measured" if distances else "not_measured",
    )
    return metrics, []


def score_racket_pose_labels(*, labels_dir: Path, racket_pose: RacketPose) -> tuple[dict[str, EvalMetric], list[str]]:
    payload, note = _load_authoritative_payload(labels_dir / "racket_pose.json")
    if payload is None:
        return _not_measured_metrics(RACKET_LABEL_GATES), [note]

    labels = [
        {
            "frame_index": frame_index,
            "player_id": _player_id(item),
            "face_normal": face_normal,
            "contact_point_face_cm": contact_point,
        }
        for item in _payload_items(payload)
        if _is_accepted_item(item)
        for frame_index in [_frame_index(item) if _frame_index(item) is not None else _frame_index_from_time(item, racket_pose.fps)]
        for face_normal in [_vector3(item, "face_normal")]
        for contact_point in [_vector2(item, "contact_point_face_cm")]
        if frame_index is not None and (face_normal is not None or contact_point is not None)
    ]
    if not labels:
        return _not_measured_metrics(RACKET_LABEL_GATES), [
            "racket_pose.json has no reviewed racket pose reference labels"
        ]

    predictions_by_frame: dict[int, list[dict[str, Any]]] = {}
    for player in racket_pose.players:
        for contact in player.contacts:
            frame_index = int(round(float(contact.t) * float(racket_pose.fps)))
            predictions_by_frame.setdefault(frame_index, []).append(
                {
                    "player_id": int(player.id),
                    "face_normal": tuple(float(value) for value in contact.face_normal),
                    "contact_point_face_cm": tuple(float(value) for value in contact.contact_point_face_cm),
                }
            )

    face_angle_errors: list[float] = []
    contact_point_errors: list[float] = []
    matched_count = 0
    for label in labels:
        prediction = _matching_racket_prediction(
            predictions_by_frame.get(label["frame_index"], []),
            player_id=label["player_id"],
        )
        if prediction is not None:
            matched_count += 1
        if label["face_normal"] is not None:
            if prediction is None:
                face_angle_errors.append(180.0)
            else:
                face_angle_errors.append(_angle_deg(label["face_normal"], prediction["face_normal"]))
        if label["contact_point_face_cm"] is not None:
            if prediction is None:
                contact_point_errors.append(999.0)
            else:
                contact_point_errors.append(
                    _distance(label["contact_point_face_cm"], prediction["contact_point_face_cm"])
                )

    metrics = evaluate_numeric_gates(
        {
            "racket_face_angle_p90_error_deg": _percentile(face_angle_errors, 90.0),
            "racket_contact_point_p90_error_cm": _percentile(contact_point_errors, 90.0),
        },
        RACKET_LABEL_GATES,
    )
    metrics["racket_pose_label_count"] = metric(
        value=len(labels),
        unit="labels",
        gate="label_check.racket_pose_labels_present",
        passed=None,
    )
    metrics["racket_pose_match_count"] = metric(
        value=matched_count,
        unit="matches",
        gate="label_check.racket_pose_matches_recorded",
        passed=None,
    )
    return metrics, []


def score_shot_labels(
    *,
    labels_dir: Path,
    metrics_artifact: RacketSportMetrics,
    tolerance_s: float = 0.30,
) -> tuple[dict[str, EvalMetric], list[str]]:
    payload, note = _load_authoritative_payload(labels_dir / "events.json")
    if payload is None:
        metrics = _not_measured_metrics(SHOT_LABEL_GATES)
        metrics.update(_shot_record_metrics(None))
        return metrics, [note]

    truth_events = [
        event
        for item in _payload_items(payload)
        if _is_accepted_item(item)
        for event in [_shot_truth_event(item)]
        if event is not None
    ]
    if not truth_events:
        metrics = _not_measured_metrics(SHOT_LABEL_GATES)
        metrics.update(_shot_record_metrics(None))
        return metrics, ["events.json has no reviewed shot labels"]

    score = score_shot_events(
        truth_events,
        _shot_predictions_from_metrics(metrics_artifact),
        tolerance_s=tolerance_s,
    )
    metrics = evaluate_numeric_gates(
        {
            "shot_label_macro_f1": score["macro_f1"],
            "shot_label_top2_accuracy": score["top2_accuracy"],
        },
        SHOT_LABEL_GATES,
    )
    metrics.update(_shot_record_metrics(score))
    return metrics, []


def _load_authoritative_payload(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{path.name} could not be read as reviewed labels: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name} is not an object label payload"
    if payload.get("not_ground_truth") is True:
        return None, f"{path.name} has not_ground_truth=true; label_check metrics not measured"
    status = str(payload.get("status", "")).lower()
    if status not in {"human_reviewed", "accepted", "reviewed"}:
        return None, f"{path.name} status is {payload.get('status')!r}; label_check metrics not measured"
    return payload, ""


def _not_measured_metrics(gates: dict[str, NumericGate]) -> dict[str, EvalMetric]:
    return evaluate_numeric_gates({name: None for name in gates}, gates)


def _payload_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = payload.get("annotation")
    if isinstance(annotation, dict) and isinstance(annotation.get("items"), list):
        return [item for item in annotation["items"] if isinstance(item, dict)]
    if isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload.get("frames"), list):
        return [item for item in payload["frames"] if isinstance(item, dict)]
    return []


def _is_accepted_item(item: dict[str, Any]) -> bool:
    status = str(item.get("status", "")).lower()
    return status == "accepted"


def _frame_index(item: dict[str, Any]) -> int | None:
    raw = item.get("frame_index")
    if isinstance(raw, int) and raw >= 0:
        return raw
    for key in ("frame", "image"):
        value = item.get(key)
        if isinstance(value, str):
            matches = re.findall(r"\d+", value)
            if matches:
                return int(matches[-1])
    return None


def _bbox_xyxy(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = item.get("bbox_xyxy")
    if isinstance(raw, list) and len(raw) == 4:
        return _tuple4(raw)
    raw = item.get("bbox")
    if isinstance(raw, list) and len(raw) == 4:
        x, y, width, height = _tuple4(raw)
        return (x, y, x + width, y + height)
    return None


def _xy_px(item: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("xy_px", "ball_xy", "xy"):
        raw = item.get(key)
        if isinstance(raw, list) and len(raw) == 2 and all(isinstance(value, (int, float)) for value in raw):
            return _tuple2(raw)
    return None


def _is_visible_ball_label(item: dict[str, Any], xy: tuple[float, float] | None) -> bool | None:
    visible = item.get("visible")
    if isinstance(visible, bool):
        return visible
    visibility = str(item.get("visibility", "")).lower()
    if visibility in {"visible", "present"}:
        return True
    if visibility in {"hidden", "missing", "not_visible", "occluded"}:
        return False
    return True if xy is not None else None


def _player_id(item: dict[str, Any]) -> int | None:
    raw = item.get("player_id", item.get("id"))
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _frame_index_from_time(item: dict[str, Any], fps: float) -> int | None:
    raw = item.get("t", item.get("time_s"))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(round(float(raw) * float(fps)))


def _vector3(item: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    raw = item.get(key)
    if isinstance(raw, list) and len(raw) == 3 and all(isinstance(value, (int, float)) for value in raw):
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return None


def _vector2(item: dict[str, Any], key: str) -> tuple[float, float] | None:
    raw = item.get(key)
    if isinstance(raw, list) and len(raw) == 2 and all(isinstance(value, (int, float)) for value in raw):
        return _tuple2(raw)
    return None


def _matching_racket_prediction(predictions: list[dict[str, Any]], *, player_id: int | None) -> dict[str, Any] | None:
    if not predictions:
        return None
    if player_id is not None:
        for prediction in predictions:
            if prediction.get("player_id") == player_id:
                return prediction
    return predictions[0]


def _shot_truth_event(item: dict[str, Any]) -> dict[str, Any] | None:
    label = item.get("shot_label", item.get("label"))
    if not isinstance(label, str) or not label:
        return None
    time_s = _event_time_s(item)
    if time_s is None:
        return None
    event = {
        "id": str(item.get("id", f"truth_{label}_{time_s:.3f}")),
        "t": time_s,
        "shot_label": label,
    }
    player_id = _player_id(item)
    if player_id is not None:
        event["player_id"] = player_id
    frame_index = _frame_index(item)
    if frame_index is not None:
        event["frame"] = frame_index
    return event


def _shot_predictions_from_metrics(metrics_artifact: RacketSportMetrics) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for player in metrics_artifact.players:
        for index, shot in enumerate(player.shots):
            predictions.append(
                {
                    "id": f"player_{player.id}_shot_{index:04d}",
                    "t": float(shot.t),
                    "player_id": int(player.id),
                    "type": str(shot.type),
                    "type_conf": float(shot.type_conf),
                    "top2": [
                        {"type": str(item.type), "confidence": float(item.confidence)}
                        for item in shot.top2[:2]
                    ],
                    "gated": bool(
                        shot.type == "unknown"
                        or any(value.gated is True for value in shot.metrics.values())
                    ),
                }
            )
    return predictions


def _shot_record_metrics(score: dict[str, Any] | None) -> dict[str, EvalMetric]:
    if score is None:
        return {
            "shot_label_sample_count": metric(
                value=None,
                unit="shots",
                gate="label_check.shot_labels_present",
                passed=None,
                status="not_measured",
            ),
            "shot_label_accuracy": metric(
                value=None,
                unit=None,
                gate="label_check.shot_accuracy_recorded",
                passed=None,
                status="not_measured",
            ),
            "shot_label_unknown_rate": metric(
                value=None,
                unit=None,
                gate="label_check.shot_unknown_rate_recorded",
                passed=None,
                status="not_measured",
            ),
            "shot_label_gated_rate": metric(
                value=None,
                unit=None,
                gate="label_check.shot_gated_rate_recorded",
                passed=None,
                status="not_measured",
            ),
            "shot_label_missing_prediction_rate": metric(
                value=None,
                unit=None,
                gate="label_check.shot_missing_prediction_rate_recorded",
                passed=None,
                status="not_measured",
            ),
        }
    return {
        "shot_label_sample_count": metric(
            value=int(score["sample_count"]),
            unit="shots",
            gate="label_check.shot_labels_present",
            passed=int(score["sample_count"]) > 0,
        ),
        "shot_label_accuracy": metric(
            value=float(score["accuracy"]),
            unit=None,
            gate="label_check.shot_accuracy_recorded",
            passed=None,
        ),
        "shot_label_unknown_rate": metric(
            value=float(score["unknown_rate"]),
            unit=None,
            gate="label_check.shot_unknown_rate_recorded",
            passed=None,
        ),
        "shot_label_gated_rate": metric(
            value=float(score["gated_rate"]),
            unit=None,
            gate="label_check.shot_gated_rate_recorded",
            passed=None,
        ),
        "shot_label_missing_prediction_rate": metric(
            value=float(score["missing_prediction_rate"]),
            unit=None,
            gate="label_check.shot_missing_prediction_rate_recorded",
            passed=None,
        ),
    }


def _event_time_s(item: dict[str, Any]) -> float | None:
    for key in ("t", "time_s"):
        raw = item.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
            return float(raw)
    raw_ms = item.get("contact_time_ms")
    if isinstance(raw_ms, (int, float)) and not isinstance(raw_ms, bool) and raw_ms >= 0:
        return float(raw_ms) / 1000.0
    return None


def _angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 180.0
    dot = sum(a[index] * b[index] for index in range(3)) / (norm_a * norm_b)
    clamped = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(clamped))


def _tuple4(values: Any) -> tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _tuple2(values: Any) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return 0.0 if precision == 0.0 or recall == 0.0 else None
    return 2.0 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
