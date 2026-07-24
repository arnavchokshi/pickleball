from __future__ import annotations

import pytest

from scripts.racketsport.eval_event_head import owner_val_metrics_from_predictions
from threed.racketsport.event_head.matcher import Event


def _owner_manifest() -> dict:
    rows = []
    decisions = ["HIT"] * 15 + ["BOUNCE"] * 4 + ["none"] * 22
    for index, decision in enumerate(decisions):
        events = [] if decision == "none" else [{"frame": 32, "class": decision}]
        rows.append({
            "source": "synthetic_owner_reviewed",
            "source_video": f"source_{index // 21}",
            "video_path": f"synthetic_{index // 21}.mp4",
            "media_present": True,
            "split": "val",
            "fps": 30.0,
            "source_start_frame": index * 64,
            "num_frames": 64,
            "events": events,
            "loss_validity_mask": [True, True, True],
            "license_posture": "synthetic_fixture",
            "label_id": f"owner_{index:02d}",
        })
    return {
        "schema_version": 1,
        "artifact_type": "event_head_owner_reviewed_dataset_manifest",
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "config": {"window_frames": 64},
        "rows": rows,
    }


def test_owner_val_synthetic_predictions_emit_every_decision_gate_field_deterministically() -> None:
    manifest = _owner_manifest()
    predictions = {}
    for row in manifest["rows"]:
        row_id = row["label_id"]
        if row["events"]:
            class_id = 1 if row["events"][0]["class"] == "HIT" else 2
            predictions[row_id] = [Event(32, class_id, 0.9)]
        else:
            predictions[row_id] = []
    predictions["owner_19"] = [Event(12, 1, 0.8)]
    predictions["owner_20"] = [Event(18, 1, 0.7)]

    kwargs = {
        "arm": "B",
        "seed": 20260721,
        "completed_steps": 1000,
        "target_steps": 1000,
        "full_video_event_count": 6,
        "full_video_duration_s": 10.0,
    }
    first = owner_val_metrics_from_predictions(manifest, predictions, **kwargs)
    second = owner_val_metrics_from_predictions(manifest, predictions, **kwargs)

    assert first == second
    assert first["artifact_type"] == "event_head_abc_arm_eval"
    assert first["selection_scope"] == "owner_validation_41"
    assert first["selection_rows"] == 41
    assert first["protected_50_touched"] is False
    assert first["arm"] == "B"
    assert first["seed"] == 20260721
    assert first["completed_steps"] == first["target_steps"] == 1000
    assert first["negative_rows"] == 22
    assert first["negative_false_positives"] == 2
    assert first["timing_error_p90_frames"] == 0
    assert first["full_video_events_per_second"] == 0.6
    tolerance_two = next(
        item for item in first["tolerance_sweep"] if item["tolerance_frames"] == 2
    )
    assert tolerance_two["per_class"]["HIT"]["f1"] == 0.9375
    assert tolerance_two["per_class"]["BOUNCE"]["f1"] == 1.0
    assert first["macro_f1_at_2"] == 0.96875


def test_frozen_judge_macro_f1_at_2_protocol_is_unchanged() -> None:
    """E1 frozen judge: macro-F1 is scored at tolerance ±2 frames, exactly.

    A prediction offset by exactly 2 frames must count as a match; an offset
    of 3 frames must not. The gate's headline macro_f1_at_2 must equal the
    hand-computed mean of the per-class F1 values at the tolerance-2 sweep.
    """

    manifest = _owner_manifest()
    predictions = {}
    for row in manifest["rows"]:
        row_id = row["label_id"]
        if row["events"]:
            class_id = 1 if row["events"][0]["class"] == "HIT" else 2
            predictions[row_id] = [Event(32, class_id, 0.9)]
        else:
            predictions[row_id] = []
    # owner_00 (HIT): offset exactly +2 -> matched at the frozen +/-2 tolerance.
    predictions["owner_00"] = [Event(34, 1, 0.9)]
    # owner_01 (HIT): offset +3 -> outside the frozen tolerance, FP + FN.
    predictions["owner_01"] = [Event(35, 1, 0.9)]

    metrics = owner_val_metrics_from_predictions(
        manifest,
        predictions,
        arm="B",
        seed=20260721,
        completed_steps=1000,
        target_steps=1000,
        full_video_event_count=6,
        full_video_duration_s=10.0,
    )
    tolerance_two = next(
        item for item in metrics["tolerance_sweep"] if item["tolerance_frames"] == 2
    )
    tolerance_one = next(
        item for item in metrics["tolerance_sweep"] if item["tolerance_frames"] == 1
    )
    # HIT at +/-2: 14 TP (13 exact + the +2 offset), 1 FP, 1 FN -> F1 = 14/15.
    assert tolerance_two["per_class"]["HIT"]["tp"] == 14
    assert tolerance_two["per_class"]["HIT"]["f1"] == pytest.approx(14 / 15)
    assert tolerance_two["per_class"]["BOUNCE"]["f1"] == 1.0
    # At +/-1 the +2 offset would NOT match; the protocol is +/-2, not +/-1.
    assert tolerance_one["per_class"]["HIT"]["tp"] == 13
    assert metrics["macro_f1_at_2"] == pytest.approx((14 / 15 + 1.0) / 2)
