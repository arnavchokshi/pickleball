#!/usr/bin/env python3
"""Promote the calibrations that earned it, and only those.

Promotion writes TWO new files beside the raw solve and modifies NOTHING that
already exists:

* `court_calibration_metric15pt_promoted.json` -- the refit lane's refined
  artifact, re-serialised through the `CourtCalibration` model so it actually
  validates (the lane's `provenance` block used ad-hoc keys that the strict
  schema forbids, so the refined artifacts as-shipped fail
  `validate_artifact_file("court_calibration", ...)`).
* `court_calibration_selected.json` -- the checksummed pointer that makes it the
  selected input. See `threed.racketsport.court_calibration_selection`.

The raw solve is read, digested, and left alone. Its digest goes in the pointer,
so a later edit to a "raw" artifact is detectable rather than silent.

Who is promoted, and why, is a decision -- not something this script computes.
The evidence is `floor_before_promotion.json` vs `floor_refined_candidates.json`,
both produced by `measure_calibration_floor.py`, and the reasoning is in
REPORT.md. Refusals are recorded here as data so they survive as measured
negatives rather than as absence.

Usage:
    python3 runs/lanes/calib_promote_cleanup_20260727/promote_calibrations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration_selection import (  # noqa: E402
    SELECTION_ARTIFACT_TYPE,
    SELECTION_POINTER_FILENAME,
    SELECTION_SCHEMA_VERSION,
    file_sha256,
    resolve_selected_calibration_path,
)
from threed.racketsport.schemas import validate_artifact_file  # noqa: E402

LANE = "runs/lanes/calib_promote_cleanup_20260727"
REFIT_LANE = "runs/lanes/calib_distortion_fit_20260726"
PROMOTED_NAME = "court_calibration_metric15pt_promoted.json"

#: Base commit this promotion was decided on.
BASE_COMMIT = "e209112"

# --------------------------------------------------------------------------
# The decision. Both held-out criteria -- median plane error in metres and
# median reprojection in pixels -- must agree that the refined artifact is
# better before it is promoted. Where they disagree, or where neither moves,
# the raw solve stays selected.
# --------------------------------------------------------------------------

PROMOTE: dict[str, dict[str, str]] = {
    "pbvision_11min_20260713_demo_seed": {
        "raw": "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_metric15pt.json",
        "reviewed_keypoints": "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_keypoints_reviewed.json",
        "rationale": (
            "The defect the North Star names. The raw solve declared its 3 net keypoints at "
            "0.9144 m while the labels are at ~0 m, so its own correspondence set was wrong. "
            "Held-out median plane error 2.7452 -> 0.1770 m (-93.6%) and held-out median "
            "reprojection 23.317 -> 8.772 px (-62.4%): both held-out criteria agree by a wide "
            "margin."
        ),
    },
    "owner_IMG_1605_8a193402780b": {
        "raw": "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_calibration_metric15pt.json",
        "reviewed_keypoints": "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoints.json",
        "rationale": (
            "The raw artifact was internally inconsistent: it declared net points at 0.9144 m "
            "and then excluded them from its own fit. Held-out median plane error "
            "2.4198 -> 0.1065 m (-95.6%) and held-out median reprojection 26.847 -> 3.448 px "
            "(-87.2%): both held-out criteria agree. The in-sample floor barely moves "
            "(0.0933 -> 0.0899 m) because the shipped camera was already fit floor-only -- the "
            "repair is to the declared correspondence set, which is exactly what held-out sees "
            "and in-sample cannot."
        ),
    },
}

REFUSE: dict[str, dict[str, object]] = {
    "burlington_gold_0300_low_steep_corner": {
        "held_out_plane_error_m": {"selected": 0.268336, "refined": 0.269263, "change_pct": +0.35},
        "held_out_median_px": {"selected": 10.742, "refined": 11.050, "change_pct": +2.87},
        "in_sample_p95_px": {"selected": 19.783, "refined": 25.775, "change_pct": +30.29},
        "reason": (
            "Nothing improves. Both held-out criteria are marginally WORSE, and the in-sample "
            "p95 regresses 19.78 -> 25.78 px. That last number is not cosmetic: "
            "ball_inout_uncertainty builds the in/out abstention radius from the in-sample p95, "
            "so promoting here would widen the radius while buying no held-out accuracy."
        ),
    },
    "indoor_doubles_fwuks_0500_long_mid_baseline": {
        "held_out_plane_error_m": {"selected": 0.233780, "refined": 0.190812, "change_pct": -18.38},
        "held_out_median_px": {"selected": 5.562, "refined": 6.603, "change_pct": +18.72},
        "in_sample_median_px": {"selected": 3.719, "refined": 5.446, "change_pct": +46.44},
        "reason": (
            "The only clip where the two held-out criteria DISAGREE, and they disagree by "
            "almost exactly equal magnitude: plane error -18.4%, reprojection +18.7%. Promoting "
            "would therefore not be acting on a measurement, it would be betting on which "
            "held-out metric to rank by. The refit lane made the defensible choice of metres "
            "for model selection; that is not the same as evidence strong enough to replace a "
            "shipped artifact. Refused pending a tie-break the data does not currently supply. "
            "This is the one refusal a reasonable owner might overturn -- the numbers are here "
            "to overturn it with."
        ),
    },
    "outdoor_webcam_iynbd_1500_long_high_baseline": {
        "held_out_plane_error_m": {"selected": 0.127254, "refined": 0.127254, "change_pct": 0.0},
        "held_out_median_px": {"selected": 7.026, "refined": 7.026, "change_pct": 0.0},
        "reason": (
            "MEASURED NEGATIVE, PRESERVED. The refit refused distortion on held-out evidence "
            "for this camera and recovered the shipped solve exactly. Both held-out criteria "
            "are identical to 6 significant figures. There is nothing to promote, and the "
            "refusal of distortion here is a result about this camera, not a gap."
        ),
    },
    "wolverine_mixed_0200_mid_steep_corner": {
        "held_out_plane_error_m": {"selected": 0.158241, "refined": 0.158241, "change_pct": 0.0},
        "held_out_median_px": {"selected": 11.005, "refined": 11.005, "change_pct": 0.0},
        "reason": (
            "MEASURED NEGATIVE, PRESERVED. Same as outdoor: distortion refused on held-out "
            "evidence, shipped solve recovered, both held-out criteria unchanged. The owner's "
            "reviewed bounce labels on this clip do not materially improve either "
            "(sigma along ray 0.2253 -> 0.2249 m), which is an honest negative for the clip."
        ),
    },
}


def _promoted_payload(refined_path: Path, *, raw: str, reviewed_keypoints: str) -> dict:
    """Refined artifact, with a `provenance` block the strict schema accepts.

    The refit lane's provenance keys (`refinement_of`, `refit_reason`, ...) are
    forbidden extras on `CourtCalibrationProvenance`, which requires exactly
    `method`, `inputs`, `code_identity`. The narrative those keys carried is not
    lost -- it moves to the selection pointer, which is the artifact that records
    the promotion decision.
    """

    payload = json.loads(refined_path.read_text(encoding="utf-8"))
    payload["provenance"] = {
        "method": (
            "metric_15pt_reviewed refit with leave-one-out cross-validation over 15 folds; "
            "net-keypoint label height and radial distortion order both selected on held-out "
            "median plane error in metres"
        ),
        "inputs": [reviewed_keypoints, raw],
        "code_identity": f"{REFIT_LANE}/refit_and_measure.py at {BASE_COMMIT}",
    }
    # Drop null-valued optional declarations rather than emitting explicit nulls.
    return {key: value for key, value in payload.items() if value is not None}


def main() -> int:
    evidence = json.loads((Path(ROOT) / LANE / "floor_before_promotion.json").read_text())["clips"]
    refined_evidence = json.loads(
        (Path(ROOT) / LANE / "floor_refined_candidates.json").read_text()
    )["clips"]

    written: list[str] = []
    record: dict = {
        "schema_version": 1,
        "artifact_type": "racketsport_calibration_promotion_record",
        "lane": LANE,
        "base_commit": BASE_COMMIT,
        "verified": 0,
        "authority_note": (
            "Promotion does not change the authority class. source and intrinsics.source stay "
            "metric_15pt_reviewed on every promoted artifact, and "
            "orchestrator.TRUSTED_INTRINSICS_SOURCES gains no entries."
        ),
        "promoted": {},
        "refused": REFUSE,
    }

    for clip, spec in sorted(PROMOTE.items()):
        raw_rel = str(spec["raw"])
        raw_path = ROOT / raw_rel
        refined_path = ROOT / REFIT_LANE / "refined" / clip / "court_calibration_metric15pt_refined.json"
        raw_digest_before = file_sha256(raw_path)

        promoted_payload = _promoted_payload(
            refined_path, raw=raw_rel, reviewed_keypoints=str(spec["reviewed_keypoints"])
        )
        promoted_path = raw_path.parent / PROMOTED_NAME
        promoted_path.write_text(
            json.dumps(promoted_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_artifact_file("court_calibration", promoted_path)

        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        for field, getter in (
            ("source", lambda p: p.get("source")),
            ("intrinsics.source", lambda p: (p.get("intrinsics") or {}).get("source")),
        ):
            if getter(raw_payload) != getter(promoted_payload):
                raise SystemExit(
                    f"{clip}: promotion would change {field} "
                    f"{getter(raw_payload)!r} -> {getter(promoted_payload)!r}; that is a "
                    "source-class escalation, not a promotion"
                )

        held_now = evidence[clip]["measurement"]["held_out"]
        held_refined = refined_evidence[clip]["measurement"]["held_out"]
        pointer = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "artifact_type": SELECTION_ARTIFACT_TYPE,
            "clip": clip,
            "selected": {"path": PROMOTED_NAME, "sha256": file_sha256(promoted_path)},
            "supersedes": {"path": raw_path.name, "sha256": raw_digest_before},
            "authority": {
                "class_unchanged": True,
                "source": raw_payload.get("source"),
                "note": (
                    "A better fit of the same owner-reviewed correspondences is not a new "
                    "authority class. TRUSTED_INTRINSICS_SOURCES is unchanged."
                ),
            },
            "verified": 0,
            "promoted_by": {
                "lane": LANE,
                "base_commit": BASE_COMMIT,
                "refined_artifact": (REFIT_LANE + f"/refined/{clip}/court_calibration_metric15pt_refined.json"),
                "refit_lane": REFIT_LANE,
            },
            "evidence": {
                "protocol": (
                    "leave-one-out over 15 correspondences; each fold refits focal length, "
                    "distortion and pose on 14 and scores the 1 withheld"
                ),
                "held_out_median_plane_error_m": {
                    "superseded": held_now["held_out_median_plane_error_m"],
                    "selected": held_refined["held_out_median_plane_error_m"],
                },
                "held_out_median_reprojection_px": {
                    "superseded": held_now["held_out_median_px"],
                    "selected": held_refined["held_out_median_px"],
                },
                "in_sample_caveat": (
                    "reprojection_error_px in both artifacts is IN-SAMPLE and optimistic; the "
                    "held_out_* figures above are the honest ones"
                ),
                "rationale": spec["rationale"],
            },
        }
        pointer_path = raw_path.parent / SELECTION_POINTER_FILENAME
        pointer_path.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if file_sha256(raw_path) != raw_digest_before:
            raise SystemExit(f"{clip}: raw solve changed during promotion; aborting")
        resolved = resolve_selected_calibration_path(raw_path)
        if resolved != promoted_path:
            raise SystemExit(f"{clip}: resolver returned {resolved}, expected {promoted_path}")

        written.extend(
            [
                promoted_path.relative_to(ROOT).as_posix(),
                pointer_path.relative_to(ROOT).as_posix(),
            ]
        )
        record["promoted"][clip] = {
            "raw_untouched": raw_rel,
            "raw_sha256": raw_digest_before,
            "promoted_artifact": promoted_path.relative_to(ROOT).as_posix(),
            "promoted_sha256": file_sha256(promoted_path),
            "selection_pointer": pointer_path.relative_to(ROOT).as_posix(),
            "held_out_plane_error_m": {
                "before": held_now["held_out_median_plane_error_m"],
                "after": held_refined["held_out_median_plane_error_m"],
            },
            "held_out_median_px": {
                "before": held_now["held_out_median_px"],
                "after": held_refined["held_out_median_px"],
            },
            "rationale": spec["rationale"],
        }
        print(f"promoted {clip}")
        for path in written[-2:]:
            print(f"    wrote {path}")

    out = Path(ROOT) / LANE / "promotion_record.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nrefused: {', '.join(sorted(REFUSE))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
