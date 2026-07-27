"""Promotion pointers must redirect the selected calibration, or fail loud.

The raw solve is immutable, so a promoted calibration can only ever be a *new*
file plus a checksummed pointer. These tests pin both halves: the redirect works,
and every way the pointer can be wrong raises instead of silently degrading back
to the superseded artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.court_calibration_selection import (
    SELECTION_ARTIFACT_TYPE,
    SELECTION_POINTER_FILENAME,
    CourtCalibrationSelectionError,
    file_sha256,
    resolve_selected_calibration_path,
)
from threed.racketsport.virtual_world import resolve_best_court_calibration_path

RAW_NAME = "court_calibration_metric15pt.json"
PROMOTED_NAME = "court_calibration_metric15pt_promoted.json"


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    raw = labels / RAW_NAME
    raw.write_text('{"source": "metric_15pt_reviewed", "which": "raw"}\n', encoding="utf-8")
    promoted = labels / PROMOTED_NAME
    promoted.write_text(
        '{"source": "metric_15pt_reviewed", "which": "promoted"}\n', encoding="utf-8"
    )
    return raw, promoted


def _pointer_payload(raw: Path, promoted: Path) -> dict:
    """Pointer references are relative to the pointer's own directory."""

    return {
        "schema_version": 1,
        "artifact_type": SELECTION_ARTIFACT_TYPE,
        "selected": {"path": promoted.name, "sha256": file_sha256(promoted)},
        "supersedes": {"path": raw.name, "sha256": file_sha256(raw)},
        "authority": {"class_unchanged": True},
    }


def _write_pointer(raw: Path, promoted: Path, **overrides) -> Path:
    payload = _pointer_payload(raw, promoted)
    payload.update(overrides)
    pointer = raw.parent / SELECTION_POINTER_FILENAME
    pointer.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pointer


def test_absent_pointer_returns_the_raw_artifact_unchanged(tmp_path: Path) -> None:
    raw, _promoted = _write_pair(tmp_path)

    assert resolve_selected_calibration_path(raw) == raw


def test_valid_pointer_redirects_to_the_promoted_artifact(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(raw, promoted)

    resolved = resolve_selected_calibration_path(raw)

    assert resolved == promoted
    assert json.loads(resolved.read_text(encoding="utf-8"))["which"] == "promoted"
    # The raw solve is still on disk, byte-for-byte.
    assert json.loads(raw.read_text(encoding="utf-8"))["which"] == "raw"


def test_naming_the_pointer_directly_also_resolves(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    pointer = _write_pointer(raw, promoted)

    assert resolve_selected_calibration_path(pointer) == promoted


def test_promoted_artifact_digest_mismatch_raises(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(raw, promoted)
    promoted.write_text('{"which": "tampered"}\n', encoding="utf-8")

    with pytest.raises(CourtCalibrationSelectionError, match="selected artifact digest mismatch"):
        resolve_selected_calibration_path(raw)


def test_edited_raw_solve_raises_because_raw_solves_are_immutable(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(raw, promoted)
    raw.write_text('{"which": "raw-but-edited"}\n', encoding="utf-8")

    with pytest.raises(CourtCalibrationSelectionError, match="immutable"):
        resolve_selected_calibration_path(raw)


def test_missing_promoted_artifact_raises_instead_of_falling_back(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(raw, promoted)
    promoted.unlink()

    with pytest.raises(CourtCalibrationSelectionError, match="does not exist"):
        resolve_selected_calibration_path(raw)


def test_pointer_may_not_supersede_an_artifact_in_another_directory(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    stranger = other / RAW_NAME
    stranger.write_text('{"which": "stranger"}\n', encoding="utf-8")
    _write_pointer(
        raw,
        promoted,
        supersedes={
            "path": f"../other/{stranger.name}",
            "sha256": file_sha256(stranger),
        },
    )

    with pytest.raises(CourtCalibrationSelectionError, match="stay inside the pointer"):
        resolve_selected_calibration_path(raw)


def test_authority_class_change_is_refused(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(raw, promoted, authority={"class_unchanged": False})

    with pytest.raises(CourtCalibrationSelectionError, match="class_unchanged must be true"):
        resolve_selected_calibration_path(raw)


def test_authority_block_is_mandatory(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    payload = _pointer_payload(raw, promoted)
    payload.pop("authority")
    (raw.parent / SELECTION_POINTER_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(CourtCalibrationSelectionError, match="authority is required"):
        resolve_selected_calibration_path(raw)


def test_absolute_paths_in_a_pointer_are_refused(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(
        raw,
        promoted,
        selected={"path": str(promoted), "sha256": file_sha256(promoted)},
    )

    with pytest.raises(CourtCalibrationSelectionError, match="must be relative to the pointer"):
        resolve_selected_calibration_path(raw)


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    raw, promoted = _write_pair(tmp_path)
    _write_pointer(raw, promoted, schema_version=99)

    with pytest.raises(CourtCalibrationSelectionError, match="unsupported schema_version"):
        resolve_selected_calibration_path(raw)


def test_virtual_world_resolver_follows_a_pointer_in_a_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw, promoted = _write_pair(run_dir)
    _write_pointer(raw, promoted)

    assert resolve_best_court_calibration_path(run_dir) == promoted


def test_virtual_world_explicit_path_is_never_redirected_by_a_sibling_pointer(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw, promoted = _write_pair(run_dir)
    _write_pointer(raw, promoted)

    # An explicit --court-calibration choice stays explicit.
    assert resolve_best_court_calibration_path(run_dir, explicit=raw) == raw
    # ...but naming the pointer explicitly resolves.
    assert (
        resolve_best_court_calibration_path(
            run_dir, explicit=raw.parent / SELECTION_POINTER_FILENAME
        )
        == promoted
    )


def test_pointer_must_supersede_the_artifact_it_sits_beside(tmp_path: Path) -> None:
    """A same-directory pointer that names the wrong neighbour is still wrong."""

    raw, promoted = _write_pair(tmp_path)
    decoy = raw.parent / "court_calibration_metric15pt_other.json"
    decoy.write_text('{"which": "decoy"}\n', encoding="utf-8")
    _write_pointer(
        raw, promoted, supersedes={"path": decoy.name, "sha256": file_sha256(decoy)}
    )

    with pytest.raises(CourtCalibrationSelectionError, match="may only supersede"):
        resolve_selected_calibration_path(raw)


# ---------------------------------------------------------------- real repo
REPO_ROOT = Path(__file__).resolve().parents[2]

PROMOTED_CLIPS = {
    "owner_IMG_1605_8a193402780b": (
        "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_calibration_metric15pt.json"
    ),
    "pbvision_11min_20260713_demo_seed": (
        "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/"
        "court_calibration_metric15pt.json"
    ),
}

REFUSED_CLIPS = {
    "burlington_gold_0300_low_steep_corner",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "wolverine_mixed_0200_mid_steep_corner",
}


@pytest.mark.parametrize("clip,raw_rel", sorted(PROMOTED_CLIPS.items()))
def test_promoted_clip_resolves_to_its_promoted_artifact(clip: str, raw_rel: str) -> None:
    """The two promotions are live, checksummed, and still consistent on disk.

    This fails loudly if anyone edits a raw solve (they are immutable) or
    regenerates a promoted artifact without refreshing its pointer.
    """

    raw = REPO_ROOT / raw_rel
    if not raw.is_file():  # pragma: no cover - artifact not present in this checkout
        pytest.skip(f"{raw_rel} not present")

    resolved = resolve_selected_calibration_path(raw)

    assert resolved != raw
    assert resolved.name == PROMOTED_NAME
    assert resolved.parent == raw.parent


@pytest.mark.parametrize("clip,raw_rel", sorted(PROMOTED_CLIPS.items()))
def test_promotion_never_escalates_the_authority_class(clip: str, raw_rel: str) -> None:
    """A better fit of the same reviewed correspondences is not a new authority."""

    raw = REPO_ROOT / raw_rel
    if not raw.is_file():  # pragma: no cover
        pytest.skip(f"{raw_rel} not present")
    promoted = resolve_selected_calibration_path(raw)

    raw_payload = json.loads(raw.read_text(encoding="utf-8"))
    promoted_payload = json.loads(promoted.read_text(encoding="utf-8"))

    assert promoted_payload["source"] == raw_payload["source"] == "metric_15pt_reviewed"
    assert (
        promoted_payload["intrinsics"]["source"]
        == raw_payload["intrinsics"]["source"]
        == "metric_15pt_reviewed"
    )
    assert "reviewed_15pt_correspondences" in promoted_payload["capture_quality"]["reasons"]


@pytest.mark.parametrize("clip,raw_rel", sorted(PROMOTED_CLIPS.items()))
def test_promoted_artifact_validates_against_the_court_calibration_schema(
    clip: str, raw_rel: str
) -> None:
    """The refit lane's refined artifacts do NOT validate; the promoted ones must."""

    from threed.racketsport.schemas import validate_artifact_file

    raw = REPO_ROOT / raw_rel
    if not raw.is_file():  # pragma: no cover
        pytest.skip(f"{raw_rel} not present")

    validate_artifact_file("court_calibration", resolve_selected_calibration_path(raw))


@pytest.mark.parametrize("clip", sorted(REFUSED_CLIPS))
def test_refused_clips_still_resolve_to_their_raw_solve(clip: str) -> None:
    """Refusals are results. burlington/indoor/outdoor/wolverine were not promoted."""

    raw = REPO_ROOT / "eval_clips" / "ball" / clip / "labels" / RAW_NAME
    if not raw.is_file():  # pragma: no cover
        pytest.skip(f"{clip} not present")

    assert resolve_selected_calibration_path(raw) == raw
    assert not (raw.parent / SELECTION_POINTER_FILENAME).exists()
