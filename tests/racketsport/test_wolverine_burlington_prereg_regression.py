"""Regression pin for the TRK-10/TRK-11 pre-registered association baseline.

Ledger: ``runs/manager/heldout_eval_ledger.md`` rows TRK-10/TRK-11. TRK-10
pre-registered two per-clip configs *before* execution; TRK-11 recorded their
scored results in ``runs/phase2/trk_assoc_prereg_20260701T233015Z/``:

  * Wolverine (``wolverine_iter2_margin2``): IDF1 0.8678111587982833
  * Burlington (``burlington_iter5_minconf05_appw2``): IDF1 0.9112426035502958

Both numbers are the anchor for every downstream comparison against this
candidate family -- see task #39's regression investigation
(``runs/wolverine_assoc_regression_20260702T050000Z/REPORT.md``), which
traced a suspected Wolverine regression (reported as reproducing at 0.8390)
to a misattributed number: 0.8390 is Burlington's *non-selected*
``iter1_margin2`` config (default min-conf/appearance-weight, not iter5's),
reproduced by the TRK-SPEED lane's device/batch benchmark, which only ever
touched Burlington. Wolverine's own pre-registered config was never actually
re-run by that lane. Direct re-execution of both pre-registered configs on
current code reproduces the TRK-11 numbers exactly (see the REPORT.md above
for the full byte-level diff), so this test exists to keep it that way:
catch any *future* change to ``player_global_association.py`` /
``raw_pool_person_authority.py`` (e.g. the 2026-07-02T04:18Z
``court_polygon_filter`` refactor's neighborhood) that silently shifts the
default candidate-construction court-filter path or flips the new opt-in
``post_association_court_margin_m`` capability on by default.

This test intentionally runs the real association+scoring path (not a mock)
against the real committed 10s clips and the real reused OSNet embedding
exports from the original sweep, so it is real, non-synthetic regression
coverage -- but ``models/checkpoints/`` and ``runs/`` are both gitignored in
this repo (large local artifacts), so every required path is guarded by
``pytest.mark.skipif`` and the test is skipped (not failed) on a fresh
checkout or CI environment that does not have this session's local run
history synced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threed.racketsport.raw_pool_person_authority import (
    RawPoolAuthorityConfig,
    run_raw_pool_authority_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_ROOT = REPO_ROOT / "runs/phase2/trk_sam4dbody_jointid_20260701T073206Z/a100_yolo26m_botsort_reid_loose_eval"
SWEEP_ROOT = REPO_ROOT / "runs/phase2/trk_offline_authority_rawpool_20260701T222255Z"
CAND = "base_yolo26m_botsortreidloose_img1536_conf005_raw"
REID_MODEL = REPO_ROOT / "models/checkpoints/osnet_x1_0_market1501.pt"


def _clip_paths(clip_id: str, video_name: str) -> dict[str, Path]:
    return {
        "video_path": REPO_ROOT / "cvat_upload" / video_name,
        "raw_pool_dir": POOL_ROOT / clip_id / CAND,
        "calibration_path": REPO_ROOT / f"runs/eval0/prototype_gate_h100_v2/{clip_id}/court_calibration.json",
        "embedding_export_path": SWEEP_ROOT / clip_id / "reid_embeddings.json",
        "ground_truth_path": REPO_ROOT / f"runs/cvat_imports/2026_06_30/{clip_id}/person_ground_truth.json",
    }


WOLVERINE = _clip_paths("wolverine_mixed_0200_mid_steep_corner", "02_wolverine_mixed_0200_mid_steep_corner_10s.mp4")
BURLINGTON = _clip_paths("burlington_gold_0300_low_steep_corner", "01_burlington_gold_0300_low_steep_corner_10s.mp4")

_REQUIRED_PATHS = [REID_MODEL, *WOLVERINE.values(), *BURLINGTON.values()]
_MISSING = [str(path) for path in _REQUIRED_PATHS if not path.exists()]

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "TRK-10/TRK-11 pre-registered regression fixtures are local-only "
        f"(gitignored runs/ and models/checkpoints/) and are not present here: {_MISSING[:1]}"
    ),
)


def _run(clip_id: str, candidate: str, paths: dict[str, Path], config: RawPoolAuthorityConfig, out_dir: Path) -> dict:
    return run_raw_pool_authority_candidate(
        clip_id=clip_id,
        candidate=candidate,
        video_path=paths["video_path"],
        raw_pool_dir=paths["raw_pool_dir"],
        calibration_path=paths["calibration_path"],
        out_dir=out_dir,
        reid_model_path=REID_MODEL,
        embedding_export_path=paths["embedding_export_path"],
        ground_truth_path=paths["ground_truth_path"],
        expected_players=4,
        config=config,
    )


def test_wolverine_iter2_prereg_config_pins_trk11_idf1(tmp_path: Path) -> None:
    # TRK-10 pre-registration: Wolverine = iter2 (court-margin 2.0m only, all
    # other knobs at the BASE default profile).
    config = RawPoolAuthorityConfig(expected_players=4, court_margin_m=2.0)
    report = _run(
        "wolverine_mixed_0200_mid_steep_corner",
        CAND,
        WOLVERINE,
        config,
        tmp_path / "wolverine_iter2",
    )

    assert report["global_association"]["court_filter_skipped_reason"] == ""
    assert report["global_association"]["post_association_court_rejected_frame_count"] == 0

    score = report["score"]
    assert score["idf1"] == 0.8678111587982833
    assert score["id_switches"] == 4
    assert score["false_negatives"] == 185
    assert score["false_positives"] == 115
    assert score["off_court_false_positive_frames"] == 25
    assert score["four_player_coverage"] == pytest.approx(0.8233333333333334)


def test_burlington_iter5_prereg_config_pins_trk11_idf1(tmp_path: Path) -> None:
    # TRK-10 pre-registration: Burlington = iter5 (min-conf 0.5,
    # appearance-weight 2.0, court-margin 2.0m).
    config = RawPoolAuthorityConfig(expected_players=4, court_margin_m=2.0, min_conf=0.5, appearance_weight=2.0)
    report = _run(
        "burlington_gold_0300_low_steep_corner",
        CAND,
        BURLINGTON,
        config,
        tmp_path / "burlington_iter5",
    )

    assert report["global_association"]["court_filter_skipped_reason"] == ""
    assert report["global_association"]["post_association_court_rejected_frame_count"] == 0

    score = report["score"]
    assert score["idf1"] == 0.9112426035502958
    assert score["id_switches"] == 0
    assert score["false_negatives"] == 244
    assert score["off_court_false_positive_frames"] == 30
    assert score["four_player_coverage"] == pytest.approx(0.8866666666666667)
