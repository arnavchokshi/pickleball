import json
from pathlib import Path

import pytest

from scripts.racketsport.verify_process_video_viewer import (
    assert_non_empty_entity_counts,
    viewer_url_for_manifest,
    write_headless_verify_report,
)


def test_viewer_url_requires_real_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest path is required"):
        viewer_url_for_manifest(None)

    with pytest.raises(FileNotFoundError, match="manifest does not exist"):
        viewer_url_for_manifest(tmp_path / "missing_replay_viewer_manifest.json")


def test_viewer_url_adds_manifest_query_param(tmp_path: Path) -> None:
    manifest = tmp_path / "replay_viewer_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    url = viewer_url_for_manifest(manifest)

    assert url.startswith("http://127.0.0.1:5173/?manifest=/@fs")
    assert str(manifest.resolve()) in url


def test_assert_non_empty_entity_counts_rejects_silent_empty_viewer() -> None:
    counts = {
        "Players": 0,
        "Mesh Frames": 0,
        "Solid Mesh Frames": 0,
        "Floor Frames": 0,
        "Ball Contacts": 0,
        "Replay Points": 0,
    }

    with pytest.raises(AssertionError, match="empty viewer"):
        assert_non_empty_entity_counts(counts)


def test_assert_non_empty_entity_counts_allows_explicit_empty_opt_out() -> None:
    assert_non_empty_entity_counts({"Players": 0}, allow_empty=True)


def test_write_headless_verify_report_records_counts_and_page_errors(tmp_path: Path) -> None:
    report_path = write_headless_verify_report(
        tmp_path,
        {
            "ok": False,
            "url": "http://127.0.0.1:5173/?manifest=/@fs/tmp/replay_viewer_manifest.json",
            "loaded_counts": {"Players": 4, "Ball Contacts": 63},
            "page_errors": ["contact_windows.events[0].sources.audio must be a number"],
        },
    )

    assert report_path == tmp_path / "headless_verify.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["loaded_counts"]["Players"] == 4
    assert payload["page_errors"] == ["contact_windows.events[0].sources.audio must be a number"]
