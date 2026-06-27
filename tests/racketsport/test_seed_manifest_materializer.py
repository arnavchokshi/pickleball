from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.materialize_seed_manifest import materialize_seed_manifest


def _write_seed_manifest(path: Path, clips: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "source manifest, not registrar input",
                "download_tool": "yt-dlp",
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )


def _clip(name: str, **metadata_overrides) -> dict:
    metadata = {
        "camera_height": "high",
        "camera_angle": "shallow_baseline",
        "play_type": "doubles",
        "environment": "outdoor",
        "frame_rate_fps": 30,
        "duration_s": 90,
        "racket_gt": False,
    }
    metadata.update(metadata_overrides)
    return {
        "name": name,
        "url": "https://www.youtube.com/watch?v=example",
        "title": "Seed source row",
        "time_range": "00:12:00-00:13:30",
        "verified_media": {"width": 1920, "height": 1080, "fps": 30},
        "metadata": metadata,
        "visual_check": "all players visible",
    }


def test_materialize_seed_manifest_writes_registrar_manifest_for_downloaded_sources(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "baseline.mp4").write_bytes(b"fake mp4")
    (downloaded / "side_view.mov").write_bytes(b"fake mov")
    seed = tmp_path / "seed.json"
    output = tmp_path / "registrar.json"
    _write_seed_manifest(
        seed,
        [
            _clip("baseline"),
            _clip(
                "side_view",
                camera_height="mid",
                camera_angle="side_fence",
                play_type="messy_real_world",
                environment="indoor",
                frame_rate_fps=60,
                duration_s=75,
                racket_gt=True,
            ),
        ],
    )

    summary = materialize_seed_manifest(
        source_manifest_path=seed,
        output_manifest_path=output,
        downloaded_source_root=downloaded,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["materialized_count"] == 2
    assert payload["schema_version"] == 1
    assert payload["source_manifest"] == str(seed)
    assert payload["downloaded_source_root"] == str(downloaded)
    assert payload["clips"] == [
        {
            "source": str(downloaded / "baseline.mp4"),
            "name": "baseline",
            "camera_height": "high",
            "camera_angle": "shallow_baseline",
            "play_type": "doubles",
            "environment": "outdoor",
            "frame_rate_fps": 30,
            "duration_s": 90,
            "racket_gt": False,
        },
        {
            "source": str(downloaded / "side_view.mov"),
            "name": "side_view",
            "camera_height": "mid",
            "camera_angle": "side_fence",
            "play_type": "messy_real_world",
            "environment": "indoor",
            "frame_rate_fps": 60,
            "duration_s": 75,
            "racket_gt": True,
        },
    ]
    assert "url" not in payload["clips"][0]
    assert "metadata" not in payload["clips"][0]


def test_materialize_seed_manifest_requires_source_file_by_default(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    seed = tmp_path / "seed.json"
    output = tmp_path / "registrar.json"
    _write_seed_manifest(seed, [_clip("missing_clip")])

    with pytest.raises(FileNotFoundError, match="missing_clip"):
        materialize_seed_manifest(
            source_manifest_path=seed,
            output_manifest_path=output,
            downloaded_source_root=downloaded,
        )

    assert not output.exists()


def test_materialize_seed_manifest_allow_missing_writes_expected_source_path(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    seed = tmp_path / "seed.json"
    output = tmp_path / "registrar.json"
    _write_seed_manifest(seed, [_clip("missing_clip")])

    summary = materialize_seed_manifest(
        source_manifest_path=seed,
        output_manifest_path=output,
        downloaded_source_root=downloaded,
        allow_missing=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["missing_count"] == 1
    assert payload["clips"][0]["source"] == str(downloaded / "missing_clip.mp4")
    assert payload["clips"][0]["name"] == "missing_clip"


def test_materialize_seed_manifest_cli_does_not_download_and_writes_json(tmp_path):
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "baseline.mkv").write_bytes(b"fake mkv")
    seed = tmp_path / "seed.json"
    output = tmp_path / "registrar.json"
    _write_seed_manifest(seed, [_clip("baseline")])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/materialize_seed_manifest.py",
            "--source-manifest",
            str(seed),
            "--downloaded-source-root",
            str(downloaded),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["materialized_count"] == 1
    assert payload["clips"][0]["source"] == str(downloaded / "baseline.mkv")
