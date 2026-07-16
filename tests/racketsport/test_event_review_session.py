from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from scripts.racketsport.build_event_review_session import (
    ANCHOR_MARGIN_S,
    CROSS_STRATUM_RADIUS_S,
    PROTECTED_SEED_RADIUS_S,
    SIGNAL_MIN_SEPARATION_S,
    SOURCES,
    TARGETS,
    UNIFORM_MIN_SEPARATION_S,
    _render_html,
    build_session,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def session_manifest() -> dict:
    return build_session(ROOT, seed=20260715)


def test_sampler_is_deterministic_and_hits_exact_strata(session_manifest: dict) -> None:
    rerun = build_session(ROOT, seed=20260715)
    encoded = json.dumps(session_manifest, indent=2, sort_keys=True) + "\n"
    assert encoded == json.dumps(rerun, indent=2, sort_keys=True) + "\n"
    assert len(session_manifest["rows"]) == 300
    assert Counter(row["stratum"] for row in session_manifest["rows"]) == Counter(TARGETS)
    assert session_manifest["universe"]["clip_count"] == 40
    assert session_manifest["universe"]["online_harvest_20260712_rally_count"] == 0
    assert session_manifest["anchor_sanity_check"]["within_one_frame"] >= 45
    assert session_manifest["anchor_sanity_check"]["status"] == "pass"


def test_sampler_enforces_all_four_exclusions_and_allocations(session_manifest: dict) -> None:
    protected_payload = json.loads(
        (ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json").read_text(encoding="utf-8")
    )
    protected: dict[str, list[float]] = defaultdict(list)
    for label in protected_payload["labels"]:
        protected[label["source"]["clip_id"]].append(float(label["anchor"]["pts_s"]))
    duration_by_clip: dict[str, float] = {}
    for path in (ROOT / "data/online_harvest_20260706/rallies").glob("*/*.mp4"):
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        duration_by_clip[path.stem] = float(completed.stdout)

    for row in session_manifest["rows"]:
        # E1: protected eval seed radius.
        assert all(abs(row["anchor_pts_s"] - anchor) > PROTECTED_SEED_RADIUS_S for anchor in protected[row["clip_id"]])
        serialized = json.dumps(row).lower()
        # E2: competitor R&D reference never enters any row.
        assert "pbvision_11min_20260713" not in serialized
        # E3: only the declared rally tree, never protected/test video.
        assert row["video_path"].startswith("data/online_harvest_20260706/rallies/")
        assert "eval_clips/" not in row["video_path"]
        assert "data/testclips/" not in row["video_path"]
        # E4: the render-safe anchor interval.
        assert ANCHOR_MARGIN_S <= row["anchor_pts_s"] <= duration_by_clip[row["clip_id"]] - ANCHOR_MARGIN_S

    for stratum, target in TARGETS.items():
        counts = Counter(row["source_group"] for row in session_manifest["rows"] if row["stratum"] == stratum)
        assert set(counts) == set(SOURCES)
        assert min(counts.values()) >= 8
        assert max(counts.values()) <= int(target * 0.30)
        assert sum(counts.values()) == target


def test_sampler_separation_and_tercile_metadata(session_manifest: dict) -> None:
    by_stratum_clip: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in session_manifest["rows"]:
        by_stratum_clip[(row["stratum"], row["clip_id"])].append(row["anchor_pts_s"])
        if row["stratum"] in {"audio_onset", "track_discontinuity"}:
            assert row["score_band"] in {"low", "mid", "high"}
        else:
            assert row["score_band"] is None
    for (stratum, _clip), anchors in by_stratum_clip.items():
        minimum = UNIFORM_MIN_SEPARATION_S if stratum == "uniform_random" else SIGNAL_MIN_SEPARATION_S
        ordered = sorted(anchors)
        assert all(right - left >= minimum - 1e-8 for left, right in zip(ordered, ordered[1:]))
    clips = {row["clip_id"] for row in session_manifest["rows"]}
    for clip in clips:
        audio = by_stratum_clip[("audio_onset", clip)]
        track = by_stratum_clip[("track_discontinuity", clip)]
        assert all(abs(left - right) > CROSS_STRATUM_RADIUS_S for left in audio for right in track)


def test_blind_page_contains_only_allowed_item_fields_and_keyboard_controls(session_manifest: dict) -> None:
    html = _render_html(session_manifest)
    lowered = html.lower()
    assert "stratum" not in lowered
    assert "audio_onset" not in lowered
    assert "track_discontinuity" not in lowered
    assert "uniform_random" not in lowered
    assert 'const KEY="event_labels_20260715_answers_v2"' in html
    assert "ArrowLeft" in html and "ArrowRight" in html
    assert "2/ITEMS[i].source_fps" in html
    assert "results_schema_version:2" in html
    assert "label_id:item.label_id" in html
    # dt-integrity regressions (owner-found corruption, 2026-07-16): native controls
    # let a phase-2 click toggle playback so dt was read from a moving currentTime;
    # the rewatch button (and browser video context menu) could do the same at Confirm.
    assert re.search(r"<video[^>]*\bcontrols\b", html) is None
    assert '<video id="player" playsinline></video>' in html
    # click-capture pauses playback before recording coordinates
    assert 'player.addEventListener("click",e=>{if(!pending)return;player.pause();' in html
    # commit pauses at the dt read site, covering every play-entry path (rewatch, context menu)
    assert "if(!click)return;player.pause();rec.x=click.x" in html
    # phase 1 must autoplay a loop from t=0 (no native controls exist to start playback)
    assert "It loops — watch with sound." in html
    assert 'player.loop=true;player.currentTime=0;player.play().catch(()=>{})}' in html
    # onloadedmetadata must respect phase: no pending decision -> loop+play; pending -> centerPaused
    assert (
        "player.onloadedmetadata=()=>{if(!pending){player.loop=true;player.currentTime=0;"
        "player.play().catch(()=>{})}else{centerPaused()}}" in html
    )


def test_build_event_review_session_direct_cli_render(tmp_path: Path) -> None:
    """Direct subprocess coverage for scripts/racketsport/build_event_review_session.py."""
    clip_id = "73VurrTKCZ8_rally_fixture"
    source = tmp_path / "data/online_harvest_20260706/rallies/73VurrTKCZ8" / f"{clip_id}.mp4"
    source.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=2",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(source),
        ],
        check=True,
    )
    manifest = {
        "session_id": "fixture",
        "rows": [
            {
                "row": 1,
                "label_id": "els_fixture_001",
                "clip_id": clip_id,
                "video_path": source.relative_to(tmp_path).as_posix(),
                "source_fps": 30.0,
                "anchor_pts_s": 1.0,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_dir = tmp_path / "pack"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_event_review_session.py",
            "render",
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["rendered"] == 1
    assert report["validated"] == 1
    assert report["all_have_audio"] is True
    assert report["all_decode_exit_zero"] is True
    assert 2.25 <= report["duration_range_s"][0] <= 2.55
    html = (out_dir / "START_HERE.html").read_text(encoding="utf-8")
    assert "stratum" not in html.lower()


def test_renderer_rejects_video_outside_universe(tmp_path: Path) -> None:
    manifest = {
        "session_id": "fixture",
        "rows": [{"row": 1, "label_id": "x", "clip_id": "protected", "video_path": "eval_clips/ball/protected.mp4", "source_fps": 30, "anchor_pts_s": 1.0}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_event_review_session.py", "render", "--root", str(tmp_path), "--manifest", str(manifest_path), "--out-dir", str(tmp_path / "pack")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "refusing to render non-universe path" in completed.stderr
