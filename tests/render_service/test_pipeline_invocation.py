import json
from pathlib import Path

from server.pipeline_invocation import (
    build_process_video_args,
    copy_source_video_artifact,
    prepare_render_artifacts,
    remote_model_root,
    rewrite_manifest_urls,
    safe_slug,
)


def test_build_process_video_args_produces_expected_arg_list() -> None:
    args = build_process_video_args(
        python="/srv/pickleball/.venv/bin/python",
        script="scripts/racketsport/process_video.py",
        video="/srv/pickleball/runs/render_jobs/job_1/input/clip.mp4",
        out="/srv/pickleball/runs/render_jobs/job_1/out",
        clip="clip_1",
        model_root="/srv/pickleball",
        sidecar="/srv/pickleball/runs/render_jobs/job_1/input/capture_sidecar.json",
        max_frames=12,
        wasb_repo="/srv/pickleball_git/third_party/WASB-SBDT",
        wasb_checkpoint="/srv/pickleball_git/models/checkpoints/wasb/wasb_tennis_best.pth.tar",
        allow_auto_court=True,
    )

    assert args[:2] == ["/srv/pickleball/.venv/bin/python", "scripts/racketsport/process_video.py"]
    assert args[args.index("--video") + 1] == "/srv/pickleball/runs/render_jobs/job_1/input/clip.mp4"
    assert args[args.index("--out") + 1] == "/srv/pickleball/runs/render_jobs/job_1/out"
    assert args[args.index("--clip") + 1] == "clip_1"
    assert "--body-local" in args
    assert args[args.index("--device") + 1] == "cuda:0"
    assert "--json" in args
    assert args[args.index("--manifest") + 1] == "/srv/pickleball/models/MANIFEST.json"
    assert args[args.index("--reid-model") + 1] == "/srv/pickleball/models/checkpoints/osnet_x1_0_market1501.pt"
    assert "--allow-auto-court-corners-preview" in args
    assert args[args.index("--wasb-repo") + 1] == "/srv/pickleball_git/third_party/WASB-SBDT"
    assert (
        args[args.index("--wasb-checkpoint") + 1]
        == "/srv/pickleball_git/models/checkpoints/wasb/wasb_tennis_best.pth.tar"
    )
    assert args[args.index("--max-frames") + 1] == "12"
    assert (
        args[args.index("--capture-sidecar") + 1]
        == "/srv/pickleball/runs/render_jobs/job_1/input/capture_sidecar.json"
    )


def test_build_process_video_args_omits_optional_flags_when_absent() -> None:
    args = build_process_video_args(
        python="python3",
        script="scripts/racketsport/process_video.py",
        video="/tmp/in/clip.mp4",
        out="/tmp/out",
        clip="clip_1",
        model_root="/opt/pickleball",
        sidecar=None,
        max_frames=None,
    )

    assert "--allow-auto-court-corners-preview" not in args
    assert "--wasb-repo" not in args
    assert "--wasb-checkpoint" not in args
    assert "--max-frames" not in args
    assert "--capture-sidecar" not in args
    assert "--court-corners" not in args
    assert "--court-calibration" not in args


def test_build_process_video_args_rejects_unsafe_clip_slug() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_process_video_args(
            python="python3",
            script="scripts/racketsport/process_video.py",
            video="/tmp/in/clip.mp4",
            out="/tmp/out",
            clip="clip; rm -rf /",
            model_root="/opt/pickleball",
            sidecar=None,
            max_frames=None,
        )


def test_remote_model_root_splits_on_venv_marker() -> None:
    assert remote_model_root("/srv/pickleball/.venv/bin/python") == "/srv/pickleball"


def test_remote_model_root_falls_back_to_parent_parent_parent_without_venv_marker() -> None:
    assert remote_model_root("/opt/pickleball/bin/python3") == "/opt"


def test_rewrite_manifest_urls_with_resolver_yields_bare_bundle_names(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    video_path = tmp_path / "input" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    manifest_path = artifacts_dir / "replay_viewer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "video_url": "/@fs//srv/pickleball/runs/render_jobs/job_1/input/clip.mp4",
                "virtual_world_url": "/@fs//srv/pickleball/runs/render_jobs/job_1/out/clip_1/confidence_gated_world.json",
                "unrelated_field": "left alone",
            }
        ),
        encoding="utf-8",
    )

    rewrite_manifest_urls(
        artifacts_dir=artifacts_dir,
        video_path=video_path,
        resolve=lambda name: f"bundles/clip_1/{name}",
    )

    rewritten = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rewritten["video_url"] == "bundles/clip_1/source.mp4"
    assert rewritten["virtual_world_url"] == "bundles/clip_1/confidence_gated_world.json"
    assert rewritten["unrelated_field"] == "left alone"


def test_rewrite_manifest_urls_is_a_noop_without_a_manifest_file(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    # Must not raise even though replay_viewer_manifest.json does not exist.
    rewrite_manifest_urls(artifacts_dir=artifacts_dir, video_path=video_path, resolve=lambda name: name)

    assert not (artifacts_dir / "replay_viewer_manifest.json").exists()


def test_copy_source_video_artifact_writes_source_file_and_returns_name(tmp_path: Path) -> None:
    video_path = tmp_path / "input" / "clip.mov"
    video_path.parent.mkdir()
    video_path.write_bytes(b"hello")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    name = copy_source_video_artifact(video_path=video_path, artifacts_dir=artifacts_dir)

    assert name == "source.mov"
    assert (artifacts_dir / "source.mov").read_bytes() == b"hello"


def test_prepare_render_artifacts_combines_copy_and_rewrite(tmp_path: Path) -> None:
    video_path = tmp_path / "input" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video-bytes")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "replay_viewer_manifest.json").write_text(
        json.dumps({"video_url": "/@fs//srv/x/input/clip.mp4"}), encoding="utf-8"
    )

    prepare_render_artifacts(
        artifacts_dir=artifacts_dir,
        video_path=video_path,
        resolve=lambda name: f"/api/jobs/job_1/artifacts/{name}",
    )

    assert (artifacts_dir / "source.mp4").read_bytes() == b"video-bytes"
    manifest = json.loads((artifacts_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_url"] == "/api/jobs/job_1/artifacts/source.mp4"


def test_prepare_render_artifacts_creates_artifacts_dir_if_missing(tmp_path: Path) -> None:
    video_path = tmp_path / "input" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"x")
    artifacts_dir = tmp_path / "does_not_exist_yet"

    prepare_render_artifacts(artifacts_dir=artifacts_dir, video_path=video_path, resolve=lambda name: name)

    assert artifacts_dir.is_dir()
    assert (artifacts_dir / "source.mp4").is_file()


def test_safe_slug_rejects_shell_metacharacters() -> None:
    import pytest

    assert safe_slug("match_01-clip.a") == "match_01-clip.a"
    with pytest.raises(ValueError):
        safe_slug("match; rm -rf /")
