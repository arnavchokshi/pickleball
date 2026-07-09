import json
from pathlib import Path

import pytest

import server.pipeline_invocation as pipeline_invocation
from server.pipeline_invocation import (
    build_process_video_args,
    collect_manifest_asset_closure,
    copy_source_video_artifact,
    prepare_render_artifacts,
    remote_model_root,
    rewrite_manifest_urls,
    safe_slug,
    stage_manifest_delivery_bundle,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    (artifacts_dir / "confidence_gated_world.json").write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "clip": "clip_1",
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


def test_recursive_manifest_closure_and_staging_preserve_nested_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    video_path = tmp_path / "input" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")

    remote_root = "/@fs//srv/pickleball/runs/render_jobs/job_1/out/clip_1"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {
            "clip": "clip_1",
            "video_url": "/@fs//srv/pickleball/runs/render_jobs/job_1/input/clip.mp4",
            "body_mesh_index_url": f"{remote_root}/body_mesh_index/body_mesh_index.json",
            "replay_scene_url": f"{remote_root}/replay_scene.json",
            "label_overlays": [{"url": f"{remote_root}/replay_scene.json"}],
        },
    )
    _write_json(
        source_dir / "body_mesh_index" / "body_mesh_index.json",
        {
            "faces_url": "body_mesh_faces.json",
            "windows": [
                {"url": "body_mesh_chunks/window_000.bin.gz"},
                {"url": "body_mesh_chunks/window_000.bin.gz"},
            ],
        },
    )
    _write_json(source_dir / "body_mesh_index" / "body_mesh_faces.json", {"faces": []})
    chunk_path = source_dir / "body_mesh_index" / "body_mesh_chunks" / "window_000.bin.gz"
    chunk_path.parent.mkdir(parents=True)
    chunk_path.write_bytes(b"chunk")
    _write_json(
        source_dir / "replay_scene.json",
        {
            "court_glb": "replay_review/court.glb",
            "points": [
                {"glb_url": "replay_review/points/point_001.glb"},
                {"glb_url": "replay_review/points/point_001.glb"},
            ],
        },
    )
    court_glb = source_dir / "replay_review" / "court.glb"
    point_glb = source_dir / "replay_review" / "points" / "point_001.glb"
    court_glb.parent.mkdir(parents=True)
    point_glb.parent.mkdir(parents=True)
    court_glb.write_bytes(b"court")
    point_glb.write_bytes(b"point")

    closure = collect_manifest_asset_closure(
        manifest_path=source_dir / "replay_viewer_manifest.json",
        video_path=video_path,
    )
    relative_paths = [asset.relative_path.as_posix() for asset in closure]
    assert relative_paths == sorted(set(relative_paths))
    assert set(relative_paths) == {
        "body_mesh_index/body_mesh_chunks/window_000.bin.gz",
        "body_mesh_index/body_mesh_faces.json",
        "body_mesh_index/body_mesh_index.json",
        "replay_review/court.glb",
        "replay_review/points/point_001.glb",
        "replay_scene.json",
        "replay_viewer_manifest.json",
        "source.mp4",
    }

    bundle_dir = tmp_path / "bundle"
    staged = stage_manifest_delivery_bundle(
        source_dir=source_dir,
        bundle_dir=bundle_dir,
        video_path=video_path,
        resolve=lambda path: f"/api/jobs/job_1/artifacts/{path}",
    )

    assert set(staged) == {Path(path) for path in relative_paths}
    for relative_path in relative_paths:
        assert (bundle_dir / relative_path).is_file()
    manifest = json.loads((bundle_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["body_mesh_index_url"] == (
        "/api/jobs/job_1/artifacts/body_mesh_index/body_mesh_index.json"
    )
    assert manifest["replay_scene_url"] == "/api/jobs/job_1/artifacts/replay_scene.json"
    assert manifest["label_overlays"][0]["url"] == "/api/jobs/job_1/artifacts/replay_scene.json"
    body_index = json.loads(
        (bundle_dir / "body_mesh_index" / "body_mesh_index.json").read_text(encoding="utf-8")
    )
    assert body_index["faces_url"] == "body_mesh_faces.json"
    assert body_index["windows"][0]["url"] == "body_mesh_chunks/window_000.bin.gz"


def test_recursive_manifest_closure_rejects_missing_advertised_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"body_mesh_index_url": "body_mesh_index/body_mesh_index.json"},
    )
    _write_json(
        source_dir / "body_mesh_index" / "body_mesh_index.json",
        {"faces_url": "missing_faces.json"},
    )

    with pytest.raises(FileNotFoundError, match="missing_faces.json"):
        collect_manifest_asset_closure(manifest_path=source_dir / "replay_viewer_manifest.json")


def test_recursive_manifest_closure_rebundles_already_delivered_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    video_path = tmp_path / "input" / "clip.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {
            "clip": "clip_1",
            "video_url": "/api/jobs/job_1/artifacts/source.mp4",
            "virtual_world_url": "bundles/clip_1/nested/world.json",
            "ball_arc_url": "bundles/clip_1/jobs/job_1/generations/gen_1/arc.json",
        },
    )
    _write_json(source_dir / "nested" / "world.json", {"world": True})
    _write_json(source_dir / "arc.json", {"arc": True})

    bundle_dir = tmp_path / "bundle"
    staged = stage_manifest_delivery_bundle(
        source_dir=source_dir,
        bundle_dir=bundle_dir,
        video_path=video_path,
        resolve=lambda path: f"bundles/clip_1/{path}",
    )

    assert Path("nested/world.json") in staged
    assert Path("arc.json") in staged
    assert (bundle_dir / "nested" / "world.json").is_file()
    manifest = json.loads((bundle_dir / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["virtual_world_url"] == "bundles/clip_1/nested/world.json"


def test_staging_prunes_obsolete_files_only_after_success(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _write_json(source_dir / "replay_viewer_manifest.json", {"virtual_world_url": "world.json"})
    _write_json(source_dir / "world.json", {})
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "obsolete.bin").write_bytes(b"old")

    stage_manifest_delivery_bundle(
        source_dir=source_dir,
        bundle_dir=bundle_dir,
        video_path=video_path,
        resolve=lambda path: f"bundles/clip_1/{path}",
        prune_destination=True,
    )

    assert not (bundle_dir / "obsolete.bin").exists()
    assert (bundle_dir / "world.json").is_file()


def test_delivered_url_detection_does_not_strip_ordinary_artifacts_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"virtual_world_url": "foo/artifacts/world.json"},
    )
    _write_json(source_dir / "foo" / "artifacts" / "world.json", {})

    closure = collect_manifest_asset_closure(manifest_path=source_dir / "replay_viewer_manifest.json")

    assert {asset.relative_path.as_posix() for asset in closure} == {
        "foo/artifacts/world.json",
        "replay_viewer_manifest.json",
    }


def test_staging_rewrites_absolute_child_json_references_to_relative_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    remote_root = "/@fs//srv/runs/clip_1/body_mesh_index"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"clip": "clip_1", "body_mesh_index_url": f"{remote_root}/body_mesh_index.json"},
    )
    _write_json(
        source_dir / "body_mesh_index" / "body_mesh_index.json",
        {
            "faces_url": f"{remote_root}/body_mesh_faces.json",
            "windows": [{"url": f"{remote_root}/body_mesh_chunks/window_000.bin.gz"}],
        },
    )
    _write_json(source_dir / "body_mesh_index" / "body_mesh_faces.json", {})
    chunk = source_dir / "body_mesh_index" / "body_mesh_chunks" / "window_000.bin.gz"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"chunk")

    bundle_dir = tmp_path / "bundle"
    stage_manifest_delivery_bundle(
        source_dir=source_dir,
        bundle_dir=bundle_dir,
        video_path=video_path,
        resolve=lambda path: f"/artifacts/{path}",
    )

    child = json.loads((bundle_dir / "body_mesh_index" / "body_mesh_index.json").read_text(encoding="utf-8"))
    assert child["faces_url"] == "body_mesh_faces.json"
    assert child["windows"][0]["url"] == "body_mesh_chunks/window_000.bin.gz"


def test_recursive_manifest_closure_does_not_parse_large_leaf_json(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"clip": "clip_1", "virtual_world_url": "confidence_gated_world.json"},
    )
    (source_dir / "confidence_gated_world.json").write_text("not parsed as a manifest", encoding="utf-8")

    closure = collect_manifest_asset_closure(manifest_path=source_dir / "replay_viewer_manifest.json")

    assert {asset.relative_path.as_posix() for asset in closure} == {
        "confidence_gated_world.json",
        "replay_viewer_manifest.json",
    }


def test_staging_stream_compacts_leaf_json_without_changing_string_whitespace(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _write_json(source_dir / "replay_viewer_manifest.json", {"virtual_world_url": "world.json"})
    original = {"message": "keep spaces and \\n escapes", "values": [1, 2, 3]}
    world_path = source_dir / "world.json"
    world_path.write_text(json.dumps(original, indent=4) + "\n", encoding="utf-8")
    original_bytes = world_path.stat().st_size

    bundle_dir = tmp_path / "bundle"
    stage_manifest_delivery_bundle(
        source_dir=source_dir,
        bundle_dir=bundle_dir,
        video_path=video_path,
        resolve=lambda path: f"bundles/clip_1/{path}",
    )

    staged_world = bundle_dir / "world.json"
    assert json.loads(staged_world.read_text(encoding="utf-8")) == original
    assert staged_world.stat().st_size < original_bytes


def test_recursive_manifest_closure_rejects_wrong_same_basename_fallback(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {
            "clip": "clip_1",
            "virtual_world_url": "/@fs//srv/runs/clip_1/nested/world.json",
        },
    )
    _write_json(source_dir / "world.json", {"wrong": True})

    with pytest.raises(FileNotFoundError, match="nested/world.json"):
        collect_manifest_asset_closure(manifest_path=source_dir / "replay_viewer_manifest.json")


def test_recursive_manifest_closure_rejects_path_traversal(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"body_mesh_index_url": "body_mesh_index/body_mesh_index.json"},
    )
    _write_json(
        source_dir / "body_mesh_index" / "body_mesh_index.json",
        {"windows": [{"url": "../outside.bin"}]},
    )
    (source_dir / "outside.bin").write_bytes(b"outside")

    with pytest.raises(ValueError, match="traversal"):
        collect_manifest_asset_closure(manifest_path=source_dir / "replay_viewer_manifest.json")


def test_recursive_manifest_closure_rejects_absolute_escape(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"body_mesh_index_url": "body_mesh_index/body_mesh_index.json"},
    )
    _write_json(
        source_dir / "body_mesh_index" / "body_mesh_index.json",
        {"faces_url": f"/@fs/{outside}"},
    )

    with pytest.raises(ValueError, match="escapes allowed root"):
        collect_manifest_asset_closure(manifest_path=source_dir / "replay_viewer_manifest.json")


def test_staging_does_not_publish_new_manifest_when_copy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "source"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _write_json(
        source_dir / "replay_viewer_manifest.json",
        {"video_url": "clip.mp4", "virtual_world_url": "nested/world.json"},
    )
    _write_json(source_dir / "nested" / "world.json", {})
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    old_manifest = b'{"generation":"old"}'
    (bundle_dir / "replay_viewer_manifest.json").write_bytes(old_manifest)
    original_compact = pipeline_invocation._compact_json_file

    def fail_world_copy(source: Path, destination: Path, *, chunk_bytes: int = 1024 * 1024) -> None:
        if Path(source).name == "world.json":
            raise OSError("injected copy failure")
        original_compact(source, destination, chunk_bytes=chunk_bytes)

    monkeypatch.setattr(pipeline_invocation, "_compact_json_file", fail_world_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        stage_manifest_delivery_bundle(
            source_dir=source_dir,
            bundle_dir=bundle_dir,
            video_path=video_path,
            resolve=lambda path: f"/artifacts/{path}",
        )

    assert (bundle_dir / "replay_viewer_manifest.json").read_bytes() == old_manifest
    assert not list(tmp_path.glob(".bundle.delivery-*"))


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
    assert safe_slug("match_01-clip.a") == "match_01-clip.a"
    with pytest.raises(ValueError):
        safe_slug("match; rm -rf /")
