"""Shared `process_video.py` invocation + render-artifact plumbing (INFRA-2).

Extracted from `server/gpu_runner.py` so `SshGpuRunner` (SSH-push, one-wave
fallback) and the pull-worker daemon (`server/worker/daemon.py`) share ONE
definition of how the pipeline is invoked and how its outputs are prepared
for the replay viewer. Pure stdlib + `pathlib.Path` only -- no torch/cv2/
fastapi/boto3 -- so the tiny worker venv (httpx + boto3) never has to import
the heavy render-service dependency graph just to build a command line.

This is a PURE REFACTOR of pre-existing `server/gpu_runner.py` logic: every
function here reproduces the exact string/arg shape the SSH runner produced
before extraction. `tests/render_service/test_gpu_runner.py` is the proof --
it stays green, unmodified, against the refactored `SshGpuRunner`.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable

SAFE_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

RESOURCE_USAGE_ARTIFACT = "gpu_resource_usage.json"
PIPELINE_SUMMARY_ARTIFACT = "PIPELINE_SUMMARY.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_MONITOR_SOURCE = REPO_ROOT / "scripts" / "racketsport" / "monitor_process_resources.py"

# Resolves a bare artifact filename (e.g. "source.mp4") to whatever the
# manifest should point at: SSH keeps `/api/jobs/{id}/artifacts/{name}`,
# the pull-worker daemon rewrites to a bare `bundles/{clip_id}/{name}` S3 key
# (the API presigns a GET for it at serve time).
ManifestUrlResolver = Callable[[str], str]


def safe_slug(value: str) -> str:
    if not value or not SAFE_SLUG_PATTERN.match(value):
        raise ValueError(f"unsafe slug {value!r}; expected {SAFE_SLUG_PATTERN.pattern}")
    return value


def build_process_video_args(
    *,
    python: str,
    script: str,
    video: str,
    out: str,
    clip: str,
    model_root: str,
    sidecar: str | None,
    max_frames: int | None,
    wasb_repo: str | None = None,
    wasb_checkpoint: str | None = None,
    allow_auto_court: bool = False,
) -> list[str]:
    """Build the `process_video.py` argv, minus the caller's own extras
    (SSH appends `--court-corners`/`--court-calibration` itself; the daemon
    has no such inputs on the queue path). Caller resolves every path to
    whatever namespace it runs in (remote SSH paths, or local worker paths)
    before calling this -- this function is a pure string/list builder.
    """
    args = [
        python,
        script,
        "--video",
        video,
        "--out",
        out,
        "--clip",
        safe_slug(clip),
        "--body-local",
        "--device",
        "cuda:0",
        "--json",
        "--manifest",
        f"{model_root}/models/MANIFEST.json",
        "--reid-model",
        f"{model_root}/models/checkpoints/osnet_x1_0_market1501.pt",
    ]
    if allow_auto_court:
        args.append("--allow-auto-court-corners-preview")
    if wasb_repo:
        args.extend(["--wasb-repo", wasb_repo])
    if wasb_checkpoint:
        args.extend(["--wasb-checkpoint", wasb_checkpoint])
    if max_frames is not None:
        args.extend(["--max-frames", str(max_frames)])
    if sidecar:
        args.extend(["--capture-sidecar", sidecar])
    return args


def remote_model_root(python_path: str) -> str:
    """Given a venv python path, return the repo/model root above it.

    `.../<root>/.venv/bin/python` -> `.../<root>`. Works identically for a
    remote SSH path string or a local worker path string -- purely textual.
    """
    marker = "/.venv/"
    if marker in python_path:
        return python_path.split(marker, 1)[0].rstrip("/")
    return str(Path(python_path).parent.parent.parent)


def copy_resource_monitor_to_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESOURCE_MONITOR_SOURCE, input_dir / RESOURCE_MONITOR_SOURCE.name)


def copy_source_video_artifact(*, video_path: Path, artifacts_dir: Path) -> str:
    """Copy the source video into `artifacts_dir` as `source<ext>`.

    Returns the artifact filename written (e.g. `source.mp4`).
    """
    suffix = video_path.suffix.lower() or ".mp4"
    target = artifacts_dir / f"source{suffix}"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        target.unlink()
    if not target.is_file():
        shutil.copy2(video_path, target)
    return target.name


def rewrite_manifest_urls(
    *,
    artifacts_dir: Path,
    video_path: Path,
    resolve: ManifestUrlResolver,
) -> None:
    """Rewrite every `*_url` string in `replay_viewer_manifest.json` (if
    present) from a raw pipeline filesystem path to whatever `resolve`
    returns for the bare artifact filename. `video_url` always resolves to
    the copied `source<ext>` artifact, not the original upload path.
    """
    manifest_path = artifacts_dir / "replay_viewer_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    source_artifact_name = f"source{video_path.suffix.lower() or '.mp4'}"
    rewritten = _rewrite_manifest_value(
        payload,
        resolve=resolve,
        source_artifact_name=source_artifact_name,
    )
    manifest_path.write_text(json.dumps(rewritten, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_manifest_value(
    value: object,
    *,
    resolve: ManifestUrlResolver,
    source_artifact_name: str,
    key: str | None = None,
) -> object:
    if isinstance(value, dict):
        return {
            str(child_key): _rewrite_manifest_value(
                child_value,
                resolve=resolve,
                source_artifact_name=source_artifact_name,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_manifest_value(item, resolve=resolve, source_artifact_name=source_artifact_name, key=key)
            for item in value
        ]
    if isinstance(value, str) and key is not None and key.endswith("_url"):
        artifact_name = _artifact_name_from_pipeline_url(value)
        if artifact_name is not None:
            if key == "video_url":
                artifact_name = source_artifact_name
            return resolve(artifact_name)
    return value


def _artifact_name_from_pipeline_url(value: str) -> str | None:
    if value.startswith("/@fs//"):
        return Path(value.removeprefix("/@fs//")).name
    if value.startswith("/@fs/"):
        return Path(value.removeprefix("/@fs/")).name
    if value.startswith("/"):
        path = Path(value)
        if "runs" in path.parts and path.name:
            return path.name
    return None


def prepare_render_artifacts(*, artifacts_dir: Path, video_path: Path, resolve: ManifestUrlResolver) -> None:
    """Copy the source video into `artifacts_dir` and rewrite the manifest's
    `*_url` fields via `resolve`. Shared tail step for every runner (SSH,
    local, and the pull-worker's bundle-prep) that produces a
    `replay_viewer_manifest.json`.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copy_source_video_artifact(video_path=video_path, artifacts_dir=artifacts_dir)
    rewrite_manifest_urls(artifacts_dir=artifacts_dir, video_path=video_path, resolve=resolve)
