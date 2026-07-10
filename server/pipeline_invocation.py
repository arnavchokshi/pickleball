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

import ctypes
import json
import os
import posixpath
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

from .bundle_policy import iter_policy_package_files

SAFE_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

RESOURCE_USAGE_ARTIFACT = "gpu_resource_usage.json"
PIPELINE_SUMMARY_ARTIFACT = "PIPELINE_SUMMARY.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_MONITOR_SOURCE = REPO_ROOT / "scripts" / "racketsport" / "monitor_process_resources.py"

# Resolves a bundle-relative artifact path (e.g. "source.mp4" or
# "body_mesh_index/body_mesh_index.json") to whatever the manifest should
# point at: SSH keeps `/api/jobs/{id}/artifacts/{path}`, the pull-worker daemon
# rewrites to a bare `bundles/{clip_id}/{path}` S3 key
# (the API presigns a GET for it at serve time).
ManifestUrlResolver = Callable[[str], str]

REPLAY_MANIFEST_ARTIFACT = "replay_viewer_manifest.json"
_ADDITIONAL_ASSET_REFERENCE_KEYS = frozenset({"court_glb"})


@dataclass(frozen=True)
class ManifestAsset:
    """One file in the transitive replay delivery closure.

    ``relative_path`` is the path the bundle must preserve. ``source_path``
    may be elsewhere under the caller-approved staging root (the local runner
    first writes to a sibling ``process_video`` directory), but it is never an
    unchecked path supplied by a manifest.
    """

    relative_path: Path
    source_path: Path


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
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix.lower() or ".mp4"
    target = artifacts_dir / f"source{suffix}"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        target.unlink()
    if not target.is_file():
        shutil.copy2(video_path, target)
    return target.name


def collect_manifest_asset_closure(
    *,
    manifest_path: Path,
    asset_root: Path | None = None,
    video_path: Path | None = None,
    allowed_source_root: Path | None = None,
    allow_missing_assets: bool = False,
) -> tuple[ManifestAsset, ...]:
    """Collect and validate the replay manifest's transitive local assets.

    Local references are values whose key is ``url`` or ends in ``_url``.
    ``court_glb`` is also included because it is the replay-scene court asset.
    Referenced JSON is traversed recursively, so a manifest's body index pulls
    in its faces/chunks and a replay scene pulls in its point/court GLBs.

    Every returned path is unique and bundle-relative. Relative traversal,
    absolute paths outside ``allowed_source_root``, symlink escapes, ambiguous
    remote-path suffixes, and missing advertised files fail closed.
    """
    closure, _ = _collect_manifest_asset_closure(
        manifest_path=manifest_path,
        asset_root=asset_root,
        video_path=video_path,
        allowed_source_root=allowed_source_root,
        allow_missing_assets=allow_missing_assets,
    )
    return closure


def _collect_manifest_asset_closure(
    *,
    manifest_path: Path,
    asset_root: Path | None,
    video_path: Path | None,
    allowed_source_root: Path | None,
    allow_missing_assets: bool = False,
) -> tuple[tuple[ManifestAsset, ...], dict[str, object]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"replay manifest is missing: {manifest_path}")

    root = (asset_root or manifest_path.parent).resolve()
    allowed_root = (allowed_source_root or root).resolve()
    manifest_source = manifest_path.resolve()
    _require_path_within(manifest_source, root, label="manifest")
    _require_path_within(root, allowed_root, label="asset root")

    manifest_relative = manifest_source.relative_to(root)
    payload = _read_json_object(manifest_source, label=manifest_relative.as_posix())
    clip_value = payload.get("clip")
    clip = str(clip_value) if isinstance(clip_value, str) and clip_value else None

    assets: dict[str, ManifestAsset] = {}
    manifest_asset = ManifestAsset(relative_path=manifest_relative, source_path=manifest_source)
    _add_manifest_asset(assets, manifest_asset)
    if video_path is not None:
        video_source = video_path.resolve()
        if not video_source.is_file():
            raise FileNotFoundError(f"advertised video is missing: {video_path}")
        _add_manifest_asset(
            assets,
            ManifestAsset(
                relative_path=Path(f"source{video_path.suffix.lower() or '.mp4'}"),
                source_path=video_source,
            ),
        )

    pending_json: list[tuple[ManifestAsset, dict[str, object]]] = [(manifest_asset, payload)]
    visited_json: set[str] = set()
    while pending_json:
        document, document_payload = pending_json.pop()
        document_key = document.relative_path.as_posix()
        if document_key in visited_json:
            continue
        visited_json.add(document_key)

        for key, value in _iter_asset_references(document_payload):
            try:
                asset = _resolve_manifest_reference(
                    key=key,
                    value=value,
                    document=document,
                    asset_root=root,
                    allowed_source_root=allowed_root,
                    clip=clip,
                    video_path=video_path,
                )
            except FileNotFoundError:
                if allow_missing_assets:
                    continue
                raise
            if asset is None:
                continue
            is_new = _add_manifest_asset(assets, asset)
            if is_new and _should_traverse_json(asset.relative_path):
                child_payload = _read_json_object(asset.source_path, label=asset.relative_path.as_posix())
                pending_json.append((asset, child_payload))

    closure = tuple(assets[key] for key in sorted(assets))
    return closure, payload


def stage_manifest_delivery_bundle(
    *,
    source_dir: Path,
    bundle_dir: Path,
    video_path: Path,
    resolve: ManifestUrlResolver,
    allowed_source_root: Path | None = None,
    prune_destination: bool = False,
    allow_missing_assets: bool = False,
) -> tuple[Path, ...]:
    """Validate and stage a complete replay bundle, publishing manifest last.

    All source reads and copies finish in a sibling temporary directory before
    any new destination file is installed. Files are then atomically replaced
    one by one and the rewritten manifest is replaced last, so a failed copy
    never publishes a manifest that advertises an incomplete closure.
    """
    source_root = source_dir.resolve()
    destination_root = bundle_dir.resolve()
    manifest_path = source_root / REPLAY_MANIFEST_ARTIFACT
    closure, manifest_payload = _collect_manifest_asset_closure(
        manifest_path=manifest_path,
        asset_root=source_root,
        video_path=video_path,
        allowed_source_root=allowed_source_root or source_root,
        allow_missing_assets=allow_missing_assets,
    )
    assets = {asset.relative_path.as_posix(): asset for asset in closure}
    for source_path in iter_policy_package_files(source_root):
        relative_path = source_path.relative_to(source_root)
        _add_manifest_asset(
            assets,
            ManifestAsset(relative_path=relative_path, source_path=source_path),
        )
    closure = tuple(assets[key] for key in sorted(assets))
    manifest_asset = next(asset for asset in closure if asset.relative_path == Path(REPLAY_MANIFEST_ARTIFACT))
    clip_value = manifest_payload.get("clip")
    clip = str(clip_value) if isinstance(clip_value, str) and clip_value else None
    manifest_payload = _advertise_policy_artifacts(manifest_payload, source_root)
    rewritten_manifest = _rewrite_manifest_value(
        manifest_payload,
        resolve=resolve,
        document=manifest_asset,
        asset_root=source_root,
        allowed_source_root=(allowed_source_root or source_root).resolve(),
        clip=clip,
        video_path=video_path,
        allow_missing_assets=allow_missing_assets,
    )
    rewritten_children: dict[Path, object] = {}
    for asset in closure:
        if asset.relative_path == Path(REPLAY_MANIFEST_ARTIFACT) or not _should_traverse_json(asset.relative_path):
            continue
        child_payload = _read_json_object(asset.source_path, label=asset.relative_path.as_posix())
        child_parent = asset.relative_path.parent.as_posix()
        if child_parent == ".":
            child_parent = ""

        def resolve_child(target: str, *, _parent: str = child_parent) -> str:
            return posixpath.relpath(target, start=_parent or ".")

        rewritten_children[asset.relative_path] = _rewrite_manifest_value(
            child_payload,
            resolve=resolve_child,
            document=asset,
            asset_root=source_root,
            allowed_source_root=(allowed_source_root or source_root).resolve(),
            clip=clip,
            video_path=video_path,
            allow_missing_assets=allow_missing_assets,
        )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.delivery-",
            dir=str(destination_root.parent),
        )
    )
    staged_paths: list[Path] = []
    try:
        for asset in closure:
            if asset.relative_path == Path(REPLAY_MANIFEST_ARTIFACT):
                continue
            rewritten_child = rewritten_children.get(asset.relative_path)
            staged = _safe_bundle_destination(stage_root, asset.relative_path)
            staged.parent.mkdir(parents=True, exist_ok=True)
            if rewritten_child is None:
                if asset.relative_path.suffix.lower() == ".json" and destination_root != source_root:
                    _compact_json_file(asset.source_path, staged)
                else:
                    shutil.copy2(asset.source_path, staged)
            else:
                staged.write_text(
                    json.dumps(rewritten_child, separators=(",", ":"), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            staged_paths.append(asset.relative_path)

        staged_manifest = stage_root / REPLAY_MANIFEST_ARTIFACT
        staged_manifest.write_text(
            json.dumps(rewritten_manifest, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_paths.append(Path(REPLAY_MANIFEST_ARTIFACT))
        _verify_staged_delivery_bundle(stage_root, staged_paths)
        if destination_root.exists():
            if not prune_destination:
                raise FileExistsError(f"delivery bundle destination already exists: {destination_root}")
            _atomic_exchange_directories(stage_root, destination_root)
            # The old generation now lives at stage_root and is not visible at
            # the published path. It is removed only after the atomic swap.
        else:
            os.replace(stage_root, destination_root)
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)

    return tuple(staged_paths)


def rewrite_manifest_urls(
    *,
    artifacts_dir: Path,
    video_path: Path,
    resolve: ManifestUrlResolver,
    allow_missing_assets: bool = False,
) -> None:
    """Validate/stage the recursive closure and rewrite its manifest URLs.

    Nested paths are preserved. The source video is always delivered as
    ``source<ext>``. Missing manifests retain the historical no-op behavior.
    """
    manifest_path = artifacts_dir / REPLAY_MANIFEST_ARTIFACT
    if not manifest_path.is_file():
        return
    stage_manifest_delivery_bundle(
        source_dir=artifacts_dir,
        bundle_dir=artifacts_dir,
        video_path=video_path,
        resolve=resolve,
        # The local runner writes its first bundle into a sibling directory;
        # SSH/worker assets already live under artifacts_dir. Both are bounded
        # by the per-job parent directory.
        allowed_source_root=artifacts_dir.parent,
        prune_destination=True,
        allow_missing_assets=allow_missing_assets,
    )


def _advertise_policy_artifacts(payload: dict[str, object], source_root: Path) -> dict[str, object]:
    """Add same-run stats/coaching URLs before the manifest is staged.

    Older runners create the manifest before these facts. The server may add
    links to artifacts that actually exist, but it never invents an artifact
    or upgrades bundle status when either file is absent.
    """

    advertised = dict(payload)
    for field, filename in (
        ("match_stats_url", "match_stats.json"),
        ("coaching_card_facts_url", "coaching_card_facts.json"),
    ):
        if (source_root / filename).is_file():
            advertised[field] = filename
    return advertised


def _rewrite_manifest_value(
    value: object,
    *,
    resolve: ManifestUrlResolver,
    document: ManifestAsset,
    asset_root: Path,
    allowed_source_root: Path,
    clip: str | None,
    video_path: Path,
    key: str | None = None,
    allow_missing_assets: bool = False,
) -> object:
    if isinstance(value, dict):
        return {
            str(child_key): _rewrite_manifest_value(
                child_value,
                resolve=resolve,
                document=document,
                asset_root=asset_root,
                allowed_source_root=allowed_source_root,
                clip=clip,
                video_path=video_path,
                key=str(child_key),
                allow_missing_assets=allow_missing_assets,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_manifest_value(
                item,
                resolve=resolve,
                document=document,
                asset_root=asset_root,
                allowed_source_root=allowed_source_root,
                clip=clip,
                video_path=video_path,
                key=key,
                allow_missing_assets=allow_missing_assets,
            )
            for item in value
        ]
    if isinstance(value, str) and key is not None and _is_asset_reference_key(key):
        try:
            asset = _resolve_manifest_reference(
                key=key,
                value=value,
                document=document,
                asset_root=asset_root,
                allowed_source_root=allowed_source_root,
                clip=clip,
                video_path=video_path,
            )
        except FileNotFoundError:
            if allow_missing_assets:
                return value
            raise
        if asset is not None:
            return resolve(asset.relative_path.as_posix())
    return value


def _iter_asset_references(value: object) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []

    def visit(current: object, key: str | None = None) -> None:
        if isinstance(current, dict):
            for child_key, child_value in current.items():
                visit(child_value, str(child_key))
        elif isinstance(current, list):
            for item in current:
                visit(item, key)
        elif isinstance(current, str) and key is not None and _is_asset_reference_key(key):
            references.append((key, current))

    visit(value)
    return references


def _is_asset_reference_key(key: str) -> bool:
    return key == "url" or key.endswith("_url") or key in _ADDITIONAL_ASSET_REFERENCE_KEYS


def _should_traverse_json(relative_path: Path) -> bool:
    """Return true only for small JSON documents that advertise more assets."""

    return relative_path.name in {
        REPLAY_MANIFEST_ARTIFACT,
        "body_mesh_index.json",
        "replay_scene.json",
    }


def _resolve_manifest_reference(
    *,
    key: str,
    value: str,
    document: ManifestAsset,
    asset_root: Path,
    allowed_source_root: Path,
    clip: str | None,
    video_path: Path | None,
) -> ManifestAsset | None:
    if key == "video_url" and video_path is not None:
        source = video_path.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"advertised video is missing: {video_path}")
        return ManifestAsset(
            relative_path=Path(f"source{video_path.suffix.lower() or '.mp4'}"),
            source_path=source,
        )

    parsed = urlsplit(value)
    if parsed.scheme:
        if parsed.scheme.lower() == "file":
            raise ValueError(f"absolute file URL is not allowed in {document.relative_path}: {value!r}")
        return None
    if parsed.netloc or value.startswith("//"):
        raise ValueError(f"protocol-relative asset URL is not allowed in {document.relative_path}: {value!r}")

    raw_path = unquote(parsed.path)
    if not raw_path:
        return None
    if "\\" in raw_path:
        raise ValueError(f"backslash asset path is not allowed in {document.relative_path}: {value!r}")

    if raw_path.startswith("/@fs/"):
        filesystem_value = raw_path.removeprefix("/@fs/")
        if not filesystem_value.startswith("/"):
            filesystem_value = f"/{filesystem_value}"
        return _resolve_absolute_reference(
            Path(filesystem_value),
            document=document,
            asset_root=asset_root,
            allowed_source_root=allowed_source_root,
            clip=clip,
            original=value,
            allow_remote_mapping=True,
        )

    delivered_relative = _delivered_relative_path(raw_path, clip=clip, original=value)
    if delivered_relative is not None:
        return _asset_from_relative_path(
            delivered_relative,
            asset_root=asset_root,
            allowed_source_root=allowed_source_root,
        )

    reference_path = Path(raw_path)
    if reference_path.is_absolute():
        return _resolve_absolute_reference(
            reference_path,
            document=document,
            asset_root=asset_root,
            allowed_source_root=allowed_source_root,
            clip=clip,
            original=value,
            allow_remote_mapping="runs" in reference_path.parts,
        )

    parts = tuple(part for part in reference_path.parts if part not in ("", "."))
    if not parts or ".." in parts:
        raise ValueError(f"asset path traversal is not allowed in {document.relative_path}: {value!r}")
    relative_path = document.relative_path.parent.joinpath(*parts)
    _validate_relative_bundle_path(relative_path, original=value)

    source_candidate = document.source_path.parent.joinpath(*parts)
    if not source_candidate.exists():
        source_candidate = asset_root / relative_path
    source = _validated_source_file(
        source_candidate,
        allowed_source_root=allowed_source_root,
        advertised_path=relative_path,
    )
    return ManifestAsset(relative_path=relative_path, source_path=source)


def _resolve_absolute_reference(
    absolute_path: Path,
    *,
    document: ManifestAsset,
    asset_root: Path,
    allowed_source_root: Path,
    clip: str | None,
    original: str,
    allow_remote_mapping: bool,
) -> ManifestAsset:
    if not absolute_path.is_absolute():
        raise ValueError(f"expected absolute asset path in {document.relative_path}: {original!r}")

    if absolute_path.exists():
        source = _validated_source_file(
            absolute_path,
            allowed_source_root=allowed_source_root,
            advertised_path=absolute_path,
        )
        try:
            relative_path = source.relative_to(asset_root)
        except ValueError:
            relative_path = _relative_suffix_for_absolute_path(absolute_path, asset_root=asset_root, clip=clip)
        _validate_relative_bundle_path(relative_path, original=original)
        return ManifestAsset(relative_path=relative_path, source_path=source)

    if not allow_remote_mapping and not clip:
        raise ValueError(f"absolute asset escapes the delivery root in {document.relative_path}: {original!r}")
    matched = _match_remote_asset_within_root(absolute_path, asset_root=asset_root, clip=clip)
    if matched is None:
        raise FileNotFoundError(f"advertised asset is missing: {original!r}")
    relative_path, source = matched
    return ManifestAsset(relative_path=relative_path, source_path=source)


def _relative_suffix_for_absolute_path(absolute_path: Path, *, asset_root: Path, clip: str | None) -> Path:
    if clip:
        candidates = _suffixes_after_component(absolute_path, clip)
        if candidates:
            return candidates[-1]
    raise ValueError(f"cannot preserve a unique bundle-relative path for absolute asset: {absolute_path}")


def _match_remote_asset_within_root(
    absolute_path: Path,
    *,
    asset_root: Path,
    clip: str | None,
) -> tuple[Path, Path] | None:
    if not clip:
        return None
    candidates = _suffixes_after_component(absolute_path, clip)

    seen: set[str] = set()
    matches: list[tuple[int, Path, Path]] = []
    for candidate in candidates:
        candidate_key = candidate.as_posix()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        try:
            _validate_relative_bundle_path(candidate, original=str(absolute_path))
        except ValueError:
            continue
        source_candidate = asset_root / candidate
        if source_candidate.is_file():
            source = _validated_source_file(
                source_candidate,
                allowed_source_root=asset_root,
                advertised_path=candidate,
            )
            matches.append((len(candidate.parts), candidate, source))

    if not matches:
        return None
    max_depth = max(depth for depth, _, _ in matches)
    deepest = [(relative, source) for depth, relative, source in matches if depth == max_depth]
    unique = {(relative.as_posix(), str(source)): (relative, source) for relative, source in deepest}
    if len(unique) != 1:
        raise ValueError(f"ambiguous remote asset path cannot be staged safely: {absolute_path}")
    return next(iter(unique.values()))


def _suffixes_after_component(path: Path, component: str) -> list[Path]:
    parts = tuple(part for part in path.parts if part not in ("", "/"))
    return [Path(*parts[index + 1 :]) for index, part in enumerate(parts[:-1]) if part == component]


def _delivered_relative_path(raw_path: str, *, clip: str | None, original: str) -> Path | None:
    stripped = raw_path.lstrip("/")
    parts = tuple(Path(stripped).parts)
    if not parts:
        return None
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"asset path traversal is not allowed: {original!r}")

    relative_parts: tuple[str, ...] | None = None
    if parts[0] == "artifacts":
        relative_parts = parts[1:]
    elif len(parts) >= 5 and parts[0:2] == ("api", "jobs") and parts[3] == "artifacts":
        relative_parts = parts[4:]
    elif parts[0] == "bundles":
        relative_parts = parts[1:]
        if clip and relative_parts and relative_parts[0] == clip:
            relative_parts = relative_parts[1:]
        if (
            len(relative_parts) >= 5
            and relative_parts[0] == "jobs"
            and relative_parts[2] == "generations"
        ):
            relative_parts = relative_parts[4:]
    if relative_parts is None:
        return None
    if not relative_parts:
        raise ValueError(f"delivered asset URL has no relative path: {original!r}")
    relative_path = Path(*relative_parts)
    _validate_relative_bundle_path(relative_path, original=original)
    return relative_path


def _asset_from_relative_path(
    relative_path: Path,
    *,
    asset_root: Path,
    allowed_source_root: Path,
) -> ManifestAsset:
    source = _validated_source_file(
        asset_root / relative_path,
        allowed_source_root=allowed_source_root,
        advertised_path=relative_path,
    )
    return ManifestAsset(relative_path=relative_path, source_path=source)


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in advertised asset {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"advertised JSON asset must be an object: {label}")
    return payload


def _add_manifest_asset(assets: dict[str, ManifestAsset], asset: ManifestAsset) -> bool:
    key = asset.relative_path.as_posix()
    existing = assets.get(key)
    if existing is None:
        assets[key] = asset
        return True
    if existing.source_path != asset.source_path:
        raise ValueError(
            f"two source files map to the same delivery path {key!r}: "
            f"{existing.source_path} and {asset.source_path}"
        )
    return False


def _validated_source_file(
    path: Path,
    *,
    allowed_source_root: Path,
    advertised_path: Path,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"advertised asset is missing: {advertised_path}")
    source = path.resolve()
    _require_path_within(source, allowed_source_root, label=f"advertised asset {advertised_path}")
    if not source.is_file():
        raise FileNotFoundError(f"advertised asset is not a file: {advertised_path}")
    return source


def _validate_relative_bundle_path(path: Path, *, original: str) -> None:
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe bundle-relative asset path {original!r}")


def _require_path_within(path: Path, root: Path, *, label: str) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"{label} escapes allowed root {root}: {path}")


def _safe_bundle_destination(root: Path, relative_path: Path) -> Path:
    _validate_relative_bundle_path(relative_path, original=relative_path.as_posix())
    destination = (root / relative_path).resolve(strict=False)
    _require_path_within(destination, root, label="bundle destination")
    return destination


def _verify_staged_delivery_bundle(stage_root: Path, staged_paths: list[Path]) -> None:
    expected = {path.as_posix() for path in staged_paths}
    actual = {
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if expected != actual:
        raise RuntimeError(
            "staged delivery bundle verification failed: "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
        )
    manifest = stage_root / REPLAY_MANIFEST_ARTIFACT
    _read_json_object(manifest, label=REPLAY_MANIFEST_ARTIFACT)


def _atomic_exchange_directories(staged: Path, published: Path) -> None:
    """Atomically swap two directories or fail closed on this platform."""

    staged_bytes = os.fsencode(staged)
    published_bytes = os.fsencode(published)
    libc = ctypes.CDLL(None, use_errno=True)
    result = -1
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename_swap = 0x00000002
        result = libc.renamex_np(
            ctypes.c_char_p(staged_bytes),
            ctypes.c_char_p(published_bytes),
            ctypes.c_uint(rename_swap),
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        at_fdcwd = -100
        rename_exchange = 0x2
        result = libc.renameat2(
            ctypes.c_int(at_fdcwd),
            ctypes.c_char_p(staged_bytes),
            ctypes.c_int(at_fdcwd),
            ctypes.c_char_p(published_bytes),
            ctypes.c_uint(rename_exchange),
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            "atomic directory exchange is unavailable; refusing non-atomic bundle replacement",
            str(published),
        )


def _compact_json_file(source: Path, destination: Path, *, chunk_bytes: int = 1024 * 1024) -> None:
    """Strip insignificant JSON whitespace with bounded memory.

    This is a lexical transform: UTF-8 bytes and whitespace inside strings are
    copied exactly, while JSON whitespace outside strings is omitted.
    """

    in_string = False
    escaped = False
    with source.open("rb") as src, destination.open("wb") as dst:
        while chunk := src.read(chunk_bytes):
            compacted = bytearray()
            for value in chunk:
                if in_string:
                    compacted.append(value)
                    if escaped:
                        escaped = False
                    elif value == ord("\\"):
                        escaped = True
                    elif value == ord('"'):
                        in_string = False
                elif value == ord('"'):
                    in_string = True
                    compacted.append(value)
                elif value not in (9, 10, 13, 32):
                    compacted.append(value)
            dst.write(compacted)
    if in_string or escaped:
        raise ValueError(f"unterminated JSON string in advertised asset: {source}")


def prepare_render_artifacts(
    *,
    artifacts_dir: Path,
    video_path: Path,
    resolve: ManifestUrlResolver,
    allow_missing_assets: bool = False,
) -> None:
    """Atomically stage the recursive replay closure and rewrite its URLs.

    Shared tail step for every runner (SSH, local, and the pull-worker's
    bundle-prep) that produces a `replay_viewer_manifest.json`.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not (artifacts_dir / REPLAY_MANIFEST_ARTIFACT).is_file():
        copy_source_video_artifact(video_path=video_path, artifacts_dir=artifacts_dir)
        return
    rewrite_manifest_urls(
        artifacts_dir=artifacts_dir,
        video_path=video_path,
        resolve=resolve,
        allow_missing_assets=allow_missing_assets,
    )
