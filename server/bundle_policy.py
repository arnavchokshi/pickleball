"""Fail-closed NS-01.5 policy for server-visible replay bundles.

The pipeline's process exit code is deliberately not an input to this policy.
Only the runner summary, the owned artifact tree, and the URLs advertised by
the published manifest decide whether a job is ``complete``, ``partial``, or
``failed``.
"""

from __future__ import annotations

import copy
import json
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import unquote, urlsplit

PIPELINE_SUMMARY_NAME = "PIPELINE_SUMMARY.json"
REPLAY_MANIFEST_NAME = "replay_viewer_manifest.json"
VALID_BUNDLE_STATUSES = frozenset({"complete", "partial", "failed"})

# Files which belong to the minimum inspectable bundle even when an older
# runner did not advertise them in replay_viewer_manifest.json. Directory
# assets are collected recursively by ``iter_policy_package_files``.
_PACKAGE_FILES = (
    PIPELINE_SUMMARY_NAME,
    "gpu_resource_usage.json",
    "source_identity.json",
    "capture_sidecar.json",
    "court_calibration.json",
    "frame_times.json",
    "tracks.json",
    "body_full_clip_gate.json",
    "body_mesh_index.json",
    "body_mesh.json",
    "ball_track.json",
    "ball_track_arc_solved.json",
    "contact_windows.json",
    "racket_pose_estimate.json",
    "virtual_world.json",
    "confidence_gated_world.json",
    "match_stats.json",
    "rally_metrics.json",
    "coaching_card_facts.json",
    "trust_bands.json",
    "replay_scene.json",
)
_PACKAGE_DIRECTORIES = ("body_mesh_index", "replay_review", "assets")


UrlExists = Callable[[str, str], bool]


@dataclass(frozen=True)
class BundlePolicyResult:
    status: str
    missing_capabilities: tuple[Any, ...]
    trust_bands: dict[str, Any]
    runner_status: str | None
    advertised_urls: tuple[str, ...]
    missing_urls: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "missing_capabilities": copy.deepcopy(list(self.missing_capabilities)),
            "trust_bands": copy.deepcopy(self.trust_bands),
            "runner_status": self.runner_status,
            "advertised_urls": list(self.advertised_urls),
            "missing_urls": list(self.missing_urls),
        }


def iter_policy_package_files(run_dir: Path) -> tuple[Path, ...]:
    """Return existing minimum-policy files, including directory contents."""

    root = run_dir.resolve()
    paths: set[Path] = set()
    for name in _PACKAGE_FILES:
        candidate = root / name
        if candidate.is_file():
            paths.add(candidate)
    for name in _PACKAGE_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            continue
        for candidate in directory.rglob("*"):
            if candidate.is_file():
                paths.add(candidate)
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def evaluate_bundle(
    run_dir: Path,
    *,
    runner_status: str | None = None,
    pipeline_summary: Mapping[str, Any] | None = None,
    url_exists: UrlExists | None = None,
) -> BundlePolicyResult:
    """Evaluate one local or published bundle without ever upgrading status.

    ``url_exists`` is used after publication. It receives the advertised URL
    and the relative JSON document containing it, allowing S3 key membership
    or an in-process API resolver to prove every URL without a socket bind.
    """

    root = run_dir.resolve()
    summary, summary_error = _load_summary(root, pipeline_summary)
    summary_status = summary.get("status") if isinstance(summary.get("status"), str) else None
    effective_runner_status = summary_status or runner_status

    runner_missing = summary.get("missing_capabilities")
    missing: list[Any] = copy.deepcopy(runner_missing) if isinstance(runner_missing, list) else []

    summary_trust = summary.get("trust_bands")
    trust_bands = copy.deepcopy(summary_trust) if isinstance(summary_trust, dict) else {}

    if summary_error is not None:
        _append_missing(missing, "summary", summary_error)
    if effective_runner_status not in VALID_BUNDLE_STATUSES:
        _append_missing(missing, "summary", "runner summary has no valid complete|partial|failed status")

    for capability, candidates, reason in _mandatory_requirements():
        if not any((root / candidate).is_file() for candidate in candidates):
            _append_missing(missing, capability, reason)
    if not (root / "body_full_clip_gate.json").is_file():
        _append_missing(missing, "body", "declared BODY full-clip coverage artifact is missing")

    if not _has_source_identity(root, summary):
        _append_missing(
            missing,
            "source_identity",
            "missing content-addressed source identity (source_identity.json or summary video sha256+size)",
        )

    trust_file = _read_json_object(root / "trust_bands.json")
    if trust_file is None:
        _append_missing(missing, "trust_bands", "trust_bands.json is missing or invalid")
    elif not trust_bands:
        _append_missing(missing, "trust_bands", "runner summary does not carry trust_bands")
    elif trust_file != trust_bands:
        _append_missing(missing, "trust_bands", "runner summary trust_bands differ from trust_bands.json")

    advertised_urls, missing_urls = _check_advertised_urls(root, url_exists=url_exists)
    if missing_urls:
        _append_missing(
            missing,
            "advertised_urls",
            "advertised URL(s) do not resolve: " + ", ".join(missing_urls),
        )

    if effective_runner_status == "failed" or summary_error is not None or effective_runner_status is None:
        status = "failed"
    elif effective_runner_status == "partial" or missing:
        status = "partial"
    else:
        status = "complete"

    return BundlePolicyResult(
        status=status,
        missing_capabilities=tuple(missing),
        trust_bands=trust_bands,
        runner_status=effective_runner_status,
        advertised_urls=tuple(advertised_urls),
        missing_urls=tuple(missing_urls),
    )


def s3_url_exists(*, published_keys: set[str], bundle_prefix: str) -> UrlExists:
    """Build a URL resolver for an immutable S3 bundle generation."""

    def exists(url: str, document_relative: str) -> bool:
        target = _published_target(url, document_relative=document_relative, bundle_prefix=bundle_prefix)
        return target is not None and target in published_keys

    return exists


def gate_reported_status(
    *,
    status: str,
    missing_capabilities: list[Any] | None,
    trust_bands: dict[str, Any] | None,
    bundle_policy: Mapping[str, Any] | None,
) -> str:
    """Fail closed when a worker completion report is inconsistent."""

    if status == "failed":
        return "failed"
    if status not in {"complete", "partial"}:
        return "partial"
    if not isinstance(bundle_policy, Mapping):
        return "partial"
    policy_status = bundle_policy.get("status")
    if policy_status == "failed":
        return "failed"
    if policy_status != status:
        return "partial"
    policy_missing = bundle_policy.get("missing_capabilities")
    policy_trust = bundle_policy.get("trust_bands")
    missing_urls = bundle_policy.get("missing_urls")
    if policy_missing != (missing_capabilities or []):
        return "partial"
    if policy_trust != (trust_bands or {}):
        return "partial"
    if status == "complete" and (policy_missing or missing_urls):
        return "partial"
    return status


def local_api_url_exists(*, bundle_root: Path, artifacts_marker: str = "/artifacts/") -> UrlExists:
    """Build an in-process resolver for local/SSH API artifact URLs."""

    root = bundle_root.resolve()

    def exists(url: str, document_relative: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            return False
        path = unquote(parsed.path)
        if artifacts_marker in path:
            relative = path.split(artifacts_marker, 1)[1]
        elif path.startswith("/"):
            return False
        else:
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(document_relative), path))
        return _safe_local_file(root, relative)

    return exists


def _mandatory_requirements() -> tuple[tuple[str, tuple[str, ...], str], ...]:
    return (
        ("capture_sidecar", ("capture_sidecar.json",), "capture sidecar is missing"),
        ("calibration", ("court_calibration.json",), "court/camera calibration is missing"),
        ("tracks", ("tracks.json",), "persistent player tracks are missing"),
        (
            "body",
            ("body_mesh_index/body_mesh_index.json", "body_mesh_index.json", "body_mesh.json"),
            "BODY mesh/coverage artifact is missing",
        ),
        ("ball", ("ball_track.json",), "ball artifact is missing"),
        ("events", ("contact_windows.json",), "event/contact artifact is missing"),
        ("ball_arc", ("ball_track_arc_solved.json",), "ball arc artifact is missing"),
        ("paddle", ("racket_pose_estimate.json",), "paddle artifact is missing"),
        (
            "fusion",
            ("confidence_gated_world.json", "virtual_world.json"),
            "fused-world artifact is missing",
        ),
        ("stats", ("match_stats.json",), "deterministic stats artifact is missing"),
        ("coaching", ("coaching_card_facts.json",), "deterministic coaching facts are missing"),
        ("assets", ("replay_scene.json",), "recursive replay assets are missing"),
        ("manifest", (REPLAY_MANIFEST_NAME,), "replay manifest is missing"),
        ("summary", (PIPELINE_SUMMARY_NAME,), "pipeline summary is missing"),
    )


def _load_summary(
    root: Path, supplied: Mapping[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    if supplied is not None:
        supplied_copy = copy.deepcopy(dict(supplied))
        path = root / PIPELINE_SUMMARY_NAME
        on_disk = _read_json_object(path)
        if on_disk is None:
            return supplied_copy, "PIPELINE_SUMMARY.json is missing or invalid"
        if on_disk != supplied_copy:
            return supplied_copy, "worker summary differs from packaged PIPELINE_SUMMARY.json"
        return supplied_copy, None
    path = root / PIPELINE_SUMMARY_NAME
    if not path.is_file():
        return {}, "PIPELINE_SUMMARY.json is missing"
    payload = _read_json_object(path)
    if payload is None:
        return {}, "PIPELINE_SUMMARY.json is invalid"
    return payload, None


def _has_source_identity(root: Path, summary: Mapping[str, Any]) -> bool:
    if (root / "source_identity.json").is_file():
        return True
    video = summary.get("video")
    if not isinstance(video, Mapping):
        return False
    sha256 = video.get("sha256")
    size = video.get("size") if isinstance(video.get("size"), int) else video.get("size_bytes")
    return isinstance(sha256, str) and len(sha256) == 64 and isinstance(size, int) and size >= 0


def _append_missing(missing: list[Any], capability: str, reason: str) -> None:
    if any(_missing_capability_name(item) == capability for item in missing):
        return
    missing.append({"capability": capability, "reason": reason})


def _missing_capability_name(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("capability", "name", "id"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def _check_advertised_urls(root: Path, *, url_exists: UrlExists | None) -> tuple[list[str], list[str]]:
    manifest = root / REPLAY_MANIFEST_NAME
    payload = _read_json_object(manifest)
    if payload is None:
        return [], []

    advertised: list[str] = []
    missing: list[str] = []
    pending: list[tuple[str, dict[str, Any]]] = [(REPLAY_MANIFEST_NAME, payload)]
    visited: set[str] = set()
    while pending:
        document_relative, document = pending.pop()
        if document_relative in visited:
            continue
        visited.add(document_relative)
        for url in _iter_url_values(document):
            advertised.append(url)
            exists = (
                url_exists(url, document_relative)
                if url_exists is not None
                else _local_reference_exists(root, url, document_relative=document_relative)
            )
            if not exists:
                missing.append(url)
                continue
            child_relative = _local_reference_relative(url, document_relative=document_relative)
            if child_relative and Path(child_relative).name in {"body_mesh_index.json", "replay_scene.json"}:
                child = _read_json_object(root / child_relative)
                if child is not None:
                    pending.append((child_relative, child))
    return _stable_unique(advertised), _stable_unique(missing)


def _iter_url_values(value: Any, key: str | None = None) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            yield from _iter_url_values(child_value, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _iter_url_values(child, key)
    elif isinstance(value, str) and key is not None and (
        key == "url" or key.endswith("_url") or key == "court_glb"
    ):
        yield value


def _local_reference_exists(root: Path, url: str, *, document_relative: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme.lower() == "file":
        return False
    if parsed.scheme or parsed.netloc:
        return False
    path = unquote(parsed.path)
    if path.startswith("/@fs/"):
        filesystem_path = path.removeprefix("/@fs/")
        if not filesystem_path.startswith("/"):
            filesystem_path = "/" + filesystem_path
        return Path(filesystem_path).is_file()
    relative = _local_reference_relative(url, document_relative=document_relative)
    return relative is not None and _safe_local_file(root, relative)


def _local_reference_relative(url: str, *, document_relative: str) -> str | None:
    path = unquote(urlsplit(url).path)
    if not path or path.startswith("/"):
        if "/artifacts/" in path:
            path = path.split("/artifacts/", 1)[1]
        else:
            return None
    parts = tuple(part for part in Path(path).parts if part not in ("", "."))
    if "generations" in parts:
        index = parts.index("generations")
        if len(parts) > index + 2:
            path = "/".join(parts[index + 2 :])
    elif parts[:1] == ("bundles",) and len(parts) >= 3:
        path = "/".join(parts[2:])
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(document_relative), path))
    if relative == ".." or relative.startswith("../"):
        return None
    return relative


def _published_target(url: str, *, document_relative: str, bundle_prefix: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    if path.startswith(bundle_prefix):
        return path
    if path.startswith("/"):
        return None
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(document_relative), path))
    if relative == ".." or relative.startswith("../"):
        return None
    return f"{bundle_prefix}{relative}"


def _safe_local_file(root: Path, relative: str) -> bool:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
