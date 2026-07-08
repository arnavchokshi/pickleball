"""Owner-capture intake, prelabel specs, and reviewed-corpus materialization."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eval_guard import EvalClipLeakError, PROTECTED_EVAL_CLIP_IDS, assert_not_training_on_eval_clip
from .io_decode import probe_clip

UTC = timezone.utc
PROTECTED_OWNER_EVAL_SLUGS: tuple[str, ...] = tuple(PROTECTED_EVAL_CLIP_IDS)
DEFAULT_OWNER_DATA_MANIFEST = Path("runs/owner_data/OWNER_DATA_MANIFEST.json")
DEFAULT_OWNER_DATA_ROOT = Path("runs/owner_data")
DEFAULT_OWNER_CORPUS_MANIFEST = Path("runs/training_corpora_20260702/owner_capture/manifest.json")


class OwnerCaptureError(ValueError):
    """Base class for owner-capture data-engine failures."""


class ProtectedEvalCaptureError(OwnerCaptureError):
    """Raised when intake sees a protected eval clip by slug or content hash."""


class ReviewExportError(OwnerCaptureError):
    """Raised when a post-review flip is requested without reviewed CVAT payload evidence."""


class CandidatePredictionStatusError(OwnerCaptureError):
    """Raised when prelabel code attempts to emit anything other than candidate predictions."""


@dataclass(frozen=True)
class OwnerCaptureVideoMetadata:
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video_metadata(path: str | Path) -> OwnerCaptureVideoMetadata:
    source = probe_clip(path)
    return OwnerCaptureVideoMetadata(
        width=source.width,
        height=source.height,
        fps=source.fps,
        duration_s=source.duration_s,
        frame_count=source.frame_count,
    )


def ingest_owner_capture(
    input_path: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_OWNER_DATA_MANIFEST,
    protected_video_shas: Mapping[str, str] | None = None,
    repo_root: str | Path = ".",
    account_id: str | None = None,
    profiles_root: str | Path = "runs/profiles",
) -> dict[str, Any]:
    """Register an owner capture package or bare video in the append-only manifest."""

    resolved = _resolve_owner_input(input_path)
    guard_protected_eval_reference(resolved["guard_values"], repo_root=repo_root, protected_video_shas=protected_video_shas)
    video_sha = sha256_file(resolved["video_path"])
    protected_shas = dict(protected_video_shas) if protected_video_shas is not None else collect_protected_eval_video_shas(repo_root)
    if video_sha in protected_shas:
        raise ProtectedEvalCaptureError(
            f"refusing owner-capture intake: video sha256 matches protected eval clip {protected_shas[video_sha]!r}"
        )

    sidecar = _load_json_optional(resolved["sidecar_path"])
    if sidecar is not None and account_id is not None:
        sidecar_with_profile_dist = inject_device_profile_distortion(
            sidecar,
            account_id=account_id,
            profiles_root=profiles_root,
        )
        if sidecar_with_profile_dist != sidecar:
            sidecar = sidecar_with_profile_dist
            if resolved["sidecar_path"] is not None:
                _write_json(resolved["sidecar_path"], sidecar)
    metadata = probe_video_metadata(resolved["video_path"])
    capture_id = _assign_capture_id(resolved, sidecar, video_sha)
    manifest_file = Path(manifest_path)
    manifest = load_owner_data_manifest(manifest_file)
    validate_owner_data_manifest(manifest)

    for existing in manifest["captures"]:
        if existing["sha256"] == video_sha:
            return {
                "status": "already_registered",
                "capture_id": existing["capture_id"],
                "manifest_path": str(manifest_file),
                "row": existing,
            }

    capture_id = _dedupe_capture_id(capture_id, manifest["captures"], video_sha)
    now = _utc_now()
    row = {
        "schema_version": 1,
        "capture_id": capture_id,
        "source": "owner_capture",
        "sha256": video_sha,
        "video_path": str(resolved["video_path"]),
        "package_path": str(resolved["package_path"]) if resolved["package_path"] else None,
        "sidecar_path": str(resolved["sidecar_path"]) if resolved["sidecar_path"] else None,
        "sidecar_provenance": _sidecar_provenance(sidecar),
        "camera_fingerprint": camera_fingerprint(metadata, sidecar),
        "video_metadata": metadata.to_dict(),
        "review_status": "unreviewed",
        "train_eligible": False,
        "registered_at_utc": now,
        "updated_at_utc": now,
    }
    manifest["captures"].append(row)
    _write_json(manifest_file, manifest)
    return {"status": "registered", "capture_id": capture_id, "manifest_path": str(manifest_file), "row": row}


def guard_protected_eval_reference(
    values: Sequence[Any],
    *,
    repo_root: str | Path = ".",
    protected_video_shas: Mapping[str, str] | None = None,
) -> None:
    """Refuse known protected eval slugs before any capture can enter owner data."""

    try:
        assert_not_training_on_eval_clip(values, allow_internal_val=False)
    except EvalClipLeakError as exc:
        raise ProtectedEvalCaptureError(str(exc)) from exc

    lowered_values = [str(value).lower() for value in values]
    for slug in PROTECTED_OWNER_EVAL_SLUGS:
        if any(slug in value for value in lowered_values):
            raise ProtectedEvalCaptureError(f"refusing owner-capture intake: protected eval slug {slug!r} was referenced")

    # Keep the content-hash registry warm for callers that want guard-only use.
    if protected_video_shas is None:
        collect_protected_eval_video_shas(repo_root)


def collect_protected_eval_video_shas(repo_root: str | Path = ".") -> dict[str, str]:
    root = Path(repo_root)
    candidates: list[tuple[str, Path]] = []
    for slug in PROTECTED_OWNER_EVAL_SLUGS:
        candidates.append((slug, root / "eval_clips" / "ball" / slug / "source.mp4"))
        candidates.extend((slug, path) for path in sorted((root / "cvat_upload").glob(f"*{slug}*.mp4")))

    shas: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for slug, path in candidates:
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        shas[sha256_file(path)] = slug
    return shas


def load_owner_data_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"schema_version": 1, "artifact_type": "racketsport_owner_data_manifest", "captures": []}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_owner_data_manifest(payload)
    return payload


def validate_owner_data_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("owner data manifest schema_version must be 1")
    if payload.get("artifact_type") != "racketsport_owner_data_manifest":
        raise ValueError("owner data manifest artifact_type must be racketsport_owner_data_manifest")
    captures = payload.get("captures")
    if not isinstance(captures, list):
        raise ValueError("owner data manifest captures must be a list")
    capture_ids: set[str] = set()
    shas: set[str] = set()
    for row in captures:
        if not isinstance(row, Mapping):
            raise ValueError("owner data manifest capture rows must be objects")
        for key in ("capture_id", "sha256", "source", "camera_fingerprint", "review_status", "train_eligible"):
            if key not in row:
                raise ValueError(f"owner data manifest row missing {key}")
        if row["capture_id"] in capture_ids:
            raise ValueError(f"duplicate owner capture_id {row['capture_id']!r}")
        if row["sha256"] in shas:
            raise ValueError(f"duplicate owner capture sha256 {row['sha256']!r}")
        capture_ids.add(str(row["capture_id"]))
        shas.add(str(row["sha256"]))
        if row["source"] != "owner_capture":
            raise ValueError("owner data manifest source must be owner_capture")
        if row["review_status"] not in {"unreviewed", "reviewed"}:
            raise ValueError("owner data manifest review_status must be unreviewed or reviewed")
        if not isinstance(row["train_eligible"], bool):
            raise ValueError("owner data manifest train_eligible must be boolean")


def camera_fingerprint(metadata: OwnerCaptureVideoMetadata, sidecar: Mapping[str, Any] | None) -> str:
    intrinsics_hash = _intrinsics_hash(sidecar)
    return f"{metadata.width}x{metadata.height}@{metadata.fps:.3f}:{intrinsics_hash}"


def inject_device_profile_distortion(
    sidecar: Mapping[str, Any],
    *,
    account_id: str,
    profiles_root: str | Path = "runs/profiles",
) -> dict[str, Any]:
    """Copy matched ChArUco lens/zoom distortion from a DeviceProfile into a sidecar."""

    updated = copy.deepcopy(dict(sidecar))
    intrinsics = updated.get("intrinsics")
    if not isinstance(intrinsics, dict):
        return updated

    device_key = _sidecar_string(updated, ("device_key", "device_profile_key", "deviceKey", "deviceProfileKey"))
    lens = _sidecar_string(updated, ("camera_lens", "lens", "cameraLens"))
    zoom = _sidecar_float(updated, ("camera_zoom", "zoom", "zoom_factor", "zoomFactor"))
    if device_key is None or lens is None or zoom is None:
        return updated

    try:
        from .profile_registry import load_profile_registry

        registry = load_profile_registry(account_id, profiles_root=profiles_root)
    except (FileNotFoundError, ValueError):
        return updated

    lens_normalized = lens.strip().lower()
    for profile in registry.device_profiles.values():
        if profile.device_key != device_key:
            continue
        for entry in profile.intrinsics_by_lens_zoom:
            if entry.lens.strip().lower() != lens_normalized:
                continue
            if not math.isclose(float(entry.zoom), zoom, rel_tol=1e-6, abs_tol=1e-6):
                continue
            intrinsics["dist"] = [float(value) for value in entry.intrinsics.dist]
            return updated
    return updated


def prelabel_owner_capture(
    capture_id: str,
    *,
    manifest_path: str | Path = DEFAULT_OWNER_DATA_MANIFEST,
    owner_data_root: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Write a candidate-only prelabel job spec and review manifest for a registered capture."""

    manifest_file = Path(manifest_path)
    manifest = load_owner_data_manifest(manifest_file)
    row = _find_capture(manifest, capture_id)
    if row["review_status"] != "unreviewed":
        raise OwnerCaptureError(f"capture {capture_id!r} is already reviewed; prelabels only apply to unreviewed captures")
    root = Path(owner_data_root) if owner_data_root is not None else manifest_file.parent
    prelabels_dir = root / capture_id / "prelabels"
    prelabels_dir.mkdir(parents=True, exist_ok=True)
    job_spec = build_prelabel_job_spec(row, prelabels_dir=prelabels_dir, dry_run=dry_run)
    enforce_candidate_prediction_status(job_spec, artifact_name="prelabel_job_spec.json")
    review_manifest = build_review_manifest(capture_id, [])
    _write_json(prelabels_dir / "prelabel_job_spec.json", job_spec)
    _write_json(prelabels_dir / "review_manifest.json", review_manifest)
    if not dry_run:
        raise OwnerCaptureError(
            "local prelabel execution is intentionally not run by this harness; execute the job_spec commands on the A100"
        )
    return {
        "status": "dry_run_ready",
        "capture_id": capture_id,
        "prelabels_dir": str(prelabels_dir),
        "job_spec_path": str(prelabels_dir / "prelabel_job_spec.json"),
        "review_manifest_path": str(prelabels_dir / "review_manifest.json"),
    }


def build_prelabel_job_spec(row: Mapping[str, Any], *, prelabels_dir: Path, dry_run: bool) -> dict[str, Any]:
    capture_id = str(row["capture_id"])
    video_path = str(row["video_path"])
    fps = float(row.get("video_metadata", {}).get("fps") or 30.0)
    person_out = prelabels_dir / "person_tracks.json"
    ball_out = prelabels_dir / "ball_track.json"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_owner_capture_prelabel_job_spec",
        "capture_id": capture_id,
        "status": "candidate_prediction",
        "dry_run": dry_run,
        "video_path": video_path,
        "expected_outputs": {
            "person_tracks": str(person_out),
            "ball_track": str(ball_out),
            "review_manifest": str(prelabels_dir / "review_manifest.json"),
        },
        "commands": {
            "person_tracks": (
                "python scripts/racketsport/run_offline_person_authority.py "
                f"--clip-id {capture_id} --candidate owner_prelabel --video {video_path} "
                "--source-run-dir <A100_TRACK_RUN_DIR_WITH_TRACKS_JSON> "
                f"--out-dir {prelabels_dir} --reid-model <A100_REID_MODEL_PATH>"
            ),
            "ball_track": (
                "python scripts/racketsport/run_wasb_ball.py "
                f"--video {video_path} --fps {fps:.6f} --out {ball_out} "
                f"--metadata-out {prelabels_dir / 'ball_track_metadata.json'} "
                "--checkpoint <A100_WASB_CHECKPOINT> --wasb-repo <A100_WASB_REPO> --device cuda"
            ),
            "cvat_export": (
                "python scripts/racketsport/export_cvat_tasks.py "
                f"--review-manifest {prelabels_dir / 'review_manifest.json'} "
                f"--out {prelabels_dir / 'cvat_task'}"
            ),
        },
        "notes": [
            "All outputs from these commands are candidate predictions until reviewed in CVAT.",
            "Do not use candidate_prediction artifacts as training rows.",
        ],
    }


def enforce_candidate_prediction_status(payload: Mapping[str, Any], *, artifact_name: str) -> None:
    if payload.get("status") != "candidate_prediction":
        raise CandidatePredictionStatusError(f"{artifact_name} must carry status='candidate_prediction'")


def build_review_manifest(capture_id: str, segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(segments, key=lambda segment: float(segment.get("confidence", 1.0)))
    manifest_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(ordered, start=1):
        item = dict(segment)
        item["review_priority"] = index
        item["reason"] = "lowest_confidence_first"
        manifest_segments.append(item)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_owner_capture_review_manifest",
        "capture_id": capture_id,
        "status": "candidate_prediction",
        "segments": manifest_segments,
    }


def apply_reviewed_cvat_export(
    capture_id: str,
    *,
    reviewed_export_path: str | Path,
    manifest_path: str | Path = DEFAULT_OWNER_DATA_MANIFEST,
    corpus_manifest_path: str | Path = DEFAULT_OWNER_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Flip one registered capture to train-eligible only from a reviewed CVAT payload."""

    export_path = Path(reviewed_export_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    _require_reviewed_cvat_payload(payload, capture_id=capture_id)

    manifest_file = Path(manifest_path)
    manifest = load_owner_data_manifest(manifest_file)
    row = _find_capture(manifest, capture_id)
    now = _utc_now()
    row["review_status"] = "reviewed"
    row["train_eligible"] = True
    row["reviewed_cvat_export_path"] = str(export_path)
    row["reviewed_at_utc"] = now
    row["updated_at_utc"] = now
    _write_json(manifest_file, manifest)

    corpus_file = Path(corpus_manifest_path)
    corpus_manifest = _load_owner_corpus_manifest(corpus_file)
    sample = {
        "schema_version": 1,
        "capture_id": capture_id,
        "source": "owner_capture",
        "sha256": row["sha256"],
        "video_path": row["video_path"],
        "reviewed_cvat_export_path": str(export_path),
        "camera_fingerprint": row["camera_fingerprint"],
        "review_status": "reviewed",
        "train_eligible": True,
        "materialized_at_utc": now,
    }
    corpus_manifest["samples"] = [existing for existing in corpus_manifest["samples"] if existing["capture_id"] != capture_id]
    corpus_manifest["samples"].append(sample)
    _write_json(corpus_file, corpus_manifest)
    return {
        "status": "reviewed_materialized",
        "capture_id": capture_id,
        "manifest_path": str(manifest_file),
        "corpus_manifest_path": str(corpus_file),
        "sample": sample,
    }


def _resolve_owner_input(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if path.is_dir():
        video_path = path / "clip.mov"
        if not video_path.is_file():
            raise FileNotFoundError(f"capture package is missing clip.mov: {path}")
        sidecar_path = path / "capture_sidecar.json"
        return {
            "input_path": path,
            "package_path": path,
            "video_path": video_path,
            "sidecar_path": sidecar_path if sidecar_path.is_file() else None,
            "guard_values": [path, video_path, sidecar_path],
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar_path = path.parent / "capture_sidecar.json"
    return {
        "input_path": path,
        "package_path": None,
        "video_path": path,
        "sidecar_path": sidecar_path if sidecar_path.is_file() else None,
        "guard_values": [path, sidecar_path],
    }


def _load_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar_provenance(sidecar: Mapping[str, Any] | None) -> str | None:
    if sidecar is None:
        return None
    provenance = sidecar.get("provenance")
    return str(provenance) if provenance is not None else None


def _intrinsics_hash(sidecar: Mapping[str, Any] | None) -> str:
    if not sidecar or "intrinsics" not in sidecar:
        return "no_intrinsics"
    canonical = json.dumps(sidecar["intrinsics"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def _sidecar_string(sidecar: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value = _sidecar_nested_value(sidecar, keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sidecar_float(sidecar: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    value = _sidecar_nested_value(sidecar, keys)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _sidecar_nested_value(sidecar: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in sidecar:
            return sidecar[key]
    for parent_key in ("metadata", "capture_metadata", "device", "camera"):
        child = sidecar.get(parent_key)
        if not isinstance(child, Mapping):
            continue
        for key in keys:
            if key in child:
                return child[key]
    return None


def _assign_capture_id(resolved: Mapping[str, Any], sidecar: Mapping[str, Any] | None, video_sha: str) -> str:
    if sidecar and isinstance(sidecar.get("capture_id"), str) and sidecar["capture_id"].strip():
        return _safe_id(sidecar["capture_id"])
    package_path = resolved.get("package_path")
    if isinstance(package_path, Path):
        return _safe_id(package_path.name)
    return f"owner_{Path(resolved['video_path']).stem}_{video_sha[:12]}"


def _safe_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    if not safe:
        raise OwnerCaptureError("capture_id cannot be empty")
    return safe


def _dedupe_capture_id(capture_id: str, rows: Sequence[Mapping[str, Any]], video_sha: str) -> str:
    existing_ids = {str(row["capture_id"]) for row in rows}
    if capture_id not in existing_ids:
        return capture_id
    return f"{capture_id}_{video_sha[:8]}"


def _find_capture(manifest: Mapping[str, Any], capture_id: str) -> dict[str, Any]:
    for row in manifest["captures"]:
        if row["capture_id"] == capture_id:
            return row
    raise KeyError(f"unknown owner capture_id {capture_id!r}")


def _require_reviewed_cvat_payload(payload: Mapping[str, Any], *, capture_id: str) -> None:
    status = payload.get("review_status", payload.get("status"))
    artifact_type = str(payload.get("artifact_type", ""))
    source_format = str(payload.get("source_format", ""))
    if "cvat" not in artifact_type.lower() and "cvat" not in source_format.lower():
        raise ReviewExportError("reviewed export payload must be a CVAT artifact")
    if status not in {None, "reviewed"}:
        raise ReviewExportError("post-review materialization requires a reviewed CVAT export payload")
    payload_capture_id = payload.get("clip_id") or payload.get("capture_id")
    if payload_capture_id is not None and payload_capture_id != capture_id:
        raise ReviewExportError(f"reviewed export clip_id {payload_capture_id!r} does not match capture_id {capture_id!r}")


def _load_owner_corpus_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_owner_capture_training_corpus_manifest",
            "samples": [],
            "policy": "Only reviewed owner-capture CVAT exports with train_eligible=true may appear here.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "racketsport_owner_capture_training_corpus_manifest":
        raise ValueError("owner corpus manifest artifact_type mismatch")
    if not isinstance(payload.get("samples"), list):
        raise ValueError("owner corpus manifest samples must be a list")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "CandidatePredictionStatusError",
    "DEFAULT_OWNER_CORPUS_MANIFEST",
    "DEFAULT_OWNER_DATA_MANIFEST",
    "DEFAULT_OWNER_DATA_ROOT",
    "OwnerCaptureError",
    "OwnerCaptureVideoMetadata",
    "PROTECTED_OWNER_EVAL_SLUGS",
    "ProtectedEvalCaptureError",
    "ReviewExportError",
    "apply_reviewed_cvat_export",
    "build_prelabel_job_spec",
    "build_review_manifest",
    "camera_fingerprint",
    "collect_protected_eval_video_shas",
    "enforce_candidate_prediction_status",
    "guard_protected_eval_reference",
    "inject_device_profile_distortion",
    "ingest_owner_capture",
    "load_owner_data_manifest",
    "prelabel_owner_capture",
    "probe_video_metadata",
    "sha256_file",
    "validate_owner_data_manifest",
]
