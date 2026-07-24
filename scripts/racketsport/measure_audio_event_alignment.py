#!/usr/bin/env python3
"""Measure audio-pop onset alignment to independent pb.vision event teachers.

This is a measurement-only CLI.  It reuses ``build_audio_onsets_v2.py`` with
its committed defaults, preserves the raw onset artifact, and maps both the
teacher and decoded-audio sample clocks onto media PTS before matching.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence
import unicodedata
from urllib.parse import unquote
import warnings

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.audio_onsets_v2 import (  # noqa: E402
    DEFAULT_ADAPTIVE_WINDOW_S,
    DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DEFAULT_FRAME_SIZE_S,
    DEFAULT_HOP_S,
    DEFAULT_MIN_HFC_EVIDENCE,
    DEFAULT_MIN_POP_BAND_RATIO,
    DEFAULT_MIN_SEPARATION_S,
    DEFAULT_MIN_SPECTRAL_EVIDENCE,
    DEFAULT_THRESHOLD_MAD,
    DETECTOR_VERSION,
    _adaptive_positive_z,
    _as_float_array,
    _bandpass,
    _frame_signal,
    _resample,
)


TOLERANCES_S = (0.033, 0.066, 0.100)
NULL_SHIFTS_S = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 17.0, 23.0, 31.0, 41.0, 53.0, 67.0, 79.0, 97.0, 113.0, 137.0)
RALLY_NULL_SEED = 20260722
RALLY_NULL_DRAWS = 500
FEATURES = (
    "onset_strength",
    "score",
    "spectral_flux",
    "high_frequency_content",
    "band_energy_delta",
    "pop_band_ratio",
)
FORBIDDEN_PBVISION_IDS = {"83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr"}
CANONICAL_DETECTOR_SOURCE = ROOT / "threed/racketsport/audio_onsets_v2.py"
AUTHORITATIVE_DATA_LEDGER = ROOT / "runs/manager/data_ledger.json"
REPAIR_BUNDLE = ROOT / "runs/lanes/ball_audio_repair2_20260722"
REPAIR_ALIGNMENT_REPORT = REPAIR_BUNDLE / "alignment_report_v3.json"
TRACKD_FINDINGS = ROOT / "runs/lanes/trackD_owner_queue_20260722/BATCH01_FINDINGS.md"
ONSET_BUILDER_SOURCE = ROOT / "scripts/racketsport/build_audio_onsets_v2.py"

# This lane registry is deliberately closed. Input paths are matched lexically
# before stat/read/probe/decode, so a caller cannot rename a protected source,
# supply a symlink alias, or pair forbidden bytes with a permitted clip ID.
REFERENCE_INPUT_REGISTRY: dict[str, dict[str, str]] = {
    "xkadsq9bli3h": {
        "video_path": "data/pbv_replay_20260720/xkadsq9bli3h/max.mp4",
        "cv_export_path": "data/pbvision_gallery_20260719/xkadsq9bli3h/cv_export.json",
        "insights_path": "data/pbvision_gallery_20260719/xkadsq9bli3h/insights.json",
        "ledger_asset": "pbv_replay_xkadsq9bli3h_20260720",
    },
}
VIABILITY_INPUT_REGISTRY: dict[str, dict[str, str]] = {
    clip_id: {
        "video_path": f"data/online_harvest_20260706/raw/{clip_id}.mp4",
        "ledger_asset": "online_harvest_20260706",
    }
    for clip_id in ("Ezz6HDNHlnk", "HyUqT7zFiwk", "73VurrTKCZ8", "wBu8bC4OfUY")
}
TT_INPUT_REGISTRY = {
    "labels_csv": "data/event_public_20260713/tt_sounds_data/full.csv",
    "snippets_dir": "data/event_public_20260713/tt_sounds_data/sounds_extracted/sounds",
    "anchor_report_path": "runs/lanes/ball_audio_repair2_20260722/alignment_report_v3.json",
    "trackd_findings_path": "runs/lanes/trackD_owner_queue_20260722/BATCH01_FINDINGS.md",
    "detector_source": "threed/racketsport/audio_onsets_v2.py",
    "raw_sound_candidates": (
        "data/event_public_20260713/tt_sounds_data/raw_sounds",
        "data/event_public_20260713/tt_sounds_data/sounds_extracted/raw_sounds",
    ),
}
TT_SURFACES = {"racket", "table", "floor", "other"}
TT_CORE_FEATURES = (
    "onset_strength",
    "high_frequency_content",
    "spectral_flux",
    "pop_band_ratio",
)


class RuntimeAccessObserver:
    """Record every explicit filesystem access made by this measurement CLI.

    Direct Python opens/stats/scans and subprocess probe/decode inputs are routed
    through the helpers below.  Decisions are derived only when the completed
    event stream is reconciled with the pre-access registry validation.
    """

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self._events: list[dict[str, Any]] = []

    @staticmethod
    def _display_path(path: Path | str) -> str:
        raw = os.fspath(path)
        absolute = os.path.abspath(
            raw if os.path.isabs(raw) else os.path.join(os.fspath(ROOT), raw)
        )
        root = os.path.abspath(os.fspath(ROOT))
        try:
            if os.path.commonpath((root, absolute)) == root:
                return Path(os.path.relpath(absolute, root)).as_posix()
        except ValueError:
            pass
        return Path(absolute).as_posix()

    def observe(
        self,
        path: Path | str,
        *,
        operation: str,
        via: str = "audited_python_wrapper",
    ) -> None:
        self._events.append({
            "sequence": len(self._events) + 1,
            "operation": operation,
            "path": self._display_path(path),
            "via": via,
        })

    def snapshot(
        self,
        *,
        pre_access_validation: Mapping[str, Any],
        output_roots: Sequence[Path | str],
    ) -> dict[str, Any]:
        validated = [
            dict(item)
            for item in pre_access_validation.get("validated_inputs") or []
            if isinstance(item, Mapping)
        ]
        directory_roles = {"tt_snippets"}
        normalized_outputs = [self._display_path(path) for path in output_roots]
        normalized_output_parents = {
            Path(path).parent.as_posix() for path in normalized_outputs
        }
        runtime_support = {
            self._display_path(AUTHORITATIVE_DATA_LEDGER): "authoritative_data_ledger",
            self._display_path(ONSET_BUILDER_SOURCE): "committed_onset_builder",
            self._display_path(ROOT): "git_repository_metadata_query",
        }
        write_operations = {
            "create_output_directory",
            "subprocess_write_output",
            "write_json",
        }

        observed: list[dict[str, Any]] = []
        forbidden: list[dict[str, Any]] = []
        for raw_event in self._events:
            event = dict(raw_event)
            path = str(event["path"])
            matched = next(
                (
                    item
                    for item in validated
                    if path == str(item.get("path"))
                    or (
                        str(item.get("role")) in directory_roles
                        and _path_is_within(path, str(item.get("path")))
                    )
                ),
                None,
            )
            output_root = next(
                (root for root in normalized_outputs if _path_is_within(path, root)),
                None,
            )
            is_output_parent_creation = (
                str(event["operation"]) == "create_output_directory"
                and path in normalized_output_parents
            )
            if output_root is not None or is_output_parent_creation:
                event.update({
                    "classification": "GENERATED_OUTPUT_OR_RELOAD",
                    "registry_role": "measurement_output",
                    "allowed": True,
                })
            elif matched is not None:
                event.update({
                    "classification": "VALIDATED_INPUT",
                    "registry_role": matched.get("role"),
                    "allowed": True,
                })
            elif path in runtime_support:
                event.update({
                    "classification": "RUNTIME_SUPPORT",
                    "registry_role": runtime_support[path],
                    "allowed": True,
                })
            elif str(event["operation"]) in write_operations:
                event.update({
                    "classification": "UNREGISTERED_OUTPUT",
                    "registry_role": None,
                    "allowed": False,
                })
            else:
                event.update({
                    "classification": "FORBIDDEN_UNREGISTERED_INPUT",
                    "registry_role": None,
                    "allowed": False,
                })
                forbidden.append(dict(event))
            observed.append(event)

        observed_encoded = json.dumps(
            observed, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            **dict(pre_access_validation),
            "audit_kind": "runtime_observed_filesystem_access",
            "observation_scope": (
                "all explicit data/support/output opens, stats, scans, hashes, probes, "
                "decodes, and writes made by this measurement command; child probe/decode "
                "paths are recorded at the audited subprocess invocation boundary"
            ),
            "observation_mechanism": (
                "audited Python filesystem wrappers plus audited ffprobe/git/onset-builder "
                "subprocess path boundaries"
            ),
            "observed_access_event_count": len(observed),
            "observed_unique_paths": sorted({str(item["path"]) for item in observed}),
            "observed_unique_path_count": len({str(item["path"]) for item in observed}),
            "observed_accesses_sha256": hashlib.sha256(observed_encoded).hexdigest(),
            "observed_accesses": observed,
            "forbidden_input_access_count": len(forbidden),
            "forbidden_input_accesses": forbidden,
            "all_observed_input_accesses_allowed": not forbidden,
        }


_ACTIVE_ACCESS_OBSERVER: RuntimeAccessObserver | None = None


@contextmanager
def _runtime_access_observation(observer: RuntimeAccessObserver):
    global _ACTIVE_ACCESS_OBSERVER
    previous = _ACTIVE_ACCESS_OBSERVER
    _ACTIVE_ACCESS_OBSERVER = observer
    try:
        yield observer
    finally:
        _ACTIVE_ACCESS_OBSERVER = previous


def _observe_access(
    path: Path | str, *, operation: str, via: str = "audited_python_wrapper"
) -> None:
    if _ACTIVE_ACCESS_OBSERVER is not None:
        _ACTIVE_ACCESS_OBSERVER.observe(path, operation=operation, via=via)


def _read_text(path: Path, *, encoding: str = "utf-8") -> str:
    _observe_access(path, operation="read_text")
    return path.read_text(encoding=encoding)


def _path_is_file(path: Path) -> bool:
    _observe_access(path, operation="stat_file")
    return path.is_file()


def _path_is_dir(path: Path) -> bool:
    _observe_access(path, operation="stat_directory")
    return path.is_dir()


def _glob_paths(path: Path, pattern: str) -> list[Path]:
    _observe_access(path, operation="scan_directory")
    return list(path.glob(pattern))


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(_read_text(path, encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _observe_access(path.parent, operation="create_output_directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    _observe_access(path, operation="write_json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _data_fences_from_observed_audit(
    access_audit: Mapping[str, Any], *, include_excluded: bool
) -> dict[str, Any]:
    output = {
        "derivation": "derived_from_runtime_observed_access_events",
        "scope": "measurement_command_only_not_process_wide_lane_activity",
        "all_declared_inputs_validated_before_access": bool(
            access_audit.get("all_declared_inputs_validated_before_access")
        ),
        "all_observed_input_accesses_allowed": bool(
            access_audit.get("all_observed_input_accesses_allowed")
        ),
        "observed_access_event_count": int(
            access_audit.get("observed_access_event_count", 0)
        ),
        "observed_unique_path_count": int(
            access_audit.get("observed_unique_path_count", 0)
        ),
        "observed_accesses_sha256": access_audit.get("observed_accesses_sha256"),
        "observed_access_list_verbatim_at": "access_audit.observed_accesses",
        "forbidden_input_access_count": int(
            access_audit.get("forbidden_input_access_count", -1)
        ),
        "validated_input_roles": [
            item.get("role") for item in access_audit.get("validated_inputs") or []
        ],
    }
    if include_excluded:
        output["excluded_without_access"] = list(
            access_audit.get("excluded_without_access") or []
        )
    return output


def _write_observed_report_json(
    path: Path,
    payload: dict[str, Any],
    *,
    observer: RuntimeAccessObserver,
    pre_access_validation: Mapping[str, Any],
    output_roots: Sequence[Path | str],
    include_excluded: bool,
) -> None:
    """Finalize the audit after recording the report write, then serialize once."""

    _observe_access(path.parent, operation="create_output_directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    _observe_access(path, operation="write_json")
    access_audit = observer.snapshot(
        pre_access_validation=pre_access_validation,
        output_roots=output_roots,
    )
    payload["access_audit"] = access_audit
    payload["data_fences"] = _data_fences_from_observed_audit(
        access_audit, include_excluded=include_excluded
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalise_guard_text(value: Any) -> str:
    text = str(value or "")
    for _ in range(4):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if character.isalnum()
    )


def _lexical_repo_path(path: Path | str) -> str:
    """Return a normalized repo-relative path without stat or symlink access."""

    raw = os.fspath(path)
    absolute = os.path.abspath(raw if os.path.isabs(raw) else os.path.join(os.fspath(ROOT), raw))
    root = os.path.abspath(os.fspath(ROOT))
    try:
        common = os.path.commonpath((root, absolute))
    except ValueError as exc:
        raise ValueError(f"input path is outside the repository registry: {raw}") from exc
    if common != root:
        raise ValueError(f"input path is outside the repository registry: {raw}")
    return Path(os.path.relpath(absolute, root)).as_posix()


def _path_is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _load_authoritative_lane_registry() -> dict[str, Any]:
    """Load and validate policy registry before touching caller-supplied inputs."""

    ledger = _read_object(AUTHORITATIVE_DATA_LEDGER)
    assets = ledger.get("assets")
    if not isinstance(assets, list):
        raise ValueError("authoritative data ledger must contain an assets array")
    by_id = {
        str(asset.get("asset_id")): asset
        for asset in assets
        if isinstance(asset, Mapping) and asset.get("asset_id")
    }
    required_assets = {
        "pbv_replay_xkadsq9bli3h_20260720",
        "pbvision_gallery_20260719",
        "online_harvest_20260706",
        "event_public_tt_sounds_20260713",
        "eval_clips_ball_protected_4",
        "protected_event_seed_50_20260713",
        "owner_event_labels_102_20260719",
    }
    missing = sorted(required_assets - set(by_id))
    if missing:
        raise ValueError(f"authoritative data ledger is missing lane assets: {missing}")

    required_registered_roots = {
        "pbv_replay_xkadsq9bli3h_20260720": "data/pbv_replay_20260720/xkadsq9bli3h/max.mp4",
        "pbvision_gallery_20260719": "data/pbvision_gallery_20260719",
        "online_harvest_20260706": "data/online_harvest_20260706",
        "event_public_tt_sounds_20260713": "data/event_public_20260713/tt_sounds_data",
        "owner_event_labels_102_20260719": "data/event_labels_owner_20260719",
    }
    for asset_id, expected_root in required_registered_roots.items():
        paths = by_id[asset_id].get("paths") or []
        registered = {
            str(item.get("path"))
            for item in paths
            if isinstance(item, Mapping) and item.get("path")
        }
        if not any(
            _path_is_within(expected_root, candidate)
            or _path_is_within(candidate, expected_root)
            for candidate in registered
        ):
            raise ValueError(
                f"authoritative data ledger asset {asset_id} does not register {expected_root}"
            )

    ledger_compare_ids = {
        str(identity.get("identity"))
        for identity in (
            by_id["pbvision_gallery_20260719"].get("protection", {}).get("identities")
            or []
        )
        if isinstance(identity, Mapping) and identity.get("posture") == "compare_only"
    }
    if ledger_compare_ids != FORBIDDEN_PBVISION_IDS:
        raise ValueError(
            "authoritative compare-only registry drift: "
            f"expected={sorted(FORBIDDEN_PBVISION_IDS)} actual={sorted(ledger_compare_ids)}"
        )

    forbidden_roots = {
        "data/pbvision_11min_20260713",
        "data/event_labels_owner_20260719",
        "runs/lanes/ball_audio_anchor_20260722",
        "runs/lanes/ball_audio_ttcal_20260722",
    }
    for asset_id in ("eval_clips_ball_protected_4", "protected_event_seed_50_20260713"):
        for item in by_id[asset_id].get("paths") or []:
            if isinstance(item, Mapping) and item.get("path"):
                forbidden_roots.add(str(item["path"]))

    return {
        "ledger_path": _lexical_repo_path(AUTHORITATIVE_DATA_LEDGER),
        "ledger_sha256": _sha256(AUTHORITATIVE_DATA_LEDGER),
        "ledger_schema_version": ledger.get("schema_version"),
        "required_assets": sorted(required_assets),
        "compare_only_ids": sorted(ledger_compare_ids),
        "forbidden_roots": sorted(forbidden_roots),
    }


def _reject_forbidden_identity_or_path(
    *, clip_id: str | None, path: Path | str, registry: Mapping[str, Any]
) -> str:
    normalized_id = _normalise_guard_text(clip_id)
    forbidden_tokens = {
        _normalise_guard_text(item) for item in registry["compare_only_ids"]
    }
    if normalized_id and normalized_id in forbidden_tokens:
        raise ValueError(
            f"registry refusal before input access: compare-only identity {clip_id}"
        )
    lexical = _lexical_repo_path(path)
    normalized_path = _normalise_guard_text(lexical)
    if any(token and token in normalized_path for token in forbidden_tokens):
        raise ValueError(
            f"registry refusal before input access: compare-only path provenance {lexical}"
        )
    for forbidden_root in registry["forbidden_roots"]:
        if _path_is_within(lexical, str(forbidden_root)):
            raise ValueError(
                f"registry refusal before input access: protected/owner path {lexical}"
            )
    return lexical


def _validate_registered_path(
    *,
    clip_id: str | None,
    path: Path | str,
    expected: str,
    role: str,
    registry: Mapping[str, Any],
    validated_inputs: list[dict[str, Any]],
) -> None:
    lexical = _reject_forbidden_identity_or_path(
        clip_id=clip_id, path=path, registry=registry
    )
    if lexical != expected:
        raise ValueError(
            f"registry refusal before input access: unregistered {role} path {lexical}; "
            f"expected {expected}"
        )
    validated_inputs.append({
        "role": role,
        "clip_id": clip_id,
        "path": lexical,
        "decision": "ALLOW_EXACT_REGISTERED_PATH",
    })


def _validate_alignment_inputs_before_access(
    references: Sequence[Mapping[str, Any]],
    viability: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    detector_source: Path,
) -> dict[str, Any]:
    registry = _load_authoritative_lane_registry()
    validated_inputs: list[dict[str, Any]] = []
    measured_ids: set[str] = set()
    for spec in references:
        offered_id = str(spec.get("clip_id") or "")
        if _normalise_guard_text(offered_id) in {
            _normalise_guard_text(item) for item in registry["compare_only_ids"]
        }:
            raise ValueError(
                f"registry refusal before input access: compare-only identity {offered_id}"
            )
        if offered_id not in REFERENCE_INPUT_REGISTRY:
            raise ValueError(
                f"registry refusal before input access: unregistered reference identity {offered_id}"
            )
        expected = REFERENCE_INPUT_REGISTRY[offered_id]
        measured_ids.add(offered_id)
        for key, role in (
            ("video_path", "reference_video"),
            ("cv_export_path", "reference_teacher_export"),
            ("insights_path", "reference_teacher_insights"),
        ):
            _validate_registered_path(
                clip_id=offered_id,
                path=spec[key],
                expected=expected[key],
                role=role,
                registry=registry,
                validated_inputs=validated_inputs,
            )

    for spec in viability:
        offered_id = str(spec.get("clip_id") or "")
        if _normalise_guard_text(offered_id) in {
            _normalise_guard_text(item) for item in registry["compare_only_ids"]
        }:
            raise ValueError(
                f"registry refusal before input access: compare-only identity {offered_id}"
            )
        if offered_id not in VIABILITY_INPUT_REGISTRY:
            raise ValueError(
                f"registry refusal before input access: unregistered viability identity {offered_id}"
            )
        _validate_registered_path(
            clip_id=offered_id,
            path=spec["video_path"],
            expected=VIABILITY_INPUT_REGISTRY[offered_id]["video_path"],
            role="viability_audio_stream",
            registry=registry,
            validated_inputs=validated_inputs,
        )

    excluded_rows: list[dict[str, Any]] = []
    for item in excluded:
        clip_id = str(item.get("clip_id") or "")
        external_id = str(item.get("external_id") or "")
        if external_id not in FORBIDDEN_PBVISION_IDS:
            raise ValueError(
                f"excluded reference is not in the compare-only registry: {external_id}"
            )
        if clip_id in measured_ids or external_id in measured_ids:
            raise ValueError(
                f"reference is simultaneously measured and excluded: {clip_id or external_id}"
            )
        excluded_rows.append({
            "clip_id": clip_id,
            "external_id": external_id,
            "decision": "DENY_WITHOUT_PATH_CONSTRUCTION_OR_ACCESS",
        })

    _validate_registered_path(
        clip_id=None,
        path=detector_source,
        expected=TT_INPUT_REGISTRY["detector_source"],
        role="canonical_detector_source",
        registry=registry,
        validated_inputs=validated_inputs,
    )

    return {
        "mode": "alignment",
        "pre_access_validation_kind": "closed_registry_and_provenance",
        "validation_order": "registry_and_provenance_before_any_input_stat_read_probe_parse_or_decode",
        "all_declared_inputs_validated_before_access": True,
        "authoritative_registry": {
            key: registry[key]
            for key in ("ledger_path", "ledger_sha256", "ledger_schema_version", "required_assets")
        },
        "validated_inputs": validated_inputs,
        "excluded_without_access": excluded_rows,
    }


def _validate_tt_inputs_before_access(
    *,
    labels_csv: Path,
    snippets_dir: Path,
    anchor_report_path: Path,
    trackd_findings_path: Path,
    detector_source: Path,
) -> dict[str, Any]:
    registry = _load_authoritative_lane_registry()
    validated_inputs: list[dict[str, Any]] = []
    for role, path, expected in (
        ("tt_labels", labels_csv, TT_INPUT_REGISTRY["labels_csv"]),
        ("tt_snippets", snippets_dir, TT_INPUT_REGISTRY["snippets_dir"]),
        ("clean_anchor_report", anchor_report_path, TT_INPUT_REGISTRY["anchor_report_path"]),
        ("trackd_findings", trackd_findings_path, TT_INPUT_REGISTRY["trackd_findings_path"]),
        ("canonical_detector_source", detector_source, TT_INPUT_REGISTRY["detector_source"]),
    ):
        _validate_registered_path(
            clip_id=None,
            path=path,
            expected=expected,
            role=role,
            registry=registry,
            validated_inputs=validated_inputs,
        )
    for expected in TT_INPUT_REGISTRY["raw_sound_candidates"]:
        _validate_registered_path(
            clip_id=None,
            path=expected,
            expected=expected,
            role="tt_continuous_raw_sound_candidate",
            registry=registry,
            validated_inputs=validated_inputs,
        )
    return {
        "mode": "tt_snippet_diagnostics",
        "pre_access_validation_kind": "closed_registry_and_provenance",
        "validation_order": "registry_and_provenance_before_any_input_stat_read_probe_parse_or_decode",
        "all_declared_inputs_validated_before_access": True,
        "authoritative_registry": {
            key: registry[key]
            for key in ("ledger_path", "ledger_sha256", "ledger_schema_version", "required_assets")
        },
        "validated_inputs": validated_inputs,
        "excluded_without_access": [],
    }


def _validate_output_destination(path: Path, *, role: str) -> None:
    raw = os.fspath(path)
    absolute = os.path.abspath(raw if os.path.isabs(raw) else os.path.join(os.fspath(ROOT), raw))
    root = os.path.abspath(os.fspath(ROOT))
    try:
        common = os.path.commonpath((root, absolute))
    except ValueError:
        return
    if common != root:
        return
    lexical = Path(os.path.relpath(absolute, root)).as_posix()
    immutable_roots = (
        "runs/lanes/ball_audio_anchor_20260722",
        "runs/lanes/ball_audio_ttcal_20260722",
    )
    if any(_path_is_within(lexical, root) for root in immutable_roots):
        raise ValueError(f"refusing {role} inside immutable prior audio lane: {lexical}")


def _sha256(path: Path) -> str:
    _observe_access(path, operation="sha256_read")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_json(
    command: Sequence[str], *, observed_path: Path, operation: str
) -> dict[str, Any]:
    _observe_access(
        observed_path,
        operation=operation,
        via="audited_subprocess_invocation_boundary",
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"command did not emit a JSON object: {shlex.join(command)}")
    return payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _ratio(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    return _optional_float(text)


def probe_media_pts(path: Path, *, audio_only: bool) -> dict[str, Any]:
    if not _path_is_file(path):
        raise FileNotFoundError(path)
    audio_stream_command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries",
        "stream=index,codec_name,codec_type,sample_rate,channels,time_base,start_pts,start_time,duration_ts,duration,nb_frames",
        "-of", "json", str(path),
    ]
    audio_packet_command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_packets", "-show_entries",
        "packet=pts,pts_time,dts_time,duration_time,flags,side_data_list",
        "-read_intervals", "%+#4", "-of", "json", str(path),
    ]
    format_command = [
        "ffprobe", "-v", "error", "-show_entries", "format=start_time,duration",
        "-of", "json", str(path),
    ]
    audio_stream_payload = _run_json(
        audio_stream_command, observed_path=path, operation="ffprobe_audio_stream"
    )
    audio_packet_payload = _run_json(
        audio_packet_command, observed_path=path, operation="ffprobe_audio_packets"
    )
    format_payload = _run_json(
        format_command, observed_path=path, operation="ffprobe_media_format"
    )
    audio_streams = audio_stream_payload.get("streams") or []
    if not audio_streams:
        return {
            "status": "NO_AUDIO_STREAM",
            "commands": [shlex.join(audio_stream_command), shlex.join(audio_packet_command), shlex.join(format_command)],
        }
    audio_stream = dict(audio_streams[0])
    audio_origin = _effective_audio_origin(audio_stream, audio_packet_payload.get("packets") or [])
    output: dict[str, Any] = {
        "status": "PTS_PROBED",
        "probe_scope": "audio_stream_only_no_video_frame_decode" if audio_only else "audio_and_video_packet_pts_no_frame_decode",
        "commands": [shlex.join(audio_stream_command), shlex.join(audio_packet_command), shlex.join(format_command)],
        "format": dict(format_payload.get("format") or {}),
        "audio_stream": audio_stream,
        "audio_first_packets": audio_packet_payload.get("packets") or [],
        "audio_effective_origin_pts_s": audio_origin,
    }
    if audio_only:
        return output

    video_stream_command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=index,codec_name,codec_type,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration_ts,duration,nb_frames",
        "-of", "json", str(path),
    ]
    video_packet_command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_packets", "-show_entries",
        "packet=pts,pts_time,dts_time,duration_time,flags",
        "-read_intervals", "%+#4", "-of", "json", str(path),
    ]
    video_stream_payload = _run_json(
        video_stream_command, observed_path=path, operation="ffprobe_video_stream"
    )
    video_packet_payload = _run_json(
        video_packet_command, observed_path=path, operation="ffprobe_video_packets"
    )
    video_streams = video_stream_payload.get("streams") or []
    if not video_streams:
        output["status"] = "NO_VIDEO_STREAM"
        return output
    video_stream = dict(video_streams[0])
    video_pts_values = [
        value for value in (_optional_float(item.get("pts_time")) for item in video_packet_payload.get("packets") or [])
        if value is not None
    ]
    video_origin = _optional_float(video_stream.get("start_time"))
    if video_origin is None and video_pts_values:
        video_origin = min(video_pts_values)
    output.update({
        "commands": output["commands"] + [shlex.join(video_stream_command), shlex.join(video_packet_command)],
        "video_stream": video_stream,
        "video_first_packets": video_packet_payload.get("packets") or [],
        "video_effective_origin_pts_s": video_origin,
        "media_fps": _ratio(video_stream.get("avg_frame_rate")) or _ratio(video_stream.get("r_frame_rate")),
    })
    return output


def _effective_audio_origin(stream: Mapping[str, Any], packets: Sequence[Mapping[str, Any]]) -> float | None:
    sample_rate = _optional_float(stream.get("sample_rate"))
    for packet in packets:
        packet_pts = _optional_float(packet.get("pts_time"))
        if packet_pts is None:
            continue
        skip_samples = 0.0
        for side_data in packet.get("side_data_list") or []:
            if side_data.get("side_data_type") == "Skip Samples":
                skip_samples = _optional_float(side_data.get("skip_samples")) or 0.0
                break
        if skip_samples and sample_rate:
            effective = packet_pts + skip_samples / sample_rate
            # ffprobe renders pts_time to six decimals; snap sub-sample residue.
            return 0.0 if abs(effective) <= 0.5 / sample_rate else effective
        return packet_pts
    return _optional_float(stream.get("start_time"))


def extract_pbvision_events(cv_export: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    camera = cv_export.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("cv_export.camera must be an object")
    export_fps = _optional_float(camera.get("fps"))
    if export_fps is None or export_fps <= 0:
        raise ValueError("cv_export.camera.fps must be positive")
    events: list[dict[str, Any]] = []
    sessions = cv_export.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("cv_export.sessions must be an array")
    for session_index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            continue
        for rally_index, rally in enumerate(session.get("rallies") or []):
            if not isinstance(rally, Mapping):
                continue
            first_frame = rally.get("frame_index")
            if not isinstance(first_frame, int):
                raise ValueError("cv_export rally.frame_index must be an integer")
            frames = rally.get("frames") or []
            if not isinstance(frames, list):
                raise ValueError("cv_export rally.frames must be an array")
            for offset, frame in enumerate(frames):
                if not isinstance(frame, Mapping):
                    continue
                balls = frame.get("balls")
                if not isinstance(balls, Mapping):
                    continue
                selected = balls.get("selected")
                if selected not in {"shot", "bounce"} or not isinstance(balls.get(selected), Mapping):
                    continue
                absolute_frame = first_frame + offset
                detail = balls[selected]
                actions = frame.get("actions") if isinstance(frame.get("actions"), Mapping) else {}
                action = actions.get(selected) if isinstance(actions.get(selected), Mapping) else {}
                events.append({
                    "event_index": len(events),
                    "event_type": "hit" if selected == "shot" else "bounce",
                    "pbvision_selected": selected,
                    "frame_index_export": absolute_frame,
                    "export_time_s": absolute_frame / export_fps,
                    "session_index": session_index,
                    "rally_index": rally_index,
                    "confidence": _optional_float(action.get("confidence")),
                    "interpolated": bool(detail.get("interpolated", False)),
                    "out_of_sequence": bool(detail.get("out_of_sequence", False)),
                })
    events.sort(key=lambda item: (float(item["export_time_s"]), str(item["event_type"])))
    for event_index, event in enumerate(events):
        event["event_index"] = event_index
    return events, export_fps


def extract_pbvision_rally_intervals(
    cv_export: Mapping[str, Any], *, export_fps: float
) -> list[dict[str, Any]]:
    """Extract half-open rally intervals on the pb.vision export clock."""

    intervals: list[dict[str, Any]] = []
    sessions = cv_export.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("cv_export.sessions must be an array")
    for session_index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            continue
        for rally_index, rally in enumerate(session.get("rallies") or []):
            if not isinstance(rally, Mapping):
                continue
            first_frame = rally.get("frame_index")
            frames = rally.get("frames")
            if not isinstance(first_frame, int) or not isinstance(frames, list):
                raise ValueError("cv_export rally requires integer frame_index and frames array")
            start_s = first_frame / export_fps
            end_s = (first_frame + len(frames)) / export_fps
            if end_s <= start_s:
                raise ValueError("cv_export rally interval must have positive duration")
            intervals.append({
                "session_index": session_index,
                "rally_index": rally_index,
                "start_export_s": start_s,
                "end_export_s": end_s,
                "duration_s": end_s - start_s,
                "frame_count": len(frames),
            })
    intervals.sort(key=lambda item: (float(item["start_export_s"]), float(item["end_export_s"])))
    for previous, current in zip(intervals, intervals[1:]):
        if float(current["start_export_s"]) < float(previous["end_export_s"]):
            raise ValueError("cv_export rally intervals overlap; conditioned null is ambiguous")
    return intervals


def crosscheck_insights_timebase(
    cv_export: Mapping[str, Any],
    insights: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    export_fps: float,
) -> dict[str, Any]:
    insight_rallies = insights.get("rallies") or []
    cv_sessions = cv_export.get("sessions") or []
    cv_rallies = cv_sessions[0].get("rallies") if cv_sessions and isinstance(cv_sessions[0], Mapping) else []
    first_rally_delta_s: float | None = None
    if insight_rallies and cv_rallies:
        insight_start = _optional_float(insight_rallies[0].get("start_ms"))
        cv_start = _optional_float(cv_rallies[0].get("frame_index"))
        if insight_start is not None and cv_start is not None:
            first_rally_delta_s = cv_start / export_fps - insight_start / 1000.0

    insight_shot_times = [
        float(shot["start_ms"]) / 1000.0
        for rally in insight_rallies if isinstance(rally, Mapping)
        for shot in rally.get("shots") or []
        if isinstance(shot, Mapping) and _optional_float(shot.get("start_ms")) is not None
    ]
    selected_shot_times = [float(event["export_time_s"]) for event in events if event["event_type"] == "hit"]
    tolerance_s = max(0.0011, 0.51 / export_fps)
    matched = sum(
        1 for time_s in insight_shot_times
        if selected_shot_times and min(abs(time_s - candidate) for candidate in selected_shot_times) <= tolerance_s
    )
    return {
        "method": "insights shot.start_ms and rally.start_ms compared to cv_export selected-shot frame/export_fps",
        "insights_shot_count": len(insight_shot_times),
        "insights_shots_matching_selected_frames": matched,
        "insights_shot_match_fraction": matched / len(insight_shot_times) if insight_shot_times else None,
        "crosscheck_tolerance_s": tolerance_s,
        "first_rally_cv_minus_insights_s": first_rally_delta_s,
    }


def derive_reference_mapping(
    probe: Mapping[str, Any],
    *,
    export_fps: float,
    events: Sequence[Mapping[str, Any]],
    insights_crosscheck: Mapping[str, Any],
) -> dict[str, Any]:
    audio_origin = _optional_float(probe.get("audio_effective_origin_pts_s"))
    video_origin = _optional_float(probe.get("video_effective_origin_pts_s"))
    media_fps = _optional_float(probe.get("media_fps"))
    video_stream = probe.get("video_stream") if isinstance(probe.get("video_stream"), Mapping) else {}
    duration_s = _optional_float(video_stream.get("duration"))
    blockers: list[str] = []
    if audio_origin is None:
        blockers.append("audio_effective_origin_pts_unavailable")
    if video_origin is None:
        blockers.append("video_effective_origin_pts_unavailable")
    if media_fps is None or media_fps <= 0:
        blockers.append("media_fps_unavailable")
    match_fraction = _optional_float(insights_crosscheck.get("insights_shot_match_fraction"))
    if match_fraction is None or match_fraction < 0.95:
        blockers.append("insights_to_selected_event_timebase_crosscheck_below_0.95")
    first_rally_delta = _optional_float(insights_crosscheck.get("first_rally_cv_minus_insights_s"))
    if first_rally_delta is None or abs(first_rally_delta) > max(0.002, 0.51 / export_fps):
        blockers.append("rally_start_timebase_crosscheck_failed")
    if duration_s is not None and events:
        last_export_time = max(float(item["export_time_s"]) for item in events)
        if last_export_time > duration_s + 1.0 / export_fps:
            blockers.append("export_event_exceeds_video_duration")
    return {
        "status": "AUDIO_REFERENCE_UNALIGNABLE" if blockers else "ALIGNED",
        "blockers": blockers,
        "mapping": {
            "reference_media_time_s": "video_effective_origin_pts_s + frame_index_export / cv_export.camera.fps",
            "audio_media_time_s": "audio_effective_origin_pts_s + detector_raw_time_s",
            "video_effective_origin_pts_s": video_origin,
            "audio_effective_origin_pts_s": audio_origin,
            "cv_export_fps": export_fps,
            "media_fps": media_fps,
            "acoustic_propagation_correction": "not_applied_distance_unknown",
        },
        "insights_crosscheck": dict(insights_crosscheck),
    }


def greedy_nearest_one_to_one(
    reference_times_s: Sequence[float], onset_times_s: Sequence[float], tolerance_s: float
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, float, float, int, int]] = []
    for ref_index, ref_time in enumerate(reference_times_s):
        for onset_index, onset_time in enumerate(onset_times_s):
            absolute_offset = abs(onset_time - ref_time)
            if absolute_offset <= tolerance_s + 1e-12:
                candidates.append((absolute_offset, ref_time, onset_time, ref_index, onset_index))
    candidates.sort()
    used_refs: set[int] = set()
    used_onsets: set[int] = set()
    matched: list[dict[str, Any]] = []
    for absolute_offset, ref_time, onset_time, ref_index, onset_index in candidates:
        if ref_index in used_refs or onset_index in used_onsets:
            continue
        used_refs.add(ref_index)
        used_onsets.add(onset_index)
        matched.append({
            "reference_index": ref_index,
            "onset_index": onset_index,
            "reference_time_s": ref_time,
            "onset_time_s": onset_time,
            "offset_ms": (onset_time - ref_time) * 1000.0,
            "absolute_offset_ms": absolute_offset * 1000.0,
        })
    matched.sort(key=lambda item: int(item["reference_index"]))
    return matched


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), quantile, method="linear"))


def measure_tolerance(
    reference_times_s: Sequence[float],
    onset_times_s: Sequence[float],
    *,
    tolerance_s: float,
    media_fps: float,
    duration_s: float,
) -> dict[str, Any]:
    matched = greedy_nearest_one_to_one(reference_times_s, onset_times_s, tolerance_s)
    offsets = [float(item["absolute_offset_ms"]) for item in matched]
    signed_offsets = [float(item["offset_ms"]) for item in matched]
    unmatched_onsets = len(onset_times_s) - len(matched)
    duration_minutes = duration_s / 60.0
    return {
        "tolerance_ms": tolerance_s * 1000.0,
        "tolerance_frames_at_media_fps": tolerance_s * media_fps,
        "reference_event_count": len(reference_times_s),
        "onset_count": len(onset_times_s),
        "matched_count": len(matched),
        "unmatched_reference_count": len(reference_times_s) - len(matched),
        "unmatched_onset_count": unmatched_onsets,
        "recall": len(matched) / len(reference_times_s) if reference_times_s else None,
        "precision_proxy": len(matched) / len(onset_times_s) if onset_times_s else None,
        "median_absolute_offset_ms": _percentile(offsets, 50),
        "p90_absolute_offset_ms": _percentile(offsets, 90),
        "median_signed_offset_ms_onset_minus_reference": _percentile(signed_offsets, 50),
        "p10_signed_offset_ms_onset_minus_reference": _percentile(signed_offsets, 10),
        "p90_signed_offset_ms_onset_minus_reference": _percentile(signed_offsets, 90),
        "unmatched_onset_rate_per_min": unmatched_onsets / duration_minutes if duration_minutes > 0 else None,
        "matched_pairs": matched,
    }


def measure_circular_shift_null(
    reference_times_s: Sequence[float],
    onset_times_s: Sequence[float],
    *,
    tolerance_s: float,
    duration_s: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shift_s in NULL_SHIFTS_S:
        shifted = sorted((time_s + shift_s) % duration_s for time_s in onset_times_s)
        matched_count = len(greedy_nearest_one_to_one(reference_times_s, shifted, tolerance_s))
        rows.append({
            "shift_s": shift_s,
            "matched_count": matched_count,
            "recall": matched_count / len(reference_times_s) if reference_times_s else None,
            "precision_proxy": matched_count / len(onset_times_s) if onset_times_s else None,
        })
    recalls = [float(row["recall"]) for row in rows if row["recall"] is not None]
    precisions = [float(row["precision_proxy"]) for row in rows if row["precision_proxy"] is not None]
    return {
        "method": (
            "activity-confounded weak null: fixed circular shifts of the full onset sequence; "
            "spacing/count are preserved but rally occupancy is not"
        ),
        "interpretation": (
            "Whole-clip shifts can over-credit local alignment because teachers and detections "
            "both concentrate during play; this null is not sufficient evidence by itself."
        ),
        "shift_count": len(rows),
        "shifts_s": list(NULL_SHIFTS_S),
        "median_recall": _percentile(recalls, 50),
        "p90_recall": _percentile(recalls, 90),
        "median_precision_proxy": _percentile(precisions, 50),
        "p90_precision_proxy": _percentile(precisions, 90),
        "rows": rows,
    }


def _times_inside_intervals(
    times_s: Sequence[float], intervals_s: Sequence[tuple[float, float]]
) -> list[float]:
    return [
        float(time_s)
        for time_s in times_s
        if any(start_s <= float(time_s) < end_s for start_s, end_s in intervals_s)
    ]


def measure_rally_conditioned_null(
    reference_times_s: Sequence[float],
    onset_times_s: Sequence[float],
    *,
    rally_intervals_s: Sequence[tuple[float, float]],
    tolerance_s: float,
    seed: int = RALLY_NULL_SEED,
    draws: int = RALLY_NULL_DRAWS,
) -> dict[str, Any]:
    """Shift onset sequences independently inside every rally interval."""

    if draws <= 0:
        raise ValueError("rally-conditioned null draws must be positive")
    intervals = sorted((float(start), float(end)) for start, end in rally_intervals_s)
    if not intervals or any(end <= start for start, end in intervals):
        raise ValueError("rally-conditioned null requires positive rally intervals")
    if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
        raise ValueError("rally-conditioned null intervals must not overlap")

    rally_references = _times_inside_intervals(reference_times_s, intervals)
    onsets_by_interval = [
        [float(time_s) for time_s in onset_times_s if start <= float(time_s) < end]
        for start, end in intervals
    ]
    rally_onsets = sorted(time_s for group in onsets_by_interval for time_s in group)
    actual_matched_count = len(
        greedy_nearest_one_to_one(rally_references, rally_onsets, tolerance_s)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for draw_index in range(draws):
        shifted: list[float] = []
        for (start_s, end_s), interval_onsets in zip(intervals, onsets_by_interval):
            duration_s = end_s - start_s
            shift_s = float(rng.uniform(0.0, duration_s))
            shifted.extend(
                start_s + ((time_s - start_s + shift_s) % duration_s)
                for time_s in interval_onsets
            )
        matched_count = len(
            greedy_nearest_one_to_one(
                rally_references, sorted(shifted), tolerance_s
            )
        )
        rows.append({
            "draw_index": draw_index,
            "matched_count": matched_count,
            "recall": (
                matched_count / len(rally_references) if rally_references else None
            ),
            "precision_proxy": (
                matched_count / len(rally_onsets) if rally_onsets else None
            ),
        })
    matched_counts = [int(row["matched_count"]) for row in rows]
    exceedance_count = sum(value >= actual_matched_count for value in matched_counts)
    return {
        "method": (
            "seeded independent circular shift of the observed onset sequence within each "
            "half-open pb.vision rally interval; per-rally occupancy/count are preserved"
        ),
        "seed": seed,
        "draw_count": draws,
        "rally_interval_count": len(intervals),
        "rally_duration_s": sum(end - start for start, end in intervals),
        "reference_event_count_in_rallies": len(rally_references),
        "onset_count_in_rallies": len(rally_onsets),
        "actual_matched_count": actual_matched_count,
        "actual_recall": (
            actual_matched_count / len(rally_references) if rally_references else None
        ),
        "actual_precision_proxy": (
            actual_matched_count / len(rally_onsets) if rally_onsets else None
        ),
        "null_median_matched_count": _percentile(matched_counts, 50),
        "null_p90_matched_count": _percentile(matched_counts, 90),
        "null_max_matched_count": max(matched_counts) if matched_counts else None,
        "draws_reaching_or_exceeding_actual": exceedance_count,
        "empirical_exceedance_fraction": exceedance_count / draws,
        "plus_one_empirical_p": (exceedance_count + 1) / (draws + 1),
        "rows": rows,
    }


def _feature_value(onset: Mapping[str, Any], feature: str) -> float | None:
    if feature in {"onset_strength", "score"}:
        return _optional_float(onset.get(feature))
    nested = onset.get("features") if isinstance(onset.get("features"), Mapping) else {}
    return _optional_float(nested.get(feature))


def summarize_features(onsets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature in FEATURES:
        values = [value for value in (_feature_value(item, feature) for item in onsets) if value is not None]
        output[feature] = {
            "n": len(values),
            "min": min(values) if values else None,
            "p10": _percentile(values, 10),
            "median": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "max": max(values) if values else None,
        }
    return output


def compare_features_to_reference(
    onsets: Sequence[Mapping[str, Any]], reference_summary: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature in FEATURES:
        values = [value for value in (_feature_value(item, feature) for item in onsets) if value is not None]
        baseline = reference_summary.get(feature) if isinstance(reference_summary.get(feature), Mapping) else {}
        baseline_median = _optional_float(baseline.get("median"))
        lower = _optional_float(baseline.get("p10"))
        upper = _optional_float(baseline.get("p90"))
        median = _percentile(values, 50)
        in_reference_band = [value for value in values if lower is not None and upper is not None and lower <= value <= upper]
        output[feature] = {
            "n": len(values),
            "median_ratio_to_reference_matched": (
                median / baseline_median if median is not None and baseline_median not in {None, 0.0} else None
            ),
            "fraction_inside_reference_matched_p10_p90": len(in_reference_band) / len(values) if values else None,
            "reference_matched_p10_p90": [lower, upper],
        }
    return output


def _build_pts_identity(path: Path, *, media_sha256: str, probe: Mapping[str, Any], clip_id: str) -> None:
    _write_json(path, {
        "schema_version": 1,
        "artifact_type": "audio_alignment_pts_identity",
        "clip_id": clip_id,
        "source_video_sha256": media_sha256,
        "media_sha256": media_sha256,
        "probe_scope": probe.get("probe_scope"),
        "audio_effective_origin_pts_s": probe.get("audio_effective_origin_pts_s"),
        "video_effective_origin_pts_s": probe.get("video_effective_origin_pts_s"),
        "commands": probe.get("commands"),
    })


def run_committed_onset_pipeline(
    *, clip_id: str, video_path: Path, probe: Mapping[str, Any], raw_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    media_sha256 = _sha256(video_path)
    pts_path = raw_dir / f"{clip_id}.pts_identity.json"
    onset_path = raw_dir / f"{clip_id}.audio_onsets_v2.json"
    _build_pts_identity(pts_path, media_sha256=media_sha256, probe=probe, clip_id=clip_id)
    command = [
        sys.executable,
        str(ROOT / "scripts/racketsport/build_audio_onsets_v2.py"),
        "--input", str(video_path),
        "--frame-times", str(pts_path),
        "--out", str(onset_path),
        "--clip", clip_id,
    ]
    media_fps = _optional_float(probe.get("media_fps"))
    if media_fps is not None and media_fps > 0:
        command.extend(["--frame-rate", str(media_fps)])
    _observe_access(
        ONSET_BUILDER_SOURCE,
        operation="subprocess_execute_script",
        via="audited_subprocess_invocation_boundary",
    )
    _observe_access(
        CANONICAL_DETECTOR_SOURCE,
        operation="subprocess_import_detector",
        via="audited_subprocess_invocation_boundary",
    )
    _observe_access(
        video_path,
        operation="subprocess_probe_and_decode_input",
        via="audited_subprocess_invocation_boundary",
    )
    _observe_access(
        pts_path,
        operation="subprocess_read_pts_identity",
        via="audited_subprocess_invocation_boundary",
    )
    _observe_access(
        onset_path,
        operation="subprocess_write_output",
        via="audited_subprocess_invocation_boundary",
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    invocation = {
        "command": shlex.join(command),
        "artifact_path": str(onset_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "detector_parameters": committed_detector_config(),
        "parameter_fitting_against_references": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"audio onset pipeline failed for {clip_id}: {completed.stderr}")
    payload = _read_object(onset_path)
    return payload, invocation


def committed_detector_config() -> dict[str, Any]:
    return {
        "detector_version": DETECTOR_VERSION,
        "implementation": "threed.racketsport.audio_onsets_v2 via scripts/racketsport/build_audio_onsets_v2.py",
        "analysis_sample_rate_hz": DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
        "bandpass_low_hz": DEFAULT_BANDPASS_LOW_HZ,
        "bandpass_high_hz": DEFAULT_BANDPASS_HIGH_HZ,
        "frame_size_s": DEFAULT_FRAME_SIZE_S,
        "hop_s": DEFAULT_HOP_S,
        "min_separation_s": DEFAULT_MIN_SEPARATION_S,
        "threshold_mad": DEFAULT_THRESHOLD_MAD,
        "adaptive_window_s": DEFAULT_ADAPTIVE_WINDOW_S,
        "min_pop_band_ratio": DEFAULT_MIN_POP_BAND_RATIO,
        "min_spectral_evidence": DEFAULT_MIN_SPECTRAL_EVIDENCE,
        "min_hfc_evidence": DEFAULT_MIN_HFC_EVIDENCE,
        "tuned_on_teacher_references": False,
    }


def extract_tt_snippet_core_features(
    samples: Sequence[float], *, sample_rate_hz: int
) -> dict[str, Any]:
    """Extract the frozen detector's core values at one snippet-local candidate.

    This intentionally does not call the detector an event pipeline: a 15 ms
    snippet cannot supply its 0.5 s adaptive context or exercise its 80 ms
    separation policy.  The feature equations and committed constants are
    nevertheless identical to ``audio_onsets_v2._detect_onsets``.  All four
    reported values are taken at the frame with maximum onset strength so the
    committed threshold conjunction is evaluated at one coherent candidate.
    """

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    sample_array = _as_float_array(samples)
    if sample_rate_hz != DEFAULT_ANALYSIS_SAMPLE_RATE_HZ:
        sample_array = _resample(
            sample_array,
            source_rate_hz=sample_rate_hz,
            target_rate_hz=DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
        )
    frame_size = max(16, int(round(DEFAULT_FRAME_SIZE_S * DEFAULT_ANALYSIS_SAMPLE_RATE_HZ)))
    hop = max(1, int(round(DEFAULT_HOP_S * DEFAULT_ANALYSIS_SAMPLE_RATE_HZ)))
    if len(sample_array) < frame_size:
        raise ValueError(
            f"snippet shorter than one committed analysis frame: "
            f"samples={len(sample_array)} frame_size={frame_size}"
        )

    centered = sample_array - float(np.mean(sample_array))
    raw_frames = _frame_signal(centered, frame_size=frame_size, hop=hop)
    filtered = _bandpass(
        sample_array,
        sample_rate_hz=DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
        low_hz=DEFAULT_BANDPASS_LOW_HZ,
        high_hz=DEFAULT_BANDPASS_HIGH_HZ,
    )
    frames = _frame_signal(filtered, frame_size=frame_size, hop=hop)
    if len(frames) < 3:
        raise ValueError(
            f"snippet yields fewer than three committed analysis frames: frames={len(frames)}"
        )

    window = np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    raw_spectrum = np.abs(np.fft.rfft(raw_frames * window, axis=1))
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / DEFAULT_ANALYSIS_SAMPLE_RATE_HZ)
    pop_band = (frequencies >= DEFAULT_BANDPASS_LOW_HZ) & (
        frequencies
        <= min(DEFAULT_BANDPASS_HIGH_HZ, DEFAULT_ANALYSIS_SAMPLE_RATE_HZ * 0.5 - 1.0)
    )
    low_band = (frequencies >= 50.0) & (frequencies < DEFAULT_BANDPASS_LOW_HZ)
    if not np.any(pop_band):
        raise ValueError("committed pop band has no FFT bins")

    raw_pop_power = np.sum(raw_spectrum[:, pop_band] ** 2, axis=1)
    raw_low_power = (
        np.sum(raw_spectrum[:, low_band] ** 2, axis=1)
        if np.any(low_band)
        else np.zeros_like(raw_pop_power)
    )
    pop_band_ratio = raw_pop_power / np.maximum(raw_pop_power + raw_low_power, 1e-12)
    band_magnitude = spectrum[:, pop_band]
    spectral_flux = np.r_[
        0.0,
        np.sum(np.maximum(0.0, band_magnitude[1:] - band_magnitude[:-1]), axis=1),
    ]
    hfc_weights = np.maximum(frequencies[pop_band] / DEFAULT_BANDPASS_LOW_HZ, 1.0)
    high_frequency_content = np.sum((band_magnitude**2) * hfc_weights, axis=1)
    band_rms = np.sqrt(np.mean(frames**2, axis=1))
    band_energy_delta = np.r_[0.0, np.maximum(0.0, np.diff(band_rms))]

    adaptive_window_frames = max(3, int(round(DEFAULT_ADAPTIVE_WINDOW_S / DEFAULT_HOP_S)))
    flux_z = _adaptive_positive_z(spectral_flux, window_frames=adaptive_window_frames)
    hfc_z = _adaptive_positive_z(
        high_frequency_content, window_frames=adaptive_window_frames
    )
    energy_z = _adaptive_positive_z(band_energy_delta, window_frames=adaptive_window_frames)
    onset_strength = 0.45 * flux_z + 0.35 * hfc_z + 0.20 * energy_z
    candidate_index = int(np.argmax(onset_strength))
    values = {
        "onset_strength": float(onset_strength[candidate_index]),
        "high_frequency_content": float(hfc_z[candidate_index]),
        "spectral_flux": float(flux_z[candidate_index]),
        "pop_band_ratio": float(pop_band_ratio[candidate_index]),
    }
    threshold_checks = {
        "onset_strength_gte_threshold_mad": (
            values["onset_strength"] >= DEFAULT_THRESHOLD_MAD
        ),
        "pop_band_ratio_gte_minimum": (
            values["pop_band_ratio"] >= DEFAULT_MIN_POP_BAND_RATIO
        ),
        "spectral_plus_hfc_gte_minimum": (
            values["spectral_flux"] + values["high_frequency_content"]
            >= DEFAULT_MIN_SPECTRAL_EVIDENCE
        ),
        "hfc_gte_minimum": (
            values["high_frequency_content"] >= DEFAULT_MIN_HFC_EVIDENCE
        ),
    }
    return {
        **values,
        "threshold_eligible": all(threshold_checks.values()),
        "threshold_checks": threshold_checks,
        "candidate_frame_index": candidate_index,
        "analysis_frame_count": len(frames),
        "analysis_sample_count": len(sample_array),
    }


def _read_tt_snippet(path: Path) -> tuple[np.ndarray, int, float]:
    _observe_access(path, operation="read_wav")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_rate_hz, samples = wavfile.read(path)
    sample_array = np.asarray(samples)
    if sample_array.ndim == 2:
        sample_array = np.mean(sample_array.astype(np.float64), axis=1)
    if sample_array.ndim != 1:
        raise ValueError(f"expected mono or channel-last WAV: shape={sample_array.shape}")
    if np.issubdtype(sample_array.dtype, np.integer):
        info = np.iinfo(sample_array.dtype)
        scale = float(max(abs(info.min), info.max))
        sample_array = sample_array.astype(np.float64) / scale
    else:
        sample_array = sample_array.astype(np.float64)
    return sample_array, int(sample_rate_hz), len(sample_array) / float(sample_rate_hz)


def _binary_auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC with average ranks for exact ties."""

    if not positive or not negative:
        return None
    values = np.asarray([*positive, *negative], dtype=np.float64)
    positive_count = len(positive)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[:positive_count]))
    mann_whitney_u = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return mann_whitney_u / (positive_count * len(negative))


def _feature_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature in TT_CORE_FEATURES:
        values = [float(row[feature]) for row in rows]
        output[feature] = {
            "n": len(values),
            "min": min(values) if values else None,
            "p10": _percentile(values, 10),
            "median": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "max": max(values) if values else None,
        }
    return output


def _threshold_confusion(
    positive_rows: Sequence[Mapping[str, Any]],
    background_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    true_positive = sum(bool(row["threshold_eligible"]) for row in positive_rows)
    false_negative = len(positive_rows) - true_positive
    false_positive = sum(bool(row["threshold_eligible"]) for row in background_rows)
    true_negative = len(background_rows) - false_positive
    predicted_positive = true_positive + false_positive
    return {
        "positive_n": len(positive_rows),
        "background_n": len(background_rows),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "precision_at_observed_class_mix": (
            true_positive / predicted_positive if predicted_positive else None
        ),
        "recall": true_positive / len(positive_rows) if positive_rows else None,
        "specificity": (
            true_negative / len(background_rows) if background_rows else None
        ),
        "false_positive_rate": (
            false_positive / len(background_rows) if background_rows else None
        ),
    }


def _source_grouped_feature_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["original_file"]), str(row["semantic_class"]))
        grouped.setdefault(key, []).append(row)
    medians = [
        {
            "original_file": original_file,
            "semantic_class": semantic_class,
            "snippet_count": len(group_rows),
            **{
                feature: _percentile(
                    [float(row[feature]) for row in group_rows], 50
                )
                for feature in TT_CORE_FEATURES
            },
        }
        for (original_file, semantic_class), group_rows in sorted(grouped.items())
    ]
    by_class = {
        name: [row for row in medians if row["semantic_class"] == name]
        for name in ("hit", "bounce", "background")
    }
    by_class["pooled_event"] = by_class["hit"] + by_class["bounce"]
    comparisons: dict[str, Any] = {}
    for comparison_name, positive_name in (
        ("hit_vs_background", "hit"),
        ("bounce_vs_background", "bounce"),
        ("pooled_event_vs_background", "pooled_event"),
    ):
        positive_rows = by_class[positive_name]
        background_rows = by_class["background"]
        comparisons[comparison_name] = {
            "positive_source_class_n": len(positive_rows),
            "background_source_class_n": len(background_rows),
            "auroc_on_source_class_medians": {
                feature: _binary_auroc(
                    [float(row[feature]) for row in positive_rows],
                    [float(row[feature]) for row in background_rows],
                )
                for feature in TT_CORE_FEATURES
            },
        }
    return {
        "unit": "original-file by semantic-class median",
        "unique_source_recordings": len({str(row["original_file"]) for row in rows}),
        "source_class_group_sizes": {
            name: len(class_rows) for name, class_rows in by_class.items()
        },
        "comparisons": comparisons,
        "caveat": (
            "Snippets from the same original recording are correlated. Snippet n is not an "
            "independent effective sample size; source-class medians are sensitivity diagnostics, "
            "not a replacement full-pipeline estimate."
        ),
    }


def _git_blob_hash(path: Path, *, at_head: bool) -> str | None:
    try:
        relative = Path(_lexical_repo_path(path))
    except ValueError:
        return None
    command = (
        ["git", "rev-parse", f"HEAD:{relative.as_posix()}"]
        if at_head
        else ["git", "hash-object", str(path)]
    )
    _observe_access(
        ROOT if at_head else path,
        operation=("git_head_blob_lookup" if at_head else "git_hash_object_read"),
        via="audited_subprocess_invocation_boundary",
    )
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _validate_detector_source_identity(detector_source: Path) -> dict[str, str]:
    lexical = _lexical_repo_path(detector_source)
    expected = _lexical_repo_path(CANONICAL_DETECTOR_SOURCE)
    if lexical != expected:
        raise ValueError(
            "detector source must be the canonical committed implementation: "
            f"expected {expected}, got {lexical}"
        )
    if not _path_is_file(detector_source):
        raise FileNotFoundError(f"canonical detector source is absent: {detector_source}")
    working_sha256 = _sha256(detector_source)
    working_blob = _git_blob_hash(detector_source, at_head=False)
    head_blob = _git_blob_hash(detector_source, at_head=True)
    if not working_sha256 or not working_blob or not head_blob:
        raise RuntimeError(
            "canonical detector identity requires non-null working SHA-256, working Git blob, "
            "and committed HEAD Git blob"
        )
    if working_blob != head_blob:
        raise RuntimeError(
            "canonical detector source does not match the committed HEAD blob: "
            f"working={working_blob} head={head_blob}"
        )
    return {
        "canonical_repo_path": lexical,
        "working_sha256": working_sha256,
        "working_git_blob": working_blob,
        "committed_head_git_blob": head_blob,
    }


def _anchor_lane_reference(anchor_report_path: Path) -> dict[str, Any]:
    anchor = _read_object(anchor_report_path)
    anchor_access_audit = anchor.get("access_audit")
    if not isinstance(anchor_access_audit, Mapping):
        raise ValueError("clean anchor report must contain a derived access audit")
    if (
        not anchor_access_audit.get("all_declared_inputs_validated_before_access")
        or not anchor_access_audit.get("all_observed_input_accesses_allowed")
        or int(anchor_access_audit.get("forbidden_input_access_count", -1)) != 0
    ):
        raise ValueError("clean anchor report failed its pre-access registry audit")
    metrics = []
    for row in anchor.get("pooled_reference_metrics") or []:
        metrics.append({
            key: row.get(key)
            for key in (
                "tolerance_ms",
                "reference_event_count",
                "onset_count",
                "matched_count",
                "recall",
                "precision_proxy",
                "median_absolute_offset_ms",
                "p90_absolute_offset_ms",
                "unmatched_onset_rate_per_min",
            )
        })
    viability = [
        {
            key: row.get(key)
            for key in (
                "clip_id",
                "domain",
                "status",
                "duration_s",
                "onset_count",
                "onset_rate_per_min",
                "mean_core_feature_fraction_inside_reference_p10_p90",
            )
        }
        for row in anchor.get("viability_probes") or []
    ]
    return {
        "source_path": str(anchor_report_path),
        "source_sha256": _sha256(anchor_report_path),
        "copied_not_recomputed": True,
        "reference_metrics_by_tolerance": metrics,
        "rally_only_reference_metrics_by_tolerance": (
            (anchor.get("reference_clips") or [{}])[0].get(
                "rally_only_metrics_by_tolerance"
            )
            if anchor.get("reference_clips")
            else []
        ),
        "denominator_views": (
            (anchor.get("reference_clips") or [{}])[0].get("denominator_views")
            if anchor.get("reference_clips")
            else {}
        ),
        "rally_conditioned_null_by_tolerance": (
            (anchor.get("reference_clips") or [{}])[0].get(
                "rally_conditioned_null_by_tolerance"
            )
            if anchor.get("reference_clips")
            else []
        ),
        "access_audit": dict(anchor_access_audit),
        "reference_metrics_by_event_type": anchor.get(
            "pooled_reference_metrics_by_event_type"
        ),
        "viability_probes": viability,
    }


def build_tt_sounds_calibration_report(
    *,
    labels_csv: Path,
    snippets_dir: Path,
    detector_source: Path,
    anchor_report_path: Path,
    trackd_findings_path: Path,
    access_audit: Mapping[str, Any],
) -> dict[str, Any]:
    detector_identity_before = _validate_detector_source_identity(detector_source)
    detector_sha256_before = detector_identity_before["working_sha256"]
    detector_blob_before = detector_identity_before["working_git_blob"]
    detector_head_blob = detector_identity_before["committed_head_git_blob"]
    _observe_access(labels_csv, operation="read_csv")
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"bounce-id", "original-file", "timestamp", "surface"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"TT-Sounds labels missing required columns: required={sorted(required)} "
                f"actual={reader.fieldnames}"
            )
        labels = [dict(row) for row in reader]
    bounce_ids = [str(row["bounce-id"]) for row in labels]
    if len(set(bounce_ids)) != len(bounce_ids):
        raise ValueError("TT-Sounds bounce-id values must be unique")
    invalid_surfaces = sorted({str(row["surface"]) for row in labels} - TT_SURFACES)
    if invalid_surfaces:
        raise ValueError(f"unexpected TT-Sounds surfaces: {invalid_surfaces}")

    surface_counts = {
        surface: sum(row["surface"] == surface for row in labels)
        for surface in sorted(TT_SURFACES)
    }
    all_wavs = sorted(_glob_paths(snippets_dir, "*.wav"))
    numeric_wavs = {path.stem: path for path in all_wavs if path.stem.isdigit()}
    n_prefixed_wavs = [
        path for path in all_wavs if path.stem.startswith("n") and path.stem[1:].isdigit()
    ]
    label_id_set = set(bounce_ids)
    missing_labeled_wavs = sorted(label_id_set - set(numeric_wavs), key=int)
    unlabeled_numeric_wavs = sorted(set(numeric_wavs) - label_id_set, key=int)
    raw_sound_candidates = [
        labels_csv.parent / "raw_sounds",
        snippets_dir.parent / "raw_sounds",
        snippets_dir.parent.parent / "raw_sounds",
    ]
    continuous_raw_sounds_present = any(_path_is_dir(path) for path in raw_sound_candidates)

    feature_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    duration_counts: dict[str, int] = {}
    for row in labels:
        bounce_id = str(row["bounce-id"])
        path = snippets_dir / f"{bounce_id}.wav"
        duration_s: float | None = None
        if not _path_is_file(path):
            exclusions.append({
                "bounce_id": bounce_id,
                "surface": row["surface"],
                "reason": "labeled_snippet_missing",
            })
            continue
        try:
            samples, sample_rate_hz, duration_s = _read_tt_snippet(path)
            duration_key = f"{duration_s * 1000.0:.6f}ms"
            duration_counts[duration_key] = duration_counts.get(duration_key, 0) + 1
            features = extract_tt_snippet_core_features(
                samples, sample_rate_hz=sample_rate_hz
            )
        except (OSError, ValueError) as exc:
            exclusions.append({
                "bounce_id": bounce_id,
                "surface": row["surface"],
                "duration_s": duration_s,
                "reason": str(exc),
            })
            continue
        semantic_class = {
            "racket": "hit",
            "table": "bounce",
            "floor": "background",
            "other": "background",
        }[str(row["surface"])]
        feature_rows.append({
            "bounce_id": bounce_id,
            "original_file": row["original-file"],
            "surface": row["surface"],
            "semantic_class": semantic_class,
            **features,
        })

    class_rows = {
        name: [row for row in feature_rows if row["semantic_class"] == name]
        for name in ("hit", "bounce", "background")
    }
    class_rows["pooled_event"] = class_rows["hit"] + class_rows["bounce"]
    comparisons: dict[str, Any] = {}
    for comparison_name, positive_name in (
        ("hit_vs_background", "hit"),
        ("bounce_vs_background", "bounce"),
        ("pooled_event_vs_background", "pooled_event"),
    ):
        positive_rows = class_rows[positive_name]
        background_rows = class_rows["background"]
        comparisons[comparison_name] = {
            "positive_class": positive_name,
            "positive_n": len(positive_rows),
            "background_n": len(background_rows),
            "auroc_higher_means_positive": {
                feature: _binary_auroc(
                    [float(row[feature]) for row in positive_rows],
                    [float(row[feature]) for row in background_rows],
                )
                for feature in TT_CORE_FEATURES
            },
            "committed_threshold_confusion": _threshold_confusion(
                positive_rows, background_rows
            ),
            "interpretation": (
                "Observed-mix snippet eligibility diagnostic only; not committed-detector "
                "precision/recall and not an independent-snippet uncertainty estimate."
            ),
        }

    source_grouped_diagnostics = _source_grouped_feature_diagnostics(feature_rows)

    trackd_text = _read_text(trackd_findings_path, encoding="utf-8")
    required_trackd_terms = ("62/74", "83.8%", "HARDEST", "E0-excluded")
    if any(term not in trackd_text for term in required_trackd_terms):
        raise ValueError(
            "Track D findings do not contain the dispatched hardest-first, E0-rejected "
            "62/74 = 83.8% prior"
        )

    detector_identity_after = _validate_detector_source_identity(detector_source)
    detector_sha256_after = detector_identity_after["working_sha256"]
    detector_blob_after = detector_identity_after["working_git_blob"]
    if detector_sha256_before != detector_sha256_after:
        raise RuntimeError("detector source changed during calibration")
    if detector_head_blob != detector_blob_after:
        raise RuntimeError("detector source does not match the committed HEAD blob")

    return {
        "schema_version": 2,
        "artifact_type": "audio_anchor_tt_sounds_calibration",
        "status": "SNIPPET_DIAGNOSTICS_ONLY_FULL_PIPELINE_NOT_DERIVABLE",
        "verified": False,
        "promotion": "none",
        "calibration_verdict": {
            "classification": "honestly_undecidable",
            "directional_evidence": "snippet_local_feature_separation_only",
            "reason": (
                "Controlled isolated snippets show snippet-local feature separation, but this design "
                "cannot attribute the difference to capture regime rather than sport, windowing, corpus, "
                "source, or context. Missing continuous recordings prevent full-pipeline precision, "
                "recall, offset, or false-onset-burden measurement."
            ),
        },
        "corpus_layout": {
            "labels_csv": str(labels_csv),
            "snippets_dir": str(snippets_dir),
            "label_rows": len(labels),
            "original_source_recordings": len(
                {str(row["original-file"]) for row in labels}
            ),
            "within_source_correlation_disclosed": True,
            "surface_counts": surface_counts,
            "semantic_class_counts": {
                "hit_racket": surface_counts["racket"],
                "bounce_table": surface_counts["table"],
                "background_floor_plus_other": (
                    surface_counts["floor"] + surface_counts["other"]
                ),
            },
            "all_wav_files": len(all_wavs),
            "numeric_wav_files": len(numeric_wavs),
            "label_addressed_numeric_wav_files": len(label_id_set & set(numeric_wavs)),
            "unlabeled_numeric_wav_files": len(unlabeled_numeric_wavs),
            "n_prefixed_variant_wav_files": len(n_prefixed_wavs),
            "missing_labeled_wav_files": missing_labeled_wavs,
            "labeled_snippet_duration_counts": duration_counts,
            "continuous_raw_sounds_present": continuous_raw_sounds_present,
            "continuous_raw_sound_paths_checked": [
                str(path) for path in raw_sound_candidates
            ],
            "continuous_recording_note": (
                "full.csv retains original-file timestamps, but raw_sounds was not fetched; only extracted snippets exist."
            ),
        },
        "feature_measurement": {
            "scope": "snippet_local_feature_diagnostics_only",
            "method": (
                "Resample each labeled WAV to 24 kHz, apply the committed 1-6 kHz filter and 6 ms/1 ms frame/hop, "
                "compute the exact audio_onsets_v2 core equations, and read all four features at the snippet-local "
                "maximum-onset-strength frame. No parameter was fitted."
            ),
            "limits": (
                "A 15 ms snippet cannot supply the detector's 0.5 s adaptive context or exercise its 80 ms "
                "minimum-separation policy. These are snippet-level feature AUROCs and threshold-eligibility "
                "confusions, not full-pipeline detection precision/recall/offset metrics. Snippets within "
                "each original recording are correlated, so 5,700 snippets are not 5,700 independent sources."
            ),
            "feature_definitions": {
                "onset_strength": "committed 0.45*flux_z + 0.35*hfc_z + 0.20*band_energy_delta_z",
                "high_frequency_content": "committed adaptive-positive-z HFC",
                "spectral_flux": "committed adaptive-positive-z spectral flux",
                "pop_band_ratio": "committed raw 1-6 kHz power / (1-6 kHz + 50-1000 Hz power)",
            },
            "candidate_frame_policy": "argmax onset_strength within each snippet; all feature values use that same frame",
            "usable_class_sizes": {
                name: len(rows) for name, rows in class_rows.items()
            },
            "excluded_snippets": exclusions,
            "distributions_by_class": {
                name: _feature_distribution(rows)
                for name, rows in class_rows.items()
            },
            "comparisons": comparisons,
            "source_grouped_sensitivity": source_grouped_diagnostics,
            "threshold_policy": {
                "kind": "snippet_local_threshold_conjunction_diagnostic_not_full_detector_output",
                "conjunction": (
                    "onset_strength>=4.0 AND pop_band_ratio>=0.10 AND "
                    "spectral_flux+high_frequency_content>=0.5 AND high_frequency_content>=0.7"
                ),
                "local_peak_and_minimum_separation_not_claimed": True,
                "observed_mix_eligibility_not_deployable_precision": True,
            },
        },
        "full_pipeline_detection_metrics": {
            "derivable": False,
            "status": "NOT_DERIVABLE_FROM_CORPUS_SHAPE",
            "reason": "continuous raw_sounds recordings are absent; only centered short snippets were fetched",
            "tolerance_bands_ms_that_would_apply": [33.0, 66.0, 100.0],
        },
        "frozen_detector": {
            "parameters": committed_detector_config(),
            "parameter_fit_performed": False,
            "source_path": detector_identity_before["canonical_repo_path"],
            "source_sha256_before": detector_sha256_before,
            "source_sha256_after": detector_sha256_after,
            "working_git_blob_before": detector_blob_before,
            "working_git_blob_after": detector_blob_after,
            "committed_head_git_blob": detector_head_blob,
            "unchanged_before_after": detector_sha256_before == detector_sha256_after,
            "matches_committed_head": detector_blob_after == detector_head_blob,
            "identity_fields_non_null": all(
                (detector_sha256_before, detector_sha256_after, detector_blob_before,
                 detector_blob_after, detector_head_blob)
            ),
        },
        "anchor_lane_reference": _anchor_lane_reference(anchor_report_path),
        "trackd_candidate_level_prior": {
            "source_path": str(trackd_findings_path),
            "source_sha256": _sha256(trackd_findings_path),
            "family": "audio_onset_audio_only",
            "owner_confirmed": 62,
            "candidate_count": 74,
            "candidate_precision": 62 / 74,
            "reported_percent": 83.8,
            "selection": "74 hardest-first most-uncertain candidates",
            "generation": "teacher-generated audio-only candidates",
            "e0_disposition": "rejected_by_E0_audio_weight_zero",
            "qualification": "source describes rates as lower bounds",
            "evidence_level": "owner_precision_on_hardest_first_E0_rejected_candidates_not_detector_precision_recall",
            "anchor_clip_overlap": {
                "clip_id": "xkadsq9bli3h",
                "candidate_count": 12,
                "owner_confirmed": 10,
                "leave_clip_out_owner_confirmed": 52,
                "leave_clip_out_candidate_count": 62,
                "leave_clip_out_precision": 52 / 62,
                "reported_percent": 83.9,
                "source": "ball_audio_review_20260722 independent verdict-claims audit",
            },
        },
        "consumer_implications": {
            "track_b_ball_3d": (
                "Outdoor consumer capture does not support standalone audio anchoring at +/-66 ms; audio may be "
                "retained only as untyped timing support for a type assigned independently. Controlled isolated "
                "snippets show snippet-local feature separation, but no TT full-pipeline tolerance claim is derivable."
            ),
            "track_d_late_fusion": (
                "The detector-level outdoor operating point is high-recall/low-precision and therefore supplies a "
                "noisy feature rather than an event decision. The separate owner prior comes from hardest-first, "
                "teacher-generated audio-only candidates that E0 rejected; Track D owns the adoption gate."
            ),
        },
        "access_audit": dict(access_audit),
        "data_fences": {"derivation": "pending_runtime_observation_finalization"},
        "cross_signal": {
            "consumes": [
                "TT-Sounds labeled corpus from Track D audit",
                "ball_audio_repair2_20260722/alignment_report_v3.json",
                "Track D batch-01 owner candidate precision",
            ],
            "feeds": [
                "Track B 3D bounce anchoring decision",
                "Track D audio late-fusion gate",
            ],
        },
        "best_stack_delta": {
            "classification": "c",
            "delta": "none",
            "configs/racketsport/best_stack.json_touched_by_lane": False,
            "configs/racketsport/best_stack.json_accessed_by_cli": False,
        },
    }


def _onsets_on_media_clock(payload: Mapping[str, Any], audio_origin_s: float) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for onset in payload.get("onsets") or []:
        if not isinstance(onset, Mapping):
            continue
        raw_time = _optional_float(onset.get("raw_time_s"))
        if raw_time is None:
            continue
        item = dict(onset)
        item["detector_raw_time_s"] = raw_time
        item["media_time_s"] = audio_origin_s + raw_time
        mapped.append(item)
    return mapped


def measure_reference_clip(spec: Mapping[str, Any], *, raw_dir: Path) -> dict[str, Any]:
    clip_id = str(spec["clip_id"])
    video_path = Path(spec["video_path"])
    cv_path = Path(spec["cv_export_path"])
    insights_path = Path(spec["insights_path"])
    probe = probe_media_pts(video_path, audio_only=False)
    cv_export = _read_object(cv_path)
    insights = _read_object(insights_path)
    events, export_fps = extract_pbvision_events(cv_export)
    rally_intervals = extract_pbvision_rally_intervals(
        cv_export, export_fps=export_fps
    )
    crosscheck = crosscheck_insights_timebase(cv_export, insights, events, export_fps=export_fps)
    mapping = derive_reference_mapping(
        probe, export_fps=export_fps, events=events, insights_crosscheck=crosscheck
    )
    base = {
        "clip_id": clip_id,
        "domain": spec["domain"],
        "status": mapping["status"],
        "video_path": str(video_path),
        "cv_export_path": str(cv_path),
        "insights_path": str(insights_path),
        "reference_source": "pbvision_model_teacher_not_ground_truth_not_frozen_judge",
        "pts_evidence": probe,
        "timebase_alignment": mapping,
        "reference_event_count": len(events),
        "reference_event_counts_by_type": {
            "hit": sum(item["event_type"] == "hit" for item in events),
            "bounce": sum(item["event_type"] == "bounce" for item in events),
        },
        "reference_events": events,
        "rally_intervals_export_clock": rally_intervals,
        "reference_identity": {
            "cv_export_sha256": _sha256(cv_path),
            "insights_sha256": _sha256(insights_path),
        },
    }
    if mapping["status"] != "ALIGNED":
        return base
    onset_payload, invocation = run_committed_onset_pipeline(
        clip_id=clip_id, video_path=video_path, probe=probe, raw_dir=raw_dir
    )
    audio_origin = float(mapping["mapping"]["audio_effective_origin_pts_s"])
    video_origin = float(mapping["mapping"]["video_effective_origin_pts_s"])
    media_fps = float(mapping["mapping"]["media_fps"])
    mapped_onsets = _onsets_on_media_clock(onset_payload, audio_origin)
    for event in events:
        event["media_time_s"] = video_origin + float(event["export_time_s"])
    reference_times = [float(item["media_time_s"]) for item in events]
    onset_times = [float(item["media_time_s"]) for item in mapped_onsets]
    duration_s = _optional_float((probe.get("format") or {}).get("duration")) or float(onset_payload["summary"]["duration_s"])
    rally_intervals_media = [
        (
            video_origin + float(item["start_export_s"]),
            video_origin + float(item["end_export_s"]),
        )
        for item in rally_intervals
    ]
    rally_reference_times = _times_inside_intervals(
        reference_times, rally_intervals_media
    )
    rally_onset_times = _times_inside_intervals(onset_times, rally_intervals_media)
    rally_duration_s = sum(end - start for start, end in rally_intervals_media)
    tolerance_metrics = [
        measure_tolerance(
            reference_times, onset_times, tolerance_s=tolerance,
            media_fps=media_fps, duration_s=duration_s,
        ) for tolerance in TOLERANCES_S
    ]
    rally_only_tolerance_metrics = [
        measure_tolerance(
            rally_reference_times,
            rally_onset_times,
            tolerance_s=tolerance,
            media_fps=media_fps,
            duration_s=rally_duration_s,
        )
        for tolerance in TOLERANCES_S
    ]
    metrics_by_event_type: dict[str, list[dict[str, Any]]] = {}
    for event_type in ("hit", "bounce"):
        type_times = [
            float(item["media_time_s"]) for item in events if item["event_type"] == event_type
        ]
        metrics_by_event_type[event_type] = [
            measure_tolerance(
                type_times, onset_times, tolerance_s=tolerance,
                media_fps=media_fps, duration_s=duration_s,
            ) for tolerance in TOLERANCES_S
        ]
    null_metrics = []
    rally_conditioned_null_metrics = []
    for tolerance, measured in zip(TOLERANCES_S, tolerance_metrics):
        null = measure_circular_shift_null(
            reference_times, onset_times, tolerance_s=tolerance, duration_s=duration_s
        )
        null["tolerance_ms"] = tolerance * 1000.0
        null["measured_recall_minus_null_median"] = (
            float(measured["recall"]) - float(null["median_recall"])
            if measured["recall"] is not None and null["median_recall"] is not None else None
        )
        null["measured_precision_proxy_minus_null_median"] = (
            float(measured["precision_proxy"]) - float(null["median_precision_proxy"])
            if measured["precision_proxy"] is not None and null["median_precision_proxy"] is not None else None
        )
        null_metrics.append(null)
        rally_null = measure_rally_conditioned_null(
            reference_times,
            onset_times,
            rally_intervals_s=rally_intervals_media,
            tolerance_s=tolerance,
        )
        rally_null["tolerance_ms"] = tolerance * 1000.0
        rally_conditioned_null_metrics.append(rally_null)
    largest = tolerance_metrics[-1]
    matched_indices = {int(item["onset_index"]) for item in largest["matched_pairs"]}
    matched_onsets = [item for index, item in enumerate(mapped_onsets) if index in matched_indices]
    base.update({
        "status": "MEASURED_TEACHER_ALIGNMENT",
        "duration_s": duration_s,
        "media_fps": media_fps,
        "onset_count": len(mapped_onsets),
        "onset_rate_per_min": len(mapped_onsets) / (duration_s / 60.0),
        "onset_extraction": invocation,
        "onset_artifact_summary": onset_payload.get("summary"),
        "metrics_by_tolerance": tolerance_metrics,
        "rally_only_metrics_by_tolerance": rally_only_tolerance_metrics,
        "metrics_by_event_type": metrics_by_event_type,
        "circular_time_shift_null_by_tolerance": null_metrics,
        "rally_conditioned_null_by_tolerance": rally_conditioned_null_metrics,
        "denominator_views": {
            "full_clip": {
                "onset_count": len(onset_times),
                "duration_s": duration_s,
                "description": "precision proxy uses every full-clip onset; burden uses full media duration",
            },
            "rally_only": {
                "rally_interval_count": len(rally_intervals_media),
                "reference_event_count": len(rally_reference_times),
                "onset_count": len(rally_onset_times),
                "duration_s": rally_duration_s,
                "description": "sensitivity view restricted to the union of half-open teacher rally intervals",
            },
        },
        "all_onset_feature_distribution": summarize_features(mapped_onsets),
        "matched_onset_feature_distribution_at_100ms": summarize_features(matched_onsets),
        "matched_onset_count_at_100ms": len(matched_onsets),
    })
    del onset_payload, mapped_onsets, matched_onsets
    gc.collect()
    return base


def measure_viability_clip(
    spec: Mapping[str, Any], *, raw_dir: Path, reference_matched_summary: Mapping[str, Any], reference_rate: float
) -> dict[str, Any]:
    clip_id = str(spec["clip_id"])
    video_path = Path(spec["video_path"])
    probe = probe_media_pts(video_path, audio_only=True)
    if probe.get("status") != "PTS_PROBED" or _optional_float(probe.get("audio_effective_origin_pts_s")) is None:
        return {
            "clip_id": clip_id,
            "domain": spec["domain"],
            "status": "AUDIO_VIABILITY_UNPROBEABLE",
            "pts_evidence": probe,
        }
    onset_payload, invocation = run_committed_onset_pipeline(
        clip_id=clip_id, video_path=video_path, probe=probe, raw_dir=raw_dir
    )
    audio_origin = float(probe["audio_effective_origin_pts_s"])
    onsets = _onsets_on_media_clock(onset_payload, audio_origin)
    duration_s = _optional_float((probe.get("format") or {}).get("duration")) or float(onset_payload["summary"]["duration_s"])
    rate = len(onsets) / (duration_s / 60.0)
    feature_summary = summarize_features(onsets)
    comparison = compare_features_to_reference(onsets, reference_matched_summary)
    core_overlap = [
        _optional_float(comparison[name]["fraction_inside_reference_matched_p10_p90"])
        for name in ("onset_strength", "spectral_flux", "high_frequency_content", "pop_band_ratio")
    ]
    core_overlap = [value for value in core_overlap if value is not None]
    overlap_mean = sum(core_overlap) / len(core_overlap) if core_overlap else None
    if not onsets:
        structure_verdict = "NO_POP_LIKE_TRANSIENTS_DETECTED"
    elif overlap_mean is not None and overlap_mean >= 0.25:
        structure_verdict = "POP_LIKE_TRANSIENT_STRUCTURE_PRESENT__VIABILITY_ONLY"
    else:
        structure_verdict = "TRANSIENTS_DETECTED_BUT_SPECTRALLY_SHIFTED__VIABILITY_ONLY"
    burden_ratio = rate / reference_rate if reference_rate > 0 else None
    burden_verdict = (
        "HIGH_ONSET_BURDEN_VS_REFERENCE_CLIP" if burden_ratio is not None and burden_ratio >= 2.0
        else "NOT_HIGH_BY_TWO_X_DIAGNOSTIC"
    )
    result = {
        "clip_id": clip_id,
        "domain": spec["domain"],
        "status": "AUDIO_VIABILITY_ONLY_NO_EVENT_ACCURACY_CLAIM",
        "video_path": str(video_path),
        "duration_s": duration_s,
        "onset_count": len(onsets),
        "onset_rate_per_min": rate,
        "pts_evidence": probe,
        "onset_extraction": invocation,
        "onset_artifact_summary": onset_payload.get("summary"),
        "onset_feature_distribution": feature_summary,
        "comparison_to_reference_matched_onsets_at_100ms": comparison,
        "mean_core_feature_fraction_inside_reference_p10_p90": overlap_mean,
        "onset_rate_ratio_to_reference_aligned_clip": burden_ratio,
        "qualitative_verdict": {
            "transient_structure": structure_verdict,
            "spurious_burden_diagnostic": burden_verdict,
            "limits": "No independent event timestamps were used; this is spectral/rate viability evidence only. Noise source attribution and event accuracy are not claimed.",
        },
        "audio_only_fence": "HELD: ffprobe packet metadata and ffmpeg 0:a:0 decode only; no video frames decoded",
    }
    del onset_payload, onsets
    gc.collect()
    return result


def _pool_metrics(
    clips: Sequence[Mapping[str, Any]], *, event_type: str | None = None
) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for tolerance_index, tolerance in enumerate(TOLERANCES_S):
        rows = [
            (
                clip["metrics_by_event_type"][event_type][tolerance_index]
                if event_type is not None else clip["metrics_by_tolerance"][tolerance_index]
            )
            for clip in clips
        ]
        reference_count = sum(int(row["reference_event_count"]) for row in rows)
        onset_count = sum(int(row["onset_count"]) for row in rows)
        matched_count = sum(int(row["matched_count"]) for row in rows)
        offsets = [
            float(pair["absolute_offset_ms"])
            for row in rows for pair in row["matched_pairs"]
        ]
        unmatched_onsets = onset_count - matched_count
        duration_minutes = sum(float(clip["duration_s"]) for clip in clips) / 60.0
        pooled.append({
            "tolerance_ms": tolerance * 1000.0,
            "reference_event_count": reference_count,
            "onset_count": onset_count,
            "matched_count": matched_count,
            "recall": matched_count / reference_count if reference_count else None,
            "precision_proxy": matched_count / onset_count if onset_count else None,
            "median_absolute_offset_ms": _percentile(offsets, 50),
            "p90_absolute_offset_ms": _percentile(offsets, 90),
            "unmatched_onset_count": unmatched_onsets,
            "unmatched_onset_rate_per_min": unmatched_onsets / duration_minutes if duration_minutes else None,
            "per_clip_decomposition": [
                {
                    "clip_id": clip["clip_id"],
                    "reference_event_count": row["reference_event_count"],
                    "onset_count": row["onset_count"],
                    "matched_count": row["matched_count"],
                    "recall": row["recall"],
                    "precision_proxy": row["precision_proxy"],
                }
                for clip, row in zip(clips, rows)
            ],
        })
    return pooled


def _pooled_matched_feature_summary(clips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Pooling quantiles exactly from per-pair matched feature values requires raw artifacts.
    # Reload only the small onset artifacts and select the already-recorded 100 ms indices.
    pooled_onsets: list[Mapping[str, Any]] = []
    for clip in clips:
        onset_path = Path(clip["onset_extraction"]["artifact_path"])
        payload = _read_object(onset_path)
        matched_indices = {
            int(pair["onset_index"]) for pair in clip["metrics_by_tolerance"][-1]["matched_pairs"]
        }
        payload_onsets = [item for item in payload.get("onsets") or [] if isinstance(item, Mapping)]
        pooled_onsets.extend(item for index, item in enumerate(payload_onsets) if index in matched_indices)
    return summarize_features(pooled_onsets)


def build_measurement_report(
    *,
    references: Sequence[Mapping[str, Any]],
    viability_specs: Sequence[Mapping[str, Any]],
    excluded_references: Sequence[Mapping[str, Any]],
    policy_incidents: Sequence[str],
    raw_dir: Path,
    access_audit: Mapping[str, Any],
    detector_source: Path,
) -> dict[str, Any]:
    detector_identity_before = _validate_detector_source_identity(detector_source)
    reference_results = [measure_reference_clip(spec, raw_dir=raw_dir) for spec in references]
    usable = [item for item in reference_results if item.get("status") == "MEASURED_TEACHER_ALIGNMENT"]
    matched_summary = _pooled_matched_feature_summary(usable) if usable else summarize_features([])
    reference_rate = (
        sum(float(item["onset_count"]) for item in usable)
        / (sum(float(item["duration_s"]) for item in usable) / 60.0)
        if usable else 0.0
    )
    viability_results = [
        measure_viability_clip(
            spec, raw_dir=raw_dir,
            reference_matched_summary=matched_summary,
            reference_rate=reference_rate,
        )
        for spec in viability_specs
    ]
    domains: dict[str, Any] = {}
    for clip in usable:
        domains.setdefault(str(clip["domain"]), {"reference_clips": [], "metrics_by_tolerance": []})
        domains[str(clip["domain"])]["reference_clips"].append(str(clip["clip_id"]))
    for domain, domain_payload in domains.items():
        domain_clips = [clip for clip in usable if clip["domain"] == domain]
        domain_payload["metrics_by_tolerance"] = _pool_metrics(domain_clips)
        domain_payload["metrics_by_event_type"] = {
            event_type: _pool_metrics(domain_clips, event_type=event_type)
            for event_type in ("hit", "bounce")
        }
    detector_identity_after = _validate_detector_source_identity(detector_source)
    if detector_identity_before != detector_identity_after:
        raise RuntimeError("canonical detector identity changed during alignment measurement")
    return {
        "schema_version": 2,
        "artifact_type": "audio_event_alignment_measurement",
        "status": "MEASUREMENT_ONLY_VERIFIED_0",
        "verified": False,
        "promotion": "none",
        "onset_detector": committed_detector_config(),
        "detector_source_identity": {
            **detector_identity_after,
            "identity_fields_non_null": all(detector_identity_after.values()),
            "unchanged_during_measurement": True,
        },
        "matching_algorithm": (
            "For each tolerance, enumerate all reference/onset pairs inside the band, sort by "
            "absolute offset then reference time then onset time/index, and greedily accept a pair "
            "only when neither member was used. This is deterministic nearest-neighbor one-to-one matching."
        ),
        "null_interpretation": {
            "whole_clip": "activity-confounded weak null; never sufficient by itself",
            "rally_conditioned": (
                "primary local-alignment sensitivity: independent seeded circular shifts "
                "within each rally preserve rally occupancy"
            ),
        },
        "tolerance_bands_ms": [value * 1000.0 for value in TOLERANCES_S],
        "reference_clips": reference_results,
        "excluded_references": list(excluded_references),
        "policy_incidents": list(policy_incidents),
        "pooled_reference_metrics": _pool_metrics(usable) if usable else [],
        "pooled_reference_metrics_by_event_type": (
            {
                event_type: _pool_metrics(usable, event_type=event_type)
                for event_type in ("hit", "bounce")
            }
            if usable else {"hit": [], "bounce": []}
        ),
        "reference_domains": domains,
        "reference_matched_onset_feature_distribution_at_100ms": matched_summary,
        "viability_probes": viability_results,
        "access_audit": dict(access_audit),
        "data_fences": {"derivation": "pending_runtime_observation_finalization"},
        "cross_signal": {
            "consumes": ["audio_onsets_v2 committed defaults", "pb.vision model-teacher timestamps", "media PTS"],
            "feeds": [
                "Track B untyped timing support for independently assigned event types",
                "Track D audio late-fusion gate",
                "audio-XOR-kink adjudication context",
            ],
        },
        "best_stack_delta": {
            "classification": "c",
            "delta": "none",
            "configs/racketsport/best_stack.json_touched": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure committed audio_onsets_v2 timing against pb.vision teacher events, "
            "or calibrate its frozen core features on TT-Sounds snippets."
        )
    )
    parser.add_argument(
        "--reference", nargs=5, action="append", default=[],
        metavar=("CLIP_ID", "VIDEO", "CV_EXPORT", "INSIGHTS", "DOMAIN"),
        help="Repeatable reference-aligned clip specification.",
    )
    parser.add_argument(
        "--viability", nargs=3, action="append", default=[],
        metavar=("CLIP_ID", "VIDEO", "DOMAIN"),
        help="Repeatable audio-only viability clip specification.",
    )
    parser.add_argument(
        "--excluded-reference", nargs=4, action="append", default=[],
        metavar=("CLIP_ID", "EXTERNAL_ID", "DOMAIN", "REASON"),
        help="Record a policy-excluded reference without opening its paths.",
    )
    parser.add_argument(
        "--policy-incident", action="append", default=[],
        help="Repeatable honest disclosure included verbatim in the report.",
    )
    parser.add_argument(
        "--tt-sounds-labels", type=Path,
        help="TT-Sounds full.csv; selects snippet feature-calibration mode.",
    )
    parser.add_argument(
        "--tt-sounds-snippets", type=Path,
        help="Directory containing TT-Sounds extracted WAV snippets.",
    )
    parser.add_argument(
        "--anchor-report", type=Path,
        help="Existing anchor-lane alignment_report.json to cite without recomputation.",
    )
    parser.add_argument(
        "--trackd-findings", type=Path,
        help="Track D owner-adjudication findings containing the candidate-level prior.",
    )
    parser.add_argument(
        "--detector-source", type=Path,
        default=ROOT / "threed/racketsport/audio_onsets_v2.py",
        help="Committed source defining frozen audio_onsets_v2 defaults.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output alignment_report.json path.")
    parser.add_argument(
        "--raw-dir", type=Path,
        help="Directory for raw committed-onset and PTS-evidence artifacts (alignment mode).",
    )
    return parser.parse_args()


def _main_observed(args: argparse.Namespace, observer: RuntimeAccessObserver) -> int:
    tt_arguments = (
        args.tt_sounds_labels,
        args.tt_sounds_snippets,
        args.anchor_report,
        args.trackd_findings,
    )
    if any(value is not None for value in tt_arguments):
        if not all(value is not None for value in tt_arguments):
            raise ValueError(
                "TT-Sounds mode requires --tt-sounds-labels, --tt-sounds-snippets, "
                "--anchor-report, and --trackd-findings"
            )
        if args.reference or args.viability or args.excluded_reference or args.policy_incident:
            raise ValueError("TT-Sounds mode cannot be combined with alignment-mode inputs")
        pre_access_validation = _validate_tt_inputs_before_access(
            labels_csv=args.tt_sounds_labels,
            snippets_dir=args.tt_sounds_snippets,
            anchor_report_path=args.anchor_report,
            trackd_findings_path=args.trackd_findings,
            detector_source=args.detector_source,
        )
        _validate_output_destination(args.out, role="TT report output")
        report = build_tt_sounds_calibration_report(
            labels_csv=args.tt_sounds_labels,
            snippets_dir=args.tt_sounds_snippets,
            detector_source=args.detector_source,
            anchor_report_path=args.anchor_report,
            trackd_findings_path=args.trackd_findings,
            access_audit=pre_access_validation,
        )
        _write_observed_report_json(
            args.out,
            report,
            observer=observer,
            pre_access_validation=pre_access_validation,
            output_roots=[args.out],
            include_excluded=False,
        )
        layout = report["corpus_layout"]
        print(
            f"wrote {args.out} "
            f"(labels={layout['label_rows']}, "
            f"feature_usable={sum(report['feature_measurement']['usable_class_sizes'][key] for key in ('hit', 'bounce', 'background'))})"
        )
        return 0

    if args.raw_dir is None:
        raise ValueError("alignment mode requires --raw-dir")
    references = [
        {
            "clip_id": row[0], "video_path": row[1], "cv_export_path": row[2],
            "insights_path": row[3], "domain": row[4],
        }
        for row in args.reference
    ]
    viability = [
        {"clip_id": row[0], "video_path": row[1], "domain": row[2]}
        for row in args.viability
    ]
    excluded = [
        {
            "clip_id": row[0], "external_id": row[1], "domain": row[2],
            "status": "EXCLUDED_COMPARE_ONLY_ID", "reason": row[3],
            "metrics_reported": False,
        }
        for row in args.excluded_reference
    ]
    pre_access_validation = _validate_alignment_inputs_before_access(
        references, viability, excluded, args.detector_source
    )
    _validate_output_destination(args.out, role="alignment report output")
    _validate_output_destination(args.raw_dir, role="alignment raw output")
    report = build_measurement_report(
        references=references, viability_specs=viability,
        excluded_references=excluded, policy_incidents=args.policy_incident,
        raw_dir=args.raw_dir,
        access_audit=pre_access_validation,
        detector_source=args.detector_source,
    )
    _write_observed_report_json(
        args.out,
        report,
        observer=observer,
        pre_access_validation=pre_access_validation,
        output_roots=[args.out, args.raw_dir],
        include_excluded=True,
    )
    print(
        f"wrote {args.out} "
        f"(aligned={sum(item.get('status') == 'MEASURED_TEACHER_ALIGNMENT' for item in report['reference_clips'])}, "
        f"viability={len(report['viability_probes'])})"
    )
    return 0


def main() -> int:
    args = _parse_args()
    mode = (
        "tt_snippet_diagnostics"
        if any(
            value is not None
            for value in (
                args.tt_sounds_labels,
                args.tt_sounds_snippets,
                args.anchor_report,
                args.trackd_findings,
            )
        )
        else "alignment"
    )
    observer = RuntimeAccessObserver(mode=mode)
    with _runtime_access_observation(observer):
        return _main_observed(args, observer)


if __name__ == "__main__":
    raise SystemExit(main())
