#!/usr/bin/env python3
"""Fail closed on training-input provenance before a trainer reads input bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "training_input_gate_proof"
INPUT_MANIFEST_TYPE = "training_input_manifest"
DEFAULT_VALID_FOR_SECONDS = 900
CACHE_ROOT = Path("/cache")
SHA256_PATTERN_LENGTH = 64
COMPONENTS = {"BALL", "COURT", "EVENT", "PERSON", "REID"}
TRAIN_REFUSAL_STATES = {"BLOCKED", "DEFERRED_WITH_REASON", "QUARANTINED", "REJECTED"}
CONSUMER_TRACKS = {"A", "B", "C", "D", "E"}
FORBIDDEN_IDENTITY_POSTURES = {"compare_only", "protected", "quarantine"}
TRAINING_INTENT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\btrain(?:ing)?\s+on\b",
        r"\b(?:add|feed|consume|materialize|reuse)\b.{0,240}\b(?:train(?:ing)?[- ]pool|retrain\s+pool|pretrain\s+(?:rows|experiment)|fine[- ]?tun\w*)\b",
        r"\b(?:run|start|execute)\b.{0,180}\b(?:retrain|pretrain\s+experiment|fine[- ]?tun)\w*\b",
        r"\bgradient(?:[- ](?:update|step))?[- ]supervision\b",
        r"\b(?:backprop(?:agation)?|optimizer[- ]?step|loss[- ]?bearing\s+supervision)\b",
    )
)
MIN_SUBSTANTIVE_RULING_LENGTH = 32
DIRECT_REFUSAL_CODES = {
    "CACHE_ENTRY_FORBIDS_TRAINING",
    "CACHE_SHA256_MISMATCH",
    "LEDGER_ASSET_NOT_FOUND",
    "LEDGER_COMPONENT_NOT_AUTHORIZED",
    "LEDGER_PROVENANCE_FORBIDS_TRAINING",
    "LEDGER_QUEUE_NOT_AUTHORIZED",
    "LEDGER_STATE_FORBIDS_TRAINING",
    "PROVENANCE_MARKER_FORBIDS_TRAINING",
    "PROVENANCE_MARKER_INVALID",
}
FORBIDDEN_CACHE_TOKENS = (
    "COMPARE_ONLY",
    "NEVER_TRAIN",
    "QUARANTINED",
    "SHA256_MISMATCH",
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"only local schema references are supported: {reference}")
    current: Any = root_schema
    for token in reference[2:].split("/"):
        current = current[token.replace("~1", "/").replace("~0", "~")]
    if not isinstance(current, dict):
        raise ValueError(f"schema reference does not resolve to an object: {reference}")
    return current


def _schema_errors(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    location: str,
) -> list[str]:
    if "$ref" in schema:
        return _schema_errors(
            value,
            _resolve_ref(root_schema, schema["$ref"]),
            root_schema=root_schema,
            location=location,
        )

    errors: list[str] = []
    expected_type = schema.get("type")
    type_checks = {
        "array": lambda candidate: isinstance(candidate, list),
        "boolean": lambda candidate: isinstance(candidate, bool),
        "integer": lambda candidate: isinstance(candidate, int) and not isinstance(candidate, bool),
        "number": lambda candidate: isinstance(candidate, (int, float)) and not isinstance(candidate, bool),
        "object": lambda candidate: isinstance(candidate, dict),
        "string": lambda candidate: isinstance(candidate, str),
        "null": lambda candidate: candidate is None,
    }
    if expected_type is not None:
        allowed_types = [expected_type] if isinstance(expected_type, str) else expected_type
        if not any(type_checks[item](value) for item in allowed_types):
            return [f"{location}: expected type {allowed_types}, got {type(value).__name__}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{location}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location}: {value!r} is not in {schema['enum']!r}")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{location}: string is shorter than {schema['minLength']}")
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(pattern, value) is None:
            errors.append(f"{location}: value does not match {pattern!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{location}: value is below {schema['minimum']}")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{location}: expected at least {schema['minItems']} items")
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True) for item in value]
            if len(serialized) != len(set(serialized)):
                errors.append(f"{location}: items must be unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(
                    _schema_errors(
                        item,
                        item_schema,
                        root_schema=root_schema,
                        location=f"{location}[{index}]",
                    )
                )

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{location}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, child in value.items():
            if key in properties:
                errors.extend(
                    _schema_errors(
                        child,
                        properties[key],
                        root_schema=root_schema,
                        location=f"{location}.{key}",
                    )
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{location}: unexpected property {key!r}")
    return errors


def _queue_action_errors(action: Any, *, location: str) -> list[str]:
    if not isinstance(action, dict):
        return [f"{location}: queued disposition must be an object"]
    errors: list[str] = []
    track = action.get("consumer_track")
    if track not in CONSUMER_TRACKS:
        errors.append(f"{location}.consumer_track: expected one of {sorted(CONSUMER_TRACKS)}")
    if not isinstance(action.get("training_intent"), bool):
        errors.append(f"{location}.training_intent: boolean structural declaration is required")
    for key in ("next_queue_action", "consumer_evidence"):
        value = action.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{location}.{key}: non-empty string is required")
    return errors


def disposition_violations(ledger: dict[str, Any]) -> list[str]:
    """Return one or more asset-addressed errors for every invalid queue disposition."""
    violations: list[str] = []
    assets = ledger.get("assets", [])
    if not isinstance(assets, list):
        return violations
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            continue
        asset_id = asset.get("asset_id", f"index_{index}")
        location = f"asset {asset_id} disposition"
        disposition = asset.get("disposition")
        if not isinstance(disposition, dict):
            violations.append(f"{location}: missing valid disposition block")
            continue
        is_ruled_out = "not_usable_because" in disposition
        has_queue_field = any(
            key in disposition
            for key in ("consumer_track", "next_queue_action", "consumer_evidence")
        )
        if is_ruled_out and has_queue_field:
            violations.append(f"{location}: cannot be both queued and ruled out")
            continue
        if is_ruled_out:
            reason = disposition.get("not_usable_because")
            if not isinstance(reason, str) or not reason.strip():
                violations.append(f"{location}.not_usable_because: non-empty string is required")
            elif len(reason.strip()) < MIN_SUBSTANTIVE_RULING_LENGTH or ":" not in reason:
                violations.append(
                    f"{location}.not_usable_because: substantive CODE: explanation is required"
                )
            if "secondary_queue_actions" in disposition:
                violations.append(f"{location}: ruled-out disposition cannot have secondary queue actions")
            continue
        if not has_queue_field:
            violations.append(f"{location}: must queue a consumer or provide not_usable_because")
            continue
        violations.extend(_queue_action_errors(disposition, location=location))
        secondary = disposition.get("secondary_queue_actions", [])
        if not isinstance(secondary, list):
            violations.append(f"{location}.secondary_queue_actions: must be an array")
            continue
        for secondary_index, action in enumerate(secondary):
            violations.extend(
                _queue_action_errors(
                    action,
                    location=f"{location}.secondary_queue_actions[{secondary_index}]",
                )
            )
    return sorted(violations)


def _queued_actions(asset: dict[str, Any]) -> list[dict[str, Any]]:
    disposition = asset.get("disposition", {})
    if not isinstance(disposition, dict) or "not_usable_because" in disposition:
        return []
    actions = [disposition]
    secondary = disposition.get("secondary_queue_actions", [])
    if isinstance(secondary, list):
        actions.extend(action for action in secondary if isinstance(action, dict))
    return actions


def _has_training_language(action: dict[str, Any]) -> bool:
    text = " ".join(str(action.get("next_queue_action", "")).casefold().split())
    return any(pattern.search(text) for pattern in TRAINING_INTENT_PATTERNS)


def _has_training_intent(action: dict[str, Any]) -> bool:
    return action.get("training_intent") is True or _has_training_language(action)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _canonical_family(value: str) -> str:
    """Collapse count-qualified partition entries to their ledger lineage family."""
    return value.split(":", 1)[0].strip().casefold()


def _normalized_path(value: str, repo_root: Path) -> Path | None:
    if not value or any(marker in value for marker in ("$", "*", "?", "{", "}")):
        return None
    if "://" in value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def path_registered_to_asset(
    asset: dict[str, Any],
    candidate: Path,
    *,
    repo_root: Path,
) -> bool:
    """Return whether a resolved input path is inside the asset's recorded path set."""
    resolved_candidate = candidate.resolve(strict=False)
    registered_values = [
        entry.get("path")
        for entry in asset.get("paths", [])
        if isinstance(entry, dict)
    ]
    registered_values.extend(
        binding.get("path")
        for binding in asset.get("immutable_hashes", [])
        if isinstance(binding, dict)
    )
    registered_values.extend(
        subset.get("selector_path")
        for subset in asset.get("protection", {}).get("clean_subsets", [])
        if isinstance(subset, dict)
    )
    for value in registered_values:
        if not isinstance(value, str):
            continue
        registered = _normalized_path(value, repo_root)
        if registered is not None and (
            resolved_candidate == registered
            or _path_contains(registered, resolved_candidate)
        ):
            return True
    return False


def recorded_sha256s_for_asset_path(
    asset: dict[str, Any],
    candidate: Path,
    *,
    repo_root: Path,
) -> set[str]:
    """Return exact SHA-256 bindings recorded for one resolved asset path."""
    resolved_candidate = candidate.resolve(strict=False)
    return {
        str(binding["digest"]).casefold()
        for binding in asset.get("immutable_hashes", [])
        if isinstance(binding, dict)
        and binding.get("algorithm") == "sha256"
        and isinstance(binding.get("digest"), str)
        and isinstance(binding.get("path"), str)
        and _normalized_path(binding["path"], repo_root) == resolved_candidate
    }


def training_queue_authorization_errors(
    asset: dict[str, Any],
    *,
    source_id: str | None = None,
    component: str | None = None,
) -> list[tuple[str, str]]:
    """Return typed reasons why a ledger row does not authorize training use."""
    asset_id = str(asset.get("asset_id", "<unknown>"))
    errors: list[tuple[str, str]] = []
    state = asset.get("state")
    if state in TRAIN_REFUSAL_STATES:
        errors.append(
            (
                "LEDGER_STATE_FORBIDS_TRAINING",
                f"asset {asset_id} has training-refusal state {state}",
            )
        )

    protection = asset.get("protection", {})
    if not isinstance(protection, dict):
        protection = {}
    if protection.get("trainer_forbidden") is True:
        errors.append(
            (
                "LEDGER_PROVENANCE_FORBIDS_TRAINING",
                f"asset {asset_id} is marked trainer_forbidden",
            )
        )

    training_actions = [
        action for action in _queued_actions(asset) if _has_training_intent(action)
    ]
    if not training_actions:
        errors.append(
            (
                "LEDGER_QUEUE_NOT_AUTHORIZED",
                f"asset {asset_id} has no queue-authorized training disposition",
            )
        )

    forbidden_identities = {
        identity.get("identity")
        for identity in protection.get("identities", [])
        if isinstance(identity, dict)
        and identity.get("posture") in FORBIDDEN_IDENTITY_POSTURES
        and isinstance(identity.get("identity"), str)
    }
    training_allowed_ids = {
        value
        for value in protection.get("training_allowed_ids", [])
        if isinstance(value, str)
    }
    if source_id is not None and source_id in forbidden_identities:
        errors.append(
            (
                "LEDGER_PROVENANCE_FORBIDS_TRAINING",
                f"source {source_id} has a training-refusal provenance posture in asset {asset_id}",
            )
        )
    elif forbidden_identities and source_id is None:
        errors.append(
            (
                "LEDGER_PROVENANCE_FORBIDS_TRAINING",
                f"asset {asset_id} mixes training-refusal identities and requires an explicit source_id",
            )
        )
    if training_allowed_ids and source_id is not None and source_id not in training_allowed_ids:
        errors.append(
            (
                "LEDGER_PROVENANCE_FORBIDS_TRAINING",
                f"source {source_id} is absent from asset {asset_id}'s training allowlist",
            )
        )

    if component is not None:
        ruling = asset.get("rights", {}).get("component_rulings", {}).get(component)
        if not isinstance(ruling, dict):
            errors.append(
                (
                    "LEDGER_COMPONENT_NOT_AUTHORIZED",
                    f"asset {asset_id} has no {component} component ruling",
                )
            )
        elif ruling.get("decision") != "ALLOW":
            errors.append(
                (
                    "LEDGER_COMPONENT_NOT_AUTHORIZED",
                    f"asset {asset_id} has {component}={ruling.get('decision')}",
                )
            )
    return sorted(set(errors))


def forbidden_ledger_sha256s(ledger: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Index recorded content hashes whose ledger row refuses unscoped training."""
    reasons_by_digest: dict[str, set[str]] = {}
    for asset in ledger.get("assets", []):
        if not isinstance(asset, dict):
            continue
        protection = asset.get("protection", {})
        if not isinstance(protection, dict):
            protection = {}
        has_refusal_identity = any(
            isinstance(identity, dict)
            and identity.get("posture") in FORBIDDEN_IDENTITY_POSTURES
            for identity in protection.get("identities", [])
        )
        refuses = (
            asset.get("state") in TRAIN_REFUSAL_STATES
            or protection.get("trainer_forbidden") is True
            or has_refusal_identity
        )
        if not refuses:
            continue
        asset_id = str(asset.get("asset_id", "<unknown>"))
        for binding in asset.get("immutable_hashes", []):
            if (
                isinstance(binding, dict)
                and binding.get("algorithm") == "sha256"
                and isinstance(binding.get("digest"), str)
            ):
                reasons_by_digest.setdefault(
                    binding["digest"].casefold(), set()
                ).add(f"ledger asset {asset_id}")
    return {
        digest: tuple(sorted(reasons))
        for digest, reasons in reasons_by_digest.items()
    }


def validate_ledger(ledger: dict[str, Any]) -> list[str]:
    schema = ledger.get("schema")
    if not isinstance(schema, dict):
        return ["$.schema: embedded JSON schema is required"]
    errors = _schema_errors(ledger, schema, root_schema=schema, location="$")
    assets = ledger.get("assets")
    if not isinstance(assets, list):
        return errors

    try:
        _parse_utc(ledger["generated_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"$.generated_utc: {exc}")

    asset_ids = [asset.get("asset_id") for asset in assets if isinstance(asset, dict)]
    duplicates = sorted(asset_id for asset_id, count in Counter(asset_ids).items() if count > 1)
    if duplicates:
        errors.append(f"$.assets: duplicate asset IDs: {duplicates}")

    errors.extend(disposition_violations(ledger))

    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            continue
        location = f"$.assets[{index}]"
        try:
            _parse_utc(asset["acquired_utc"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{location}.acquired_utc: {exc}")
        component_rulings = asset.get("rights", {}).get("component_rulings", {})
        missing_components = sorted(COMPONENTS - set(component_rulings))
        if missing_components:
            errors.append(f"{location}.rights.component_rulings: missing {missing_components}")
        state = asset.get("state")
        consumers = asset.get("consumers", [])
        if state == "CONSUMED" and not consumers:
            errors.append(f"{location}: CONSUMED assets require at least one consumer")
        if state == "READY" and asset.get("state_reason") != "ready_for_named_consumer":
            errors.append(f"{location}: READY state_reason must be 'ready_for_named_consumer'")
        if asset.get("protection", {}).get("trainer_forbidden") and not asset.get("protection", {}).get(
            "identities"
        ):
            errors.append(f"{location}.protection: trainer-forbidden assets require an identity")
        identities = {
            identity["identity"]
            for identity in asset.get("protection", {}).get("identities", [])
            if isinstance(identity, dict) and isinstance(identity.get("identity"), str)
        }
        binding_paths = {
            binding["path"]
            for binding in asset.get("immutable_hashes", [])
            if isinstance(binding, dict) and isinstance(binding.get("path"), str)
        }
        for subset_index, subset in enumerate(asset.get("protection", {}).get("clean_subsets", [])):
            subset_location = f"{location}.protection.clean_subsets[{subset_index}]"
            if subset.get("selector_path") not in binding_paths:
                errors.append(f"{subset_location}: selector_path must be an immutable hash binding")
            missing_exclusions = identities - set(subset.get("excluded_identities", []))
            if missing_exclusions:
                errors.append(
                    f"{subset_location}: protected identities not excluded: {sorted(missing_exclusions)}"
                )
            subset_train = {_canonical_family(value) for value in subset.get("train_families", [])}
            subset_holdout = {_canonical_family(value) for value in subset.get("holdout_families", [])}
            subset_overlap = sorted((subset_train & subset_holdout) - {""})
            if subset_overlap:
                errors.append(f"{subset_location}: train/holdout overlap: {subset_overlap}")
    return sorted(errors)


class GateProofError(ValueError):
    """Typed downstream refusal for a missing, stale, failed, or changed proof."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_gate_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path, *, _visited: set[Path] | None = None) -> tuple[str, str]:
    """Hash one file's bytes or one directory's stable names-and-content tree."""
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return _sha256_file(resolved), "file_bytes"
    if not resolved.is_dir():
        raise ValueError(f"unsupported training input type: {resolved}")

    visited = set() if _visited is None else _visited
    if resolved in visited:
        raise ValueError(f"directory cycle while hashing training input: {resolved}")
    visited.add(resolved)
    digest = hashlib.sha256()
    digest.update(b"training_input_directory_tree_v1\0")
    for child in sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()):
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        if child.is_symlink():
            target = child.resolve(strict=True)
            digest.update(b"symlink\0")
            digest.update(str(target).encode("utf-8"))
            digest.update(b"\0")
            target_digest, target_kind = _sha256_path(target, _visited=visited)
            digest.update(target_kind.encode("ascii"))
            digest.update(b"\0")
            digest.update(target_digest.encode("ascii"))
        elif child.is_file():
            digest.update(b"file\0")
            digest.update(_sha256_file(child).encode("ascii"))
        elif child.is_dir():
            digest.update(b"directory\0")
        else:
            raise ValueError(f"unsupported entry in training input tree: {child}")
        digest.update(b"\0")
    visited.remove(resolved)
    return digest.hexdigest(), "directory_tree_v1"


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _proof_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("proof_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _repo_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise ValueError("repository HEAD SHA is unavailable")
    return value


def _absolute_path(value: str, *, repo_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.absolute()


def _is_beneath(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _is_cache_path(candidate: Path, resolved: Path) -> bool:
    return _is_beneath(CACHE_ROOT, candidate) or _is_beneath(CACHE_ROOT, resolved)


def _reason(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _add_reason(reasons: list[dict[str, str]], code: str, detail: str) -> None:
    candidate = _reason(code, detail)
    if candidate not in reasons:
        reasons.append(candidate)


def _cache_tokens(entry: dict[str, Any]) -> set[str]:
    values: list[Any] = [entry.get("status")]
    flags = entry.get("flags", [])
    if isinstance(flags, list):
        values.extend(flags)
    return {
        str(value).strip().upper().replace("-", "_")
        for value in values
        if value is not None
    }


def _cache_entry_forbids_training(entry: dict[str, Any]) -> bool:
    return any(
        marker in token
        for token in _cache_tokens(entry)
        for marker in FORBIDDEN_CACHE_TOKENS
    )


def _cache_entry_has_sha_mismatch(entry: dict[str, Any]) -> bool:
    tokens = _cache_tokens(entry)
    expected = entry.get("expected_sha256")
    actual = entry.get("sha256")
    return (
        any("SHA256_MISMATCH" in token for token in tokens)
        or (
            isinstance(expected, str)
            and isinstance(actual, str)
            and expected.casefold() != actual.casefold()
        )
    )


def _iter_cache_entries(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and any(
            key in value
            for key in ("flags", "status", "sha256", "expected_sha256")
        ):
            yield value
        for child in value.values():
            yield from _iter_cache_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_cache_entries(child)


def _cache_entry_match(
    entries: Sequence[dict[str, Any]],
    *,
    requested: Path,
    resolved: Path,
    source_id: str | None,
) -> dict[str, Any] | None:
    path_matches: list[dict[str, Any]] = []
    for entry in entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        entry_path = Path(raw_path)
        if (
            requested == entry_path
            or resolved == entry_path
            or _is_beneath(entry_path, requested)
            or _is_beneath(entry_path, resolved)
        ):
            path_matches.append(entry)
    if len(path_matches) == 1:
        return path_matches[0]
    if len(path_matches) > 1:
        path_matches.sort(key=lambda entry: len(str(entry["path"])), reverse=True)
        return path_matches[0]
    if source_id is not None:
        id_matches = [entry for entry in entries if entry.get("id") == source_id]
        if len(id_matches) == 1:
            return id_matches[0]
    return None


def _forbidden_cache_sha256s(
    entries: Sequence[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    reasons: dict[str, set[str]] = {}
    for entry in entries:
        if not _cache_entry_forbids_training(entry):
            continue
        entry_id = str(entry.get("id", entry.get("path", "<unknown>")))
        for key in ("sha256", "expected_sha256", "sha256_provenance_only"):
            digest = entry.get(key)
            if isinstance(digest, str) and len(digest) == SHA256_PATTERN_LENGTH:
                reasons.setdefault(digest.casefold(), set()).add(
                    f"cache entry {entry_id} field {key}"
                )
    return {
        digest: tuple(sorted(values)) for digest, values in reasons.items()
    }


def _find_usage_marker(
    candidate: Path,
    *,
    repo_root: Path,
    explicit: str | None,
) -> tuple[Path | None, str | None]:
    if explicit is not None:
        marker = _absolute_path(explicit, repo_root=repo_root)
        if not marker.is_file():
            return None, f"declared provenance marker is absent: {marker}"
        return marker, None

    resolved = candidate.resolve(strict=False)
    start = resolved if resolved.is_dir() else resolved.parent
    stop_roots = {repo_root.resolve(strict=False), CACHE_ROOT}
    current = start
    while True:
        marker = current / "USAGE_POSTURE.json"
        if marker.is_file():
            return marker, None
        if current in stop_roots or current.parent == current:
            break
        current = current.parent
    return None, None


def _marker_policy_reasons(marker: dict[str, Any]) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    posture = marker.get("posture")
    if posture is not None and posture != "train":
        violations.append(
            (
                "PROVENANCE_MARKER_FORBIDS_TRAINING",
                f"provenance marker posture is {posture!r}, not 'train'",
            )
        )
    if marker.get("never_train") is True:
        violations.append(
            (
                "PROVENANCE_MARKER_FORBIDS_TRAINING",
                "provenance marker sets never_train=true",
            )
        )
    if marker.get("trainer_forbidden") is True:
        violations.append(
            (
                "PROVENANCE_MARKER_FORBIDS_TRAINING",
                "provenance marker sets trainer_forbidden=true",
            )
        )
    if marker.get("training_eligible") is False:
        violations.append(
            (
                "PROVENANCE_MARKER_FORBIDS_TRAINING",
                "provenance marker sets training_eligible=false",
            )
        )
    usage_posture = marker.get("usage_posture")
    if isinstance(usage_posture, str) and any(
        token in usage_posture.casefold()
        for token in (
            "compare-only",
            "compare_only",
            "eval-only",
            "frozen-protected",
            "never train",
            "never-train",
            "never trainable",
            "quarantin",
        )
    ):
        violations.append(
            (
                "PROVENANCE_MARKER_FORBIDS_TRAINING",
                "usage_posture text carries a training-refusal marker",
            )
        )
    if _cache_entry_forbids_training(marker):
        violations.append(
            (
                "PROVENANCE_MARKER_FORBIDS_TRAINING",
                "provenance marker status or flags refuse training",
            )
        )
    return sorted(set(violations))


def _source_record(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {"path": str(path.absolute()), "sha256": _sha256_file(path)}


def _manifest_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("artifact_type") != INPUT_MANIFEST_TYPE:
        errors.append(f"artifact_type must be {INPUT_MANIFEST_TYPE!r}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("inputs must be a non-empty array")
        return errors
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            errors.append(f"inputs[{index}] must be an object")
            continue
        for key in ("path", "asset_id"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                errors.append(f"inputs[{index}].{key} must be a non-empty string")
        digest = item.get("sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != SHA256_PATTERN_LENGTH
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            errors.append(f"inputs[{index}].sha256 must be a 64-character hex digest")
        for key in ("source_id", "component", "provenance_marker"):
            if key in item and (
                not isinstance(item[key], str) or not item[key].strip()
            ):
                errors.append(f"inputs[{index}].{key} must be a non-empty string")
    return errors


def verify_training_inputs(
    *,
    input_manifest_path: Path,
    ledger_path: Path,
    repo_root: Path,
    proof_path: Path,
    cache_manifest_path: Path | None = None,
    now: datetime | None = None,
    valid_for_seconds: int = DEFAULT_VALID_FOR_SECONDS,
) -> dict[str, Any]:
    """Evaluate every intended input and always write a pass/fail proof artifact."""
    generated_at = (now or _utc_now()).astimezone(timezone.utc)
    repo_root = repo_root.resolve()
    input_manifest_path = input_manifest_path.absolute()
    ledger_path = ledger_path.absolute()
    cache_manifest_path = (
        cache_manifest_path.absolute() if cache_manifest_path is not None else None
    )
    global_reasons: list[dict[str, str]] = []

    input_manifest: dict[str, Any] = {}
    ledger: dict[str, Any] = {}
    cache_manifest: dict[str, Any] | None = None
    try:
        input_manifest = load_json(input_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _add_reason(
            global_reasons,
            "INPUT_MANIFEST_INVALID",
            str(exc),
        )
    for detail in _manifest_errors(input_manifest):
        _add_reason(global_reasons, "INPUT_MANIFEST_INVALID", detail)

    try:
        ledger = load_json(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _add_reason(global_reasons, "LEDGER_INVALID", str(exc))
    for detail in validate_ledger(ledger) if ledger else []:
        _add_reason(global_reasons, "LEDGER_INVALID", detail)

    if cache_manifest_path is not None:
        try:
            cache_manifest = load_json(cache_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add_reason(global_reasons, "CACHE_MANIFEST_INVALID", str(exc))

    cache_entries = list(_iter_cache_entries(cache_manifest or {}))
    forbidden_hashes = {
        digest: set(reasons)
        for digest, reasons in forbidden_ledger_sha256s(ledger).items()
    }
    for digest, reasons in _forbidden_cache_sha256s(cache_entries).items():
        forbidden_hashes.setdefault(digest, set()).update(reasons)
    assets = {
        asset.get("asset_id"): asset
        for asset in ledger.get("assets", [])
        if isinstance(asset, dict) and isinstance(asset.get("asset_id"), str)
    }

    verdicts: list[dict[str, Any]] = []
    items = input_manifest.get("inputs", [])
    if not isinstance(items, list):
        items = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        asset_id = item.get("asset_id")
        if not isinstance(raw_path, str) or not isinstance(asset_id, str):
            continue
        requested = _absolute_path(raw_path, repo_root=repo_root)
        resolved = requested.resolve(strict=False)
        source_id = item.get("source_id")
        component = item.get("component")
        reasons: list[dict[str, str]] = []
        cache_sha256s: dict[str, str] = {}
        cache_entry = _cache_entry_match(
            cache_entries,
            requested=requested,
            resolved=resolved,
            source_id=source_id if isinstance(source_id, str) else None,
        )

        under_cache = _is_cache_path(requested, resolved)
        if under_cache and cache_manifest is None:
            _add_reason(
                reasons,
                "CACHE_MANIFEST_REQUIRED",
                "an input under /cache requires --cache-manifest",
            )
        if cache_manifest is not None and under_cache and cache_entry is None:
            _add_reason(
                reasons,
                "CACHE_ENTRY_NOT_FOUND",
                f"no cache manifest entry binds {requested}",
            )
        if cache_entry is not None:
            for key in ("expected_sha256", "sha256"):
                value = cache_entry.get(key)
                if isinstance(value, str):
                    cache_sha256s[key] = value
            if _cache_entry_forbids_training(cache_entry):
                _add_reason(
                    reasons,
                    "CACHE_ENTRY_FORBIDS_TRAINING",
                    f"cache entry {cache_entry.get('id', cache_entry.get('path'))} "
                    f"has status/flags {sorted(_cache_tokens(cache_entry))}",
                )
            if _cache_entry_has_sha_mismatch(cache_entry):
                _add_reason(
                    reasons,
                    "CACHE_SHA256_MISMATCH",
                    f"cache entry {cache_entry.get('id', cache_entry.get('path'))} "
                    "records contradictory SHA-256 values",
                )

        asset = assets.get(asset_id)
        if asset is None:
            _add_reason(
                reasons,
                "LEDGER_ASSET_NOT_FOUND",
                f"asset {asset_id} is absent from the data ledger",
            )
        else:
            for code, detail in training_queue_authorization_errors(
                asset,
                source_id=source_id if isinstance(source_id, str) else None,
                component=component if isinstance(component, str) else None,
            ):
                _add_reason(reasons, code, detail)
            if cache_entry is None and not path_registered_to_asset(
                asset,
                resolved,
                repo_root=repo_root,
            ):
                _add_reason(
                    reasons,
                    "LEDGER_PATH_UNBOUND",
                    f"{resolved} is outside asset {asset_id}'s registered paths",
                )
            if (
                cache_entry is not None
                and isinstance(source_id, str)
                and isinstance(cache_entry.get("id"), str)
                and cache_entry["id"] != source_id
            ):
                _add_reason(
                    reasons,
                    "CACHE_SOURCE_ID_MISMATCH",
                    f"declared source_id {source_id} differs from cache entry "
                    f"{cache_entry['id']}",
                )

        marker_path, marker_error = _find_usage_marker(
            resolved,
            repo_root=repo_root,
            explicit=(
                item.get("provenance_marker")
                if isinstance(item.get("provenance_marker"), str)
                else None
            ),
        )
        marker_sha256: str | None = None
        if marker_error is not None:
            _add_reason(reasons, "PROVENANCE_MARKER_INVALID", marker_error)
        elif marker_path is not None:
            try:
                marker_payload = load_json(marker_path)
                marker_sha256 = _sha256_file(marker_path)
                for code, detail in _marker_policy_reasons(marker_payload):
                    _add_reason(reasons, code, detail)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                _add_reason(
                    reasons,
                    "PROVENANCE_MARKER_INVALID",
                    f"{marker_path}: {exc}",
                )

        actual_sha256: str | None = None
        sha256_kind: str | None = None
        input_bytes_read = False
        direct_refusal = any(
            reason["code"] in DIRECT_REFUSAL_CODES for reason in reasons
        )
        if not direct_refusal:
            if not requested.exists():
                _add_reason(
                    reasons,
                    "INPUT_NOT_FOUND",
                    f"training input is absent: {requested}",
                )
            else:
                try:
                    actual_sha256, sha256_kind = _sha256_path(requested)
                    input_bytes_read = True
                except (OSError, ValueError) as exc:
                    _add_reason(reasons, "INPUT_HASH_FAILED", str(exc))
                if actual_sha256 is not None:
                    recorded = item.get("sha256")
                    if (
                        isinstance(recorded, str)
                        and recorded.casefold() != actual_sha256
                    ):
                        _add_reason(
                            reasons,
                            "RECORDED_SHA256_MISMATCH",
                            f"input manifest={recorded.casefold()} current={actual_sha256}",
                        )
                    if asset is not None:
                        ledger_digests = recorded_sha256s_for_asset_path(
                            asset,
                            resolved,
                            repo_root=repo_root,
                        )
                        if ledger_digests and actual_sha256 not in ledger_digests:
                            _add_reason(
                                reasons,
                                "LEDGER_SHA256_MISMATCH",
                                f"ledger={sorted(ledger_digests)} current={actual_sha256}",
                            )
                    if cache_entry is not None:
                        cache_expected = cache_entry.get(
                            "expected_sha256", cache_entry.get("sha256")
                        )
                        if (
                            isinstance(cache_expected, str)
                            and cache_expected.casefold() != actual_sha256
                        ):
                            _add_reason(
                                reasons,
                                "CACHE_FILE_SHA256_MISMATCH",
                                f"cache manifest={cache_expected.casefold()} "
                                f"current={actual_sha256}",
                            )
                    if actual_sha256 in forbidden_hashes:
                        _add_reason(
                            reasons,
                            "CONTENT_SHA256_FORBIDS_TRAINING",
                            "content SHA-256 matches a training-refusal record: "
                            + "; ".join(sorted(forbidden_hashes[actual_sha256])),
                        )

        reasons.sort(key=lambda reason: (reason["code"], reason["detail"]))
        verdicts.append(
            {
                "index": index,
                "requested_path": str(requested),
                "resolved_path": str(resolved),
                "asset_id": asset_id,
                "source_id": source_id if isinstance(source_id, str) else None,
                "component": component if isinstance(component, str) else None,
                "verdict": "PASS" if not reasons else "FAIL",
                "reasons": reasons,
                "actual_sha256": actual_sha256,
                "sha256_kind": sha256_kind,
                "input_manifest_sha256": (
                    item.get("sha256").casefold()
                    if isinstance(item.get("sha256"), str)
                    else None
                ),
                "input_bytes_read_for_integrity": input_bytes_read,
                "cache_entry_id": (
                    cache_entry.get("id") if cache_entry is not None else None
                ),
                "cache_entry_sha256s": cache_sha256s,
                "provenance_marker_path": (
                    str(marker_path) if marker_path is not None else None
                ),
                "provenance_marker_sha256": marker_sha256,
            }
        )

    source_manifests: dict[str, dict[str, str] | None] = {}
    for key, path in (
        ("training_inputs", input_manifest_path),
        ("data_ledger", ledger_path),
        ("cache_manifest", cache_manifest_path),
    ):
        try:
            source_manifests[key] = _source_record(path)
        except OSError as exc:
            source_manifests[key] = None
            _add_reason(global_reasons, "SOURCE_MANIFEST_UNREADABLE", f"{key}: {exc}")

    status = (
        "PASS"
        if not global_reasons
        and verdicts
        and all(verdict["verdict"] == "PASS" for verdict in verdicts)
        else "FAIL"
    )
    proof: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "generated_at_utc": _utc_text(generated_at),
        "expires_at_utc": _utc_text(
            generated_at + timedelta(seconds=valid_for_seconds)
        ),
        "repo_head_sha": _repo_head(repo_root),
        "repo_root": str(repo_root),
        "manifest_sha256s": {
            key: value["sha256"] if value is not None else None
            for key, value in source_manifests.items()
        },
        "source_manifests": source_manifests,
        "inputs": verdicts,
        "global_reasons": sorted(
            global_reasons, key=lambda reason: (reason["code"], reason["detail"])
        ),
    }
    proof["proof_sha256"] = _proof_sha256(proof)
    _atomic_write_json(proof_path, proof)
    return proof


def assert_gate_proof(
    proof_path: Path | None,
    *,
    repo_root: Path,
    required_input_paths: Sequence[Path] = (),
    max_age_seconds: int = DEFAULT_VALID_FOR_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate proof integrity, freshness, source manifests, HEAD, and input bytes."""
    if proof_path is None:
        raise GateProofError(
            "GATE_PROOF_MISSING",
            "a passing --gate-proof is required before input read",
        )
    proof_path = proof_path.absolute()
    if not proof_path.is_file():
        raise GateProofError(
            "GATE_PROOF_MISSING",
            f"gate proof is absent before input read: {proof_path}",
        )
    try:
        proof = load_json(proof_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GateProofError("GATE_PROOF_INVALID", str(exc)) from exc

    recorded_proof_sha = proof.get("proof_sha256")
    current_proof_sha = _proof_sha256(proof)
    if recorded_proof_sha != current_proof_sha:
        raise GateProofError(
            "GATE_PROOF_INTEGRITY_MISMATCH",
            f"recorded={recorded_proof_sha} current={current_proof_sha}",
        )
    if (
        proof.get("schema_version") != SCHEMA_VERSION
        or proof.get("artifact_type") != ARTIFACT_TYPE
    ):
        raise GateProofError(
            "GATE_PROOF_SCHEMA_INVALID",
            "unrecognized schema_version or artifact_type",
        )
    if proof.get("status") != "PASS" or proof.get("global_reasons"):
        raise GateProofError(
            "GATE_PROOF_FAILED",
            "proof status is not PASS",
        )
    inputs = proof.get("inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or any(
            not isinstance(verdict, dict) or verdict.get("verdict") != "PASS"
            for verdict in inputs
        )
    ):
        raise GateProofError(
            "GATE_PROOF_FAILED",
            "proof does not contain only passing per-input verdicts",
        )

    current_time = (now or _utc_now()).astimezone(timezone.utc)
    try:
        generated_at = _parse_gate_utc(proof.get("generated_at_utc"))
        expires_at = _parse_gate_utc(proof.get("expires_at_utc"))
    except ValueError as exc:
        raise GateProofError("GATE_PROOF_SCHEMA_INVALID", str(exc)) from exc
    age_seconds = (current_time - generated_at).total_seconds()
    if age_seconds < -5:
        raise GateProofError(
            "GATE_PROOF_CLOCK_SKEW",
            f"proof timestamp is {-age_seconds:.3f}s in the future",
        )
    if age_seconds > max_age_seconds or current_time > expires_at:
        raise GateProofError(
            "GATE_PROOF_STALE",
            f"proof age is {age_seconds:.3f}s; limit is {max_age_seconds}s",
        )

    repo_root = repo_root.resolve()
    current_head = _repo_head(repo_root)
    if proof.get("repo_head_sha") != current_head:
        raise GateProofError(
            "GATE_PROOF_HEAD_MISMATCH",
            f"proof={proof.get('repo_head_sha')} current={current_head}",
        )

    sources = proof.get("source_manifests")
    if not isinstance(sources, dict):
        raise GateProofError(
            "GATE_PROOF_SCHEMA_INVALID",
            "source_manifests must be an object",
        )
    for name, source in sources.items():
        if source is None:
            continue
        if not isinstance(source, dict):
            raise GateProofError(
                "GATE_PROOF_SCHEMA_INVALID",
                f"source_manifests.{name} must be an object or null",
            )
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            raise GateProofError(
                "GATE_PROOF_SOURCE_CHANGED",
                f"source manifest is absent: {path}",
            )
        current_sha = _sha256_file(path)
        if current_sha != source.get("sha256"):
            raise GateProofError(
                "GATE_PROOF_SOURCE_CHANGED",
                f"{name} proof={source.get('sha256')} current={current_sha}",
            )

    ledger_source = sources.get("data_ledger")
    canonical_ledger = (repo_root / "runs/manager/data_ledger.json").resolve()
    if (
        not isinstance(ledger_source, dict)
        or Path(str(ledger_source.get("path", ""))).resolve(strict=False)
        != canonical_ledger
    ):
        raise GateProofError(
            "GATE_PROOF_LEDGER_UNTRUSTED",
            f"proof must bind the canonical data ledger: {canonical_ledger}",
        )

    proof_paths: set[str] = set()
    proof_resolved_paths: set[str] = set()
    for verdict in inputs:
        requested = Path(str(verdict.get("requested_path", "")))
        resolved = Path(str(verdict.get("resolved_path", "")))
        proof_paths.add(str(requested))
        proof_resolved_paths.add(str(resolved))
        recorded_sha = verdict.get("actual_sha256")
        if not isinstance(recorded_sha, str):
            raise GateProofError(
                "GATE_PROOF_INPUT_HASH_MISSING",
                f"passing input lacks actual_sha256: {requested}",
            )
        if not requested.exists():
            raise GateProofError(
                "GATE_PROOF_INPUT_CHANGED",
                f"input is absent: {requested}",
            )
        try:
            current_sha, current_kind = _sha256_path(requested)
        except (OSError, ValueError) as exc:
            raise GateProofError("GATE_PROOF_INPUT_CHANGED", str(exc)) from exc
        if (
            current_sha != recorded_sha
            or current_kind != verdict.get("sha256_kind")
        ):
            raise GateProofError(
                "GATE_PROOF_INPUT_CHANGED",
                f"{requested} proof={recorded_sha}/{verdict.get('sha256_kind')} "
                f"current={current_sha}/{current_kind}",
            )

    for required in required_input_paths:
        requested = (
            required if required.is_absolute() else repo_root / required
        ).absolute()
        resolved = requested.resolve(strict=False)
        if (
            str(requested) not in proof_paths
            and str(resolved) not in proof_resolved_paths
        ):
            raise GateProofError(
                "GATE_PROOF_INPUT_MISSING",
                f"trainer input is absent from the proof: {requested}",
            )

    uses_cache = any(
        _is_cache_path(
            Path(str(verdict.get("requested_path", ""))),
            Path(str(verdict.get("resolved_path", ""))),
        )
        for verdict in inputs
    )
    cache_source = sources.get("cache_manifest")
    if uses_cache:
        canonical_cache_manifest = CACHE_ROOT / "CACHE_MANIFEST.json"
        if (
            not isinstance(cache_source, dict)
            or Path(str(cache_source.get("path", ""))).resolve(strict=False)
            != canonical_cache_manifest
        ):
            raise GateProofError(
                "GATE_PROOF_CACHE_MANIFEST_UNTRUSTED",
                "a /cache proof must bind /cache/CACHE_MANIFEST.json",
            )

    input_source = sources.get("training_inputs")
    if not isinstance(input_source, dict):
        raise GateProofError(
            "GATE_PROOF_SCHEMA_INVALID",
            "passing proof lacks the intended-input source manifest",
        )
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix="training_gate_recheck_",
        suffix=".json",
    )
    os.close(temporary_fd)
    temporary_proof = Path(temporary_name)
    try:
        recomputed = verify_training_inputs(
            input_manifest_path=Path(str(input_source["path"])),
            ledger_path=canonical_ledger,
            cache_manifest_path=(
                Path(str(cache_source["path"]))
                if isinstance(cache_source, dict)
                else None
            ),
            repo_root=repo_root,
            proof_path=temporary_proof,
            now=current_time,
            valid_for_seconds=max_age_seconds,
        )
    finally:
        temporary_proof.unlink(missing_ok=True)
    if recomputed.get("status") != "PASS":
        reason_codes = sorted(
            {
                reason.get("code")
                for verdict in recomputed.get("inputs", [])
                if isinstance(verdict, dict)
                for reason in verdict.get("reasons", [])
                if isinstance(reason, dict) and isinstance(reason.get("code"), str)
            }
        )
        raise GateProofError(
            "GATE_PROOF_REVALIDATION_FAILED",
            f"current policy evaluation refused the inputs: {reason_codes}",
        )
    recomputed_by_path = {
        verdict["requested_path"]: verdict
        for verdict in recomputed["inputs"]
        if isinstance(verdict, dict)
    }
    for verdict in inputs:
        current_verdict = recomputed_by_path.get(verdict["requested_path"])
        if (
            current_verdict is None
            or current_verdict.get("actual_sha256") != verdict.get("actual_sha256")
            or current_verdict.get("sha256_kind") != verdict.get("sha256_kind")
        ):
            raise GateProofError(
                "GATE_PROOF_REVALIDATION_MISMATCH",
                f"current gate result differs for {verdict['requested_path']}",
            )
    return proof


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--gate-proof", type=Path, default=Path("gate_proof.json"))
    parser.add_argument(
        "--valid-for-seconds",
        type=int,
        default=DEFAULT_VALID_FOR_SECONDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.valid_for_seconds < 1:
        raise SystemExit("--valid-for-seconds must be positive")
    proof = verify_training_inputs(
        input_manifest_path=args.inputs,
        ledger_path=args.ledger,
        cache_manifest_path=args.cache_manifest,
        repo_root=args.repo_root,
        proof_path=args.gate_proof,
        valid_for_seconds=args.valid_for_seconds,
    )
    print(json.dumps(proof, sort_keys=True))
    return 0 if proof["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
