#!/usr/bin/env python3
"""Validate the data ledger, gate dispatch contracts, and report never-queued assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = Path("runs/manager/data_ledger.json")
DEFAULT_VIEW = Path("runs/manager/DATA_LEDGER.md")
COMPONENTS = {"BALL", "COURT", "EVENT", "PERSON", "REID"}
DATA_INPUT_ROLES = {"ground_truth", "teacher_supervision", "training_data"}
SHORT_VALUE_FLAGS = {"-c", "-d", "-i", "-l", "-m", "-o", "-t"}
TRAIN_REFUSAL_STATES = {"BLOCKED", "DEFERRED_WITH_REASON", "QUARANTINED", "REJECTED"}
TERMINAL_UNQUEUED_STATES = {
    "BLOCKED",
    "QUARANTINED",
    "CONSUMED",
    "REJECTED",
    "DEFERRED_WITH_REASON",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def sha_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def verify_hashes(ledger: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    for asset in ledger.get("assets", []):
        asset_id = asset["asset_id"]
        for binding in asset["immutable_hashes"]:
            path = repo_root / binding["path"]
            if not path.is_file():
                errors.append(f"{asset_id}: hash input is absent: {binding['path']}")
                continue
            actual = sha_digest(path, binding["algorithm"])
            if actual != binding["digest"]:
                errors.append(
                    f"{asset_id}: {binding['algorithm']} differs for {binding['path']}: "
                    f"ledger={binding['digest']} current={actual}"
                )
    return sorted(errors)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def never_queued_assets(ledger: dict[str, Any], *, as_of: datetime) -> list[dict[str, Any]]:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    report: list[dict[str, Any]] = []
    for asset in ledger["assets"]:
        age_hours = (as_of.astimezone(timezone.utc) - _parse_utc(asset["acquired_utc"])).total_seconds() / 3600
        counts = asset["counts"]
        has_payload = counts["byte_count"] > 0 or counts["label_count"] > 0
        has_disposition = asset["state"] in TERMINAL_UNQUEUED_STATES
        if age_hours > 24 and has_payload and not asset["consumers"] and not has_disposition:
            report.append(
                {
                    "acquired_utc": asset["acquired_utc"],
                    "age_hours": round(age_hours, 3),
                    "asset_id": asset["asset_id"],
                    "byte_count": counts["byte_count"],
                    "label_count": counts["label_count"],
                    "state": asset["state"],
                }
            )
    return sorted(report, key=lambda row: (row["acquired_utc"], row["asset_id"]))


def _asset_index(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {asset["asset_id"]: asset for asset in ledger["assets"]}


def _canonical_family(value: str) -> str:
    """Collapse count-qualified partition entries to their ledger lineage family."""
    return value.split(":", 1)[0].strip().casefold()


def _partition_families(asset: dict[str, Any]) -> tuple[set[str], set[str]]:
    partitions = asset["partitions"]
    train = {_canonical_family(value) for value in partitions["train"]}
    holdout = {
        _canonical_family(value)
        for value in (*partitions["val"], *partitions["test"])
    }
    return train - {""}, holdout - {""}


def _identity_aliases(identity: str) -> set[str]:
    aliases = {identity.strip().casefold()}
    aliases.update(
        token.casefold()
        for token in re.split(r"[/,:;]+", identity)
        if len(token) >= 8
    )
    return {alias for alias in aliases if alias}


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


def _command_pairs(argv: list[str]) -> tuple[list[tuple[str | None, str, int]], list[str]]:
    """Return every argv value as (flag, value, index), refusing opaque dash forms."""
    pairs: list[tuple[str | None, str, int]] = []
    errors: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            errors.append(
                f"dispatch.command.argv[{index}]: unsupported dash-prefixed form is ambiguous: {token}"
            )
        elif token.startswith("--"):
            if "=" in token:
                flag, value = token.split("=", 1)
                if re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", flag) is None or not value:
                    errors.append(
                        f"dispatch.command.argv[{index}]: unsupported dash-prefixed form is ambiguous: {token}"
                    )
                pairs.append((flag.casefold(), value, index))
            elif re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", token) is None:
                errors.append(
                    f"dispatch.command.argv[{index}]: unsupported dash-prefixed form is ambiguous: {token}"
                )
            elif index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                pairs.append((token.casefold(), argv[index + 1], index + 1))
                index += 1
            else:
                pairs.append((token.casefold(), "", index))
        elif token.startswith("-"):
            flag = token[:2].casefold()
            attached = token[2:]
            supported = flag in SHORT_VALUE_FLAGS
            if not supported:
                errors.append(
                    f"dispatch.command.argv[{index}]: unsupported dash-prefixed form is ambiguous: {token}"
                )
            if attached:
                value = attached[1:] if attached.startswith("=") else attached
                if not value:
                    errors.append(
                        f"dispatch.command.argv[{index}]: short option has no value: {token}"
                    )
                pairs.append((flag, value, index))
            elif index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                pairs.append((flag, argv[index + 1], index + 1))
                index += 1
            else:
                errors.append(
                    f"dispatch.command.argv[{index}]: short option has no value: {token}"
                )
                pairs.append((flag, "", index))
        else:
            pairs.append((None, token, index))
        index += 1
    return pairs, errors


def _flag_role(flag: str | None) -> str:
    normalized = (flag or "").replace("_", "-")
    if normalized in {"-o", "-l"} or any(
        marker in normalized for marker in ("output", "save", "result", "log")
    ):
        return "output"
    if normalized == "-c" or any(
        marker in normalized for marker in ("config", "checkpoint", "weight", "resume")
    ):
        return "configuration"
    if normalized in {"-d", "-i"}:
        return "training_data"
    if any(marker in normalized for marker in ("teacher", "pseudo")):
        return "teacher_supervision"
    if any(marker in normalized for marker in ("ground-truth", "-gt", "label", "annotation", "target")):
        return "ground_truth"
    if any(marker in normalized for marker in ("data", "dataset", "input", "source", "video", "image", "manifest")):
        return "training_data"
    return "ambiguous"


def _looks_data_reference(flag: str | None, value: str, index: int) -> bool:
    role = _flag_role(flag)
    if role in {"configuration", "output"}:
        return False
    if role in DATA_INPUT_ROLES:
        return True
    lowered = value.casefold()
    if lowered.endswith((".json", ".jsonl", ".csv", ".tsv", ".parquet", ".mp4", ".mov", ".wav", ".jpg", ".jpeg", ".png", ".zip", ".npy", ".npz")):
        return True
    if index > 1 and any(root in lowered for root in ("data/", "eval_clips/", "cvat_upload/")):
        return True
    return False


def _classify_command(
    argv: list[str],
    pairs: list[tuple[str | None, str, int]],
) -> tuple[str | None, bool, str | None, list[str]]:
    errors: list[str] = []
    script_tokens = [
        value.casefold()
        for flag, value, index in pairs
        if flag is None and (index <= 2 or value.casefold().endswith(".py"))
    ]
    explicit_modes = {
        value.casefold()
        for flag, value, _ in pairs
        if flag in {"--mode", "--operation", "--action"}
    }
    semantic = " ".join(script_tokens)
    training = bool(
        re.search(r"(?:^|[\s/_.-])(train|trainer|training|fit|finetune|fine-tune)(?:$|[\s/_.-])", semantic)
        or explicit_modes & {"train", "training", "fit", "finetune", "fine-tune"}
        or any("torchrun" in token for token in script_tokens)
    )
    evaluation = bool(
        re.search(r"(?:^|[\s/_.-])(eval|evaluate|score|benchmark|infer|inference|predict|audit)(?:$|[\s/_.-])", semantic)
        or explicit_modes & {"eval", "evaluate", "score", "infer", "inference", "predict", "audit"}
    )
    operation: str | None = None
    if training and evaluation:
        errors.append("dispatch.command: argv operation is ambiguous (both train and evaluation markers)")
    elif training:
        operation = "train"
    elif evaluation:
        operation = "evaluate"
    else:
        errors.append("dispatch.command: argv operation is unresolvable; use an explicit train/eval argv form")

    gpu = any(re.search(r"(?:^|[/_.:=-])(gpu|cuda|nvidia)(?:$|[/_.:=-])", token.casefold()) for token in argv)
    gpu = gpu or any(
        (
            flag is not None
            and "gpu" in flag
            and value.casefold() not in {"0", "false", "none", "off"}
        )
        or (
            flag in {"--accelerator", "--device"}
            and re.search(r"(?:^|[:/_.=-])(gpu|cuda|nvidia)(?:$|[:/_.=-])", value.casefold())
        )
        for flag, value, _ in pairs
    )
    gpu = gpu or any("torchrun" in token for token in script_tokens)

    explicit_components = {
        value.upper()
        for flag, value, _ in pairs
        if flag in {"--component", "--task"} and value.upper() in COMPONENTS
    }
    script_components = {
        component
        for component in COMPONENTS
        if re.search(rf"(?:^|[/_.-]){component.casefold()}(?:$|[/_.-])", semantic)
    }
    components = explicit_components | script_components
    component: str | None = None
    if len(components) == 1:
        component = next(iter(components))
    elif not components:
        errors.append("dispatch.command: dispatching component is unresolvable from argv")
    else:
        errors.append(f"dispatch.command: dispatching component is ambiguous in argv: {sorted(components)}")
    return operation, gpu, component, errors


def _clean_subset_for_path(asset: dict[str, Any], candidate: Path, repo_root: Path) -> dict[str, Any] | None:
    for subset in asset.get("protection", {}).get("clean_subsets", []):
        selector = _normalized_path(subset["selector_path"], repo_root)
        if selector is not None and candidate == selector:
            return subset
    return None


def _resolve_command_assets(
    ledger: dict[str, Any],
    pairs: list[tuple[str | None, str, int]],
    *,
    repo_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    resolved: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    path_registry: list[tuple[Path, dict[str, Any], str]] = []
    identity_registry: list[tuple[str, dict[str, Any]]] = []
    for asset in ledger["assets"]:
        for entry in asset["paths"]:
            normalized = _normalized_path(entry["path"], repo_root)
            if normalized is not None:
                path_registry.append((normalized, asset, "asset_path"))
        for binding in asset["immutable_hashes"]:
            normalized = _normalized_path(binding["path"], repo_root)
            if normalized is not None:
                path_registry.append((normalized, asset, "immutable_binding"))
        for subset in asset.get("protection", {}).get("clean_subsets", []):
            normalized = _normalized_path(subset["selector_path"], repo_root)
            if normalized is not None:
                path_registry.append((normalized, asset, "clean_subset"))
        for identity in asset["protection"]["identities"]:
            identity_registry.extend((alias, asset) for alias in _identity_aliases(identity["identity"]))

    for flag, value, index in pairs:
        if index <= 1 and flag is None and value.casefold().endswith(("python", "python3", ".py")):
            continue
        candidate = _normalized_path(value, repo_root)
        matches: list[tuple[dict[str, Any], str]] = []
        if candidate is not None:
            for registered, asset, match_kind in path_registry:
                if candidate == registered or _path_contains(registered, candidate) or _path_contains(candidate, registered):
                    matches.append((asset, match_kind))
        lowered = value.casefold()
        for alias, asset in identity_registry:
            if alias in lowered:
                matches.append((asset, "protected_identity"))
        for asset in ledger["assets"]:
            if asset["asset_id"].casefold() in lowered:
                matches.append((asset, "asset_id"))

        if matches:
            for asset, match_kind in matches:
                record = {
                    "argv_index": index,
                    "flag": flag,
                    "value": value,
                    "path": candidate,
                    "role": _flag_role(flag),
                    "match_kind": match_kind,
                    "clean_subset": (
                        _clean_subset_for_path(asset, candidate, repo_root) if candidate is not None else None
                    ),
                }
                records = resolved.setdefault(asset["asset_id"], [])
                signature = (index, match_kind, str(candidate), asset["asset_id"])
                if not any(
                    (row["argv_index"], row["match_kind"], str(row["path"]), asset["asset_id"]) == signature
                    for row in records
                ):
                    records.append(record)
        elif _looks_data_reference(flag, value, index):
            errors.append(f"dispatch.command.argv[{index}]: data-looking reference is absent from ledger: {value}")

    if not resolved:
        errors.append("dispatch.command: argv resolves no ledger asset; refusing unbound dispatch")
    return resolved, errors


def _missing_gate_value(contract: dict[str, Any], key: str) -> bool:
    value = contract.get(key)
    if not isinstance(value, dict):
        return True
    return any(value.get(field) in (None, "", []) for field in ("metric", "threshold"))


def audit_dispatch_contract(
    ledger: dict[str, Any],
    contract: dict[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    errors: list[str] = []
    assets = _asset_index(ledger)
    for key in ("baseline", "check", "kill_threshold"):
        if _missing_gate_value(contract, key):
            errors.append(f"dispatch.{key}: metric and threshold are required")

    command = contract.get("command")
    if not isinstance(command, dict):
        errors.append("dispatch.command: object is required")
        command = {}
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(token, str) or not token for token in argv):
        errors.append("dispatch.command.argv: non-empty string array is required")
        argv = []

    pairs, parse_errors = _command_pairs(argv)
    errors.extend(parse_errors)
    operation, gpu, component, command_errors = _classify_command(argv, pairs)
    errors.extend(command_errors)
    resolved, resolution_errors = _resolve_command_assets(ledger, pairs, repo_root=repo_root)
    errors.extend(resolution_errors)

    # Legacy caller declarations are advisory only. They can add errors, never grant access.
    inputs = contract.get("inputs", [])
    if inputs is not None and not isinstance(inputs, list):
        errors.append("dispatch.inputs: must be an array when provided")
        inputs = []
    for item in inputs:
        if not isinstance(item, dict):
            errors.append("dispatch.inputs: advisory entries must be objects")
            continue
        asset_id = item.get("asset_id")
        asset = assets.get(asset_id)
        if asset is None:
            errors.append(f"dispatch.inputs: asset is absent from ledger: {asset_id}")
            continue
        declared_hashes = item.get("immutable_hashes")
        if declared_hashes is not None:
            ledger_bindings = {
                (binding["path"], binding["algorithm"], binding["digest"])
                for binding in asset["immutable_hashes"]
            }
            for binding in declared_hashes if isinstance(declared_hashes, list) else []:
                identity = (binding.get("path"), binding.get("algorithm"), binding.get("digest"))
                if identity not in ledger_bindings:
                    errors.append(f"dispatch.inputs[{asset_id}]: hash differs from ledger: {identity}")

    resolved_train: set[str] = set()
    resolved_holdout: set[str] = set()
    command_selector_paths: set[str] = set()
    for asset_id, references in sorted(resolved.items()):
        asset = assets[asset_id]
        input_references = [
            reference for reference in references if reference["role"] in DATA_INPUT_ROLES
        ]
        non_input_references = [
            reference for reference in references if reference["role"] not in DATA_INPUT_ROLES
        ]
        for reference in non_input_references:
            errors.append(
                f"dispatch.command.argv[{reference['argv_index']}]: resolved ledger reference for "
                f"{asset_id} requires a recognized data-bearing input role; got {reference['role']}"
            )

        subsets = [reference["clean_subset"] for reference in input_references if reference["clean_subset"]]
        selector_paths = {subset["selector_path"] for subset in subsets}
        command_selector_paths.update(selector_paths)
        clean_subset = (
            subsets[0]
            if input_references
            and len(subsets) == len(input_references)
            and len(selector_paths) == 1
            else None
        )
        if subsets and clean_subset is None:
            errors.append(
                f"dispatch.inputs[{asset_id}]: every resolved data reference must use the same immutable selector"
            )

        if gpu and asset["counts"]["decoded_count"] == 0:
            errors.append(f"dispatch.command: GPU asset has zero decoded rows: {asset_id}")

        if operation == "train":
            if asset["state"] in TRAIN_REFUSAL_STATES:
                errors.append(f"dispatch.inputs[{asset_id}]: ledger state {asset['state']} refuses train use")
            overlap_status = asset["protection"]["overlap_check_coverage"]["status"]
            if overlap_status != "PASS" and clean_subset is None:
                errors.append(
                    f"dispatch.inputs[{asset_id}]: ledger overlap coverage {overlap_status} refuses unscoped train use"
                )
            if asset["protection"]["trainer_forbidden"]:
                errors.append(f"dispatch.inputs[{asset_id}]: ledger protection forbids trainer reachability")
            identities = asset["protection"]["identities"]
            if identities and not asset["protection"]["trainer_forbidden"]:
                excluded = set(clean_subset.get("excluded_identities", [])) if clean_subset else set()
                required = {identity["identity"] for identity in identities}
                if not clean_subset or not required.issubset(excluded):
                    errors.append(
                        f"dispatch.inputs[{asset_id}]: mixed protected/compare identities require an immutable clean subset"
                    )
            if component is not None:
                ruling = asset["rights"]["component_rulings"][component]
                if ruling["decision"] == "FORBID":
                    errors.append(f"dispatch.inputs[{asset_id}]: {component}=FORBID in ledger component ruling")
                elif ruling["decision"] == "CONDITIONAL" and (
                    clean_subset is None or component not in clean_subset["allowed_components"]
                ):
                    errors.append(
                        f"dispatch.inputs[{asset_id}]: {component}=CONDITIONAL lacks an immutable component-authorized clean subset proof"
                    )
            if "teacher" in asset["label_authority"]:
                roles = {reference["role"] for reference in references}
                if "ground_truth" in roles:
                    errors.append(
                        f"dispatch.inputs[{asset_id}]: ledger authority teacher cannot be used as ground truth"
                    )
                elif "teacher_supervision" not in roles:
                    errors.append(
                        f"dispatch.inputs[{asset_id}]: ledger authority teacher requires an argv teacher/pseudo supervision role"
                    )

        if clean_subset:
            resolved_train.update(_canonical_family(value) for value in clean_subset["train_families"])
            resolved_holdout.update(_canonical_family(value) for value in clean_subset["holdout_families"])
        else:
            train, holdout = _partition_families(asset)
            resolved_train.update(train)
            resolved_holdout.update(holdout)

    if len(command_selector_paths) > 1:
        errors.append(
            "dispatch.command: every resolved data reference must use the same immutable selector; "
            f"found {sorted(command_selector_paths)}"
        )

    overlap = sorted((resolved_train & resolved_holdout) - {""})
    if overlap:
        errors.append(f"dispatch.ledger_partitions: train/holdout lineage overlap: {overlap}")

    current_hash_errors = verify_hashes(
        {"assets": [assets[asset_id] for asset_id in resolved]},
        repo_root,
    )
    errors.extend(f"dispatch.current_hash: {error}" for error in current_hash_errors)
    return sorted(set(errors))


def render_markdown(ledger: dict[str, Any]) -> str:
    assets = sorted(ledger["assets"], key=lambda row: row["asset_id"])
    state_counts = Counter(asset["state"] for asset in assets)
    lines = [
        "<!-- GENERATED, do not hand-edit. Source: runs/manager/data_ledger.json via scripts/racketsport/audit_data_utilization.py. -->",
        "# Data Ledger (generated view)",
        "",
        "This is a coordination view for data lineage and utilization only. `NORTH_STAR_ROADMAP.md` remains product truth.",
        "",
        f"- Ledger schema: `{ledger['schema_version']}`",
        f"- Snapshot UTC: `{ledger['generated_utc']}`",
        f"- Assets: `{len(assets)}`",
        "- States: " + ", ".join(f"`{key}={state_counts[key]}`" for key in sorted(state_counts)),
        "",
        "| Asset ID | State | Bytes | Raw | Kept | Decoded | Labels | Authority | Owner | Next check |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for asset in assets:
        counts = asset["counts"]
        authority = ", ".join(asset["label_authority"])
        next_check = asset["next_check"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{asset['asset_id']}` | `{asset['state']}` | {counts['byte_count']} | "
            f"{counts['raw_count']} {counts['raw_unit']} | {counts['dedup_kept_count']} "
            f"{counts['dedup_unit']} | {counts['decoded_count']} {counts['decoded_unit']} | "
            f"{counts['label_count']} {counts['label_unit']} | {authority} | {asset['owner']} | {next_check} |"
        )

    lines.extend(["", "## Per-asset rulings and utilization", ""])
    for asset in assets:
        rendered_paths = ", ".join(f"`{item['path']}`" for item in asset["paths"])
        lines.extend(
            [
                f"### `{asset['asset_id']}`",
                "",
                f"- State reason: {asset['state_reason']}",
                f"- Paths: {rendered_paths}",
                f"- Source families: {', '.join(asset['source_lineage']['original_sources']) or 'none recorded'}",
                f"- Partition: train={asset['partitions']['train']}; val={asset['partitions']['val']}; test={asset['partitions']['test']}",
                f"- Overlap coverage: {asset['protection']['overlap_check_coverage']['status']} — {asset['protection']['overlap_check_coverage']['scope']}",
                f"- Immutable clean-subset selectors: {len(asset['protection'].get('clean_subsets', []))}",
                f"- Consumers: {len(asset['consumers'])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    ledger: dict[str, Any],
    *,
    repo_root: Path,
    as_of: datetime,
    contract: dict[str, Any] | None = None,
    check_view: Path | None = None,
) -> dict[str, Any]:
    ledger_errors = validate_ledger(ledger)
    hash_errors = [] if ledger_errors else verify_hashes(ledger, repo_root)
    dispatch_errors = []
    if contract is not None and not ledger_errors:
        dispatch_errors = audit_dispatch_contract(ledger, contract, repo_root=repo_root)
    view_errors: list[str] = []
    if check_view is not None and not ledger_errors:
        expected = render_markdown(ledger)
        if not check_view.is_file():
            view_errors.append(f"generated view is absent: {check_view}")
        elif check_view.read_text(encoding="utf-8") != expected:
            view_errors.append(f"generated view differs: {check_view}")

    never_queued = [] if ledger_errors else never_queued_assets(ledger, as_of=as_of)
    state_distribution = (
        dict(sorted(Counter(asset["state"] for asset in ledger.get("assets", [])).items()))
        if not ledger_errors
        else {}
    )
    errors = ledger_errors + hash_errors + dispatch_errors + view_errors
    return {
        "artifact_type": "data_utilization_audit",
        "status": "fail" if errors else "pass",
        "asset_count": len(ledger.get("assets", [])),
        "state_distribution": state_distribution,
        "never_queued_count": len(never_queued),
        "never_queued": never_queued,
        "ledger_errors": ledger_errors,
        "hash_errors": hash_errors,
        "dispatch_errors": dispatch_errors,
        "view_errors": view_errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the machine-readable data ledger and optionally gate a dispatch contract."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dispatch-contract", type=Path)
    parser.add_argument("--write-view", type=Path)
    parser.add_argument("--check-view", type=Path)
    parser.add_argument("--as-of", help="UTC/offset ISO timestamp used by the >24h never-queued audit")
    parser.add_argument("--json", action="store_true", help="Emit the full machine-readable audit report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    ledger_path = args.ledger if args.ledger.is_absolute() else repo_root / args.ledger
    try:
        ledger = load_json(ledger_path)
        contract_path = None
        if args.dispatch_contract:
            contract_path = (
                args.dispatch_contract
                if args.dispatch_contract.is_absolute()
                else repo_root / args.dispatch_contract
            )
        contract = load_json(contract_path) if contract_path else None
        as_of = _parse_utc(args.as_of) if args.as_of else datetime.now(timezone.utc)
        if args.write_view:
            validation_errors = validate_ledger(ledger)
            if validation_errors:
                raise ValueError("cannot generate view from invalid ledger: " + "; ".join(validation_errors))
            output = args.write_view if args.write_view.is_absolute() else repo_root / args.write_view
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(ledger), encoding="utf-8")
        check_view = None
        if args.check_view:
            check_view = args.check_view if args.check_view.is_absolute() else repo_root / args.check_view
        report = build_report(
            ledger,
            repo_root=repo_root,
            as_of=as_of,
            contract=contract,
            check_view=check_view,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "artifact_type": "data_utilization_audit",
            "status": "fail",
            "asset_count": 0,
            "state_distribution": {},
            "never_queued_count": 0,
            "never_queued": [],
            "ledger_errors": [str(exc)],
            "hash_errors": [],
            "dispatch_errors": [],
            "view_errors": [],
        }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DATA UTILIZATION AUDIT: {report['status'].upper()}")
        print(f"assets={report['asset_count']}")
        print(f"states={json.dumps(report['state_distribution'], sort_keys=True)}")
        print(f"NEVER-QUEUED ({report['never_queued_count']})")
        for row in report["never_queued"]:
            print(
                f"{row['acquired_utc']} {row['asset_id']} bytes={row['byte_count']} "
                f"labels={row['label_count']} state={row['state']}"
            )
        for category in ("ledger_errors", "hash_errors", "dispatch_errors", "view_errors"):
            for error in report[category]:
                print(f"ERROR {category}: {error}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
