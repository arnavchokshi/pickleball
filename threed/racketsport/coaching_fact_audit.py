"""Fail-closed zero-fabrication audit for deterministic coaching facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from threed.racketsport.rally_metrics import build_rally_metrics


AUDIT_ARTIFACT_TYPE = "coaching_fact_zero_fabrication_audit"
ALLOWED_FACT_TYPES = {"rally", "movement", "positioning", "recovery"}
ALLOWED_AUTHORITIES = {"preview", "low_confidence", "too_close_to_call"}
TOP_FIELDS = {
    "schema_version",
    "artifact_type",
    "source_run_dir",
    "rally_scope",
    "build_order",
    "compatibility",
    "priority_rule",
    "source_artifacts",
    "facts",
    "audited_facts",
    "omissions",
}
SOURCE_FIELDS = {"source_id", "path", "sha256", "artifact_type"}
FACT_FIELDS = {
    "fact_id",
    "fact_type",
    "metric",
    "value",
    "unit",
    "rally_id",
    "entity",
    "player_id",
    "interval",
    "coordinate_space",
    "time_space",
    "trust",
    "coverage",
    "rule",
    "source_artifacts",
    "evidence_locator",
    "numeric_lineage",
}
LEGACY_FIELDS = {
    "rally_id",
    "rally_scope",
    "player_id",
    "metric",
    "value",
    "unit",
    "trust",
    "frames_used",
    "frames_total",
    "coverage_fraction",
}
OMISSION_FIELDS = {
    "fact_type",
    "status",
    "reason_code",
    "required_artifacts",
    "missing_artifacts",
    "required_gate_ids",
    "observed_artifacts",
}
FREE_FORM_KEYS = {
    "claim",
    "coach_text",
    "description",
    "explanation",
    "language",
    "message",
    "narrative",
    "phrasing",
    "recommendation",
    "reason",
    "summary",
    "text",
}


def audit_coaching_facts(
    payload: Mapping[str, Any],
    *,
    facts_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Audit an in-memory facts artifact against its immutable JSON sources."""

    issues: list[dict[str, str]] = []
    _check_keys(payload, TOP_FIELDS, TOP_FIELDS, path="/", issues=issues)
    _check_free_form_keys(payload, path="", issues=issues)
    _require_equal(payload.get("schema_version"), 2, "/schema_version", "invalid_schema", issues)
    _require_equal(payload.get("artifact_type"), "coaching_card_facts", "/artifact_type", "invalid_schema", issues)
    _require_equal(
        payload.get("build_order"),
        "coaching_facts_before_manifest",
        "/build_order",
        "post_manifest_fact",
        issues,
    )
    compatibility = payload.get("compatibility")
    if isinstance(compatibility, Mapping):
        _check_keys(
            compatibility,
            {"facts_field", "authoritative_field", "user_facing"},
            {"facts_field", "authoritative_field", "user_facing"},
            path="/compatibility",
            issues=issues,
        )
        _require_equal(compatibility.get("facts_field"), "legacy_v1_projection", "/compatibility/facts_field", "invalid_schema", issues)
        _require_equal(compatibility.get("authoritative_field"), "audited_facts", "/compatibility/authoritative_field", "invalid_schema", issues)
        _require_equal(compatibility.get("user_facing"), False, "/compatibility/user_facing", "authority_boundary", issues)
    else:
        _issue(issues, "invalid_schema", "/compatibility")

    top_sources = _source_records(payload.get("source_artifacts"), "/source_artifacts", issues)
    for index, legacy in enumerate(_as_list(payload.get("facts"))):
        if not isinstance(legacy, Mapping):
            _issue(issues, "invalid_schema", f"/facts/{index}")
            continue
        _check_keys(legacy, LEGACY_FIELDS, LEGACY_FIELDS, path=f"/facts/{index}", issues=issues)

    audited_facts = _as_list(payload.get("audited_facts"))
    fact_ids: set[str] = set()
    for index, fact in enumerate(audited_facts):
        fact_path = f"/audited_facts/{index}"
        if not isinstance(fact, Mapping):
            _issue(issues, "invalid_schema", fact_path)
            continue
        _check_keys(fact, FACT_FIELDS, FACT_FIELDS, path=fact_path, issues=issues)
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id.startswith("ns051."):
            _issue(issues, "invalid_schema", f"{fact_path}/fact_id")
        elif fact_id in fact_ids:
            _issue(issues, "duplicate_fact_id", f"{fact_path}/fact_id")
        else:
            fact_ids.add(fact_id)
        if fact.get("fact_type") not in ALLOWED_FACT_TYPES:
            _issue(issues, "authority_boundary", f"{fact_path}/fact_type")
        trust = fact.get("trust")
        if isinstance(trust, Mapping):
            _check_keys(
                trust,
                {"provenance_band", "authority_band", "gate_id", "gate_status"},
                {"provenance_band", "authority_band", "gate_id", "gate_status"},
                path=f"{fact_path}/trust",
                issues=issues,
            )
            if trust.get("authority_band") not in ALLOWED_AUTHORITIES:
                _issue(issues, "authority_boundary", f"{fact_path}/trust/authority_band")
            if trust.get("gate_status") != "unpassed":
                _issue(issues, "authority_boundary", f"{fact_path}/trust/gate_status")
        else:
            _issue(issues, "invalid_schema", f"{fact_path}/trust")
        for field_name, allowed_fields in (
            ("entity", {"type", "id"}),
            ("interval", {"frame_start", "frame_end_exclusive", "pts_start_s", "pts_end_s"}),
            ("coverage", {"frames_used", "frames_total", "fraction"}),
            ("rule", {"id", "version"}),
            ("evidence_locator", {"uri", "source_id", "json_pointer"}),
        ):
            field = fact.get(field_name)
            if isinstance(field, Mapping):
                _check_keys(field, allowed_fields, allowed_fields, path=f"{fact_path}/{field_name}", issues=issues)
            else:
                _issue(issues, "invalid_schema", f"{fact_path}/{field_name}")
        fact_sources = _source_records(fact.get("source_artifacts"), f"{fact_path}/source_artifacts", issues)
        _check_fact_sources_against_top(fact_sources, top_sources, fact_path, issues)
        _check_numeric_lineage(fact, fact_path, issues)
        _check_evidence_locator(fact.get("evidence_locator"), fact_sources, fact_path, issues)

    _check_omissions(payload.get("omissions"), issues)
    _check_no_manifest_sources(top_sources, issues)
    _check_manifest_order(payload, facts_path=facts_path, manifest_path=manifest_path, issues=issues)
    _check_canonical_rebuild(payload, issues)

    codes = sorted({issue["code"] for issue in issues})
    return {
        "schema_version": 1,
        "artifact_type": AUDIT_ARTIFACT_TYPE,
        "verdict": "pass" if not issues else "reject",
        "checked_fact_count": len(audited_facts),
        "checked_source_count": len(top_sources),
        "issue_codes": codes,
        "issues": issues,
    }


def audit_coaching_facts_file(
    path: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and audit a coaching facts JSON artifact."""

    facts_path = Path(path)
    payload = json.loads(facts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {
            "schema_version": 1,
            "artifact_type": AUDIT_ARTIFACT_TYPE,
            "verdict": "reject",
            "checked_fact_count": 0,
            "checked_source_count": 0,
            "issue_codes": ["invalid_schema"],
            "issues": [{"code": "invalid_schema", "path": "/"}],
        }
    return audit_coaching_facts(payload, facts_path=facts_path, manifest_path=manifest_path)


def _source_records(value: Any, path: str, issues: list[dict[str, str]]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    if not isinstance(value, list) or not value:
        _issue(issues, "invalid_schema", path)
        return records
    for index, source in enumerate(value):
        source_path = f"{path}/{index}"
        if not isinstance(source, Mapping):
            _issue(issues, "invalid_schema", source_path)
            continue
        _check_keys(source, SOURCE_FIELDS, SOURCE_FIELDS, path=source_path, issues=issues)
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            _issue(issues, "invalid_schema", f"{source_path}/source_id")
            continue
        if source_id in records and records[source_id] != source:
            _issue(issues, "duplicate_source_id", f"{source_path}/source_id")
        records[source_id] = source
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            _issue(issues, "missing_source_hash", f"{source_path}/sha256")
        local_path = Path(str(source.get("path", "")))
        if not local_path.is_file():
            _issue(issues, "source_not_openable", f"{source_path}/path")
        elif isinstance(digest, str) and hashlib.sha256(local_path.read_bytes()).hexdigest() != digest:
            _issue(issues, "source_hash_mismatch", f"{source_path}/sha256")
    return records


def _check_fact_sources_against_top(
    fact_sources: Mapping[str, Mapping[str, Any]],
    top_sources: Mapping[str, Mapping[str, Any]],
    fact_path: str,
    issues: list[dict[str, str]],
) -> None:
    for source_id, source in fact_sources.items():
        if source_id not in top_sources or top_sources[source_id] != source:
            _issue(issues, "source_binding_mismatch", f"{fact_path}/source_artifacts/{source_id}")


def _check_numeric_lineage(fact: Mapping[str, Any], fact_path: str, issues: list[dict[str, str]]) -> None:
    expected = set(_numeric_leaf_pointers({key: value for key, value in fact.items() if key != "numeric_lineage"}))
    lineage = fact.get("numeric_lineage")
    if not isinstance(lineage, list):
        _issue(issues, "unlinked_number", f"{fact_path}/numeric_lineage")
        return
    actual: set[str] = set()
    source_ids = {
        source.get("source_id")
        for source in _as_list(fact.get("source_artifacts"))
        if isinstance(source, Mapping)
    }
    source_by_id = {
        source.get("source_id"): source
        for source in _as_list(fact.get("source_artifacts"))
        if isinstance(source, Mapping)
    }
    for index, item in enumerate(lineage):
        item_path = f"{fact_path}/numeric_lineage/{index}"
        if not isinstance(item, Mapping):
            _issue(issues, "invalid_schema", item_path)
            continue
        allowed = {"output_pointer", "formula_id", "source_ids", "source_json_pointers"}
        _check_keys(item, allowed, allowed, path=item_path, issues=issues)
        output_pointer = item.get("output_pointer")
        if isinstance(output_pointer, str):
            actual.add(output_pointer)
        linked_ids = item.get("source_ids")
        pointers = item.get("source_json_pointers")
        if not isinstance(linked_ids, list) or not linked_ids or not set(linked_ids).issubset(source_ids):
            _issue(issues, "unlinked_number", f"{item_path}/source_ids")
            continue
        if not isinstance(pointers, list) or not pointers:
            _issue(issues, "unlinked_number", f"{item_path}/source_json_pointers")
            continue
        for source_id, pointer in zip(linked_ids, pointers):
            source = source_by_id.get(source_id)
            if source is None or not _source_pointer_opens(source, pointer):
                _issue(issues, "source_number_not_openable", f"{item_path}/source_json_pointers")
    for pointer in sorted(expected - actual):
        _issue(issues, "unlinked_number", f"{fact_path}{pointer}")
    for pointer in sorted(actual - expected):
        _issue(issues, "invalid_numeric_lineage", f"{fact_path}/numeric_lineage/{pointer}")


def _check_evidence_locator(
    locator: Any,
    sources: Mapping[str, Mapping[str, Any]],
    fact_path: str,
    issues: list[dict[str, str]],
) -> None:
    if not isinstance(locator, Mapping):
        return
    uri = locator.get("uri")
    source_id = locator.get("source_id")
    pointer = locator.get("json_pointer")
    if not isinstance(uri, str) or not isinstance(source_id, str) or not isinstance(pointer, str):
        _issue(issues, "evidence_not_openable", f"{fact_path}/evidence_locator")
        return
    source = sources.get(source_id)
    parsed = urlsplit(uri)
    uri_path = Path(unquote(parsed.path)) if parsed.scheme == "file" else None
    if source is None or uri_path is None or uri_path != Path(str(source.get("path", ""))):
        _issue(issues, "evidence_not_openable", f"{fact_path}/evidence_locator/uri")
        return
    if parsed.fragment != pointer or not _source_pointer_opens(source, pointer):
        _issue(issues, "evidence_not_openable", f"{fact_path}/evidence_locator/json_pointer")


def _check_omissions(value: Any, issues: list[dict[str, str]]) -> None:
    omissions = _as_list(value)
    if {item.get("fact_type") for item in omissions if isinstance(item, Mapping)} != {"shot", "landing", "contact"}:
        _issue(issues, "unsupported_advanced_fact", "/omissions")
    for index, omission in enumerate(omissions):
        path = f"/omissions/{index}"
        if not isinstance(omission, Mapping):
            _issue(issues, "invalid_schema", path)
            continue
        _check_keys(omission, OMISSION_FIELDS, OMISSION_FIELDS, path=path, issues=issues)
        if omission.get("status") != "absent":
            _issue(issues, "authority_boundary", f"{path}/status")
        if omission.get("reason_code") not in {"required_artifact_missing", "required_gate_unpassed"}:
            _issue(issues, "invalid_schema", f"{path}/reason_code")
        _source_records(omission.get("observed_artifacts"), f"{path}/observed_artifacts", issues) if omission.get("observed_artifacts") else None


def _check_no_manifest_sources(sources: Mapping[str, Mapping[str, Any]], issues: list[dict[str, str]]) -> None:
    for source_id, source in sources.items():
        if Path(str(source.get("path", ""))).name == "replay_viewer_manifest.json":
            _issue(issues, "post_manifest_fact", f"/source_artifacts/{source_id}/path")


def _check_manifest_order(
    payload: Mapping[str, Any],
    *,
    facts_path: Path | str | None,
    manifest_path: Path | str | None,
    issues: list[dict[str, str]],
) -> None:
    if facts_path is None:
        return
    facts = Path(facts_path)
    manifest = Path(manifest_path) if manifest_path is not None else Path(str(payload.get("source_run_dir", ""))) / "replay_viewer_manifest.json"
    if manifest.is_file() and facts.is_file() and facts.stat().st_mtime_ns > manifest.stat().st_mtime_ns:
        _issue(issues, "post_manifest_fact", "/build_order")


def _check_canonical_rebuild(payload: Mapping[str, Any], issues: list[dict[str, str]]) -> None:
    run_dir = payload.get("source_run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        _issue(issues, "invalid_schema", "/source_run_dir")
        return
    try:
        canonical = build_rally_metrics(Path(run_dir))["coaching_card_facts"]
    except (FileNotFoundError, NotADirectoryError, ValueError, json.JSONDecodeError):
        _issue(issues, "source_not_openable", "/source_run_dir")
        return
    if canonical != payload:
        _issue(issues, "canonical_mismatch", "/")


def _source_pointer_opens(source: Mapping[str, Any], pointer: Any) -> bool:
    if not isinstance(pointer, str):
        return False
    try:
        payload = json.loads(Path(str(source["path"])).read_text(encoding="utf-8"))
        _resolve_json_pointer(payload, pointer)
    except (KeyError, IndexError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _resolve_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError(pointer)
    current = payload
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, Mapping):
            current = current[token]
        else:
            raise TypeError(pointer)
    return current


def _numeric_leaf_pointers(value: Any, pointer: str = "") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [pointer or "/"]
    if isinstance(value, Mapping):
        pointers: list[str] = []
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            pointers.extend(_numeric_leaf_pointers(item, f"{pointer}/{escaped}"))
        return pointers
    if isinstance(value, list):
        pointers = []
        for index, item in enumerate(value):
            pointers.extend(_numeric_leaf_pointers(item, f"{pointer}/{index}"))
        return pointers
    return []


def _check_free_form_keys(value: Any, *, path: str, issues: list[dict[str, str]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}/{key}"
            if str(key).lower() in FREE_FORM_KEYS:
                _issue(issues, "free_form_language", child_path)
            if key == "value" and _contains_uncontrolled_string(item):
                _issue(issues, "free_form_language", child_path)
            _check_free_form_keys(item, path=child_path, issues=issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_free_form_keys(item, path=f"{path}/{index}", issues=issues)


def _contains_uncontrolled_string(value: Any) -> bool:
    controlled_value_tokens = {"kitchen", "transition", "baseline", "out_of_court"}
    if isinstance(value, str):
        return value not in controlled_value_tokens
    if isinstance(value, Mapping):
        return any(_contains_uncontrolled_string(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_uncontrolled_string(item) for item in value)
    return False


def _check_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    required: set[str],
    *,
    path: str,
    issues: list[dict[str, str]],
) -> None:
    for key in sorted(set(value) - allowed):
        _issue(issues, "unknown_field", f"{path.rstrip('/')}/{key}")
    for key in sorted(required - set(value)):
        _issue(issues, "missing_field", f"{path.rstrip('/')}/{key}")


def _require_equal(value: Any, expected: Any, path: str, code: str, issues: list[dict[str, str]]) -> None:
    if value != expected:
        _issue(issues, code, path)


def _issue(issues: list[dict[str, str]], code: str, path: str) -> None:
    issue = {"code": code, "path": path or "/"}
    if issue not in issues:
        issues.append(issue)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


__all__ = ["AUDIT_ARTIFACT_TYPE", "audit_coaching_facts", "audit_coaching_facts_file"]
