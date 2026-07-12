from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from tests.racketsport.test_coaching_facts import _write_fact_run
from threed.racketsport.coaching_fact_audit import audit_coaching_facts, audit_coaching_facts_file
from threed.racketsport.rally_metrics import build_rally_metrics


def test_zero_fabrication_audit_accepts_canonical_artifact(tmp_path: Path) -> None:
    payload = build_rally_metrics(_write_fact_run(tmp_path / "run"))["coaching_card_facts"]

    report = audit_coaching_facts(payload)

    assert report["verdict"] == "pass"
    assert report["issues"] == []
    assert report["checked_fact_count"] == 4
    assert report["checked_source_count"] == 4


def test_zero_fabrication_audit_accepts_clip_fallback_lineage(tmp_path: Path) -> None:
    run_dir = _write_fact_run(tmp_path / "run")
    (run_dir / "rally_spans.json").unlink()
    payload = build_rally_metrics(run_dir)["coaching_card_facts"]

    report = audit_coaching_facts(payload)

    assert report["verdict"] == "pass"
    rally_fact = next(fact for fact in payload["audited_facts"] if fact["fact_type"] == "rally")
    assert rally_fact["source_artifacts"][0]["source_id"] == "virtual_world"
    assert rally_fact["trust"]["provenance_band"] == "model_estimated"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("fact_value", "canonical_mismatch"),
        ("interval_number", "canonical_mismatch"),
        ("coverage_number", "canonical_mismatch"),
        ("missing_numeric_lineage", "unlinked_number"),
        ("top_source_hash", "source_hash_mismatch"),
        ("inline_source_hash", "source_hash_mismatch"),
        ("missing_source_hash", "missing_source_hash"),
        ("unknown_field", "unknown_field"),
        ("free_form_language", "free_form_language"),
        ("free_form_value", "free_form_language"),
        ("verified_authority", "authority_boundary"),
        ("advanced_fact", "authority_boundary"),
        ("bad_evidence_uri", "evidence_not_openable"),
        ("post_manifest_marker", "post_manifest_fact"),
    ],
)
def test_mutation_audit_rejects_each_output_number_hash_and_policy_class(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    payload = build_rally_metrics(_write_fact_run(tmp_path / "run"))["coaching_card_facts"]
    mutated = copy.deepcopy(payload)
    movement = next(fact for fact in mutated["audited_facts"] if fact["fact_type"] == "movement")
    if mutation == "fact_value":
        movement["value"] += 1.0
    elif mutation == "interval_number":
        movement["interval"]["pts_end_s"] += 0.01
    elif mutation == "coverage_number":
        movement["coverage"]["fraction"] = 0.123
    elif mutation == "missing_numeric_lineage":
        movement["numeric_lineage"] = [
            item for item in movement["numeric_lineage"] if item["output_pointer"] != "/value"
        ]
    elif mutation == "top_source_hash":
        mutated["source_artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "inline_source_hash":
        movement["source_artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "missing_source_hash":
        del movement["source_artifacts"][0]["sha256"]
    elif mutation == "unknown_field":
        movement["invented"] = 7
    elif mutation == "free_form_language":
        movement["recommendation"] = "swing harder"
    elif mutation == "free_form_value":
        movement["value"] = "swing harder"
    elif mutation == "verified_authority":
        movement["trust"]["authority_band"] = "verified"
    elif mutation == "advanced_fact":
        movement["fact_type"] = "contact"
    elif mutation == "bad_evidence_uri":
        movement["evidence_locator"]["uri"] = "file:///does/not/exist.json#/players"
    elif mutation == "post_manifest_marker":
        mutated["build_order"] = "after_manifest"
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    report = audit_coaching_facts(mutated)

    assert report["verdict"] == "reject"
    assert expected_code in report["issue_codes"]


@pytest.mark.parametrize(
    ("source_name", "mutate"),
    [
        ("virtual_world.json", lambda payload: payload["players"][0]["frames"][0].__setitem__("t", 0.01)),
        (
            "virtual_world.json",
            lambda payload: payload["players"][0]["frames"][1]["track_world_xy"].__setitem__(0, 0.25),
        ),
        ("court_zones.json", lambda payload: payload["zones"]["court"][0].__setitem__(0, -2.9)),
        ("rally_spans.json", lambda payload: payload["spans"][0].__setitem__("t1", 0.39)),
        ("contact_windows.json", lambda payload: payload["events"][0].__setitem__("t", 0.11)),
    ],
)
def test_mutation_audit_rejects_every_source_numeric_class(
    tmp_path: Path,
    source_name: str,
    mutate,
) -> None:
    run_dir = _write_fact_run(tmp_path / "run")
    payload = build_rally_metrics(run_dir)["coaching_card_facts"]
    source_path = run_dir / source_name
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mutate(source)
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = audit_coaching_facts(payload)

    assert report["verdict"] == "reject"
    assert "source_hash_mismatch" in report["issue_codes"]


def test_audit_rejects_facts_file_written_after_manifest(tmp_path: Path) -> None:
    run_dir = _write_fact_run(tmp_path / "run")
    payload = build_rally_metrics(run_dir)["coaching_card_facts"]
    manifest_path = run_dir / "replay_viewer_manifest.json"
    facts_path = run_dir / "coaching_card_facts.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    facts_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_mtime = manifest_path.stat().st_mtime_ns
    os.utime(facts_path, ns=(manifest_mtime + 1_000_000, manifest_mtime + 1_000_000))

    report = audit_coaching_facts_file(facts_path, manifest_path=manifest_path)

    assert report["verdict"] == "reject"
    assert "post_manifest_fact" in report["issue_codes"]


def test_schema_declares_strict_unknown_field_rejection() -> None:
    schema_path = Path("docs/racketsport/coaching_facts_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["auditedFact"]["additionalProperties"] is False
    assert schema["$defs"]["sourceArtifact"]["properties"]["sha256"]["pattern"] == "^[0-9a-f]{64}$"
