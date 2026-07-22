from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.racketsport.audit_data_utilization import (
    audit_dispatch_contract,
    load_json,
    never_queued_assets,
    render_markdown,
    validate_ledger,
    verify_hashes,
)


ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = "scripts/racketsport/audit_data_utilization.py"
LEDGER_PATH = ROOT / "runs" / "manager" / "data_ledger.json"
VIEW_PATH = ROOT / "runs" / "manager" / "DATA_LEDGER.md"
PERSON_SELECTOR_PATH = (
    ROOT
    / "runs"
    / "lanes"
    / "data_steward_ledger_20260721"
    / "person_core_commercial_15312_selector.json"
)
PERSON_SOURCE_INDEX_PATH = (
    ROOT
    / "data"
    / "roboflow_universe_20260706"
    / "aggregated"
    / "subset_indexes"
    / "person_index.json"
)


def _ledger() -> dict[str, object]:
    return load_json(LEDGER_PATH)


def _asset(ledger: dict[str, object], asset_id: str) -> dict[str, object]:
    return next(asset for asset in ledger["assets"] if asset["asset_id"] == asset_id)


def _advisory_input(asset: dict[str, object], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "asset_id": asset["asset_id"],
        "immutable_hashes": copy.deepcopy(asset["immutable_hashes"]),
        "represented_label_authority": "benign_caller_claim",
        "trainer_reachable": False,
    }
    payload.update(overrides)
    return payload


def _train_contract(
    path: str,
    component: str,
    *,
    flag: str = "--data",
    advisory_asset: dict[str, object] | None = None,
    asset_ids: list[str] | None = None,
) -> dict[str, object]:
    script = {
        "BALL": "train_ball_detector.py",
        "COURT": "train_court_model.py",
        "EVENT": "train_event_head.py",
        "PERSON": "train_person_detector.py",
        "REID": "train_reid_model.py",
    }[component]
    return {
        "dispatch_id": "adversarial_dispatch",
        "inputs": [_advisory_input(advisory_asset)] if advisory_asset else [],
        "source_families": {"train": ["caller_train"], "holdout": ["caller_holdout"]},
        "baseline": {"metric": "f1", "threshold": ">=0.30"},
        "check": {"metric": "f1_delta", "threshold": ">=0.10"},
        "kill_threshold": {"metric": "hidden_fp", "threshold": "<=0.05"},
        "command": {
            "kind": "CPU",
            "argv": ["python", script, "--accelerator=cuda", flag, path],
            "asset_ids": [] if asset_ids is None else asset_ids,
        },
    }


def _first_path(asset: dict[str, object]) -> str:
    return asset["paths"][0]["path"]


def _errors_for(asset_id: str, component: str, *, flag: str = "--data") -> list[str]:
    ledger = _ledger()
    asset = _asset(ledger, asset_id)
    contract = _train_contract(
        _first_path(asset),
        component,
        flag=flag,
        advisory_asset=asset,
    )
    return audit_dispatch_contract(ledger, contract, repo_root=ROOT)


def test_seeded_ledger_validates_and_generated_view_round_trips(tmp_path: Path) -> None:
    ledger = _ledger()

    assert validate_ledger(ledger) == []
    assert verify_hashes(ledger, ROOT) == []
    expected = render_markdown(ledger)
    assert VIEW_PATH.read_text(encoding="utf-8") == expected

    generated = tmp_path / "DATA_LEDGER.md"
    generated.write_text(expected, encoding="utf-8")
    first_bytes = generated.read_bytes()
    generated.write_text(render_markdown(json.loads(json.dumps(ledger))), encoding="utf-8")
    assert generated.read_bytes() == first_bytes


def test_probe_protected_event_seed_refuses_gpu_despite_false_reachability() -> None:
    errors = _errors_for("protected_event_seed_50_20260713", "EVENT")

    assert any("state QUARANTINED refuses train use" in error for error in errors)
    assert any("ledger protection forbids trainer reachability" in error for error in errors)


def test_probe_eval_clips_refuses_gpu_despite_false_reachability() -> None:
    errors = _errors_for("eval_clips_ball_protected_4", "PERSON")

    assert any("state QUARANTINED refuses train use" in error for error in errors)
    assert any("PERSON=FORBID" in error for error in errors)


def test_probe_nc_licensed_person_asset_refuses_training() -> None:
    errors = _errors_for("roboflow_person_nc_20260706", "PERSON")

    assert any("state QUARANTINED refuses train use" in error for error in errors)
    assert any("PERSON=FORBID" in error for error in errors)


def test_probe_compare_only_gallery_refuses_without_clean_subset() -> None:
    errors = _errors_for("pbvision_gallery_20260719", "REID", flag="--teacher-data")

    assert any("mixed protected/compare identities require an immutable clean subset" in error for error in errors)


def test_probe_iynbdrs1jdk_court_pack_refuses_without_clean_subset() -> None:
    errors = _errors_for("court_diversity_100_20260712", "COURT")

    assert any("state BLOCKED refuses train use" in error for error in errors)
    assert any("mixed protected/compare identities require an immutable clean subset" in error for error in errors)


def test_probe_rejected_tt_sounds_refuses_event_training() -> None:
    errors = _errors_for("event_public_tt_sounds_20260713", "EVENT")

    assert any("state REJECTED refuses train use" in error for error in errors)
    assert any("EVENT=FORBID" in error for error in errors)


def test_probe_zero_decoded_refuses_gpu_when_asset_ids_are_empty() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "pbv_pickleball_teacher_events_20260720")
    contract = _train_contract(
        _first_path(asset),
        "EVENT",
        flag="--teacher-data",
        advisory_asset=asset,
        asset_ids=[],
    )

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("GPU asset has zero decoded rows: pbv_pickleball_teacher_events_20260720" in error for error in errors)


@pytest.mark.parametrize("gpu_argv", [["--device", "cuda:0"], ["--gpus", "1"]])
def test_zero_decoded_refuses_alternate_gpu_argv_forms(gpu_argv: list[str]) -> None:
    ledger = _ledger()
    asset = _asset(ledger, "pbv_pickleball_teacher_events_20260720")
    contract = _train_contract(
        _first_path(asset),
        "EVENT",
        flag="--teacher-data",
        advisory_asset=asset,
        asset_ids=[],
    )
    contract["command"]["argv"] = [
        "python",
        "train_event_head.py",
        *gpu_argv,
        "--teacher-data",
        _first_path(asset),
    ]

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("GPU asset has zero decoded rows: pbv_pickleball_teacher_events_20260720" in error for error in errors)


def test_probe_protected_argv_refuses_with_benign_declared_contract() -> None:
    ledger = _ledger()
    benign = _asset(ledger, "owner_event_labels_102_20260719")
    protected = _asset(ledger, "protected_event_seed_50_20260713")
    contract = _train_contract(
        _first_path(protected),
        "EVENT",
        advisory_asset=benign,
        asset_ids=[benign["asset_id"]],
    )

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("protected_event_seed_50_20260713" in error and "QUARANTINED" in error for error in errors)
    assert not any("owner_event_labels_102_20260719" in error and "protection forbids" in error for error in errors)


@pytest.mark.parametrize(
    "data_argv",
    [
        lambda path: f"-d{path}",
        lambda path: f"-d={path}",
        lambda path: f"--data={path}",
        lambda path: f"--teacher-data={path}",
    ],
)
def test_protected_path_refuses_concatenated_and_equals_forms(data_argv) -> None:
    ledger = _ledger()
    asset = _asset(ledger, "protected_event_seed_50_20260713")
    contract = _train_contract(_first_path(asset), "EVENT", advisory_asset=asset)
    contract["command"]["argv"] = [
        "python",
        "train_event_head.py",
        "--accelerator=cuda",
        data_argv(_first_path(asset)),
    ]

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any(
        "protected_event_seed_50_20260713" in error and "QUARANTINED" in error
        for error in errors
    )


def test_unsupported_concatenated_short_option_is_ambiguous_and_still_scanned() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "protected_event_seed_50_20260713")
    contract = _train_contract(_first_path(asset), "EVENT", advisory_asset=asset)
    contract["command"]["argv"] = [
        "python",
        "train_event_head.py",
        "--accelerator=cuda",
        f"-x{_first_path(asset)}",
    ]

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("unsupported dash-prefixed form is ambiguous" in error for error in errors)
    assert any("protected_event_seed_50_20260713" in error for error in errors)


def test_dispatch_fails_when_actual_argv_data_is_absent_from_ledger(tmp_path: Path) -> None:
    ledger = _ledger()
    unknown = tmp_path / "not_registered.jsonl"
    contract = _train_contract(str(unknown), "EVENT")

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("data-looking reference is absent from ledger" in error for error in errors)
    assert any("argv resolves no ledger asset" in error for error in errors)


def test_dispatch_fails_when_advisory_hash_differs_but_does_not_trust_it() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "owner_event_labels_102_20260719")
    contract = _train_contract(_first_path(asset), "EVENT", advisory_asset=asset)
    contract["inputs"][0]["immutable_hashes"][0]["digest"] = "0" * 64

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("hash differs from ledger" in error for error in errors)


def test_hash_verification_fails_after_input_bytes_change(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text('{"version": 1}\n', encoding="utf-8")
    ledger = _ledger()
    asset = copy.deepcopy(ledger["assets"][0])
    asset["immutable_hashes"] = [
        {
            "path": str(source),
            "algorithm": "sha256",
            "digest": hashlib.sha256(source.read_bytes()).hexdigest(),
            "role": "test manifest",
        }
    ]
    source.write_text('{"version": 2}\n', encoding="utf-8")

    errors = verify_hashes({"assets": [asset]}, ROOT)

    assert len(errors) == 1
    assert "sha256 differs" in errors[0]


def test_dispatch_overlap_is_ledger_derived_and_ignores_caller_family_strings() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "owner_event_labels_102_20260719")
    contract = _train_contract(_first_path(asset), "EVENT", advisory_asset=asset)
    contract["source_families"] = {"train": ["caller_says_clean"], "holdout": ["different_value"]}

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("ledger_partitions" in error and "owner_102_manifest" in error for error in errors)


def test_teacher_authority_is_derived_from_ledger_and_argv_role() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "pbv_pickleball_teacher_events_20260720")
    contract = _train_contract(_first_path(asset), "EVENT", flag="--labels", advisory_asset=asset)
    contract["inputs"][0]["represented_label_authority"] = "gold_labels"

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("ledger authority teacher cannot be used as ground truth" in error for error in errors)


def test_nonpass_overlap_coverage_refuses_unscoped_training() -> None:
    errors = _errors_for("roboflow_person_core_20260706", "PERSON")

    assert any("ledger overlap coverage NOT_RUN refuses unscoped train use" in error for error in errors)


@pytest.mark.parametrize("missing_key", ["baseline", "check", "kill_threshold"])
def test_dispatch_fails_when_required_gate_threshold_is_missing(missing_key: str) -> None:
    ledger = _ledger()
    asset = _asset(ledger, "owner_event_labels_102_20260719")
    contract = _train_contract(_first_path(asset), "EVENT", advisory_asset=asset)
    del contract[missing_key]

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any(f"dispatch.{missing_key}: metric and threshold are required" in error for error in errors)


def test_immutable_clean_subset_can_prove_mixed_identity_exclusion(tmp_path: Path) -> None:
    selector = tmp_path / "clean_train_manifest.json"
    selector.write_text('{"sources":["allowed_source"]}\n', encoding="utf-8")
    digest = hashlib.sha256(selector.read_bytes()).hexdigest()
    ledger = _ledger()
    asset = copy.deepcopy(_asset(ledger, "pbvision_gallery_20260719"))
    asset["paths"] = [{"path": str(tmp_path), "role": "synthetic mixed asset", "present": True}]
    asset["immutable_hashes"] = [
        {"path": str(selector), "algorithm": "sha256", "digest": digest, "role": "clean selector"}
    ]
    asset["source_lineage"]["original_sources"] = ["allowed_source", "heldout_source", "blocked_id"]
    asset["partitions"] = {
        "strategy": "immutable synthetic selector",
        "train": ["allowed_source"],
        "val": [],
        "test": ["heldout_source"],
    }
    asset["label_authority"] = ["human_gt"]
    asset["rights"]["component_rulings"]["EVENT"] = {"decision": "ALLOW", "ruling": "synthetic test"}
    asset["protection"] = {
        "trainer_forbidden": False,
        "identities": [
            {"identity": "blocked_id", "posture": "compare_only", "ruling": "synthetic test"}
        ],
        "clean_subsets": [
            {
                "selector_path": str(selector),
                "excluded_identities": ["blocked_id"],
                "train_families": ["allowed_source"],
                "holdout_families": ["heldout_source"],
                "allowed_components": ["EVENT"],
            }
        ],
        "overlap_check_coverage": {
            "method": "synthetic exact selector",
            "coverage_count": 3,
            "scope": "test fixture",
            "status": "PASS",
        },
    }
    ledger["assets"] = [asset]
    contract = _train_contract(str(selector), "EVENT", advisory_asset=asset)

    assert validate_ledger(ledger) == []
    assert audit_dispatch_contract(ledger, contract, repo_root=ROOT) == []


@pytest.mark.parametrize(
    "non_input_flag",
    ["--teacher-output", "--save-path", "--log-file", "--config"],
)
def test_clean_subset_under_non_input_role_never_grants_access(
    tmp_path: Path,
    non_input_flag: str,
) -> None:
    selector = tmp_path / "clean_train_manifest.json"
    selector.write_text('{"sources":["allowed_source"]}\n', encoding="utf-8")
    digest = hashlib.sha256(selector.read_bytes()).hexdigest()
    ledger = _ledger()
    asset = copy.deepcopy(_asset(ledger, "pbvision_gallery_20260719"))
    asset["paths"] = [{"path": str(tmp_path), "role": "synthetic mixed asset", "present": True}]
    asset["immutable_hashes"] = [
        {"path": str(selector), "algorithm": "sha256", "digest": digest, "role": "clean selector"}
    ]
    asset["source_lineage"]["original_sources"] = ["allowed_source", "blocked_id"]
    asset["partitions"] = {
        "strategy": "immutable synthetic selector",
        "train": ["allowed_source"],
        "val": [],
        "test": [],
    }
    asset["label_authority"] = ["human_gt"]
    asset["rights"]["component_rulings"]["EVENT"] = {
        "decision": "CONDITIONAL",
        "ruling": "synthetic test",
    }
    asset["protection"] = {
        "trainer_forbidden": False,
        "identities": [
            {"identity": "blocked_id", "posture": "compare_only", "ruling": "synthetic test"}
        ],
        "clean_subsets": [
            {
                "selector_path": str(selector),
                "excluded_identities": ["blocked_id"],
                "train_families": ["allowed_source"],
                "holdout_families": [],
                "allowed_components": ["EVENT"],
            }
        ],
        "overlap_check_coverage": {
            "method": "synthetic exact selector",
            "coverage_count": 2,
            "scope": "test fixture",
            "status": "PASS",
        },
    }
    ledger["assets"] = [asset]
    contract = _train_contract(str(selector), "EVENT", advisory_asset=asset)
    contract["command"]["argv"] = [
        "python",
        "train_event_head.py",
        "--accelerator=cuda",
        non_input_flag,
        str(selector),
    ]

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("requires a recognized data-bearing input role" in error for error in errors)
    assert any("mixed protected/compare identities require an immutable clean subset" in error for error in errors)


def test_every_resolved_data_reference_must_use_the_same_selector(tmp_path: Path) -> None:
    selector = tmp_path / "clean_train_manifest.json"
    selector.write_text('{"sources":["allowed_source"]}\n', encoding="utf-8")
    digest = hashlib.sha256(selector.read_bytes()).hexdigest()
    ledger = _ledger()
    asset = copy.deepcopy(_asset(ledger, "pbvision_gallery_20260719"))
    asset["paths"] = [{"path": str(tmp_path), "role": "synthetic mixed asset", "present": True}]
    asset["immutable_hashes"] = [
        {"path": str(selector), "algorithm": "sha256", "digest": digest, "role": "clean selector"}
    ]
    asset["source_lineage"]["original_sources"] = ["allowed_source", "blocked_id"]
    asset["partitions"] = {
        "strategy": "immutable synthetic selector",
        "train": ["allowed_source"],
        "val": [],
        "test": [],
    }
    asset["label_authority"] = ["human_gt"]
    asset["rights"]["component_rulings"]["EVENT"] = {
        "decision": "CONDITIONAL",
        "ruling": "synthetic test",
    }
    asset["protection"] = {
        "trainer_forbidden": False,
        "identities": [
            {"identity": "blocked_id", "posture": "compare_only", "ruling": "synthetic test"}
        ],
        "clean_subsets": [
            {
                "selector_path": str(selector),
                "excluded_identities": ["blocked_id"],
                "train_families": ["allowed_source"],
                "holdout_families": [],
                "allowed_components": ["EVENT"],
            }
        ],
        "overlap_check_coverage": {
            "method": "synthetic exact selector",
            "coverage_count": 2,
            "scope": "test fixture",
            "status": "PASS",
        },
    }
    ledger["assets"] = [asset]
    contract = _train_contract(str(selector), "EVENT", advisory_asset=asset)
    contract["command"]["argv"].extend(["--source-data", str(tmp_path)])

    errors = audit_dispatch_contract(ledger, contract, repo_root=ROOT)

    assert any("every resolved data reference must use the same immutable selector" in error for error in errors)


def test_never_queued_report_is_sorted_and_excludes_ruled_assets() -> None:
    ledger = _ledger()
    first = copy.deepcopy(_asset(ledger, "owner_event_labels_102_20260719"))
    second = copy.deepcopy(_asset(ledger, "event_bootstrap_audio_20260713"))
    ruled = copy.deepcopy(_asset(ledger, "event_public_tt_sounds_20260713"))
    for asset, asset_id in ((first, "z_ready"), (second, "a_ready")):
        asset["asset_id"] = asset_id
        asset["acquired_utc"] = "2026-07-18T00:00:00Z"
        asset["consumers"] = []
        asset["state"] = "READY"
        asset["state_reason"] = "ready_for_named_consumer"
    ruled["asset_id"] = "ruled_rejected"
    ruled["consumers"] = []
    ruled["state"] = "REJECTED"
    ledger["assets"] = [first, second, ruled]

    report = never_queued_assets(ledger, as_of=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert [row["asset_id"] for row in report] == ["a_ready", "z_ready"]


def test_protected_eval_person_binding_lists_hashes_and_exact_box_total() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "eval_clips_ball_protected_4")
    bindings = {
        binding["path"]: binding
        for binding in asset["immutable_hashes"]
        if binding["path"].endswith("person_ground_truth.json")
    }

    assert len(bindings) == 4
    label_counts = []
    for relative_path in sorted(bindings):
        payload = load_json(ROOT / relative_path)
        label_counts.append(sum(len(frame["labels"]) for frame in payload["frames"]))

    assert sorted(label_counts) == [1200, 2400, 3379, 4480]
    assert sum(label_counts) == asset["counts"]["label_count"] == 11459
    assert asset["counts"]["byte_count"] == 24804487
    assert all(path["path"] != "eval_clips/ball" for path in asset["paths"])


def test_roboflow_person_selector_is_exact_and_consumer_is_ball_only() -> None:
    ledger = _ledger()
    person = _asset(ledger, "roboflow_person_core_20260706")
    ball = _asset(ledger, "roboflow_ball_core_pretrain_20260706")
    selector = load_json(PERSON_SELECTOR_PATH)
    source = load_json(PERSON_SOURCE_INDEX_PATH)
    expected_ids = [
        sample["sample_id"]
        for sample in source["samples"]
        if sample["bucket"] == "core_pickleball"
        and sample["source_slug"] != "testing-esifc/pickle-ball-labeling-mff1d"
    ]

    assert selector["artifact_type"] == "racketsport_roboflow_row_selector"
    assert selector["label_kind"] == "person"
    assert selector["row_count"] == len(selector["sample_ids"]) == 15312
    assert len(set(selector["sample_ids"])) == 15312
    assert selector["sample_ids"] == expected_ids
    assert person["consumers"] == []
    assert person["state"] == "BLOCKED"
    assert person["counts"]["dedup_kept_count"] == 15312
    assert ball["state"] == "CONSUMED"
    assert ball["counts"]["raw_count"] == 34658
    assert [consumer["rows_loaded"] for consumer in ball["consumers"]] == [34658]
    assert all("BALL pretrain" in consumer["lane"] for consumer in ball["consumers"])


def test_unrelated_eval_derivatives_have_separate_ledger_lineage() -> None:
    ledger = _ledger()
    protected = _asset(ledger, "eval_clips_ball_protected_4")
    owner = _asset(ledger, "owner_img_1605_court_review_20260721")
    pbvision = _asset(ledger, "pbvision_gallery_20260719")

    assert owner["counts"]["byte_count"] == 520522
    assert any("owner_IMG_1605" in path["path"] for path in owner["paths"])
    assert any("pbvision_11min_20260713" in path["path"] for path in pbvision["paths"])
    assert not any(
        "owner_IMG_1605" in path["path"] or "pbvision_11min_20260713" in path["path"]
        for path in protected["paths"]
    )


def test_direct_cli_reference_and_seeded_snapshot() -> None:
    help_result = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--dispatch-contract" in help_result.stdout
    assert "--check-view" in help_result.stdout

    audit_result = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--ledger",
            "runs/manager/data_ledger.json",
            "--check-view",
            "runs/manager/DATA_LEDGER.md",
            "--as-of",
            "2026-07-21T23:59:59Z",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(audit_result.stdout)
    assert audit_result.returncode == 0, report
    assert report["status"] == "pass"
    assert report["asset_count"] == 27
    assert report["never_queued_count"] == 0
    assert report["state_distribution"] == {
        "BLOCKED": 9,
        "CONSUMED": 6,
        "QUARANTINED": 6,
        "REJECTED": 6,
    }
    assert report["never_queued"] == sorted(
        report["never_queued"], key=lambda row: (row["acquired_utc"], row["asset_id"])
    )
