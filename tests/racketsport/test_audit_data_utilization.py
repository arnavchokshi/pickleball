from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import scripts.racketsport.audit_data_utilization as audit_module
from scripts.racketsport.audit_data_utilization import (
    REQUIRED_CONTRACT_ASSET_IDS,
    audit_dispatch_contract,
    disposition_violations,
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
PBVISION_MANIFEST_PATH = ROOT / "data" / "pbvision_gallery_20260719" / "MANIFEST.json"
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


def _run_enforced_ledger(
    tmp_path: Path,
    ledger: dict[str, object],
    *,
    filename: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    ledger_path = tmp_path / filename
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--repo-root",
            str(ROOT),
            "--ledger",
            str(ledger_path),
            "--enforce-queued",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, json.loads(result.stdout)


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


def test_probe_nc_person_asset_is_not_blocked_by_license_but_still_requires_technical_gates() -> None:
    errors = _errors_for("roboflow_person_nc_20260706", "PERSON")

    assert any("state BLOCKED refuses train use" in error for error in errors)
    assert any("PERSON=CONDITIONAL" in error for error in errors)
    assert not any("license" in error.casefold() for error in errors)


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
    asset = copy.deepcopy(_asset(ledger, "event_public_f3set_20260713"))
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
    errors = _errors_for("roboflow_person_nc_20260706", "PERSON")

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
        asset.pop("disposition", None)
    ruled["asset_id"] = "ruled_rejected"
    ruled["consumers"] = []
    ruled["state"] = "REJECTED"
    ledger["assets"] = [first, second, ruled]

    report = never_queued_assets(ledger, as_of=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert [row["asset_id"] for row in report] == ["a_ready", "z_ready"]


def test_current_ledger_has_zero_disposition_violations() -> None:
    ledger = _ledger()
    assert disposition_violations(ledger) == []
    queued_actions = [
        action
        for asset in ledger["assets"]
        if "not_usable_because" not in asset["disposition"]
        for action in [
            asset["disposition"],
            *asset["disposition"].get("secondary_queue_actions", []),
        ]
    ]
    assert queued_actions
    assert all(isinstance(action["training_intent"], bool) for action in queued_actions)


def test_data_debt_required_mappings_are_pinned() -> None:
    ledger = _ledger()

    assert ledger["schema_version"] == 3
    assert ledger["policy_directives"]["license_is_state_gate"] is False
    assert all(asset["license_fyi"] for asset in ledger["assets"])

    gallery = _asset(ledger, "pbvision_gallery_20260719")
    assert gallery["disposition"]["consumer_track"] == "A"
    allowed_gallery_ids = {
        "0tmdeghtfvjx",
        "143sf3gdwxsa",
        "98z43hspqz13",
        "bewqc0glhgpq",
        "pldtjpw3h0jw",
        "st0epgnab7dr",
        "td2szayjwtrj",
        "tqjlrcntpjvt",
        "utasf5hnozwz",
        "xkadsq9bli3h",
    }
    assert set(gallery["protection"]["training_allowed_ids"]) == allowed_gallery_ids
    assert all(
        video_id in gallery["disposition"]["next_queue_action"]
        for video_id in allowed_gallery_ids
    )
    assert "iottnc0h3ekn" not in gallery["disposition"]["next_queue_action"]
    assert "o4dee9dn0ccr" not in gallery["disposition"]["next_queue_action"]
    assert {action["consumer_track"] for action in gallery["disposition"]["secondary_queue_actions"]} == {
        "A",
        "B",
    }
    assert "10-frame owner-tap spot-check" in gallery["disposition"]["next_queue_action"]

    opentt = _asset(ledger, "event_public_extended_opentt_20260713")
    assert opentt["counts"]["label_count"] == 52987
    assert opentt["disposition"]["consumer_track"] == "B"
    assert "official control" in opentt["disposition"]["next_queue_action"]
    assert "w7_ballretrain" in opentt["disposition"]["consumer_evidence"]

    track_d_inventory_ids = {
        "event_public_f3set_20260713",
        "event_public_golfdb_20260713",
        "event_public_padeltracker100_20260713",
        "event_public_shuttleset_20260713",
        "event_public_squash_figshare_20260713",
        "event_public_shuttlecock_zenodo_20260713",
        "event_public_tt_sounds_20260713",
    }
    for asset_id in track_d_inventory_ids - {"event_public_padeltracker100_20260713"}:
        assert _asset(ledger, asset_id)["disposition"]["consumer_track"] == "D"
    padel = _asset(ledger, "event_public_padeltracker100_20260713")
    assert padel["counts"]["dedup_kept_count"] == 906
    assert padel["disposition"]["consumer_track"] == "C"
    assert [action["consumer_track"] for action in padel["disposition"]["secondary_queue_actions"]] == ["D"]

    online = _asset(ledger, "online_harvest_20260706")
    assert online["disposition"]["consumer_track"] == "A"
    # Court-calibration derivatives are now fenced OUT by path-narrowing (a stronger guard
    # than the former disposition_derives marker): only the rally media + harvest manifest are
    # registered paths, so court_calibrations/ + prelabels/ cannot be a training input at all.
    assert {p["path"] for p in online["paths"]} == {
        "data/online_harvest_20260706/rallies",
        "data/online_harvest_20260706/manifest.json",
    }
    assert online["rights"]["component_rulings"]["COURT"]["decision"] == "FORBID"
    assert online["rights"]["component_rulings"]["EVENT"]["decision"] == "ALLOW"
    assert "audit-only" in online["disposition"]["next_queue_action"]
    assert "exclude every derivative from training" in online["disposition"]["next_queue_action"]
    # The only training authorization is the secondary Track D event action, scoped to rallies.
    assert any(
        action["consumer_track"] == "D"
        and action.get("training_intent") is True
        and "data/online_harvest_20260706/rallies" in action["next_queue_action"]
        for action in online["disposition"]["secondary_queue_actions"]
    )
    assert "CVAT_CLOSED_TO_PERSON_TASKS" in _asset(
        ledger, "online_harvest_person_gap_20260706"
    )["disposition"]["not_usable_because"]
    assert _asset(ledger, "court_diversity_100_20260712")["disposition"]["consumer_track"] == "A"
    assert _asset(ledger, "court_keypoints_6_20260707")["protection"]["trainer_forbidden"] is True

    replay = _asset(ledger, "pbv_replay_xkadsq9bli3h_20260720")
    assert replay["state"] == "CONSUMED"
    assert replay["disposition"]["consumer_track"] == "B"
    assert "secondary_queue_actions" not in replay["disposition"]
    assert "remaining sources tqjlrcntpjvt and xkadsq9bli3h" in replay["disposition"][
        "next_queue_action"
    ]
    assert replay["consumers"][0]["result_path"] == (
        "runs/lanes/pbv_replay_20260720/MANAGER_RULING.md"
    )
    assert "81.8%" in replay["consumers"][0]["metric"]

    assert "PROVEN_NEGATIVE_TRANSFER" in _asset(
        ledger, "roboflow_ball_core_pretrain_20260706"
    )["disposition"]["not_usable_because"]
    assert _asset(ledger, "roboflow_court_taxonomy_20260706")["disposition"]["consumer_track"] == "A"
    person_core = _asset(ledger, "roboflow_person_core_20260706")
    assert person_core["state"] == "REJECTED"
    assert "PERSON_RF_POOL_TOO_THIN" in person_core["disposition"]["not_usable_because"]
    assert "Any Track C aux/eval use requires a NEW ruling" in person_core["disposition"][
        "not_usable_because"
    ]
    golfdb = _asset(ledger, "event_public_golfdb_20260713")
    assert "BLOCKED_NO_LABEL_MAPPED_SOURCE_RESOLVED_PIXELS" in golfdb["state_reason"]
    assert "test_video.mp4 exists" in golfdb["state_reason"]
    assert _asset(ledger, "eval_clips_ball_protected_4")["disposition"]["consumer_track"] == "C"
    assert "NO_MEDIA_ON_DISK" in _asset(
        ledger, "data_testclips_metadata_4"
    )["disposition"]["not_usable_because"]


def test_person_negative_and_license_only_unblock_are_pinned() -> None:
    ledger = _ledger()
    core = _asset(ledger, "roboflow_person_core_20260706")
    formerly_nc = _asset(ledger, "roboflow_person_nc_20260706")
    pointer = _asset(ledger, "person_mixed_pool_no_lift_20260722")

    assert core["state"] == "REJECTED"
    assert core["protection"]["trainer_forbidden"] is True
    assert core["protection"]["overlap_check_coverage"]["status"] == "PASS"
    assert core["protection"]["overlap_check_coverage"]["coverage_count"] == 45844128
    core_ruling = core["disposition"]["not_usable_because"]
    assert "PERSON_RF_POOL_TOO_THIN" in core_ruling
    assert "REJECTED_FOR_TRAINING" in core_ruling
    assert "P2: NO_ATTEMPT_PREREQ, permanently closed for this export" in core_ruling
    assert "protected-collision audit ALREADY PASSED" in core_ruling
    assert "Human quality card: NOT_COMPLETED_PROTOCOL" in core_ruling
    assert "runs/lanes/person_p1_roboflow_20260721/RULING.md" in core_ruling
    assert "runs/lanes/person_p1_roboflow_20260721/report_fix2.json" in core_ruling
    assert formerly_nc["state"] == "BLOCKED"
    assert formerly_nc["protection"]["trainer_forbidden"] is False
    assert "only remaining blocker is an exhaustive protected-collision audit" in formerly_nc["state_reason"]
    for asset in (core, formerly_nc, pointer):
        finding = next(
            finding
            for finding in asset["named_findings"]
            if finding["finding"] == "PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED"
        )
        assert "13.5x" in finding["interpretation"]
        assert "6.75x" in finding["interpretation"]
        assert "od8al precision -0.1924 (F1 -0.0842) LOSS" in finding["interpretation"]
        assert "hemel_test F1 +0.046 WIN" in finding["interpretation"]
        assert len(finding["evidence"]) == 2
        assert finding["evidence"][0] == "runs/lanes/person_mixed_20260722/gpu_phase_report.json"


def test_pbvision_usage_posture_preserves_protocol_quarantines() -> None:
    manifest = load_json(PBVISION_MANIFEST_PATH)
    posture = manifest["usage_posture"]

    assert "PBV-FULL-USAGE-20260720" in posture
    assert "Training and commercial use authorized" in posture
    assert "83gyqyc10y8f/iottnc0h3ekn/o4dee9dn0ccr" in posture
    assert "compare/eval only" in posture
    assert "never protected-eval" in posture
    assert "never redistributed" in posture
    assert manifest["usage_posture_provenance_note"].startswith("2026-07-22:")


def test_enforce_queued_cli_lists_every_missing_asset(tmp_path: Path) -> None:
    ledger = _ledger()
    missing_ids = [ledger["assets"][0]["asset_id"], ledger["assets"][1]["asset_id"]]
    for asset in ledger["assets"][:2]:
        asset.pop("disposition")
    ledger_path = tmp_path / "missing_dispositions.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--ledger",
            str(ledger_path),
            "--enforce-queued",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert result.returncode != 0
    assert report["queue_violation_count"] == 2
    assert all(any(asset_id in error for error in report["queue_errors"]) for asset_id in missing_ids)


def test_enforce_queued_rejects_deleted_contract_asset_exact_review_mutant(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    ledger["assets"] = [
        asset
        for asset in ledger["assets"]
        if asset["asset_id"] != "pbvision_gallery_20260719"
    ]

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="drop_contract_asset.json",
    )

    assert result.returncode != 0
    assert set(REQUIRED_CONTRACT_ASSET_IDS) - {
        asset["asset_id"] for asset in ledger["assets"]
    } == {"pbvision_gallery_20260719"}
    assert any(
        "pbvision_gallery_20260719 contract baseline: required asset is missing" in error
        for error in report["queue_errors"]
    )


def test_enforce_queued_rejects_nonexistent_disposition_evidence(tmp_path: Path) -> None:
    ledger = _ledger()
    _asset(ledger, "event_public_f3set_20260713")["disposition"][
        "consumer_evidence"
    ] = "does/not/exist.json"

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="nonexistent_evidence.json",
    )

    assert result.returncode != 0
    assert any(
        "event_public_f3set_20260713" in error
        and "path does not exist: does/not/exist.json" in error
        for error in report["queue_errors"]
    )


def test_enforce_queued_rejects_protected_training_exact_review_mutant(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    _asset(ledger, "eval_clips_ball_protected_4")["disposition"] = {
        "consumer_track": "C",
        "next_queue_action": "Train on all protected rows immediately.",
        "consumer_evidence": "does/not/exist.json",
    }

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="protected_to_train.json",
    )

    assert result.returncode != 0
    errors = report["queue_errors"]
    assert any("eval_clips_ball_protected_4" in error and "path does not exist" in error for error in errors)
    assert any(
        "eval_clips_ball_protected_4" in error
        and "global forbidden reference eval_clips_ball_protected_4" in error
        for error in errors
    )


def test_enforce_queued_rejects_trivial_reason_exact_review_mutant(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    _asset(ledger, "person_mixed_pool_no_lift_20260722")["disposition"] = {
        "not_usable_because": "x",
        "secondary_queue_actions": [],
    }

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="trivial_reason.json",
    )

    assert result.returncode != 0
    errors = report["queue_errors"]
    assert any("substantive CODE: explanation is required" in error for error in errors)
    assert any("ruled-out disposition cannot have secondary queue actions" in error for error in errors)


def test_enforce_queued_rejects_frozen_audit_derivative_training_mutant(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    _asset(ledger, "online_harvest_20260706")["disposition"]["next_queue_action"] = (
        "Add the six source calibration JSON files plus coverage_report.json to the "
        "Track A court training-pool adapter."
    )

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="frozen_audit_to_train.json",
    )

    assert result.returncode != 0
    # Path-narrowing removed the per-asset derivative marker, so the frozen court derivative is
    # now caught by the global forbidden-reference guard (online_harvest's disposition evidence
    # binds the trainer_forbidden court asset). The mutant is still refused and the error names
    # the frozen court asset.
    assert any(
        "online_harvest_20260706" in error
        and "court_keypoints_6_20260707" in error
        and "forbidden reference" in error
        for error in report["queue_errors"]
    )


def test_enforce_queued_rejects_pbvision_all_twelve_training_mutant(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    _asset(ledger, "pbvision_gallery_20260719")["disposition"]["next_queue_action"] = (
        "After the owner-tap spot-check, add all 12 videos with court-calibration keypoints "
        "to the Track A court-keypoint retrain pool."
    )

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="pbvision_all_twelve_to_train.json",
    )

    assert result.returncode != 0
    assert any(
        "pbvision_gallery_20260719" in error
        and "training action does not enumerate the full allowed ID set" in error
        for error in report["queue_errors"]
    )


@pytest.mark.parametrize(
    ("forbidden_reference", "action_text"),
    [
        (
            "eval_clips_ball_protected_4",
            "Train on protected asset eval_clips_ball_protected_4 immediately.",
        ),
        (
            "iottnc0h3ekn",
            "Train on compare-only clip iottnc0h3ekn immediately.",
        ),
        (
            "court_keypoints_6_20260707",
            "Train on frozen-audit asset court_keypoints_6_20260707 immediately.",
        ),
    ],
)
def test_enforce_queued_rejects_cross_row_global_forbidden_reference_mutants(
    tmp_path: Path,
    forbidden_reference: str,
    action_text: str,
) -> None:
    ledger = _ledger()
    _asset(ledger, "event_public_f3set_20260713")["disposition"][
        "secondary_queue_actions"
    ] = [
        {
            "consumer_track": "D",
            "next_queue_action": action_text,
            "consumer_evidence": "data/event_public_20260713/f3set/manifest.json",
        }
    ]

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename=f"cross_row_{forbidden_reference}.json",
    )

    assert result.returncode != 0
    assert any(
        "event_public_f3set_20260713" in error
        and "global forbidden reference" in error
        and forbidden_reference in error
        for error in report["queue_errors"]
    )


def test_enforce_queued_rejects_gradient_supervision_synonym_mutant(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    _asset(ledger, "eval_clips_ball_protected_4")["disposition"] = {
        "consumer_track": "C",
        "next_queue_action": "Use all 11,459 protected rows as gradient-update supervision immediately.",
        "consumer_evidence": "eval_clips/ball/manifest.json",
    }

    result, report = _run_enforced_ledger(
        tmp_path,
        ledger,
        filename="gradient_supervision_synonym.json",
    )

    assert result.returncode != 0
    assert any(
        "eval_clips_ball_protected_4" in error
        and "global forbidden reference eval_clips_ball_protected_4" in error
        for error in report["queue_errors"]
    )


def test_protected_eval_person_binding_uses_recorded_values_only() -> None:
    ledger = _ledger()
    asset = _asset(ledger, "eval_clips_ball_protected_4")
    recorded = {
        (binding["path"], binding["algorithm"]): binding["digest"]
        for binding in asset["immutable_hashes"]
    }
    expected_recorded = {
        ("eval_clips/ball/manifest.json", "sha256"): (
            "1efe76ed15c6fb2fe4eedfbdb7ad6e4cd567af4c08a58e87cc35a859daa15762"
        ),
        (
            "runs/cvat_imports/2026_06_30/burlington_gold_0300_low_steep_corner/"
            "person_ground_truth.json",
            "sha256",
        ): "4aafefd495b1023c4a0d4616a51d42caebc3a38f50dbf5be1f9104f72b42c399",
        (
            "runs/cvat_imports/2026_06_30/indoor_doubles_fwuks_0500_long_mid_baseline/"
            "person_ground_truth.json",
            "sha256",
        ): "d47726e1457f53163cf547401a37610ff2a991be47abb39350bad870df8d3132",
        (
            "runs/cvat_imports/2026_06_30/outdoor_webcam_iynbd_1500_long_high_baseline/"
            "person_ground_truth.json",
            "sha256",
        ): "be0123400cc4bcab5de7711b52213323e60356f5dc98b9f25297a80a5c52ed37",
        (
            "runs/cvat_imports/2026_06_30/wolverine_mixed_0200_mid_steep_corner/"
            "person_ground_truth.json",
            "sha256",
        ): "96b76c8f5eb0a6815ef443400a65b54396ccb1f3a05e699e2ad57f0db48178b7",
    }

    assert recorded == expected_recorded
    assert asset["protection"]["trainer_forbidden"] is True
    assert asset["partitions"]["train"] == []


def test_verify_hashes_uses_recorded_only_integrity_for_protected_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger()
    recorded_by_path = {
        (ROOT / binding["path"]).resolve(): binding["digest"]
        for asset in ledger["assets"]
        for binding in asset["immutable_hashes"]
    }
    protected_paths = {
        (ROOT / binding["path"]).resolve()
        for asset in ledger["assets"]
        if audit_module._uses_recorded_only_integrity(asset)
        for binding in asset["immutable_hashes"]
    }
    hashed_paths: set[Path] = set()

    def recorded_digest(path: Path, algorithm: str) -> str:
        assert algorithm == "sha256"
        resolved = path.resolve()
        hashed_paths.add(resolved)
        return recorded_by_path[resolved]

    monkeypatch.setattr(audit_module, "sha_digest", recorded_digest)

    assert verify_hashes(ledger, ROOT) == []
    assert hashed_paths.isdisjoint(protected_paths)

    shared_nonprotected_assets = [
        asset
        for asset in ledger["assets"]
        if not audit_module._uses_recorded_only_integrity(asset)
        and any(
            (ROOT / binding["path"]).resolve() in protected_paths
            for binding in asset["immutable_hashes"]
        )
    ]
    assert shared_nonprotected_assets
    hashed_paths.clear()

    assert verify_hashes(
        {"assets": shared_nonprotected_assets},
        ROOT,
        protection_ledger=ledger,
    ) == []
    assert hashed_paths.isdisjoint(protected_paths)


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
    assert person["state"] == "REJECTED"
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
    assert "--enforce-queued" in help_result.stdout

    audit_result = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--ledger",
            "runs/manager/data_ledger.json",
            "--check-view",
            "runs/manager/DATA_LEDGER.md",
            "--enforce-queued",
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
    assert report["asset_count"] == 32
    assert report["never_queued_count"] == 0
    assert report["queue_violation_count"] == 0
    assert report["queue_errors"] == []
    assert report["state_distribution"] == {
        "BLOCKED": 10,
        "CONSUMED": 7,
        "DEFERRED_WITH_REASON": 2,
        "QUARANTINED": 5,
        "READY": 1,
        "REJECTED": 7,
    }
    assert report["never_queued"] == sorted(
        report["never_queued"], key=lambda row: (row["acquired_utc"], row["asset_id"])
    )
