#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
LANE_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[4]
OUT_DIR = LANE_ROOT / "out"
MUTANTS_DIR = LANE_ROOT / "mutants"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MUTANTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "mutation": mutation_proof(),
        "fallback": fallback_proof(),
        "threshold_stat": threshold_stat_proof(),
        "escape_hatch": escape_hatch_proof(),
    }
    write_json(OUT_DIR / "adversarial_proofs_summary.json", results)

    required = [
        results["mutation"]["omit_pred_cam_t"]["verdict"] == "tests_failed",
        results["mutation"]["double_apply_pred_cam_t"]["verdict"] == "tests_failed",
        results["fallback"]["absent_index"]["raw_emit_source"]["status"] == "not_found",
        results["fallback"]["absent_index"]["captured_ground_call"]["pred_cam_t"] is None,
        results["fallback"]["unreadable_index"]["raised"] == "HarnessInputError",
        results["threshold_stat"]["byte_equivalent"] is True,
        results["escape_hatch"]["raw_default_false"]["pred_cam_t_already_applied"] is False,
        results["escape_hatch"]["raw_source_precedence_over_body_true"]["pred_cam_t_already_applied"] is False,
        results["escape_hatch"]["flag_without_translation_ignored"]["pred_cam_t"] is None,
        results["escape_hatch"]["flag_without_translation_ignored"]["pred_cam_t_already_applied"] is False,
        results["escape_hatch"]["explicit_true_only"]["pred_cam_t_already_applied"] is True,
    ]
    return 0 if all(required) else 1


def mutation_proof() -> dict[str, Any]:
    source = REPO_ROOT / "scripts" / "racketsport" / "gate_check_body_decode.py"
    text = source.read_text(encoding="utf-8")

    omit_text = text.replace(
        "                pred_cam_t=frame.pred_cam_t,\n"
        "                pred_cam_t_already_applied=frame.pred_cam_t_already_applied,\n",
        "",
        1,
    )
    double_text = text.replace(
        '                joints_camera=decoded["joints_camera"][0],\n'
        '                vertices_camera=decoded["vertices_camera"][0] if decoded["vertices_camera"] is not None else [],\n',
        '                joints_camera=mhr_decode.apply_pred_cam_t_once(\n'
        '                    decoded["joints_camera"][0],\n'
        "                    pred_cam_t=frame.pred_cam_t,\n"
        "                    already_applied=False,\n"
        "                ),\n"
        "                vertices_camera=mhr_decode.apply_pred_cam_t_once(\n"
        '                    decoded["vertices_camera"][0] if decoded["vertices_camera"] is not None else [],\n'
        "                    pred_cam_t=frame.pred_cam_t,\n"
        "                    already_applied=False,\n"
        "                ),\n",
        1,
    )

    cases = {
        "omit_pred_cam_t": omit_text,
        "double_apply_pred_cam_t": double_text,
    }
    results: dict[str, Any] = {}
    for name, mutant_text in cases.items():
        if mutant_text == text:
            results[name] = {"verdict": "mutation_not_applied"}
            continue
        mutant_root = MUTANTS_DIR / name
        mutant_file = mutant_root / "scripts" / "racketsport" / "gate_check_body_decode.py"
        mutant_test_file = mutant_root / "tests" / "racketsport" / "test_gate_check_body_decode.py"
        mutant_conftest = mutant_root / "conftest.py"
        if mutant_root.exists():
            shutil.rmtree(mutant_root)
        mutant_file.parent.mkdir(parents=True, exist_ok=True)
        mutant_file.write_text(mutant_text, encoding="utf-8")
        mutant_test_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "tests" / "racketsport" / "test_gate_check_body_decode.py", mutant_test_file)
        mutant_conftest.write_text(
            "from __future__ import annotations\n"
            "import importlib.util\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "mutant_path = Path(__file__).parent / 'scripts' / 'racketsport' / 'gate_check_body_decode.py'\n"
            "spec = importlib.util.spec_from_file_location('scripts.racketsport.gate_check_body_decode', mutant_path)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules['scripts.racketsport.gate_check_body_decode'] = module\n"
            "assert spec.loader is not None\n"
            "spec.loader.exec_module(module)\n"
            "try:\n"
            "    import scripts.racketsport as package\n"
            "    package.gate_check_body_decode = module\n"
            "except Exception:\n"
            "    pass\n",
            encoding="utf-8",
        )
        mutant_tests = [
            "tests/racketsport/test_gate_check_body_decode.py::test_gate_1b_regrounding_uses_raw_pred_cam_t_once",
            "tests/racketsport/test_gate_check_body_decode.py::test_gate_1b_respects_pred_cam_t_already_applied_escape_hatch",
        ]
        import_check = run_cmd(
            [
                str(PYTHON),
                "-c",
                "import scripts.racketsport.gate_check_body_decode as g; print(g.__file__)",
            ],
            env=mutant_env(mutant_root),
            cwd=mutant_root,
        )
        pytest_result = run_cmd([str(PYTHON), "-m", "pytest", *mutant_tests, "-q"], env=mutant_env(mutant_root), cwd=mutant_root)
        write_text(OUT_DIR / f"{name}_pytest.stdout.txt", pytest_result["stdout"])
        write_text(OUT_DIR / f"{name}_pytest.stderr.txt", pytest_result["stderr"])
        results[name] = {
            "mutant_path": str(mutant_file.relative_to(REPO_ROOT)),
            "imported_module": import_check["stdout"].strip(),
            "command": pytest_result["command"],
            "returncode": pytest_result["returncode"],
            "verdict": "tests_failed" if pytest_result["returncode"] != 0 else "tests_passed_vacuous",
            "stdout_path": str((OUT_DIR / f"{name}_pytest.stdout.txt").relative_to(REPO_ROOT)),
            "stderr_path": str((OUT_DIR / f"{name}_pytest.stderr.txt").relative_to(REPO_ROOT)),
        }
    write_json(OUT_DIR / "mutation_proof.json", results)
    return results


def fallback_proof() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.racketsport import gate_check_body_decode as gate

    fixture_dir = OUT_DIR / "fallback_fixture"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)
    body_mesh_path = fixture_dir / "body_mesh.json"
    court_cal_path = fixture_dir / "court_calibration.json"
    write_json(body_mesh_path, fixture_body_mesh())
    write_json(court_cal_path, {"stub": True})

    captured_ground_calls: list[dict[str, Any]] = []

    class DummyDecoder:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def decode_euler_frame(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "joints_camera": [[[1.0, 2.0, 3.0]]],
                "vertices_camera": [[[4.0, 5.0, 6.0]]],
            }

        def mesh_skeleton_divergence_mm(self, **_kwargs: Any) -> dict[str, Any]:
            return {"p95_mm": 1.25}

    def fake_gate_1a(_global_orient: Any, _body_pose: Any) -> dict[str, Any]:
        return {"gate": gate.GATE_1A_NAME, "max_abs_error_deg": 0.0, "passed": True}

    def fake_ground(**kwargs: Any) -> dict[str, Any]:
        captured_ground_calls.append(
            {
                "pred_cam_t": kwargs.get("pred_cam_t"),
                "pred_cam_t_already_applied": kwargs.get("pred_cam_t_already_applied"),
            }
        )
        return {
            "joints_world": kwargs["joints_camera"],
            "vertices_world": kwargs["vertices_camera"],
        }

    originals = {
        "runtime_available": gate.mhr_decode.MHR_RUNTIME_AVAILABLE,
        "runtime_import_error": gate.mhr_decode.MHR_RUNTIME_IMPORT_ERROR,
        "decoder": gate.mhr_decode.MHRDecoder,
        "gate_1a": gate.mhr_decode.gate_1a_euler_round_trip,
        "ground": gate.mhr_decode.ground_decoded_camera_frame,
        "model_validate": gate.CourtCalibration.model_validate,
        "provenance": gate.build_decoder_provenance,
    }
    try:
        gate.mhr_decode.MHR_RUNTIME_AVAILABLE = True
        gate.mhr_decode.MHR_RUNTIME_IMPORT_ERROR = None
        gate.mhr_decode.MHRDecoder = DummyDecoder
        gate.mhr_decode.gate_1a_euler_round_trip = fake_gate_1a
        gate.mhr_decode.ground_decoded_camera_frame = fake_ground
        gate.CourtCalibration.model_validate = classmethod(lambda cls, payload: object())
        gate.build_decoder_provenance = lambda **_kwargs: {"stubbed_for_proof": True, "mhr_runtime_available": True}

        parser = gate.build_arg_parser()
        absent_args = parser.parse_args(
            [
                "--body-mesh",
                str(body_mesh_path),
                "--court-calibration",
                str(court_cal_path),
                "--out",
                str(fixture_dir / "absent_report.json"),
                "--scale-source",
                "field",
            ]
        )
        absent_report = gate.run(absent_args)

        unreadable: dict[str, Any]
        unreadable_args = parser.parse_args(
            [
                "--body-mesh",
                str(body_mesh_path),
                "--court-calibration",
                str(court_cal_path),
                "--out",
                str(fixture_dir / "unreadable_report.json"),
                "--scale-source",
                "field",
                "--sam3d-output-index",
                str(fixture_dir / "does_not_exist" / "index.json"),
            ]
        )
        try:
            gate.run(unreadable_args)
        except Exception as exc:  # noqa: BLE001 - this proof records the exact hard-fail type/message.
            unreadable = {"raised": type(exc).__name__, "message": str(exc)}
        else:
            unreadable = {"raised": None, "message": None}
    finally:
        gate.mhr_decode.MHR_RUNTIME_AVAILABLE = originals["runtime_available"]
        gate.mhr_decode.MHR_RUNTIME_IMPORT_ERROR = originals["runtime_import_error"]
        gate.mhr_decode.MHRDecoder = originals["decoder"]
        gate.mhr_decode.gate_1a_euler_round_trip = originals["gate_1a"]
        gate.mhr_decode.ground_decoded_camera_frame = originals["ground"]
        gate.CourtCalibration.model_validate = originals["model_validate"]
        gate.build_decoder_provenance = originals["provenance"]

    result = {
        "absent_index": {
            "measurement_status": absent_report["measurement_status"],
            "raw_emit_source": absent_report["raw_emit_source"],
            "captured_ground_call": captured_ground_calls[0],
            "report_path": str((fixture_dir / "absent_report.json").relative_to(REPO_ROOT)),
        },
        "unreadable_index": unreadable,
    }
    write_json(OUT_DIR / "fallback_proof.json", result)
    return result


def threshold_stat_proof() -> dict[str, Any]:
    old_gate = git_show("HEAD~2:scripts/racketsport/gate_check_body_decode.py")
    new_gate = git_show("HEAD~1:scripts/racketsport/gate_check_body_decode.py")
    old_mhr = git_show("HEAD~2:threed/racketsport/mhr_decode.py")
    new_mhr = git_show("HEAD~1:threed/racketsport/mhr_decode.py")

    labels = [
        ("gate_check_body_decode.py", "per_player_gate1b", assignment_segment),
        ("gate_check_body_decode.py", "per_player_divergence", assignment_segment),
        ("gate_check_body_decode.py", "gate1b_summary", assignment_segment),
        ("gate_check_body_decode.py", "divergence_summary", assignment_segment),
        ("gate_check_body_decode.py", "_blocked_gate_1b", function_return_segment),
        ("gate_check_body_decode.py", "_blocked_mesh_divergence", function_return_segment),
    ]
    comparisons: dict[str, Any] = {}
    for file_name, label, extractor in labels:
        old_segment = extractor(old_gate, label)
        new_segment = extractor(new_gate, label)
        comparisons[f"{file_name}:{label}"] = segment_comparison(old_segment, new_segment)

    for label in ("GATE_1B_MAX_ABS_ERROR_MM", "MESH_SKELETON_DIVERGENCE_P95_MM"):
        old_segment = assignment_segment(old_mhr, label)
        new_segment = assignment_segment(new_mhr, label)
        comparisons[f"mhr_decode.py:{label}"] = segment_comparison(old_segment, new_segment)

    result = {
        "base": "HEAD~2",
        "fixed": "HEAD~1",
        "byte_equivalent": all(item["byte_equal"] for item in comparisons.values()),
        "comparisons": comparisons,
    }
    write_json(OUT_DIR / "threshold_stat_proof.json", result)
    return result


def escape_hatch_proof() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.racketsport import gate_check_body_decode as gate

    base = fixture_body_mesh()
    raw_default = gate.extract_frames(base, raw_records_by_request_id={"7:42": {"pred_cam_t": [0.25, -0.5, 1.0]}})[
        "42"
    ][0]

    body_true = fixture_body_mesh()
    body_frame = body_true["players"][0]["frames"][0]
    body_frame["camera_translation"] = [9.0, 9.0, 9.0]
    body_frame["pred_cam_t_already_applied"] = True
    precedence = gate.extract_frames(
        body_true,
        raw_records_by_request_id={"7:42": {"pred_cam_t": [0.25, -0.5, 1.0]}},
    )["42"][0]

    flag_only = gate.extract_frames(
        fixture_body_mesh(),
        raw_records_by_request_id={"7:42": {"pred_cam_t_already_applied": True}},
    )["42"][0]

    explicit_true = gate.extract_frames(
        fixture_body_mesh(),
        raw_records_by_request_id={"7:42": {"pred_cam_t": [0.25, -0.5, 1.0], "pred_cam_t_already_applied": "yes"}},
    )["42"][0]

    result = {
        "raw_default_false": frame_cam_summary(raw_default),
        "raw_source_precedence_over_body_true": frame_cam_summary(precedence),
        "flag_without_translation_ignored": frame_cam_summary(flag_only),
        "explicit_true_only": frame_cam_summary(explicit_true),
        "origin_trace": [
            "extract_frames computes request_id as '<frame_idx>:<player_id>'.",
            "_camera_translation_for_frame searches raw_record, then body_mesh_frame, then smplx_params.",
            "The already-applied flag is read only from the same source that provided pred_cam_t/camera_translation/transl.",
            "The default for pred_cam_t_already_applied is False; no inference path defaults it to True.",
        ],
    }
    write_json(OUT_DIR / "escape_hatch_proof.json", result)
    return result


def fixture_body_mesh() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from threed.racketsport import mhr_decode

    def numbers(count: int, start: float, step: float) -> list[float]:
        return [start + step * idx for idx in range(count)]

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "adversarial_fixture",
        "players": [
            {
                "id": 42,
                "frames": [
                    {
                        "frame_idx": 7,
                        "t": 7.0 / 30.0,
                        "smplx_params": {
                            "global_orient": [0.1, 0.2, 0.3],
                            "body_pose": numbers(mhr_decode.BODY_POSE_EULER_DIM, 0.01, 0.002),
                            "betas": numbers(mhr_decode.SHAPE_DIM, 0.2, 0.003),
                            "scale": numbers(mhr_decode.SCALE_DIM, 0.5, 0.01),
                            "left_hand_pose": numbers(mhr_decode.HAND_COMPS_DIM, 1.0, 0.01),
                            "right_hand_pose": numbers(mhr_decode.HAND_COMPS_DIM, -1.0, -0.01),
                            "transl_world": [1.25, -2.5, 0.0],
                        },
                        "joints_world": [[1.0, 2.0, 3.0]],
                        "mesh_vertices_world": [[4.0, 5.0, 6.0]],
                    }
                ],
            }
        ],
    }


def frame_cam_summary(frame: Any) -> dict[str, Any]:
    return {
        "pred_cam_t": frame.pred_cam_t,
        "pred_cam_t_already_applied": frame.pred_cam_t_already_applied,
    }


def mutant_env(mutant_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(mutant_root), str(REPO_ROOT), existing]) if existing else os.pathsep.join(
        [str(mutant_root), str(REPO_ROOT)]
    )
    env["MPLBACKEND"] = "Agg"
    return env


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(cmd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def git_show(spec: str) -> str:
    completed = subprocess.run(
        ["git", "show", spec],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def assignment_segment(source: str, target_label: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_label:
                    return ast.get_source_segment(source, node) or ""
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == target_label
                ):
                    return ast.get_source_segment(source, node) or ""
    raise KeyError(target_label)


def function_return_segment(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    return ast.get_source_segment(source, child) or ""
    raise KeyError(function_name)


def segment_comparison(old_segment: str, new_segment: str) -> dict[str, Any]:
    return {
        "byte_equal": old_segment == new_segment,
        "old_sha256": hashlib.sha256(old_segment.encode("utf-8")).hexdigest(),
        "new_sha256": hashlib.sha256(new_segment.encode("utf-8")).hexdigest(),
        "old_excerpt": old_segment.splitlines()[0] if old_segment else "",
        "new_excerpt": new_segment.splitlines()[0] if new_segment else "",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
