#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import smoke_models

GIB = 1024**3
DEFAULT_MIN_FREE_GB = 15.0


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _status_counts(checks: list[Check]) -> dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0, "info": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def check_python_repo_venv(repo_root: Path) -> Check:
    executable = Path(sys.executable).absolute()
    active_prefix = Path(sys.prefix).resolve()
    expected = (repo_root / ".venv").resolve()
    if _is_relative_to(executable, expected) or active_prefix == expected:
        return Check(
            "python_repo_venv",
            "pass",
            "running under the repo .venv Python",
            {"executable": str(executable), "active_prefix": str(active_prefix), "expected_prefix": str(expected)},
        )
    return Check(
        "python_repo_venv",
        "fail",
        "not running under the repo .venv; use .venv/bin/python to avoid Anaconda/MPS drift",
        {"executable": str(executable), "active_prefix": str(active_prefix), "expected_prefix": str(expected)},
    )


def check_torch_mps() -> Check:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return Check("torch_mps", "warn", f"torch import failed: {type(exc).__name__}: {exc}")

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    is_built = bool(mps_backend is not None and mps_backend.is_built())
    is_available = bool(mps_backend is not None and mps_backend.is_available())
    details = {
        "torch_version": getattr(torch, "__version__", None),
        "mps_built": is_built,
        "mps_available": is_available,
    }
    if is_available:
        return Check("torch_mps", "pass", "torch MPS backend is available", details)
    return Check(
        "torch_mps",
        "warn",
        "torch MPS backend is not available in this process; local MPS smoke/training may fall back or fail",
        details,
    )


def check_mplbackend() -> Check:
    value = os.environ.get("MPLBACKEND", "")
    if value.lower() == "agg":
        return Check("mplbackend", "pass", "MPLBACKEND=Agg is set", {"value": value})
    return Check(
        "mplbackend",
        "warn",
        "set MPLBACKEND=Agg before pytest to avoid GUI backend failures",
        {"value": value or None, "recommended": "MPLBACKEND=Agg .venv/bin/python -m pytest ..."},
    )


def check_disk_headroom(path: Path, *, min_free_gb: float) -> Check:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / GIB
    details = {
        "path": str(path),
        "free_gb": round(free_gb, 3),
        "total_gb": round(usage.total / GIB, 3),
        "min_free_gb": min_free_gb,
    }
    if free_gb < min_free_gb:
        return Check(
            "disk_headroom",
            "warn",
            f"only {free_gb:.1f} GB free; pipeline/body artifacts can exhaust local disk",
            details,
        )
    return Check("disk_headroom", "pass", f"{free_gb:.1f} GB free on the checked filesystem", details)


def check_model_weights(manifest: Path) -> Check:
    try:
        model_manifest = smoke_models.load_model_manifest(manifest)
        results = [smoke_models.check_model(entry, check_files_only=True) for entry in model_manifest.models]
        summary = smoke_models.summarize(results)
    except Exception as exc:  # noqa: BLE001
        return Check(
            "model_weights",
            "fail",
            f"could not read model manifest with smoke_models.py logic: {type(exc).__name__}: {exc}",
            {"manifest": str(manifest)},
        )

    details = {
        "manifest": str(manifest),
        "declared_h100_checkpoint_files": summary["declared_h100_checkpoint_files"],
        "integrity_failed": summary["integrity_failed"],
        "failed_model_ids": [
            result["id"]
            for result in summary["models"]
            if result.get("status") == "available_on_h100" and result.get("integrity_ok") is not True
        ],
    }
    if summary["integrity_failed"]:
        return Check(
            "model_weights",
            "warn",
            "declared H100 checkpoint files are missing or mismatched here; rerun on the GPU host or fix the manifest/runtime paths",
            details,
        )
    return Check("model_weights", "pass", "declared H100 checkpoint files passed smoke_models.py file/hash checks", details)


def check_web_replay_node_modules(web_replay_dir: Path) -> Check:
    node_modules = web_replay_dir / "node_modules"
    if node_modules.is_dir():
        return Check(
            "web_replay_node_modules",
            "pass",
            "web/replay node_modules is present",
            {"path": str(node_modules)},
        )
    return Check(
        "web_replay_node_modules",
        "warn",
        "web/replay dependencies are not installed; run npm install before npm run dev",
        {"path": str(node_modules), "command": "cd web/replay && npm install"},
    )


def check_known_hosts_hint(path: Path) -> Check:
    if path.is_file():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        return Check(
            "known_hosts_hint",
            "info",
            "remote BODY host keys are pinned; refresh recycled fleet IPs before trusting SSH failures",
            {
                "known_hosts_file": str(path),
                "mtime_utc": mtime,
                "refresh_command": "scripts/fleet/refresh_remote_host.sh --host <ip> --alias <name>",
                "remote_probe": "pass --remote-host to run SSH/disk/nvidia/version checks",
            },
        )
    return Check(
        "known_hosts_hint",
        "info",
        "no pinned known_hosts file found at the default path",
        {
            "known_hosts_file": str(path),
            "refresh_command": "scripts/fleet/refresh_remote_host.sh --host <ip> --alias <name>",
        },
    )


def generated_artifacts_audit_hint() -> Check:
    return Check(
        "generated_artifacts_audit_hint",
        "info",
        "when following the runbook after local tests, use audit_storage_policy.py --ignore-generated-artifacts to avoid failing on fresh pytest/build outputs",
        {
            "command": "python3 scripts/racketsport/audit_storage_policy.py --root . --ignore-generated-artifacts --json"
        },
    )


def remote_not_requested() -> Check:
    return Check(
        "remote_probe",
        "info",
        "remote checks were skipped; pass --remote-host to run SSH, df -h, nvidia-smi, and version-stamp checks",
    )


def _run_remote(config: Any, remote_command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*config.ssh_base(), remote_command],
        check=False,
        capture_output=True,
        text=True,
        timeout=config.connect_timeout_s + 30,
    )


def remote_checks(args: argparse.Namespace, *, repo_root: Path) -> list[Check]:
    try:
        import remote_body_dispatch as remote
    except Exception as exc:  # noqa: BLE001
        return [
            Check(
                "remote_body_helpers",
                "warn",
                f"remote_body_dispatch helpers were not importable: {type(exc).__name__}: {exc}",
            )
        ]

    config = remote.RemoteConfig(
        host=args.remote_host,
        ssh_key=args.remote_ssh_key,
        repo=args.remote_repo,
        python=args.remote_python,
        fast_sam_python=args.remote_fast_sam_python,
        fast_sam_root=args.remote_fast_sam_root,
        known_hosts_file=args.known_hosts_file,
    )
    checks: list[Check] = []
    reachable = remote.check_remote_reachable(config)
    checks.append(
        Check(
            "remote_ssh_reachable",
            "pass" if reachable else "fail",
            "remote host is reachable over SSH" if reachable else "remote host is not reachable over SSH",
            {"host": args.remote_host, "known_hosts_file": args.known_hosts_file},
        )
    )
    if not reachable:
        return checks

    df_result = _run_remote(config, f"df -h {remote.shlex.quote(args.remote_repo)}")
    checks.append(
        Check(
            "remote_disk",
            "pass" if df_result.returncode == 0 else "fail",
            "remote df -h completed" if df_result.returncode == 0 else "remote df -h failed",
            {"stdout": df_result.stdout.strip(), "stderr": df_result.stderr.strip()},
        )
    )

    nvidia_result = _run_remote(config, "nvidia-smi")
    checks.append(
        Check(
            "remote_nvidia_smi",
            "pass" if nvidia_result.returncode == 0 else "fail",
            "remote nvidia-smi completed" if nvidia_result.returncode == 0 else "remote nvidia-smi failed",
            {"stdout_tail": nvidia_result.stdout.strip()[-2000:], "stderr_tail": nvidia_result.stderr.strip()[-2000:]},
        )
    )

    try:
        stamp = remote.build_version_stamp(
            repo_root=repo_root,
            remote_run_dir="/tmp/racketsport_doctor_preflight",
            generated_runner_sha256="doctor-preflight-no-runner",
            allow_dirty=args.allow_dirty,
        )
        remote_head = remote._remote_git_head_sha(config, run=remote._run)
        local_head = stamp["git_head_sha"]
        matched = bool(local_head) and local_head == remote_head
        checks.append(
            Check(
                "remote_version_stamp",
                "pass" if matched else "fail",
                "remote git HEAD matches the local BODY version stamp"
                if matched
                else "remote git HEAD differs from the local BODY version stamp; run --sync-remote-code or verify with --verify-version-stamp before trusting VM numbers",
                {
                    "local_git_head_sha": local_head,
                    "remote_git_head_sha": remote_head,
                    "stamp_helper": "scripts/racketsport/remote_body_dispatch.py --verify-version-stamp",
                    "dirty_tracked_runtime_files": stamp.get("dirty_tracked_runtime_files", []),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check(
                "remote_version_stamp",
                "fail",
                f"could not build/compare remote BODY version stamp: {type(exc).__name__}: {exc}",
                {"stamp_helper": "scripts/racketsport/remote_body_dispatch.py --verify-version-stamp"},
            )
        )

    return checks


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    checks = [
        check_python_repo_venv(repo_root),
        check_torch_mps(),
        check_mplbackend(),
        check_disk_headroom(args.disk_path.resolve(), min_free_gb=args.min_free_gb),
        check_model_weights(args.manifest.resolve()),
        check_web_replay_node_modules(args.web_replay_dir.resolve()),
        check_known_hosts_hint(Path(args.known_hosts_file).expanduser().resolve())
        if args.known_hosts_file
        else check_known_hosts_hint(Path("")),
        generated_artifacts_audit_hint(),
    ]
    if args.remote_host:
        checks.extend(remote_checks(args, repo_root=repo_root))
    else:
        checks.append(remote_not_requested())

    counts = _status_counts(checks)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_doctor_report",
        "status": "fail" if counts["fail"] else "pass",
        "repo_root": str(repo_root),
        "summary": counts,
        "checks": {check.check_id: {"status": check.status, "message": check.message, "details": check.details} for check in checks},
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"doctor status: {report['status']}",
        "summary: "
        + " ".join(f"{key}={value}" for key, value in report["summary"].items()),
    ]
    for check_id, check in report["checks"].items():
        lines.append(f"- {check_id}: {check['status']} - {check['message']}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only preflight for running and debugging the pickleball pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Incident-backed checks: use .venv/bin/python instead of Anaconda when MPS matters; "
            "set MPLBACKEND=Agg before pytest; keep more than 15GB free for generated artifacts; "
            "reuse smoke_models.py for model file/hash checks; install web/replay node_modules before "
            "opening http://127.0.0.1:5173/?manifest=/@fs/.../replay_viewer_manifest.json; "
            "run audit_storage_policy.py --ignore-generated-artifacts after local tests; inspect "
            "remote_body_stdout.log for remote BODY failures; rerun BODY with --fetch-body-monoliths "
            "when monoliths are needed; verify VM code drift with remote_body_dispatch.py "
            "--verify-version-stamp before trusting VM numbers; refresh recycled fleet IPs with "
            "scripts/fleet/refresh_remote_host.sh."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="Repository root for .venv and git/runtime checks.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "models" / "MANIFEST.json")
    parser.add_argument("--web-replay-dir", type=Path, default=ROOT / "web" / "replay")
    parser.add_argument("--disk-path", type=Path, default=ROOT, help="Filesystem path to check for local free space.")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--known-hosts-file", default=str(ROOT / "configs" / "ssh" / "a100_known_hosts"))
    parser.add_argument("--remote-host", default="", help="SSH host to probe. No network calls are made unless this is set.")
    parser.add_argument("--remote-ssh-key", default="~/.ssh/google_compute_engine")
    parser.add_argument("--remote-repo", default="/home/arnavchokshi/pickleball_git")
    parser.add_argument("--remote-python", default="/opt/conda/envs/fast_sam_3d_body/bin/python")
    parser.add_argument("--remote-fast-sam-python", default="/opt/conda/envs/fast_sam_3d_body/bin/python")
    parser.add_argument("--remote-fast-sam-root", default="/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty tracked runtime files in the local version-stamp comparison metadata.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
