#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.model_manifest import ModelEntry, load_model_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(entry: ModelEntry, *, check_files_only: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": entry.id,
        "status": entry.status,
        "local_path": entry.local_path,
        "file_ok": None,
        "checksum_ok": None,
        "integrity_ok": None,
        "forward_smoke_status": "not_run",
        "detail": "skipped; entry is not a declared H100 checkpoint file",
    }

    if entry.status != "available_on_h100":
        return result

    if not entry.local_path:
        result.update(file_ok=False, checksum_ok=False, integrity_ok=False, detail="missing local_path")
        return result

    path = Path(entry.local_path)
    if not path.is_file():
        result.update(file_ok=False, checksum_ok=False, integrity_ok=False, detail=f"missing file: {path}")
        return result

    result["file_ok"] = True
    actual_sha = sha256_file(path)
    result["actual_sha256"] = actual_sha
    result["checksum_ok"] = actual_sha == entry.sha256

    if not result["checksum_ok"]:
        result.update(integrity_ok=False, detail="sha256 mismatch")
        return result

    result["integrity_ok"] = True
    if check_files_only:
        result["detail"] = "file and checksum verified; forward smoke not run"
        return result

    # Phase 0 model-specific forward passes will be wired here as each upstream
    # runner is stabilized. For now, exact file+hash verification is the shared
    # prerequisite that fails fast before expensive GPU-specific smoke tests.
    result["detail"] = "file and checksum verified; model-specific forward pass not wired"
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    available = [result for result in results if result["status"] == "available_on_h100"]
    failures = [result for result in available if result["integrity_ok"] is not True]
    return {
        "schema_version": 1,
        "total_models": len(results),
        "declared_h100_checkpoint_files": len(available),
        "integrity_failed": len(failures),
        "models": results,
    }


def render_text(summary: dict[str, Any]) -> str:
    rows = [
        f"total_models={summary['total_models']}",
        f"declared_h100_checkpoint_files={summary['declared_h100_checkpoint_files']}",
        f"integrity_failed={summary['integrity_failed']}",
    ]
    for result in summary["models"]:
        rows.append(
            f"{result['id']}: status={result['status']} "
            f"file_ok={result['file_ok']} checksum_ok={result['checksum_ok']} "
            f"integrity_ok={result['integrity_ok']} forward_smoke_status={result['forward_smoke_status']} detail={result['detail']}"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify declared model file presence and sha256 integrity before GPU smoke tests.")
    parser.add_argument("--manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument(
        "--check-files-only",
        action="store_true",
        help="Only verify file presence and sha256 checksums for available_on_h100 entries.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    manifest = load_model_manifest(args.manifest)
    results = [check_model(entry, check_files_only=args.check_files_only) for entry in manifest.models]
    summary = summarize(results)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_text(summary))

    return 1 if summary["integrity_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
