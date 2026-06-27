from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _candidate(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "variant_id": "sam3dbody_fast_baseline",
        "stage": "EVAL-0",
        "clip_id": "clip_001",
        "overlay_path": "runs/eval0/clip_001/overlay.mp4",
        "accuracy_metric": 0.82,
        "latency_ms": 31.5,
        "vram_gb": 7.25,
        "notes": ["seed scaffold only"],
    }
    payload.update(overrides)
    return payload


def test_variant_comparison_writes_pending_markdown_and_json_summary(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.json"
    out_dir = tmp_path / "comparison"
    candidates_path.write_text(
        json.dumps(
            [
                _candidate(),
                _candidate(
                    variant_id="sam3dbody_accurate_candidate",
                    clip_id="clip_002",
                    overlay_path="runs/eval0/clip_002/overlay.png",
                    accuracy_metric=0.86,
                    latency_ms=58.0,
                    vram_gb=10.75,
                    notes="higher accuracy, slower",
                ),
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_variant_comparison.py",
            "--candidates",
            str(candidates_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    markdown = (out_dir / "variant_selection.md").read_text(encoding="utf-8")
    payload = json.loads((out_dir / "variant_selection.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["approval_status"] == "pending"
    assert payload["candidate_count"] == 2
    assert payload["candidates"][0]["variant_id"] == "sam3dbody_fast_baseline"
    assert payload["candidates"][1]["overlay_path"] == "runs/eval0/clip_002/overlay.png"
    assert "# Variant Selection Comparison" in markdown
    assert "approval_status: pending" in markdown
    assert "This report does not approve or lock any model variant." in markdown
    assert "sam3dbody_accurate_candidate" in markdown
    assert "models/MANIFEST.json" in markdown


def test_variant_comparison_rejects_unsafe_overlay_paths(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.json"
    for unsafe_path in ["../outside/overlay.mp4", "https://example.test/overlay.mp4"]:
        candidates_path.write_text(json.dumps([_candidate(overlay_path=unsafe_path)]), encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/racketsport/build_variant_comparison.py",
                "--candidates",
                str(candidates_path),
                "--out-dir",
                str(tmp_path / "comparison"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 2
        assert "unsafe relative path" in completed.stderr


def test_variant_comparison_auto_finalized_requires_reason(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([_candidate()]), encoding="utf-8")

    missing_reason = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_variant_comparison.py",
            "--candidates",
            str(candidates_path),
            "--out-dir",
            str(tmp_path / "missing_reason"),
            "--auto-finalized-obvious",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing_reason.returncode == 2
    assert "--finalized-reason is required" in missing_reason.stderr

    empty_reason = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_variant_comparison.py",
            "--candidates",
            str(candidates_path),
            "--out-dir",
            str(tmp_path / "empty_reason"),
            "--auto-finalized-obvious",
            "--finalized-reason",
            " ",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert empty_reason.returncode == 2
    assert "--finalized-reason must be non-empty" in empty_reason.stderr

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_variant_comparison.py",
            "--candidates",
            str(candidates_path),
            "--out-dir",
            str(tmp_path / "with_reason"),
            "--auto-finalized-obvious",
            "--finalized-reason",
            "Only one complete candidate in dry-run comparison.",
        ],
        check=True,
    )

    payload = json.loads((tmp_path / "with_reason" / "variant_selection.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "with_reason" / "variant_selection.md").read_text(encoding="utf-8")
    assert payload["approval_status"] == "auto_finalized_obvious"
    assert payload["finalized_reason"] == "Only one complete candidate in dry-run comparison."
    assert "approval_status: auto_finalized_obvious" in markdown
    assert "Only one complete candidate in dry-run comparison." in markdown
