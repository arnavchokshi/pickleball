from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "runs/lanes/w1b_abc_loader_20260721/VM_ABC_RUN.md"


def _atomic_guard() -> str:
    text = RUNBOOK.read_text()
    start = text.index("# BEGIN ONE_TOUCH_ATOMIC_GUARD")
    end = text.index("# END ONE_TOUCH_ATOMIC_GUARD")
    return text[start:end]


@pytest.mark.parametrize(
    ("rg_exit", "expected_exit"),
    ((0, 70), (1, 0), (2, 71)),
)
def test_one_touch_atomic_guard_handles_all_rg_exit_branches(
    tmp_path: Path, rg_exit: int, expected_exit: int
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rg = fake_bin / "rg"
    fake_rg.write_text(f"#!/bin/sh\nexit {rg_exit}\n")
    fake_rg.chmod(0o755)
    ledger = tmp_path / "heldout_eval_ledger.md"
    ledger.write_text("# synthetic held-out ledger\n")
    lock = tmp_path / "event_head_abc_protected50_20260721.one_touch.lock"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "ONE_TOUCH_TOKEN": "event_head_abc_protected50_20260721",
        "HELDOUT_LEDGER": str(ledger),
        "ONE_TOUCH_LOCK": str(lock),
    }

    completed = subprocess.run(
        ["bash", "-c", _atomic_guard()],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_exit
    assert lock.is_dir()
    if rg_exit == 0:
        assert "token is already present" in completed.stderr
    elif rg_exit == 1:
        assert completed.stderr == ""
    else:
        assert "ledger search errored" in completed.stderr
