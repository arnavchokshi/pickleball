from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from threed.racketsport import append_lock


def test_write_atomic_leaves_no_temp_file_and_full_content(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    append_lock.write_atomic(target, "hello world\n")

    assert target.read_text(encoding="utf-8") == "hello world\n"
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [target]


def test_write_atomic_overwrite_never_exposes_partial_content(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    append_lock.write_atomic(target, "A" * 10_000)

    append_lock.write_atomic(target, "B" * 5_000)

    content = target.read_text(encoding="utf-8")
    # A reader can only ever see the fully-old or fully-new content -- never
    # a mix (e.g. "AAAABBBB...") -- because os.replace() is atomic.
    assert content == "B" * 5_000


def test_write_atomic_cleans_up_temp_file_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "out.txt"

    class _Boom(Exception):
        pass

    real_fdopen = os.fdopen

    def _boom_fdopen(fd, mode="r", **kwargs):  # noqa: ANN001
        handle = real_fdopen(fd, mode, **kwargs)
        real_write = handle.write

        def _write(data):
            raise _Boom("disk full")

        handle.write = _write  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(append_lock.os, "fdopen", _boom_fdopen)

    with pytest.raises(_Boom):
        append_lock.write_atomic(target, "will not land")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_file_lock_serializes_two_threads(tmp_path: Path) -> None:
    import threading

    lock_path = tmp_path / ".test.lock"
    events: list[str] = []

    def worker(tag: str) -> None:
        with append_lock.file_lock(lock_path, timeout_s=5.0):
            events.append(f"{tag}-start")
            time.sleep(0.05)
            events.append(f"{tag}-end")

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    time.sleep(0.01)
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # Whichever thread goes first, its start/end pair must not be interleaved
    # with the other thread's start/end pair.
    assert events[0].endswith("start")
    assert events[1].endswith("end")
    assert events[0].split("-")[0] == events[1].split("-")[0]
    assert events[2].endswith("start")
    assert events[3].endswith("end")


def test_file_lock_times_out_when_already_held(tmp_path: Path) -> None:
    lock_path = tmp_path / ".test.lock"
    with append_lock.file_lock(lock_path, timeout_s=5.0):
        with pytest.raises(append_lock.AppendLockTimeoutError):
            with append_lock.file_lock(lock_path, timeout_s=0.15, poll_interval_s=0.02):
                pass  # pragma: no cover - should never enter


def _append_worker(path_str: str, lock_str: str, tag: str, count: int) -> None:
    path = Path(path_str)
    for i in range(count):
        append_lock.append_text(path, f"{tag}-{i}\n", lock_path=Path(lock_str))


def test_append_text_concurrent_processes_never_interleave_lines(tmp_path: Path) -> None:
    target = tmp_path / "ledger.md"
    lock_path = tmp_path / ".ledger.lock"
    target.write_text("# ledger\n", encoding="utf-8")

    procs = [
        multiprocessing.Process(target=_append_worker, args=(str(target), str(lock_path), tag, 40))
        for tag in ("p1", "p2", "p3")
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# ledger"
    body = lines[1:]
    assert len(body) == 120
    # Every line must be a complete, well-formed "tagN-i" line -- if two
    # writers' bytes had interleaved mid-write, we would see garbled or
    # merged lines here instead of exactly 120 clean ones.
    for line in body:
        tag, _, idx = line.partition("-")
        assert tag in {"p1", "p2", "p3"}
        assert idx.isdigit()
    for tag in ("p1", "p2", "p3"):
        indices = [int(line.split("-", 1)[1]) for line in body if line.startswith(f"{tag}-")]
        assert indices == list(range(40))


def test_append_text_returns_offset_at_end_of_prior_content(tmp_path: Path) -> None:
    target = tmp_path / "ledger.md"
    target.write_text("12345", encoding="utf-8")

    offset = append_lock.append_text(target, "67890")

    assert offset == 5
    assert target.read_text(encoding="utf-8") == "1234567890"


def test_locked_update_text_inserts_before_marker_under_lock(tmp_path: Path) -> None:
    target = tmp_path / "ledger.md"
    target.write_text("# ledger\n\nrow-1\n\n---\n\n## POLICY\n", encoding="utf-8")

    def _insert(current: str) -> str:
        return current.replace("\n---\n", "\nrow-2\n\n---\n", 1)

    updated = append_lock.locked_update_text(target, _insert)

    assert updated == target.read_text(encoding="utf-8")
    assert "row-1" in updated and "row-2" in updated
    assert updated.index("row-1") < updated.index("row-2") < updated.index("## POLICY")


def test_locked_update_text_creates_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "new" / "ledger.md"

    append_lock.locked_update_text(target, lambda current: current + "first line\n")

    assert target.read_text(encoding="utf-8") == "first line\n"


def test_default_lock_path_is_hidden_sidecar(tmp_path: Path) -> None:
    target = tmp_path / "heldout_eval_ledger.md"

    lock_path = append_lock.default_lock_path(target)

    assert lock_path.name == ".heldout_eval_ledger.md.append.lock"
    assert lock_path.parent == tmp_path
