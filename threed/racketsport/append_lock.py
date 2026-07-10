"""Portable cross-process locking + atomic/append writes for shared text files.

Why this exists
----------------
Several files in this repo are edited by multiple concurrent agent sessions
that cannot talk to each other directly, most notably
`runs/manager/heldout_eval_ledger.md`, `runs/manager/inflight_lanes.md`, and
the fleet ledger. Root narrative documents are not coordination logs. Plain `Path.write_text(...)`
or an editor's "insert this text before that line" edit is not safe under
concurrency: two writers can interleave partial writes (a reader sees a
truncated/corrupt file), or one writer's read-modify-write can silently
clobber a row another writer appended a moment earlier ("lost update").

This module gives every writer of a shared file the same, simple discipline:

1. Take an exclusive `fcntl.flock` on a dedicated lock file next to the
   target (works across processes/hosts sharing a filesystem; `fcntl` is
   available on both macOS, where this repo's local checkout runs, and the
   Linux A100 host -- there is no portable `flock(1)` *binary* on macOS, but
   `fcntl.flock` is a stdlib call and needs no external binary).
2. While holding the lock, either:
   - append new bytes to the end of the file with a single `os.write()` on
     an `O_APPEND`-opened fd (`append_text`), which is the right primitive
     when new content always goes at the very end of the file (e.g. a
     brand-new trailing section); or
   - read-modify-write the whole file (`locked_update_text`), which is the
     right primitive when new content must be inserted at a specific place
     (e.g. a new row inserted into an existing Markdown table before a
     `---` delimiter, which is how most of this repo's ledger appends
     actually work) -- the write-back itself still goes through
     `write_atomic` (temp file + fsync + `os.replace`) so a crash mid-write
     never leaves a truncated file on disk.

Convention for this repo
-------------------------
Any code (Python) that programmatically appends a row/section to a shared
coordination file such as `runs/manager/heldout_eval_ledger.md` MUST go
through `append_text` or `locked_update_text` from this module instead of a
bare `open(path, "a")` / `Path.write_text(...)`. Interactive edits made by an
agent's editor tool are outside this module's reach (there is no lock across
a human/agent editing session), but agents making a single, fast,
programmatic append (e.g. a CLI or a report-writer step) should use this
module so at least the machine-driven writes cannot corrupt each other.

Example
-------
    from pathlib import Path
    from threed.racketsport.append_lock import append_text, locked_update_text

    # Pure end-of-file append (safe even if another writer appends at the
    # same instant -- both appends land, in some order, never interleaved):
    append_text(Path("runs/manager/heldout_eval_ledger.md"), "\n" + new_section_markdown)

    # Insert a new table row before a known marker, holding the lock across
    # the whole read-modify-write so no other writer's append/insert is lost:
    def _insert_row(current_text: str) -> str:
        marker = "\n---\n\n## POLICY"
        return current_text.replace(marker, f"\n{new_row_markdown}\n{marker}", 1)

    locked_update_text(Path("runs/manager/heldout_eval_ledger.md"), _insert_row)
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator

DEFAULT_LOCK_SUFFIX = ".append.lock"
DEFAULT_LOCK_TIMEOUT_S = 30.0
DEFAULT_POLL_INTERVAL_S = 0.05


class AppendLockTimeoutError(RuntimeError):
    """Raised when a lock cannot be acquired within the configured timeout."""


class AppendWriteError(RuntimeError):
    """Raised when a write did not complete as expected (e.g. a short write)."""


def default_lock_path(target: Path | str) -> Path:
    """The default sidecar lock file path for ``target``: ``.<name>.append.lock``."""

    target = Path(target)
    return target.parent / f".{target.name}{DEFAULT_LOCK_SUFFIX}"


@contextlib.contextmanager
def file_lock(
    lock_path: Path | str,
    *,
    timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> Iterator[None]:
    """Hold an exclusive cross-process lock on ``lock_path`` for the block body.

    Uses ``fcntl.flock`` (POSIX; available on macOS and Linux without any
    external binary). Blocks with polling up to ``timeout_s`` seconds, then
    raises :class:`AppendLockTimeoutError`. The lock file itself is never
    deleted (deleting a lock file while another process still holds it open
    can create a second, unsynchronized lock file at the same path and
    silently defeat mutual exclusion), only its lock is released.
    """

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise AppendLockTimeoutError(
                        f"could not acquire lock {lock_path} within {timeout_s}s "
                        "(another process is holding it -- if this persists, check "
                        "for a stuck/crashed writer holding this lock file open)"
                    ) from None
                time.sleep(poll_interval_s)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _fsync_dir_best_effort(dir_path: Path) -> None:
    try:
        dir_fd = os.open(str(dir_path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def write_atomic(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically: temp file + fsync + ``os.replace``.

    Readers always see either the previous complete content or the new
    complete content, never a truncated/partial file, even if the writer
    process crashes or the machine loses power mid-write. This function does
    **not** itself take a lock -- callers writing from multiple
    threads/processes should hold :func:`file_lock` (or rely on a
    module-level ``threading.Lock`` for same-process callers) around the
    read-modify-write sequence that produces ``text``, so two writers cannot
    both compute ``text`` from a stale read and then have the second
    `os.replace` silently discard the first writer's update.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
    _fsync_dir_best_effort(path.parent)


def append_text(
    path: Path | str,
    text: str,
    *,
    lock_path: Path | str | None = None,
    timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
) -> int:
    """Append ``text`` to the end of ``path`` under an exclusive lock.

    Uses a single ``os.write()`` on an ``O_APPEND``-opened file descriptor,
    which POSIX guarantees is atomic with respect to other ``O_APPEND``
    writers on local filesystems (the seek-to-end and the write happen as
    one kernel operation, so two concurrent appenders' bytes cannot
    interleave). The extra ``fcntl.flock`` (see :func:`file_lock`) is
    defense in depth and also serializes any read-modify-write a caller does
    just before appending. Returns the byte offset the text was written at.

    Only appropriate when new content always belongs at the very end of the
    file. For inserting content at a specific location (e.g. a Markdown
    table row before a delimiter), use :func:`locked_update_text` instead.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path) if lock_path is not None else default_lock_path(path)
    data = text.encode("utf-8")
    with file_lock(lock_path, timeout_s=timeout_s):
        fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        try:
            offset = os.lseek(fd, 0, os.SEEK_END)
            written = os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    if written != len(data):
        raise AppendWriteError(f"partial append write to {path}: wrote {written} of {len(data)} bytes")
    return offset


def locked_update_text(
    path: Path | str,
    transform: Callable[[str], str],
    *,
    lock_path: Path | str | None = None,
    timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    encoding: str = "utf-8",
) -> str:
    """Read ``path`` under lock, apply ``transform``, write back atomically.

    The lock is held across the whole read -> transform -> write sequence,
    so ``transform`` always sees the latest on-disk content (no other writer
    can sneak an update in between the read and the write) and the write
    itself is corruption-safe (:func:`write_atomic`). Returns the new text.

    This is the right primitive for the common "insert a new row into an
    existing Markdown table/section" append pattern used by this repo's
    ledgers, where the new content is not simply tacked onto EOF.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path) if lock_path is not None else default_lock_path(path)
    with file_lock(lock_path, timeout_s=timeout_s):
        current = path.read_text(encoding=encoding) if path.is_file() else ""
        updated = transform(current)
        write_atomic(path, updated, encoding=encoding)
    return updated
