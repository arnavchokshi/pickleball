from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable


LARGE_TRACKED_THRESHOLD_BYTES = 5 * 1024 * 1024
LARGE_UNTRACKED_SOURCE_THRESHOLD_BYTES = 1 * 1024 * 1024
IGNORED_DIR_PARTS = {
    ".git",
    ".venv",
    ".venv_yolo_coreml",
    "node_modules",
    "runs",
    "third_party",
}
GENERATED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".build",
}
GENERATED_FILE_NAMES = {".DS_Store"}
GENERATED_FILE_SUFFIXES = {".pyc", ".pyo"}
GENERATED_RELATIVE_DIRS = {
    "ios/.build",
    "web/replay/dist",
}

ALLOWED_LARGE_TRACKED_FILES = {
    "cvat_upload/04_indoor_doubles_fwuks_0500_long_mid_baseline_30s.mp4",
    "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4",
    "models_coreml/yolo26m_img416_int8/yolo26m.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "models_coreml/yolo26s_img416_int8/yolo26s.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
}

ALLOWED_LARGE_UNTRACKED_SOURCE_FILES = {
    "ios/Replay/Sources/PickleballReplay/Resources/RealityReplayFixture/body_mesh_animated_budget53.usdz",
    "ios/Replay/Sources/PickleballReplay/Resources/WorldFixture/virtual_world.json",
    "tests/racketsport/fixtures/solid_mesh_real_window_000/body_mesh_faces.json",
}


def build_storage_report(root: Path, *, check_generated_artifacts: bool = True) -> dict[str, object]:
    root = root.resolve()
    tracked_large = _large_files(_git_paths(root, ["ls-files"]), root, LARGE_TRACKED_THRESHOLD_BYTES)
    untracked_large = _large_files(
        _git_paths(root, ["ls-files", "--others", "--exclude-standard"]),
        root,
        LARGE_UNTRACKED_SOURCE_THRESHOLD_BYTES,
    )

    unknown_tracked = tracked_large - ALLOWED_LARGE_TRACKED_FILES
    unknown_untracked = untracked_large - ALLOWED_LARGE_UNTRACKED_SOURCE_FILES
    generated_artifacts = _generated_artifacts(root) if check_generated_artifacts else set()

    return {
        "root": root.as_posix(),
        "thresholds": {
            "large_tracked_bytes": LARGE_TRACKED_THRESHOLD_BYTES,
            "large_untracked_source_bytes": LARGE_UNTRACKED_SOURCE_THRESHOLD_BYTES,
        },
        "allowed_large_tracked_files": sorted(ALLOWED_LARGE_TRACKED_FILES),
        "observed_large_tracked_files": sorted(tracked_large),
        "unknown_large_tracked_files": sorted(unknown_tracked),
        "missing_allowed_large_tracked_files": sorted(ALLOWED_LARGE_TRACKED_FILES - tracked_large),
        "allowed_large_untracked_source_files": sorted(ALLOWED_LARGE_UNTRACKED_SOURCE_FILES),
        "observed_large_untracked_source_files": sorted(untracked_large),
        "unknown_large_untracked_source_files": sorted(unknown_untracked),
        "missing_allowed_large_untracked_source_files": sorted(ALLOWED_LARGE_UNTRACKED_SOURCE_FILES - untracked_large),
        "check_generated_artifacts": check_generated_artifacts,
        "generated_artifacts": sorted(generated_artifacts),
        "status": "fail" if unknown_tracked or unknown_untracked or generated_artifacts else "pass",
    }


def _git_paths(root: Path, args: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _large_files(paths: Iterable[str], root: Path, threshold_bytes: int) -> set[str]:
    large: set[str] = set()
    for relpath in paths:
        path = root / relpath
        if path.is_file() and path.stat().st_size > threshold_bytes:
            large.add(relpath)
    return large


def _generated_artifacts(root: Path) -> set[str]:
    artifacts: set[str] = set()
    for relpath in GENERATED_RELATIVE_DIRS:
        path = root / relpath
        if path.exists():
            artifacts.add(relpath)
    for path in _walk_non_ignored(root):
        relpath = path.relative_to(root).as_posix()
        if relpath in GENERATED_RELATIVE_DIRS:
            continue
        if path.is_dir() and path.name in GENERATED_DIR_NAMES:
            artifacts.add(relpath)
        elif path.is_file() and (path.name in GENERATED_FILE_NAMES or path.suffix in GENERATED_FILE_SUFFIXES):
            artifacts.add(relpath)
    return artifacts


def _walk_non_ignored(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda child: child.as_posix(), reverse=True)
        except OSError:
            continue
        for child in children:
            rel_parts = child.relative_to(root).parts
            if any(part in IGNORED_DIR_PARTS for part in rel_parts):
                continue
            yield child
            if child.is_dir() and child.name not in GENERATED_DIR_NAMES:
                stack.append(child)


def _format_human_report(report: dict[str, object]) -> str:
    lines = [
        f"status: {report['status']}",
        f"root: {report['root']}",
        "",
        "unknown_large_tracked_files:",
    ]
    lines.extend(f"- {path}" for path in report["unknown_large_tracked_files"])
    lines.append("")
    lines.append("unknown_large_untracked_source_files:")
    lines.extend(f"- {path}" for path in report["unknown_large_untracked_source_files"])
    lines.append("")
    lines.append("generated_artifacts:")
    lines.extend(f"- {path}" for path in report["generated_artifacts"])
    if lines[-1].endswith(":"):
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--ignore-generated-artifacts",
        action="store_true",
        help="Skip generated cache/build leftover checks; intended for pytest processes that create caches while running.",
    )
    args = parser.parse_args()

    report = build_storage_report(args.root, check_generated_artifacts=not args.ignore_generated_artifacts)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human_report(report))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
