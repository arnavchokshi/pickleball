#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


SOURCE_PREFIXES = ("scripts/", "threed/racketsport/")
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv_yolo_coreml",
    "node_modules",
    "runs",
    "third_party",
}


def build_dead_code_reference_audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    _validate_root(root)
    worktree_paths = _git_worktree_paths(root)
    source_paths = _python_source_paths(worktree_paths)
    text_by_path = _read_text_corpus(root, worktree_paths)
    import_refs = _python_import_references(root, text_by_path)

    entries = [
        _source_entry(relpath, text_by_path=text_by_path, import_refs=import_refs)
        for relpath in source_paths
    ]
    unknown = [entry for entry in entries if entry["status"] == "unknown"]
    kind_counts: dict[str, int] = {}
    for entry in entries:
        kind_counts[entry["kind"]] = kind_counts.get(entry["kind"], 0) + 1

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_dead_code_reference_audit",
        "root": root.as_posix(),
        "scope": {
            "source_prefixes": list(SOURCE_PREFIXES),
            "ignored_parts": sorted(IGNORED_PARTS),
            "uses_git_exclude_standard": True,
            "runs_source_code": False,
            "claims_semantic_reachability": False,
        },
        "summary": {
            "python_sources": len(entries),
            "unknown_python_sources": len(unknown),
            "kind_counts": dict(sorted(kind_counts.items())),
        },
        "unknown_python_sources": [entry["path"] for entry in unknown],
        "python_sources": entries,
        "status": "fail" if unknown else "pass",
    }


def _validate_root(root: Path) -> None:
    if not root.exists():
        raise ValueError(f"root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")
    if not (root / "scripts" / "racketsport").is_dir():
        raise ValueError(f"scripts/racketsport does not exist under: {root}")
    if not (root / "threed" / "racketsport").is_dir():
        raise ValueError(f"threed/racketsport does not exist under: {root}")


def _git_worktree_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in completed.stdout.splitlines() if line]
    return [path for path in sorted(paths) if not IGNORED_PARTS.intersection(Path(path).parts)]


def _python_source_paths(paths: Iterable[str]) -> list[str]:
    source_paths: list[str] = []
    for relpath in paths:
        path = Path(relpath)
        if path.suffix != ".py" or path.name == "__init__.py":
            continue
        if relpath.startswith("tests/"):
            continue
        if relpath.startswith("scripts/") or relpath.startswith("threed/racketsport/"):
            source_paths.append(relpath)
    return sorted(source_paths)


def _read_text_corpus(root: Path, paths: Iterable[str]) -> dict[str, str]:
    text_by_path: dict[str, str] = {}
    for relpath in paths:
        path = root / relpath
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text_by_path[relpath] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return text_by_path


def _python_import_references(root: Path, text_by_path: dict[str, str]) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for relpath, text in text_by_path.items():
        if not relpath.endswith(".py"):
            continue
        try:
            tree = ast.parse(text, filename=relpath)
        except SyntaxError:
            continue
        source_package = _package_for_python_file(Path(relpath))
        for imported_module in _imports_from_ast(tree, source_package=source_package):
            refs.setdefault(imported_module, set()).add(relpath)
    return refs


def _package_for_python_file(path: Path) -> str | None:
    if path.parts[:2] == ("threed", "racketsport"):
        package_parts = list(path.with_suffix("").parts[:-1])
        return ".".join(package_parts)
    return None


def _imports_from_ast(tree: ast.AST, *, source_package: str | None) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _resolve_import_from_module(node, source_package=source_package)
        if module:
            imports.add(module)
        for alias in node.names:
            if alias.name == "*":
                continue
            if module:
                imports.add(f"{module}.{alias.name}")
    return imports


def _resolve_import_from_module(node: ast.ImportFrom, *, source_package: str | None) -> str | None:
    if node.level == 0:
        return node.module
    if source_package is None:
        return None
    package_parts = source_package.split(".")
    keep_count = len(package_parts) - node.level + 1
    if keep_count <= 0:
        return node.module
    base = ".".join(package_parts[:keep_count])
    if node.module:
        return f"{base}.{node.module}"
    return base


def _source_entry(
    relpath: str,
    *,
    text_by_path: dict[str, str],
    import_refs: dict[str, set[str]],
) -> dict[str, object]:
    path = Path(relpath)
    kind = "module" if relpath.startswith("threed/racketsport/") else "cli"
    module_name = _module_name(path) if kind == "module" else None
    path_reference_files = _path_reference_files(relpath, text_by_path)
    module_text_reference_files = _path_reference_files(module_name, text_by_path) if module_name else []
    import_reference_files = _import_reference_files(module_name, import_refs) if module_name else []
    matching_test_files = _matching_test_files(path.stem, text_by_path)
    has_reference = bool(path_reference_files or module_text_reference_files or import_reference_files or matching_test_files)

    return {
        "path": relpath,
        "kind": kind,
        "module": module_name,
        "path_reference_files": path_reference_files,
        "module_text_reference_files": module_text_reference_files,
        "import_reference_files": import_reference_files,
        "matching_test_files": matching_test_files,
        "status": "referenced" if has_reference else "unknown",
    }


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def _path_reference_files(relpath: str, text_by_path: dict[str, str]) -> list[str]:
    return sorted(path for path, text in text_by_path.items() if path != relpath and relpath in text)


def _import_reference_files(module_name: str | None, import_refs: dict[str, set[str]]) -> list[str]:
    if module_name is None:
        return []
    files: set[str] = set()
    for imported_module, source_files in import_refs.items():
        if imported_module == module_name or imported_module.startswith(f"{module_name}."):
            files.update(source for source in source_files if source != module_name.replace(".", "/") + ".py")
    return sorted(files)


def _matching_test_files(stem: str, text_by_path: dict[str, str]) -> list[str]:
    normalized_stem = re.escape(stem)
    pattern = re.compile(rf"(^|_)({normalized_stem})(_|$)")
    matches: list[str] = []
    for relpath in text_by_path:
        path = Path(relpath)
        if path.parts[:2] != ("tests", "racketsport") or path.suffix != ".py":
            continue
        test_stem = path.stem.removeprefix("test_")
        if pattern.search(test_stem):
            matches.append(relpath)
    return sorted(matches)


def _format_human_report(report: dict[str, object]) -> str:
    lines = [
        f"status: {report['status']}",
        f"root: {report['root']}",
        f"python_sources: {report['summary']['python_sources']}",
        f"unknown_python_sources: {report['summary']['unknown_python_sources']}",
        "",
        "unknown:",
    ]
    unknown = report["unknown_python_sources"]
    if unknown:
        lines.extend(f"- {path}" for path in unknown)
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Python source surfaces for exact reference signals.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = build_dead_code_reference_audit(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human_report(report))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
