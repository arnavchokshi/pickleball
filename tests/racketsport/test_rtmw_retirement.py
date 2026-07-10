from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ACTIVE_SOURCE_ROOTS = (
    ROOT / "scripts" / "racketsport",
    ROOT / "threed" / "racketsport",
    ROOT / "ios",
)

RTMW_MARKERS = (
    "RTMW",
    "RTMW3D",
    "RTMPose",
    "MMPose",
    "rtmw",
    "rtmw3d",
    "rtmpose",
    "mmpose",
)


def test_active_pipeline_source_has_no_rtmw_runtime_or_references() -> None:
    assert not (ROOT / "threed" / "racketsport" / "pose_fast.py").exists()

    offenders: list[str] = []
    for root in ACTIVE_SOURCE_ROOTS:
        for path in root.rglob("*"):
            if path.is_dir() or _ignored_path(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(marker in text for marker in RTMW_MARKERS):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_canonical_docs_state_rtmw_is_retired_for_fast_sam3d_body() -> None:
    for relpath in ("NORTH_STAR_ROADMAP.md", "RUNBOOK.md"):
        text = (ROOT / relpath).read_text(encoding="utf-8")
        compact = " ".join(text.split())
        assert "RTMW" in text, relpath
        assert "retired" in compact.lower(), relpath
        assert "SAM-3D-Body only" in text, relpath


def _ignored_path(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    ignored_parts = {
        ".build",
        "__pycache__",
    }
    if set(rel_parts) & ignored_parts:
        return True
    return path.suffix not in {".py", ".swift", ".md", ".json"}
