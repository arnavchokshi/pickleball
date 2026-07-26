"""NS-04 trust-contract regression: the compositor must never be sold as fusion.

`scripts/racketsport/process_video.py` runs a `world` stage that stitches
already-finished artifacts together (`threed/racketsport/virtual_world.py`
"assembles" them). It performs no joint refinement. Reporting its output under a
capability named "fusion" told every reader of `PIPELINE_SUMMARY.json` that
confidence-weighted joint refinement had happened when nothing in the default
stack ever fused anything.

These tests pin the honest vocabulary so the mislabel cannot come back.
"""

from __future__ import annotations

from pathlib import Path

from scripts.racketsport import process_video
from server import bundle_policy


ROOT = Path(__file__).resolve().parents[2]

# Words that promise joint refinement / state estimation over multiple evidence
# sources. A pure compositor may not be credited with any of them.
JOINT_REFINEMENT_WORDS = ("fusion", "fused", "fuse", "refined", "refinement", "solved", "estimated")

# The stage that only composites. Its capability names are the ones under test.
COMPOSITOR_STAGE = "world"


def _missing_capability_names(clip_dir: Path, preset: str) -> dict[str, str]:
    """The requirement tables are function-local tuples, so exercise the real
    function rather than reaching into private state: an empty clip dir reports
    every capability as missing, which enumerates the whole vocabulary."""

    missing = process_video._minimum_bundle_missing_capabilities(
        clip_dir=clip_dir,
        stage_outcomes=[],
        video_identity=None,
        pipeline_preset=preset,
    )
    return {item["capability"]: item["reason"] for item in missing}


def test_compositor_output_is_not_reported_under_a_joint_refinement_name(tmp_path: Path) -> None:
    """The core regression. An empty bundle reports every capability as missing,
    so this enumerates the complete worker capability vocabulary for both presets."""

    for preset in ("full", "court_skeletons"):
        reported = _missing_capability_names(tmp_path, preset)
        assert "fusion" not in reported, (
            f"preset {preset}: capability 'fusion' is back. The world stage composites "
            "finished artifacts and never jointly refines them."
        )
        assert "composited_world" in reported, preset

        for capability, reason in reported.items():
            for word in JOINT_REFINEMENT_WORDS:
                assert word not in capability, (
                    f"preset {preset}: capability {capability!r} implies joint refinement "
                    f"via {word!r}; only a real joint-refinement stage may claim that"
                )
            if capability == "composited_world":
                assert "fus" not in reason.lower(), (
                    f"preset {preset}: composited_world reason still advertises fusion: {reason!r}"
                )


def test_composited_world_is_satisfied_by_the_compositor_artifact_alone(tmp_path: Path) -> None:
    """Positive control: the capability still tracks the same artifacts, so this is
    a rename of the claim, not a change to what the bundle requires."""

    for artifact in ("virtual_world.json", "confidence_gated_world.json"):
        clip_dir = tmp_path / artifact.replace(".json", "")
        clip_dir.mkdir()
        (clip_dir / artifact).write_text("{}", encoding="utf-8")

        reported = _missing_capability_names(clip_dir, "full")
        assert "composited_world" not in reported, (
            f"{artifact} should satisfy composited_world"
        )


def test_worker_and_server_capability_vocabularies_stay_lockstep(tmp_path: Path) -> None:
    """`server.bundle_policy.gate_reported_status` compares the worker's
    missing_capabilities list against the list the server re-derives. A one-sided
    rename silently downgrades every complete bundle to `partial`, so the two
    vocabularies must not drift."""

    server_names = {capability for capability, _candidates, _reason in bundle_policy._mandatory_requirements()}
    worker_names = set(_missing_capability_names(tmp_path, "full"))

    assert "fusion" not in server_names
    assert "composited_world" in server_names

    # Every server-side capability that the worker also evaluates must use the
    # same spelling. (The server adds bundle-only names like `summary`.)
    server_only = server_names - worker_names
    assert server_only == {"summary"}, (
        f"server capability vocabulary drifted from the worker's: {sorted(server_only)}"
    )


def test_server_capability_reasons_do_not_advertise_fusion() -> None:
    for capability, _candidates, reason in bundle_policy._mandatory_requirements():
        for word in JOINT_REFINEMENT_WORDS:
            assert word not in capability, f"{capability!r} implies joint refinement via {word!r}"
        if capability == "composited_world":
            assert "fus" not in reason.lower(), reason


def test_no_stage_in_the_graph_claims_fusion_by_name() -> None:
    """The compositor stage keeps its honest name. Any stage that does claim joint
    refinement is covered by tests/racketsport/test_one_world_stage.py."""

    assert COMPOSITOR_STAGE in {node.name for node in process_video.AUTHORITATIVE_STAGE_GRAPH}
    for node in process_video.AUTHORITATIVE_STAGE_GRAPH:
        assert "fusion" not in node.name, node.name


def test_capability_rename_is_recorded_for_bundle_consumers() -> None:
    """The capability name is a published contract surface (it appears in
    `PIPELINE_SUMMARY.json` -> `missing_capabilities[].capability`). Keep the
    documented break discoverable next to the code that made it."""

    text = (ROOT / "scripts" / "racketsport" / "process_video.py").read_text(encoding="utf-8")
    assert "must not be named \"fusion\"" in text

    server_text = (ROOT / "server" / "bundle_policy.py").read_text(encoding="utf-8")
    assert "lockstep-identical" in server_text


def test_no_stray_fusion_capability_literals_remain_in_bundle_code() -> None:
    for relpath in ("scripts/racketsport/process_video.py", "server/bundle_policy.py"):
        text = (ROOT / relpath).read_text(encoding="utf-8")
        assert '"fusion",' not in text, f"{relpath} still declares a 'fusion' capability tuple"
