from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


LANE = Path(__file__).resolve().parents[1]
ROOT = LANE.parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sampler = _load("owner_person_sample_frames", LANE / "tools/sample_frames.py")
ingest = _load("owner_person_ingest_labels", LANE / "tools/ingest_labels.py")


def _candidate(index: int, stratum: str, session: str) -> dict:
    return {
        "candidate_id": f"{index:020x}",
        "clip_id": f"IMG_{index:04d}",
        "clip_name": f"IMG_{index:04d}.mov",
        "video_path": f"runs/owner_footage_intake_20260702/raw/IMG_{index:04d}.mov",
        "video_sha256": f"sha-{index}",
        "session_id": session,
        "timestamp_s": float(index),
        "source_fps": 30.0,
        "stratum": stratum,
        "proposals": [
            {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 500.0, "confidence": 0.9}
            for _ in range(5 if stratum == "spectator_rich" else 1 if stratum == "empty_sparse" else 4)
        ],
    }


def _small_manifest() -> dict:
    candidates = []
    index = 1
    for stratum, count in {"gameplay": 12, "spectator_rich": 6, "empty_sparse": 6}.items():
        for _ in range(count):
            candidates.append(_candidate(index, stratum, f"session_{(index % 6) + 1:02d}"))
            index += 1
    return sampler.build_pack_manifest(
        candidates,
        {"eligible_gameplay_clips": 24, "sessions": {}},
        seed=sampler.SEED,
        targets={"gameplay": 8, "spectator_rich": 4, "empty_sparse": 4},
    )


def test_sampler_manifest_is_byte_deterministic() -> None:
    manifest = _small_manifest()
    left = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    right = json.dumps(_small_manifest(), indent=2, sort_keys=True) + "\n"
    assert left.encode() == right.encode()
    validation_sessions = [
        session for session, split in manifest["split"]["session_to_split"].items() if split == "validation"
    ]
    assert 1 <= len(validation_sessions) <= 2
    assert manifest["split"]["actual_validation_frames"] == sum(
        1 for frame in manifest["frames"] if frame["session_id"] in validation_sessions
    )


def test_scratch_withholding_and_page_stratum_blindness() -> None:
    manifest = _small_manifest()
    html = sampler.render_review_html(manifest)
    sampler.validate_blind_page(html, manifest)
    items = {item["id"]: item for item in sampler._page_items(manifest)}
    assert any(frame["scratch"] for frame in manifest["frames"])
    for frame in manifest["frames"]:
        if frame["scratch"]:
            assert items[frame["frame_id"]]["boxes"] == []
    lowered = html.lower()
    for forbidden in ("scratch", "spectator_rich", "empty_sparse", "clip_name", "video_path", "<video"):
        assert forbidden not in lowered
    for frame in manifest["frames"]:
        assert frame["clip_name"].lower() not in lowered
    assert 'KEY="person_labels_20260716"' in html
    assert "localStorage.setItem(KEY,JSON.stringify(state))" in html
    assert "const stored=JSON.parse(localStorage.getItem(KEY))" in html
    assert "beforeunload" in html
    assert "ArrowLeft" in html and "ArrowRight" in html
    assert "proposal_deleted" in html
    assert "mutate(rec=>{rec.empty_confirmed=true;rec.reviewed=true},false)" in html
    assert "if(clearEmpty)rec.empty_confirmed=false" in html
    assert "rec.empty_confirmed=false;rec.reviewed=true;rec.boxes.push" in html


def test_render_page_direct_cli(tmp_path: Path) -> None:
    manifest = _small_manifest()
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    for frame in manifest["frames"]:
        Image.new("RGB", (1920, 1080), (20, 40, 30)).save(candidate_dir / frame["filename"], quality=95)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stage = tmp_path / "stage"
    completed = subprocess.run(
        [
            sys.executable, str(LANE / "tools/sample_frames.py"), "render-page",
            "--manifest", str(manifest_path), "--candidate-dir", str(candidate_dir), "--staging-dir", str(stage),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["frames"] == len(manifest["frames"])
    assert report["page_blind"] is True
    assert (stage / "START_HERE.html").is_file()


def test_generated_page_state_machine_executes_and_exports_stored_state(tmp_path: Path) -> None:
    html = sampler.render_review_html(_small_manifest())
    match = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert match
    browser_script = match.group(1)
    harness = r'''
const elements = new Map();
function element(id="") {
  return {
    id, style:{}, dataset:{}, textContent:"", innerHTML:"", src:"", tagName:"DIV",
    classList:{add(){},remove(){},toggle(){},contains(){return false}},
    addEventListener(){}, setPointerCapture(){},
    getBoundingClientRect(){return {left:0,top:0,width:1920,height:1080}},
    click(){},
  };
}
globalThis.document = {
  getElementById(id){if(!elements.has(id)) elements.set(id,element(id)); return elements.get(id)},
  addEventListener(){},
  createElement(tag){const e=element(); e.tagName=tag.toUpperCase(); return e},
};
const storage = new Map();
globalThis.localStorage = {
  getItem(k){return storage.has(k)?storage.get(k):null},
  setItem(k,v){storage.set(k,String(v))},
};
globalThis.window = {addEventListener(){}};
globalThis.Image = class {set src(value){this._src=value} get src(){return this._src}};
globalThis.URL = {createObjectURL(){return "blob:test"},revokeObjectURL(){}};
globalThis.Blob = class {constructor(parts,opts){this.parts=parts;this.opts=opts;globalThis.lastBlob=this}};
'''
    assertions = r'''
const noProposalIndex=ITEMS.findIndex(item=>item.boxes.length===0);
if(noProposalIndex<0) throw new Error("fixture has no proposal-withheld item");
state.index=noProposalIndex; entered=performance.now(); render(); confirmEmpty();
const emptyId=current().id;
if(!state.frames[emptyId].empty_confirmed||!state.frames[emptyId].reviewed) throw new Error("empty confirmation did not persist");
const emptySnapshot=JSON.stringify({boxes:state.frames[emptyId].boxes,empty_confirmed:state.frames[emptyId].empty_confirmed,reviewed:state.frames[emptyId].reviewed});
go(1); go(-1);
if(JSON.stringify({boxes:state.frames[emptyId].boxes,empty_confirmed:state.frames[emptyId].empty_confirmed,reviewed:state.frames[emptyId].reviewed})!==emptySnapshot) throw new Error("navigation changed earlier answer");
const proposalIndex=ITEMS.findIndex(item=>item.boxes.length>0);
state.index=proposalIndex; entered=performance.now(); render(); selected=0; toggleClass();
let proposal=ensure().boxes[0];
if(proposal.class!=="off_court_person"||proposal.source!=="proposal_adjusted") throw new Error("class adjustment provenance failed");
removeSelected(); proposal=ensure().boxes[0];
if(!proposal.deleted||proposal.source!=="proposal_deleted") throw new Error("delete provenance failed");
exportNow();
const exported=JSON.parse(globalThis.lastBlob.parts.join(""));
const stored=JSON.parse(localStorage.getItem(KEY));
const exportedEmpty=exported.frames.find(frame=>frame.frame_id===emptyId);
if(!exportedEmpty||!exportedEmpty.empty_confirmed) throw new Error("export missed stored empty confirmation");
if(exportedEmpty.ms_spent!==Math.round(stored.frames[emptyId].ms_spent||0)) throw new Error("export timing differs from stored state");
const proposalId=current().id;
const exportedProposal=exported.frames.find(frame=>frame.frame_id===proposalId);
if(!exportedProposal||exportedProposal.boxes[0].source!=="proposal_deleted"||!exportedProposal.boxes[0].deleted) throw new Error("export differs from stored deletion state");
const roundTrip=JSON.parse(JSON.stringify(stored));
if(JSON.stringify(roundTrip)!==JSON.stringify(stored)) throw new Error("stored state does not round trip exactly");
process.stdout.write("page_state_machine PASS\n");
'''
    script_path = tmp_path / "page_state_harness.js"
    script_path.write_text(harness + "\n" + browser_script + "\n" + assertions, encoding="utf-8")
    completed = subprocess.run(["node", str(script_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "page_state_machine PASS\n"


def _ingest_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    frames = [
        {
            "frame_id": "opaque_a", "filename": "opaque_a.jpg", "clip_id": "IMG_1001", "clip_name": "IMG_1001.mov",
            "video_path": "runs/owner_footage_intake_20260702/raw/IMG_1001.mov", "video_sha256": "sha-a",
            "session_id": "session_01", "timestamp_s": 1.0, "source_fps": 30.0, "stratum": "gameplay",
            "detected_stratum": "gameplay", "backfilled_from": None, "scratch": False,
            "proposals": [{"x1": 10, "y1": 10, "x2": 110, "y2": 210, "confidence": 0.9}], "presentation_index": 1,
        },
        {
            "frame_id": "opaque_b", "filename": "opaque_b.jpg", "clip_id": "IMG_2001", "clip_name": "IMG_2001.mov",
            "video_path": "runs/owner_footage_intake_20260702/raw/IMG_2001.mov", "video_sha256": "sha-b",
            "session_id": "session_02", "timestamp_s": 2.0, "source_fps": 30.0, "stratum": "spectator_rich",
            "detected_stratum": "spectator_rich", "backfilled_from": None, "scratch": True,
            "proposals": [{"x1": 20, "y1": 20, "x2": 120, "y2": 220, "confidence": 0.8}], "presentation_index": 2,
        },
        {
            "frame_id": "opaque_c", "filename": "opaque_c.jpg", "clip_id": "IMG_3001", "clip_name": "IMG_3001.mov",
            "video_path": "runs/owner_footage_intake_20260702/raw/IMG_3001.mov", "video_sha256": "sha-c",
            "session_id": "session_03", "timestamp_s": 3.0, "source_fps": 30.0, "stratum": "empty_sparse",
            "detected_stratum": "empty_sparse", "backfilled_from": None, "scratch": True,
            "proposals": [], "presentation_index": 3,
        },
    ]
    manifest = {
        "schema_version": 1, "session_id": sampler.SESSION_ID, "created_at": sampler.CREATED_AT,
        "image": {"width": 1920, "height": 1080}, "frames": frames,
        "split": {"session_to_split": {"session_01": "train", "session_02": "train", "session_03": "validation"}},
    }
    export = {
        "schema_version": 1, "session_id": sampler.SESSION_ID, "storage_key": sampler.PAGE_STORAGE_KEY,
        "frames": [
            {
                "frame_id": "opaque_a", "boxes": [
                    {"x1": 10, "y1": 10, "x2": 110, "y2": 210, "class": "player", "source": "proposal_confirmed"},
                    {"x1": 400, "y1": 50, "x2": 500, "y2": 250, "class": "off_court_person", "source": "drawn"},
                    {"x1": 700, "y1": 10, "x2": 800, "y2": 210, "class": "player", "source": "proposal_deleted", "deleted": True},
                ], "empty_confirmed": False, "ms_spent": 12000,
            },
            {
                "frame_id": "opaque_b", "boxes": [
                    {"x1": 20, "y1": 20, "x2": 120, "y2": 220, "class": "player", "source": "drawn"},
                ], "empty_confirmed": False, "ms_spent": 9000,
            },
            {"frame_id": "opaque_c", "boxes": [], "empty_confirmed": True, "ms_spent": 3000},
        ],
    }
    manifest_path = tmp_path / "pack_manifest.json"
    export_path = tmp_path / "person_labels_export.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    export_path.write_text(json.dumps(export), encoding="utf-8")
    return manifest_path, export_path, export


def test_schema_round_trip_audit_math_and_empty_confirm(tmp_path: Path) -> None:
    manifest_path, export_path, _ = _ingest_fixture(tmp_path)
    payload = ingest.build_outputs(export_path, manifest_path, allow_partial=False)
    audit = payload["audit"]
    assert audit["frames_ingested"] == 3
    assert audit["active_label_count"] == 3
    assert audit["labels_by_class"] == {"off_court_person": 1, "player": 2}
    assert audit["proposal_review_sources"] == {"drawn": 2, "proposal_confirmed": 1, "proposal_deleted": 1}
    assert audit["empty_confirmed_count"] == 1
    assert audit["review_ms_total"] == 24000
    scratch = audit["scratch_vs_withheld_proposals"]
    assert scratch["frames"] == 2
    assert scratch["micro_iou_match_precision"] == 1.0
    assert scratch["micro_iou_match_recall"] == 1.0
    assert audit["eval_protected_disjointness"]["assertion"] is True
    assert payload["split"]["session_disjoint"] is True
    assert len(payload["empties"]) == 1
    assert all(label["provenance"]["reviewer"] == "owner" for label in payload["labels"])


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    manifest_path, export_path, _ = _ingest_fixture(tmp_path)
    payload = ingest.build_outputs(export_path, manifest_path, allow_partial=False)
    out = tmp_path / "dataset"
    ingest.write_outputs(payload, out)
    first = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    ingest.write_outputs(payload, out)
    second = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    assert first == second


def test_ingest_refuses_partial_without_override(tmp_path: Path) -> None:
    manifest_path, export_path, export = _ingest_fixture(tmp_path)
    export["frames"].pop()
    export_path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="partial export refused"):
        ingest.build_outputs(export_path, manifest_path, allow_partial=False)
    payload = ingest.build_outputs(export_path, manifest_path, allow_partial=True)
    assert payload["audit"]["partial"] is True


def test_empty_confirm_rejects_active_boxes(tmp_path: Path) -> None:
    manifest_path, export_path, export = _ingest_fixture(tmp_path)
    export["frames"][0]["empty_confirmed"] = True
    export_path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="empty-confirmed frame carries active boxes"):
        ingest.build_outputs(export_path, manifest_path, allow_partial=False)


def test_ingest_direct_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    manifest_path, export_path, _ = _ingest_fixture(tmp_path)
    out = tmp_path / "dataset"
    completed = subprocess.run(
        [
            sys.executable, str(LANE / "tools/ingest_labels.py"), "--export", str(export_path),
            "--manifest", str(manifest_path), "--out-dir", str(out), "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["dry_run"] is True
    assert report["audit"]["frames_ingested"] == 3
    assert not out.exists()


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda box: box.update({"class": "spectator"}), "invalid class"),
        (lambda box: box.update({"x2": 2000}), "outside image"),
        (lambda box: box.update({"source": "proposal_deleted"}), "deleted flag/source mismatch"),
    ],
)
def test_ingest_rejects_bad_box_schema(tmp_path: Path, mutation, message: str) -> None:
    manifest_path, export_path, export = _ingest_fixture(tmp_path)
    mutation(export["frames"][0]["boxes"][0])
    export_path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        ingest.build_outputs(export_path, manifest_path, allow_partial=False)
