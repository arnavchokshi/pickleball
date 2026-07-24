from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.racketsport import build_person_fewshot_pack as pack


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_RELATIVE = "scripts/racketsport/build_person_fewshot_pack.py"
SCRIPT = ROOT / SCRIPT_RELATIVE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(count: int, *, frame_count: int = 1_000, media_sha256: str = "a" * 64) -> list[dict]:
    return [
        {
            "source_id": f"source-{index:02d}",
            "source_pool": "fixture",
            "source_family_id": f"fixture:source-{index:02d}",
            "venue_source_family_id": f"fixture:venue-{index:02d}",
            "duration_s": frame_count / 10,
            "fps_inventory": 10.0,
            "expected_frame_count": frame_count,
            "expected_media_sha256": media_sha256,
            "expected_media_location": f"fixture/source-{index:02d}.avi",
        }
        for index in range(count)
    ]


def _local_media(sources: list[dict]) -> list[dict]:
    return [
        {
            "source_id": row["source_id"],
            "venue_id": row["venue_source_family_id"],
            "source_pool": row["source_pool"],
            "expected_media_sha256": row["expected_media_sha256"],
            "local_media_present": False,
            "matching_paths": [],
            "sha_mismatches": [],
        }
        for row in sources
    ]


def _teacher(checkpoint: Path) -> dict:
    model_manifest = checkpoint.with_suffix(".manifest.json")
    model_manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "yolo26m",
                        "sha256": _sha256(checkpoint),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "model_id": "yolo26m",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_manifest_path": str(model_manifest),
        "model_manifest_sha256": _sha256(model_manifest),
        "class_filter": {"class_id": 0, "class_name": "person"},
        "confidence_min": 0.25,
        "nms": "ultralytics_defaults_no_override",
    }


@pytest.mark.parametrize("clip_id", pack.PROTECTED_CLIP_IDS)
def test_quarantine_refuses_every_protected_clip_id(clip_id: str) -> None:
    with pytest.raises(pack.QuarantinedSourceError, match="PROTECTED_EVAL_CLIP_ID"):
        pack.assert_source_allowed({"source_id": clip_id})


@pytest.mark.parametrize("media_sha256", sorted(pack.PROTECTED_MEDIA_SHA256))
def test_quarantine_refuses_every_protected_media_sha(media_sha256: str) -> None:
    with pytest.raises(pack.QuarantinedSourceError, match="PROTECTED_EVAL_MEDIA_SHA256"):
        pack.assert_source_allowed({"source_id": "innocent", "media_sha256": media_sha256})


@pytest.mark.parametrize("source_id", sorted(pack.COMPARE_ONLY_PBVISION_IDS))
def test_quarantine_refuses_every_compare_only_id(source_id: str) -> None:
    with pytest.raises(pack.QuarantinedSourceError, match="PBVISION_COMPARE_ONLY"):
        pack.assert_source_allowed({"source_id": source_id})


@pytest.mark.parametrize("media_sha256", sorted(pack.COMPARE_ONLY_MEDIA_SHA256))
def test_quarantine_refuses_every_compare_only_media_sha(media_sha256: str) -> None:
    with pytest.raises(pack.QuarantinedSourceError, match="PBVISION_COMPARE_ONLY_MEDIA_SHA256"):
        pack.assert_source_allowed({"source_id": "permitted-looking", "video_sha256": media_sha256})


@pytest.mark.parametrize(
    "identity",
    [
        "IYnbdRs1Jdk",
        "%49%59%6e%62%64%52%73%31%4a%64%6b",
        {"parent": {"lineage": "prefix_IYnbdRs1Jdk_derivative"}},
    ],
)
def test_quarantine_refuses_iynbd_derivatives_after_canonical_decode(identity: object) -> None:
    with pytest.raises(pack.QuarantinedSourceError, match="IYNBDRS1JDK_DERIVATIVE"):
        pack.assert_source_allowed({"source_id": "other", "lineage": identity})


def _nested_percent_encoding(value: str, depth: int) -> str:
    encoded = "".join(f"%{ord(character):02X}" for character in value)
    for _ in range(depth - 1):
        encoded = encoded.replace("%", "%25")
    return encoded


@pytest.mark.parametrize("depth", range(1, pack.MAX_CANONICAL_DECODE_PASSES + 1))
def test_quarantine_depth_one_through_eight_preserves_canonical_refusal(depth: int) -> None:
    encoded = _nested_percent_encoding(pack.IYNBD_DERIVATIVE_TOKEN, depth)
    with pytest.raises(pack.QuarantinedSourceError, match="IYNBDRS1JDK_DERIVATIVE"):
        pack.assert_source_allowed({"source_id": "other", "lineage": encoded})


def test_quarantine_depth_nine_fails_closed_when_decode_budget_is_exhausted() -> None:
    encoded = _nested_percent_encoding(
        pack.IYNBD_DERIVATIVE_TOKEN,
        pack.MAX_CANONICAL_DECODE_PASSES + 1,
    )
    with pytest.raises(
        pack.QuarantinedSourceError,
        match="CANONICAL_DECODE_BUDGET_EXHAUSTED",
    ):
        pack.assert_source_allowed({"source_id": "other", "lineage": encoded})


def test_quarantine_regenerated_owner_artifacts_contain_no_denied_source_identity() -> None:
    lane = ROOT / "runs/lanes/trkC_fewshot_pack_20260722"
    regenerated_text = "\n".join(
        [
            (lane / "review/START_HERE.html").read_text(encoding="utf-8"),
            (lane / "review/index.html").read_text(encoding="utf-8"),
            (lane / "OWNER_ASK.md").read_text(encoding="utf-8"),
        ]
    )
    normalized = pack.mixed_pack._normalise_guard_text(regenerated_text)
    denied_ids = (
        set(pack.PROTECTED_CLIP_IDS)
        | set(pack.COMPARE_ONLY_PBVISION_IDS)
        | {pack.IYNBD_DERIVATIVE_TOKEN}
    )
    assert all(pack.mixed_pack._normalise_guard_text(value) not in normalized for value in denied_ids)
    assert all(value not in regenerated_text.lower() for value in pack.PROTECTED_MEDIA_SHA256)
    assert all(value not in regenerated_text.lower() for value in pack.COMPARE_ONLY_MEDIA_SHA256)


def test_uniform_sampler_is_deterministic_and_trims_intro_outro() -> None:
    first = pack.sample_frame_indices(10_000)
    second = pack.sample_frame_indices(10_000)
    assert first == second
    assert len(first) == 32
    assert len(set(first)) == 32
    assert first[0] == 300
    assert first[-1] == 9_699
    strides = [right - left for left, right in zip(first, first[1:], strict=False)]
    assert max(strides) - min(strides) <= 1


def test_secondary_selection_is_seeded_deterministic_and_venue_unique() -> None:
    sources = _sources(8)
    # A second clip from an existing venue cannot create an extra venue stratum.
    duplicate = dict(sources[0])
    duplicate["source_id"] = "source-duplicate"
    duplicate["source_family_id"] = "fixture:source-duplicate"
    sources.append(duplicate)
    first = pack.select_secondary_harvest_sources(sources, seed=20260722, limit=5)
    second = pack.select_secondary_harvest_sources(list(reversed(sources)), seed=20260722, limit=5)
    assert first == second
    assert len(first) == 5
    assert len({row["venue_source_family_id"] for row in first}) == 5


def test_partition_is_hash_deterministic_disjoint_and_roughly_70_30() -> None:
    sources = [pack._with_venue_identity(row) for row in _sources(15)]
    first = pack.build_partition_assignment(sources, seed=20260722)
    second = pack.build_partition_assignment(list(reversed(sources)), seed=20260722)
    assert first == second
    assert first["counts"] == {"FINETUNE_MATERIAL": 10, "PROMOTION_JUDGE": 5}
    fine = {row["venue_id"] for row in first["assignments"] if row["partition"] == "FINETUNE_MATERIAL"}
    judge = {row["venue_id"] for row in first["assignments"] if row["partition"] == "PROMOTION_JUDGE"}
    assert len(judge) >= 3
    assert fine.isdisjoint(judge)
    assert fine | judge == {row["venue_id"] for row in sources}


def test_partition_refuses_known_physical_venue_alias_across_sides() -> None:
    sources = [pack._with_venue_identity(row) for row in _sources(15)]
    baseline = pack.build_partition_assignment(sources, seed=20260722)
    fine_source = next(
        row["source_id"]
        for row in baseline["assignments"]
        if row["partition"] == pack.PARTITION_FINETUNE
    )
    judge_source = next(
        row["source_id"]
        for row in baseline["assignments"]
        if row["partition"] == pack.PARTITION_JUDGE
    )
    aliases = {
        fine_source: "physical-venue:fixture-collision",
        judge_source: "physical-venue:fixture-collision",
    }
    with pytest.raises(pack.VenueAliasCollisionError, match="crosses partition sides"):
        pack.build_partition_assignment(
            sources,
            seed=20260722,
            venue_aliases=aliases,
        )


def test_real_assignment_passes_person_p1_and_person_mixed_alias_taxonomy() -> None:
    lane = ROOT / "runs/lanes/trkC_fewshot_pack_20260722"
    manifest = pack._load_json(lane / "pack_manifest.json")
    assignment = pack._load_json(lane / "partition_assignment.json")
    aliases = pack.load_person_family_alias_taxonomy()
    result = pack.validate_partition_venue_aliases(
        manifest["sources"],
        assignment,
        aliases=aliases,
    )
    assert result == {
        "known_source_count": 15,
        "known_alias_count": 15,
        "cross_partition_alias_collisions": 0,
    }
    assert manifest["venue_alias_risk"] == {
        "cross_partition_alias_collisions": 0,
        "known_alias_count": 15,
        "known_source_count": 15,
        "person_mixed_manifest": "runs/lanes/person_mixed_20260722/pack_manifest.json",
        "person_p1_manifest": (
            "runs/lanes/person_p1_roboflow_20260721/roboflow_person/dataset_manifest.json"
        ),
        "statement": (
            "Partitioning is name-based. Aliases known by the person_p1/person_mixed family "
            "taxonomy are refused across sides, but physical venue identity is not fully "
            "provable from these manifests."
        ),
    }


def test_decode_plan_hits_registered_counts_cap_and_every_row_has_partition() -> None:
    sources = [pack._with_venue_identity(row) for row in _sources(15)]
    assignment = pack.build_partition_assignment(sources)
    rows = pack.build_decode_rows(sources, assignment)
    counts = Counter(row["venue_id"] for row in rows)
    assert len(rows) == 480
    assert set(counts.values()) == {32}
    assert all(row["partition"] in pack.PARTITIONS for row in rows)
    assert all(row["teacher_derived"] is True for row in rows)
    assert all(row["ground_truth"] is False for row in rows)
    assert all(row["verified_by_owner"] is False for row in rows)


def test_same_seed_writes_byte_identical_plan_and_partition(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    sources = _sources(15)
    first = tmp_path / "first"
    second = tmp_path / "second"
    pack.write_plan(
        out_dir=first,
        sources=sources,
        teacher=_teacher(checkpoint),
        local_media=_local_media(sources),
        seed=20260722,
    )
    pack.write_plan(
        out_dir=second,
        sources=list(reversed(sources)),
        teacher=_teacher(checkpoint),
        local_media=list(reversed(_local_media(sources))),
        seed=20260722,
    )
    assert (first / "decode_plan.jsonl").read_bytes() == (second / "decode_plan.jsonl").read_bytes()
    assert (first / "partition_assignment.json").read_bytes() == (
        second / "partition_assignment.json"
    ).read_bytes()
    manifest = json.loads((first / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["partition_assignment"]["sha256"] == _sha256(
        first / "partition_assignment.json"
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"teacher_derived": False}, "teacher_derived=true"),
        ({"ground_truth": True}, "ground_truth=false"),
        ({"verified_by_owner": True}, "verified_by_owner=false"),
    ],
)
def test_pseudo_rows_can_never_claim_verified_or_ground_truth(mutation: dict, match: str) -> None:
    expected_sha256 = pack._expected_teacher_checkpoint_sha256(ROOT / "models/MANIFEST.json")
    row = {
        "partition": "FINETUNE_MATERIAL",
        "teacher_derived": True,
        "ground_truth": False,
        "verified_by_owner": False,
        "teacher_conf": 0.5,
        "teacher_checkpoint_sha256": expected_sha256,
    }
    row.update(mutation)
    with pytest.raises(ValueError, match=match):
        pack.validate_prelabel_rows([row])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"teacher_conf": 0.01}, "below the registered 0.25 floor"),
        ({"teacher_checkpoint_sha256": "0" * 64}, "checkpoint SHA does not match"),
    ],
)
def test_prelabel_validator_rejects_low_confidence_and_manifest_sha_drift(
    mutation: dict,
    match: str,
) -> None:
    row = {
        "partition": "FINETUNE_MATERIAL",
        "teacher_derived": True,
        "ground_truth": False,
        "verified_by_owner": False,
        "teacher_conf": 0.5,
        "teacher_checkpoint_sha256": pack._expected_teacher_checkpoint_sha256(
            ROOT / "models/MANIFEST.json"
        ),
    }
    row.update(mutation)
    with pytest.raises(ValueError, match=match):
        pack.validate_prelabel_rows([row])


def test_prelabel_validator_enforces_teacher_confidence_upper_bound() -> None:
    row = {
        "partition": "FINETUNE_MATERIAL",
        "teacher_derived": True,
        "ground_truth": False,
        "verified_by_owner": False,
        "teacher_conf": 1.0,
        "teacher_checkpoint_sha256": pack._expected_teacher_checkpoint_sha256(
            ROOT / "models/MANIFEST.json"
        ),
    }
    pack.validate_prelabel_rows([row])
    for invalid in (1.01, 1.0000001):
        row["teacher_conf"] = invalid
        with pytest.raises(ValueError, match="exceeds the registered 1.0 ceiling"):
            pack.validate_prelabel_rows([row])


def _write_test_video(path: Path, *, frame_count: int = 120) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (96, 64),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :, 1] = index % 255
            writer.write(frame)
    finally:
        writer.release()


def test_materialize_writes_separate_prelabels_offline_html_and_cvat_jobs(tmp_path: Path) -> None:
    video = tmp_path / "venue.avi"
    _write_test_video(video)
    checkpoint = tmp_path / "yolo26m.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    sources = _sources(10, frame_count=120, media_sha256=_sha256(video))
    out = tmp_path / "pack"
    teacher = _teacher(checkpoint)
    pack.write_plan(
        out_dir=out,
        sources=sources,
        teacher=teacher,
        local_media=_local_media(sources),
    )

    def fake_predictor(
        image_paths: list[Path],
        checkpoint_path: Path,
        confidence: float,
        device: str,
    ) -> list[list[dict]]:
        assert checkpoint_path == checkpoint
        assert confidence == 0.25
        assert device == "cpu"
        return [[{"bbox_xyxy": [10.0, 8.0, 40.0, 55.0], "confidence": 0.91}] for _ in image_paths]

    result = pack.materialize_source(
        out_dir=out,
        source_id="source-00",
        media_path=video,
        checkpoint_path=checkpoint,
        predictor=fake_predictor,
        model_manifest_path=Path(teacher["model_manifest_path"]),
    )
    assert result["decoded_frames"] == 32
    assert result["teacher_boxes"] == 32
    assert result["verified_by_owner"] is False
    prelabels = pack._load_jsonl(out / "prelabels_teacher.jsonl")
    pack.validate_prelabel_rows(
        prelabels,
        model_manifest_path=pack._teacher_manifest_path(pack._load_json(out / "pack_manifest.json")),
    )
    assert len(prelabels) == 32
    assert (out / "verified_labels.jsonl").read_bytes() == b""
    assert len(list((out / "materialized").glob("*/frames/*.jpg"))) == 32

    review_path = out / "review" / "START_HERE.html"
    review = review_path.read_text(encoding="utf-8")
    assert "FINETUNE MATERIAL" in review
    assert "PROMOTION JUDGE" in review
    assert "Export results.json" in review
    assert "1 · ALL CORRECT" in review
    assert "2 · BOX WRONG" in review
    assert "3 · PERSON MISSED" in review
    assert "4 · UNSURE" in review
    assert "Saved locally" in review
    assert "Resume saved progress" in review
    assert "teacher_conf" not in review  # confidence is shown numerically, not misnamed as GT.
    assert review.count('"image":"../materialized/') == 32
    assert review.count('data-status="pending"') == 9
    assert not re.search(r"(?:https?:)?//", review)
    assert (out / "review" / "index.html").read_text(encoding="utf-8").count(
        "START_HERE.html"
    ) == 3

    items_match = re.search(
        r'<script id="review-items" type="application/json">(.*?)</script>',
        review,
    )
    assert items_match is not None
    items = json.loads(items_match.group(1))
    assert len(items) == 32
    assert all(len(item["boxes"]) == 1 for item in items)
    assert all(not item["image"].startswith(("/", "http://", "https://")) for item in items)
    before = review_path.read_bytes()
    emitted = pack.emit_owner_review(
        out,
        model_manifest_path=Path(teacher["model_manifest_path"]),
    )
    assert emitted == {
        "review_html": "review/START_HERE.html",
        "index_redirect": "review/index.html",
        "materialized_venues": 1,
        "materialized_frames": 32,
        "pending_venues": 9,
        "teacher_boxes": 32,
    }
    assert review_path.read_bytes() == before

    validation = pack.validate_cvat_package(out)
    assert validation["status"] == "PASS"
    assert validation["venue_disjoint"] is True
    assert (out / "cvat_upload" / "FINETUNE_MATERIAL").is_dir()
    assert (out / "cvat_upload" / "PROMOTION_JUDGE").is_dir()
    materialized = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))[
        "materialized_venues"
    ]
    assert materialized[0]["frame_count"] == 32
    assert len(materialized[0]["frame_md5_by_name"]) == 32
    assert materialized[0]["verified_by_owner"] is False
    assert materialized[0]["training_eligible"] is False


def _emit_runtime_review_fixture(tmp_path: Path) -> Path:
    sources = _sources(10)
    checkpoint = tmp_path / "runtime-teacher.pt"
    checkpoint.write_bytes(b"runtime-teacher")
    teacher = _teacher(checkpoint)
    out = tmp_path / "runtime-pack"
    pack.write_plan(
        out_dir=out,
        sources=sources,
        teacher=teacher,
        local_media=_local_media(sources),
    )
    planned = [
        row
        for row in pack._load_jsonl(out / "decode_plan.jsonl")
        if row["source_id"] == "source-00"
    ]
    frames: list[dict] = []
    prelabels: list[dict] = []
    for row in planned:
        frame = {
            **row,
            "image_path": row["output_relpath"],
            "image_width": 96,
            "image_height": 64,
        }
        image_path = out / frame["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"offline-runtime-fixture")
        frames.append(frame)
        prelabels.append(
            {
                **frame,
                "box_id": 1,
                "class_id": 0,
                "class_name": "person",
                "bbox_xyxy": [10.0, 8.0, 40.0, 55.0],
                "teacher_conf": 0.91,
                "teacher_model_id": pack.TEACHER_ID,
                "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
                "teacher_derived": True,
                "ground_truth": False,
                "verified_by_owner": False,
                "label_state": "PRELABEL_ONLY",
                "training_eligible": False,
                "production_eligible": False,
                "do_not_promote": True,
            }
        )
    venue_dir = out / "materialized" / pack._slug(str(planned[0]["venue_id"]))
    pack._write_jsonl(venue_dir / "frames.jsonl", frames)
    pack._write_jsonl(venue_dir / "prelabels_teacher.jsonl", prelabels)
    pack.emit_owner_review(
        out,
        model_manifest_path=Path(teacher["model_manifest_path"]),
    )
    return out / "review" / "START_HERE.html"


class TestEmittedReviewJavaScriptRuntime:
    def test_regenerated_start_here_node_check(self, tmp_path: Path) -> None:
        node = shutil.which("node")
        assert node is not None, "node is required for the emitted-page runtime regression"
        review = (
            ROOT / "runs/lanes/trkC_fewshot_pack_20260722/review/START_HERE.html"
        ).read_text(encoding="utf-8")
        runtime_matches = re.findall(r"<script>(.*?)</script>", review, flags=re.DOTALL)
        assert len(runtime_matches) == 1
        runtime_path = tmp_path / "regenerated-start-here.js"
        runtime_path.write_text(runtime_matches[0], encoding="utf-8")
        completed = subprocess.run(
            [node, "--check", str(runtime_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        print(
            json.dumps(
                {
                    "page": "runs/lanes/trkC_fewshot_pack_20260722/review/START_HERE.html",
                    "node_check_exit": completed.returncode,
                },
                sort_keys=True,
            )
        )

    def test_node_vm_compiles_and_executes_keymap_autoadvance_storage_and_export(
        self,
        tmp_path: Path,
    ) -> None:
        node = shutil.which("node")
        assert node is not None, "node is required for the emitted-page runtime regression"
        review_path = _emit_runtime_review_fixture(tmp_path)
        review = review_path.read_text(encoding="utf-8")
        runtime_matches = re.findall(r"<script>(.*?)</script>", review, flags=re.DOTALL)
        assert len(runtime_matches) == 1
        runtime_path = tmp_path / "emitted-review-runtime.js"
        runtime_path.write_text(runtime_matches[0], encoding="utf-8")

        check = subprocess.run(
            [node, "--check", str(runtime_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr

        contract_match = re.search(
            r'<script id="answer-contract" type="application/json">(.*?)</script>',
            review,
        )
        assert contract_match is not None
        items_match = re.search(
            r'<script id="review-items" type="application/json">(.*?)</script>',
            review,
        )
        assert items_match is not None
        harness_path = tmp_path / "runtime-harness.js"
        harness_path.write_text(
            r'''
const fs = require("node:fs");
const vm = require("node:vm");
const runtimeSource = fs.readFileSync(process.argv[2], "utf8");
const contractText = process.argv[3];
const itemsText = process.argv[4];
class FakeClassList {
  constructor(values = []) { this.values = new Set(values); }
  contains(value) { return this.values.has(value); }
  toggle(value, force) {
    const enabled = force === undefined ? !this.values.has(value) : Boolean(force);
    if (enabled) this.values.add(value); else this.values.delete(value);
    return enabled;
  }
}
const listeners = {};
const storage = new Map();
const state = { downloaded: null, blob: null, revoked: null };
const elements = {};
function element(id, classes = []) {
  if (!elements[id]) {
    elements[id] = {
      id,
      classList: new FakeClassList(classes),
      textContent: "",
      style: {},
      dataset: {},
      src: "",
      alt: "",
      innerHTML: "",
      onclick: null,
      click() { state.downloaded = this.download || id; },
    };
  }
  return elements[id];
}
element("startScreen");
element("reviewScreen", ["hidden"]);
element("doneScreen", ["hidden"]);
element("answer-contract").textContent = contractText;
element("review-items").textContent = itemsText;
const keyButtons = ["1", "2", "3", "4"].map(key => {
  const button = element("key-" + key);
  button.dataset.key = key;
  return button;
});
const context = {
  console,
  document: {
    getElementById: id => element(id),
    querySelectorAll: selector => selector === "[data-key]" ? keyButtons : [],
    addEventListener: (name, handler) => { listeners[name] = handler; },
    createElement: tag => element("created-" + tag),
  },
  localStorage: {
    getItem: key => storage.has(key) ? storage.get(key) : null,
    setItem: (key, value) => storage.set(key, value),
  },
  Blob: class FakeBlob {
    constructor(parts, options) { this.parts = parts; this.options = options; state.blob = this; }
  },
  URL: {
    createObjectURL: blob => { state.blob = blob; return "blob:fixture"; },
    revokeObjectURL: value => { state.revoked = value; },
  },
};
vm.createContext(context);
new vm.Script(runtimeSource, { filename: "START_HERE.inline.js" }).runInContext(context);
elements.start.onclick();
if (elements.reviewScreen.classList.contains("hidden")) throw new Error("review did not start");
const firstFrame = elements.frameImage.alt;
let prevented = false;
listeners.keydown({ key: "2", repeat: false, preventDefault() { prevented = true; } });
if (!prevented) throw new Error("keymap did not prevent default");
const storedRows = JSON.parse([...storage.values()][0]);
const firstSaved = Object.values(storedRows)[0];
if (firstSaved.answer !== "BOX_WRONG") throw new Error("key 2 mapped incorrectly");
if (elements.frameImage.alt === firstFrame) throw new Error("answer did not auto-advance");
if (!elements.progress.textContent.includes("1 of 32")) throw new Error("progress did not update");
if (elements.start.textContent !== "Resume review") throw new Error("resume indicator did not update");
elements.export.onclick();
if (state.downloaded !== "results.json") throw new Error("results.json was not downloaded");
const exported = state.blob.parts.join("");
if (!exported.endsWith("\n")) throw new Error("export lacks final newline");
if (JSON.parse(exported)[0].answer !== "BOX_WRONG") throw new Error("export answer drifted");
console.log(JSON.stringify({
  compiled: true,
  keymap_answer: firstSaved.answer,
  auto_advanced: true,
  local_storage_rows: Object.keys(storedRows).length,
  resume_text: elements.start.textContent,
  export_name: state.downloaded,
  export_final_newline: true,
}));
''',
            encoding="utf-8",
        )
        smoke = subprocess.run(
            [
                node,
                str(harness_path),
                str(runtime_path),
                contract_match.group(1),
                items_match.group(1),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert smoke.returncode == 0, smoke.stderr
        print(smoke.stdout.strip())
        assert json.loads(smoke.stdout) == {
            "compiled": True,
            "keymap_answer": "BOX_WRONG",
            "auto_advanced": True,
            "local_storage_rows": 1,
            "resume_text": "Resume review",
            "export_name": "results.json",
            "export_final_newline": True,
        }


def test_regeneration_refuses_nonempty_verified_labels_and_force_backs_up_first(
    tmp_path: Path,
) -> None:
    sources = _sources(10)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    out = tmp_path / "pack"
    pack.write_plan(
        out_dir=out,
        sources=sources,
        teacher=_teacher(checkpoint),
        local_media=_local_media(sources),
    )
    verified = out / "verified_labels.jsonl"
    original = b'{"frame_id":"owner-verified","verified_by_owner":true}\n'
    verified.write_bytes(original)

    with pytest.raises(pack.VerifiedLabelsOverwriteError, match="force-with-backup"):
        pack._regenerate_review_artifacts(out)
    assert verified.read_bytes() == original

    pack._regenerate_review_artifacts(out, force_with_backup=True)
    assert verified.read_bytes() == b""
    backups = list((out / "verified_label_backups").glob("*/verified_labels.jsonl"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_repeated_force_backup_excludes_prior_backups_from_live_label_scan(
    tmp_path: Path,
) -> None:
    sources = _sources(10)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    out = tmp_path / "pack"
    pack.write_plan(
        out_dir=out,
        sources=sources,
        teacher=_teacher(checkpoint),
        local_media=_local_media(sources),
    )
    verified = out / "verified_labels.jsonl"
    backup_root = out / "verified_label_backups"
    stale = backup_root / "stale" / "verified_labels.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'{"frame_id":"stale"}\n')

    for run in (1, 2):
        live = f'{{"frame_id":"live-{run}","verified_by_owner":true}}\n'.encode()
        verified.write_bytes(live)
        assert pack._nonempty_verified_label_paths(out) == [verified]
        before = set(backup_root.glob("*/verified_labels.jsonl"))
        pack._regenerate_review_artifacts(out, force_with_backup=True)
        after = set(backup_root.glob("*/verified_labels.jsonl"))
        created = after - before
        assert len(created) == 1
        assert created.pop().read_bytes() == live
        assert not list(backup_root.glob("*/verified_label_backups/**/verified_labels.jsonl"))


def test_review_items_data_island_html_escapes_script_terminators_and_round_trips(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    assert node is not None, "node is required for the emitted-page security regression"
    venue = "x</script><script>y"
    frame = {
        "output_name": "frame.jpg",
        "venue_id": venue,
        "partition": pack.PARTITION_FINETUNE,
        "sample_ordinal": 1,
        "frame_index": 7,
        "timestamp_s": 0.7,
        "image_path": "materialized/frame.jpg",
        "image_width": 96,
        "image_height": 64,
    }
    assignment = {
        "seed": pack.DEFAULT_SEED,
        "assignments": [{"venue_id": venue, "partition": pack.PARTITION_FINETUNE}],
    }
    review = pack._review_html([frame], [], assignment, [frame])
    script_bodies = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", review, flags=re.DOTALL)
    assert script_bodies
    assert all("</script>" not in body.lower() for body in script_bodies)

    items_match = re.search(
        r'<script id="review-items" type="application/json">(.*?)</script>', review
    )
    assert items_match is not None
    assert json.loads(items_match.group(1))[0]["venue"] == venue
    runtime_matches = re.findall(r"<script>(.*?)</script>", review, flags=re.DOTALL)
    assert len(runtime_matches) == 1
    runtime_path = tmp_path / "xss-shaped-review.js"
    runtime_path.write_text(runtime_matches[0], encoding="utf-8")
    completed = subprocess.run(
        [node, "--check", str(runtime_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_fix1_report_correction_records_the_honest_attribution_split() -> None:
    correction = (
        ROOT / "runs/lanes/trkC_fewshot_pack_fix1_20260722/report_correction.md"
    ).read_text(encoding="utf-8")
    assert "36 reproduced pre-existing" in correction
    assert "5 non-lane current-tree drift" in correction
    assert "1 non-lane timing-flaky" in correction
    assert "0 lane-caused" in correction
    assert "supersedes the fix1 report's `42/42 pre-existing` attribution claim" in correction


def test_review_answer_contract_embedded_mapping_never_auto_verifies_2_3_4(
    tmp_path: Path,
) -> None:
    sources = _sources(10)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    out = tmp_path / "pack"
    pack.write_plan(
        out_dir=out,
        sources=sources,
        teacher=_teacher(checkpoint),
        local_media=_local_media(sources),
    )
    pack.emit_owner_review(out)
    review = (out / "review" / "START_HERE.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script id="answer-contract" type="application/json">(.*?)</script>',
        review,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded == pack.REVIEW_ANSWER_CONTRACT
    for key in ("2", "3", "4"):
        assert embedded[key]["ground_truth"] is False
        assert embedded[key]["verified_by_owner"] is False


def test_direct_cli_reference_exposes_plan_materialize_validate_and_emit_review() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{plan,materialize,validate,emit-review}" in completed.stdout
    assert "venue-disjoint PERSON few-shot" in completed.stdout


def test_best_stack_delta_is_exact_manager_text() -> None:
    assert pack.BEST_STACK_DELTA == (
        "(c) none — data-pack tooling; the fine-tune lane that consumes this pack will carry the manifest delta."
    )
