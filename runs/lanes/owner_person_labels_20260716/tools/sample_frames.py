#!/usr/bin/env python3
"""Build the deterministic, blind owner person-box review pack.

The real ``sample`` command extracts a seeded candidate pool with ffmpeg,
runs the repository YOLO26m checkpoint locally, selects the requested strata,
and writes an offline still-image review page.  The pure selection and page
functions are intentionally public so determinism and blindness can be tested
without rerunning the detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


SEED = 20260716
SESSION_ID = "person_labels_20260716"
SCHEMA_VERSION = 1
CREATED_AT = "2026-07-16T00:00:00Z"
TARGETS = {"gameplay": 300, "spectator_rich": 100, "empty_sparse": 50}
SCRATCH_FRACTION = 0.18
CONFIDENCE = 0.25
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
PAGE_STORAGE_KEY = "person_labels_20260716"
ELIGIBLE_CLIP_COUNT = 30
PORTRAIT_COUNT = 2
SCREEN_RECORDING_COUNT = 9


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_int(seed: int, *parts: object) -> int:
    encoded = "|".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _opaque_id(seed: int, clip_id: str, timestamp_s: float) -> str:
    value = f"{seed}|{clip_id}|{timestamp_s:.6f}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _probe(path: Path) -> dict[str, Any]:
    payload = _run_json(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ]
    )
    stream = payload["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    return {
        "duration_s": float(payload["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "fps_rational": stream["avg_frame_rate"],
    }


def _image_number(name: str) -> int:
    match = re.fullmatch(r"IMG_(\d{4})(?: \(\d+\))?\.mov", name)
    if not match:
        raise ValueError(f"not an IMG gameplay filename: {name}")
    return int(match.group(1))


def _session_map(names: Iterable[str], *, max_gap: int = 100) -> dict[str, str]:
    """Cluster adjacent IMG numbers; duplicate names stay in the same session."""
    materialized = list(names)
    ordered_numbers = sorted({_image_number(name) for name in materialized})
    number_to_session: dict[int, str] = {}
    cluster = 0
    previous: int | None = None
    for number in ordered_numbers:
        if previous is None or number - previous > max_gap:
            cluster += 1
        number_to_session[number] = f"session_{cluster:02d}"
        previous = number
    return {name: number_to_session[_image_number(name)] for name in materialized}


def discover_clips(root: Path, probe_list: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    probe_list = probe_list.resolve()
    rows: list[tuple[str, int, int, float, str]] = []
    pattern = re.compile(r"^(LANDSCAPE|portrait) (\d+)x(\d+) ([0-9.]+)s (.+\.mov)$")
    for raw_line in probe_list.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(raw_line.strip())
        if not match:
            raise ValueError(f"unparseable probe row: {raw_line!r}")
        orientation, width, height, duration, name = match.groups()
        rows.append((orientation, int(width), int(height), float(duration), name))

    portrait = [row for row in rows if row[0] == "portrait"]
    screens = [row for row in rows if row[0] == "LANDSCAPE" and row[4].startswith("Screen Recording")]
    eligible_rows = [row for row in rows if row[0] == "LANDSCAPE" and row[4].startswith("IMG_")]
    if len(portrait) != PORTRAIT_COUNT:
        raise ValueError(f"expected {PORTRAIT_COUNT} portrait exclusions, found {len(portrait)}")
    if len(screens) != SCREEN_RECORDING_COUNT:
        raise ValueError(f"expected {SCREEN_RECORDING_COUNT} landscape screen recordings, found {len(screens)}")
    if len(eligible_rows) != ELIGIBLE_CLIP_COUNT:
        raise ValueError(f"expected {ELIGIBLE_CLIP_COUNT} eligible gameplay clips, found {len(eligible_rows)}")

    raw_root = (root / "runs/owner_footage_intake_20260702/raw").resolve()
    mapping = _session_map(row[4] for row in eligible_rows)
    clips: list[dict[str, Any]] = []
    for _orientation, listed_width, listed_height, listed_duration, name in sorted(eligible_rows, key=lambda row: row[4]):
        path = (raw_root / name).resolve()
        if not path.is_relative_to(raw_root) or not path.is_file():
            raise ValueError(f"missing or escaped gameplay clip: {path}")
        probe = _probe(path)
        if probe["width"] <= probe["height"]:
            raise ValueError(f"portrait clip entered gameplay universe: {path}")
        if (probe["width"], probe["height"]) != (listed_width, listed_height):
            raise ValueError(f"probe-list resolution drift for {name}")
        if abs(probe["duration_s"] - listed_duration) > 1.1:
            raise ValueError(f"probe-list duration drift for {name}")
        clips.append(
            {
                "clip_id": path.stem,
                "clip_name": name,
                "video_path": path.relative_to(root).as_posix(),
                "video_sha256": _sha256(path),
                "session_id": mapping[name],
                **probe,
            }
        )
    return clips, {
        "probe_rows": len(rows),
        "landscape_rows": sum(row[0] == "LANDSCAPE" for row in rows),
        "eligible_gameplay_clips": len(clips),
        "excluded_landscape_screen_recordings": len(screens),
        "excluded_portrait_files": len(portrait),
        "session_cluster_rule": "sort numeric IMG suffixes; start a new cluster when adjacent suffix gap exceeds 100",
        "sessions": {session: sorted(clip["clip_id"] for clip in clips if clip["session_id"] == session) for session in sorted({clip["session_id"] for clip in clips})},
    }


def _largest_remainder(total: int, weights: dict[str, float], *, minimum: int = 0) -> dict[str, int]:
    if total < minimum * len(weights):
        raise ValueError("allocation total cannot satisfy minimum")
    result = {key: minimum for key in weights}
    remaining = total - sum(result.values())
    weight_sum = sum(weights.values())
    ideals = {key: remaining * value / weight_sum for key, value in weights.items()}
    for key, value in ideals.items():
        result[key] += math.floor(value)
    leftovers = total - sum(result.values())
    ordered = sorted(weights, key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key))
    for key in ordered[:leftovers]:
        result[key] += 1
    return result


def candidate_plan(clips: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_session[clip["session_id"]].append(clip)
    session_quotas = _largest_remainder(
        count,
        {session: sum(float(clip["duration_s"]) for clip in rows) for session, rows in by_session.items()},
        minimum=8,
    )
    plan: list[dict[str, Any]] = []
    for session, session_clips in sorted(by_session.items()):
        clip_quotas = _largest_remainder(
            session_quotas[session],
            {clip["clip_id"]: float(clip["duration_s"]) for clip in session_clips},
        )
        for clip in sorted(session_clips, key=lambda row: row["clip_id"]):
            quota = clip_quotas[clip["clip_id"]]
            rng = random.Random(_stable_int(seed, "candidate", session, clip["clip_id"]))
            timestamps: set[float] = set()
            low, high = 0.25, max(0.251, float(clip["duration_s"]) - 0.25)
            attempts = 0
            while len(timestamps) < quota:
                attempts += 1
                if attempts > max(10000, quota * 1000):
                    raise ValueError(f"unable to draw {quota} unique timestamps for {clip['clip_id']}")
                frame = round(rng.uniform(low, high) * float(clip["fps"]))
                timestamp = round(frame / float(clip["fps"]), 6)
                if low <= timestamp <= high:
                    timestamps.add(timestamp)
            for timestamp in sorted(timestamps):
                plan.append(
                    {
                        "candidate_id": _opaque_id(seed, clip["clip_id"], timestamp),
                        "clip_id": clip["clip_id"],
                        "clip_name": clip["clip_name"],
                        "video_path": clip["video_path"],
                        "video_sha256": clip["video_sha256"],
                        "session_id": session,
                        "timestamp_s": timestamp,
                        "source_fps": clip["fps"],
                    }
                )
    return sorted(plan, key=lambda row: row["candidate_id"])


def extract_candidates(root: Path, plan: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(plan, start=1):
        output = out_dir / f"{row['candidate_id']}.jpg"
        if output.is_file():
            continue
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{float(row['timestamp_s']):.6f}", "-i", str(root / row["video_path"]),
            "-frames:v", "1", "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",
            "-q:v", "2", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode or not output.is_file():
            raise RuntimeError(f"ffmpeg extraction failed for candidate {index}: {completed.stderr.strip()}")
        row["candidate_file"] = output.name


def _attach_detection(row: dict[str, Any], result: Any) -> None:
    boxes: list[dict[str, Any]] = []
    if result.boxes is not None:
        coords = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        for xyxy, confidence in zip(coords, confidences, strict=True):
            boxes.append(
                {
                    "x1": round(max(0.0, min(OUTPUT_WIDTH, float(xyxy[0]))), 3),
                    "y1": round(max(0.0, min(OUTPUT_HEIGHT, float(xyxy[1]))), 3),
                    "x2": round(max(0.0, min(OUTPUT_WIDTH, float(xyxy[2]))), 3),
                    "y2": round(max(0.0, min(OUTPUT_HEIGHT, float(xyxy[3]))), 3),
                    "confidence": round(float(confidence), 6),
                }
            )
    row["proposals"] = boxes
    count = len(boxes)
    row["stratum"] = "spectator_rich" if count > 4 else "empty_sparse" if count <= 2 else "gameplay"


def detect_people(
    plan: list[dict[str, Any]],
    image_dir: Path,
    model_path: Path,
    *,
    device: str,
    checkpoint_path: Path | None = None,
    batch_size: int = 8,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency blocker exercised operationally
        raise RuntimeError("STOP_MISSING_VENV_DEPENDENCY: ultralytics") from exc
    cached: dict[str, Any] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("model_sha256") != _sha256(model_path) or payload.get("confidence") != CONFIDENCE:
            raise ValueError("detection checkpoint model/config does not match")
        cached = dict(payload.get("detections", {}))
    for row in plan:
        detection = cached.get(row["candidate_id"])
        if detection is not None:
            row["proposals"] = detection["proposals"]
            row["stratum"] = detection["stratum"]

    remaining = [row for row in plan if "proposals" not in row]
    if not remaining:
        return
    model = YOLO(str(model_path))
    total = len(plan)
    completed_before = total - len(remaining)
    for offset in range(0, len(remaining), batch_size):
        chunk = remaining[offset : offset + batch_size]
        paths = [str(image_dir / f"{row['candidate_id']}.jpg") for row in chunk]
        results = model.predict(
            paths,
            conf=CONFIDENCE,
            classes=[0],
            imgsz=640,
            device=device,
            batch=batch_size,
            verbose=False,
            stream=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"detector returned {len(results)} results for batch of {len(chunk)}")
        for row, result in zip(chunk, results, strict=True):
            _attach_detection(row, result)
            cached[row["candidate_id"]] = {"proposals": row["proposals"], "stratum": row["stratum"]}
        if checkpoint_path is not None:
            _write_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "model_sha256": _sha256(model_path),
                    "confidence": CONFIDENCE,
                    "detections": cached,
                },
            )
        print(
            f"detected {completed_before + min(offset + len(chunk), len(remaining))}/{total}",
            file=sys.stderr,
            flush=True,
        )
    if any("proposals" not in row for row in plan):
        raise RuntimeError("detector checkpoint is incomplete after prediction")


def _select_across_sessions(pool: list[dict[str, Any]], target: int, *, seed: int, label: str) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_session[row["session_id"]].append(row)
    for session, rows in by_session.items():
        rows.sort(key=lambda row: _stable_int(seed, "select", label, session, row["candidate_id"]))
    sessions = sorted(by_session, key=lambda session: _stable_int(seed, "session-order", label, session))
    selected: list[dict[str, Any]] = []
    while len(selected) < target and sessions:
        next_sessions: list[str] = []
        for session in sessions:
            if by_session[session] and len(selected) < target:
                selected.append(by_session[session].pop())
            if by_session[session]:
                next_sessions.append(session)
        sessions = next_sessions
    return selected


def build_pack_manifest(
    candidates: list[dict[str, Any]],
    universe: dict[str, Any],
    *,
    seed: int = SEED,
    targets: dict[str, int] | None = None,
    generator_sha256: str = "test-fixture",
    model_sha256: str = "test-fixture",
) -> dict[str, Any]:
    targets = dict(targets or TARGETS)
    pools = {name: [dict(row) for row in candidates if row["stratum"] == name] for name in targets}
    selected: list[dict[str, Any]] = []
    actuals: Counter[str] = Counter()
    backfills: list[dict[str, Any]] = []
    used: set[str] = set()
    for stratum in ("spectator_rich", "empty_sparse"):
        rows = _select_across_sessions(pools[stratum], targets[stratum], seed=seed, label=stratum)
        selected.extend(rows)
        used.update(row["candidate_id"] for row in rows)
        actuals[stratum] += len(rows)
        deficit = targets[stratum] - len(rows)
        if deficit:
            gameplay_available = [row for row in pools["gameplay"] if row["candidate_id"] not in used]
            fills = _select_across_sessions(gameplay_available, deficit, seed=seed, label=f"{stratum}-gameplay-fill")
            for row in fills:
                row["selection_stratum"] = stratum
                row["backfilled_from"] = "gameplay"
            selected.extend(fills)
            used.update(row["candidate_id"] for row in fills)
            actuals[stratum] += len(fills)
            backfills.append({"target_stratum": stratum, "requested": deficit, "filled_from_gameplay": len(fills)})
    gameplay_available = [row for row in pools["gameplay"] if row["candidate_id"] not in used]
    gameplay_rows = _select_across_sessions(gameplay_available, targets["gameplay"], seed=seed, label="gameplay")
    selected.extend(gameplay_rows)
    actuals["gameplay"] += len(gameplay_rows)
    if len(gameplay_rows) != targets["gameplay"]:
        raise ValueError(f"gameplay pool shortfall: needed {targets['gameplay']}, found {len(gameplay_rows)}")
    if len(selected) != sum(targets.values()):
        raise ValueError(f"final pack shortfall: needed {sum(targets.values())}, found {len(selected)}")

    scratch_targets = _largest_remainder(
        round(len(selected) * SCRATCH_FRACTION),
        {key: float(value) for key, value in actuals.items()},
    )
    for stratum in sorted(actuals):
        rows = [row for row in selected if row.get("selection_stratum", row["stratum"]) == stratum]
        ranked = sorted(rows, key=lambda row: _stable_int(seed, "scratch", stratum, row["candidate_id"]))
        scratch_ids = {row["candidate_id"] for row in ranked[: scratch_targets[stratum]]}
        for row in rows:
            row["scratch"] = row["candidate_id"] in scratch_ids

    random.Random(_stable_int(seed, "presentation")).shuffle(selected)
    frames: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        frame_id = row["candidate_id"]
        frames.append(
            {
                "frame_id": frame_id,
                "presentation_index": index,
                "filename": f"{frame_id}.jpg",
                "clip_id": row["clip_id"],
                "clip_name": row.get("clip_name", f"{row['clip_id']}.mov"),
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "session_id": row["session_id"],
                "timestamp_s": row["timestamp_s"],
                "source_fps": row["source_fps"],
                "stratum": row.get("selection_stratum", row["stratum"]),
                "detected_stratum": row["stratum"],
                "backfilled_from": row.get("backfilled_from"),
                "scratch": bool(row["scratch"]),
                "proposals": row["proposals"],
            }
        )

    session_frame_counts = Counter(frame["session_id"] for frame in frames)
    sessions = sorted(session_frame_counts)
    holdout_target = round(len(frames) * 0.20)
    holdout_options = [
        combo
        for size in (1, 2)
        for combo in combinations(sessions, size)
    ]
    validation_combo = min(
        holdout_options,
        key=lambda combo: (
            abs(sum(session_frame_counts[session] for session in combo) - holdout_target),
            len(combo),
            _stable_int(seed, "validation", *combo),
        ),
    )
    validation_sessions = set(validation_combo)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "owner_person_box_review_pack",
        "session_id": SESSION_ID,
        "seed": seed,
        "created_at": CREATED_AT,
        "generator_sha256": generator_sha256,
        "model": {"path": "models/checkpoints/yolo26m.pt", "sha256": model_sha256, "confidence": CONFIDENCE, "class": "person"},
        "image": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT, "format": "jpeg", "ffmpeg_qscale": 2},
        "universe": universe,
        "candidate_count": len(candidates),
        "candidate_strata": dict(sorted(Counter(row["stratum"] for row in candidates).items())),
        "targets": targets,
        "actuals": dict(sorted(actuals.items())),
        "backfills": backfills,
        "scratch": {"count": sum(frame["scratch"] for frame in frames), "fraction": round(sum(frame["scratch"] for frame in frames) / len(frames), 6), "targets_by_stratum": scratch_targets},
        "split": {
            "policy": "session-disjoint; deterministic 1-2 whole-session validation holdout closest to 20% of frames",
            "target_validation_frames": holdout_target,
            "actual_validation_frames": sum(session_frame_counts[session] for session in validation_sessions),
            "session_to_split": {session: "validation" if session in validation_sessions else "train" for session in sessions},
        },
        "frames": frames,
        "verified": False,
    }


def _page_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for frame in manifest["frames"]:
        proposals = [] if frame["scratch"] else [
            {
                "x1": box["x1"], "y1": box["y1"], "x2": box["x2"], "y2": box["y2"],
                "class": "player", "origin": f"p{index}",
            }
            for index, box in enumerate(frame["proposals"])
        ]
        items.append({"id": frame["frame_id"], "file": f"frames/{frame['filename']}", "boxes": proposals})
    return items


def render_review_html(manifest: dict[str, Any]) -> str:
    """Return a file://-safe page containing no source/selection metadata."""
    items_json = json.dumps(_page_items(manifest), separators=(",", ":"), sort_keys=True)
    session_json = json.dumps(manifest["session_id"])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Person box review</title><style>
:root{{--ink:#17211d;--paper:#f5f0e4;--panel:#fffdf6;--green:#18864b;--orange:#db711d;--yellow:#f1d848;--muted:#6c746f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.35 system-ui,-apple-system,sans-serif}}
header{{display:flex;gap:16px;align-items:center;padding:12px 18px;background:var(--ink);color:white;position:sticky;top:0;z-index:5}}
#progress{{font-size:20px;font-weight:800}}#doneState{{color:var(--yellow);font-weight:800}}#status{{margin-left:auto;color:#dbe3df}}
main{{max-width:1280px;margin:0 auto;padding:14px}}#stage{{position:relative;aspect-ratio:16/9;background:#111;border-radius:12px;overflow:hidden;box-shadow:0 12px 35px #0002;touch-action:none}}
#photo,#overlay{{position:absolute;inset:0;width:100%;height:100%}}#photo{{object-fit:contain;pointer-events:none}}#overlay{{cursor:default}}
.box{{fill:transparent;stroke-width:5;vector-effect:non-scaling-stroke}}.player{{stroke:var(--green)}}.off_court_person{{stroke:var(--orange)}}
.selected{{stroke-dasharray:10 7}}.handle{{fill:white;stroke:var(--ink);stroke-width:2;vector-effect:non-scaling-stroke}}
.toolbar{{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px;align-items:center}}button{{border:0;border-radius:10px;padding:12px 16px;font-weight:750;background:var(--ink);color:white;cursor:pointer}}
button.alt{{background:white;color:var(--ink);border:2px solid #d5d0c5}}button.active{{outline:4px solid var(--yellow)}}button.danger{{background:#a6322b}}button.save{{background:#176b90}}button.export{{background:var(--green);font-size:18px}}
#legend{{margin-left:auto;color:var(--muted)}}.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin:0 5px 0 12px}}.green{{background:var(--green)}}.orange{{background:var(--orange)}}
.help{{padding:10px 2px;color:var(--muted)}}#toast{{min-height:24px;color:#a6322b;font-weight:700}}@media(max-width:700px){{button{{padding:10px}}#legend{{width:100%;margin:0}}}}
</style></head><body><header><div id="progress"></div><div id="doneState"></div><div id="status"></div></header>
<main><div id="stage"><img id="photo" alt="Review frame"><svg id="overlay" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid meet"></svg></div>
<div class="toolbar"><button id="prev" class="alt">← Previous</button><button id="next">Next →</button><button id="draw">W · New box</button><button id="toggle" class="alt">C · Toggle class</button><button id="remove" class="danger">X · Delete</button><button id="empty" class="alt">E · Confirm empty</button><button id="save" class="save">S · Save progress</button><button id="export" class="export">Export labels</button><span id="legend"><i class="dot green"></i>player <i class="dot orange"></i>off-court person</span></div>
<div class="help">Drag to draw. Select a box, then drag it or its corner/edge handles to adjust. A/D or ←/→ navigate. Every change saves automatically.</div><div id="toast"></div></main>
<script>
const ITEMS={items_json},SESSION_ID={session_json},KEY="{PAGE_STORAGE_KEY}",W=1920,H=1080;
const $=id=>document.getElementById(id),photo=$("photo"),svg=$("overlay");
const fresh=()=>({{schema_version:1,session_id:SESSION_ID,index:0,frames:{{}}}});
let state;try{{state=JSON.parse(localStorage.getItem(KEY)||"null")||fresh()}}catch(_e){{state=fresh()}}
if(state.session_id!==SESSION_ID||!state.frames)state=fresh();state.index=Math.max(0,Math.min(ITEMS.length-1,Number(state.index)||0));
let selected=null,mode="select",gesture=null,entered=performance.now();
function clone(x){{return JSON.parse(JSON.stringify(x))}}function current(){{return ITEMS[state.index]}}
function ensure(){{const id=current().id;if(!state.frames[id])state.frames[id]={{boxes:current().boxes.map(b=>({{...clone(b),source:"proposal_confirmed",deleted:false}})),empty_confirmed:false,reviewed:false,ms_spent:0}};return state.frames[id]}}
function active(rec=ensure()){{return rec.boxes.filter(b=>!b.deleted)}}function done(rec=ensure()){{return !!rec.reviewed&&(rec.empty_confirmed||active(rec).length>0)}}
function account(){{const rec=ensure();rec.ms_spent=Math.round((Number(rec.ms_spent)||0)+performance.now()-entered);entered=performance.now()}}
function persist(){{account();localStorage.setItem(KEY,JSON.stringify(state));$("status").textContent="Saved locally"}}
function mutate(fn,clearEmpty=true){{const rec=ensure();fn(rec);if(clearEmpty)rec.empty_confirmed=false;rec.reviewed=true;persist();renderBoxes();renderHeader()}}
function esc(s){{return String(s).replace(/[&<>\"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]))}}
function handles(b,i){{if(i!==selected)return"";const x=(b.x1+b.x2)/2,y=(b.y1+b.y2)/2,pts=[[b.x1,b.y1,"nw"],[x,b.y1,"n"],[b.x2,b.y1,"ne"],[b.x2,y,"e"],[b.x2,b.y2,"se"],[x,b.y2,"s"],[b.x1,b.y2,"sw"],[b.x1,y,"w"]];return pts.map(p=>`<circle class="handle" data-i="${{i}}" data-h="${{p[2]}}" cx="${{p[0]}}" cy="${{p[1]}}" r="10"/>`).join("")}}
function renderBoxes(){{const rec=ensure();svg.innerHTML=rec.boxes.map((b,i)=>b.deleted?"":`<g><rect data-i="${{i}}" class="box ${{esc(b.class)}} ${{i===selected?"selected":""}}" x="${{b.x1}}" y="${{b.y1}}" width="${{b.x2-b.x1}}" height="${{b.y2-b.y1}}"/>${{handles(b,i)}}</g>`).join("");svg.style.cursor=mode==="draw"?"crosshair":"default";$("draw").classList.toggle("active",mode==="draw")}}
function renderHeader(){{$("progress").textContent=`${{state.index+1}} of ${{ITEMS.length}}`;$("doneState").textContent=done()?"✓ Done":"○ Needs review"}}
function preload(){{if(state.index+1<ITEMS.length){{const im=new Image();im.src=ITEMS[state.index+1].file}}}}
function render(){{selected=null;mode="select";const item=current();photo.src=item.file;ensure();renderBoxes();renderHeader();preload();persist()}}
function go(delta){{if(delta>0){{const rec=ensure();if(active(rec).length||rec.empty_confirmed)rec.reviewed=true}}persist();state.index=Math.max(0,Math.min(ITEMS.length-1,state.index+delta));entered=performance.now();render()}}
function point(e){{const r=svg.getBoundingClientRect();return{{x:Math.max(0,Math.min(W,(e.clientX-r.left)/r.width*W)),y:Math.max(0,Math.min(H,(e.clientY-r.top)/r.height*H))}}}}
function changed(b){{if(b.origin&&b.source==="proposal_confirmed")b.source="proposal_adjusted"}}
svg.addEventListener("pointerdown",e=>{{const p=point(e),target=e.target,i=Number(target.dataset.i);svg.setPointerCapture(e.pointerId);if(mode==="draw"){{const rec=ensure();rec.empty_confirmed=false;rec.reviewed=true;rec.boxes.push({{x1:p.x,y1:p.y,x2:p.x,y2:p.y,class:"player",origin:null,source:"drawn",deleted:false}});selected=rec.boxes.length-1;gesture={{kind:"draw",i:selected,start:p}};return}}if(Number.isInteger(i)){{selected=i;const b=ensure().boxes[i];gesture={{kind:target.dataset.h?"resize":"move",handle:target.dataset.h||null,i,start:p,box:clone(b)}};renderBoxes()}}else{{selected=null;renderBoxes()}}}});
svg.addEventListener("pointermove",e=>{{if(!gesture)return;const p=point(e),b=ensure().boxes[gesture.i];if(gesture.kind==="draw"){{b.x1=Math.min(gesture.start.x,p.x);b.y1=Math.min(gesture.start.y,p.y);b.x2=Math.max(gesture.start.x,p.x);b.y2=Math.max(gesture.start.y,p.y)}}else if(gesture.kind==="move"){{const dx=p.x-gesture.start.x,dy=p.y-gesture.start.y,w=gesture.box.x2-gesture.box.x1,h=gesture.box.y2-gesture.box.y1;b.x1=Math.max(0,Math.min(W-w,gesture.box.x1+dx));b.y1=Math.max(0,Math.min(H-h,gesture.box.y1+dy));b.x2=b.x1+w;b.y2=b.y1+h;changed(b)}}else{{const h=gesture.handle;if(h.includes("w"))b.x1=Math.min(p.x,b.x2-2);if(h.includes("e"))b.x2=Math.max(p.x,b.x1+2);if(h.includes("n"))b.y1=Math.min(p.y,b.y2-2);if(h.includes("s"))b.y2=Math.max(p.y,b.y1+2);changed(b)}}renderBoxes()}});
function endGesture(){{if(!gesture)return;const rec=ensure(),b=rec.boxes[gesture.i];if(b.x2-b.x1<4||b.y2-b.y1<4){{b.deleted=true}}rec.reviewed=true;if(active(rec).length)rec.empty_confirmed=false;gesture=null;mode="select";persist();renderBoxes();renderHeader()}}svg.addEventListener("pointerup",endGesture);svg.addEventListener("pointercancel",endGesture);
function toggleClass(){{if(selected===null)return;mutate(rec=>{{const b=rec.boxes[selected];b.class=b.class==="player"?"off_court_person":"player";changed(b)}})}}
function removeSelected(){{if(selected===null)return;mutate(rec=>{{const b=rec.boxes[selected];if(b.origin){{b.deleted=true;b.source="proposal_deleted"}}else rec.boxes.splice(selected,1);selected=null}})}}
function confirmEmpty(){{if(active().length){{$("toast").textContent="Delete all visible boxes before confirming empty.";return}}mutate(rec=>{{rec.empty_confirmed=true;rec.reviewed=true}},false);$("toast").textContent="Empty frame confirmed."}}
function exportBox(b){{return{{x1:+b.x1.toFixed(3),y1:+b.y1.toFixed(3),x2:+b.x2.toFixed(3),y2:+b.y2.toFixed(3),class:b.class,source:b.source,...(b.deleted?{{deleted:true}}:{{}})}}}}
function exportNow(){{persist();const stored=JSON.parse(localStorage.getItem(KEY));const frames=ITEMS.filter(it=>stored.frames[it.id]).map(it=>{{const r=stored.frames[it.id];return{{frame_id:it.id,boxes:r.boxes.map(exportBox),empty_confirmed:!!r.empty_confirmed,ms_spent:Math.round(r.ms_spent||0)}}}});const payload={{schema_version:1,session_id:SESSION_ID,storage_key:KEY,frames}};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:"application/json"}}));a.download="person_labels_export.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}
$("prev").onclick=()=>go(-1);$("next").onclick=()=>go(1);$("draw").onclick=()=>{{mode=mode==="draw"?"select":"draw";renderBoxes()}};$("toggle").onclick=toggleClass;$("remove").onclick=removeSelected;$("empty").onclick=confirmEmpty;$("save").onclick=()=>{{persist();$("toast").textContent="Progress saved in this browser."}};$("export").onclick=exportNow;
document.addEventListener("keydown",e=>{{if(["INPUT","TEXTAREA"].includes(e.target.tagName))return;const k=e.key.toLowerCase();if(k==="a"||e.key==="ArrowLeft"){{e.preventDefault();go(-1)}}else if(k==="d"||e.key==="ArrowRight"){{e.preventDefault();go(1)}}else if(k==="w"){{mode=mode==="draw"?"select":"draw";renderBoxes()}}else if(k==="x")removeSelected();else if(k==="c")toggleClass();else if(k==="e")confirmEmpty();else if(k==="s")persist()}});window.addEventListener("beforeunload",persist);render();
</script></body></html>'''


def validate_blind_page(html: str, manifest: dict[str, Any]) -> None:
    lowered = html.lower()
    forbidden = ["scratch", "spectator_rich", "empty_sparse", "detected_stratum", "clip_name", "video_path"]
    forbidden.extend(str(frame["clip_name"]).lower() for frame in manifest["frames"])
    hits = sorted({term for term in forbidden if term in lowered})
    if hits:
        raise AssertionError(f"blind page leaks protected sampling/source metadata: {hits[:5]}")
    if "<video" in lowered:
        raise AssertionError("review page must contain no native video element")
    items = _page_items(manifest)
    by_id = {frame["frame_id"]: frame for frame in manifest["frames"]}
    for item in items:
        if by_id[item["id"]]["scratch"] and item["boxes"]:
            raise AssertionError(f"withheld proposals leaked for {item['id']}")


def stage_pack(manifest: dict[str, Any], candidate_dir: Path, staging_dir: Path) -> dict[str, Any]:
    frames_dir = staging_dir / "frames"
    if staging_dir.exists():
        for child in staging_dir.iterdir():
            if child.name not in {"START_HERE.html", "frames"}:
                raise ValueError(f"refusing to replace unexpected staging artifact: {child}")
        shutil.rmtree(staging_dir)
    frames_dir.mkdir(parents=True)
    for frame in manifest["frames"]:
        source = candidate_dir / f"{frame['frame_id']}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, frames_dir / frame["filename"])
    html = render_review_html(manifest)
    validate_blind_page(html, manifest)
    (staging_dir / "START_HERE.html").write_text(html, encoding="utf-8")
    return {
        "frames": len(list(frames_dir.glob("*.jpg"))),
        "page_bytes": (staging_dir / "START_HERE.html").stat().st_size,
        "pack_bytes": sum(path.stat().st_size for path in staging_dir.rglob("*") if path.is_file()),
        "page_blind": True,
        "scratch_withheld": True,
        "contains_video_element": False,
    }


def run_sample(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    clips, universe = discover_clips(root, args.probe_list)
    candidate_dir = args.work_dir / "candidate_frames"
    candidate_manifest_path = args.work_dir / "candidate_manifest.json"
    if args.reuse_candidates:
        candidates = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))["candidates"]
    else:
        candidates = candidate_plan(clips, count=args.candidate_count, seed=args.seed)
        extract_candidates(root, candidates, candidate_dir)
        detect_people(
            candidates,
            candidate_dir,
            args.model,
            device=args.device,
            checkpoint_path=args.work_dir / "candidate_detections.json",
        )
        _write_json(candidate_manifest_path, {"schema_version": 1, "seed": args.seed, "universe": universe, "candidates": candidates})
    manifest = build_pack_manifest(
        candidates,
        universe,
        seed=args.seed,
        generator_sha256=_sha256(Path(__file__)),
        model_sha256=_sha256(args.model),
    )
    _write_json(args.manifest, manifest)
    staging = stage_pack(manifest, candidate_dir, args.staging_dir)
    return {"manifest": str(args.manifest), "candidate_manifest": str(candidate_manifest_path), "staging": staging, "actuals": manifest["actuals"], "scratch": manifest["scratch"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="Run extraction, YOLO stratification, selection, and staging")
    sample.add_argument("--root", type=Path, default=Path("."))
    sample.add_argument("--probe-list", type=Path, default=Path("runs/lanes/trk_detbench_20260716/owner_footage_probe_20260716.txt"))
    sample.add_argument("--model", type=Path, default=Path("models/checkpoints/yolo26m.pt"))
    sample.add_argument("--work-dir", type=Path, required=True)
    sample.add_argument("--manifest", type=Path, required=True)
    sample.add_argument("--staging-dir", type=Path, required=True)
    sample.add_argument("--candidate-count", type=int, default=1200)
    sample.add_argument("--seed", type=int, default=SEED)
    sample.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    sample.add_argument("--reuse-candidates", action="store_true")
    page = sub.add_parser("render-page", help="Regenerate a blind page/frames from a manifest and candidate directory")
    page.add_argument("--manifest", type=Path, required=True)
    page.add_argument("--candidate-dir", type=Path, required=True)
    page.add_argument("--staging-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "sample":
        payload = run_sample(args)
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        payload = stage_pack(manifest, args.candidate_dir, args.staging_dir)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
