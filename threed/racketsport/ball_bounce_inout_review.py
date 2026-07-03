"""Review packets for BALL bounce timing and in/out labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .schemas import BallTrack, validate_artifact_file


REVIEW_ARTIFACT_TYPE = "racketsport_ball_bounce_inout_review"
REVIEW_EXPORT_ARTIFACT_TYPE = "racketsport_ball_bounce_inout_review_export"
REVIEWED_BOUNCES_ARTIFACT_TYPE = "racketsport_reviewed_ball_bounces"
REVIEWED_INOUT_ARTIFACT_TYPE = "racketsport_reviewed_ball_inout"
ACCEPTED_STATUSES = {"accepted", "human_reviewed"}


def export_ball_bounce_inout_review_bundle(
    *,
    video_path: str | Path,
    ball_track_path: str | Path,
    out_dir: str | Path,
    clip: str | None = None,
    context_frames: int = 2,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Export a fail-closed review packet for candidate bounces/in-out calls."""

    video = Path(video_path)
    track_path = Path(ball_track_path)
    out = Path(out_dir)
    if not video.is_file():
        raise ValueError(f"missing source video: {video}")
    if context_frames < 0:
        raise ValueError("context_frames must be >= 0")
    track = validate_artifact_file("ball_track", track_path)
    if not isinstance(track, BallTrack):
        raise ValueError(f"{track_path} did not validate as BallTrack")

    cv2 = cv2_module or _cv2()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")

    try:
        total_frames = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        fps = _positive_float(cap.get(cv2.CAP_PROP_FPS)) or float(track.fps)
        width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
        height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
        if total_frames <= 0:
            raise ValueError(f"cannot determine video frame count: {video}")

        images_dir = out / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, Any]] = []
        for bounce_index, bounce in enumerate(track.bounces):
            predicted_frame = int(bounce.frame) if bounce.frame is not None else round(float(bounce.t) * float(track.fps))
            context_images: list[dict[str, Any]] = []
            for frame_index in range(predicted_frame - context_frames, predicted_frame + context_frames + 1):
                if frame_index < 0 or frame_index >= total_frames:
                    continue
                image_name = f"bounce_{bounce_index:04d}_frame_{frame_index:06d}.jpg"
                image_path = images_dir / image_name
                _write_video_frame(cv2=cv2, cap=cap, frame_index=frame_index, image_path=image_path)
                context_images.append(
                    {
                        "frame": frame_index,
                        "t": frame_index / fps,
                        "image": (Path("images") / image_name).as_posix(),
                    }
                )
            items.append(_review_item_from_bounce(bounce_index, bounce, fps=float(track.fps), context_images=context_images))
    finally:
        cap.release()

    clip_name = clip or video.stem
    payload = {
        "schema_version": 1,
        "artifact_type": REVIEW_ARTIFACT_TYPE,
        "status": "needs_human_review",
        "clip": clip_name,
        "source_video": str(video),
        "ball_track_path": str(track_path),
        "fps": float(track.fps),
        "video_fps": fps,
        "video_frame_count": total_frames,
        "video_resolution": [width, height],
        "coordinate_frame": "image_pixels_video_space",
        "instructions": {
            "reviewed_bounce_frame": "Set to the visually reviewed bounce frame, within the provided context when possible.",
            "reviewed_call": "Set to in, out, or too_close_to_call after visual review; leave null if unreviewed.",
        },
        "items": items,
        "not_ground_truth": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    review_json = out / "ball_bounce_inout_review.json"
    review_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_review_html(out / "review.html", payload)
    summary = {
        "schema_version": 1,
        "artifact_type": REVIEW_EXPORT_ARTIFACT_TYPE,
        "status": "needs_human_review",
        "clip": clip_name,
        "source_video": str(video),
        "ball_track_path": str(track_path),
        "out_dir": str(out),
        "candidate_count": len(items),
        "review_json": str(review_json),
        "review_html": str(out / "review.html"),
        "not_ground_truth": True,
    }
    (out / "ball_bounce_inout_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_reviewed_bounce_inout_labels(review: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build M4/M5 reviewed-label artifacts from explicit human review decisions."""

    payload = _load_json_or_mapping(review)
    if payload.get("artifact_type") != REVIEW_ARTIFACT_TYPE:
        raise ValueError(f"review artifact_type must be {REVIEW_ARTIFACT_TYPE}")
    fps = _positive_float(payload.get("fps"))
    if fps is None:
        raise ValueError("review fps must be finite and > 0")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("review items must be a list")

    reviewed_bounces: list[dict[str, Any]] = []
    reviewed_calls: list[dict[str, Any]] = []
    pending_count = 0
    rejected_count = 0
    for item in items:
        if not isinstance(item, Mapping):
            pending_count += 1
            continue
        status = item.get("status")
        if status not in ACCEPTED_STATUSES:
            if status == "rejected":
                rejected_count += 1
            else:
                pending_count += 1
            continue
        frame = _int_or_none(item.get("reviewed_bounce_frame"))
        call = item.get("reviewed_call")
        if frame is None or frame < 0 or call not in {"in", "out", "too_close_to_call"}:
            pending_count += 1
            continue
        review_id = str(item.get("review_id") or f"bounce_{len(reviewed_bounces):04d}")
        reviewed_bounces.append({"frame": frame, "t": frame / fps, "review_id": review_id})
        if call in {"in", "out"}:
            reviewed_calls.append({"frame": frame, "t": frame / fps, "call": call, "review_id": review_id})

    status = "human_reviewed" if pending_count == 0 else "partial_human_review"
    common = {
        "schema_version": 1,
        "clip": payload.get("clip"),
        "fps": fps,
        "status": status,
        "source": "human_review",
        "review_artifact_type": REVIEW_ARTIFACT_TYPE,
        "reviewed_item_count": len(reviewed_bounces),
        "pending_review_count": pending_count,
        "rejected_review_count": rejected_count,
    }
    return (
        {
            **common,
            "artifact_type": REVIEWED_BOUNCES_ARTIFACT_TYPE,
            "bounces": reviewed_bounces,
        },
        {
            **common,
            "artifact_type": REVIEWED_INOUT_ARTIFACT_TYPE,
            "calls": reviewed_calls,
        },
    )


def write_reviewed_bounce_inout_labels(
    *,
    review_path: str | Path,
    out_bounces: str | Path,
    out_inout: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bounces, inout = build_reviewed_bounce_inout_labels(review_path)
    out_bounces_path = Path(out_bounces)
    out_inout_path = Path(out_inout)
    out_bounces_path.parent.mkdir(parents=True, exist_ok=True)
    out_inout_path.parent.mkdir(parents=True, exist_ok=True)
    out_bounces_path.write_text(json.dumps(bounces, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_inout_path.write_text(json.dumps(inout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bounces, inout


def write_review_html(path: str | Path, payload: Mapping[str, Any]) -> None:
    html_path = Path(path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    review_json = json.dumps(payload, indent=2, sort_keys=True)
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ball Bounce/In-Out Review - {payload.get("clip", "clip")}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101318; color: #eef2f6; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 16px; }}
    header {{ display: flex; gap: 12px; align-items: center; justify-content: space-between; }}
    button, input, select {{ height: 34px; border: 1px solid #3a4655; border-radius: 6px; background: #202833; color: #eef2f6; padding: 0 10px; font: inherit; }}
    button {{ cursor: pointer; }}
    button:hover {{ background: #2c3644; }}
    label {{ display: grid; gap: 4px; color: #a9b5c4; font-size: 12px; }}
    .candidate {{ border-top: 1px solid #2b3542; padding: 14px 0; }}
    .candidate.reviewed {{ border-left: 4px solid #40c463; padding-left: 10px; }}
    .candidate.rejected {{ border-left: 4px solid #ef6b73; padding-left: 10px; }}
    .frames {{ display: flex; gap: 8px; overflow-x: auto; margin: 10px 0; }}
    .frame-card {{ display: grid; gap: 4px; color: #a9b5c4; font-size: 12px; text-align: center; }}
    img {{ max-height: 260px; border: 1px solid #303a48; background: #050608; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: end; }}
    pre {{ white-space: pre-wrap; background: #171c23; border: 1px solid #2b3542; padding: 12px; max-height: 280px; overflow: auto; }}
    .muted {{ color: #a9b5c4; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Ball Bounce/In-Out Review</h1>
      <p class="muted">Review each candidate visually. Download the updated JSON, then run build_reviewed_ball_bounce_inout.py.</p>
    </div>
    <button type="button" id="downloadJson">Download JSON</button>
  </header>
  <div id="items"></div>
  <h2>Current JSON</h2>
  <pre id="json"></pre>
</main>
<script>
const reviewData = {review_json};
const items = document.getElementById("items");
const jsonPreview = document.getElementById("json");

function allowedCall(value) {{
  return ["in", "out", "too_close_to_call"].includes(value) ? value : "too_close_to_call";
}}

function refreshJson() {{
  jsonPreview.textContent = JSON.stringify(reviewData, null, 2);
}}

function saveCurrentReview(index) {{
  const item = reviewData.items[index];
  const section = document.querySelector(`[data-review-index="${{index}}"]`);
  const frameInput = section.querySelector("[data-reviewed-frame]");
  const callInput = section.querySelector("[data-reviewed-call]");
  const notesInput = section.querySelector("[data-review-notes]");
  const frameValue = Number.parseInt(frameInput.value, 10);
  item.reviewed_bounce_frame = Number.isFinite(frameValue) ? frameValue : null;
  item.reviewed_call = callInput.value || null;
  item.review_notes = notesInput.value || "";
  if (item.status !== "rejected") {{
    item.status = item.reviewed_bounce_frame !== null && item.reviewed_call ? "accepted" : "needs_human_review";
  }}
  section.classList.toggle("reviewed", item.status === "accepted");
  section.classList.toggle("rejected", item.status === "rejected");
  refreshJson();
}}

function setPredicted(index) {{
  const item = reviewData.items[index];
  item.status = "accepted";
  item.reviewed_bounce_frame = item.predicted_frame;
  item.reviewed_call = allowedCall(item.predicted_call);
  render();
}}

function setTooClose(index) {{
  const item = reviewData.items[index];
  item.status = "accepted";
  item.reviewed_bounce_frame = item.reviewed_bounce_frame ?? item.predicted_frame;
  item.reviewed_call = "too_close_to_call";
  render();
}}

function rejectCandidate(index) {{
  const item = reviewData.items[index];
  item.status = "rejected";
  item.reviewed_bounce_frame = null;
  item.reviewed_call = null;
  render();
}}

function downloadReviewJson() {{
  const blob = new Blob([JSON.stringify(reviewData, null, 2) + "\\n"], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "ball_bounce_inout_review.json";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}}

function render() {{
  items.innerHTML = "";
  (reviewData.items || []).forEach((item, index) => {{
    const node = document.createElement("section");
    node.className = "candidate";
    node.dataset.reviewIndex = String(index);
    node.classList.toggle("reviewed", item.status === "accepted");
    node.classList.toggle("rejected", item.status === "rejected");
    const frames = (item.context_images || []).map(image => `<div class="frame-card"><img src="${{image.image}}" alt="frame ${{image.frame}}"><span>frame ${{image.frame}}</span></div>`).join("");
    node.innerHTML = `<h2>${{item.review_id}} · predicted frame ${{item.predicted_frame}} · ${{item.predicted_call}}</h2>
      <p class="muted">margin=${{item.predicted_margin_m ?? "n/a"}}m · uncertainty=${{item.predicted_uncertainty_m ?? "n/a"}}m · line=${{item.nearest_line ?? "n/a"}}</p>
      <div class="frames">${{frames}}</div>
      <div class="controls">
        <label>reviewed_bounce_frame<input data-reviewed-frame type="number" step="1" value="${{item.reviewed_bounce_frame ?? ""}}" placeholder="${{item.predicted_frame}}"></label>
        <label>reviewed_call<select data-reviewed-call>
          <option value="">unreviewed</option>
          <option value="in" ${{item.reviewed_call === "in" ? "selected" : ""}}>in</option>
          <option value="out" ${{item.reviewed_call === "out" ? "selected" : ""}}>out</option>
          <option value="too_close_to_call" ${{item.reviewed_call === "too_close_to_call" ? "selected" : ""}}>too_close_to_call</option>
        </select></label>
        <label>notes<input data-review-notes type="text" value="${{item.review_notes || ""}}"></label>
        <button type="button" data-save>Save item</button>
        <button type="button" data-accept>Accept predicted</button>
        <button type="button" data-gray>Mark too close</button>
        <button type="button" data-reject>Reject candidate</button>
      </div>
      <pre>${{JSON.stringify(item, null, 2)}}</pre>`;
    node.querySelector("[data-save]").addEventListener("click", () => saveCurrentReview(index));
    node.querySelector("[data-accept]").addEventListener("click", () => setPredicted(index));
    node.querySelector("[data-gray]").addEventListener("click", () => setTooClose(index));
    node.querySelector("[data-reject]").addEventListener("click", () => rejectCandidate(index));
    items.appendChild(node);
  }});
  refreshJson();
}}

document.getElementById("downloadJson").addEventListener("click", downloadReviewJson);
render();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _review_item_from_bounce(
    bounce_index: int,
    bounce: Any,
    *,
    fps: float,
    context_images: list[dict[str, Any]],
) -> dict[str, Any]:
    frame = int(bounce.frame) if bounce.frame is not None else round(float(bounce.t) * fps)
    return {
        "review_id": f"bounce_{bounce_index:04d}",
        "status": "needs_human_review",
        "predicted_t": float(bounce.t),
        "predicted_frame": frame,
        "predicted_world_xy": list(bounce.world_xy),
        "predicted_contact_xy_img": list(bounce.contact_xy_img) if bounce.contact_xy_img is not None else None,
        "predicted_p_bounce": float(bounce.p_bounce) if bounce.p_bounce is not None else None,
        "predicted_call": bounce.call,
        "predicted_margin_m": float(bounce.margin_m) if bounce.margin_m is not None else None,
        "predicted_uncertainty_m": float(bounce.uncertainty_m) if bounce.uncertainty_m is not None else None,
        "predicted_confidence": float(bounce.confidence) if bounce.confidence is not None else None,
        "nearest_line": bounce.nearest_line,
        "region": bounce.region,
        "source": bounce.source,
        "context_images": context_images,
        "reviewed_bounce_frame": None,
        "reviewed_call": None,
        "review_notes": "",
    }


def _write_video_frame(*, cv2: Any, cap: Any, frame_index: int, image_path: Path) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"failed reading frame {frame_index}")
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"failed writing review frame: {image_path}")


def _load_json_or_mapping(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _cv2() -> Any:
    import cv2  # type: ignore

    return cv2


__all__ = [
    "build_reviewed_bounce_inout_labels",
    "export_ball_bounce_inout_review_bundle",
    "write_reviewed_bounce_inout_labels",
]
