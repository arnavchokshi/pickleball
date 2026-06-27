"""Human click-review bundle for sparse ball labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def sample_frame_indices(*, total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    count = min(int(sample_count), int(total_frames))
    if count == 1:
        return [0]
    return [int(round(idx * (total_frames - 1) / (count - 1))) for idx in range(count)]


def build_ball_points_template(
    *,
    clip: str,
    source_video: str | Path,
    fps: float,
    frame_indices: list[int],
    image_paths: list[str | Path],
) -> dict[str, Any]:
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be > 0")
    if len(frame_indices) != len(image_paths):
        raise ValueError("frame_indices and image_paths must have the same length")

    items: list[dict[str, Any]] = []
    for frame_index, image_path in zip(frame_indices, image_paths, strict=True):
        frame_idx = int(frame_index)
        image_name = Path(image_path).name
        items.append(
            {
                "review_id": f"ball_frame_{frame_idx:06d}",
                "frame_index": frame_idx,
                "frame": image_name,
                "t": frame_idx / float(fps),
                "image": Path(image_path).as_posix(),
                "ball_xy": None,
                "xy_px": None,
                "visible": None,
                "visibility": None,
                "source": "human_review",
                "class_id": 32,
                "class_name": "sports ball",
                "label": "ball",
                "confidence": None,
                "status": "needs_human_review",
                "notes": "",
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_click_review",
        "status": "needs_human_review",
        "clip": clip,
        "target_file": "ball.json",
        "review_items": [str(item["review_id"]) for item in items],
        "source_video": str(source_video),
        "coordinate_frame": "image_pixels_video_space",
        "items": items,
        "not_ground_truth": True,
    }


def export_ball_click_review_bundle(
    *,
    video_path: str | Path,
    out_dir: str | Path,
    clip: str | None = None,
    sample_count: int = 30,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    video = Path(video_path)
    if not video.is_file():
        raise ValueError(f"missing source video: {video}")

    cv2 = cv2_module or _cv2()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")

    try:
        total_frames = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        fps = _positive_float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
        height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
        if total_frames <= 0:
            raise ValueError(f"cannot determine video frame count: {video}")
        frame_indices = sample_frame_indices(total_frames=total_frames, sample_count=sample_count)

        out = Path(out_dir)
        images_dir = out / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        relative_images: list[Path] = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"failed reading frame {frame_index} from {video}")
            image_name = f"frame_{frame_index:06d}.jpg"
            image_path = images_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"failed writing review frame: {image_path}")
            relative_images.append(Path("images") / image_name)
    finally:
        cap.release()

    clip_name = clip or video.stem
    payload = build_ball_points_template(
        clip=clip_name,
        source_video=video,
        fps=fps,
        frame_indices=frame_indices,
        image_paths=relative_images,
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "ball_points.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_review_html(out / "review.html", payload)

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_click_review_export",
        "status": "needs_human_review",
        "clip": clip_name,
        "source_video": str(video),
        "out_dir": str(out),
        "frame_count": len(frame_indices),
        "frame_indices": frame_indices,
        "fps": fps,
        "width": width,
        "height": height,
        "ball_points": str(out / "ball_points.json"),
        "review_html": str(out / "review.html"),
        "not_ground_truth": True,
    }
    (out / "ball_click_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_review_html(path: str | Path, payload: dict[str, Any]) -> None:
    html_path = Path(path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_review_html(payload), encoding="utf-8")


def _review_html(payload: dict[str, Any]) -> str:
    review_json = json.dumps(payload, indent=2, sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ball Click Review - {payload.get("clip", "clip")}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101318;
      color: #eef2f6;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }}
    header, footer {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
      background: #171c23;
      border-color: #28313d;
    }}
    header {{
      border-bottom: 1px solid #28313d;
    }}
    footer {{
      border-top: 1px solid #28313d;
      flex-wrap: wrap;
    }}
    button {{
      height: 34px;
      border: 1px solid #3a4655;
      border-radius: 6px;
      background: #242c37;
      color: #eef2f6;
      padding: 0 12px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{
      background: #303a48;
    }}
    .status {{
      margin-left: auto;
      color: #a9b5c4;
      font-size: 13px;
    }}
    main {{
      display: grid;
      place-items: center;
      overflow: auto;
      padding: 14px;
    }}
    .frame-wrap {{
      position: relative;
      max-width: min(100%, 1500px);
      width: 100%;
    }}
    #frame {{
      display: block;
      width: 100%;
      height: auto;
      background: #050608;
      cursor: crosshair;
      border: 1px solid #303a48;
    }}
    #marker {{
      position: absolute;
      width: 18px;
      height: 18px;
      border: 2px solid #ffdf52;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      box-shadow: 0 0 0 2px #101318;
      display: none;
    }}
    .meta {{
      color: #a9b5c4;
      font-size: 13px;
    }}
    .hint {{
      color: #7f8b9a;
      font-size: 12px;
      margin-left: auto;
    }}
    input {{
      min-width: 240px;
      height: 32px;
      border: 1px solid #3a4655;
      border-radius: 6px;
      background: #101318;
      color: #eef2f6;
      padding: 0 10px;
    }}
  </style>
</head>
<body>
  <header>
    <strong id="clip"></strong>
    <span class="meta" id="frameMeta"></span>
    <span class="status" id="progress"></span>
  </header>
  <main>
    <div class="frame-wrap">
      <img id="frame" alt="review frame">
      <div id="marker"></div>
    </div>
  </main>
  <footer>
    <button type="button" id="prev">Previous</button>
    <button type="button" id="next">Next</button>
    <button type="button" id="missing">Mark missing</button>
    <button type="button" id="occluded">Mark occluded</button>
    <input id="notes" type="text" placeholder="notes">
    <button type="button" id="download">Download ball_points.json</button>
    <span class="hint">Keyboard: A/Left = previous, D/Right = next</span>
  </footer>
  <script>
    const reviewData = {review_json};
    let index = 0;
    const frame = document.getElementById("frame");
    const marker = document.getElementById("marker");
    const notes = document.getElementById("notes");

    function currentItem() {{
      return reviewData.items[index];
    }}

    function render() {{
      const item = currentItem();
      document.getElementById("clip").textContent = reviewData.clip;
      document.getElementById("frameMeta").textContent = `frame ${{item.frame_index}}  t=${{item.t.toFixed(3)}}s`;
      document.getElementById("progress").textContent = `${{index + 1}} / ${{reviewData.items.length}}`;
      frame.src = item.image;
      notes.value = item.notes || "";
      updateMarker();
    }}

    function updateMarker() {{
      const item = currentItem();
      if (!item.ball_xy || !frame.naturalWidth || !frame.naturalHeight) {{
        marker.style.display = "none";
        return;
      }}
      const rect = frame.getBoundingClientRect();
      marker.style.display = "block";
      marker.style.left = `${{(item.ball_xy[0] / frame.naturalWidth) * rect.width}}px`;
      marker.style.top = `${{(item.ball_xy[1] / frame.naturalHeight) * rect.height}}px`;
    }}

    function navigatePrevious() {{
      currentItem().notes = notes.value;
      index = Math.max(0, index - 1);
      render();
    }}

    function navigateNext() {{
      currentItem().notes = notes.value;
      index = Math.min(reviewData.items.length - 1, index + 1);
      render();
    }}

    frame.addEventListener("click", (event) => {{
      const rect = frame.getBoundingClientRect();
      const naturalWidth = frame.naturalWidth || rect.width;
      const naturalHeight = frame.naturalHeight || rect.height;
      const x = ((event.clientX - rect.left) / rect.width) * naturalWidth;
      const y = ((event.clientY - rect.top) / rect.height) * naturalHeight;
      const item = currentItem();
      item.ball_xy = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
      item.xy_px = item.ball_xy;
      item.visible = true;
      item.visibility = "visible";
      item.confidence = 1.0;
      item.status = "corrected_unverified";
      item.notes = notes.value;
      updateMarker();
    }});

    frame.addEventListener("load", updateMarker);
    window.addEventListener("resize", updateMarker);
    notes.addEventListener("input", () => {{
      currentItem().notes = notes.value;
    }});
    document.getElementById("prev").addEventListener("click", navigatePrevious);
    document.getElementById("next").addEventListener("click", navigateNext);
    document.addEventListener("keydown", (event) => {{
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {{
        return;
      }}
      if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") {{
        event.preventDefault();
        navigatePrevious();
      }}
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") {{
        event.preventDefault();
        navigateNext();
      }}
    }});
    document.getElementById("missing").addEventListener("click", () => {{
      const item = currentItem();
      item.ball_xy = null;
      item.xy_px = null;
      item.visible = false;
      item.visibility = "missing";
      item.confidence = 1.0;
      item.status = "corrected_unverified";
      item.notes = notes.value || "missing";
      render();
    }});
    document.getElementById("occluded").addEventListener("click", () => {{
      const item = currentItem();
      item.ball_xy = null;
      item.xy_px = null;
      item.visible = false;
      item.visibility = "occluded";
      item.confidence = 1.0;
      item.status = "corrected_unverified";
      item.notes = notes.value || "occluded";
      render();
    }});
    document.getElementById("download").addEventListener("click", () => {{
      currentItem().notes = notes.value;
      reviewData.status = "human_reviewed";
      reviewData.not_ground_truth = true;
      const blob = new Blob([JSON.stringify(reviewData, null, 2) + "\\n"], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "ball_points.json";
      link.click();
      URL.revokeObjectURL(url);
    }});

    render();
  </script>
</body>
</html>
"""


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number) and number > 0.0:
        return number
    return None


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("ball click review export requires opencv-python") from exc
    return cv2


__all__ = [
    "build_ball_points_template",
    "export_ball_click_review_bundle",
    "sample_frame_indices",
    "write_review_html",
]
