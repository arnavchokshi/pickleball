"""One-page local review server for BALL bounce and in/out decisions."""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse


REVIEW_ARTIFACT_TYPE = "racketsport_ball_bounce_inout_review"
REVIEW_FILE = "ball_bounce_inout_review.json"
VALID_ACTIONS = {"in", "out", "too_close", "reject"}


def build_simple_review_state(*, root: str | Path, clips: Iterable[str]) -> dict[str, Any]:
    root_path = Path(root)
    flattened: list[dict[str, Any]] = []
    clip_counts: dict[str, dict[str, int]] = {}

    for clip in clips:
        review_path = root_path / clip / REVIEW_FILE
        payload = _load_review(review_path)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{review_path} items must be a list")

        counts = _count_items(items)
        clip_counts[clip] = counts
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            context_images = []
            for image in item.get("context_images") or []:
                if not isinstance(image, Mapping):
                    continue
                rel_image = str(image.get("image") or "")
                context_images.append(
                    {
                        "frame": image.get("frame"),
                        "t": image.get("t"),
                        "url": f"/media/{clip}/{rel_image}",
                    }
                )
            flattened.append(
                {
                    "clip": clip,
                    "clip_label": _friendly_clip_name(clip),
                    "clip_position": index + 1,
                    "clip_total": len(items),
                    "review_id": item.get("review_id"),
                    "status": item.get("status"),
                    "predicted_frame": item.get("predicted_frame"),
                    "predicted_call": item.get("predicted_call"),
                    "predicted_margin_m": item.get("predicted_margin_m"),
                    "reviewed_bounce_frame": item.get("reviewed_bounce_frame"),
                    "reviewed_call": item.get("reviewed_call"),
                    "context_images": context_images,
                }
            )

    return {
        "clips": list(clips),
        "clip_counts": clip_counts,
        "items": flattened,
        "totals": _count_items(flattened),
    }


def apply_simple_review_decision(
    *,
    review_path: str | Path,
    clip: str,
    review_id: str,
    action: str,
    frame: int | None,
) -> dict[str, int]:
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}")

    path = Path(review_path)
    payload = _load_review(path)
    if payload.get("clip") != clip:
        raise ValueError(f"review clip mismatch: expected {clip!r}, got {payload.get('clip')!r}")

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} items must be a list")

    item = _find_item(items, review_id)
    if action == "reject":
        item["status"] = "rejected"
        item["reviewed_bounce_frame"] = None
        item["reviewed_call"] = None
        item["review_notes"] = "simple_review: rejected"
    else:
        selected_frame = _valid_frame(frame, item)
        item["status"] = "accepted"
        item["reviewed_bounce_frame"] = selected_frame
        item["reviewed_call"] = "too_close_to_call" if action == "too_close" else action
        item["review_notes"] = f"simple_review: {action}"

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _count_items(items)


def run_simple_review_server(*, root: str | Path, clips: Iterable[str], host: str, port: int) -> None:
    root_path = Path(root).resolve()
    clip_list = list(clips)
    handler = _make_handler(root=root_path, clips=clip_list)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"BALL review UI: http://{host}:{server.server_port}/")
    server.serve_forever()


def _load_review(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if payload.get("artifact_type") != REVIEW_ARTIFACT_TYPE:
        raise ValueError(f"{path} artifact_type must be {REVIEW_ARTIFACT_TYPE}")
    return payload


def _find_item(items: list[Any], review_id: str) -> dict[str, Any]:
    for item in items:
        if isinstance(item, dict) and item.get("review_id") == review_id:
            return item
    raise ValueError(f"unknown review_id {review_id!r}")


def _valid_frame(frame: int | None, item: Mapping[str, Any]) -> int:
    candidate = frame if isinstance(frame, int) and frame >= 0 else item.get("predicted_frame")
    if not isinstance(candidate, int) or candidate < 0:
        raise ValueError("frame must be a non-negative integer")
    return candidate


def _count_items(items: Iterable[Any]) -> dict[str, int]:
    accepted = 0
    rejected = 0
    pending = 0
    too_close = 0
    inout = 0
    total = 0
    for item in items:
        if not isinstance(item, Mapping):
            pending += 1
            total += 1
            continue
        total += 1
        status = item.get("status")
        reviewed_call = item.get("reviewed_call")
        if status in {"accepted", "human_reviewed"}:
            accepted += 1
            if reviewed_call == "too_close_to_call":
                too_close += 1
            if reviewed_call in {"in", "out"}:
                inout += 1
        elif status == "rejected":
            rejected += 1
        else:
            pending += 1
    return {
        "accepted": accepted,
        "rejected": rejected,
        "pending": pending,
        "too_close": too_close,
        "inout": inout,
        "total": total,
    }


def _friendly_clip_name(clip: str) -> str:
    return clip.replace("_", " ").title()


def _make_handler(*, root: Path, clips: list[str]) -> type[BaseHTTPRequestHandler]:
    class SimpleReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(_html(), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                self._send_json(build_simple_review_state(root=root, clips=clips))
                return
            if parsed.path.startswith("/media/"):
                self._send_media(parsed.path)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/decision":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be a JSON object")
                clip = str(payload.get("clip") or "")
                if clip not in clips:
                    raise ValueError(f"unknown clip {clip!r}")
                result = apply_simple_review_decision(
                    review_path=root / clip / REVIEW_FILE,
                    clip=clip,
                    review_id=str(payload.get("review_id") or ""),
                    action=str(payload.get("action") or ""),
                    frame=payload.get("frame") if isinstance(payload.get("frame"), int) else None,
                )
                self._send_json({"ok": True, "clip_counts": result})
            except Exception as exc:  # pragma: no cover - exercised through integration smoke.
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, *, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, request_path: str) -> None:
            parts = [unquote(part) for part in request_path.split("/") if part]
            if len(parts) < 3:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            _, clip, *relative_parts = parts
            if clip not in clips or any(part in {"", ".", ".."} for part in relative_parts):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            media_path = (root / clip / Path(*relative_parts)).resolve()
            clip_root = (root / clip).resolve()
            if not media_path.is_file() or clip_root not in [media_path, *media_path.parents]:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = media_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(media_path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return SimpleReviewHandler


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BALL Bounce Review</title>
  <style>
    :root {
      --bg: #f6f3ed;
      --ink: #1f252b;
      --muted: #65707a;
      --line: #d7d0c4;
      --panel: #fffdfa;
      --selected: #135f68;
      --in: #127a4a;
      --out: #9d2f24;
      --close: #8a5a09;
      --no: #3b4652;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
      padding: 18px;
    }
    header, footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }
    h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1;
      letter-spacing: 0;
    }
    .stats {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-weight: 700;
      font-size: 13px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      background: rgba(255,255,255,0.55);
    }
    .stage {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 12px;
      align-content: center;
    }
    .meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
      font-weight: 700;
    }
    .filmstrip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }
    .frame {
      border: 3px solid transparent;
      border-radius: 8px;
      background: var(--panel);
      padding: 8px;
      box-shadow: 0 8px 24px rgba(31,37,43,0.08);
      cursor: pointer;
    }
    .frame.selected {
      border-color: var(--selected);
      box-shadow: 0 0 0 4px rgba(19,95,104,0.18);
    }
    .frame img {
      width: 100%;
      height: min(42vh, 330px);
      object-fit: contain;
      background: #101318;
      border-radius: 5px;
      display: block;
    }
    .frame span {
      display: block;
      padding-top: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
      text-align: center;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    button {
      border: 0;
      border-radius: 8px;
      min-height: 74px;
      padding: 12px;
      color: white;
      font: inherit;
      font-size: clamp(16px, 2vw, 24px);
      font-weight: 900;
      letter-spacing: 0;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .in { background: var(--in); }
    .out { background: var(--out); }
    .close { background: var(--close); }
    .no { background: var(--no); }
    .nav {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .nav button {
      min-height: 42px;
      padding: 8px 13px;
      color: var(--ink);
      background: var(--panel);
      border: 1px solid var(--line);
      font-size: 14px;
    }
    .done {
      display: none;
      align-self: center;
      justify-self: center;
      text-align: center;
      max-width: 720px;
      padding: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .done h2 {
      margin: 0 0 8px;
      font-size: clamp(28px, 5vw, 56px);
      letter-spacing: 0;
    }
    @media (max-width: 760px) {
      main { padding: 12px; }
      header, footer, .meta { align-items: stretch; flex-direction: column; }
      .stats { justify-content: flex-start; }
      .actions { grid-template-columns: 1fr 1fr; }
      button { min-height: 62px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>BALL Bounce Review</h1>
    <div class="stats" id="stats"></div>
  </header>
  <section class="stage" id="stage">
    <div class="meta">
      <div id="clipName"></div>
      <div id="candidateNumber"></div>
    </div>
    <div class="filmstrip" id="filmstrip"></div>
  </section>
  <section class="done" id="done">
    <h2>All reviewed</h2>
    <p>The JSON files in the four clip folders are updated.</p>
  </section>
  <footer>
    <div class="actions">
      <button class="in" data-action="in">IN</button>
      <button class="out" data-action="out">OUT</button>
      <button class="close" data-action="too_close">TOO CLOSE</button>
      <button class="no" data-action="reject">NOT A BOUNCE</button>
    </div>
    <div class="nav">
      <button id="prev">Previous</button>
      <button id="skip">Skip</button>
    </div>
  </footer>
</main>
<script>
let state = null;
let index = 0;
let selectedFrame = null;

async function loadState(preferredNext) {
  const response = await fetch("/api/state");
  state = await response.json();
  index = typeof preferredNext === "number" ? preferredNext : firstPendingIndex();
  render();
}

function firstPendingIndex() {
  const pending = state.items.findIndex(item => item.status !== "accepted" && item.status !== "human_reviewed" && item.status !== "rejected");
  return pending >= 0 ? pending : 0;
}

function currentItem() {
  return state.items[index];
}

function render() {
  const item = currentItem();
  const stage = document.getElementById("stage");
  const done = document.getElementById("done");
  const footer = document.querySelector("footer");
  const allDone = state.totals.pending === 0;
  stage.style.display = allDone ? "none" : "grid";
  footer.style.display = allDone ? "none" : "flex";
  done.style.display = allDone ? "block" : "none";
  renderStats();
  if (allDone || !item) return;

  selectedFrame = item.reviewed_bounce_frame ?? item.predicted_frame;
  document.getElementById("clipName").textContent = item.clip_label;
  document.getElementById("candidateNumber").textContent = `${index + 1} of ${state.items.length} · picked frame ${selectedFrame}`;

  const filmstrip = document.getElementById("filmstrip");
  filmstrip.innerHTML = "";
  item.context_images.forEach(image => {
    const frame = document.createElement("button");
    frame.type = "button";
    frame.className = "frame";
    frame.dataset.frame = String(image.frame);
    frame.innerHTML = `<img src="${image.url}" alt=""><span>frame ${image.frame}</span>`;
    frame.addEventListener("click", () => {
      selectedFrame = Number(image.frame);
      document.querySelectorAll(".frame").forEach(node => node.classList.remove("selected"));
      frame.classList.add("selected");
      document.getElementById("candidateNumber").textContent = `${index + 1} of ${state.items.length} · picked frame ${selectedFrame}`;
    });
    if (Number(image.frame) === Number(selectedFrame)) {
      frame.classList.add("selected");
    }
    filmstrip.appendChild(frame);
  });
}

function renderStats() {
  const totals = state.totals;
  document.getElementById("stats").innerHTML = [
    `${totals.total - totals.pending}/${totals.total} done`,
    `${totals.pending} left`,
    `${totals.inout} in/out`,
    `${totals.too_close} too close`,
    `${totals.rejected} rejected`,
  ].map(text => `<span class="pill">${text}</span>`).join("");
}

async function decide(action) {
  const item = currentItem();
  setButtons(false);
  const response = await fetch("/api/decision", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      clip: item.clip,
      review_id: item.review_id,
      frame: selectedFrame,
      action,
    }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    alert(payload.error || "Save failed");
    setButtons(true);
    return;
  }
  await loadState(nextReviewIndex(index));
  setButtons(true);
}

function nextReviewIndex(fromIndex) {
  for (let offset = 1; offset <= state.items.length; offset += 1) {
    const next = (fromIndex + offset) % state.items.length;
    const item = state.items[next];
    if (item.status !== "accepted" && item.status !== "human_reviewed" && item.status !== "rejected") return next;
  }
  return fromIndex;
}

function setButtons(enabled) {
  document.querySelectorAll("button").forEach(button => { button.disabled = !enabled; });
}

document.querySelectorAll("[data-action]").forEach(button => {
  button.addEventListener("click", () => decide(button.dataset.action));
});
document.getElementById("prev").addEventListener("click", () => {
  index = Math.max(0, index - 1);
  render();
});
document.getElementById("skip").addEventListener("click", () => {
  index = Math.min(state.items.length - 1, index + 1);
  render();
});

loadState();
</script>
</body>
</html>
"""


__all__ = [
    "apply_simple_review_decision",
    "build_simple_review_state",
    "run_simple_review_server",
]
