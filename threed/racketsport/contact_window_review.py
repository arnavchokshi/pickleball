"""Human review templates and fail-closed promotion for contact windows."""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Mapping

from .schemas import ContactWindowCandidates, ContactWindowReview, ContactWindows


ARTIFACT_TYPE = "racketsport_contact_window_review"
SCHEMA_VERSION = 1


def build_contact_window_review_template(
    candidates: ContactWindowCandidates | Mapping[str, Any],
    *,
    candidate_path: str | Path,
) -> dict[str, Any]:
    """Create an editable review file with every candidate left pending."""

    artifact = _candidates(candidates)
    decisions = [
        {
            "review_id": candidate.review_id,
            "decision": "pending",
            "reviewer": "",
            "reason": "",
            "player_id": None,
            "t_override": None,
            "frame_override": None,
            "confidence_override": None,
            "window_override": None,
        }
        for candidate in artifact.candidates
    ]
    return _review_payload(
        clip=artifact.clip,
        candidate_path=str(candidate_path),
        decisions=decisions,
    )


def promote_reviewed_contact_windows(
    candidates: ContactWindowCandidates | Mapping[str, Any],
    review: ContactWindowReview | Mapping[str, Any],
) -> dict[str, Any]:
    """Promote accepted review decisions into schema-valid ContactWindows."""

    candidate_artifact = _candidates(candidates)
    review_artifact = _review(review)
    if review_artifact.clip != candidate_artifact.clip:
        raise ValueError(f"review clip {review_artifact.clip} does not match candidates clip {candidate_artifact.clip}")

    candidates_by_id = _candidate_map(candidate_artifact)
    seen_decisions: set[str] = set()
    events: list[dict[str, Any]] = []
    for decision in review_artifact.decisions:
        if decision.review_id in seen_decisions:
            raise ValueError(f"duplicate review decision for {decision.review_id}")
        seen_decisions.add(decision.review_id)
        if decision.review_id not in candidates_by_id:
            raise ValueError(f"review decision {decision.review_id} does not match any candidate")
        if decision.decision != "accepted":
            continue
        if not decision.reviewer.strip() or not decision.reason.strip():
            raise ValueError(f"accepted decision {decision.review_id} requires reviewer and reason")

        candidate = candidates_by_id[decision.review_id]
        if candidate.type != "contact":
            raise ValueError(f"accepted decision {decision.review_id} has type {candidate.type}; only contact can promote")
        window = decision.window_override or candidate.window
        events.append(
            {
                "type": candidate.type,
                "t": _finite(decision.t_override if decision.t_override is not None else candidate.t, "t"),
                "frame": _frame(decision.frame_override if decision.frame_override is not None else candidate.frame),
                "player_id": decision.player_id,
                "confidence": _confidence(
                    decision.confidence_override
                    if decision.confidence_override is not None
                    else candidate.candidate_confidence,
                    "confidence",
                ),
                "sources": {
                    "audio": 0.0,
                    "wrist_vel": 0.0,
                    "ball_inflection": 0.0,
                    "human_review": 1.0,
                },
                "window": {
                    "t0": _finite(window.t0, "window.t0"),
                    "t1": _finite(window.t1, "window.t1"),
                    "importance": _confidence(window.importance, "window.importance"),
                },
            }
        )

    payload = {"schema_version": SCHEMA_VERSION, "events": events}
    ContactWindows.model_validate(payload)
    return payload


def render_contact_window_review_html(
    candidates: ContactWindowCandidates | Mapping[str, Any],
    review: ContactWindowReview | Mapping[str, Any],
    *,
    review_filename: str = "contact_window_review.json",
    media_paths: list[Mapping[str, str]] | tuple[Mapping[str, str], ...] = (),
) -> str:
    """Render a browser-friendly contact review page."""

    candidate_artifact = _candidates(candidates)
    review_artifact = _review(review)
    decisions = {decision.review_id: decision for decision in review_artifact.decisions}
    rows = []
    target_cards = []
    for candidate in candidate_artifact.candidates:
        decision = decisions.get(candidate.review_id)
        target_cards.append(
            "<article class=\"target-card\">"
            f"<h3>{escape(candidate.review_id)}</h3>"
            "<p>"
            f"Review <strong>{candidate.window.t0:.3f}-{candidate.window.t1:.3f}s</strong> "
            f"(frame <strong>{candidate.frame}</strong>). Decide whether this is a real paddle-ball contact."
            "</p>"
            f"<p class=\"muted\">Source label: {escape(candidate.source_label)}; "
            f"confidence {candidate.source_confidence:.2f}; current decision "
            f"<strong>{escape(decision.decision if decision else 'missing')}</strong>.</p>"
            "</article>"
        )
        rows.append(
            "<tr>"
            f"<td><code>{escape(candidate.review_id)}</code></td>"
            f"<td>{escape(candidate.type)}</td>"
            f"<td>{candidate.frame}</td>"
            f"<td>{candidate.t:.3f}</td>"
            f"<td>{escape(candidate.source_label)}</td>"
            f"<td>{escape(candidate.source_status)} / {candidate.source_confidence:.2f}</td>"
            f"<td>{candidate.window.t0:.3f}-{candidate.window.t1:.3f}</td>"
            f"<td><strong>{escape(decision.decision if decision else 'missing')}</strong></td>"
            f"<td>{escape(str(decision.player_id) if decision and decision.player_id is not None else '')}</td>"
            f"<td>{escape(decision.reviewer if decision else '')}</td>"
            f"<td>{escape(decision.reason if decision else '')}</td>"
            "</tr>"
        )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>Contact Window Review - {escape(candidate_artifact.clip)}</title>",
            "<style>",
            _review_html_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            f"<h1>{escape(candidate_artifact.clip)}</h1>",
            '<p class="lede">Contact-window review for adaptive BODY scheduling.</p>',
            '<dl class="summary">',
            _summary_item("Candidate file", str(review_artifact.candidate_path)),
            _summary_item("Review file", review_filename),
            _summary_item("Promotion target", str(review_artifact.promotion_target)),
            _summary_item("Status", str(review_artifact.status)),
            _summary_item("Pending", str(review_artifact.summary.pending_count)),
            _summary_item("Accepted", str(review_artifact.summary.accepted_count)),
            _summary_item("Rejected", str(review_artifact.summary.rejected_count)),
            "</dl>",
            "</header>",
            "<section class=\"review-guide\">",
            "<h2>What to review</h2>",
            "<p>Watch the visual context at the listed time window. Mark the row accepted only if the ball visibly contacts a paddle; otherwise mark it rejected. If accepted, include the player id when it is obvious.</p>",
            "<div class=\"target-grid\">",
            *target_cards,
            "</div>",
            "</section>",
            _media_section(media_paths, start_time=_first_candidate_start(candidate_artifact)),
            "<section>",
            "<h2>Decision table</h2>",
            "<table>",
            "<thead><tr>"
            "<th>Review ID</th><th>Type</th><th>Frame</th><th>Time</th><th>Label</th>"
            "<th>Source</th><th>Window</th><th>Decision</th><th>Player</th><th>Reviewer</th><th>Reason</th>"
            "</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def read_contact_window_candidates(path: str | Path) -> ContactWindowCandidates:
    return ContactWindowCandidates.model_validate(_read_json(path))


def read_contact_window_review(path: str | Path) -> ContactWindowReview:
    return ContactWindowReview.model_validate(_read_json(path))


def write_contact_window_review(path: str | Path, payload: Mapping[str, Any]) -> None:
    ContactWindowReview.model_validate(payload)
    _write_json(path, payload)


def write_contact_windows(path: str | Path, payload: Mapping[str, Any]) -> None:
    ContactWindows.model_validate(payload)
    _write_json(path, payload)


def write_contact_window_review_html(path: str | Path, html: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


def contact_review_media_paths(clip_dir: str | Path) -> list[dict[str, str]]:
    root = Path(clip_dir)
    candidates = [
        {
            "label": "Ball overlay",
            "path": root / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj_overlay_h264.mp4",
        },
        {"label": "Player tracks", "path": root / "player_tracks" / "player_track_overlay_h264.mp4"},
        {"label": "Paddle candidates", "path": root / "racket_candidates" / "racket_candidate_overlay_h264.mp4"},
    ]
    media: list[dict[str, str]] = []
    for item in candidates:
        path = item["path"]
        if path.is_file():
            media.append({"label": str(item["label"]), "path": path.name if path.parent == root else str(path.relative_to(root))})
    return media


def _review_html_css() -> str:
    return """
:root {
  color-scheme: light;
  font-family: Avenir Next, ui-sans-serif, system-ui, sans-serif;
  color: #171717;
  background: #f5f3ee;
}
body {
  margin: 0;
}
main {
  margin: 0 auto;
  padding: 24px 16px 40px;
  width: min(1280px, 100%);
}
h1, p {
  margin-top: 0;
}
h1 {
  font-size: 28px;
  line-height: 1.15;
  margin-bottom: 6px;
}
.lede {
  color: #57534e;
  margin-bottom: 18px;
}
.summary {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 0 0 18px;
}
.summary div {
  background: #ffffff;
  border: 1px solid #d8d3c8;
  border-radius: 6px;
  padding: 10px 12px;
}
dt {
  color: #6b7280;
  font-size: 11px;
  letter-spacing: 0;
  text-transform: uppercase;
}
dd {
  margin: 4px 0 0;
  overflow-wrap: anywhere;
}
section {
  overflow-x: auto;
  margin-top: 18px;
}
h2 {
  font-size: 18px;
  line-height: 1.2;
  margin: 0 0 10px;
}
h3 {
  font-size: 14px;
  line-height: 1.25;
  margin: 0 0 6px;
}
.review-guide,
.media-card,
.target-card {
  background: #ffffff;
  border: 1px solid #d8d3c8;
  border-radius: 6px;
}
.review-guide {
  padding: 14px;
}
.review-guide p {
  margin-bottom: 10px;
}
.target-grid,
.media-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.target-card,
.media-card {
  padding: 12px;
}
.muted {
  color: #57534e;
}
.media-card video {
  aspect-ratio: 16 / 9;
  background: #111111;
  border-radius: 4px;
  display: block;
  margin-top: 10px;
  width: 100%;
}
table {
  background: #ffffff;
  border-collapse: collapse;
  border: 1px solid #d8d3c8;
  min-width: 1120px;
  width: 100%;
}
th, td {
  border-bottom: 1px solid #e7e2d8;
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #22211f;
  color: #ffffff;
  font-size: 12px;
}
td {
  font-size: 13px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
strong {
  color: #1d4ed8;
}
@media (max-width: 700px) {
  main {
    padding-inline: 10px;
  }
  h1 {
    font-size: 22px;
  }
}
""".strip()


def _summary_item(label: str, value: str) -> str:
    return f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"


def _first_candidate_start(candidates: ContactWindowCandidates) -> float:
    starts = [candidate.window.t0 for candidate in candidates.candidates]
    return max(0.0, min(starts)) if starts else 0.0


def _media_section(media_paths: list[Mapping[str, str]] | tuple[Mapping[str, str], ...], *, start_time: float) -> str:
    if not media_paths:
        return (
            '<section class="review-guide">'
            "<h2>Visual context</h2>"
            "<p>No local review videos were found beside this contact review file. Use the frame/time in the table with the source clip.</p>"
            "</section>"
        )
    cards = []
    fragment = f"#t={start_time:.3f}"
    for media in media_paths:
        label = str(media.get("label", "Video"))
        path = str(media.get("path", ""))
        if not path:
            continue
        src = escape(path + fragment, quote=True)
        cards.append(
            '<article class="media-card">'
            f"<h3>{escape(label)}</h3>"
            f'<video controls preload="metadata" src="{src}"></video>'
            "</article>"
        )
    return "\n".join(
        [
            '<section class="media-section">',
            "<h2>Visual context</h2>",
            '<div class="media-grid">',
            *cards,
            "</div>",
            "</section>",
        ]
    )


def _review_payload(*, clip: str, candidate_path: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summary(decisions)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "candidate_path": candidate_path,
        "promotion_target": "contact_windows.json",
        "status": _status(summary),
        "decisions": decisions,
        "summary": summary,
    }


def _summary(decisions: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"pending": 0, "accepted": 0, "rejected": 0}
    for decision in decisions:
        value = str(decision.get("decision", "pending"))
        if value not in counts:
            raise ValueError(f"unsupported decision {value}")
        counts[value] += 1
    return {
        "candidate_count": len(decisions),
        "pending_count": counts["pending"],
        "accepted_count": counts["accepted"],
        "rejected_count": counts["rejected"],
    }


def _status(summary: Mapping[str, int]) -> str:
    if summary["pending_count"] == summary["candidate_count"]:
        return "pending_review"
    if summary["pending_count"] > 0:
        return "partially_reviewed"
    return "reviewed"


def _candidate_map(artifact: ContactWindowCandidates) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for candidate in artifact.candidates:
        if candidate.review_id in by_id:
            raise ValueError(f"duplicate candidate review_id {candidate.review_id}")
        by_id[candidate.review_id] = candidate
    return by_id


def _candidates(value: ContactWindowCandidates | Mapping[str, Any]) -> ContactWindowCandidates:
    if isinstance(value, ContactWindowCandidates):
        return value
    return ContactWindowCandidates.model_validate(value)


def _review(value: ContactWindowReview | Mapping[str, Any]) -> ContactWindowReview:
    if isinstance(value, ContactWindowReview):
        return value
    return ContactWindowReview.model_validate(value)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: float | int, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _confidence(value: float | int, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return result


def _frame(value: int) -> int:
    frame = int(value)
    if frame < 0:
        raise ValueError("frame must be non-negative")
    return frame


__all__ = [
    "ARTIFACT_TYPE",
    "build_contact_window_review_template",
    "contact_review_media_paths",
    "promote_reviewed_contact_windows",
    "read_contact_window_candidates",
    "read_contact_window_review",
    "render_contact_window_review_html",
    "write_contact_window_review",
    "write_contact_window_review_html",
    "write_contact_windows",
]
