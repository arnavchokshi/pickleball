"""Rally-span gating: derive rally spans from cheap, already-computed signals.

Owner directive (2026-07-02): "We do mesh people only at contact windows,
otherwise joints [locked decision]. Find MORE cuts like that -- faster while
still an accurate representation." This module is the biggest single documented
cut from ``BALL_TRACKING_PIPELINE.md`` section 5.1: "process rally spans only,
dead-time skipped".

This is a **runtime optimization, not a model**. It does not predict anything
about shot quality, in/out calls, or biomechanics -- it only decides which
seconds of a clip are worth spending expensive compute on (BALL heatmap
inference, BODY mesh, court-relative association, world build, ...). Because a
false negative here silently deletes a real point from every downstream stage,
every design choice is biased toward **over-inclusion**:

- Signals are fused with logical OR (any one cheap signal marking a moment
  "live" is enough to keep it), never AND.
- Each signal's own thresholds are deliberately loose (see the constants
  below), and callers are expected to validate the fused result against
  reviewed contact/bounce timestamps before trusting a tighter setting.
- Padding (default +/-0.5s, matching ``BALL_TRACKING_PIPELINE.md`` 4.2/5.1) is
  applied *after* fusing all signals, so a slightly early/late signal onset
  still buys a full half-second of margin around the true rally boundary.

Inputs consumed (all already produced by earlier, cheap pipeline stages --
this module does no model inference of its own):

- ``ball_track.json`` (schema ``BallTrack``): ball visibility over time is the
  strongest single "something is happening" signal.
- ``tracks.json`` (schema ``Tracks``): per-player bbox/world_xy motion energy
  catches rallies where the ball track is temporarily lost (occlusion, blur)
  but players are visibly moving to the ball.
- ``audio_onsets.json`` (schema from ``audio_onsets.py`` / ``audio_onsets_v2.py``):
  optional -- many clips have no audio stream (``onsets: []`` /
  ``blocked``); when present, onset timestamps are strong point-in-time
  evidence of contact/bounce sound and are folded in as another OR term.

Output: a plain JSON-serializable dict (``artifact_type ==
"racketsport_rally_spans"``) with padded, merged ``[t0, t1]`` spans plus
per-span provenance and clip-level dead-time statistics. A companion
``frame_schedule`` / ``in_rally_span`` helper lets any stage (BALL, BODY,
TRK, association, world build) filter frames without re-deriving spans.

Nothing in this module reads or writes orchestrator/pipeline_cli state; it is
a standalone importable library plus a CLI (see
``scripts/racketsport/build_rally_spans.py``) that any stage can adopt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_rally_spans"

# --- Fusion / padding defaults -------------------------------------------------
#
# Defaults are deliberately loose (biased toward keeping frames, not dropping
# them). BALL_TRACKING_PIPELINE.md 4.2 defines the on-device rally on/off rule
# this mirrors: "rally = ball visible & moving for >= 5 consecutive frames; end
# after >= 0.8s with no valid ball ... Emit rally_spans[] with +/-0.5s padding."

DEFAULT_PAD_SECONDS = 0.5
"""Padding applied to each fused active interval before merging (BALL_TRACKING_PIPELINE.md 5.1)."""

DEFAULT_BALL_GAP_SECONDS = 0.8
"""Max gap between ball-visible timestamps that still counts as one continuous interval."""

DEFAULT_PLAYER_GAP_SECONDS = 0.8
"""Max gap between above-threshold player-motion samples that still counts as continuous."""

DEFAULT_AUDIO_MERGE_GAP_SECONDS = 0.4
"""Max gap between audio onsets that still counts as one continuous interval."""

DEFAULT_MERGE_GAP_SECONDS = 1.0
"""After padding, intervals separated by less than this are merged into one span."""

DEFAULT_PLAYER_SPEED_THRESHOLD_M_S = 0.35
"""World-space (meters/second) player centroid speed above which a player counts as "moving".

Loose on purpose: a shuffle-step while waiting between points can be ~0.2-0.3 m/s;
0.35 m/s sits just above normal idle sway so it does not manufacture spans out of
standing-still noise, while still firing well before a real split-step/approach.
"""

DEFAULT_PLAYER_SPEED_THRESHOLD_PX_S = 40.0
"""Fallback bbox-center speed (px/s @ ~1080p) used only when world_xy is unavailable."""

DEFAULT_AUDIO_ONSET_WINDOW_SECONDS = 0.25
"""Half-window placed around each audio onset timestamp before merging with other signals."""


@dataclass(frozen=True)
class _Interval:
    t0: float
    t1: float
    source: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sorted_unique(values: Iterable[float]) -> list[float]:
    return sorted(set(round(float(v), 6) for v in values))


def _points_to_intervals(points: Sequence[float], *, gap_seconds: float, source: str) -> list[_Interval]:
    """Collapse a sorted set of "active" timestamps into merged [t0, t1] intervals.

    Two consecutive points are joined into the same interval when their gap is
    <= gap_seconds; a lone point becomes a zero-width interval (padding later
    gives it real width).
    """

    if not points:
        return []
    pts = sorted(points)
    intervals: list[_Interval] = []
    start = pts[0]
    prev = pts[0]
    for t in pts[1:]:
        if t - prev > gap_seconds:
            intervals.append(_Interval(t0=start, t1=prev, source=source))
            start = t
        prev = t
    intervals.append(_Interval(t0=start, t1=prev, source=source))
    return intervals


def ball_activity_intervals(
    ball_frames: Sequence[Mapping[str, Any]],
    *,
    gap_seconds: float = DEFAULT_BALL_GAP_SECONDS,
) -> list[_Interval]:
    """Raw intervals where the ball track reports ``visible`` (approx frames count too).

    Ball visibility (not "visible and moving fast") is used deliberately: slow
    dinks/resets near the kitchen are real rally content with low ball speed,
    and requiring a speed floor here would risk dropping exactly the shots the
    owner directive cares most about ("still an accurate representation").
    """

    visible_times = [float(f["t"]) for f in ball_frames if f.get("visible")]
    return _points_to_intervals(_sorted_unique(visible_times), gap_seconds=gap_seconds, source="ball")


def player_motion_intervals(
    players: Sequence[Mapping[str, Any]],
    *,
    speed_threshold_m_s: float = DEFAULT_PLAYER_SPEED_THRESHOLD_M_S,
    speed_threshold_px_s: float = DEFAULT_PLAYER_SPEED_THRESHOLD_PX_S,
    gap_seconds: float = DEFAULT_PLAYER_GAP_SECONDS,
) -> list[_Interval]:
    """Raw intervals where any tracked player's centroid speed exceeds the idle threshold.

    Falls back from world_xy (meters) to bbox-center pixels when a frame has
    no world projection yet (e.g. calibration not run), so this signal stays
    usable even upstream of the CAL stage.
    """

    active_times: list[float] = []
    for player in players:
        frames = sorted(player.get("frames", []), key=lambda f: float(f["t"]))
        prev = None
        for frame in frames:
            t = float(frame["t"])
            world_xy = frame.get("world_xy")
            bbox = frame.get("bbox")
            point = None
            unit = None
            if world_xy is not None:
                point = (float(world_xy[0]), float(world_xy[1]))
                unit = "m"
            elif bbox is not None:
                point = ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
                unit = "px"
            if point is None:
                prev = None
                continue
            if prev is not None:
                prev_t, prev_point, prev_unit = prev
                dt = t - prev_t
                if dt > 0 and prev_unit == unit:
                    dist = ((point[0] - prev_point[0]) ** 2 + (point[1] - prev_point[1]) ** 2) ** 0.5
                    speed = dist / dt
                    threshold = speed_threshold_m_s if unit == "m" else speed_threshold_px_s
                    if speed >= threshold:
                        active_times.append(prev_t)
                        active_times.append(t)
            prev = (t, point, unit)
    return _points_to_intervals(_sorted_unique(active_times), gap_seconds=gap_seconds, source="player_motion")


def audio_onset_intervals(
    onsets: Sequence[float],
    *,
    window_seconds: float = DEFAULT_AUDIO_ONSET_WINDOW_SECONDS,
    merge_gap_seconds: float = DEFAULT_AUDIO_MERGE_GAP_SECONDS,
) -> list[_Interval]:
    """Raw intervals of +/-window_seconds around each audio onset timestamp."""

    if not onsets:
        return []
    windows = [(_clamp(float(t) - window_seconds, 0.0, float("inf")), float(t) + window_seconds) for t in onsets]
    windows.sort()
    merged: list[_Interval] = []
    cur_t0, cur_t1 = windows[0]
    for t0, t1 in windows[1:]:
        if t0 - cur_t1 <= merge_gap_seconds:
            cur_t1 = max(cur_t1, t1)
        else:
            merged.append(_Interval(t0=cur_t0, t1=cur_t1, source="audio"))
            cur_t0, cur_t1 = t0, t1
    merged.append(_Interval(t0=cur_t0, t1=cur_t1, source="audio"))
    return merged


def merge_intervals(
    intervals: Sequence[_Interval],
    *,
    gap_seconds: float = 0.0,
    duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Merge (possibly overlapping/adjacent) intervals, tracking which sources contributed.

    Returns a list of dicts sorted by t0: ``{"t0": float, "t1": float, "sources": [str, ...]}``.
    """

    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: (iv.t0, iv.t1))
    merged: list[dict[str, Any]] = []
    cur_t0 = ordered[0].t0
    cur_t1 = ordered[0].t1
    cur_sources = {ordered[0].source}
    for iv in ordered[1:]:
        if iv.t0 - cur_t1 <= gap_seconds:
            cur_t1 = max(cur_t1, iv.t1)
            cur_sources.add(iv.source)
        else:
            merged.append({"t0": cur_t0, "t1": cur_t1, "sources": sorted(cur_sources)})
            cur_t0, cur_t1 = iv.t0, iv.t1
            cur_sources = {iv.source}
    merged.append({"t0": cur_t0, "t1": cur_t1, "sources": sorted(cur_sources)})
    if duration_s is not None:
        for span in merged:
            span["t0"] = _clamp(span["t0"], 0.0, duration_s)
            span["t1"] = _clamp(span["t1"], 0.0, duration_s)
    return merged


def derive_rally_spans(
    *,
    ball_frames: Sequence[Mapping[str, Any]] | None = None,
    players: Sequence[Mapping[str, Any]] | None = None,
    audio_onsets: Sequence[float] | None = None,
    duration_s: float,
    pad_seconds: float = DEFAULT_PAD_SECONDS,
    ball_gap_seconds: float = DEFAULT_BALL_GAP_SECONDS,
    player_gap_seconds: float = DEFAULT_PLAYER_GAP_SECONDS,
    player_speed_threshold_m_s: float = DEFAULT_PLAYER_SPEED_THRESHOLD_M_S,
    player_speed_threshold_px_s: float = DEFAULT_PLAYER_SPEED_THRESHOLD_PX_S,
    audio_onset_window_seconds: float = DEFAULT_AUDIO_ONSET_WINDOW_SECONDS,
    audio_merge_gap_seconds: float = DEFAULT_AUDIO_MERGE_GAP_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
) -> list[dict[str, Any]]:
    """Fuse cheap signals (OR) into padded, merged rally spans.

    Returns a list of ``{"t0": float, "t1": float, "sources": [str, ...]}`` dicts
    sorted by ``t0``, clamped to ``[0, duration_s]``. Empty input signals are
    skipped silently (e.g. no audio stream) rather than treated as "no rally".
    """

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    # Defensive: some source artifacts are computed from a longer/shorter clip
    # than the one being gated (e.g. a tracks.json re-used across a longer
    # capture window than the ball_track.json it is paired with). Timestamps
    # outside [0, duration_s] must not be allowed to fabricate coverage at the
    # clip boundary via clamping, so they are dropped before interval-building.
    ball_frames = [f for f in (ball_frames or []) if 0.0 <= float(f["t"]) <= duration_s]
    players = [
        {**player, "frames": [f for f in player.get("frames", []) if 0.0 <= float(f["t"]) <= duration_s]}
        for player in (players or [])
    ]
    audio_onsets = [t for t in (audio_onsets or []) if 0.0 <= float(t) <= duration_s]

    raw: list[_Interval] = []
    if ball_frames:
        raw.extend(ball_activity_intervals(ball_frames, gap_seconds=ball_gap_seconds))
    if players:
        raw.extend(
            player_motion_intervals(
                players,
                speed_threshold_m_s=player_speed_threshold_m_s,
                speed_threshold_px_s=player_speed_threshold_px_s,
                gap_seconds=player_gap_seconds,
            )
        )
    if audio_onsets:
        raw.extend(
            audio_onset_intervals(
                audio_onsets,
                window_seconds=audio_onset_window_seconds,
                merge_gap_seconds=audio_merge_gap_seconds,
            )
        )

    if not raw:
        return []

    # Pad each raw interval before merging so two nearby-but-separate bursts of
    # activity that pad-overlap become one span (matches the on-device
    # rally-boundary rule of padding, then treating a short silent gap as
    # still "in the rally").
    padded = [
        _Interval(
            t0=_clamp(iv.t0 - pad_seconds, 0.0, duration_s),
            t1=_clamp(iv.t1 + pad_seconds, 0.0, duration_s),
            source=iv.source,
        )
        for iv in raw
    ]
    return merge_intervals(padded, gap_seconds=merge_gap_seconds, duration_s=duration_s)


def dead_time_fraction(spans: Sequence[Mapping[str, Any]], duration_s: float) -> float:
    """Fraction of ``[0, duration_s]`` NOT covered by any span (0=no dead time, 1=all dead time)."""

    if duration_s <= 0:
        return 0.0
    covered = sum(max(0.0, float(s["t1"]) - float(s["t0"])) for s in spans)
    covered = min(covered, duration_s)
    return _clamp(1.0 - covered / duration_s, 0.0, 1.0)


def in_rally_span(t: float, spans: Sequence[Mapping[str, Any]]) -> bool:
    """True if timestamp ``t`` (seconds) falls inside any span (inclusive bounds)."""

    return any(float(s["t0"]) <= t <= float(s["t1"]) for s in spans)


def frame_schedule(
    spans: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    frame_count: int,
) -> list[int]:
    """0-indexed frame numbers inside any span, for a clip decoded at ``fps`` with
    ``frame_count`` total frames.

    This is the "frame-schedule filter any stage can consume": BALL, BODY, TRK,
    association, and world-build stages can all intersect their own native
    frame loop against ``set(frame_schedule(...))`` (or reuse
    ``in_rally_span`` directly on a timestamp) instead of re-deriving rally
    spans themselves.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if not spans:
        return []
    scheduled: list[int] = []
    for idx in range(frame_count):
        t = idx / fps
        if in_rally_span(t, spans):
            scheduled.append(idx)
    return scheduled


def missed_events(events_s: Sequence[float], spans: Sequence[Mapping[str, Any]]) -> list[float]:
    """Reviewed event timestamps (seconds) that fall OUTSIDE every span.

    Intended for validation, not production filtering: any non-empty result
    means the current signal/threshold/padding configuration is unsafe to
    ship and must be loosened before use.
    """

    return [t for t in events_s if not in_rally_span(t, spans)]


# --- Artifact I/O ---------------------------------------------------------------


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def ball_frames_from_ball_track(ball_track: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(ball_track.get("frames", []))


def players_from_tracks(tracks: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(tracks.get("players", []))


def onsets_from_audio_onsets(audio_onsets: Mapping[str, Any]) -> list[float]:
    onsets = audio_onsets.get("onsets", [])
    times: list[float] = []
    for onset in onsets:
        if isinstance(onset, Mapping):
            time_s = onset.get("time_s", onset.get("raw_time_s"))
            if time_s is not None:
                times.append(float(time_s))
        else:
            times.append(float(onset))
    return times


def build_rally_spans_artifact(
    *,
    clip_id: str,
    duration_s: float,
    ball_track: Mapping[str, Any] | None = None,
    tracks: Mapping[str, Any] | None = None,
    audio_onsets: Mapping[str, Any] | None = None,
    ball_track_path: str | None = None,
    tracks_path: str | None = None,
    audio_onsets_path: str | None = None,
    pad_seconds: float = DEFAULT_PAD_SECONDS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the full ``rally_spans.json`` payload (spans + provenance + stats).

    Accepts already-loaded artifact dicts (preferred, avoids re-reading large
    files when a caller already has them in memory) or leaves them ``None`` if
    that signal is unavailable for this clip.
    """

    ball_frames = ball_frames_from_ball_track(ball_track) if ball_track else None
    players = players_from_tracks(tracks) if tracks else None
    onsets = onsets_from_audio_onsets(audio_onsets) if audio_onsets else None

    spans = derive_rally_spans(
        ball_frames=ball_frames,
        players=players,
        audio_onsets=onsets,
        duration_s=duration_s,
        pad_seconds=pad_seconds,
        **kwargs,
    )

    signals_used = []
    if ball_frames:
        signals_used.append("ball_track")
    if players:
        signals_used.append("player_motion")
    if onsets:
        signals_used.append("audio_onsets")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": clip_id,
        "duration_s": duration_s,
        "pad_seconds": pad_seconds,
        "signals_used": signals_used,
        "signal_sources": {
            "ball_track_path": ball_track_path,
            "tracks_path": tracks_path,
            "audio_onsets_path": audio_onsets_path,
        },
        "spans": spans,
        "span_count": len(spans),
        "dead_time_fraction": dead_time_fraction(spans, duration_s),
        "not_ground_truth": True,
        "notes": [
            "Runtime optimization only, not an accuracy model.",
            "Signals fused with OR and biased toward over-inclusion; validate "
            "against reviewed contact/bounce timestamps before tightening thresholds.",
        ],
    }


def build_rally_spans_artifact_from_paths(
    *,
    clip_id: str,
    duration_s: float,
    ball_track_path: str | Path | None = None,
    tracks_path: str | Path | None = None,
    audio_onsets_path: str | Path | None = None,
    pad_seconds: float = DEFAULT_PAD_SECONDS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper: load artifacts from disk paths, then delegate to
    :func:`build_rally_spans_artifact`. Missing/None paths are skipped."""

    ball_track = load_json(ball_track_path) if ball_track_path else None
    tracks = load_json(tracks_path) if tracks_path else None
    audio_onsets = load_json(audio_onsets_path) if audio_onsets_path else None
    return build_rally_spans_artifact(
        clip_id=clip_id,
        duration_s=duration_s,
        ball_track=ball_track,
        tracks=tracks,
        audio_onsets=audio_onsets,
        ball_track_path=str(ball_track_path) if ball_track_path else None,
        tracks_path=str(tracks_path) if tracks_path else None,
        audio_onsets_path=str(audio_onsets_path) if audio_onsets_path else None,
        pad_seconds=pad_seconds,
        **kwargs,
    )
