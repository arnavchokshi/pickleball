#!/usr/bin/env python3
"""Build review-only v2 audio-pop onset cue artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.audio_onsets_v2 import (  # noqa: E402
    DEFAULT_ADAPTIVE_WINDOW_S,
    DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DEFAULT_FRAME_SIZE_S,
    DEFAULT_HOP_S,
    DEFAULT_MIN_HFC_EVIDENCE,
    DEFAULT_MIN_POP_BAND_RATIO,
    DEFAULT_MIN_SEPARATION_S,
    DEFAULT_MIN_SPECTRAL_EVIDENCE,
    DEFAULT_THRESHOLD_MAD,
    build_audio_onsets_v2_from_video,
    build_audio_onsets_v2_from_wav,
    write_audio_onsets_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review-only v2 audio pop onsets from WAV or video.")
    parser.add_argument("--input", type=Path, required=True, help="Input WAV or video file.")
    parser.add_argument(
        "--frame-times", type=Path, required=True,
        help="PTS JSON bound to the same input media; its exact bytes are SHA-256 recorded.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output audio_onsets_v2.json path.")
    parser.add_argument("--clip", help="Optional clip id to store in the review artifact.")
    parser.add_argument("--frame-rate", type=float, help="Optional frame rate for nearest-frame onset metadata.")
    parser.add_argument("--analysis-sample-rate-hz", type=int, default=DEFAULT_ANALYSIS_SAMPLE_RATE_HZ)
    parser.add_argument("--bandpass-low-hz", type=float, default=DEFAULT_BANDPASS_LOW_HZ)
    parser.add_argument("--bandpass-high-hz", type=float, default=DEFAULT_BANDPASS_HIGH_HZ)
    parser.add_argument("--frame-size-s", type=float, default=DEFAULT_FRAME_SIZE_S)
    parser.add_argument("--hop-s", type=float, default=DEFAULT_HOP_S)
    parser.add_argument("--min-separation-s", type=float, default=DEFAULT_MIN_SEPARATION_S)
    parser.add_argument("--threshold-mad", type=float, default=DEFAULT_THRESHOLD_MAD)
    parser.add_argument("--adaptive-window-s", type=float, default=DEFAULT_ADAPTIVE_WINDOW_S)
    parser.add_argument("--min-pop-band-ratio", type=float, default=DEFAULT_MIN_POP_BAND_RATIO)
    parser.add_argument("--min-spectral-evidence", type=float, default=DEFAULT_MIN_SPECTRAL_EVIDENCE)
    parser.add_argument("--min-hfc-evidence", type=float, default=DEFAULT_MIN_HFC_EVIDENCE)
    parser.add_argument("--start-s", type=float, default=0.0, help="Start time inside the input media for video audio.")
    parser.add_argument("--duration-s", type=float, help="Optional video audio duration to analyze.")
    args = parser.parse_args()

    media_sha256 = _sha256_file(args.input)
    pts_source = _load_pts_source(args.frame_times, media_sha256=media_sha256)

    common = {
        "analysis_sample_rate_hz": args.analysis_sample_rate_hz,
        "bandpass_low_hz": args.bandpass_low_hz,
        "bandpass_high_hz": args.bandpass_high_hz,
        "frame_size_s": args.frame_size_s,
        "hop_s": args.hop_s,
        "min_separation_s": args.min_separation_s,
        "threshold_mad": args.threshold_mad,
        "adaptive_window_s": args.adaptive_window_s,
        "min_pop_band_ratio": args.min_pop_band_ratio,
        "min_spectral_evidence": args.min_spectral_evidence,
        "min_hfc_evidence": args.min_hfc_evidence,
        "clip": args.clip,
        "frame_rate": args.frame_rate,
        "media_sha256": media_sha256,
        "pts_source": pts_source,
    }
    if args.input.suffix.lower() == ".wav":
        payload = build_audio_onsets_v2_from_wav(args.input, **common)
    else:
        payload = build_audio_onsets_v2_from_video(
            args.input,
            start_s=args.start_s,
            duration_s=args.duration_s,
            **common,
        )
    write_audio_onsets_v2(args.out, payload)
    print(f"wrote {args.out} ({payload['status']}, onsets={payload['summary']['onset_count']})")
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pts_source(path: Path, *, media_sha256: str) -> dict[str, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("frame-times artifact must contain an object")
    declared_media_sha256 = payload.get(
        "source_video_sha256", payload.get("media_sha256")
    )
    if declared_media_sha256 != media_sha256:
        raise ValueError("frame-times artifact is not SHA-bound to the input media")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_video_sha256": media_sha256,
    }


if __name__ == "__main__":
    raise SystemExit(main())
