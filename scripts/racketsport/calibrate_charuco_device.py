#!/usr/bin/env python3
"""Fit per-lens ChArUco intrinsics and persist them into the profile registry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.profile_registry import (  # noqa: E402
    DeviceProfile,
    SourceTrace,
    load_profile_registry,
    update_profile,
)
from threed.racketsport.schemas import CameraIntrinsics  # noqa: E402


@dataclass(frozen=True)
class CharucoObservation:
    corners: Any
    ids: Any
    image_size: tuple[int, int]
    video_index: int
    board_area_px: float


@dataclass(frozen=True)
class CharucoFit:
    rms: float
    camera_matrix: Any
    dist: Any
    observation_count: int


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", required=True, help="ChArUco sweep video; repeatable.")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--device-key", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--zoom", type=float, required=True)
    parser.add_argument("--profiles-root", type=Path, default=Path("runs/profiles"))
    parser.add_argument("--source-clip-id")
    parser.add_argument("--source-clip-ref")
    parser.add_argument("--max-frames-per-video", type=int, default=300)
    parser.add_argument("--min-observations", type=int, default=8)
    parser.add_argument("--min-subset-observations", type=int, default=6)
    parser.add_argument("--rms-threshold", type=float, default=1.0)
    parser.add_argument("--spread-threshold", type=float, default=0.20)
    args = parser.parse_args(argv)

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        return _refuse("opencv-python with aruco support is required", {"error": str(exc)})

    if not hasattr(cv2, "aruco"):
        return _refuse("opencv aruco module is required", {})

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, dictionary)
    observations = _collect_observations(
        cv2=cv2,
        videos=args.video,
        dictionary=dictionary,
        board=board,
        max_frames_per_video=args.max_frames_per_video,
        min_corners=8,
    )
    if len(observations) < args.min_observations:
        return _refuse(
            "insufficient ChArUco observations",
            {"observations": len(observations), "min_observations": args.min_observations},
        )

    try:
        fit = _fit_charuco(cv2=cv2, board=board, observations=observations)
    except ValueError as exc:
        return _refuse(str(exc), {"observations": len(observations)})
    if fit.rms > args.rms_threshold:
        return _refuse(
            "RMS reprojection exceeds acceptance threshold; refusing to persist",
            {"rms_reprojection_px": fit.rms, "rms_threshold": args.rms_threshold},
        )

    subset_fits = _fit_subsets(
        cv2=cv2,
        board=board,
        observations=observations,
        video_count=len(args.video),
        min_subset_observations=args.min_subset_observations,
    )
    if len(subset_fits) < 3:
        return _refuse(
            "fewer than three distance subsets had enough ChArUco observations; refusing to persist",
            {"subset_count": len(subset_fits), "min_subset_observations": args.min_subset_observations},
        )
    spread = _k1_k2_spread(np, subset_fits)
    if spread["k1_relative_spread"] >= args.spread_threshold or spread["k2_relative_spread"] >= args.spread_threshold:
        return _refuse(
            "k1/k2 spread across distance subsets exceeds acceptance threshold; refusing to persist",
            {"spread": spread, "spread_threshold": args.spread_threshold},
        )

    intrinsics = _intrinsics_from_fit(fit)
    trace = SourceTrace(
        source_clip_id=args.source_clip_id or args.video[0].stem,
        source_clip_ref=args.source_clip_ref or str(args.video[0]),
        source_profile_id=None,
    )
    profile = _upsert_device_intrinsics(
        account_id=args.account_id,
        device_key=args.device_key,
        profile_id=args.profile_id,
        display_name=args.display_name,
        lens=args.lens,
        zoom=args.zoom,
        intrinsics=intrinsics,
        source_trace=trace,
        profiles_root=args.profiles_root,
    )
    summary = {
        "status": "persisted",
        "profile_id": profile.profile_id,
        "device_key": profile.device_key,
        "lens": args.lens,
        "zoom": args.zoom,
        "rms_reprojection_px": fit.rms,
        "observation_count": fit.observation_count,
        "subset_count": len(subset_fits),
        "spread": spread,
        "intrinsics": intrinsics.model_dump(mode="json"),
        "profiles_root": str(args.profiles_root),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _collect_observations(
    *,
    cv2: Any,
    videos: Sequence[Path],
    dictionary: Any,
    board: Any,
    max_frames_per_video: int,
    min_corners: int,
) -> list[CharucoObservation]:
    observations: list[CharucoObservation] = []
    for video_index, video in enumerate(videos):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {video}")
        frames_read = 0
        while frames_read < max_frames_per_video:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary)
            if ids is None or len(ids) == 0:
                continue
            count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if not count or count < min_corners or charuco_corners is None or charuco_ids is None:
                continue
            hull = cv2.convexHull(charuco_corners.reshape(-1, 2).astype("float32"))
            observations.append(
                CharucoObservation(
                    corners=charuco_corners,
                    ids=charuco_ids,
                    image_size=(int(gray.shape[1]), int(gray.shape[0])),
                    video_index=video_index,
                    board_area_px=float(cv2.contourArea(hull)),
                )
            )
        capture.release()
    return observations


def _fit_charuco(*, cv2: Any, board: Any, observations: Sequence[CharucoObservation]) -> CharucoFit:
    if not observations:
        raise ValueError("no ChArUco observations to fit")
    image_size = observations[0].image_size
    if any(observation.image_size != image_size for observation in observations):
        raise ValueError("all ChArUco videos must have the same frame size")
    rms, camera_matrix, dist, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
        [observation.corners for observation in observations],
        [observation.ids for observation in observations],
        board,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
    )
    return CharucoFit(
        rms=float(rms),
        camera_matrix=camera_matrix,
        dist=dist,
        observation_count=len(observations),
    )


def _fit_subsets(
    *,
    cv2: Any,
    board: Any,
    observations: Sequence[CharucoObservation],
    video_count: int,
    min_subset_observations: int,
) -> list[CharucoFit]:
    groups: list[list[CharucoObservation]] = []
    if video_count >= 3:
        for video_index in range(video_count):
            group = [observation for observation in observations if observation.video_index == video_index]
            if len(group) >= min_subset_observations:
                groups.append(group)
    else:
        ordered = sorted(observations, key=lambda observation: observation.board_area_px)
        for index in range(3):
            start = round(index * len(ordered) / 3)
            end = round((index + 1) * len(ordered) / 3)
            group = ordered[start:end]
            if len(group) >= min_subset_observations:
                groups.append(group)
    return [_fit_charuco(cv2=cv2, board=board, observations=group) for group in groups]


def _k1_k2_spread(np: Any, fits: Sequence[CharucoFit]) -> dict[str, float]:
    values = np.asarray([fit.dist.reshape(-1)[:2] for fit in fits], dtype=float)
    means = np.mean(values, axis=0)
    relative = (np.max(values, axis=0) - np.min(values, axis=0)) / np.maximum(np.abs(means), 1e-9)
    return {
        "k1_relative_spread": float(relative[0]),
        "k2_relative_spread": float(relative[1]),
        "k1_values": [float(value) for value in values[:, 0]],
        "k2_values": [float(value) for value in values[:, 1]],
    }


def _intrinsics_from_fit(fit: CharucoFit) -> CameraIntrinsics:
    camera_matrix = fit.camera_matrix
    dist = fit.dist.reshape(-1).tolist()
    return CameraIntrinsics(
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        dist=[float(value) for value in dist[:4]],
        source="charuco_sweep",
    )


def _upsert_device_intrinsics(
    *,
    account_id: str,
    device_key: str,
    profile_id: str,
    display_name: str,
    lens: str,
    zoom: float,
    intrinsics: CameraIntrinsics,
    source_trace: SourceTrace,
    profiles_root: Path,
) -> DeviceProfile:
    profile = _existing_device_profile(
        account_id=account_id,
        device_key=device_key,
        profile_id=profile_id,
        profiles_root=profiles_root,
    )
    entry = {
        "lens": lens,
        "zoom": zoom,
        "intrinsics": intrinsics.model_dump(mode="json"),
        "source_trace": source_trace.model_dump(mode="json"),
    }
    if profile is None:
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_device_profile",
            "account_id": account_id,
            "profile_id": profile_id,
            "display_name": display_name,
            "version": 1,
            "source_trace": source_trace.model_dump(mode="json"),
            "retention": _default_retention(),
            "device_key": device_key,
            "intrinsics_by_lens_zoom": [entry],
            "exposure_constant": 1.0,
        }
    else:
        payload = profile.model_dump(mode="json")
        payload["display_name"] = display_name
        payload["device_key"] = device_key
        replaced = False
        entries = []
        for existing in payload["intrinsics_by_lens_zoom"]:
            if str(existing["lens"]).lower() == lens.lower() and abs(float(existing["zoom"]) - zoom) <= 1e-6:
                entries.append(entry)
                replaced = True
            else:
                entries.append(existing)
        if not replaced:
            entries.append(entry)
        payload["intrinsics_by_lens_zoom"] = entries
    updated = DeviceProfile.model_validate(payload)
    registry = update_profile(account_id, updated, profiles_root=profiles_root)
    return registry.device_profiles[updated.profile_id]


def _existing_device_profile(
    *,
    account_id: str,
    device_key: str,
    profile_id: str,
    profiles_root: Path,
) -> DeviceProfile | None:
    try:
        registry = load_profile_registry(account_id, profiles_root=profiles_root)
    except FileNotFoundError:
        return None
    if profile_id in registry.device_profiles:
        return registry.device_profiles[profile_id]
    for profile in registry.device_profiles.values():
        if profile.device_key == device_key:
            return profile
    return None


def _default_retention() -> dict[str, Any]:
    return {
        "scope": "account_lifetime",
        "delete_with_source_clip": True,
        "delete_with_source_profile": True,
        "retention_days": None,
        "legal_basis": "owner_setup",
    }


def _refuse(reason: str, details: dict[str, Any]) -> int:
    print(json.dumps({"status": "refused", "reason": reason, **details}, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
