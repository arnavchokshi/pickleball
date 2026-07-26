"""Adapter mapping the TT3D table-tennis evaluation set onto DinkVision solver conventions.

TT3D: Gossard/Ziegler/Zell, Univ. Tuebingen, CVPR 2025 CVSports.
Source: https://github.com/cogsys-tuebingen/tt3d/tree/main/data/evaluation

LICENSE CONSTRAINT (BINDING): the upstream repo carries NO LICENSE file.
This data is INTERNAL VALIDATION ONLY. Never train on it, never ship it, never
admit it to the product data ledger as training data.

=== COORDINATE MAPPING ===

TT3D world frame (per dataset README):
    X along the width of the table, Y along the length of the table, Z upwards.
    Origin at the CENTRE OF THE TABLE SURFACE. Units: METRES.

TT3D camera model (per the *.yaml files: rvec, tvec, f, w, h):
    Standard OpenCV pinhole. R = Rodrigues(rvec) maps world -> camera:
        X_cam = R @ X_world + tvec
    Projection with a single focal length f and principal point at image centre:
        u = f * x_cam / z_cam + w/2
        v = f * y_cam / z_cam + h/2
    No distortion coefficients are supplied (synthetic observations).

DinkVision convention (threed/racketsport/ball_arc_solver.py::pixel_ray_world):
    camera_ray = ((u - cx)/fx, (v - cy)/fy, 1)
    origin     = R^T @ (-t)          # camera centre in world coords
    direction  = normalize(R^T @ camera_ray)
    This is the exact inverse of X_cam = R @ X_world + t, i.e. the SAME
    OpenCV convention. Z is up and `intersect_ray_z` intersects a horizontal
    plane, matching TT3D's Z-up frame.

=> The mapping is an IDENTITY mapping, not a change of basis:
       extrinsics.R = Rodrigues(rvec)   (world -> camera rotation)
       extrinsics.t = tvec              (metres, camera frame)
       intrinsics: fx = fy = f, cx = w/2, cy = h/2
   Both sides are metres and both are Z-up. There is no unit conversion and no
   axis permutation. This is verified numerically by `verify_mapping`, which
   projects the CSV's own GT 3D points and compares against the CSV's (u, v).

The one semantic difference: in pickleball Z=0 is the COURT floor; in TT3D Z=0
is the TABLE SURFACE. `intersect_ray_z(..., z=ball_radius)` therefore anchors a
bounce on the table rather than the floor. That is a change of INPUT (which
plane is "ground"), not a change of solver math.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import yaml

# --- Table-tennis ball constants (ITTF): 40 mm diameter, 2.7 g ---
TT_BALL_DIAMETER_M = 0.040
TT_BALL_RADIUS_M = TT_BALL_DIAMETER_M / 2.0
TT_BALL_MASS_KG = 0.0027
# Drag coefficient for a smooth sphere at table-tennis Reynolds numbers
# (Re ~ 1e4-1e5). Standard value used in the table-tennis trajectory
# literature, incl. the TT3D paper's own physical model.
TT_DRAG_CD = 0.40


def rodrigues(rvec: Sequence[float]) -> np.ndarray:
    """Rodrigues' rotation formula: axis-angle 3-vector -> 3x3 rotation matrix.

    Implemented directly (rather than via cv2.Rodrigues) so the adapter has no
    OpenCV dependency and so the convention is auditable in-place.
    """
    r = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


@dataclass(frozen=True)
class TT3DView:
    """A TT3D camera view, already mapped into DinkVision calibration form."""

    name: str
    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]
    f: float
    w: int
    h: int

    @property
    def R(self) -> np.ndarray:
        return rodrigues(self.rvec)

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.tvec, dtype=float).reshape(3)

    @property
    def camera_center_world(self) -> np.ndarray:
        """C = -R^T t."""
        return -(self.R.T @ self.t)

    def calibration(self) -> dict[str, Any]:
        """Emit a DinkVision `CameraCalibration`-shaped mapping.

        Consumed by ball_arc_solver's `_camera_arrays` / `_intrinsics`.
        """
        return {
            "intrinsics": {
                "fx": float(self.f),
                "fy": float(self.f),
                "cx": self.w / 2.0,
                "cy": self.h / 2.0,
            },
            "extrinsics": {
                "R": [[float(v) for v in row] for row in self.R],
                "t": [float(v) for v in self.t],
            },
            "image_size": {"width": int(self.w), "height": int(self.h)},
            "source": f"tt3d_{self.name}",
        }

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """World (N,3) metres -> pixels (N,2), OpenCV pinhole, no distortion."""
        pts = np.atleast_2d(np.asarray(xyz, dtype=float))
        cam = pts @ self.R.T + self.t  # X_cam = R @ X_world + t
        z = cam[:, 2]
        u = self.f * cam[:, 0] / z + self.w / 2.0
        v = self.f * cam[:, 1] / z + self.h / 2.0
        return np.stack([u, v], axis=1)

    def depth_of(self, xyz: np.ndarray) -> np.ndarray:
        """Camera-frame Z (depth along optical axis), metres."""
        pts = np.atleast_2d(np.asarray(xyz, dtype=float))
        return (pts @ self.R.T + self.t)[:, 2]

    def camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World-frame unit vectors of the camera axes (right, down, forward).

        Rows of R are the camera axes expressed in world coordinates, because
        X_cam = R @ X_world means the i-th camera axis is R[i, :].
        """
        R = self.R
        return R[0, :].copy(), R[1, :].copy(), R[2, :].copy()

    @classmethod
    def from_yaml(cls, path: str | Path, name: str | None = None) -> "TT3DView":
        p = Path(path)
        with p.open() as fh:
            raw = yaml.safe_load(fh)
        return cls(
            name=name or p.stem,
            rvec=tuple(float(x) for x in raw["rvec"]),
            tvec=tuple(float(x) for x in raw["tvec"]),
            f=float(raw["f"]),
            w=int(raw["w"]),
            h=int(raw["h"]),
        )


@dataclass(frozen=True)
class TT3DTrajectory:
    """One TT3D trajectory: timestamps, GT 3D metres, and 2D observations."""

    traj_id: str
    view: str
    t: np.ndarray          # (N,) seconds
    xyz: np.ndarray        # (N,3) metres, TT3D world frame
    uv: np.ndarray         # (N,2) pixels
    blur_l: np.ndarray     # (N,) simulated blur length
    blur_theta: np.ndarray # (N,) simulated blur angle

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def fps(self) -> float:
        if len(self) < 2:
            return 25.0
        dt = float(np.median(np.diff(self.t)))
        return 1.0 / dt if dt > 0 else 25.0


def load_trajectory(csv_path: str | Path, view: str) -> TT3DTrajectory:
    """Load a per-view TT3D CSV (Timestamp,X,Y,Z,u,v,l,theta)."""
    p = Path(csv_path)
    rows: list[dict[str, str]] = []
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError(f"empty trajectory csv: {p}")

    def col(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=float)

    return TT3DTrajectory(
        traj_id=p.stem,
        view=view,
        t=col("Timestamp"),
        xyz=np.stack([col("X"), col("Y"), col("Z")], axis=1),
        uv=np.stack([col("u"), col("v")], axis=1),
        blur_l=col("l") if "l" in rows[0] else np.zeros(len(rows)),
        blur_theta=col("theta") if "theta" in rows[0] else np.zeros(len(rows)),
    )


def iter_trajectories(root: str | Path, view: str) -> Iterator[TT3DTrajectory]:
    """Yield every trajectory for a view directory, ordered by id."""
    d = Path(root) / view
    for csv_path in sorted(d.glob("*.csv")):
        yield load_trajectory(csv_path, view)


def verify_mapping(
    view: TT3DView,
    trajectories: Sequence[TT3DTrajectory],
) -> dict[str, Any]:
    """THE GATE. Project each CSV's own GT 3D points and compare to its (u, v).

    If the mapping (units, frame, sign, principal point) is right this residual
    is ~0 for the *_no_noise views. A non-trivial residual means the mapping is
    wrong and every downstream number is meaningless.
    """
    residuals: list[np.ndarray] = []
    for traj in trajectories:
        pred = view.project(traj.xyz)
        residuals.append(np.linalg.norm(pred - traj.uv, axis=1))
    if not residuals:
        raise ValueError("no trajectories supplied to verify_mapping")
    allr = np.concatenate(residuals)
    return {
        "view": view.name,
        "n_trajectories": len(trajectories),
        "n_points": int(allr.size),
        "mean_px": float(np.mean(allr)),
        "median_px": float(np.median(allr)),
        "p95_px": float(np.percentile(allr, 95)),
        "max_px": float(np.max(allr)),
    }
