#!/usr/bin/env python3
"""Measure lens distortion from the outdoor clip's own pixels. Read-only, CPU only.

The discriminating test. A straight world line images as a straight image line
under ANY rectilinear pinhole, whatever the pose or the focal length. Curvature
is distortion and nothing else -- so this involves no ball, no court template,
and none of the calibration correspondences, and it therefore cannot be
confounded by the ball genuinely travelling further out while also travelling
towards the frame edge.

Method: median-stack the clip's frames (the broadcast camera is static to
sub-pixel), track long structures to sub-pixel accuracy by parabolic
interpolation of the directional gradient, fit a total-least-squares line, and
scan the radial coefficient that minimises the residual.

Sign convention: the scan applies ``p -> C + (p - C)(1 + k * r^2)`` to the
OBSERVED pixels, which to first order recovers ``-k1`` in the Brown model where
``x_d = x_u (1 + k1 r_u^2)``. The script asserts this against synthetic lines
distorted by a known Brown k1 before reporting anything.

The composited broadcast scoreboard is the control: it is drawn after the lens,
so it must come back perfectly straight with k ~ 0. If it does not, the
measurement is biased and the rest is noise.

    python3 runs/lanes/farfield_extrapolation_20260727/plumbline_distortion_test.py \
        --frames <dir of NNNNNN.jpg> \
        --out runs/lanes/farfield_extrapolation_20260727/plumbline.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CX, CY = 960.0, 540.0
FOCAL_PX = 2537.9139649536
WIDTH, HEIGHT = 1920.0, 1080.0
HALF_DIAGONAL = math.hypot(WIDTH / 2.0, HEIGHT / 2.0)


def median_background(frames_dir: Path, step: int = 10) -> np.ndarray:
    paths = sorted(frames_dir.glob("*.jpg"))[::step]
    images = [cv2.imread(str(path), 0) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        raise SystemExit(f"no frames under {frames_dir}")
    return np.median(np.stack(images, 0), 0).astype(np.uint8)


class Tracker:
    def __init__(self, background: np.ndarray) -> None:
        smooth = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 1.0)
        self.gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        self.gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

    def _sample(self, image: np.ndarray, x: float, y: float) -> float:
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        if x0 < 0 or y0 < 0 or x0 + 1 >= image.shape[1] or y0 + 1 >= image.shape[0]:
            return float("nan")
        ax, ay = x - x0, y - y0
        return float(
            (1 - ax) * (1 - ay) * image[y0, x0]
            + ax * (1 - ay) * image[y0, x0 + 1]
            + (1 - ax) * ay * image[y0 + 1, x0]
            + ax * ay * image[y0 + 1, x0 + 1]
        )

    def track_horizontal(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        *,
        polarity: int = 1,
        halfwidth: float = 3.0,
        min_peak: float = 25.0,
        step: float = 0.2,
        dx: float = 2.0,
    ) -> np.ndarray:
        """Follow a near-horizontal edge, one sub-pixel y per column."""

        points: list[tuple[float, float]] = []
        y = float(y_start)
        columns = (
            np.arange(x_start, x_end + 1e-9, dx)
            if x_end > x_start
            else np.arange(x_start, x_end - 1e-9, -dx)
        )
        offsets = np.arange(-halfwidth, halfwidth + 1e-9, step)
        for x in columns:
            values = np.array(
                [polarity * self._sample(self.gy, x, y + offset) for offset in offsets],
                float,
            )
            if np.isnan(values).any():
                break
            peak = int(np.argmax(values))
            if peak == 0 or peak == len(values) - 1 or values[peak] < min_peak:
                break
            curvature = values[peak - 1] - 2 * values[peak] + values[peak + 1]
            shift = (
                0.0
                if abs(curvature) < 1e-9
                else 0.5 * (values[peak - 1] - values[peak + 1]) / curvature
            )
            if abs(shift) > 1.0:
                break
            y = y + offsets[peak] + shift * step
            points.append((float(x), y))
        return np.array(points)


def tls_residuals(points: np.ndarray, sigma_cut: float = 2.5, iterations: int = 4):
    keep = np.ones(len(points), bool)
    for _ in range(iterations):
        centred = points[keep] - points[keep].mean(0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        offset = float(normal @ points[keep].mean(0))
        residual = points @ normal - offset
        spread = float(np.std(residual[keep]))
        if spread < 1e-9:
            break
        updated = np.abs(residual) < sigma_cut * spread
        if updated.sum() < 8 or bool((updated == keep).all()):
            keep = updated
            break
        keep = updated
    centred = points[keep] - points[keep].mean(0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    offset = float(normal @ points[keep].mean(0))
    return keep, points[keep] @ normal - offset


def apply_radial(points: np.ndarray, k: float) -> np.ndarray:
    xn = (points[:, 0] - CX) / FOCAL_PX
    yn = (points[:, 1] - CY) / FOCAL_PX
    scale = 1.0 + k * (xn * xn + yn * yn)
    return np.stack([CX + xn * scale * FOCAL_PX, CY + yn * scale * FOCAL_PX], 1)


def scan_k(points: np.ndarray, lo: float = -0.4, hi: float = 0.4, step: float = 0.002):
    best = None
    for k in np.arange(lo, hi + 1e-12, step):
        _, residual = tls_residuals(apply_radial(points, k))
        score = float(np.sqrt((residual**2).mean()))
        if best is None or score < best[1]:
            best = (float(k), score)
    _, zero = tls_residuals(apply_radial(points, 0.0))
    return best[0], best[1], float(np.sqrt((zero**2).mean()))


def radius_pct(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0] - CX, points[:, 1] - CY) / HALF_DIAGONAL * 100.0


def check_sign_convention() -> dict[str, float]:
    """Fail loudly if the scan's sign is not the negative of Brown's k1."""

    recovered: dict[str, float] = {}
    for brown_k1 in (-0.10, 0.10):
        xs = np.linspace(200.0, 1790.0, 300)
        xn = (xs - CX) / FOCAL_PX
        yn = (50.0 - CY) / FOCAL_PX
        scale = 1.0 + brown_k1 * (xn * xn + yn * yn)
        observed = np.stack([CX + xn * scale * FOCAL_PX, CY + yn * scale * FOCAL_PX], 1)
        k, _, _ = scan_k(observed, step=0.001)
        recovered[f"brown_k1={brown_k1:+.2f}"] = k
        if not math.isclose(k, -brown_k1, abs_tol=0.02):
            raise SystemExit(
                f"sign check failed: brown k1 {brown_k1:+.2f} recovered as {k:+.4f}"
            )
    return recovered


CANDIDATES = [
    # (name, seed_x, seed_y, x_left, x_right, polarity, what it is)
    ("wall_top_dark", 960, 53, 200, 1900, -1, "top edge of the sponsor wall (physical)"),
    ("wall_top_bright", 960, 48, 200, 1900, 1, "highlight strip above it (physical)"),
    ("wall_base", 960, 244, 300, 1900, 1, "wall meets the surround (physical)"),
    ("court_near_baseline", 960, 709, 430, 1500, 1, "painted court line (physical)"),
    ("court_far_baseline", 960, 322, 660, 1290, 1, "painted court line (physical)"),
    ("scoreboard_separator", 340, 961, 40, 630, 1, "CONTROL: composited overlay"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    background = median_background(Path(args.frames), step=args.step)
    tracker = Tracker(background)
    sign_check = check_sign_convention()

    results: list[dict[str, Any]] = []
    for name, seed_x, seed_y, x_left, x_right, polarity, description in CANDIDATES:
        halfwidth = 4.0 if "scoreboard" in name else 3.0
        min_peak = 15.0 if "scoreboard" in name else (20.0 if "court" in name else 25.0)
        left = tracker.track_horizontal(
            seed_x - 2, seed_y, x_left, polarity=polarity, halfwidth=halfwidth, min_peak=min_peak
        )
        right = tracker.track_horizontal(
            seed_x + 2, seed_y, x_right, polarity=polarity, halfwidth=halfwidth, min_peak=min_peak
        )
        parts = [part for part in (left[::-1] if len(left) else left, right) if len(part)]
        if not parts:
            results.append({"line": name, "tracked": False, "description": description})
            continue
        points = np.vstack(parts)
        if len(points) < 60:
            results.append(
                {"line": name, "tracked": False, "point_count": int(len(points)),
                 "description": description}
            )
            continue
        _, residual = tls_residuals(points)
        k, rms_k, rms_zero = scan_k(points)
        radii = radius_pct(points)
        results.append(
            {
                "line": name,
                "description": description,
                "tracked": True,
                "point_count": int(len(points)),
                "radius_pct_min": round(float(radii.min()), 2),
                "radius_pct_max": round(float(radii.max()), 2),
                "straight_line_rms_px": round(float(np.sqrt((residual**2).mean())), 4),
                "best_scan_k": round(k, 4),
                "implied_brown_k1": round(-k, 4),
                "rms_px_at_k_zero": round(rms_zero, 4),
                "rms_px_at_best_k": round(rms_k, 4),
            }
        )

    report = {
        "method": "plumb_line_straightness_v1",
        "sign_convention_check_recovered_scan_k": sign_check,
        "sign_convention": "scan k ~ -brown_k1 (asserted above against synthetic lines)",
        "focal_px_used_for_normalisation": FOCAL_PX,
        "lines": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    for row in results:
        if not row.get("tracked"):
            print(f"{row['line']:<22} NOT TRACKED")
            continue
        print(
            "%-22s n=%4d r%% %5.1f..%5.1f  straight_rms=%.3f px  brown k1=%+.4f "
            "(rms %.3f -> %.3f)  %s"
            % (
                row["line"],
                row["point_count"],
                row["radius_pct_min"],
                row["radius_pct_max"],
                row["straight_line_rms_px"],
                row["implied_brown_k1"],
                row["rms_px_at_k_zero"],
                row["rms_px_at_best_k"],
                row["description"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
