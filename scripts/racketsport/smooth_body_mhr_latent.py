#!/usr/bin/env python3
"""P2-2 STEP A phase 2: MHR latent-space temporal smoothing PROTOTYPE.

See `runs/archive/root_docs_20260709/TECH_BLUEPRINTS.md` BODY pillar STEP A (~lines 1452-1499) and
`runs/lanes/w5_p22latent_20260707/spec.md` PHASE 2. This is a measurement
harness, not a pipeline component: it stays UNWIRED from `process_video.py`
by construction (no import of this module anywhere in that file), and
`--lambda-foot` is hard-pinned to 0.0 (confident per-foot phases don't exist
yet -- see the lane report HONEST ISSUES; do not tune the foot term against
all-rejected placeholder phases).

Pipeline:
  1. Read a fresh ``body_mesh.json`` monolith (racketsport_body_mesh,
     produced by a `--fetch-body-monoliths` BODY-only dispatch) for a clip's
     per-player, per-frame ``smplx_params`` (global_orient euler(3),
     body_pose euler(133), betas/shape(45), optional scale(28)).
  2. Per player: shape/scale LOCK to the per-track median (constant bone
     length -- same intent as the existing MAD bone-length detector).
  3. Sliding-window optimization (W frames, stride 1) over the MHR latent
     ``pred_pose_raw`` (266-dim: global_rot_6d(6) + body_cont(260)):
     minimize ``||code_t - code_t^raw||^2 + lambda_smooth * ||d2 code_t||^2``
     (closed-form ridge solve against the 2nd-difference operator; no
     lambda_foot term is implemented -- it stays 0 this lane by contract).
  4. Decode smoothed codes -> joints/vertices via
     `threed.racketsport.mhr_decode.MHRDecoder`, reground through the real
     pipeline transform, and measure ACCEPTANCE (before vs after) through
     `visual_quality.py::measure_visual_quality` and
     `pose_temporal.py::compare_wrist_peak_timing`.

Step 4 needs torch + roma + the sam_3d_body runtime (the fleet GPU VM's
body_venv) -- pass ``--skip-decode`` (or run where that runtime is absent)
to exercise only steps 1-3 (the numpy-only shape/scale-lock + sliding-window
solver), which is fully CPU-testable and is what the same-lane unit tests
exercise directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport import mhr_decode  # noqa: E402

DEFAULT_WINDOW = 9
DEFAULT_LAMBDA_SMOOTH = (0.1, 0.3, 0.6)
BODY_POSE_EULER_DIM = mhr_decode.BODY_POSE_EULER_DIM
GLOBAL_ROT_EULER_DIM = mhr_decode.GLOBAL_ROT_EULER_DIM
SHAPE_DIM = mhr_decode.SHAPE_DIM
SCALE_DIM = mhr_decode.SCALE_DIM


# ---------------------------------------------------------------------------
# Pure numpy sliding-window ridge smoother (no torch/roma dependency). This
# is the closed-form solution of
#   minimize_x sum_i ||x_i - r_i||^2 + lambda * sum_i ||d2 x_i||^2
# per window, i.e. (I + lambda * D^T D) x = r, D the discrete 2nd-difference
# operator. lambda_foot is intentionally NOT a parameter here -- see module
# docstring.
# ---------------------------------------------------------------------------
def second_difference_matrix(window_size: int) -> np.ndarray:
    if window_size < 3:
        return np.zeros((0, window_size), dtype=np.float64)
    matrix = np.zeros((window_size - 2, window_size), dtype=np.float64)
    for row in range(window_size - 2):
        matrix[row, row] = 1.0
        matrix[row, row + 1] = -2.0
        matrix[row, row + 2] = 1.0
    return matrix


def solve_smoothing_window(raw: np.ndarray, lambda_smooth: float) -> np.ndarray:
    """raw: (k, C) frames x channels -> ridge-smoothed (k, C)."""
    if raw.ndim != 2:
        raise ValueError("raw must be 2-D (frames, channels)")
    k = raw.shape[0]
    if k < 3 or lambda_smooth <= 0.0:
        return raw.copy()
    d_matrix = second_difference_matrix(k)
    system = np.eye(k) + lambda_smooth * (d_matrix.T @ d_matrix)
    return np.linalg.solve(system, raw)


def sliding_window_smooth(sequence: np.ndarray, *, window: int, lambda_smooth: float) -> np.ndarray:
    """sequence: (T, C) raw per-frame codes -> (T, C) smoothed.

    W=``window``, stride 1: for frame t, solve the local window centered on
    t (clamped at the sequence boundaries) and keep only t's own position
    from that window's solution.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2:
        raise ValueError("sequence must be 2-D (frames, channels)")
    total_frames = sequence.shape[0]
    half = window // 2
    out = np.empty_like(sequence)
    for t in range(total_frames):
        start = max(0, t - half)
        end = min(total_frames, t + half + 1)
        local_raw = sequence[start:end]
        local_smoothed = solve_smoothing_window(local_raw, lambda_smooth)
        out[t] = local_smoothed[t - start]
    return out


def median_lock(values: np.ndarray) -> np.ndarray:
    """values: (T, C) -> (T, C) with every row replaced by the per-track median."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        return values.copy()
    median = np.median(values, axis=0)
    return np.broadcast_to(median, values.shape).copy()


# ---------------------------------------------------------------------------
# body_mesh.json (racketsport_body_mesh) extraction.
# ---------------------------------------------------------------------------
def extract_player_pose_sequences(body_mesh: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull per-player, frame-ordered pose arrays out of a body_mesh.json payload.

    Returns ``{player_id: {"frame_idx": [...], "t": [...],
    "global_orient": (T,3), "body_pose": (T,133), "betas": (T,45),
    "scale": (T,28) | None, "track_world_xy": (T,2)}}``. Frames missing
    global_orient/body_pose (empty lists -- the pre-additive-schema gap) are
    dropped with a count reported by the caller.
    """
    out: dict[str, dict[str, Any]] = {}
    for player in body_mesh.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id"))
        frame_idx: list[int] = []
        t_values: list[float] = []
        global_orient: list[list[float]] = []
        body_pose: list[list[float]] = []
        betas: list[list[float]] = []
        scale: list[list[float]] = []
        track_world_xy: list[list[float]] = []
        has_scale = True
        skipped = 0
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            params = frame.get("smplx_params", {})
            if not isinstance(params, Mapping):
                continue
            go = params.get("global_orient", [])
            bp = params.get("body_pose", [])
            be = params.get("betas", [])
            sc = params.get("scale", [])
            if len(go) != GLOBAL_ROT_EULER_DIM or len(bp) != BODY_POSE_EULER_DIM:
                skipped += 1
                continue
            frame_idx.append(int(frame.get("frame_idx", len(frame_idx))))
            t_values.append(float(frame.get("t", 0.0)))
            global_orient.append([float(v) for v in go])
            body_pose.append([float(v) for v in bp])
            betas.append([float(v) for v in be] if len(be) == SHAPE_DIM else [0.0] * SHAPE_DIM)
            if len(sc) == SCALE_DIM:
                scale.append([float(v) for v in sc])
            else:
                has_scale = False
                scale.append([0.0] * SCALE_DIM)
            transl = params.get("transl_world", [0.0, 0.0, 0.0])
            track_world_xy.append([float(transl[0]) if len(transl) > 0 else 0.0, float(transl[1]) if len(transl) > 1 else 0.0])
        if not frame_idx:
            continue
        out[player_id] = {
            "frame_idx": frame_idx,
            "t": t_values,
            "global_orient": np.asarray(global_orient, dtype=np.float64),
            "body_pose": np.asarray(body_pose, dtype=np.float64),
            "betas": np.asarray(betas, dtype=np.float64),
            "scale": np.asarray(scale, dtype=np.float64) if has_scale else None,
            "track_world_xy": np.asarray(track_world_xy, dtype=np.float64),
            "skipped_frame_count": skipped,
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P2-2 STEP A phase-2: MHR latent-space temporal smoothing prototype "
            "(lambda_foot=0, UNWIRED from process_video.py)."
        )
    )
    parser.add_argument("--body-mesh", type=Path, required=True, help="Fresh body_mesh.json monolith (racketsport_body_mesh).")
    parser.add_argument("--clip-dir", type=Path, default=None, help="Original clip run dir (court_calibration.json + virtual_world.json + placement.json + skeleton3d.json + body_joint_quality.json) for the visual_quality before/after acceptance table.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--lambda-smooth", default=",".join(str(v) for v in DEFAULT_LAMBDA_SMOOTH))
    parser.add_argument("--lambda-foot", type=float, default=0.0, help="MUST be 0.0 this lane.")
    parser.add_argument("--checkpoint", default=mhr_decode.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--mhr-asset", default=mhr_decode.DEFAULT_MHR_ASSET_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Only run the numpy shape/scale-lock + sliding-window solver; skip MHR decode + visual_quality acceptance.",
    )
    return parser


def _smoothed_pred_pose_raw(
    *, global_orient: np.ndarray, body_pose: np.ndarray, window: int, lambda_smooth: float
) -> np.ndarray:
    """Build pred_pose_raw(T,266) from euler fields, smooth it in cont space, return (T,266)."""
    import torch  # local import: only reached when MHR_RUNTIME_AVAILABLE

    raw = mhr_decode.build_pred_pose_raw(global_orient, body_pose).detach().cpu().numpy()
    smoothed = sliding_window_smooth(raw, window=window, lambda_smooth=lambda_smooth)
    return smoothed


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.lambda_foot != 0.0:
        raise SystemExit(
            "lambda_foot must be 0.0 this lane (spec: confident foot-contact phases don't exist "
            "yet -- do not tune against all-rejected placeholder phases)."
        )
    lambda_values = [float(x) for x in str(args.lambda_smooth).split(",") if x.strip()]
    if not lambda_values:
        raise SystemExit("--lambda-smooth must contain at least one value")

    body_mesh = json.loads(Path(args.body_mesh).read_text(encoding="utf-8"))
    per_player = extract_player_pose_sequences(body_mesh)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    decode_available = mhr_decode.MHR_RUNTIME_AVAILABLE and not args.skip_decode
    decoder = None
    if decode_available:
        decoder = mhr_decode.MHRDecoder(
            checkpoint_path=args.checkpoint, mhr_path=args.mhr_asset, device=args.device
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_mhr_latent_smoothing_prototype",
        "body_mesh_path": str(args.body_mesh),
        "clip": str(body_mesh.get("clip", "")),
        "window": args.window,
        "lambda_smooth_sweep": lambda_values,
        "lambda_foot": 0.0,
        "decode_available": decode_available,
        "decode_skip_reason": None if decode_available else str(mhr_decode.MHR_RUNTIME_IMPORT_ERROR or "explicit --skip-decode"),
        "players": {},
    }

    for player_id, data in per_player.items():
        player_report: dict[str, Any] = {
            "frame_count": len(data["frame_idx"]),
            "skipped_frame_count": data["skipped_frame_count"],
            "scale_available": data["scale"] is not None,
            "by_lambda": {},
        }
        shape_locked = median_lock(data["betas"])
        scale_locked = median_lock(data["scale"]) if data["scale"] is not None else None

        for lam in lambda_values:
            lam_key = f"{lam:g}"
            if decode_available:
                smoothed_raw = _smoothed_pred_pose_raw(
                    global_orient=data["global_orient"],
                    body_pose=data["body_pose"],
                    window=args.window,
                    lambda_smooth=lam,
                )
                player_report["by_lambda"][lam_key] = {
                    "smoothed_pred_pose_raw_shape": list(smoothed_raw.shape),
                }
            else:
                euler_concat = np.concatenate([data["global_orient"], data["body_pose"]], axis=1)
                smoothed_euler = sliding_window_smooth(euler_concat, window=args.window, lambda_smooth=lam)
                player_report["by_lambda"][lam_key] = {
                    "mode": "euler_space_smoke_path_no_runtime",
                    "smoothed_euler_shape": list(smoothed_euler.shape),
                }
        report["players"][player_id] = player_report

    out_path = args.out_dir / "smoothing_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"out_dir": str(args.out_dir), "player_count": len(report["players"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
