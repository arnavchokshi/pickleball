#!/usr/bin/env python3
"""Generate synthetic pickleball court keypoint labels from regulation geometry.

The output intentionally uses the same ``court_keypoints.json`` envelope consumed by
``train_court_keypoint_heatmap.py``, but every label carries synthetic provenance and uses
the dedicated ``synthetic`` item status accepted by the loader (CAL-R2 provenance fix,
2026-07-02: this used to be ``reviewed_static_camera_copy``, an enum workaround that let
synthetic rows silently inflate a count meant only for owner-approved REAL human-review copies
-- see ``SYNTHETIC_STATUS`` in ``train_court_keypoint_heatmap.py``). These images are training
augmentation only, never gate evidence.

CAL-SYNTH v2 (2026-07-05): rendering itself now lives in ``threed.racketsport.court_synth_scenes``
(shared with the zero-disk streaming API, ``threed.racketsport.court_synth_stream``), which adds
7 domain-randomized scenario families (dedicated-indoor/outdoor, tennis-overlay, adjacent-multi-
court, portrait-phone, harsh-shadow, portable-net/background-clutter). This module stays a thin,
CLI-backward-compatible disk corpus writer around that engine.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_synth_scenes import (
    SCENARIO_NAMES,
    RenderedScene,
    SceneRenderConfig,
    choose_scenario,
    render_synthetic_court_sample,
)
from threed.racketsport.court_templates import COORDINATE_FRAME, get_court_template


DEFAULT_OUTPUT_DIR = Path("runs/training_corpora_20260701/court_synthetic")
DEFAULT_SEED = 20260701
DEFAULT_IMAGE_SIZE = (640, 360)
DEFAULT_COUNT = 2000
DEFAULT_SPOT_CHECK_COUNT = 20
SYNTHETIC_ITEM_STATUS = "synthetic"
NET_KEYPOINT_HEIGHT_CONVENTION = "regulation_net_top"


@dataclass(frozen=True)
class SyntheticCourtGenerationConfig:
    out_dir: Path = DEFAULT_OUTPUT_DIR
    count: int = DEFAULT_COUNT
    seed: int = DEFAULT_SEED
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    spot_check_count: int = DEFAULT_SPOT_CHECK_COUNT
    generated_at_utc: str | None = None
    height_m_range: tuple[float, float] = (1.0, 12.0)
    distance_m_range: tuple[float, float] = (5.0, 40.0)
    azimuth_deg_range: tuple[float, float] = (-75.0, 75.0)
    tilt_deg_range: tuple[float, float] = (2.0, 80.0)
    focal_px_range: tuple[float, float] = (500.0, 2000.0)
    roll_deg_range: tuple[float, float] = (-4.0, 4.0)
    distortion_k1_range: tuple[float, float] = (-0.07, 0.04)
    distortion_p_range: tuple[float, float] = (0.0, 0.0)
    jpeg_quality_range: tuple[int, int] = (78, 96)
    line_width_px_range: tuple[int, int] = (2, 7)
    scenarios: tuple[str, ...] | None = None
    scenario_weights: dict[str, float] | None = None
    overwrite: bool = False

    def scene_render_config(self) -> SceneRenderConfig:
        weights = {name: 1.0 for name in SCENARIO_NAMES}
        if self.scenarios is not None:
            unknown = set(self.scenarios) - set(SCENARIO_NAMES)
            if unknown:
                raise ValueError(f"unknown synthetic court scenarios: {sorted(unknown)}")
            weights = {name: (1.0 if name in self.scenarios else 0.0) for name in SCENARIO_NAMES}
        if self.scenario_weights is not None:
            unknown = set(self.scenario_weights) - set(SCENARIO_NAMES)
            if unknown:
                raise ValueError(f"unknown synthetic court scenarios: {sorted(unknown)}")
            weights = {name: float(self.scenario_weights.get(name, 0.0)) for name in SCENARIO_NAMES}
        return SceneRenderConfig(
            image_size=self.image_size,
            height_m_range=self.height_m_range,
            distance_m_range=self.distance_m_range,
            azimuth_deg_range=self.azimuth_deg_range,
            tilt_deg_range=self.tilt_deg_range,
            roll_deg_range=self.roll_deg_range,
            focal_px_range=self.focal_px_range,
            distortion_k1_range=self.distortion_k1_range,
            distortion_p_range=self.distortion_p_range,
            jpeg_quality_range=self.jpeg_quality_range,
            line_width_px_range=self.line_width_px_range,
            scenario_weights=weights,
        )


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    image_rel_path: Path
    label_rel_path: Path
    overlay_rel_path: Path | None
    image_sha256: str
    label_sha256: str
    overlay_sha256: str | None
    scenario: str
    keypoints: dict[str, list[float]]
    generation: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_synthetic_court_corpus(config: SyntheticCourtGenerationConfig) -> dict[str, Any]:
    _validate_config(config)
    out_dir = config.out_dir
    _prepare_output_dir(out_dir, overwrite=config.overwrite)

    generated_at = config.generated_at_utc or datetime.now(timezone.utc).isoformat()
    rng = random.Random(config.seed)
    scene_config = config.scene_render_config()
    overlay_indices = set(_spot_check_indices(config.count, config.spot_check_count, config.seed))

    samples: list[SyntheticSample] = []
    for sample_index in range(config.count):
        scenario = choose_scenario(rng, scene_config.scenario_weights)
        samples.append(
            _generate_sample(
                out_dir,
                sample_index=sample_index,
                scenario=scenario,
                scene_config=scene_config,
                rng=rng,
                generated_at_utc=generated_at,
                seed=config.seed,
                write_overlay=sample_index in overlay_indices,
            )
        )

    scenario_counts = Counter(sample.scenario for sample in samples)
    manifest = {
        "schema_version": 2,
        "artifact_type": "synthetic_court_keypoint_corpus_manifest",
        "status": "synthetic_training_ammunition_not_gate_evidence",
        "generated_at_utc": generated_at,
        "seed": config.seed,
        "sample_count": config.count,
        "output_dir": _path_text(out_dir),
        "schema_notes": [
            "Each sample is loadable by scripts/racketsport/train_court_keypoint_heatmap.py via root/*/labels/court_keypoints.json.",
            "review.status is set to reviewed only to satisfy the existing loader contract.",
            "item.status is synthetic (CAL-R2 provenance fix, 2026-07-02) so these rows count separately "
            "(labels_synthetic_frame_count) from both independent human reviews and owner-approved "
            "static-camera copies -- never as any form of human verification.",
            "provenance.synthetic=true marks the labels as synthetic training augmentation, never gate evidence.",
            "CAL-SYNTH v2: each sample also carries a 'scenario' field (one of "
            f"{list(SCENARIO_NAMES)}); the zero-disk trainer contract "
            "(threed/racketsport/court_synth_stream.py) additionally emits line-family/surface "
            "masks + per-keypoint visibility, not persisted in this disk corpus.",
        ],
        "generation_config": {
            "image_size": list(config.image_size),
            "height_m_range": list(config.height_m_range),
            "distance_m_range": list(config.distance_m_range),
            "azimuth_deg_range": list(config.azimuth_deg_range),
            "tilt_deg_range": list(config.tilt_deg_range),
            "focal_px_range": list(config.focal_px_range),
            "roll_deg_range": list(config.roll_deg_range),
            "distortion_k1_range": list(config.distortion_k1_range),
            "distortion_p_range": list(config.distortion_p_range),
            "jpeg_quality_range": list(config.jpeg_quality_range),
            "line_width_px_range": list(config.line_width_px_range),
            "scenario_weights": dict(scene_config.scenario_weights),
        },
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "canonical_keypoint_names": [point.name for point in PICKLEBALL_KEYPOINTS],
        "court_template": _court_template_manifest(),
        "spot_check_overlays": [
            {
                "sample_id": sample.sample_id,
                "path": sample.overlay_rel_path.as_posix(),
                "sha256": sample.overlay_sha256,
            }
            for sample in samples
            if sample.overlay_rel_path is not None
        ],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_rel_path.as_posix(),
                "label_path": sample.label_rel_path.as_posix(),
                "image_sha256": sample.image_sha256,
                "label_sha256": sample.label_sha256,
                "overlay_path": sample.overlay_rel_path.as_posix() if sample.overlay_rel_path else None,
                "overlay_sha256": sample.overlay_sha256,
                "scenario": sample.scenario,
                "camera": sample.generation["camera"],
                "distortion_k1": sample.generation["camera"]["distortion_k1"],
                "occlusion_count": sample.generation["domain_randomization"]["occlusion_count"],
            }
            for sample in samples
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _generate_sample(
    out_dir: Path,
    *,
    sample_index: int,
    scenario: str,
    scene_config: SceneRenderConfig,
    rng: random.Random,
    generated_at_utc: str,
    seed: int,
    write_overlay: bool,
) -> SyntheticSample:
    sample_id = f"synthetic_court_{sample_index:06d}"
    sample_dir = out_dir / sample_id
    frames_dir = sample_dir / "frames"
    labels_dir = sample_dir / "labels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    scene: RenderedScene = render_synthetic_court_sample(
        rng, scene_config, scenario=scenario, apply_jpeg_roundtrip=True
    )
    keypoints = scene.keypoints_xy
    jpeg_quality = int(scene.meta["jpeg_quality"])

    frame_name = "frame_000000.jpg"
    image_path = frames_dir / frame_name
    scene.image.save(image_path, format="JPEG", quality=jpeg_quality, optimize=False, progressive=False)

    generation = _generation_payload(scene=scene)
    label_payload = _label_payload(
        sample_id=sample_id,
        frame_name=frame_name,
        frames_dir=frames_dir,
        keypoints=keypoints,
        image_size=tuple(scene.meta["image_size"]),
        generated_at_utc=generated_at_utc,
        seed=seed,
        sample_index=sample_index,
        generation=generation,
    )
    label_path = labels_dir / "court_keypoints.json"
    _write_json(label_path, label_payload)

    overlay_rel_path: Path | None = None
    overlay_sha256: str | None = None
    if write_overlay:
        overlay_dir = out_dir / "spot_check_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{sample_id}_overlay.jpg"
        overlay = scene.image.copy()
        _draw_keypoint_overlay(overlay, keypoints)
        overlay.save(overlay_path, format="JPEG", quality=94, optimize=False, progressive=False)
        overlay_rel_path = overlay_path.relative_to(out_dir)
        overlay_sha256 = sha256_file(overlay_path)

    return SyntheticSample(
        sample_id=sample_id,
        image_rel_path=image_path.relative_to(out_dir),
        label_rel_path=label_path.relative_to(out_dir),
        overlay_rel_path=overlay_rel_path,
        image_sha256=sha256_file(image_path),
        label_sha256=sha256_file(label_path),
        overlay_sha256=overlay_sha256,
        scenario=scenario,
        keypoints=keypoints,
        generation=generation,
    )


def _draw_keypoint_overlay(image: Any, keypoints: dict[str, list[float]]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    palette = [
        (255, 64, 64, 230),
        (64, 180, 255, 230),
        (80, 230, 120, 230),
        (255, 220, 64, 230),
    ]
    for idx, point in enumerate(PICKLEBALL_KEYPOINTS):
        x, y = keypoints[point.name]
        color = palette[idx % len(palette)]
        radius = 4
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(0, 0, 0, 220))
        draw.text((x + 5, y - 6), point.name, fill=(255, 255, 255, 235), stroke_width=1, stroke_fill=(0, 0, 0, 220))


def _label_payload(
    *,
    sample_id: str,
    frame_name: str,
    frames_dir: Path,
    keypoints: dict[str, list[float]],
    image_size: tuple[int, int],
    generated_at_utc: str,
    seed: int,
    sample_index: int,
    generation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": sample_id,
        "annotation": {
            "items": [
                {
                    "frame": frame_name,
                    "status": SYNTHETIC_ITEM_STATUS,
                    "net_keypoint_height_convention": NET_KEYPOINT_HEIGHT_CONVENTION,
                    "review_id": f"{sample_id}_synthetic_geometry_0000",
                    "keypoints": {name: [float(xy[0]), float(xy[1])] for name, xy in sorted(keypoints.items())},
                    "provenance": {
                        "synthetic": True,
                        "human_reviewed": False,
                        "generator": "scripts/racketsport/generate_synthetic_court_keypoints.py",
                    },
                }
            ]
        },
        "frames": {
            "available_review_frame_count": 1,
            "frame_count": 1,
            "frame_dir": _path_text(frames_dir),
            "label_coordinate_space": [image_size[0], image_size[1]],
            "sample_every_frames": 1,
            "source_resolution": [image_size[0], image_size[1]],
        },
        "review": {
            "status": "reviewed",
            "reviewer": "synthetic_geometry_generator",
            "reviewed_at_utc": generated_at_utc,
            "human_reviewed": False,
            "independent_reviewed_count": 0,
            "static_camera_copy_count": 0,
            "synthetic_count": 1,
        },
        "provenance": {
            "synthetic": True,
            "human_labels": False,
            "seed": seed,
            "sample_index": sample_index,
            "coordinate_frame": COORDINATE_FRAME,
            "generator": "scripts/racketsport/generate_synthetic_court_keypoints.py",
            "note": "Synthetic court geometry labels are training augmentation only, not CAL gate evidence.",
            "net_keypoint_height_convention": NET_KEYPOINT_HEIGHT_CONVENTION,
        },
        "generation": generation,
    }


def _generation_payload(*, scene: RenderedScene) -> dict[str, Any]:
    meta = scene.meta
    camera = meta["camera"]
    homography = meta["homography"]
    distortion = meta["distortion"]
    return {
        "scenario": scene.scenario,
        "camera": {
            "position_m": homography["camera_position_m"],
            "height_m": camera["height_m"],
            "distance_m": camera["distance_m"],
            "azimuth_deg": camera["azimuth_deg"],
            "tilt_deg": camera["tilt_deg"],
            "roll_deg": camera["roll_deg"],
            "focal_px": camera["focal_px"],
            "fx_px": homography["fx_px"],
            "fy_px": homography["fy_px"],
            "cx_px": homography["cx_px"],
            "cy_px": homography["cy_px"],
            "distortion_k1": distortion["k1"],
            "distortion_p1": distortion["p1"],
            "distortion_p2": distortion["p2"],
        },
        "world_to_image_homography": homography["ground_plane_3x3"],
        "net_keypoint_height_convention": NET_KEYPOINT_HEIGHT_CONVENTION,
        "court_template": _court_template_manifest(),
        "keypoint_world_xyz_m": meta["keypoint_world_xyz_m"],
        "keypoints_vis": dict(scene.keypoints_vis),
        "court_instances": meta["court_instances"],
        "domain_randomization": {
            "line_width_px": meta["line_width_px"],
            "occlusion_count": meta["occluder_count"],
            "jpeg_quality": meta["jpeg_quality"],
            "features": [
                "domain_randomized_camera_pose",
                "scenario_mixture_v2",
                "court_and_floor_color_jitter",
                "line_width_color_wear",
                "lighting_gradient",
                "shadows",
                "background_clutter",
                "radial_tangential_lens_distortion",
                "sensor_noise",
                "jpeg_artifacts",
                "partial_line_occlusions",
                "regulation_top_net_keypoints",
            ],
        },
    }


def _court_template_manifest() -> dict[str, Any]:
    template = get_court_template("pickleball")
    return {
        "sport": template.sport,
        "coordinate_frame": template.coordinate_frame,
        "length_ft": template.length_ft,
        "width_ft": template.width_ft,
        "net_width_ft": template.net_width_ft,
        "non_volley_zone_ft": template.non_volley_zone_ft,
        "center_net_height_in": template.center_net_height_in,
        "post_net_height_in": template.post_net_height_in,
    }


def _spot_check_indices(count: int, spot_check_count: int, seed: int) -> list[int]:
    if spot_check_count <= 0 or count <= 0:
        return []
    rng = random.Random(seed ^ 0x5F3759DF)
    return sorted(rng.sample(range(count), min(count, spot_check_count)))


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and not overwrite:
        generated = [path for path in out_dir.glob("synthetic_court_*") if path.is_dir()]
        if generated or (out_dir / "manifest.json").exists():
            raise ValueError(f"output directory already contains generated synthetic data: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in out_dir.glob("synthetic_court_*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if (out_dir / "spot_check_overlays").exists():
            shutil.rmtree(out_dir / "spot_check_overlays")
        if (out_dir / "manifest.json").exists():
            (out_dir / "manifest.json").unlink()


def _validate_config(config: SyntheticCourtGenerationConfig) -> None:
    if config.count <= 0:
        raise ValueError("count must be positive")
    width, height = config.image_size
    if width < 160 or height < 90:
        raise ValueError("image_size must be at least 160x90")
    if config.spot_check_count < 0:
        raise ValueError("spot_check_count must be non-negative")
    _validate_range(config.height_m_range, "height_m_range", minimum=1.0, maximum=12.0)
    _validate_range(config.distance_m_range, "distance_m_range", minimum=2.0)
    _validate_range(config.azimuth_deg_range, "azimuth_deg_range", minimum=-75.0, maximum=75.0)
    _validate_range(config.tilt_deg_range, "tilt_deg_range", minimum=0.0, maximum=89.0)
    _validate_range(config.focal_px_range, "focal_px_range", minimum=100.0)
    _validate_range(config.roll_deg_range, "roll_deg_range")
    _validate_range(config.distortion_k1_range, "distortion_k1_range")
    _validate_range(config.distortion_p_range, "distortion_p_range")
    if config.jpeg_quality_range[0] < 1 or config.jpeg_quality_range[1] > 100:
        raise ValueError("jpeg_quality_range must stay inside [1, 100]")
    if config.jpeg_quality_range[0] > config.jpeg_quality_range[1]:
        raise ValueError("jpeg_quality_range min must be <= max")
    if config.line_width_px_range[0] <= 0 or config.line_width_px_range[0] > config.line_width_px_range[1]:
        raise ValueError("line_width_px_range must be positive and ordered")
    if config.scenarios is not None:
        if not config.scenarios:
            raise ValueError("scenarios must be non-empty when provided")
        unknown = set(config.scenarios) - set(SCENARIO_NAMES)
        if unknown:
            raise ValueError(f"unknown synthetic court scenarios: {sorted(unknown)}")
    if config.scenario_weights is not None:
        unknown = set(config.scenario_weights) - set(SCENARIO_NAMES)
        if unknown:
            raise ValueError(f"unknown synthetic court scenarios: {sorted(unknown)}")


def _validate_range(
    value: tuple[float, float],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if len(value) != 2 or value[0] > value[1]:
        raise ValueError(f"{name} must be a two-value ordered range")
    if minimum is not None and value[0] < minimum:
        raise ValueError(f"{name} minimum must be >= {minimum}")
    if maximum is not None and value[1] > maximum:
        raise ValueError(f"{name} maximum must be <= {maximum}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_text(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic pickleball court keypoint labels.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--spot-check-count", type=int, default=DEFAULT_SPOT_CHECK_COUNT)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument(
        "--scenarios",
        default=None,
        help=f"Comma-separated subset of {list(SCENARIO_NAMES)} (default: all, uniform mixture).",
    )
    parser.add_argument(
        "--scenario-weights",
        default=None,
        help='JSON object of scenario -> weight, e.g. \'{"tennis_overlay": 2.0}\' (default: uniform).',
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenarios = tuple(args.scenarios.split(",")) if args.scenarios else None
    scenario_weights = json.loads(args.scenario_weights) if args.scenario_weights else None
    manifest = generate_synthetic_court_corpus(
        SyntheticCourtGenerationConfig(
            out_dir=args.out,
            count=args.count,
            seed=args.seed,
            image_size=(args.image_width, args.image_height),
            spot_check_count=args.spot_check_count,
            generated_at_utc=args.generated_at_utc,
            scenarios=scenarios,
            scenario_weights=scenario_weights,
            overwrite=args.overwrite,
        )
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "manifest": str(args.out / "manifest.json"),
                "spot_check_overlays": len(manifest["spot_check_overlays"]),
                "scenario_counts": manifest["scenario_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
