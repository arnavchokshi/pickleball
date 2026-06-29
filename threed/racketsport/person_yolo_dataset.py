"""Export normalized person GT into YOLO detection training labels."""

from __future__ import annotations

from .schemas import PersonLabel


def yolo_label_line(label: PersonLabel, *, image_width: int, image_height: int, class_id: int = 0) -> str:
    """Return one YOLO detection label line in normalized cx cy w h format."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, width, height = [float(value) for value in label.bbox_xywh]
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        raise ValueError("person bbox is outside image bounds")
    cx = (x1 + x2) * 0.5 / float(image_width)
    cy = (y1 + y2) * 0.5 / float(image_height)
    norm_w = clipped_width / float(image_width)
    norm_h = clipped_height / float(image_height)
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}"


__all__ = ["yolo_label_line"]
