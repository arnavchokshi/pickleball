import React, { useCallback, useRef, useState } from "react";

import { COURT_REVIEW_LINES, PICKLEBALL_COURT_REVIEW_POINTS, type CourtReviewPointMap, type CourtReviewPointName } from "./courtReview";
import "./CourtReviewCanvas.css";

export type CourtReviewCanvasProps = {
  imageUrl: string | null;
  imageSize: [number, number];
  points: CourtReviewPointMap;
  needsUserInput?: string[];
  onPointsChange: (next: CourtReviewPointMap) => void;
  onConfirm: () => void;
  onRepredict: () => void;
  onSkip: () => void;
  disabled?: boolean;
};

/** Clamps an image-pixel-space point to the video frame bounds. */
export function clampPointToImage([x, y]: [number, number], [width, height]: [number, number]): [number, number] {
  return [Math.max(0, Math.min(width, x)), Math.max(0, Math.min(height, y))];
}

/**
 * Pure drag-update: returns a new point map with only the named point's xy replaced
 * (clamped to the frame), preserving every other point and the moved point's confidence.
 */
export function applyPointDrag(
  points: CourtReviewPointMap,
  name: CourtReviewPointName,
  nextXy: [number, number],
  imageSize: [number, number],
): CourtReviewPointMap {
  const current = points[name];
  const clamped = clampPointToImage(nextXy, imageSize);
  return {
    ...points,
    [name]: { xy: clamped, confidence: current?.confidence ?? 0 },
  };
}

/** Computes the live court-line polylines (template edges) from the current point map. */
export function courtReviewLinePoints(points: CourtReviewPointMap): Array<{ id: string; svgPoints: string }> {
  const lines: Array<{ id: string; svgPoints: string }> = [];
  for (const line of COURT_REVIEW_LINES) {
    const [startName, endName] = line.points;
    const start = points[startName];
    const end = points[endName];
    if (!start || !end) continue;
    lines.push({ id: line.id, svgPoints: `${start.xy[0]},${start.xy[1]} ${end.xy[0]},${end.xy[1]}` });
  }
  return lines;
}

/** Converts a pointer offset within the displayed (CSS-scaled) stage into image-pixel space. */
export function imagePointFromDisplayOffset(
  offset: [number, number],
  displaySize: [number, number],
  imageSize: [number, number],
): [number, number] {
  const [offsetX, offsetY] = offset;
  const [displayWidth, displayHeight] = displaySize;
  const [imageWidth, imageHeight] = imageSize;
  const scaleX = displayWidth > 0 ? imageWidth / displayWidth : 1;
  const scaleY = displayHeight > 0 ? imageHeight / displayHeight : 1;
  return clampPointToImage([offsetX * scaleX, offsetY * scaleY], imageSize);
}

export function courtReviewPointStatus(name: string, needsUserInput: string[] | undefined): "needs_review" | "ok" {
  return needsUserInput?.includes(name) ? "needs_review" : "ok";
}

export function CourtReviewCanvas({
  imageUrl,
  imageSize,
  points,
  needsUserInput,
  onPointsChange,
  onConfirm,
  onRepredict,
  onSkip,
  disabled = false,
}: CourtReviewCanvasProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [draggingPoint, setDraggingPoint] = useState<CourtReviewPointName | null>(null);
  const [width, height] = imageSize;

  const movePointFromClient = useCallback(
    (name: CourtReviewPointName, clientX: number, clientY: number) => {
      const stage = stageRef.current;
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const nextXy = imagePointFromDisplayOffset([clientX - rect.left, clientY - rect.top], [rect.width, rect.height], imageSize);
      onPointsChange(applyPointDrag(points, name, nextXy, imageSize));
    },
    [imageSize, onPointsChange, points],
  );

  function handlePointerDown(name: CourtReviewPointName, event: React.PointerEvent<SVGCircleElement>) {
    if (disabled) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDraggingPoint(name);
    movePointFromClient(name, event.clientX, event.clientY);
  }

  function handlePointerMove(event: React.PointerEvent<SVGCircleElement>) {
    if (!draggingPoint || disabled) return;
    movePointFromClient(draggingPoint, event.clientX, event.clientY);
  }

  function handlePointerUp(event: React.PointerEvent<SVGCircleElement>) {
    if (draggingPoint) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setDraggingPoint(null);
  }

  const lines = courtReviewLinePoints(points);

  return (
    <div className="court-review-canvas" aria-label="Court review">
      <div className="court-review-stage" ref={stageRef} style={{ aspectRatio: `${width} / ${height}` }}>
        {imageUrl ? <img className="court-review-frame" src={imageUrl} alt="Predicted court frame" draggable={false} /> : null}
        <svg className="court-review-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="presentation">
          {lines.map((line) => (
            <polyline key={line.id} className="court-review-line" points={line.svgPoints} />
          ))}
          {PICKLEBALL_COURT_REVIEW_POINTS.map((name) => {
            const point = points[name];
            if (!point) return null;
            const status = courtReviewPointStatus(name, needsUserInput);
            return (
              <circle
                key={name}
                className={`court-review-point court-review-point-${status}`}
                data-point-name={name}
                cx={point.xy[0]}
                cy={point.xy[1]}
                r={Math.max(4, width * 0.012)}
                onPointerDown={(event) => handlePointerDown(name, event)}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
              >
                <title>{name}</title>
              </circle>
            );
          })}
        </svg>
      </div>
      <div className="court-review-actions">
        <button type="button" className="court-review-confirm" onClick={onConfirm} disabled={disabled}>
          Confirm court
        </button>
        <button type="button" className="court-review-repredict" onClick={onRepredict} disabled={disabled}>
          Re-predict
        </button>
        <button type="button" className="court-review-skip" onClick={onSkip} disabled={disabled}>
          Skip (no court)
        </button>
      </div>
      {needsUserInput && needsUserInput.length > 0 ? (
        <p className="court-review-hint">Drag to fix: {needsUserInput.join(", ")}</p>
      ) : null}
    </div>
  );
}
