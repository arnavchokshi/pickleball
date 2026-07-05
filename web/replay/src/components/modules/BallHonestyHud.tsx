import React, { useMemo } from "react";

import { ballHudStateForTime, samplesFromVirtualWorld, type BallTrailArtifact, type BallTrailSample } from "./ballTrail";

export function BallHonestyHud({
  samples,
  arcSolved,
  world,
  currentTime,
  confidenceThreshold = 0.5,
}: {
  samples?: BallTrailSample[] | null;
  arcSolved?: BallTrailArtifact | null;
  world?: Parameters<typeof samplesFromVirtualWorld>[0] | null;
  currentTime: number;
  confidenceThreshold?: number;
}) {
  const resolvedSamples = useMemo(() => {
    if (samples) return samples;
    if (arcSolved?.samples.length) return arcSolved.samples;
    return world ? samplesFromVirtualWorld(world) : [];
  }, [arcSolved, samples, world]);
  const readout = useMemo(
    () => ballHudStateForTime(resolvedSamples, currentTime, { confidenceThreshold }),
    [confidenceThreshold, currentTime, resolvedSamples],
  );
  return (
    <div
      aria-label="Ball tracking honesty"
      data-ball-state={readout.state}
      data-low-confidence={readout.lowConfidence ? "true" : "false"}
      style={{
        alignItems: "center",
        background: "rgba(22, 24, 27, 0.82)",
        border: "1px solid rgba(255, 255, 255, 0.18)",
        borderRadius: 8,
        color: "#f5f7f8",
        display: "inline-flex",
        fontSize: 12,
        gap: 8,
        letterSpacing: 0,
        lineHeight: 1,
        padding: "8px 10px",
        pointerEvents: "none",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          background:
            readout.state === "measured" ? "#e8ff34" : readout.state === "predicted" ? "repeating-linear-gradient(90deg,#63d9ff 0 5px,transparent 5px 9px)" : "#69717b",
          borderRadius: "50%",
          boxShadow: readout.state === "predicted" ? "0 0 0 3px rgba(99,217,255,0.16)" : "none",
          display: "inline-block",
          height: 10,
          width: 10,
        }}
      />
      <span>{readout.label}</span>
      {readout.lowConfidence ? <span style={{ color: "#ffcf5a" }}>low confidence</span> : null}
    </div>
  );
}
