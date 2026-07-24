import React, { useMemo } from "react";

import { buildCourtMapShots, sampleBallArcRenderAtTime, svgCourtProjector, type BallArcRender, type CourtMapShot } from "./ballArcRender";
import { frameForTime, type PlayerPlacementDiagnostics, type Vec2, type VirtualWorld } from "./viewerData";

const SVG_WIDTH = 305;
const SVG_HEIGHT = 520;
const SVG_PADDING = 24;

export function CourtMapPanel({
  world,
  arcRender,
  currentTime,
}: {
  world: VirtualWorld;
  arcRender: BallArcRender | null;
  currentTime: number;
}) {
  const project = useMemo(
    () => {
      const bounds = courtCoordinateBounds(world);
      return svgCourtProjector({
        widthM: world.court.width_m,
        lengthM: world.court.length_m,
        paddingPx: SVG_PADDING,
        widthPx: SVG_WIDTH,
        heightPx: SVG_HEIGHT,
        ...bounds,
      });
    },
    [world],
  );
  const shots = useMemo(() => buildCourtMapShots(arcRender, currentTime), [arcRender, currentTime]);
  const currentBall = useMemo(
    () => (arcRender ? sampleBallArcRenderAtTime(arcRender.samples, currentTime) : null),
    [arcRender, currentTime],
  );
  const playerPositions = useMemo(
    () =>
      world.players
        .map((player) => {
          const frame = frameForTime(player, currentTime);
          const xy = frame?.floor_world_xyz ? ([frame.floor_world_xyz[0], frame.floor_world_xyz[1]] as Vec2) : frame?.track_world_xy ?? null;
          return xy ? { playerId: player.id, xy, diagnostics: frame?.placement_diagnostics ?? null } : null;
        })
        .filter(
          (entry): entry is { playerId: number; xy: Vec2; diagnostics: PlayerPlacementDiagnostics | null } =>
            entry !== null,
        ),
    [currentTime, world],
  );
  const activeShot = shots.find((shot) => shot.active) ?? null;
  return (
    <div className="court-map-panel" aria-label="Top-down court map">
      <svg viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} role="img" aria-label="Top-down court map">
        <CourtOutline world={world} project={project} />
        <g className="court-map-shots">
          {shots.map((shot) => (
            <ShotPath key={String(shot.segmentId)} shot={shot} project={project} />
          ))}
        </g>
        <g className="court-map-bounces">
          {shots.map((shot) => {
            const [cx, cy] = project(shot.end);
            return <circle key={`bounce-${String(shot.segmentId)}`} className="court-map-bounce-dot" cx={cx} cy={cy} r={shot.active ? 5.6 : 3.8} />;
          })}
        </g>
        {currentBall?.world_xyz ? (
          <g className="court-map-current-ball" transform={`translate(${project([currentBall.world_xyz[0], currentBall.world_xyz[1]]).join(" ")})`}>
            <circle r="6.2" />
          </g>
        ) : null}
        <g className="court-map-players">
          {playerPositions.map(({ playerId, xy, diagnostics }) => {
            const [cx, cy] = project(xy);
            const ellipse = diagnostics?.covarianceM2
              ? placementUncertaintyEllipse(xy, diagnostics.covarianceM2, project)
              : null;
            const line = diagnostics?.nearestRegulationLine;
            const candidateSummary = diagnostics
              ? diagnostics.footCandidates
                .map((candidate) => {
                  const semanticNames = candidate.keypointCandidates.map((point) => point.semanticName).join("+");
                  return `${candidate.source}/${candidate.foot}${semanticNames ? `(${semanticNames})` : ""}${candidate.accepted ? " accepted" : ` rejected:${candidate.rejectionReason ?? "not_selected"}`}`;
                })
                .join(" | ")
              : "";
            return (
              <g key={playerId} className="court-map-player" transform={`translate(${cx} ${cy})`}>
                {ellipse ? (
                  <ellipse
                    className="court-map-player-uncertainty"
                    rx={ellipse.rx}
                    ry={ellipse.ry}
                    transform={`rotate(${ellipse.rotationDeg})`}
                  />
                ) : null}
                <circle r="6" />
                <text x="9" y="4">P{playerId}</text>
                <title>
                  {diagnostics
                    ? `P${playerId} ${diagnostics.measurementProvenance}; contact ${diagnostics.contactState}; support ${diagnostics.supportFoot ?? "--"}; signal ${diagnostics.selectedSupportSignal?.name ?? "--"}; pixel ${diagnostics.selectedSupportSignal?.pixelXY.join(", ") ?? "--"}; uncertainty ${diagnostics.uncertaintyDecomposition?.dominantInput ?? "unknown"}-dominated; candidates ${candidateSummary || "--"}; nearest ${line?.lineName ?? "--"} ${line ? line.signedDistanceM.toFixed(2) : "--"}m`
                    : `P${playerId} placement diagnostics unavailable`}
                </title>
              </g>
            );
          })}
        </g>
      </svg>
      <div className="court-map-readout">
        <span>{shots.length} shots</span>
        <span>{activeShot ? `${activeShot.speedMph.toFixed(1)} mph` : "no active shot"}</span>
        <span>{activeShot?.heightOverNetM === null || activeShot?.heightOverNetM === undefined ? "net --" : `net ${activeShot.heightOverNetM.toFixed(2)}m`}</span>
        {playerPositions.map(({ playerId, diagnostics }) => diagnostics ? (
          <span className="court-map-placement-readout" key={`placement-${playerId}`}>
            {`P${playerId} ${diagnostics.contactState} · ${diagnostics.nearestRegulationLine?.lineName ?? "line --"} ${diagnostics.nearestRegulationLine ? diagnostics.nearestRegulationLine.signedDistanceM.toFixed(2) : "--"}m · ${diagnostics.uncertaintyDecomposition?.dominantInput ?? "unknown"}-dominated · ${diagnostics.measurementProvenance}`}
          </span>
        ) : null)}
      </div>
    </div>
  );
}

function placementUncertaintyEllipse(
  center: Vec2,
  covariance: [Vec2, Vec2],
  project: (point: Vec2) => Vec2,
): { rx: number; ry: number; rotationDeg: number } {
  const a = Math.max(0, covariance[0][0]);
  const b = (covariance[0][1] + covariance[1][0]) * 0.5;
  const d = Math.max(0, covariance[1][1]);
  const trace = a + d;
  const spread = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  const major = Math.max(0, (trace + spread) * 0.5);
  const minor = Math.max(0, (trace - spread) * 0.5);
  const theta = 0.5 * Math.atan2(2 * b, a - d);
  const scale95 = 2.4477;
  const majorWorld = scale95 * Math.sqrt(major);
  const minorWorld = scale95 * Math.sqrt(minor);
  const projectedCenter = project(center);
  const majorPoint = project([center[0] + majorWorld * Math.cos(theta), center[1] + majorWorld * Math.sin(theta)]);
  const minorPoint = project([center[0] - minorWorld * Math.sin(theta), center[1] + minorWorld * Math.cos(theta)]);
  return {
    rx: Math.max(2, Math.hypot(majorPoint[0] - projectedCenter[0], majorPoint[1] - projectedCenter[1])),
    ry: Math.max(2, Math.hypot(minorPoint[0] - projectedCenter[0], minorPoint[1] - projectedCenter[1])),
    rotationDeg: Math.atan2(majorPoint[1] - projectedCenter[1], majorPoint[0] - projectedCenter[0]) * 180 / Math.PI,
  };
}

function courtCoordinateBounds(world: VirtualWorld): { xMin: number; xMax: number; yMin: number; yMax: number } {
  const points = Object.values(world.court.line_segments).flat();
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const centeredY = ys.some((y) => y < 0);
  return {
    xMin: Math.min(...xs, -world.court.width_m / 2),
    xMax: Math.max(...xs, world.court.width_m / 2),
    yMin: centeredY ? Math.min(...ys) : Math.min(...ys, 0),
    yMax: centeredY ? Math.max(...ys) : Math.max(...ys, world.court.length_m),
  };
}

function CourtOutline({ world, project }: { world: VirtualWorld; project: (point: Vec2) => Vec2 }) {
  return (
    <g className="court-map-outline">
      <rect
        x={SVG_PADDING}
        y={SVG_PADDING}
        width={SVG_WIDTH - SVG_PADDING * 2}
        height={SVG_HEIGHT - SVG_PADDING * 2}
        rx="0"
      />
      {Object.entries(world.court.line_segments).map(([name, [from, to]]) => {
        const [x1, y1] = project([from[0], from[1]]);
        const [x2, y2] = project([to[0], to[1]]);
        return <line key={name} x1={x1} y1={y1} x2={x2} y2={y2} />;
      })}
    </g>
  );
}

function ShotPath({ shot, project }: { shot: CourtMapShot; project: (point: Vec2) => Vec2 }) {
  const start = project(shot.start);
  const peak = project(shot.peak);
  const end = project(shot.end);
  const control: Vec2 = [(peak[0] + (start[0] + end[0]) * 0.5) * 0.5, peak[1] - Math.min(42, Math.max(8, shot.confidence * 30))];
  const d = `M ${start[0]} ${start[1]} Q ${control[0]} ${control[1]} ${end[0]} ${end[1]}`;
  return (
    <g className={shot.active ? "court-map-shot active" : "court-map-shot"} data-active={shot.active ? "true" : "false"}>
      <path className="court-map-shot-line" d={d} />
      <circle className="court-map-strike-dot" cx={start[0]} cy={start[1]} r={shot.active ? 5.2 : 3.4} />
    </g>
  );
}
