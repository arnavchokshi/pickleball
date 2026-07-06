import React, { useMemo } from "react";

import { buildCourtMapShots, svgCourtProjector, type BallArcRender, type CourtMapShot } from "./ballArcRender";
import { frameForTime, type Vec2, type VirtualWorld } from "./viewerData";

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
    () =>
      svgCourtProjector({
        widthM: world.court.width_m,
        lengthM: world.court.length_m,
        paddingPx: SVG_PADDING,
        widthPx: SVG_WIDTH,
        heightPx: SVG_HEIGHT,
      }),
    [world.court.length_m, world.court.width_m],
  );
  const shots = useMemo(() => buildCourtMapShots(arcRender, currentTime), [arcRender, currentTime]);
  const playerPositions = useMemo(
    () =>
      world.players
        .map((player) => {
          const frame = frameForTime(player, currentTime);
          const xy = frame?.floor_world_xyz ? ([frame.floor_world_xyz[0], frame.floor_world_xyz[1]] as Vec2) : frame?.track_world_xy ?? null;
          return xy ? { playerId: player.id, xy } : null;
        })
        .filter((entry): entry is { playerId: number; xy: Vec2 } => entry !== null),
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
        <g className="court-map-players">
          {playerPositions.map(({ playerId, xy }) => {
            const [cx, cy] = project(xy);
            return (
              <g key={playerId} className="court-map-player" transform={`translate(${cx} ${cy})`}>
                <circle r="6" />
                <text x="9" y="4">P{playerId}</text>
              </g>
            );
          })}
        </g>
      </svg>
      <div className="court-map-readout">
        <span>{shots.length} shots</span>
        <span>{activeShot ? `${activeShot.speedMph.toFixed(1)} mph` : "no active shot"}</span>
        <span>{activeShot?.heightOverNetM === null || activeShot?.heightOverNetM === undefined ? "net --" : `net ${activeShot.heightOverNetM.toFixed(2)}m`}</span>
      </div>
    </div>
  );
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
