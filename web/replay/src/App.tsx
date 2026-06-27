import { useMemo } from "react";

import { parseReplayScene, type ReplayScene } from "./replayScene";

const sampleScene = {
  schema_version: 1,
  world_frame: "court_Z0",
  fps: 60,
  court_glb: "court.glb",
  players: [1, 2],
  points: [{ id: 1, t0: 0, t1: 3.5, glb_url: "points/point_1.glb", size_mb: 2.5 }],
};

type ReplayAppProps = {
  sceneInput?: unknown;
};

export default function App({ sceneInput = sampleScene }: ReplayAppProps) {
  const scene = useMemo(() => parseReplayScene(sceneInput), [sceneInput]);

  return <ReplaySummary scene={scene} />;
}

function ReplaySummary({ scene }: { scene: ReplayScene }) {
  return (
    <main aria-label="Replay viewer">
      <h1>Replay Viewer</h1>
      <dl>
        <div>
          <dt>Scenes</dt>
          <dd>1</dd>
        </div>
        <div>
          <dt>Players</dt>
          <dd>{scene.players.length}</dd>
        </div>
        <div>
          <dt>Points</dt>
          <dd>{scene.points.length}</dd>
        </div>
      </dl>
    </main>
  );
}
