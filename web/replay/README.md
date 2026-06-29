# Racketsport Replay Viewer

Review-only replay viewer for inspecting `replay_viewer_manifest.json`,
`virtual_world.json`, `replay_scene.json` metadata, player boxes,
ball/contact cues, and physics summaries.
It is a QA surface, not the production animated GLB/USDZ replay.

This package currently includes:

- Strict TypeScript parsers for viewer, world, replay-scene, contact, and physics artifacts.
- A React Three Fiber / Three.js court-world renderer with video sync.
- Vitest and TypeScript scripts for local checks.

## Commands

```sh
npm install
npm run dev -- --host 127.0.0.1
npm test
npm run typecheck
npm run build
```

Open the viewer with a manifest URL, for example:

```text
http://127.0.0.1:5173/?manifest=/@fs/absolute/path/to/replay_viewer_manifest.json
```

`build_replay_viewer_manifest.py` validates every emitted `/@fs` source against a Vite allow root. By default that root is the repo root; pass `--vite-allow-root <path>` when reviewing artifacts from a temporary or external run directory, and start Vite with a matching `server.fs.allow` policy if needed.
