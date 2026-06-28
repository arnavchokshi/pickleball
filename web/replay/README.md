# Racketsport Replay Viewer

Review-only replay viewer for inspecting `replay_viewer_manifest.json`,
`virtual_world.json`, player boxes, ball/contact cues, and physics summaries.
It is a QA surface, not the production animated GLB/USDZ replay.

This package currently includes:

- Strict TypeScript parsers for viewer, world, contact, and physics artifacts.
- A React Three Fiber / Three.js court-world renderer with video sync.
- Vitest and TypeScript scripts for local checks.

## Commands

```sh
npm install
npm test
npm run typecheck
npm run build
```
