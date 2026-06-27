# Racketsport Replay Viewer

Minimal CPU/testable scaffold for validating and summarizing replay scene JSON.

This package currently includes:

- A strict TypeScript parser for the `threed.racketsport.schemas.ReplayScene` fields.
- A placeholder React component that displays scene, player, and point counts.
- Vitest and TypeScript scripts for local checks.

No Three.js replay rendering is implemented in this scaffold.

## Commands

```sh
npm install --package-lock=false
npm test
npm run typecheck
```
