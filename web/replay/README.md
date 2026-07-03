# Replay Viewer

`web/replay` is the browser review surface for pipeline bundles. It loads a
`replay_viewer_manifest.json`, parses the referenced world/contact/scene assets,
and renders them with trust-band badges. It is a QA/review viewer, not proof that
the production native replay gate has passed.

## Commands

```bash
npm install
npm run dev -- --host 127.0.0.1
npm test
npm run typecheck
npm run build
```

Open a local manifest:

```text
http://127.0.0.1:5173/?manifest=/@fs/absolute/path/to/replay_viewer_manifest.json
```

If the manifest references files outside the repo root, start Vite with a
matching `server.fs.allow` policy or pass `--vite-allow-root` to the manifest
builder/verifier that created the bundle.

## Verification

For replay-viewer code changes, run:

```bash
npm test -- --run --dir web/replay
npm run typecheck --prefix web/replay
```

For a specific pipeline bundle, use:

```bash
python3 scripts/racketsport/verify_process_video_viewer.py \
  --manifest <run>/replay_viewer_manifest.json \
  --out-dir <run>/viewer_verify
```
