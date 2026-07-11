# Mac real-GPU browser FPS measurement — racketsport replay viewer

Date: 2026-07-10
Lane: fps_mac_20260710
Context: runs/research_deepreview_20260710/RULINGS.md §S1 found every prior
"3D FPS" number came from headless Chromium's SwiftShader software rasterizer
(confirmed again in this lane — see "Prior artifact reconfirmed" below). This
lane produces the first FPS reading taken from a Chromium instance that is
verifiably using this Mac's real GPU via Metal.

## Machine context

- Chip: `Apple M1 Pro` (`sysctl -n machdep.cpu.brand_string`)
- OS: Darwin 25.5.0 (macOS), arm64
- GPU: Apple M1 Pro, 14 cores, Metal 4 (`system_profiler SPDisplaysDataType`)
- Display: built-in Liquid Retina XDR, 3024x1964, ProMotion (up to 120Hz) —
  relevant because the FPS ceiling observed below matches this panel's max
  refresh rate (see Honest Issues #1).

## What was served

- Vite dev server: `cd web/replay && VITE_REPLAY_VERIFY_DEV_BYPASS=1 REPLAY_VERIFY_DEV_BYPASS=1 nohup npx vite --host 127.0.0.1 --port 5183 > /tmp/vite_fps_lane.log 2>&1 & disown`
  - The auth-bypass env vars were required: without them the app shows
    `SignInScreen` and never mounts `.world-panel canvas` (confirmed by a
    first failed run — `Page.wait_for_selector: Timeout 15000ms exceeded`
    waiting for the canvas). `web/replay/src/AppShell.tsx` gates on
    `replayVerifyDevBypassFromRuntime()` (`web/replay/src/devAuthBypass.ts`),
    which requires `VITE_REPLAY_VERIFY_DEV_BYPASS=1` + a loopback hostname +
    non-production Vite mode.
- Manifest: `web/replay/public/` had no staged bundle (only a pre-existing
  `critique/` folder, untouched by this lane) — **nothing was copied into
  `web/replay/public`**. Instead the manifest was served directly from its
  `runs/` location via Vite's `/@fs<absolute path>` dev-only passthrough,
  exactly the mechanism `scripts/racketsport/verify_process_video_viewer.py`
  (`viewer_url_for_manifest`, line 51-54) already uses, and confirmed
  reachable with `curl` (HTTP 200) before launching a browser.
  `web/replay/vite.config.ts` allows `fs.allow` = repo root, so any path
  under the repo — including `runs/lanes/...` — is servable this way, no
  copy required.
- Manifest file used: `runs/lanes/demo_beststack_render_20260710/after_wolv/demo_manifest.json`
  (the "wolverine" demo world from today's earlier demo-render session).
- **Exact URL measured**:
  `http://127.0.0.1:5183/?manifest=/@fs/Users/arnavchokshi/Desktop/pickleball/runs/lanes/demo_beststack_render_20260710/after_wolv/demo_manifest.json&view=world`

## Renderer proof (real hardware GPU, not SwiftShader)

Evaluated in-page (both a scratch `<canvas>` and the viewer's actual
`.world-panel canvas`, same result for both):

```json
{
  "unmasked_vendor": "Google Inc. (Apple)",
  "unmasked_renderer": "ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro, Unspecified Version)",
  "vendor": "WebKit",
  "renderer": "WebKit WebGL"
}
```

`UNMASKED_RENDERER_WEBGL` = **"ANGLE (Apple, ANGLE Metal Renderer: Apple M1
Pro, Unspecified Version)"** — this is Chromium's ANGLE backend running on
real Metal against the M1 Pro's GPU. It does **not** say SwiftShader or
SoftwareAdapter. Chromium was launched **headed** (`headless=False`, a
visible window actually opened on this Mac's display) via Python Playwright
1.57.0 (already installed; browsers `chromium-1200`/`chromium-1228` already
cached under `~/Library/Caches/ms-playwright`) — no `--use-angle=metal` flag
was even needed once running headed; the default headed launch picked
hardware Metal automatically.

### Prior artifact reconfirmed
For contrast/documentation: the pre-existing headless run at
`runs/lanes/demo_beststack_render_20260710/frames_after_world/headless_verify.json`
recorded `"3D FPS": "1.2"` from `p.chromium.launch(headless=True)` with no
special GL flags — consistent with the deep review's SwiftShader-artifact
finding. This lane did not need to re-run that headless case to prove the
point; the recorded 1.2 fps number already stands as the software-rasterizer
artifact this lane supersedes with a real-hardware number.

## FPS measurement method

The viewer computes its own "3D FPS" honestly: `web/replay/src/App.tsx`
`FpsProbe` (line 1817-1830) samples `useFrame` (react-three-fiber's
`requestAnimationFrame` render callback) into a rolling ~500ms window
(`updateFpsSample`, `replayScene.ts` line 184-193) and displays it via
`<Metric label="3D FPS" value=... />` in `.status-grid > dt/dd`. This lane
read that same DOM text every second (`document.querySelectorAll(".status-grid > div")`,
identical selector to `verify_process_video_viewer.py`'s `_collect_loaded_counts`).

Driver script: `runs/lanes/fps_mac_20260710/measure_fps.py` (60 samples/1s
while playing, 15 samples/1s while paused). Raw data:
`runs/lanes/fps_mac_20260710/raw_samples_headed.json`,
summary: `runs/lanes/fps_mac_20260710/summary_stats.json`.
Screenshots: `runs/lanes/fps_mac_20260710/screenshot_headed.png`,
`runs/lanes/fps_mac_20260710/screenshot_meshdebug.png`.

Playback was started by clicking the viewer's own `Play` button
(`page.locator('button:has-text("Play")')` — succeeded, logged as
`play_control_note: "clicked button:has-text(\"Play\")"`).

## Results

**PLAYING (60 samples, 1 Hz, real Metal GPU, headed Chromium):**
- min = 116.0, p50 = 119.9, p95 = 120.3, max = 120.4, mean = 119.56
- fps sequence: 118.0, 120.0, 118.3, 120.0, 118.2, 118.0, 119.9, 120.0, 120.1,
  120.0, 119.9, 118.0, 120.2, 119.8, 118.0, 119.7, 120.0, 118.1, 119.9, 119.6,
  119.9, 116.0, 119.9, 120.2, 120.0, 120.4, 120.0, 119.7, 119.9, 120.0, 119.7,
  120.0, 118.3, 120.0, 119.6, 119.9, 120.2, 120.0, 119.8, 120.0, 119.9, 120.3,
  120.0, 119.9, 119.6, 120.1, 118.0, 120.0, 119.8, 118.0, 120.0, 120.1, 119.9,
  120.2, 120.1, 120.0, 118.2, 120.3, 120.1, 120.0

**PAUSED (15 samples, 1 Hz, video element paused):**
- min = 118.0, p50 = 120.0, p95 = 120.1, max = 120.1, mean = 119.45
- Statistically indistinguishable from "playing" — see Honest Issue #2.

**Other on-screen stats (status-grid), sampled at load, t=0:**
- Players: 4
- Coverage gaps: 0/4
- Contacts: 0
- Ball: "not visible" (timeline was not scrubbed to a ball-visible frame —
  see Honest Issue #3)
- Warnings: "2 notices: missing mesh vertices, 2D-only ball frames outside
  solved arc coverage" (pre-existing data-quality notices from the demo
  run's manifest, unrelated to this FPS measurement)
- Active mesh count: **not obtainable** — see Honest Issue #5.

**Console/page errors during the 75s measurement run:**
- `page_errors`: none
- `console_errors` (2): both `"[error] Failed to load resource: the server
  responded with a status of 404 (Not Found)"` — message text only, exact
  request URL not captured by this pass (see Honest Issue #4). A follow-up
  pass on the same URL recorded zero >=400 responses
  (`runs/lanes/fps_mac_20260710/meshdebug_extra.json`), so this looks
  transient/non-blocking, not a rendering-relevant failure.

## HONEST ISSUES

1. **120fps is a vsync ceiling, not a proof of unbounded throughput.** This
   MacBook's built-in display is ProMotion (up to 120Hz). The FpsProbe reads
   the render loop's actual `requestAnimationFrame` rate, and it is pegged
   at ~120 with almost no variance (min 116, one dip to 116 out of 60
   samples) — meaning the scene is *not* dropping frames against this
   display's refresh rate, but it does **not** tell us the scene's actual
   frame budget headroom (e.g., how many ms/frame is spent, or what the FPS
   would be on a 60Hz external monitor, where the same content would likely
   read ~60). This is nonetheless a categorically valid, real-hardware
   number — it replaces a 1.2 fps SwiftShader software-rasterizer artifact
   with proof the app comfortably holds the display's max refresh rate — but
   it should not be quoted as "the 3D viewer renders at 120fps" without that
   caveat.
2. **"Paused" is not a meaningful FPS contrast for this app.** Pausing only
   pauses the underlying `<video>` element; `FpsProbe`'s `useFrame` hook
   (react-three-fiber's render loop) keeps running continuously regardless
   of playback state, so paused (mean 119.45) and playing (mean 119.56) FPS
   are indistinguishable. This is accurate to how the app behaves, not a
   script bug, but it means this lane could not produce a genuine
   "idle vs. active" FPS contrast as the mission envisioned — a more
   meaningful contrast would require comparing a frame with 0 active
   overlays vs. one with all overlay layers + solid meshes + point clouds
   rendering simultaneously at a busy contact-frame timestamp, which this
   pass did not isolate.
3. **Ball/contact overlays were not scrubbed into view.** Samples were taken
   from t=0 onward via the Play button; "Ball" read "not visible" throughout
   the visible window rather than showing the "measured/predicted/hidden"
   breakdown seen in the prior headless run's screenshot capture (which used
   explicit `--screenshot-at-seconds` scrubbing). FPS was still sampled
   during genuine video playback, so this is a fair typical-playback sample,
   just not a worst-case/busiest-overlay stress test.
4. **2 console 404 errors, root cause not identified.** Only `msg.text` was
   captured (`"Failed to load resource: ... 404"`), not the failing request
   URL, and a follow-up pass on the identical URL recorded 0 such errors —
   so this could not be reproduced/pinned down. `page_errors` was empty and
   the canvas rendered correctly in the screenshot both times, so this is
   flagged as unresolved-but-apparently-non-fatal, not as a false "all
   clear."
5. **"Active mesh count" could not be read from the DOM.** The app's only
   scene-introspection readout (`MeshDebugReadout` / `.mesh-debug-readout`,
   `web/replay/src/App.tsx` line 1486-1507, exposing `rendered_player_count`
   etc.) only mounts when `replayScene !== null`
   (`web/replay/src/App.tsx` line 171), which requires the manifest's
   `replay_scene_url` field — and `demo_manifest.json`'s `replay_scene_url`
   is `null` for this particular wolverine bundle. Toggling the "Point
   clouds" debug layer on (confirmed via screenshot,
   `runs/lanes/fps_mac_20260710/screenshot_meshdebug.png`) did not surface
   it either, and it changed FPS not at all (118.4 sampled with it on),
   which at least indicates the render loop was not GPU-bound at this scene
   complexity on the M1 Pro. A direct react-three-fiber scene-graph
   introspection attempt (`canvas.__r3f`) also came back `null` — this
   viewer's r3f build doesn't expose that hook on the DOM node. Getting a
   true mesh-count would require either a manifest with `replay_scene_url`
   populated, or a source-level change to expose renderer stats (out of
   scope for this measurement-only lane — no repo files outside the allowed
   staging/output paths were edited).
6. **Headed (visible-window) Chromium, not headless.** This required an
   active logged-in GUI session on the Mac, which was available. Not tested:
   whether `headless=True` with `--use-angle=metal` (also supported by the
   driver script's `"headless"` mode) would have produced the same real-GPU
   renderer string — headed mode worked on the first labeled attempt (after
   fixing the auth-bypass gating), so the headless-Metal fallback path was
   not exercised.
7. **Display-refresh dependency not cross-checked.** All numbers here are
   specific to this MacBook's own 120Hz ProMotion panel; no external 60Hz
   display was attached to see whether the same reading drops to ~60 (which
   the vsync-bound explanation in Honest Issue #1 predicts it should).

## Cleanup

- `pkill -f "vite --host 127.0.0.1 --port 5183"` run and verified (no
  matching process afterward).
- Playwright-launched Chromium instances were closed via `browser.close()`
  inside each script (both scripts ran to completion); verified no
  orphaned `chromium` process tied to this lane remained after cleanup
  (see verification below — any `Chromium`/headless_shell process left
  running belongs to other pre-existing sessions, not this lane).
- Did not touch the two other pre-existing Vite dev servers found running
  on ports 5199 and 5173 (belong to other lanes/worktrees, out of scope).
- No repo files were edited outside `runs/lanes/fps_mac_20260710/`. Nothing
  was staged into `web/replay/public/` (unnecessary — see "What was
  served" above). No commits made.

## Files produced

- `runs/lanes/fps_mac_20260710/REPORT.md` (this file)
- `runs/lanes/fps_mac_20260710/measure_fps.py` (main driver)
- `runs/lanes/fps_mac_20260710/measure_meshdebug.py` (follow-up debug-layer/network probe)
- `runs/lanes/fps_mac_20260710/raw_samples_headed.json` (full raw per-second samples + renderer info)
- `runs/lanes/fps_mac_20260710/summary_stats.json` (computed min/p50/p95/max/mean)
- `runs/lanes/fps_mac_20260710/meshdebug_extra.json` (mesh-debug/network follow-up, mostly null per Honest Issue #5)
- `runs/lanes/fps_mac_20260710/screenshot_headed.png`, `screenshot_meshdebug.png`
