# Track H friction audit — web replay viewer (2026-07-16)

Manager-driven browser audit (Playwright 1.61.1, Chromium headed `--use-angle=metal` +
headless recon), against two REAL bundles (no fixtures):

- **Rich**: `runs/lanes/demo_beststack_render_20260710/fresh_wolv/wolverine_mixed_0200_mid_steep_corner/`
  via path-translated manifest `runs/lanes/webux3_audit_20260716/manifest_fresh_wolv_local.json`
  (translation = VM→local path prefix rewrite only; all 10 asset URLs verified on disk; note appended
  into the manifest for provenance).
- **Degraded**: `runs/lanes/trk_flip_20260713/default_production/wolverine_mixed_0200_mid_steep_corner/`
  (no BODY, no ball track, no coaching facts — honest-absence behavior probe).

Viewport 1600x1000. Screenshots in `shots/` (recon_*, int_hd_*, degraded_*).
Interaction transcripts: `recon_result.json`, `interact_hd_result.json`.

## Baseline numbers (headed Chromium, Metal ANGLE, 1600x1000)

| State | Measurement |
|---|---|
| Paused, Court preset, all default layers | app FPS badge ~118.6–119.8 |
| PLAYING, Court preset (default "2x FPS interpolated") | **37.5 rAF fps** (4s window) |
| PLAYING, Follow-player preset | **11.25 rAF fps** (4s window) — 3.3x collapse |
| Headless/SwiftShader (worst-case ref) | badge 5.8–6.2 |
| Cold load wall (dev server, manifest → panes interactive) | ~7.8s incl. 262MB bundle assets |

## Ranked friction list

**F1 (highest, hit every session) — replay workspace is below the fold / hierarchy inverted.**
Video pane starts at y=786 on a 1000px viewport (page height 1401): the reviewer lands on an
upload form ("Choose File", giant "Predict Court" button, "PIPELINE PROGRESS 0%", "Waiting for
upload", "No upload queued") plus a "Latest video runs" strip BEFORE any replay content. The most
visually prominent element on a replay-review screen is an upload button irrelevant to the loaded
bundle. Video+3D+timeline only co-exist on screen after manual scroll. Evidence: recon_01, geometry
in interact_hd_result.json.

**F2 — playback frame rate collapses; Follow preset catastrophic.**
37.5 fps playing vs ~119 paused; 11.25 fps at Follow. Named code suspect (App.tsx ~line 958-976):
the rAF loop calls `setCurrentTime()` up to video-fps rate; `currentTime` is App-level React state,
so every sampled frame re-renders the entire 3843-line component and recomputes every
currentTime-keyed useMemo (`activeBodyMeshes` → `solidBodyMeshFramesForTime`, `ballRenderInfo`,
`videoBallOverlay`, `activeContactPlayerIds`, `activePaddles`, arc samples...). Follow preset adds
per-render camera pose work (poseKey effect churn). Fix direction: playback time via ref +
subscription (scene reads in useFrame; UI chips throttled ~5Hz), reuse geometry buffers, memoized
frame lookup (binary search), avoid per-frame allocation. Evidence: int_hd_01/09, code read.

**F3 — two timelines; markers illegible; empty timeline unexplained.**
Native video controls carry the primary scrub while the event strip is a separate thin dot-row at
the very page bottom ("Rally" row label clipped). §1.3 requires ONE shared timeline. All markers
render as near-identical small orange dots — provenance legend (Measured/Model estimated/Physics
predicted) exists but is indistinguishable at marker size; no visible type distinction
(contact/bounce/inflection); no hover tooltip surfaced in audit. Degraded bundle: strip renders
fully empty with no reason, while the manifest's degraded_reasons (ball_fill blocked, body degraded,
coaching_facts degraded) stay buried behind a "+1 more" warnings chip (F8). Evidence: int_hd_08,
degraded_01.

**F4 — camera presets waste the pane / cut content.**
Court preset: court occupies well under half the 720px pane, near-court players cut at the bottom
edge in default framing. Follow preset: mostly empty court visible, target player small;
"Player 19: no detection / Player 21: no detection" toast while following P19 — follow target
defaults to a player with no detection at t. Evidence: recon_01, int_hd_02_preset_*.

**F5 — trust badges detached from entities (§1.3: every visible entity carries a compact badge).**
Trust chips form a 2-row band at page top, far from the 3D entities they describe. In-pane the
entities carry no badge. Badges must move INTO the replay surface (persistent compact per-entity
badges/legend in the 3D pane), and may not become less prominent than today.

**F6 — honest-absence toasts stack unbounded in the 3D pane corner.**
Four persistent stacked toasts ("Ball trail: no data at this time", "Contact surfaces: artifact not
supplied", "Target zones: artifact not supplied", "Ghost positioning: artifact not supplied")
overlap the scene. Keep always-visible honesty, but consolidate into one compact expandable
absent-layers chip with per-layer reasons. Evidence: int_hd_05_isolate_*, degraded_01.

**F7 — three ball-status vocabularies; contacts chip ambiguity.**
Top stat chip "Ball: not visible" vs status line "ball: video high" vs trust chip "Ball — BALL: LOW
CONFIDENCE" — three surfaces, three vocabularies, no tooltips. "Contacts 0" chip while status line
says "3D contact: p20, p21, p22". Unify vocabulary + explain each chip (what, at-current-time vs
whole-clip).

**F8 — warnings/degraded reasons unreadable.**
"3 notices: ... +1 more" has no expansion; manifest notes carry structured degraded_reasons
(stage/status/reasons) that never reach the reviewer. Click-to-expand full list.

**F9 — Shots panel empty state confuses ("0/0 shots / 0 ARC TRAILS / ARC SOURCE LOADED / No shot
selected" + 4 filter dropdowns for nothing).** Opening it also inserts a black void above the video
pane (layout jank, see int_hd_06_tab_Court_map.png left pane). Honest one-liner + hidden filters
when zero; fix the reflow.

**F10 — dev load-flow trap.** README quickstart URL (`?manifest=...`) silently lands on the sign-in
screen unless `VITE_REPLAY_VERIFY_DEV_BYPASS=1` was exported; the manifest param is swallowed. On
loopback+dev+manifest-param, show an explicit hint instead of a silent sign-in; document the env in
web/replay/README.md.

**F11 — pulled-from-VM bundles don't load (dead absolute /@fs URLs).**
All three richest recent bundles carry VM-absolute paths (`/home/arnavchokshi/...`) while every
asset sits beside the manifest locally. Manager unblocked audit via manual path translation. Viewer
should, only on fetch failure of an absolute /@fs asset URL, retry basename-beside-manifest (and
declared subdirs) with a LOUD visible "resolved manifest-relative (original path unreachable)"
notice chip — honest, no content substitution, kills a manual step for every pulled bundle.

**F12 (minor)** — "Follow P19" dropdown vocabulary vs "Player 19" everywhere else; playback-rate
button labels inconsistent for programmatic/a11y access ("original" failed role-name exact query);
FPS badge visibility.

## What already works (do not regress)
Proportional timeline scrub + marker exact-PTS + keyboard nav (wave-2 fixes); entity toggle set;
camera preset persistence + reduced-motion cuts; honest empty states and per-layer absence
reporting (presence semantics must survive any consolidation); trust pills; degraded bundle loads
without errors; 0 console/page errors on both bundles.
