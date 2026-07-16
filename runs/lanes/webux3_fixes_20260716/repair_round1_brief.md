# Repair round 1 — manager browser verification results (Track H, 2026-07-16)

Your T1-T11 wave was verified by the manager in a real browser (headed Chromium, Metal ANGLE,
1600x1000, real bundles). Most of it PASSES and will be adopted:
- T1 PASS: video pane top y=225.75 (baseline 786); upload collapsed below workspace.
- Suite re-verified locally by manager: vitest 273/273 EXIT 0, typecheck EXIT 0, build EXIT 0.
- T3/T5/T6/T7/T8 browser-verified PASS: single scrub + 168 legible markers + rally band + inline
  degraded reason on the empty degraded timeline; in-pane per-entity badges; consolidated absence
  chip; expanded warnings incl. structured degraded reasons. Zero page errors on both bundles.
- T2 partial: PLAYING at Court preset improved 37.5 → 47.7 avg rAF fps (n=3: 47.2/50.2/45.5). PASS
  on that bar.

TWO FAILURES remain. Fix them in this resume. Same fence (web/replay/** + this lane dir only),
same bars (vitest+typecheck+build all EXIT 0, zero failures), no regression to the adopted items.

## FAIL 1 — T2 Follow-player playback FPS (acceptance: >=80% of Court-playing)
Measured n=3 while video playing, follow preset: 20.5 / 20.5 / 12.75 → avg 17.9 vs court avg 47.7
(37.5%). Paused follow reads 97.4 on the app badge (court paused ~119), so fill-rate is a minor
factor; the collapse is playback-specific per-frame work that only runs (or runs much hotter) in
follow mode. PROFILE FIRST, do not guess: instrument the follow path (temporary perf accumulators
are fine), identify the dominant per-frame cost while playing (candidates: per-frame follow-target
pose lookup doing linear scans over world frames each rAF; per-frame allocations (Vector3/etc.)
causing GC churn; camera pose/effect churn re-entering React per frame; controls.update interplay;
duplicated interpolation work per rendered frame). State the measured cost breakdown in the report,
fix the dominant cost, and add a unit test pinning the invariant you fixed (e.g. follow-target
lookup is memoized/O(log n) per query, or steady-state follow tick performs zero new allocations
where testable). Manager re-measures in browser with the same harness afterwards.

## FAIL 2 — T10 recovery never fires in a real browser (stretch made non-stretch for this round)
Real repro: load
`?manifest=/@fs/Users/arnavchokshi/Desktop/pickleball/runs/lanes/demo_beststack_render_20260710/fresh_wolv/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json`
(local, valid JSON manifest whose ASSET urls are dead VM-absolute `/@fs//home/arnavchokshi/...`).
Observed: red banner `Unexpected token '<', "<!doctype "... is not valid JSON`, viewer never opens,
falls back to the VIDEO INTAKE screen; NO recovery attempt, NO plain-language error. Diagnosis to
verify: the Vite dev server answers unservable `/@fs//home/...` requests with an HTML document
(likely non-2xx or even 200 html), your loaders call `.json()` without an `ok`/content-type guard,
so your recovery predicate (network failure) never triggers and the raw JSON.parse error surfaces.
Required:
- Recovery predicate must treat as failure: `!res.ok` OR non-JSON content-type OR body starting
  with `<` on a JSON asset — then run the same-basename manifest-relative retry you built (with the
  loud counted notice, unchanged semantics).
- When recovery is impossible/fails, the surfaced error must name the failing asset URL in plain
  language (e.g. "asset unreachable: /@fs//home/... — this manifest appears to have been written on
  another machine") instead of a JSON token error.
- Unit tests: mocked fetch returning an HTML-body response (200 and 403 variants) for an asset →
  recovery fires; manifest-level fetch failing the same way → plain-language error string.
Manager will re-run the real VM manifest in the browser; acceptance = viewer opens on that URL with
the loud recovery notice visible, or (if an asset genuinely cannot be found beside the manifest) a
plain-language error naming the URL.

## Cosmetic (only if fast, do not risk the above)
Degraded bundle view: right-edge collision between the Ball trust badge ("NO TRUST BAND") and the
collapsed unavailable-layers chip (see
runs/lanes/webux3_fixes_20260716/shots_after/after_08_degraded_bottom.png). Badges and the chip
should never overlap; wrap or stack instead.

Update report.json at the same path (schema unchanged) with the repair-round results appended to
`changes`/`acceptance` and honest issues refreshed. Run the full suite again before finishing.
