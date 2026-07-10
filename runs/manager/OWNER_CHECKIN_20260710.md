# Owner check-in — 2026-07-10 (deep-review session)

⭐ **Your five video complaints are all diagnosed, three have fixes already running, and one big
one was partly an illusion: the demo video itself was assembled at 10 real frames/s (screenshot
stepping), so it understated the product. Full rulings: `runs/research_deepreview_20260710/RULINGS.md`.**

What each symptom actually was:
1. **Frame rate** — 10fps demo assembly (dominant) + no interpolation between mesh keyframes +
   unknown real-device fps (all our fps numbers were software-renderer artifacts). Viewer fix lane
   running; needs your 10-min phone test (ask #1).
2. **People missing** — detector sees ≥4 people on ~97% of frames; we then delete them: zero-margin
   court filter (feet behind the baseline = dropped!), a **silently-skipped ReID step** (its model
   file was missing and the pipeline just shrugged), and top-4 selection fragmenting identities.
   Pipeline fail-loud fix running; scored TRK recovery lane specced next.
3. **Skeleton-or-mesh rule** — the viewer fallback ALREADY exists; your clip had zero BODY output
   from the cold-clip bug (fixed this morning, ns016) and the artifacts lied ("skeleton_only" when
   nothing existed). Placeholder tier + honest labels in the running fix lanes.
4. **Ball hidden** — that's the new fail-closed honesty doing its job on junk 3D; the real waste:
   71-91% of good 2D ball detections never become 3D. pb.vision comparison says our 2D coverage
   BEATS theirs (80.6% vs 58.7%) — and their "3D" product is actually a static shot chart, not a
   live replay. Plan: predicted-band ball styling (running), then UKF gap-fill + anchor-search
   solver (specced, GPU).
5. **Paddle** — it's a wrist-glued estimate (real fix needs the gold-capture GT), PLUS the viewer
   held stale poses across 1.5s gaps and drew a debug arrow BIGGER than the paddle. Presentation
   fixes running now; accuracy waits on capture.

**Blockers:** none hard.

**Verify when back / asks (numbered, easiest-first):**
1. 10-min phone test: I'll stage a Wolverine replay URL — open it on your iPhone so we get the
   first-ever real-device fps number.
2. pb.vision: which source clip was the banked cv export from? (Enables exact-clip benchmarking.)
3. Gold capture half-day (standing) — paddle/ball-3D accuracy are capped until then.

**Money/GPU:** $0 GPU this session (audit + fixes are Codex/CPU). Court wave's VM is theirs.

**Overnight log:** 3 Codex audit lanes + 45-agent internal audit + adversarial verify; 2 Codex fix
lanes (viewer, pipeline) dispatched and in flight; North Star pointer refreshed (498 lines, tests
green); nothing promoted, VERIFIED=0 unchanged.
