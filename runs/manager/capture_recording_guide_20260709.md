# Owner capture guide — what to record and how (2026-07-09)

Manager-verified against code + live infra on 2026-07-09. Companion to
NORTH_STAR_ROADMAP §NS-01/NS-02 and the two lanes
`runs/lanes/sidecar_contract_20260709/` + `runs/lanes/upload_wiring_20260709/`.

## A. What the app captures per clip (verified in Swift today)
- Video: AVCaptureMovieFileOutput, HEVC 1080p60 default (`standard60`); modes for
  120fps/1080p, 240fps/720p, 4K60 (tier-gated). Landscape ENFORCED (start throws otherwise).
- Audio: mandatory mic track (session throws without it) — bounce/contact audio is on every clip.
- Motion/AR: bounded ARKit pre-record setup pass (intrinsics/court plane attempt), CoreMotion
  gravity + per-frame samples during recording (provenance=coremotion_only).
- Sidecar: `capture_sidecar.json` written next to `clip.mov` on stop under
  `Documents/captures/<sessionID>/`; Replays tab lists local packages.
- Being fixed by today's lanes: sidecar validates server-side byte-for-byte (P0-A);
  record→upload call path + honest states (P0-B); camera-roll import button; state chips.

## B. Recording priorities (in order — this is the data engine)
1. **Route-proof traces (NS-01.2b, ~10 min total)** — after lanes land + manager flips accounts ON:
   one ~30-60s recording in the app → upload → confirm it shows Uploaded; one camera-roll import
   → upload. Any court, any quality. This is the physical gate evidence.
2. **Fresh in-domain games (the accuracy unlock)** — full games at your usual courts:
   tripod, landscape, FULL court + both baselines in frame, 1080p60, audio on (no wind cover),
   avoid backlit low sun if possible. Target: 2-4 full games across ≥2 different
   courts/sessions. These become the fresh untouched owner holdout + labeling corpus
   (NS-02.3-2.5). Do NOT reuse courts/angles already in the protected eval sets for the holdout.
3. **Gold-capture half-day (NS-02.1)** — LATER, once engineering ships the package/checklist:
   product phone + two high-FPS phones, surveyed court/net points, ChArUco board, LED/audio
   sync, paddle markers, scripted shots/occlusions. Do not attempt before the dry-run package.

## C. Practical notes
- Storage: 1080p60 HEVC ≈ 4-6 GB/hour. Keep ≥10 GB free for a full game. Clips persist in the
  app's Documents until deleted.
- Upload v1 is foreground: keep the app open while a game uploads (multi-GB can take a while
  on LTE; Wi-Fi recommended). Background upload is a booked follow-up.
- First run: record a 2-minute test, upload it, and confirm the chip goes Uploaded before
  committing a whole game.
- Sign-in: register in-app with the invite code (manager will stage it in the check-in when the
  route is live). Recording works signed-out; upload asks for sign-in.
