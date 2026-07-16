# NS-01.2b physical-trace prep — execute the moment on-device recording works

Gate (North Star NS-01.2b): saved 30-second traces proving record/import -> upload -> job ->
artifact -> own replay with auth. HONEST SCOPE CAVEAT: full gate also needs NS-01.2a restart-safe
multipart + ready-manifest routing (P0-B, Track C surface). What Track D can bank NOW on first
working recording, without touching Track C's fence:

## Trace capture procedure (single 30 s session)
1. Terminal A — console + RecordPath/Upload diagnostics for the whole session:
   ```bash
   xcrun devicectl device process launch --device B03696B6-6481-5FCD-A79C-105DA3F08F98 --terminate-existing --console com.arnavchokshi.pickleball 2>&1 | tee runs/lanes/ios_recordpath_20260715/device_build/ns012b_console.log
   ```
2. Owner: landscape, tap record, 30 s of play (or any scene), tap stop, wait for "saved", open the
   Replays tab, tap the new row once (upload/queue state visible), leave app foreground 60 s.
3. Pull the app data container (capture package + sidecar + upload queue state) — precedent:
   runs/ios_device_gate_20260702T025809Z used `--domain-type appDataContainer`:
   ```bash
   xcrun devicectl device copy from --device B03696B6-6481-5FCD-A79C-105DA3F08F98 --domain-type appDataContainer --domain-identifier com.arnavchokshi.pickleball --source Documents --destination runs/lanes/ios_recordpath_20260715/device_build/ns012b_container_pull
   ```
4. Bank immediately (identity chain, per NS-01.1/01.3 discipline):
   - sha256 + size + duration of the recorded .mov; sidecar JSON version + sha256;
   - capture package directory listing; upload queue row (clip id / attempt state);
   - if server reachable + signed in: job id + status transitions from the console log;
   - all into runs/lanes/<current-lane>/ns012b_trace/ with a one-page TRACE.md.
5. Rule honestly: record->package->sidecar hop = Track D evidence; upload/job/manifest/replay hops
   = bank whatever the current wiring produces, label partial vs P0-B, DO NOT claim the NS-01.2b
   gate unless every hop including replay-own-run routing succeeded with auth.

Prereqs checklist before calling it a gate attempt: owner signed in (SignInView), server endpoint
reachable from the phone's network, Track C's honest-status decode on current build (it is — wave
builds are current main), and the video/sidecar pair preserved raw (no re-encode).
