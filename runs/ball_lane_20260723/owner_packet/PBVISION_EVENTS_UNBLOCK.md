# Unblocking the 4,637 pb.vision teacher events

Asset: `pbv_pickleball_teacher_events_20260720` — 4,637 teacher events / 7,314 projected windows,
state **BLOCKED** ("0/7314 windows locally decodable"; a recovered materialization accepted 292
audio-only rows, invalid). EVENT ruling: "may train only after non-audio agreement and local
media/PTS hash gates pass" (`runs/manager/data_ledger.json`). `VERIFIED=0`; nothing here is done yet.

## Holdout rule (binding, restate first)
**`83gyqyc10y8f`, `iottnc0h3ekn`, `o4dee9dn0ccr` stay compare-only — permanently.** They are
excluded from the teacher corpus and denylisted in the rebuilt builder (synthetic leakage test
passed, `runs/lanes/pbv_corpus_rebuild_20260720/REPORT.md`). Nothing in this checklist pulls them
into training; their media is needed only if compare cards are wanted, never for the corpus.

## Local inventory (verified on disk, `data/pbvision_gallery_20260719/`)
- **All 13 video IDs have `insights.json` + `cv_export.json` locally** — 12 in per-ID dirs under the
  gallery, and `83gyqyc10y8f` under `data/pbvision_11min_20260713/`. JSON exports are NOT the blocker.
- **Local pixels: 2/13** — `83gyqyc10y8f` (`data/pbvision_11min_20260713/source_video.mp4`,
  compare-only) and `xkadsq9bli3h` (`data/pbv_replay_20260720/xkadsq9bli3h/max.mp4`, SHA-matched,
  decode-verified at corrected frame 134).

## Media that must be pulled (9 IDs, 1.83 GB total — unverified, on VM)
`MANIFEST.json` records every file at `vm:pickleball-gpu-evhead:~/pbv_gallery/<id>/max.mp4` with
sha256 + size. **VM contents unverified locally**; the owner check-in (2026-07-23) says ball
VMs/disks were torn down — whether `pickleball-gpu-evhead` still exists needs a fleet check first.

| video_id | bytes | | video_id | bytes |
|---|---|---|---|---|
| 0tmdeghtfvjx | 222,315,614 | | st0epgnab7dr | 271,661,642 |
| 143sf3gdwxsa | 152,436,118 | | td2szayjwtrj | 122,096,753 |
| 98z43hspqz13 | 180,321,865 | | tqjlrcntpjvt | 305,557,352 |
| bewqc0glhgpq | 112,184,212 | | utasf5hnozwz | 175,476,682 |
| pldtjpw3h0jw | 291,343,243 | | **total (9)** | **1,833,393,481** |

(`iottnc0h3ekn` 203,665,387 + `o4dee9dn0ccr` 291,958,287 also sit on the VM — compare-only, do not
stage for training.)

## Required PTS artifact
Per `runs/lanes/pbv_corpus_rebuild_20260720/REPORT.md` blocker #1: for each training source, a
**monotonic encoded-PTS table derived from the SHA-matched local MP4** (ffprobe over the exact
staged file), SHA-bound into the corpus rebuild, until every training source has
`needs_pts_verify=false`. Teacher frame indices map to source frames by nearest encoded PTS (the
xkadsq9bli3h proof: teacher frame 67 @ 2.233333 s → source frame 134, decode verified) — nominal-fps
arithmetic is not acceptable.

## Ordered unblock procedure
1. **Fleet check**: does `pickleball-gpu-evhead` (or its disk) still exist? If yes → step 2. If no →
   re-obtain each MP4 from its recorded pb.vision share URL and verify against the MANIFEST sha256
   (the hashes make re-pull safe).
2. **Stage the 9 training MP4s** locally (~1.84 GB; add the paths to the storage allowlist first —
   the rebuild report already flags a storage-policy failure for the one staged MP4).
3. **SHA-256 verify** all 9 against `MANIFEST.json` `video_sha256`. Any mismatch stops that source.
4. **Derive per-source monotonic encoded PTS** and rebuild the corpus until all 10 training sources
   (9 pulled + xkadsq9bli3h) have `needs_pts_verify=false`.
5. **Run the deterministic non-audio agreement procedure** exactly per `ABC_STAGE.md`
   (`runs/lanes/pbv_corpus_rebuild_20260720/`): agreement count 0/1/≥2 → weight 0/0.25/0.5;
   pb.vision confidence alone earns zero; the 292 audio-only rows stay rejected.
6. **Apply + test the loader hunk** `runs/lanes/pbv_corpus_rebuild_20260720/LOADER_CHANGE_REQUIRED.diff`
   (per-frame UNKNOWN masks into loss) against the current `datasets.py` / `finetune_event_head.py`.
7. **Fresh ULTRA re-review** of code, corpus, derivatives, loader, and A/B/C inputs. Until it
   passes: `training_ready=false`, zero training-eligible events.

## Ledger status change this enables
`pbv_pickleball_teacher_events_20260720`: **BLOCKED → CONDITIONAL-consumable for EVENT training**
(teacher-weighted rows only, never human GT), with the 7/2/1 source-disjoint split and the three
compare-only IDs untouched. The `pbvision_gallery_20260719` row's `next_check` ("stage and hash the
eleven VM-only MP4s") is satisfied for the 9 training IDs at step 4.
