# LANE w3_codesync_20260707 — BODY-dispatch code-sync + version-stamp hardening (wave-3 #0, PREREQ)

## HARD RULES (violating any = lane rejected)
- You are Codex lane `w3_codesync_20260707` for the pickleball manager. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- No git branches; no `git commit`; no `git push`. Write `runs/lanes/w3_codesync_20260707/commit_manifest.md` (files + one-line message) for the manager instead.
- OWNED files (edit only these; create new files only under scripts/fleet/, tests/racketsport/, runs/lanes/w3_codesync_20260707/):
  `scripts/racketsport/remote_body_dispatch.py`, `scripts/fleet/*`, `tests/racketsport/test_remote_body_dispatch.py`, plus new test files for what you add.
- FENCED (other wave-3 lanes own these NOW — do NOT edit; propose diffs in your report if needed):
  `scripts/racketsport/process_video.py`, `threed/racketsport/camera_motion.py`, `scripts/racketsport/estimate_camera_motion.py`, `tests/racketsport/test_camera_motion.py`, `tests/racketsport/test_process_video.py`, `threed/racketsport/roboflow_corpus.py`, `threed/racketsport/ball_tracknet_cvat_dataset.py`.
  If the stamp echo genuinely requires a process_video.py change, implement everything else, and put the exact process_video diff in your report as a proposed patch file under runs/lanes/w3_codesync_20260707/deferred_patches/.
- Protected data: 4 eval clips EVAL-ONLY (Burlington/Wolverine internal-val OK; Outdoor/Indoor labels NEVER). Held-out harvest videos pwxNwFfYQlQ / vQhtz8l6VqU appear NOWHERE.
- Honest reporting: measured numbers only; PARTIAL/BLOCKED with evidence beats dressed-up PASS.
- Self-verify before done: WIDE suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q --ignore=tests/racketsport/court_finding_technology_benchmark.py` (benchmark = 22min standalone, manager adjudicates separately) + every focused suite you touched. Fix every failure you introduced incl. adjacent suites + real (non-fixture) paths. Classify residuals: REAL / PRE-EXISTING (prove at HEAD) / SANDBOX-SUSPECT (socket-bind, MPS, .git-lock) / CROSS-LANE-SUSPECT (name the lane).
- importorskip("torch") for torch tests; any new CLI ships a direct-CLI reference test same-lane; no new root-level .md. Always `.venv/bin/python`. Artifacts under runs/lanes/w3_codesync_20260707/.
- Read first: NORTH_STAR_ROADMAP.md PART 0 + Part IV; BUILD_CHECKLIST.md last ~15 bullets (esp. [MAD A/B RULED 2026-07-07] and [WAVE-2 COMPLETE 2026-07-07]).

## OBJECTIVE
Wave-2 postmortem: fleet VM `fleet1`'s repo checkout sat 16 COMMITS STALE (pinned at 5b9f132ee) through the entire wave because BODY dispatch ships DATA, never code — every remote-computed metric ran old code and survived only because the relevant files happened to be md5-identical. This is the standing failure class "offline validation of VM-side behavior is doubly blind." Your job: make it STRUCTURALLY IMPOSSIBLE to trust a remote-computed metric from drifted code — fail-loud version verification in the dispatch path, plus a supported sync path.

## THE DESIGN (manager-ruled shape; implementation details yours)
1. **Version stamp at dispatch**: when `remote_body_dispatch.py` prepares a remote run, compute locally: `{git_head_sha, git_dirty (bool + dirty file list for tracked runtime files), md5s of the remote-runtime-critical file set}`. The critical file set must be derived from what the remote side actually executes (at minimum the BODY runtime entry points the VM runs — discover the real list from the dispatch/payload code; do not hardcode a guess without checking what runs remotely). Write the stamp into the payload AND to the local run dir as `version_stamp.json`.
2. **Remote-side verification before compute**: before the remote BODY run starts, verify the remote checkout's sha + md5s against the stamp. ANY mismatch → hard fail, nonzero exit, error text naming every drifted file and both shas. No warn-and-proceed mode. A `--allow-dirty` escape hatch may exist for local dev but must be OFF by default and its use must be recorded in the stamp.
3. **Stamp echo into results**: the stamp (sha + verified=true + timestamp) must land in the remote run's output artifacts that come back (whatever BODY-run summary/provenance artifact the dispatch already returns — extend it), so any metric artifact is traceable to the code version that produced it.
4. **Sync path**: a supported command (flag on remote_body_dispatch.py or a scripts/fleet/ helper) that brings the remote checkout to local HEAD (repo files only — NEVER touch the VM's vendor pins, checkpoints dir, symlinks, or ~/coldstart_20260706; read the gpu_fleet.md fleet1 notes for what lives outside the repo). It must re-verify stamps after sync and refuse to sync a dirty local tree without explicit ack.
5. **No network in your sandbox**: you cannot reach the VM. All remote behavior must be implemented so the remote-side check runs via the same channel dispatch already uses (ssh command layer), and TESTED via local fixtures: simulate "remote" as a second local directory with a manipulated checkout (e.g. checkout an older sha into a temp clone, or alter one file). The Sonnet fleet lane will live-prove on the real VM later — your tests must make that boring.

## ACCEPTANCE (all must hold)
- Forced-drift fixture test: remote-fixture one commit behind OR one file altered → dispatch verification FAILS with nonzero status and the drifted file named in the error. Matched fixture → passes and stamp echo present in outputs.
- Dirty-local-tree test: dirty tracked runtime file → refused (or stamped as dirty with explicit --allow-dirty only).
- Sync-path fixture test: stale remote fixture → sync → verification passes; vendor-pin-style paths (fixture equivalents) untouched.
- Existing remote_body_dispatch tests still green (extend, don't regress: this file has substantial wave-2 tar_batch coverage in tests/racketsport/test_remote_body_dispatch.py — 280 lines changed there in wave 2).
- WIDE suite green per HARD RULES.

## EVIDENCE TO READ FIRST
- runs/lanes/wave2_mad_ab_20260707/ (the drift discovery + how the A/B lane hand-synced).
- runs/manager/gpu_fleet.md (fleet1 row: what lives on the VM outside the repo — checkpoints symlink, coldstart dir, vendor pins).
- scripts/racketsport/remote_body_dispatch.py + tests/racketsport/test_remote_body_dispatch.py at HEAD (tar_batch mechanics you must not break).
- BUILD_CHECKLIST [ROOTJUMP VERIFY RULED] bullet (why remote-computed metrics are the only proof for gated metrics — the reason this lane exists).

## SELF-ITERATION + BOUNDED FIX AUTHORITY
Iterate until all acceptance items pass or you hit a genuine blocker; report blockers with evidence, never paper over. You may refactor within remote_body_dispatch.py freely; you may add scripts/fleet/ helpers; you may NOT change fenced files (propose deferred patches instead).

## STRUCTURED REPORT (your final message → report.json via --output-schema)
objective_result PASS/PARTIAL/BLOCKED vs the acceptance list; acceptance table (item / measured / verdict); changes (file:line one-liners); full_suite counts + residual-failure classification (PASS with failed>0 and not-all-preexisting = auto-reject); HONEST ISSUES; NEXT; commit_manifest path; dated BUILD_CHECKLIST bullet DRAFT (do not edit BUILD_CHECKLIST.md itself — put the draft bullet text in the report; manager books it).
