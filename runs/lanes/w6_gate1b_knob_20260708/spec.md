# LANE w6_gate1b_knob_20260708 — build the BODY post-chain raw-persist knob (wave-6 queue #2, stage 1 of the P2-2 wiring ruling)

## HARD RULES (binding)
- NO git branches, NO commits, NO pushes. Working-tree changes only in your OWNED FILES. Manager commits at checkpoints.
- Do NOT edit BUILD_CHECKLIST.md or runs/manager/ boards — proposed bullet text goes in your report.
- Protected eval clips are EVAL-ONLY (Burlington/Wolverine internal scoring OK; Outdoor/Indoor LABELS never without a pre-registered heldout_eval_ledger.md row — STOP if a step seems to need them).
- Honest reporting; PASS with full_suite.failed>0 not proven pre-existing = rejected.
- .venv/bin/python; importorskip("torch") for torch tests; MPLBACKEND=Agg on wide runs; new CLI flags ship scaffold-index/direct-CLI reference-test updates same-lane.
- Artifacts under runs/lanes/w6_gate1b_knob_20260708/ ONLY. Other lanes' run dirs READ-ONLY.
- Blueprint/report line numbers have DRIFTED — re-grep symbol names at HEAD before every edit.
- NO GPU dispatch from this lane (sandbox has no network anyway). You BUILD the knob; a separate GPU lane runs the instrument dispatch.

## FILE OWNERSHIP (exclusive this wave)
- OWNED: process_video.py (SOLE owner this wave), scripts/racketsport/remote_body_dispatch.py, the BODY post-chain modules you identify (the temporal-smoothing / foot-lock / foot-pin / contact-splice / wrist-lock implementation + config surfaces, e.g. under threed/ body/world post-processing), plus their test files, plus runs/lanes/w6_gate1b_knob_20260708/**.
- DO NOT TOUCH: threed/racketsport/ball_arc_solver.py + its tests (w6_magnus lane), CAPABILITIES.md + BVP verify harness + scripts/racketsport/train_ball_stage2.py (w6_instrudocs lane), cvat_upload/** (w6_labelpack lane), web/replay/** (product-infra session fence), ios/** (owner).

## OBJECTIVE
Wave-5 PROVED the knob is absent (strict GATE-1b ruling, w5_closeproof grep evidence: only `--no-sam3d-wrist-bone-lock` exists; foot-lock / stance-smoothing / world_joint_visual_smoothing have ZERO CLI/env passthrough). Build the raw-persist knob so wave-6 can run the strict apples-to-apples GATE 1b recipe: instrument ONE clip's BODY dispatch with the post-chain OFF — temporal smoothing, foot-lock, foot-pin, contact-splice, wrist-lock — and persist raw grounded joints, so decode(emit) <=1mm and mesh-skel <=5mm p95 can be compared legitimately against the P2-2 latent decode path.

## EVIDENCE TO READ FIRST
1. runs/manager/wave6_boot_prompt.md (queue #2 + the "What wave-5 banked" P2-2 paragraph) — the ruling context.
2. runs/lanes/w5_closeproof_20260708/spec.md — the strict GATE-1b recipe + the knob-absent grep evidence.
3. runs/lanes/w5_p22latent_20260707/ (spec.md + vm_evidence/) — the P2-2 decode wrapper, GATE 1a/1b definitions, what "decode(emit)" means here.
4. runs/manager/w5_rider2_score/latent_smoothing_acceptance_report.md — the manager's scoring of the decoded lambda-sweep (why NO lambda was wiring-ready; the thin-extraction caveat).
5. Commit 2db0d1b4e (w5_p22wiring latent-smoothing acceptance harness CLI) — the harness that will consume your raw-persist output.
6. The existing `--no-sam3d-wrist-bone-lock` passthrough implementation — your pattern precedent; grep it end-to-end (CLI -> dispatch -> VM-side config -> stage).

## THE DESIGN (pinned WHAT; you own the HOW)
- One coherent knob family that can switch OFF each post-chain stage independently AND all-at-once (e.g. `--body-postchain raw` plus per-stage overrides, or five `--no-body-<stage>` flags mirroring the wrist-bone-lock precedent — pick ONE style consistent with the house pattern and say why).
- Thread it through the SAME path the existing wrist knob takes: process_video.py CLI -> orchestrator config -> remote_body_dispatch.py -> VM-side BODY stage config. The dispatch path must serialize the knobs into whatever config artifact the VM-side stage reads, so a remote dispatch honors them without code edits VM-side beyond what ships via code-sync.
- Raw persist: when the post-chain is fully OFF, persist the raw grounded joints as a first-class sidecar artifact (schema-versioned, fail-loud on missing fields) alongside the normal outputs — NOT a replacement of the default artifacts. Name it so the acceptance harness (2db0d1b4e) can consume it; document the field mapping in the lane report.
- LOUD, not silent: every bypassed stage MUST be recorded in the body runtime summary (follow the w5_transport loud-degrade precedent, commit baa7c911c) so a raw run can never masquerade as a default run. Strict-mode compatible.
- Default behavior with no flags = EXACTLY today's behavior (byte-identical artifacts on the deterministic CPU fixture).

## ACCEPTANCE (measured, all required)
1. Grep proof (in report): each of temporal-smoothing / foot-lock / foot-pin / contact-splice / wrist-lock has a CLI/env passthrough reachable from process_video.py AND from remote_body_dispatch.py — the exact inverse of the w5 knob-absent grep.
2. Pipeline-entry-point contract test: on the existing deterministic CPU-path fixture (find the one the smoke/contract tests use), running with post-chain OFF produces (a) raw joints that differ from default-processed joints, (b) the raw sidecar persisted + schema-valid, (c) bypass records present in the runtime summary. Running with NO flags produces byte-identical artifacts to HEAD behavior.
3. Per-stage unit tests: each knob verifiably bypasses its stage (not just a config echo).
4. Scaffold-index/reference tests updated for new CLI flags, same-lane.
5. FULL wide blast-radius suite green (MPLBACKEND=Agg; court benchmark may be split out standalone per house pattern); any failure proven pre-existing at HEAD or fixed.
6. Report the EXACT command line the follow-on GPU lane should run for the one-clip instrument dispatch (clip choice: wolverine_mixed_0200 — the P2-2 evidence clip), including the raw-persist output paths.

## KILL / STOP CRITERIA
- If a post-chain stage turns out to be structurally un-bypassable without breaking downstream consumers (e.g. grounding hard-depends on foot-lock output), do NOT force it: implement the bypass for the stages that can be clean, and report the dependency graph + a proposed design for the blocked stage as HONEST ISSUES + NEXT. That is a PARTIAL, not a failure.
- If process_video.py CLI surface changes would collide with the suffix-stage integration rule (TECH_BLUEPRINTS B.1.1), confine your edits to arg-parsing/config-threading — you are NOT inserting stages.

## REPORT (schema-enforced)
objective_result vs the 6 acceptance items; full_suite line; CHANGES file:line; the GPU instrument command; HONEST ISSUES; proposed BUILD_CHECKLIST bullet; NEXT.
