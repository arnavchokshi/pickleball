# LANE w4_bridgefix_20260707 — scoring-bridge fixes: torch.load compat + explicit preprocessing mode

## HARD RULES
No git branches/commits/pushes. 4 protected clips EVAL-ONLY. Honest reporting. Run the full listed
blast radius, not a subset. No new root .md. Artifacts under runs/lanes/w4_bridgefix_20260707/.

## OBJECTIVE (two live-found bugs from the w4_ballgpu GPU lane — its PROGRESS.md 21:38Z update is
the evidence; runs/lanes/w4_ballgpu_20260707/PROGRESS.md)
1. TORCH.LOAD COMPAT: `wasb_adapter`'s checkpoint load path fails on torch 2.9 for lane-trained
   checkpoints (payload contains PosixPath; torch 2.9's weights_only default refuses). Fix the
   load path to handle both official and lane checkpoints safely (explicit weights_only handling
   or allowlisted safe globals; keep it minimal and boring). The GPU lane worked around it by
   sanitizing tensor-only copies — after your fix, unsanitized lane checkpoints must load.
2. EXPLICIT PREPROCESSING MODE: training-harness preprocessing (plain resize + /255,
   `threed/racketsport/roboflow_corpus.py:981` — re-grep at HEAD) differs from the bridge's
   official inference preprocessing (affine warp + ImageNet mean/std,
   `threed/racketsport/wasb_adapter.py::_preprocess_wasb_window`). Harness-trained checkpoints
   therefore produce DEGENERATE bridge scores (measured: F1 0.0 / hidden-FP 1.0 vs the official
   ckpt's 0.714/0.783 through the identical path). Add an explicit
   `--input-preprocessing {official,harness_v0}` mode to the checkpoint-inference path
   (`scripts/racketsport/run_wasb_ball.py` + the adapter): `official` = current behavior, DEFAULT,
   bit-identical outputs for the official checkpoint (prove it); `harness_v0` = the training
   harness's exact resize+/255 transform (import/replicate the SAME code the harness uses — no
   third variant), clearly marked NON-PROMOTABLE MEASUREMENT MODE. The chosen mode MUST be stamped
   into every output artifact (ball_track.json/candidates metadata: `input_preprocessing: ...`)
   so any downstream score card carries it — un-stamped artifacts are the failure mode to prevent.
   DO NOT change the training harness in this lane (making training match official preprocessing
   is a separate wave-5 lane — note it in `next`).

## EXPLICIT FILE OWNERSHIP
`threed/racketsport/wasb_adapter.py`, `scripts/racketsport/run_wasb_ball.py`,
`tests/racketsport/test_wasb_adapter.py`, `tests/racketsport/test_ball_wasb_dataset.py` (only if
touched by imports), + a new focused test file if cleaner. DO NOT TOUCH: `train_ball_pretrain.py`,
`train_ball_stage2.py`, `roboflow_corpus.py` (read-only), `ball_arc_solver.py`,
`run_ball_tracking_eval_suite.py`, `fuse_ball_tracks.py`, `process_video.py`/`orchestrator.py`
(fenced), `ios/**`, `runs/manager/**`, eval labels, ledger.

## ACCEPTANCE
1. Official-mode bit-identity: with the official tennis checkpoint
   (`models/checkpoints/wasb/wasb_tennis_best.pth.tar`, local, sha 9d391239…), default-mode
   outputs on a small fixture window are BIT-IDENTICAL pre/post your change (hash the output
   arrays in a test; if full bit-identity is impossible due to refactor, prove numerical identity
   to 0 ulp and say why).
2. torch.load: a synthetic lane-style checkpoint (state_dict + a PosixPath in the payload) loads
   without sanitization on torch 2.9. The official .pth.tar still loads.
3. harness_v0 mode: a tiny harness-trained checkpoint fixture (train 20 CPU steps in-test via the
   stage-2/pretrain plumbing, or a checked-in miniature) produces NON-degenerate output under
   harness_v0 (not a constant point) AND its output artifact carries
   `input_preprocessing: harness_v0`. Official mode artifacts carry `input_preprocessing: official`.
4. Full blast radius: `.venv/bin/python -m pytest tests/racketsport/test_wasb_adapter.py
   tests/racketsport/test_ball_wasb_dataset.py tests/racketsport/test_scaffold_tool_index.py -q`
   plus every test file your grep shows importing `wasb_adapter` or `run_wasb_ball` (list; run ALL).

## EVIDENCE TO READ FIRST
runs/lanes/w4_ballgpu_20260707/PROGRESS.md (21:38Z update); `wasb_adapter.py` load +
`_preprocess_wasb_window`; `roboflow_corpus.py:981` transform; TECH_BLUEPRINTS BALL-2D SCORING
BRIDGE section (the two-scoring-worlds rule — your flag makes the wall explicit, never blurs it).

## STRUCTURED REPORT
objective_result; acceptance table w/ the bit-identity proof + fixture scores; CHANGES file:line;
full_suite; honest_issues; next (the wave-5 harness-alignment lane note).
