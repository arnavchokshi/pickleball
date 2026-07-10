# LANE ns014_p22residual fix-round 2 — py3.10 portability + attribution input-coherence guard

Codex micro-lane. Two surgical fixes, self-verified. Same HARD RULES as the parent lane spec
(runs/lanes/ns014_p22residual_20260709/spec.md): no branches/commits; fenced files untouched
(worldhmr.py, process_video.py, remote_body_dispatch.py, all root .md, ios/**, .gitignore, etc.);
honest reporting; artifacts under runs/lanes/ns014_p22residual_20260709/fix2/.

OWNED FILES (only these): threed/racketsport/coordinates.py,
scripts/racketsport/attribute_body_decode_residual.py,
tests/racketsport/test_coordinates_api.py, tests/racketsport/test_attribute_body_decode_residual.py,
runs/lanes/ns014_p22residual_20260709/fix2/**.

## FIX 1 — coordinates.py must import on Python 3.10
Live failure: fleet VM venvs are Python 3.10.12; `from enum import StrEnum` is 3.11+. Replace with a
3.10-compatible construct (e.g. `class CoordinateSpace(str, enum.Enum)`) preserving ALL current
member names/values and str-ness semantics used by consumers (grep the consumers first:
mhr_decode.py, gate_check_body_decode.py, synthetic_body_decode_gate.py,
attribute_body_decode_residual.py, tests). No behavior change on 3.11+. Add a test that the module
contains no `StrEnum` import (regex on source) and that members compare equal to their string values.

## FIX 2 — attribution CLI refuses incoherent raw-index/run pairings (fail-closed)
Live failure: on the GPU VM the CLI was pointed (explicit --sam3d-output-index) at a STALE chunk
index from a DIFFERENT inference execution than the body_mesh.json being scored; the resulting
replay numbers (p95 527mm) were confounded garbage. Fix in attribute_body_decode_residual.py:
- Coherence check before any measurement: the loaded raw records' request_id set must cover the
  body_mesh player-frames being scored (frame_idx:player_id coverage >= a threshold, default 100%
  of frames selected for measurement), AND if both the index and the run dir carry execution/
  provenance stamps (timestamps, batch ids, out_path), any direct mismatch is an error.
- On failure: report status `incoherent_inputs` with the exact evidence (missing request_id count,
  first 10 examples, stamp mismatches), exit nonzero, and compute NOTHING downstream of the check.
- `--allow-incoherent-inputs` opt-out flag for forensics that stamps
  `inputs_coherent: false` + the same evidence into every result block it still computes.
- Auto-discovery ambiguity (2+ candidate chunk dirs) must remain a hard error listing candidates.
- Tests: coherent fixture passes (banked 32-record fixture with request ids restricted — keep the
  existing test green, adapting only if the coverage threshold needs the restricted-id path);
  incoherent synthetic fixture -> `incoherent_inputs` + nonzero exit; opt-out flag stamps false.

## ACCEPTANCE
- MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_coordinates_api.py
  tests/racketsport/test_attribute_body_decode_residual.py tests/racketsport/test_mhr_decode.py
  tests/racketsport/test_gate_check_body_decode.py tests/racketsport/test_synthetic_body_decode_gate.py -q
  -> 0 failures.
- Re-run the banked-fixture attribution command from the parent lane (same invocation as
  runs/lanes/ns014_p22residual_20260709/attribution_banked_fixture.log) -> grounding determinism
  numbers unchanged (p95 <= 1mm, machine-precision expected); save to fix2/.
- python3.10 compatibility: prove by AST/regex (no StrEnum) + `python3 -c "import ast;
  ast.parse(open('threed/racketsport/coordinates.py').read())"`; do NOT claim a 3.10 interpreter
  test if none exists on this Mac — say so.
- BEST-STACK DELTA: (c) none — portability/guard fixes only; thresholds untouched.
Report: schema report.json via harness; honest issues; wide-suite NOT required (blast radius is the
5 focused files) but run tests/racketsport/test_worldhmr.py too as a canary.
