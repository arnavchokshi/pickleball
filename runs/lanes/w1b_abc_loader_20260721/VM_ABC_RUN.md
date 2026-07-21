# VM A/B/C frozen execution runbook

Status: `VERIFIED=0`. This is the chronological A/B/C staging, training,
owner-41 selection, and final protected-50 procedure. Development and dry runs
must use fixtures/synthetic inputs only. No protected label, scorer, or result
may be opened before section 9.

## 1. Freeze media identity before deriving anything

For every train-split source, stage the MP4 at an explicit path and verify its
SHA-256 against `row.source_video_sha256` before decode:

```bash
sha256sum "$MEDIA_PATH"
# Must equal row.source_video_sha256 exactly.
```

Stop on a missing file, decode failure, or mismatch. Never rewrite a manifest
to fit an unexpected file.

## 2. Bind encoded PTS to the staged media

Build monotonic `frame_times.json` from that MP4. The artifact must declare the
same `source_video_sha256`. Rebuild the corpus so each train row contains
`timebase_conversion.pts_media_binding` with:

- `status=sha256_bound`;
- the staged media SHA;
- the exact frame-times artifact SHA; and
- `binding_sha256=sha256(canonical(media_sha, frame_times_sha))`.

```bash
sha256sum "$MEDIA_PATH" "$FRAME_TIMES_PATH"
.venv/bin/python scripts/racketsport/build_pbvision_event_corpus.py \
  --input-root data/pbvision_gallery_20260719 \
  --media-root "$MEDIA_ROOT" \
  --frame-times-root "$FRAME_TIMES_ROOT" \
  --output-dir "$CORPUS_OUT"
```

Stop if a train row has `needs_pts_verify=true`, if the frame-times JSON omits
the media SHA, or if the row binding does not recompute exactly.

## 3. Bind audio derivatives to both media and PTS

```bash
.venv/bin/python scripts/racketsport/build_audio_onsets_v2.py \
  --input "$MEDIA_PATH" --clip "$VIDEO_ID" --frame-rate "$FPS" \
  --frame-times "$FRAME_TIMES_PATH" \
  --out "$AUDIO_ONSETS_PATH"
sha256sum "$MEDIA_PATH" "$FRAME_TIMES_PATH" "$AUDIO_ONSETS_PATH"
```

Before agreement, the audio JSON must declare the exact
`source_video_sha256` and `frame_times_sha256`. The materializer rejects either
field when absent or mismatched and records a canonical dependency binding over
`audio_sha + media_sha + frame_times_sha`. Audio remains non-emitting evidence;
it never supplies HIT/BOUNCE class identity.

## 4. Bind the BALL derivative chain at every hop

Build the 2D track from the same media/PTS pair, then build image-space kinks:

```bash
# Produce $BALL_TRACK_PATH with the manager-frozen BALL-2D command.
# Its manifest must declare the exact media and frame-times SHA values.
.venv/bin/python scripts/racketsport/build_ball_inflections.py \
  --ball-track "$BALL_TRACK_PATH" --frame-times "$FRAME_TIMES_PATH" \
  --out "$BALL_KINKS_PATH"
sha256sum "$MEDIA_PATH" "$FRAME_TIMES_PATH" "$BALL_TRACK_PATH" "$BALL_KINKS_PATH"
```

Refuse the hop unless both the BALL track and kink JSONs declare the same
`source_video_sha256` and `frame_times_sha256`. The materializer independently
rechecks the consumed kink artifact and records a canonical dependency binding
over `kink_sha + media_sha + frame_times_sha`. pb.vision ball/court output is
not an independent agreement input.

## 5. Materialize B/C and freeze the complete input chain

Pass every consumed path explicitly as `VIDEO_ID=PATH`:

```bash
.venv/bin/python scripts/racketsport/build_abc_arm_manifests.py \
  --teacher-manifest "$CORPUS_OUT/manifest.json" \
  --output-dir "$ABC_OUT" --seed 20260720 \
  --media "$VIDEO_ID=$MEDIA_PATH" \
  --frame-times "$VIDEO_ID=$FRAME_TIMES_PATH" \
  --audio-onsets "$VIDEO_ID=$AUDIO_ONSETS_PATH" \
  --ball-velocity-kinks "$VIDEO_ID=$BALL_KINKS_PATH"
sha256sum "$CORPUS_OUT/manifest.json" "$ABC_OUT"/*.json
```

Repeat each path flag once per train clip. Agreement is one-to-one within each
family at `0.035s`: zero agreements are omitted, one gets weight `0.25`, and
two or more get `0.5`. C uses B's identical pixel windows, classes, weights,
agreement metadata, and number of loss-valid frames. It changes only the focal
event time within the same rally; the vacated frame stays valid background and
the shuffled frame is selected from B-valid frames.

`input_bindings.json` is the mandatory chain ledger. For every clip it must
bind corpus row -> media SHA -> frame-times SHA -> audio/kink artifact SHA ->
their declared media/PTS identities. Missing dependency identity is a hard
materialization refusal, not an optional warning.

## 6. Pre-dispatch refusal checks

- every recorded SHA recomputes exactly on the VM;
- B/C remain `verified=false`, teacher-derived, and never GT;
- every row is schema v2 with a 64-entry UNKNOWN mask;
- every media path decodes and matches its source SHA;
- B/C row, pixel, class, weight, update, and loss-valid-frame budgets match;
- compare-only and protected artifacts are absent from all training inputs; and
- source, checkpoint, code, seed, threshold, and optimizer-step choices are
  frozen before dispatch.

## 7. Train the frozen 3 x 3 arms at equal final steps

Run A, B, and C for seeds `20260720`, `20260721`, and `20260722` with the same
model-only initialization and exactly `--steps 1000`. All other frozen flags
are identical except the pseudo manifest (none/B/C). A wall exit is a failed
arm. It cannot leave a standard checkpoint or `finetune_manifest.json`.

Each successful output must have `completed_steps == target_steps == 1000`.
`finetune_manifest.json` is published atomically after both checkpoints and all
final metrics; reusing an output directory deletes any stale completion
manifest before validation/training starts.

After all nine arms finish, write a frozen bundle containing all input,
checkpoint, finetune-manifest, code, config, seed, and threshold SHA values.
Do not change or rerun an arm after this freeze.

## 8. Owner-41 selection and causal gate only

Generate the nine selection JSONs from the fixed 41-row owner validation split
in `owner_102_manifest.json`. The evaluator also covers every distinct non-GT
validation source video once in fixed, non-overlapping checkpoint windows for
the firing-rate measurement:

```bash
OWNER_MANIFEST=runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json

eval_owner_val () {
  arm="$1"
  seed="$2"
  checkpoint="$3"
  output="$4"
  .venv/bin/python scripts/racketsport/eval_event_head.py \
    --mode owner-val --device cuda --manifest "$OWNER_MANIFEST" \
    --arm "$arm" --seed "$seed" --checkpoint "$checkpoint" \
    --threshold "$FROZEN_THRESHOLD" --out "$output"
}

eval_owner_val A 20260720 "$A_20260720_CHECKPOINT" "$A_20260720"
eval_owner_val A 20260721 "$A_20260721_CHECKPOINT" "$A_20260721"
eval_owner_val A 20260722 "$A_20260722_CHECKPOINT" "$A_20260722"
eval_owner_val B 20260720 "$B_20260720_CHECKPOINT" "$B_20260720"
eval_owner_val B 20260721 "$B_20260721_CHECKPOINT" "$B_20260721"
eval_owner_val B 20260722 "$B_20260722_CHECKPOINT" "$B_20260722"
eval_owner_val C 20260720 "$C_20260720_CHECKPOINT" "$C_20260720"
eval_owner_val C 20260721 "$C_20260721_CHECKPOINT" "$C_20260721"
eval_owner_val C 20260722 "$C_20260722_CHECKPOINT" "$C_20260722"
```

Every JSON declares:

- `selection_scope=owner_validation_41`, `selection_rows=41`, and
  `protected_50_touched=false`;
- HIT/BOUNCE F1 at +/-2 frames, timing-error p90, and the same 22-row negative
  subset FP count;
- the preregistered non-GT full-video event rate; and
- `completed_steps=target_steps=1000`.

Then run exactly one gate command:

```bash
.venv/bin/python scripts/racketsport/abc_decision_gate.py \
  --arm-a "20260720=$A_20260720" "20260721=$A_20260721" "20260722=$A_20260722" \
  --arm-b "20260720=$B_20260720" "20260721=$B_20260721" "20260722=$B_20260722" \
  --arm-c "20260720=$C_20260720" "20260721=$C_20260721" "20260722=$C_20260722" \
  --out "$ABC_OUT/owner41_abc_gate.json"
```

The executable gate refuses any protected-50 input. It requires median B-A
macro-F1 >= +0.10, exact paired-bootstrap 95% lower bound >0, every seed
non-negative, B>C, per-class regression <=0.03, B negative FP <=2/22 and <=A+1,
B timing p90 non-worse than A, B rate in 0.3-1.0/s, and exact 1000-step parity
across all nine runs. If the verdict is not `PASS`, stop permanently for this
frozen experiment; the protected 50 remains untouched.

## 9. Protected-50 one-touch claim and hard refusal

Only after section 8 passes, freeze the final candidate/checkpoint, threshold,
all arm hashes, gate JSON SHA, and scorer SHA in `$FROZEN_EVAL_BUNDLE`. The
protected result cannot select a seed, threshold, checkpoint, or arm.

Before opening any protected file, use the stable experiment token
`event_head_abc_protected50_20260721`. The permanent mkdir claim lives beside
the shared held-out ledger, so every contender serializes the token check and
claim on the same filesystem. The directory is never removed, including after
an error or interruption:

```bash
ONE_TOUCH_TOKEN=event_head_abc_protected50_20260721
HELDOUT_LEDGER=runs/manager/heldout_eval_ledger.md
ONE_TOUCH_LOCK="runs/manager/${ONE_TOUCH_TOKEN}.one_touch.lock"
ONE_TOUCH_CLAIM="$ONE_TOUCH_LOCK/claim.json"

# BEGIN ONE_TOUCH_ATOMIC_GUARD
if ! mkdir -- "$ONE_TOUCH_LOCK"; then
  echo "HARD REFUSAL: protected-50 lock already exists or cannot be created" >&2
  exit 70
fi
set +e
rg -F -- "$ONE_TOUCH_TOKEN" "$HELDOUT_LEDGER" >/dev/null 2>&1
ledger_rg_exit=$?
set -e
case "$ledger_rg_exit" in
  0)
    echo "HARD REFUSAL: protected-50 token is already present in the ledger" >&2
    exit 70
    ;;
  1)
    ;;
  *)
    echo "HARD REFUSAL: ledger search errored (rg exit $ledger_rg_exit)" >&2
    exit 71
    ;;
esac
# END ONE_TOUCH_ATOMIC_GUARD
```

The successful `mkdir` is the single atomic claim and happens before the scorer
opens protected data. While owning that permanent directory, exit 0 from `rg`
means token present/refuse, exit 1 means absent/proceed, and every exit >=2 is
an error/refusal. An interruption consumes the one touch; deleting the lock or
retrying is forbidden. `claim.json` records the frozen-eval-bundle SHA,
owner-41 gate SHA, candidate/checkpoint SHA, threshold, scorer SHA, UTC time,
and status `claimed_before_read`. Append the same token and fields as a
pre-committed EVENTS row to `runs/manager/heldout_eval_ledger.md` using
`threed.racketsport.append_lock.append_text`; never edit an earlier row.

Any existing lock, prior ledger token, ledger read error, missing frozen SHA,
or owner-41 gate other than PASS refuses before protected file access. This is
the hard second-touch wiring.

Use this claim command; it reads only already-frozen non-protected artifacts:

```bash
.venv/bin/python - \
  "$ONE_TOUCH_CLAIM" "$HELDOUT_LEDGER" "$ONE_TOUCH_TOKEN" \
  "$FROZEN_EVAL_BUNDLE" "$ABC_OUT/owner41_abc_gate.json" \
  "$FROZEN_CHECKPOINT" "$FROZEN_THRESHOLD" scripts/racketsport/eval_event_head.py \
  "$ABC_OUT/protected50_frozen_result.json" <<'PY'
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from threed.racketsport.append_lock import append_text

claim_path, ledger_path, token, bundle_path, gate_path, checkpoint_path, threshold, scorer_path, result_path = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
    Path(sys.argv[5]), Path(sys.argv[6]), sys.argv[7], Path(sys.argv[8]), Path(sys.argv[9])
)

def sha256(path):
    if not path.is_file():
        raise SystemExit(70)
    return hashlib.sha256(path.read_bytes()).hexdigest()

gate = json.loads(gate_path.read_text())
if gate.get("verdict") != "PASS":
    raise SystemExit(70)
if gate.get("selection_scope") != "owner_validation_41" or gate.get("selection_rows") != 41:
    raise SystemExit(70)
if gate.get("protected_50_touched") is not False:
    raise SystemExit(70)
if result_path.exists():
    raise SystemExit(70)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
claim = {
    "schema_version": 1,
    "artifact_type": "event_head_protected50_one_touch_claim",
    "token": token,
    "status": "claimed_before_read",
    "claimed_utc": now,
    "frozen_eval_bundle_sha256": sha256(bundle_path),
    "owner41_gate_sha256": sha256(gate_path),
    "checkpoint_sha256": sha256(checkpoint_path),
    "threshold": float(threshold),
    "scorer_sha256": sha256(scorer_path),
}
payload = (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(claim_path, os.O_CREAT | os.O_WRONLY, 0o444)
try:
    if os.write(fd, payload) != len(payload):
        raise SystemExit(70)
    os.fsync(fd)
finally:
    os.close(fd)
os.chmod(claim_path, 0o444)
append_text(
    ledger_path,
    "\n\n## EVENTS protected-50 one-touch claim\n\n"
    f"- token: `{token}`\n- claimed UTC: `{now}`\n"
    f"- status: `claimed_before_read`\n- claim: `{claim_path}`\n"
    f"- frozen bundle SHA-256: `{claim['frozen_eval_bundle_sha256']}`\n"
    f"- owner-41 gate SHA-256: `{claim['owner41_gate_sha256']}`\n"
    f"- checkpoint SHA-256: `{claim['checkpoint_sha256']}`\n"
    f"- threshold: `{claim['threshold']}`\n"
    f"- scorer SHA-256: `{claim['scorer_sha256']}`\n"
)
completed = subprocess.run([
    sys.executable,
    str(scorer_path),
    "--checkpoint", str(checkpoint_path),
    "--mode", "protected-seed",
    "--threshold", threshold,
    "--out", str(result_path),
], check=False)
result_sha = sha256(result_path) if result_path.is_file() else "absent"
append_text(
    ledger_path,
    f"\n- token completion: `{token}`\n- scorer exit: `{completed.returncode}`\n"
    f"- protected result: `{result_path}`\n- protected result SHA-256: `{result_sha}`\n"
)
raise SystemExit(completed.returncode)
PY
```

This wrapper claims first, appends the pre-commitment, and immediately performs
the sole protected scorer invocation. Re-running the wrapper hard-refuses at
the permanent atomic lock. Do not invoke the protected scorer outside it.

## 10. Final frozen evaluation and closeout

Section 9 executes the single preregistered protected-50 evaluation on the
final checkpoint selected by the predeclared owner-41 rule after all A/B/C
arms were frozen. There is no threshold sweep, debug preview, partial probe,
A/B/C reselection, or rerun.

The wrapper hashes the protected result and appends a completion entry
referencing the claim token. Report the frozen owner-41 table and protected
result without choosing a favorable subset. Copy artifacts back, verify
both-side hashes, and shut down the VM.

Whether the result passes or fails, do not reopen the protected labels, rerun
the scorer, change code/config/thresholds, or overwrite outputs. A crash,
partial output, or operator mistake is recorded as the experiment result and
the existing claim forces every later attempt to hard-refuse.
