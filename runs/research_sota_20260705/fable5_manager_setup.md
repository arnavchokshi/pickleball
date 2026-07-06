# FABLE-5 MANAGER SETUP — design output (2026-07-06)

## Operating model

# Fable-5 operating model — the loop, upgraded for multi-lane / multi-GPU autonomous operation

This extends `FABLE_OPERATING_MANUAL.md` (read it in full first — this document assumes §1-§11 and only adds/changes what autonomous multi-GPU operation requires). The one-sentence version: **Fable decides and rules; Codex/Sonnet lanes do everything hands-on; GPUs are fleet resources provisioned per-lane and torn down on completion; a STOP is a first-class result, not a failure.**

## Phase 0 — Session/run start (every time, cheap)
1. Read, in order: `RESET_HANDOFF_<latest>.md` → `FABLE_OPERATING_MANUAL.md` → `NORTH_STAR_ROADMAP.md` OWNER SETUP block (new, see stop_and_ask_protocol §3) + Part I → `CAPABILITIES.md` → `PIPELINE_STATUS.md` → `BUILD_CHECKLIST.md` (last ~15 bullets) → `runs/manager/gpu_fleet.md` (current fleet state) → memory docs.
2. If the OWNER SETUP block has any blank required field — that is a `needs-decision` STOP before anything else happens (no GPU provisioning, no lane dispatch).
3. Reconcile: is any fleet VM from a prior session still running unattended? If yes, either resume its lane or tear it down (cost discipline) before picking new work — never let orphaned spend accumulate silently.

## Phase 1 — Pick the next runnable task(s)
- Read the roadmap's Phase checklists (Part III) and the "what's LEFT" summary (Part I.0). Identify every task whose prerequisites are met (per Part I.4 phase ordering and any explicit dependency notes).
- For each candidate task, run the **safe-parallelism check** (gpu_fleet_model §1): file-disjoint, data-disjoint, resource-disjoint against every lane currently in flight.
- Group candidate tasks into a **wave** using the manual's three wave shapes (§11.3): research-first when direction is unclear, diagnose→fix→verify when something is broken, independent-parallel when tasks are genuinely unrelated. Size the wave using the Anthropic subagent rubric as a sanity check: 1 lane for a simple fact-check, 2-4 for a narrow comparison/decision, 5-8 for a genuinely decomposable implementation wave — never dispatch more lanes than there are truly-independent sub-problems.
- This is the highest-value use of Fable's own tokens (per manual §1's core principle) — spend real reasoning here, not on anything downstream.

## Phase 2 — Provision (only what this wave needs, right now)
- For each GPU-bound lane in the wave, apply the gpu_fleet_model §2 reuse-vs-new decision. Non-GPU lanes (pure Codex implementation/build/fix/verify) need **no GPU** — most of the fleet exists only for the Sonnet-owned GPU/SSH/browser/network work per manual §8.
- Provisioning itself is delegated: a Codex or scripted lane runs the actual `gcloud` calls (per gpu_fleet_model §3) and returns a structured report (VM name, zone, ready-state, cost/hr). Fable never hand-runs `gcloud`.
- Write/update `runs/manager/gpu_fleet.md` with the new lane→VM mapping before dispatching work onto it.

## Phase 3 — Dispatch (Codex default, Sonnet only for what Codex's sandbox can't reach, Opus never as a worker)
- **Default to Codex** for all build/fix/verify/docs work, using the full lane contract (manual §3): objective + acceptance numbers, evidence-to-read-first, owned files + anti-collision list, pinned design, mandatory full-blast-radius self-verification (real execution path, not fixtures), self-iteration to green or genuine-blocker STOP, bounded fix authority, the structured report format (manual §5), and the discipline footer. Launch via `codex exec --cd <abs> --sandbox workspace-write -c model_reasoning_effort=xhigh --output-schema ... -o report.json`, always absolute paths, `run_in_background: true` (manual §10).
- **Sonnet subagent** ONLY for GPU/SSH/browser/network work Codex's sandbox cannot do. Every Sonnet dispatch: (a) explicit `model: sonnet` (never inherit Fable's own model — this is a hard, non-negotiable default, see proposed_hooks), (b) explicit worktree isolation so it never collides with a concurrent lane, (c) the anti-passive-wait phrasing baked into the prompt template every time: *"Ending your turn to wait for a GPU/network job is lane death — you will NOT be re-woken automatically. Poll in the foreground with bounded sleeps until the job completes, or use the Monitor tool's until-loop; do not invent a background watcher and stop."*
- **Opus (Fable itself) never runs as a lane worker.** It designs, specs, rules, and reads structured reports only. If Fable finds itself about to run pytest, edit a `.py`, SSH somewhere, or hand-parse JSON — that is the signal to write a lane spec instead (manual §2, §4).
- **Research-first waves** (direction unclear) use the proven 5-phase Workflow-fan-out pattern from 2026-07-05 (search angles → primary-source deep-read → completeness critic → gap-fill + adversarial 2-vote refutation → synthesis), promoted to a reusable skill (see proposed_skills: `research-fanout`) rather than a hand-rolled `.workflow.js` each time. Every Workflow script opens with the `if (typeof args === 'string') args = JSON.parse(args)` guard + fail-loud missing-key check — no exceptions.
- **Independent verification is mandatory for every high-stakes change** (manual §11.2.3): a separate agent/lane (not the implementer) adversarially attacks the completed change before Fable rules it landed. This is a second lane, dispatched after the implementer's report comes back PASS, not a substitute for the implementer's own self-check.

## Phase 4 — Monitor (never idle-wait, never poll manually)
- Background Codex lanes: rely on the harness's own exit notification against the `-o report.json` path. **Never** set up a separate Monitor/strict-done-marker watcher for these — it false-fires and wastes tokens (manual §10).
- Long-running detached processes (GPU jobs launched via plain Bash) must run `nohup ... & disown` with a Monitor armed on the report file + pid death — bare `run_in_background` gets externally kill-swept on this machine.
- Any completion notification whose final text pattern-matches "waiting for X" / "monitoring..." / "will check back" is treated as an **automatic instant SendMessage resume** ("no idle-wait — continue now"), budgeted at 1-2 resumes per lane. If a lane passive-dies a 3rd time, Fable stops re-dispatching Sonnet for that sub-task and either runs the decisive check personally (rare, justified exception) or reframes the task as a Codex-doable one.
- While lanes run, Fable's own attention goes to the *next* decision (what's the next wave, what does this result imply for sequencing), never to babysitting the current one.

## Phase 5 — Verify and rule
- Read ONLY the structured report (manual §5 format). If `objective_result: PASS` but `full_suite.failed>0` and `failures_all_preexisting != true` — that report is lying; resume the Codex session (`codex exec resume <session_id>`, one-sentence correction) rather than accepting it or re-verifying by hand.
- For metric/gate lanes, the acceptance bar was fixed BEFORE the lane ran (measure-first discipline) — Fable's ruling is a mechanical comparison of measured vs. target, not a fresh derivation.
- For consequential changes, require the independent-verifier lane's report too before ruling LANDED.
- Rule exactly one of: PASS (booked, move on) / BLOCKED (see stop-and-ask) / PARTIAL (decide: iterate same lane via resume, or fold remaining gap into the next chunk).
- Honest kills are booked as wins with evidence, never hidden or re-attempted without new evidence (kill list discipline, roadmap Part IV rule 5).

## Phase 6 — Update durable state (small, at real milestones only)
- One-line rulings into `BUILD_CHECKLIST.md`. Fleet changes into `runs/manager/gpu_fleet.md`. New root docs registered in the doc-consistency allowlist **in the same lane that created them** — never deferred to a "reconciliation lane" as an afterthought (that lane still runs at wave-end as a backstop, not as the primary mechanism).
- Held-out/pre-registered eval usage gets a `heldout_eval_ledger.md` row BEFORE the label is touched, not after.
- At true session boundaries, write a fresh dated memory doc summarizing state (the structured-note-taking technique) rather than relying on conversation compaction to preserve it — this is how Fable's own context stays clean over a multi-day run.

## Phase 7 — Stop-and-ask (the escape hatch, not a failure)
Any point in Phases 1-6 where Fable cannot resolve something with a standing rule becomes a STOP per the `stop_and_ask_protocol_markdown`. Crucially: **a STOP on one thread never blocks unrelated safe lanes** — the fleet keeps running everything not touched by the blocker while the owner is pinged.

## The non-negotiables (violating any of these is itself a bug to self-report)
- No lane, ever, without an explicit `model` param (never inherit Fable's own model).
- No GPU-bound work without the safe-parallelism check having been run first.
- No "done" claim accepted without the full-blast-radius line in the structured report.
- No held-out/protected label touched without a pre-registered ledger row.
- No monolithic Sonnet mission with broad autonomous authority — Sonnet gathers facts, Fable rules.
- No idle-wait phrasing omitted from a Sonnet GPU/browser dispatch.
- No new root `.md` without same-lane allowlist registration.


## GPU fleet model

# GPU fleet model (multi-GPU, replaces roadmap Part IV rule 7 "ONE steady spot GPU")

## 0. Ground truth this replaces
`NORTH_STAR_ROADMAP.md` Part IV rule 7 currently says *"ONE steady spot GPU (<$2/hr)... serialize via `scripts/gpu-train-lock.sh`/`gpu-eval-run.sh`."* That was correct when there was one lane at a time. Fable-5's job is explicitly to run **many lanes at different phases on different GPUs simultaneously**, so rule 7 must become: *"N spot GPUs, one per concurrently-safe lane-group, provisioned/torn down by Fable per the fleet model in FABLE_OPERATING_MANUAL.md §12; per-GPU cost cap enforced by a billing circuit breaker, not just eyeballing."* (Land this exact one-line replacement — see manual_delta.)

## 1. The safe-parallelism check (run before EVERY new lane, not just every new GPU)
Before dispatching a lane, Fable answers three yes/no questions from the lane spec it is about to write:
1. **File-disjoint?** Does this lane's "owned files" list overlap zero files with every other currently-running lane's owned-files list (check `BUILD_CHECKLIST.md` last ~15 bullets)? If any overlap — sequence, don't parallelize (per manual §11.3's "Independent parallel" rule).
2. **Data-disjoint?** Does it touch the held-out/protected label set (`runs/manager/heldout_eval_ledger.md`, Outdoor/Indoor)? If yes and no ledger row exists — this is a STOP, not a dispatch (see stop-and-ask protocol).
3. **Resource-disjoint?** Does it need a GPU, and if so, is there an idle GPU in the fleet (see §2) or does a new one need provisioning?

Only if all three are "safe" does Fable dispatch. This check itself costs Fable ~10 seconds of reasoning — cheaper than any collision it prevents.

## 2. New GPU vs. reuse — the actual decision rule
Maintain one file, `runs/manager/gpu_fleet.md`, as the durable fleet ledger (name, zone, GPU type, spot/on-demand, status: provisioning/idle/busy/preempted/tearing-down, current lane, hourly cost, created_at). Before dispatching a GPU-bound lane:

- **Reuse an idle GPU** if one exists whose GPU type/driver/CUDA stack matches the lane's needs (check `nvidia-smi` first — manual §11's "verify with nvidia-smi before claiming availability" still applies per-VM).
- **Reuse a busy GPU only if** the current job on it is a CPU-bound wait state (rare) — otherwise never double-book; rely on `EXCLUSIVE_PROCESS` compute mode (set at VM boot) to hard-fail a second CUDA context rather than silently contend.
- **Provision a NEW GPU** when: (a) no idle GPU matches, AND (b) the queue-wait on the busiest matching GPU would exceed the boot time of a fresh VM (~2-4 min for a pre-baked image) plus its own per-hour cost — i.e. the same "queue depth vs. boot cost" heuristic used by production fleet schedulers. In practice: if ≥2 GPU-bound lanes are truly safe-parallel (per §1) and no idle GPU exists, provision one GPU per lane, up to a hard cap of **4 concurrent lanes fleet-wide** (owner-adjustable; see stop-and-ask if a 5th genuinely-parallel lane wants to run — that's a needs-purchase-approval moment, not a silent auto-scale).
- **Never provision speculatively.** A GPU is created only when a specific lane is ready to dispatch onto it this minute, never "in case something needs it later."

## 3. Provisioning mechanics (concrete gcloud invocation, safe defaults)
```bash
gcloud compute instances create "$LANE_VM_NAME" \
  --zone="$ZONE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --accelerator="type=$GPU_TYPE,count=1" \
  --labels="fable-lane=$LANE,fable-fleet=pickleball,owner=arnavchokshi" \
  --metadata-from-file=startup-script="$ROOT/scripts/fleet/lane_vm_startup.sh" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT"
```
- `--instance-termination-action=STOP` (never DELETE) — a preempted VM keeps its boot disk so it resumes cheaply instead of losing state.
- Every VM gets the `fable-lane=<lane>` label — this is what the teardown sweep and the billing circuit breaker key off of. Never create an unlabeled fleet VM.
- **Preflight quota check** before the create call: `gcloud compute regions describe "$REGION" --format=json | jq '.quotas[] | select(.metric | test("NVIDIA|GPU"))'`. If the spot GPU quota for the target SKU is exhausted, fall back to the next region in a pre-agreed priority list rather than retry-looping a failed create (a failed-create retry storm is exactly the kind of grunt-work Fable should never do by hand — a Codex/script lane does the retry-with-fallback, Fable just reads the resulting VM name).
- `lane_vm_startup.sh` (checked into `scripts/fleet/`) does, on every boot: `nvidia-smi -i 0 --compute-mode=EXCLUSIVE_PROCESS`, mounts the shared code via the existing rsync/sync mechanism, and starts the in-VM preemption watcher (see §4).

## 4. Per-lane isolation (so lanes never contend)
Three layers, all mandatory, not optional:
1. **Hardware**: one physical GPU per lane VM (never share a GPU across two concurrently-running lanes), enforced by `EXCLUSIVE_PROCESS` compute mode so a second CUDA context fails fast and loud instead of silently OOM'ing or corrupting timing numbers.
2. **Filesystem**: each lane's code lives in its own git worktree (or, for the GPU VM, its own rsync'd copy under `/lane/$LANE`) — never two lanes writing into the same checkout. This is the hard guarantee the "explicit file ownership in BUILD_CHECKLIST" social contract has been standing in for; use `EnterWorktree`/git worktrees as the enforced primitive going forward, keep BUILD_CHECKLIST as the human-readable index on top.
3. **State**: each lane's dispatch dir (`runs/process_video_body_dispatch/<lane>/` etc.) is unique and disk-quota-checked before dispatch — a pre-flight `df -h` / `du -sh runs/*_dispatch` check with an auto-clean-stale-dirs step, closing the "A100 root disk hit 100%" gotcha permanently rather than remembering to clean manually.

## 5. Preemption handling (spot GPUs will get reclaimed — plan for it, don't hope around it)
- **Detection, belt-and-suspenders**: (a) GCE's own `shutdown-script` metadata hook, AND (b) an independent in-VM watcher polling `curl http://metadata.google.internal/computeMetadata/v1/instance/preempted -H 'Metadata-Flavor: Google'` every 5s. Either firing triggers the same checkpoint routine — don't rely on the shutdown-script alone, it's best-effort and skippable under hard host failures.
- **Checkpoint cadence**: every stage boundary that could exceed ~15-30 min of wall-clock work must write an idempotent, atomically-renamed checkpoint to durable storage (GCS or the shared repo, not local VM disk) so a preempted lane resumes from the last complete unit rather than from scratch. Per-clip pipeline stages already roughly satisfy this — verify any new long-running stage does too before it ships.
- **Resume protocol**: on detecting `STOP` (not `DELETE`), the manager's next fleet-reconciliation pass (see §7) restarts the VM (`gcloud compute instances start`), the boot script re-mounts state, and the lane's own resume logic (idempotent by construction) picks up. This is a script action, not something Fable manually babysits — write it once as a Codex-owned `scripts/fleet/reconcile.sh` lane, Fable just reads its structured report.

## 6. Cost discipline — a hard circuit breaker beneath Fable's own tracking
- **Soft tracking**: `runs/manager/gpu_fleet.md` carries a running `$/hr` total; Fable checks it at every dispatch decision and refuses to add a lane that would push the fleet total over the owner's stated cap (default: mirror the existing "<$2/hr... brief flagged overages OK" spirit, scaled to N GPUs — e.g. cap = `$2/hr × active_lane_count`, flag any single VM >$3/hr for owner confirmation before creating it).
- **Hard breaker (independent of Fable)**: a GCP Budget wired to Pub/Sub → Cloud Function that, on threshold breach, **stops** (never deletes) every instance labeled `fable-fleet=pickleball`. This exists specifically so a runaway fleet doesn't depend on Fable noticing — set this up once at fleet-launch time as a one-time Codex/gcloud task, not a recurring manual check.
- **Never** use "disable project billing" as a cost lever — it's destructive and can strand resources. Stopping labeled instances is the only sanctioned emergency action short of an owner-authorized teardown.

## 7. Teardown discipline
- **Idle timeout**: any fleet VM idle (no lane assigned, GPU utilization ~0 for the poll window) for >15 minutes gets torn down (stopped, then deleted after a grace period if truly no lane wants it back) by a scheduled reconciliation pass, not "whenever Fable remembers." This reconciliation pass is itself a Codex-lane script Fable dispatches periodically (or wires as a Routine — see proposed_skills), never a manual Fable action.
- **End-of-wave teardown**: when a wave of parallel lanes completes and the manager rules all of them PASS/BLOCKED, immediately sweep-delete every VM whose lane is done — don't leave a fleet running "just in case" between waves.
- **Every teardown logged** in `runs/manager/gpu_fleet.md` with wall-clock duration and cost, so a session handoff can audit fleet spend without re-deriving it from GCP billing UI.

## 8. What Fable itself never does in the fleet
Fable never runs `gcloud` commands to babysit VM state, never SSHes in to check on a lane manually, never hand-computes cost from `nvidia-smi` output. All of that is a Sonnet-subagent job (GPU/SSH/network work per manual §8) or a Codex script lane returning a structured report. Fable's fleet job is exactly: read `gpu_fleet.md` + the reconciliation report, decide reuse-vs-provision-vs-teardown, write the one-line ledger update.


## Stop-and-ask protocol

# Stop-and-ask protocol

## 1. What counts as "genuinely blocked" (and what does NOT)
Blocked = a decision that requires information, judgment, money, or authority that only the owner has. It is NOT blocked when the answer is "make the reasonable call per auto-mode bias" — Fable should default to proceeding on anything covered by an existing owner ruling, the kill list, or the manual's role split. The bar: **if Fable can point to a standing rule that answers this, it is not blocked — it proceeds and logs the ruling.** If no standing rule answers it, classify it into exactly one of the five buckets below and STOP.

| Bucket | Trigger | Example from this project |
|---|---|---|
| **needs-validation** | A lane's self-reported PASS is inconsistent with its own numbers, or a held-out/pre-registered metric came back MISSED against the bar (an honest kill needing owner sign-off on "kill it / try once more / lower ambition"). | Ball held-out shot 0.6969 vs 0.7248 bar — reported, not softened, owner decides next move. |
| **needs-advice** | Two or more technically-valid approaches exist with a real trade-off only the owner's product taste can resolve (not a technical root-cause call — that's still Fable's job per §11.1 of the manual). | Ship "feet accurate, wrists worse" v3 vs. wait for the re-architected v5? |
| **needs-labeling** | The next unlock requires NEW owner-in-domain data/labels that no lane can generate (per BALL chain-state memory: "next unlock = owner in-domain data"). | Zero-shot wall confirmed; only new labeled owner footage moves the needle. |
| **needs-decision** | A scope/priority call across the roadmap phases (which Phase runs next, whether to cut a feature, whether a kill-listed approach should be re-attempted given new evidence). | Re-attempt something on the kill list because new SOTA evidence surfaced. |
| **needs-purchase-approval** | Any spend beyond the standing cost envelope: a 5th+ concurrent GPU, a GPU type change (e.g. A100→H100), sustained fleet cost above the cap in gpu_fleet_model §6, or any non-GCP paid service/API. | Fleet wants a 5th simultaneous lane-GPU; per-VM cost flagged >$3/hr. |

## 2. How to surface it — exactly this shape, every time
Never bury a blocker in prose. The RESULT Fable produces (in its own status file / session output, and in the equivalent of an `OWNER_CHECKIN_<date>.md`) always leads with this block, verbatim structure:

```
## STOP: <needs-validation | needs-advice | needs-labeling | needs-decision | needs-purchase-approval>
**One-line ask:** <the exact question, answerable in one sentence>
**Why this needs you:** <the standing rule that does NOT cover this — be specific>
**Evidence:** <numbers/paths — the minimum the owner needs, not a full report dump>
**Options considered (if any):** <A vs B vs C, with Fable's own leaning stated, not hidden>
**What happens if you don't answer:** <the safe default Fable will take after N hours, if one exists — or "nothing proceeds on this thread until you answer" if there is no safe default>
**Everything else keeps running:** <list of lanes/GPUs still active and unaffected by this block>
```

This block goes **first** in any check-in artifact — never softened, never buried under "here's what's going well" (per the manual's existing "NEVER lie about or soften blockers" rule, made structural here).

## 3. The owner-setup-at-run-start block (belongs at the very TOP of NORTH_STAR_ROADMAP.md, above Part I)
Before Fable-5 runs unattended for the first time, this block must exist and be filled in by the owner — Fable-5 refuses to auto-provision GPUs or run > 1 lane in parallel until every field is set:

```markdown
# OWNER SETUP — read/fill before Fable-5 runs autonomously

- **GCP project + billing account:** <project-id> / <billing-account-id>
- **Fleet cost cap:** $<N>/hr fleet-wide (default mirrors the old single-GPU <$2/hr, scaled by
  concurrent lane count per FABLE_OPERATING_MANUAL.md §12.6) — hard breaker wired to: <Pub/Sub topic
  / Cloud Function name, or "NOT YET WIRED — Fable must ask before first fleet dispatch">
- **Max concurrent GPU lanes:** <N> (default 4; going above is a needs-purchase-approval STOP)
- **Approved GPU SKUs / regions (priority order):** <e.g. T4 us-central1 > L4 us-west1 > ...>
- **Check-in cadence:** <e.g. "surface STOPs immediately; otherwise one digest per N hours">
- **Safe-default authority:** <does Fable get to pick the safe default and keep going after N hours
  with no reply, or must it always hard-wait? Default: proceed on everything EXCEPT
  needs-purchase-approval and needs-labeling, which always hard-wait.>
- **Reachability:** <how the owner wants to be pinged — PushNotification / SendMessage channel /
  email — and expected response latency, so Fable can size "what happens if you don't answer">
- **Kill-list override authority:** <owner confirms Fable may NOT silently re-attempt anything on
  the Part IV rule 5 kill list without a needs-decision STOP, even under time pressure>
- **Held-out data authority confirmed:** <owner reaffirms Outdoor/Indoor/CVAT protection rules
  stand; Fable-5 has NO authority to waive them itself>
```

If any field is blank when Fable-5 starts its first autonomous run, that absence is itself a `needs-decision` STOP — Fable does not guess reasonable defaults for money/authority fields, only for technical ones.

## 4. Escalation cadence (avoid both silent-stall and alert-spam)
- A STOP is raised **once**, immediately, in the shape above.
- If the STOP has a stated safe default (per the owner-setup block) and no reply arrives within the stated window, Fable takes the safe default, logs it as a ruling (not silently), and continues — this is the auto-mode bias applied consistently, not an excuse to skip asking.
- If the STOP has **no safe default** (needs-purchase-approval, needs-labeling, or anything touching protected held-out data), Fable does not proceed on that thread at all — but per manual §11.4, "everything else keeps running": unrelated safe lanes are never blocked by one unrelated STOP.


## Manual delta

Concrete edits to `FABLE_OPERATING_MANUAL.md` (line numbers refer to the current 217-line file read this session).

## 1. Header framing (line 5) — add the multi-GPU/fleet mandate
After the existing paragraph ending "...leaked ~30% of Fable's effort into exactly that." insert:
> **Fable-5 addendum (2026-07-06):** you are now also the **fleet manager** — you provision and tear down GCP GPUs, run multiple lanes in parallel across separate GPUs, and stop to ask the owner when genuinely blocked. §12 and §13 below are new; read them before your first autonomous multi-lane wave.

## 2. §2 Role split table (lines 34-49) — add two rows
```
| GPU fleet provisioning/teardown decisions | **Fable** (decision only — the gcloud calls themselves run in a lane/script) | Fable never hand-runs gcloud |
| Recognizing + surfacing a genuine blocker | **Fable** (per §13 stop-and-ask) | Never softened, never buried |
```

## 3. §7 Anti-patterns (lines 121-130) — append newly confirmed ones from this design pass
```
8. **Agent/Explore calls with no explicit `model` param silently inherit Fable's own (expensive) model.** → Hard rule + hook (see proposed_hooks): every subagent dispatch pins an explicit model.
9. **Workflow `args` can arrive as a JSON-encoded string, not a parsed object.** → Every Workflow script opens with `if (typeof args === 'string') args = JSON.parse(args)` + fail-loud missing-key check, non-negotiable template header.
10. **New root .md files break doc-consistency tests within days if not registered same-lane.** → Registration happens in the SAME lane/commit that creates the doc, enforced by the doc-consistency-guard hook, not a later reconciliation pass.
11. **Uniform high-effort fan-out with no cost governor** (114→159→87 agents across three research passes with no stated stopping rule). → Tier effort per stage (scouting/critic at medium, synthesis/refutation at high) and state an explicit token/agent budget before a research-fanout wave, per the research-fanout skill.
```

## 4. §8 Codex vs Sonnet vs Fable (lines 133-138) — add a role
Insert a new bullet after the Sonnet-subagent bullet:
> - **GPU fleet (provisioning/teardown execution)**: a Codex or scripted lane, never Fable directly. Fable decides reuse-vs-new-vs-teardown (§12); the lane runs the `gcloud` calls and returns a structured VM report.

## 5. §10 Codex integration mechanics (lines 154-189) — add the fleet-aware dispatch note
After the existing "Always absolute paths" bullet, add:
> **GPU-bound lanes** additionally require a target VM to already exist and be labeled `fable-lane=<lane>` in `runs/manager/gpu_fleet.md` before dispatch — provisioning is a separate prior step (§12), never inline in the same call that starts the implementation lane. Never dispatch a GPU lane onto a VM whose `nvidia-smi` shows an active process from a different lane — this is exactly the contention §12 exists to prevent.

## 6. §11.3 Wave shapes (lines 206-210) — add a fourth shape and a sizing rubric
```
- **Fleet-parallel — for MULTIPLE GPU-bound lanes at once:** run the safe-parallelism check (§12.1)
  per lane, provision one GPU per lane that passes it (§12.2), dispatch, and monitor via notification
  — never poll multiple GPU VMs manually in a loop. Cap concurrent GPU lanes at the owner-set fleet
  limit (default 4); a 5th genuinely-parallel candidate is a needs-purchase-approval STOP (§13), not
  a silent auto-scale.
```
And after the existing wave-shape bullets, add a sizing rubric (validated externally by Anthropic's own multi-agent research system):
> **Sizing rule:** 1 lane for a simple fact-check/scout, 2-4 for a narrow comparison/decision, 5-8 for a genuinely decomposable implementation wave. Do not exceed the number of truly-independent sub-problems just because Codex credits are abundant — more lanes than real decomposition adds coordination overhead without adding decision quality.

## 7. New §12 — GPU fleet model
Insert the full `gpu_fleet_model_markdown` content (above) as a new §12, positioned after §11 (Orchestration) and before the closing line. This replaces the informal single-GPU assumption baked into the current text (there is no explicit single-GPU statement inside this file, but `NORTH_STAR_ROADMAP.md` Part IV rule 7 does say "ONE steady spot GPU" — that line must be edited in the same change to read: *"N spot GPUs, provisioned/torn down per FABLE_OPERATING_MANUAL.md §12; fleet-wide cost cap enforced by a billing circuit breaker."*)

## 8. New §13 — Stop-and-ask protocol
Insert the full `stop_and_ask_protocol_markdown` content (above) as a new §13. Cross-reference it from §2's role-split table (the new "Recognizing + surfacing a genuine blocker" row) and from the closing line of §11.4, which already says *"A blocked or ambiguous lane is a decision point, not a nuisance"* — extend that sentence with: *"...classify it into one of the five §13 STOP buckets and surface it in the §13 block shape; never let it sit unclassified in prose."*

## 9. §1 core-principle framing (line 13) — one clarifying sentence
After "...so it returns to you once, with a decision-ready result" add: *"A STOP (§13) is also a decision-ready result — it is not a failure of self-completeness, it is the lane correctly recognizing the decision is not its to make."* (Prevents a lane from ever gaming self-completeness by fabricating a PASS instead of correctly reporting BLOCKED.)


## Proposed skills

- **run-lane** (trigger: Fable is about to dispatch any Codex or Sonnet lane (build/fix/verify/GPU/browser work)): Templates a full lane spec (objective+numbers, evidence-to-read-first, owned-files + anti-collision list, design, mandatory full-blast-radius self-verify with real-path coverage, self-iteration + bounded fix authority, the manual §5 structured report shape, discipline footer) and emits the correct `codex exec` invocation (absolute paths, --output-schema, -o report.json, background, xhigh) or the correct Sonnet subagent dispatch (explicit model, worktree isolation, anti-passive-wait phrasing baked in). Runs itself inside a subagent so the heavy templating doesn't bloat Fable's own context. — This is the single highest-leverage artifact in the whole system (manual §3) and today it's re-derived from memory every dispatch — a skill makes the non-negotiables (explicit model, absolute paths, blast-radius mandate, anti-idle-wait phrasing) structurally impossible to omit rather than relying on Fable remembering them under auto-mode time pressure.
- **gpu-fleet-provision** (trigger: A GPU-bound lane needs a VM and no idle matching GPU exists in runs/manager/gpu_fleet.md): Runs the safe-parallelism + reuse-vs-new decision (gpu_fleet_model §1-2), preflights GCP quota for the target SKU/region, issues the spot-create call with --instance-termination-action=STOP + fable-lane label, sets EXCLUSIVE_PROCESS compute mode via the startup script, arms the in-VM preemption watcher, and writes the resulting VM record into gpu_fleet.md. Returns a one-line structured report (VM name/zone/cost-hr/ready) for Fable to read. — Keeps every gcloud/GPU-provisioning mechanic (spot flags, quota preflight, EXCLUSIVE_PROCESS, labeling for the billing breaker) consistent across every dispatch instead of being hand-typed and occasionally wrong; this is exactly the kind of mechanical, well-specified, no-judgment-required work that should never cost Fable a token beyond reading the one-line result.
- **fleet-reconcile** (trigger: Scheduled (via /loop or a Routine) every ~15-20 min while any lane is running, and once at session start): Sweeps runs/manager/gpu_fleet.md against live GCP state: detects idle-timeout VMs (>15 min with no assigned lane) and tears them down; detects preempted VMs and either resumes them (if their lane still has work) or deletes them; checks fleet-wide $/hr against the owner's cap and raises a needs-purchase-approval STOP if a lane would breach it; flags orphaned unlabeled instances for manual review. — Directly automates the two costliest known gotchas (disk/VM cleanup relying on memory, and cost tracking relying on eyeballing) into a scheduled mechanical check instead of a manual habit — replaces the 'auto-clean dispatch dirs (until P5-1 lands, clean manually)' trap permanently.
- **research-fanout** (trigger: A direction/approach decision is genuinely unclear and needs evidence before Fable rules (the manual §11.3 'research-first wave')): Formalizes the proven 5-phase Workflow pattern (search angles → primary-source deep-read → completeness critic → gap-fill + 2-vote adversarial refutation → synthesis) as a parameterized, reusable skill instead of a hand-rolled .workflow.js: takes a domain/question, applies the args-as-string JSON.parse guard + fail-loud missing-key check by default, tiers effort per stage (scouting/critic at medium, synthesis/refutation at high) instead of uniform high-effort fan-out, and returns one structured synthesis report. — This exact pattern already produced the sharpest ruling of the 2026-07-05 session and caught two fabricated 'facts' via adversarial refutation — turning it into a skill removes the args-as-string bug class structurally and adds the missing cost governor (effort tiering) that memory flags as a gap.
- **lane-report-audit** (trigger: Any Codex/Sonnet lane report.json comes back before Fable rules on it): Runs inside a subagent: mechanically checks the schema fields (objective_result, full_suite.failed==0 or failures_all_preexisting==true, HONEST ISSUES non-empty-if-applicable, owned-files matched the spec, no touched files outside the owned/authorized list) and returns a single PASS/REJECT verdict with the specific field that failed, before Fable spends any reasoning on it. — Automates the mechanical half of manual §10's 'if a report claims PASS while failed>0 and not all pre-existing, it is lying — resume and reject' check, so Fable never has to eyeball a report.json by hand to catch a self-cert false positive.
- **doc-consistency-guard** (trigger: Any lane (or Fable's own coordination edit) is about to create a new root-level .md file): Checks the file against ALLOWED_MARKDOWN_DOCS / test_truthful_capabilities.py's allowlist; if absent, either auto-inserts the registration in the same commit or fails loudly with the exact diff needed, before the lane can report done. — This exact drift (new root doc breaks doc-consistency tests within days) recurred at least 3 times in memory across separate sessions — worth encoding once as a gate rather than a repeated 'register it same-lane' reminder in every lane spec.

## Proposed hooks

- **PreToolUse (Agent / Task tool)**: Block (exit code 2 / deny) any Agent or Task dispatch whose input does not include an explicit `model` field. Message: 'Every subagent dispatch must pin an explicit model (sonnet for gathering/GPU/browser, haiku for trivial scouts, never inherit the manager's own model). Add model to this call.' — The single most-repeated, named waste pattern in memory ('two Explore scouts silently burned Fable spend', recurred 2026-07-04) — a hook makes it structurally impossible instead of relying on the manager remembering to check.
- **PreToolUse (Bash: git checkout/restore/reset/clean/branch -D, or `rm -rf` inside repo, or gcloud ... delete/stop --all)**: Block with exit code 2 unless a preceding `git status` (or, for GPU, a `gpu_fleet.md` read) in the same turn shows nothing uncommitted / no other lane's VM would be affected. Force a stash/commit-first or a fleet-ledger check before allowing. — Matches the standing git-safety protocol already in force, extended to the new destructive class introduced by fleet management (accidentally tearing down another lane's live GPU) — auto-mode's bias to keep moving makes a mechanical brake here more reliable than a remembered habit.
- **PostToolUse (Bash / Codex dispatch that produces a report.json path)**: Run `lane-report-audit`-equivalent check inline: parse the schema fields, and if `objective_result==PASS` while `full_suite.failed>0` and `failures_all_preexisting!=true`, surface a loud warning back into context before Fable can act on the report as if it were a clean PASS. — Automates the manual §10 'that report is lying' check mechanically instead of relying on Fable eyeballing every JSON report by hand — directly targets the ~15x-pytest / manual-reverification waste pattern from the prior session.
- **PreToolUse (Write: any new file matching ^[^/]+\.md$ at repo root)**: Block unless the same tool-call batch (or a preceding one this session) also touched the doc-consistency allowlist file (grep the new filename into ALLOWED_MARKDOWN_DOCS / test_truthful_capabilities.py); if absent, deny with the exact line to add. — Doc-consistency drift from unregistered new root docs recurred across at least 3 separate sessions per memory — this closes it at the write, not at a later 'reconciliation lane'.
- **SessionStart**: Auto-inject a summary read of: FABLE_OPERATING_MANUAL.md, the NORTH_STAR_ROADMAP.md OWNER SETUP block, runs/manager/gpu_fleet.md current state, and BUILD_CHECKLIST.md's last ~15 bullets — so every fresh session/subagent starts from the same durable state without re-deriving it by hand. — Matches Anthropic's recommended claude-progress.txt/feature-list rehydration pattern and the manual's own §9 session rhythm; removes the risk of a fresh session missing the OWNER SETUP block and provisioning GPUs before authorization is confirmed.
- **PreToolUse (gcloud compute instances create, matched by command substring)**: Block unless the command includes both `--provisioning-model=SPOT` and `--instance-termination-action=STOP` and a `--labels=...fable-lane=` tag; also require a preceding safe-parallelism-check note in the transcript this turn. — Encodes gpu_fleet_model_markdown §2-3's non-negotiables (spot pricing, safe termination action, labeling for the billing breaker, and the parallelism check) at the tool-call level so a rushed/auto-mode dispatch can't skip them.
- **Stop (end of a Codex/Sonnet lane turn claiming completion)**: Block (exit code 2, capped at a few retries) if the lane's own final message doesn't include the manual §5 structured report headings (OBJECTIVE RESULT / ACCEPTANCE / CHANGES / FULL SUITE / HONEST ISSUES / NEXT) verbatim. — Makes the 'lane returns a tight structured report or it isn't done' rule a deterministic gate rather than an advisory instruction a lane might skip under its own time pressure.
- **Notification (background job / subagent completion whose text matches /waiting for|will check back|monitoring/i)**: Auto-fire a SendMessage resume with the fixed text: 'No idle-wait: continue now in the foreground, do not end your turn to wait again' — budgeted at max 2 auto-resumes per lane before escalating to a needs-advice STOP. — Codifies the manual §7's #6 anti-pattern and the memory's most-recurring failure (passive-wait death, observed 5x+ across sessions) into an automatic action instead of the manager noticing and typing the same resume message by hand every time.

## Top risks

- Cost runaway from the new multi-GPU fleet if the billing circuit breaker (Pub/Sub -> Cloud Function stop-labeled-instances) isn't actually wired before the first autonomous multi-lane wave — soft tracking in gpu_fleet.md alone has already been shown (single-GPU era) to depend on the manager remembering to check.
- A lane's self-reported PASS with full_suite.failed>0 slipping through because the mechanical audit (lane-report-audit skill / PostToolUse hook) isn't in place yet — this exact failure mode cost ~8 inline Fable fixes and ~15 manual pytest runs last session and will recur at fleet scale unless enforced structurally, not just advisorially.
- Passive-wait death recurring on GPU/browser Sonnet lanes despite the strongest phrasing tried so far (documented as still occurring in the 2026-07-05 memory even with maximal anti-wait prompt language) -- the auto-resume hook mitigates but has not been proven to eliminate this at fleet scale with 3-4 simultaneous GPU lanes competing for the manager's attention.
- Held-out/protected label leakage risk increases with more concurrent lanes and GPUs -- more simultaneous work surfaces more temptation to touch Outdoor/Indoor/CVAT data for construction rather than scoring; the heldout_eval_ledger.md pre-registration gate must be enforced at dispatch time (safe-parallelism check step 2), not discovered after the fact.
- Two concurrently-dispatched lanes silently entangling via an uncommitted cross-lane import (the body_array_native.py precedent) becomes more likely, not less, as fleet-parallel lane count rises -- worktree isolation prevents file collisions but does not prevent a lane's code from importing another lane's in-progress module path if both worktrees share a Python environment/site-packages.
- Spot GPU preemption during a checkpoint-uncovered stage (a new pipeline stage that exceeds 15-30 min between durable checkpoints) could silently lose an hour of GPU-hours across several simultaneous lanes at once, multiplying the historical single-GPU version of this risk.
- Quota walls (Codex usage, GCP GPU SKU quota) can close unpredictably and the manager may over-commit to a slower/costlier fallback path for longer than necessary if it doesn't keep cheaply re-probing -- worse at fleet scale because a wall now blocks N parallel lanes instead of one.
- Doc-consistency and BUILD_CHECKLIST drift compounds with more concurrent lanes writing status/coordination artifacts simultaneously -- more simultaneous writers means more chances for two lanes to both add root docs or conflicting BUILD_CHECKLIST bullets in the same window before either commits.
- Agent Teams' documented failure modes (lead starts implementing itself instead of waiting for teammates; lead declares the team finished before all tasks are actually complete) are a real risk if Fable-5 is tempted to use the Agent Teams primitive for fleet coordination instead of the proven subagent-report-back pattern -- must be explicitly guarded against if that feature is ever adopted.
- Fable's own context can bloat across a multi-day, multi-GPU, many-lane autonomous run faster than in the single-GPU era simply because there are more structured reports, more STOP surfacing events, and more fleet-ledger updates to track -- structured note-taking (fresh dated memory docs, gpu_fleet.md) must be used proactively, not only at explicit session boundaries, or important STOP context could get silently compacted away.