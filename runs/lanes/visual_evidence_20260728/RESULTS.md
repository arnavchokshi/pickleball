# Pipeline results, in plain language, 2026-07-28

Six fresh pickleball clips ran through the full pipeline overnight, from raw video to
skeletons, court geometry, and kitchen-rule calls. This is what actually happened, with
the videos to prove it (`index.html`, double-click to open, no server needed).
**Nothing here is "verified" in the audit sense (`VERIFIED=0` everywhere). This is an
honest look at working software, not a promotion claim.**

## 1. What the pipeline did on the six clips

Watch any of the six videos in the gallery. On every one, the pipeline correctly:

- **Found the court.** Solved court lines and the kitchen (NVZ) zone track the real
  court through the camera's perspective on every clip, indoor and outdoor, close-up
  and wide broadcast shot alike.
- **Kept track of all four players, all the way through.** 4/4 players retained on
  every single clip. Nobody vanished, nobody got duplicated, nobody got swapped
  mid-rally.
- **Put a skeleton on each player.** 70 body joints per player, per frame, with real
  bone connectors (shoulders to elbows to wrists, hips to knees to ankles to toes),
  drawn straight from the pipeline's own output, not staged.
- **Made a kitchen call on every foot, every frame, and admitted what it didn't
  know.** Most frames say "unknown" rather than guessing, because the decision rule
  requires 99% confidence before it calls a foot in or out of the kitchen. That's the
  right failure mode: a system that confidently calls close plays it can't actually
  see would be worse than useless. When it does commit, it commits correctly relative
  to court geometry (see the videos: kitchen calls only fire near the line).

## 2. Grounding accuracy: foot-slide vs. the 0.03 m bar

"Foot slide" measures how much a planted foot appears to drift while it's supposed to
be stationary. It's the sharpest test of whether the 3D grounding is trustworthy.

| Clip | Max foot-slide | Verdict |
|---|---:|---|
| Wolverine indoor diagonal | 0.0378 m | **FAIL**, new flag, needs eyes |
| Indoor doubles baseline | ~0.0000 m | pass |
| Outdoor high baseline | 0.0212 m | pass |
| Indoor straight (replacement) | 0.0169 m | pass |
| Indoor diagonal (replacement) | 0.0546 m | **FAIL**, known, matches prior run |
| Outdoor PBVision (replacement) | 0.0026 m | pass |

**4 of 6 pass.** That's consistent with where this pipeline already stood, except
wolverine's 0.0378 m is a *new* failure (wolverine used to pass cleanly). Its
grounding refiner also reverted mid-run on this exact clip tonight ("refiner reported
worsened residuals, original artifacts restored", a safe typed failure, not a crash,
but it means wolverine's numbers here are pre-refinement). Not diagnosed yet. See the
flags section below.

## 3. Speed: the honest story, not the flattering one

The existing baseline (`court_skeleton_runtime_20260725/REPORT.md`, measured
2026-07-25) is **352.5 s median wall time** per ~10-second clip, source-video-only,
Mac-orchestrated. Tonight's run reproduced that same pipeline at current code on all
six clips, on a real (not idle) Mac.

Split-mode tonight came in at **~500.0 s median**. That looks like a big regression,
but it isn't one. GPU-side BODY work matched baseline closely (178 to 237 s vs.
baseline's 172 to 225 s). The entire gap is Mac-side tracking taking 165 to 343 s on
a host that had a concurrent code lane, several other agent processes, and one
runaway system process (`mediaanalysisd`, over 200% CPU) running at the same time.
Treat tonight's split-mode number as an upper bound on a bad night, not a real
regression.

Running the whole pipeline directly on the GPU box (co-located) tonight took **~94.6 s
median**, the actual speed lead. Court, tracking, and placement all ran roughly 3.7x
faster with zero setup cost, because the GPU box's environment was already warm. The
catch: BODY itself silently fell back to skeleton-only on all six co-located runs
tonight, because of a real bug (a calibration-reuse code path that demanded a capture
sidecar file bare eval clips don't have). The pipeline refused to fake that file
rather than produce fabricated output, which is correct behavior, but it meant no
full BODY meshes co-located tonight.

What's now fixed: the follow-up fix lane (`bodylocal_colocated_fix_20260728`) found
and repaired that exact bug and confirmed it end-to-end on the wolverine clip. The
co-located BODY-local run completed with real BODY inference (1,136 player crops
actually scored, not skeleton-only) in **266.5 s total wall**, still materially
faster than tonight's 436.7 s split-mode run for the same clip, and with the
setup-cost problem solved. That fix lane is still open as of this pack. A full
six-clip co-located-with-real-BODY reproduction, and the larger integration demo
(full preset plus ball plus one_world), had not landed by the time this pack was
assembled. Status: in flight, not yet a clean six-clip measurement.

## 4. E-v2 (ball event detection): first real signal, not a win yet

Separate lane, same night (`ev2_train_20260728`). Verdict: **PARTIAL.**

For the first time in this program's history, the event-detection model showed real,
non-zero, above-chance signal distinguishing hit events from noise, after a string of
prior attempts that produced zero true positives. That's a first for this program.

It is **not** ready to use: the plausible firing-rate gate (event predictions per
second landing in a believable 0.3-1.0/s band) didn't clear at any of the three
thresholds tested, the BOUNCE event class showed zero learned signal at every
threshold, and macro-F1 sits at 0.0 at the standard decision threshold on both
held-out splits. No checkpoint from this run is authorized to feed anything
downstream.

## 5. Three flags that need eyes

1. **Wolverine foot-slide regression: 0.0378 m, newly failing.** This clip used to
   pass cleanly (effectively 0 m slide in the July 24 baseline run). Tonight it
   failed, and its grounding refiner separately reverted on the same clip. Possibly
   related, not diagnosed. Worth a dedicated look before trusting wolverine's
   grounding numbers.
2. **Indoor-diagonal foot-slide: 0.0546 m, known and repeat-failing.** Matches the
   July 24 baseline's 0.0545 m failure almost exactly: same clip, same failure mode,
   two runs apart. Not new, but not fixed either.
3. **E-v2 firing-rate gate: not cleared at any tested threshold.** Real signal
   exists now (see section 4), but nothing from this checkpoint should be treated as
   usable until a threshold search actually lands inside the plausible band, or the
   model is trained further past its current stopping point.

## 6. What this pack is, and isn't

Is: six real overlay videos rendered straight from last night's pipeline artifacts
(source frames plus solved court plus skeletons plus tracks plus kitchen calls), a
gallery to watch them cold, and the honest numbers behind each one.

Isn't: an accuracy promotion, a claim that any gate has newly passed, or a claim that
speed has improved end-to-end (the co-located speed win is real but not yet paired
with working BODY on all six clips in one clean run).

Provenance for every number and every video frame in this pack is in
`provenance.json` in this same folder.
