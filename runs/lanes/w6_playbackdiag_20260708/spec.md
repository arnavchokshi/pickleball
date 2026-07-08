# LANE w6_playbackdiag_20260708 — READ-ONLY diagnosis: 3D-world playback frame rate (OWNER CRITIQUE item #1; feeds the manager's P2-2 wiring ruling)

## HARD RULES (binding)
- STRICTLY READ-ONLY on the repo: you may NOT edit, create, or delete ANY file outside runs/lanes/w6_playbackdiag_20260708/. No exceptions — web/replay/ especially is another session's fence; you may read it, never touch it.
- No branches, no commits. No GPU. No network.
- Protected eval clips EVAL-ONLY; you read world manifests/mesh indexes (not labels) — that is allowed.
- Honest numbers with paths; estimates labeled as estimates and separated from measurements.

## OBJECTIVE
The owner critiqued the wave-5 worlds: 3D playback frame rate is LOW. Known mechanisms (from the wave-6 boot prompt): (1) the mesh layer is capped at ~100 selected frames per clip (`selected_mesh_frame_count`; the 290-600MB dense-monolith economy) so mesh steps at ~5fps on long clips; (2) BODY ball-aware frame scheduling skips frames. The manager must rule on a fix menu ordered by cost: render-side viewer interpolation (cheap, render-only, NEVER measured evidence) -> mesh-cap raise (disk/transfer linear) -> denser BODY scheduling (GPU linear); P2-2 latent decode enables principled inter-frame interpolation once wired. Your job: turn that menu into a MEASURED decision table.

## EVIDENCE TO READ FIRST
- runs/manager/wave6_boot_prompt.md (OWNER CRITIQUE section + P2-2 paragraph).
- The 4 wave-5 close-proof worlds under runs/lanes/w5_closeproof_20260708/{burlington,wolverine,outdoor,img1605_rerun2}/ (manifests, mesh indexes, PIPELINE_SUMMARY where present).
- web/replay/ source READ-ONLY: the existing 2x-FPS interpolation button + mesh layer rendering path.

## MEASURE (real numbers with paths, per clip where applicable)
1. Mesh density today: `selected_mesh_frame_count` vs total clip frames vs BODY-scheduled frames for each of the 4 close-proof worlds; the implied mesh-layer effective fps at playback speed 1x.
2. BODY scheduling density: which frames the ball-aware scheduler kept, gap-length histogram (max/median gap in frames), where the worst visual stutter would appear (longest gaps during rally play).
3. Mesh economy: actual bytes/mesh-frame from the shipped mesh indexes (and the monolith sizes they were cut from); linear extrapolation table for cap = 100 / 200 / 400 / all-scheduled frames — disk on VM, transfer to Mac, viewer load size.
4. Viewer capability today: what the existing 2x-FPS interpolation button actually interpolates (skeleton? trails? mesh?), grep-verified in web/replay src; whether the mesh layer has ANY inter-frame blending today; what data a render-side mesh interpolation would need that the current viewer bundle already has vs lacks (vertex correspondence? per-frame joints? MHR latents?).
5. GPU cost of denser BODY scheduling: from wave-5 logs (w5_closeproof, w5_p22latent dispatch logs), measured BODY sec/frame on H100; cost table for scheduling 2x / 4x / all frames on a typical clip.
6. P2-2 tie-in: from runs/lanes/w5_p22latent_20260707/, what the latent-decode path could interpolate BETWEEN BODY frames once wired (what exists today vs what wiring would require) — cite, do not speculate.

## DELIVERABLE
runs/lanes/w6_playbackdiag_20260708/playback_decision_table.md + .json: per-mechanism (render-interp / mesh-cap-raise / denser-scheduling / P2-2-latent-interp) — measured current state, quantified cost (disk/transfer/GPU-min/engineering scope), expected playback-fps gain, risk to the NEVER-measured-evidence rule (render-side output must be unmistakably cosmetic — how each option would be labeled/banded in the viewer so interpolated frames can never be read as measured), and which options compose. NO recommendation section — the manager rules; give the table + facts.

## REPORT (schema-enforced)
objective_result (PASS = table complete with all 6 measurement sections); full_suite N/A (read-only — say so); HONEST ISSUES (what you could not measure and why); artifact paths; proposed BUILD_CHECKLIST bullet text; NEXT.
