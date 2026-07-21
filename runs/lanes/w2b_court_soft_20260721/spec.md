# w2b_court_soft_20260721 — the ONE blocking fix from the w2a review (verbatim)

Codex gpt-5.6-sol xhigh, SMALL. From runs/lanes/ultra_review_w2a_20260721/log.txt "Blocking fix":
1. Remove `court_presence_rejected` from `geometry_rejected` and the drop-reason vocabulary (player_selection.py:347, :1515; schema) — destructive drops require appearance REJECT plus independently justified persistence/motion evidence; NO hard court boundary anywhere on a destructive path (registered soft fusion S>=0.5 only).
2. Fix the test that enshrines the unregistered behavior (test_player_selection.py:1874) and ADD the biting regression: appearance REJECT + court_presence<0.5 + adequate persistence/motion ⇒ fragment stays UNBOUND (not dropped).
3. Update the lane report/REPORT overstatements the reviewer named, regenerate the projection hashes, rerun the FOCUSED suite only (no wide, no judges).
NO other changes. NO commits. Fixtures only. Report hunks + focused results.
