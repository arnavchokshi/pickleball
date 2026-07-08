# MICRO-LANE w6_charucoskip_20260708 — cross-lane repair: guard the charuco test for envs without cv2.aruco

## HARD RULES
- NO branches/commits/pushes; no board edits. Artifacts under runs/lanes/w6_charucoskip_20260708/ only.
- SMALLEST POSSIBLE CHANGE. This is a cross-lane courtesy repair of a CONCURRENT session's file (calv1 lane, commit fcf47ec90) under the don't-leave-main-red rule — do NOT refactor, do NOT change test semantics where the dep exists.

## FILE OWNERSHIP
- OWNED (this micro-repair only): tests/racketsport/test_calibrate_charuco_device.py.
- DO NOT TOUCH anything else, especially scripts/racketsport/calibrate_charuco_device.py and all other calv1 files (threed/racketsport/court_calibration.py etc.).

## DEFECT
tests/racketsport/test_calibrate_charuco_device.py::test_calibrate_charuco_device_recovers_synthetic_barrel_distortion_and_persists_profile fails on environments whose cv2 lacks the aruco contrib module (this Mac's .venv: `python -c "import cv2; hasattr(cv2,'aruco') and hasattr(cv2.aruco,'detectMarkers')"` -> False) — the test shells a subprocess that dies with CalledProcessError. Wave-6 close adjudication: 3245/1/25 with this as the ONLY failure.

## THE FIX (pinned)
Add the house-pattern availability guard (grep how other tests skip optional deps — e.g. importorskip("torch") pattern): skip this test (and any sibling test in the file that invokes the CLI needing aruco) with a clear reason like "cv2.aruco (opencv-contrib) not available" when cv2.aruco/detectMarkers is missing. If the test uses a subprocess, the guard must check availability in the TEST process before spawning. Behavior with contrib installed must be unchanged.

## ACCEPTANCE
1. On THIS env: the test file runs green (target test SKIPPED with the clear reason; other tests in the file unaffected or likewise properly guarded).
2. Grep-proof the guard matches the house optional-dep pattern.
3. Run the focused file + the scaffold/doc guardrails; nothing else changed.

## REPORT (schema-enforced): objective_result; full_suite (the focused file); CHANGES; HONEST ISSUES; proposed bullet TEXT (manager books, incl. the note to the CALV1 session); NEXT.
