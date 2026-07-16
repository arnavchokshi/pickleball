# Fix note — 2026-07-16

Updated only `test_events_before_frames_makes_cold_mesh_plan_contact_dense`'s fake materializer to return the same validation fields as the real materializer, with expected and materialized frame indexes derived from the supplied schedule, empty missing and unexpected lists, and `equal: true`; the production frame-schedule completeness contract was not weakened.

- `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_best_stack_resolution.py -q` — 9 passed in 4.96s, literal EXIT 0.
- `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_spine_stage_contract.py tests/racketsport/test_process_video.py -q` — 176 passed in 43.76s, literal EXIT 0.
