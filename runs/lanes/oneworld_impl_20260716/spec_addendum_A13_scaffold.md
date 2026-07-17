# ADDENDUM A13 — scaffold-index naming fix (URGENT, repo-wide red; then continue where you left off)

Your docs schema filenames violate the checked-in scaffold index pattern and are breaking
tests/racketsport/test_scaffold_tool_index.py::test_real_scaffold_tool_index_matches_checked_in_schema
REPO-WIDE right now (other tracks' wave closes are hitting it). Exact error:
  $.tools[66].matching_schema: value 'docs/racketsport/one_world_v1.schema.json'
  does not match pattern '^docs/racketsport/.+_schema\.json$'

FIX (small, do it FIRST, before resuming other work):
1. Rename (house underscore convention, cf. pipeline_contracts_schema.json):
   docs/racketsport/one_world_v1.schema.json            -> one_world_v1_schema.json
   docs/racketsport/one_world_v1_metrics.schema.json    -> one_world_v1_metrics_schema.json
   docs/racketsport/one_world_v1_validation.schema.json -> one_world_v1_validation_schema.json
2. Update the three SCHEMA_OVERRIDES values in scripts/racketsport/list_scaffold_tools.py to the
   new names (your RELATED_TEST_OVERRIDES/TASK_HINTS entries are fine as-is).
3. Update every reference to the old filenames in your module/CLIs/tests (validate CLI likely
   embeds its schema path).
4. VERIFY with a REAL UNPIPED exit code (no pipes — piped $? is a known repo trap):
   MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_scaffold_tool_index.py -q
   then `echo REAL_EXIT=$?` as the immediately-next command. Must be 0. Record the literal
   output + exit code in your report (this is now acceptance item A13; the manager will rerun
   it personally before ruling — a report claiming green that isn't reproducible is a reject).
5. ALSO record in honest_issues: the window during which the repo-wide scaffold test was red due
   to this lane (Track C triage flagged it), and that the fix landed.

RULE REMINDER: do NOT attribute test_scaffold_tool_index failures to "concurrent-lane
volatility" — this one is YOURS. Wide-suite attribution must name this lane as the cause+fixer.

Then continue exactly where you left off (A1-A12 all still stand).
