# ROUND-2 RE-REVIEW (retry) of data_steward_ledger_20260721 — refusal-coverage verification (read-only)

Prior review: review.json (REJECT — the audit accepted caller-declared metadata instead of
deriving facts from the ledger and the actual command). Fix claim: report_fix1.json. Target files
per spec.md FILE OWNERSHIP.

Task: verify the fixed audit CLI derives its decisions from the ledger + the actual command line,
using the existing shipped tests plus your own read-through of the code. For each of the eight
scenarios in the prior review's independent_probes list, confirm the current CLI now REFUSES the
dispatch, and identify WHERE in the code the refusal is derived (ledger field + argv analysis).
Then examine two additional input shapes for completeness: (a) a command referencing a protected
asset via a relative or normalized path variant; (b) a consumer contract that declares a subset
selection the command does not actually enforce. State for each whether the CLI refuses, and why.
Also verify: ledger schema v2 round-trip determinism; the 25-asset snapshot integrity; that the
five mixed-identity assets refuse training use absent a proven immutable clean selector.
VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES | REJECT.
