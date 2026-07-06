---
name: research-fanout
description: Use when a direction/approach decision is genuinely unclear and needs SOTA evidence, a competitive/technology survey, or a deep literature pass before committing. Formalizes the proven 5-phase Workflow pattern (search angles → primary-source deep-read → completeness critic → gap-fill + 2-vote adversarial refute → cited synthesis). Do NOT use for a single-fact lookup (just WebSearch) or when a standing ruling already answers it.
---

# research-fanout

The proven Fable research pattern (ran 3× on the pickleball project: pass-1 SOTA survey, pass-2
citation-graph deep dive, pass-3 court/fusion — see `runs/research_sota_20260705/`). Fable designs
the angles + our-context, a Workflow fans out Sonnet subagents, Fable rules on the synthesis.

## When
- A technology/approach choice with real trade-offs and no standing ruling.
- "Is there something better/newer we should use?" across a domain.
- Competitive/landscape intel. NOT for trivial facts or already-decided questions.

## How
1. **Reuse the scripts** (don't rewrite): `scratchpad/domain_sota_research.workflow.js` (angle-based,
   exploratory) or `scratchpad/citation_graph_deepdive.workflow.js` (seed each known paper, enumerate
   its sub-tasks + backward/forward refs — this is what caught the RacketVision-TrajPred miss). Copy
   into the repo if they need to persist.
2. **Fable writes the args**: `domain`, `our_context` (our real state + constraints), 5-6 `angles`,
   `must_fetch` seeds, and for the deep dive a `known_names` set so it only surfaces what's NEW.
3. **Launch in background**, one Workflow per domain, in parallel. Each ≈28-43 agents.
4. **On completion**: read the structured output (`missed_in_pass1`/`new_adoptions`/`key_facts`/
   `claim_verdicts`), persist to `runs/research_sota_<date>/`, fold verdicts into the roadmap. Fable
   re-derives every ranking — never forward a subagent's conclusion unexamined.

## Non-negotiables (learned the hard way)
- Every workflow script opens with `if (typeof args==='string') args=JSON.parse(args)` + a fail-loud
  missing-key check (4 workflows silently ran empty before this guard).
- Tier effort: scouts/critic at `medium`, synthesis/refutation at `high`. State an agent/token budget
  before launching — don't run uniform max-effort with no stopping rule.
- Adversarially verify load-bearing claims (2-vote refute). Mark survivors `[CORROBORATED]`, NOT
  `[VERIFIED]` (reserved for passed product gates).
- Watch for empty-result agents + patent/date misattributions (we hit 3) — cross-check before folding.
