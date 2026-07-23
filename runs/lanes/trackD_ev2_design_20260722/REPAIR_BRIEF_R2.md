# Repair round 2 — six residual findings (rescore verdict REJECT; 7 of 13 round-1 fixes verified)

Read the reviewer's DETAILED interim rows in runs/lanes/trackD_ev2_review_20260722/log_rescore.txt
(the final report_rescore.json is summary-shaped; the streamed interim JSON in the log carries the
full per-finding evidence). Verified-fixed and NOT to be reopened: EV2-R1, R2, R3, F1, F2, F7, F8.

Fix EXACTLY these six, with EXECUTABLE BOUNDARY TESTS (the reviewer explicitly rejected
presence-only tests — each test must EXECUTE the violating path and prove the hard failure at the
required boundary):

1. EV2-R4 (REJECT-class, finetune_event_head.py:2185): the final-step recipe lock fires at the
   wrong boundary — the reviewer's adversary reached an INPUT READ before rejection. Required: a
   divergent OR absent registered value (sqrt-frequency, dilation=1, temporal=0.25, offset=0.2)
   must hard-fail with a typed error BEFORE ANY input artifact read (manifests, media, checkpoints,
   lock files). Structure: validate the complete registered-recipe argument set as the FIRST act of
   final-step mode. Boundary test: instrument/monkeypatch input-open paths; assert the typed
   rejection occurs with ZERO input opens for (a) each single divergent value, (b) each single
   absent value, (c) legacy defaults wholesale.
2. EV2-F3 (finetune_event_head.py:1501): make the firing-rate media inventory complete and
   independent per the reviewer's row (no manifest-conditioned 23-of-38 style subset without a
   registered derivation); re-register the exact inventory rule; boundary test executes the
   inventory builder against a synthetic manifest and proves the registered rule.
3. EV2-F4 (finetune_event_head.py:2593): enforce the registered Stage-F mining+optimizer cap in
   code (not only in the VM plan) — the cap must abort the stage, test must prove abort fires.
4. EV2-F5 (VM_RUN_PLAN.md:32): commit preflight must PROVE the reviewed code bytes are in the
   frozen commit — compare CODE_SHA256SUMS against `git show $RUN_COMMIT:<path>` hashes for every
   listed file, not against the working tree.
5. EV2-F6 (VM_RUN_PLAN.md:254): complete teardown routes for EVERY post-create failure branch
   (incl. setup/bootstrap failures before rails are armed) — each branch ends in delete+confirm.
6. EV2-F9 (VM_RUN_PLAN.md:1553): make the PASS-only best-stack handoff atomic — no partial
   production mutation on any failure path (stage to temp + single atomic move/apply, or emit a
   handoff artifact the manager applies; state which and register it).

Rules unchanged: judge frozen; fences unchanged; focused suite + repair tests + new boundary tests
all green; refresh CODE_SHA256SUMS/CONTROL_RESULTS/REGISTRATION text where operative values
changed (cap enforcement, inventory rule); DO NOT touch the manager addendum section. After your
report is written, make NO further edits (tree freezes for the canonical suite). Report:
report_repair2.json, per-finding table with boundary-test evidence.
