# w1c_unknown_bridge_20260721 — bridge: finetune consumes the loader's frame_loss_mask

Codex gpt-5.6-sol xhigh, SMALL. w1b added a per-frame UNKNOWN mask (frame_loss_mask) to datasets.py (schema-v2); w1a's finetune_event_head.py doesn't consume it — UNKNOWN frames would still train as background in the A/B/C. Fix EXACTLY:
1. finetune_event_head.py: consume frame_loss_mask from the loader batch and EXCLUDE masked frames from the loss (compose with the existing loss_validity_mask and the post-weighting pseudo cap from w1a — do not disturb either).
2. Test: a batch with UNKNOWN frames contributes ZERO loss/grad from those frames (and the pseudo-cap math is unchanged when masks are all-valid — byte-identical loss on a mask-free batch).
NO other changes. NO commits. NO judge peeking (fixtures only). Focused tests green; report the hunk.
