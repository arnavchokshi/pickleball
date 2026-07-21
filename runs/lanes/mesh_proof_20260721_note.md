# mesh_proof_20260721 — CORRECTED RULING (agent report authoritative, 2026-07-21)

SUPERSEDES my earlier "no-progress stall" note, which was WRONG on the mechanism:

1. **The w3a CUDA-mode fix is CONFIRMED WORKING**: Compute Mode Default on boot; the BODY
   subprocess got its second CUDA context and ran ~50 min of REAL GPU inference (54% util,
   9.5-10.4GB VRAM, 86 chunk buckets). Yesterday's 0-meshes blocker is CLOSED.
2. **NEW NAMED DEFECT**: host-memory exhaustion during full-tier1 BODY (RSS 76-97GB vs the
   a2-highgpu-1g's 83GB RAM / 0 swap) → kernel livelock (ssh/syslog starved). NOT a GPU bug.
   Remediation (agent-diagnosed): a2-highgpu-2g (170GB RAM, same GPU class, SPOT-quota OK) for
   full-mesh runs, AND/OR bound BODY host memory (streamed/chunked frame handling). Queued.
3. **MANAGER ATTRIBUTION**: the "stalled" state I observed was the post-hard-reset window while
   the agent was actively diagnosing/preparing a resize; my delete raced its recovery and cost
   the artifact pull. Same coordination error class as the pooling_wire run. NEW HARD RULE:
   never touch a VM owned by a live agent without SendMessage-ing the agent first.
4. Net: meshes still 0 (honest), but the path is now ONE known, remediable defect away.
   ~$2.9 total, teardown confirmed (involuntary but clean).
