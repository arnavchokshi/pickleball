# C1-C6 verdicts

| Candidate | Verdict | Evidence and limit |
|---|---|---|
| C1 | CONFIRMED | Pre-fix source set `requestingAccess` before an unbounded permission/configure chain; the button alone was disabled in that state and `blockedReason` existed only for blocked. No timeout existed. The fix adds an 8-second watchdog, late-result guards, a blocked banner, and Retry. Hosted C1 test is source-typechecked but simulator execution is sandbox-blocked. Physical occurrence remains unproven. |
| C2 | UNTESTABLE-WITHOUT-DEVICE | The real ARKit-to-AVCapture handoff and `startRunning()` hardware latency cannot be exercised without the phone. A hosted stall stub now proves the intended contract at source/typecheck level: an ARKit setup await cannot keep the UI disabled beyond the watchdog. The 4-second ARKit pass remains ordered before preview and remains advisory. |
| C3 | CONFIRMED | Pre-fix `startPreview` used `try? cameraOwnership.beginAVCapture()` plus `return`, and the queued wrapper used `_ = try?`, so an ownership failure was structurally swallowed. Both swallow sites are removed; typed owner/preview errors reach blocked UI. Whether this happened on the owner's phone is unproven. |
| C4 | CONFIRMED | Pre-fix screen `.task` and an early tap could independently call `prepare`; there was no in-flight identity/coalescing. The fix uses one preparation task/ID and logs coalescing. Hosted double-prepare regression is source-typechecked; device timing occurrence remains unproven. |
| C5 | REFUTED | A denied TCC result did not pin `requestingAccess`: pre-fix configure threw `permissionDenied` and the ViewModel caught it into blocked. The real defect was non-actionable wording, concurrent camera/mic request initiation, and no outer timeout. The fix requests camera then microphone, maps denial to `Enable Camera in Settings, then tap Retry.`, and keeps Retry enabled. Device TCC behavior remains unproven. |
| C6 | CONFIRMED | Pre-fix orientation updated only in viewport `onChange`; first appearance configured the `.landscapeRight` default. The Record screen now passes its initial viewport into prepare before configuration. Hosted orientation regression is source-typechecked; device portrait behavior remains unproven. |

`VERIFIED=0` remains binding. These verdicts distinguish confirmed code mechanisms from an unproven physical-device root cause.
