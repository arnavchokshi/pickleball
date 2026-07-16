# Post-fix RecordPath guard audit

| Chain location | Guard / early exit | User-visible consequence | Diagnostic consequence | Silent? |
|---|---|---|---|---|
| `DinkVisionTabBar` | Record disabled only while `status == requestingAccess` | Disabled state lasts at most the 8-second preparation or record-start watchdog; then Ready/Recording or blocked banner + Retry | State transition/watchdog log | No |
| `handleRecordTap` | Idle or blocked | Runs/coalesces bounded prepare, then starts only if Ready; otherwise retains visible block | Tap state and route logged | No |
| `handleRecordTap` | Requesting access | Joins the one bounded preparation instead of spawning another ARKit pass | Coalescing logged | No |
| `beginPreparation` | Existing preparation task | Both callers await one permission/configure/setup/preview attempt | Coalescing logged | No |
| `runPreparation` | Permission result arrives after timeout/replacement | Existing timeout/replacement banner remains visible; late result cannot flip Ready | Guard exit logged | No |
| `runPreparation` | Camera and/or microphone not authorized | Actionable Settings wording, blocked banner, enabled Record retry, visible Retry action | Permission terminal state + blocked transition logged | No |
| `runPreparation` | Configure/setup/preview/policy result arrives late | Existing loud terminal state remains; late completion cannot overwrite it | Guard exit logged | No |
| Preparation watchdog | Any permission/configure/setup/preview/policy await exceeds 8 seconds | `Camera setup took too long. Tap Retry.` and button re-enabled | Watchdog expiration logged | No |
| `CameraCaptureController.configure` | Landscape, permissions, camera, video/audio inputs, movie output, video connection, format | Typed error maps to actionable blocked banner | Specific failed guard logged | No |
| `performSetupPassIfNeeded` | Recording active | Recording continues visibly; setup is intentionally deferred | Skip reason logged | No |
| `performSetupPassIfNeeded` | Existing setup pass is fresh | Preview/Ready state remains valid; no redundant camera handoff | Skip reason logged | No |
| ARKit setup result unavailable | Advisory setup cannot produce a seed within its own 4-second pass | Recording path continues to AVCapture; unavailable alignment chip remains visible; AVCapture failure still blocks | Unavailable reason logged | No |
| `CameraCaptureController.startPreview` | Session already running | Success/no-op because preview is already live | Already-running guard logged | No |
| `CameraCaptureController.startPreview` | Stale AVCapture token or ARKit/AVCapture ownership conflict | Typed busy error maps to blocked banner + Retry | Active owner logged | No |
| `CameraCaptureController.startPreview` | `startRunning()` returns without a running session | Typed preview failure maps to blocked banner + Retry | Failure logged | No |
| `runRecordingToggle` | Existing record start/stop action | Caller joins one bounded action; no duplicate start/stop | Coalescing logged | No |
| `runRecordingToggle` | Already blocked | Existing reason and Retry remain visible | Retention logged | No |
| `runRecordingToggle` | Not Ready/Finished | `Camera is not ready. Tap Retry.` | Readiness guard logged | No |
| `CameraCaptureController.startRecording` | Already recording, missing policy/package/URL, portrait, unavailable camera lock | Typed error maps to blocked banner + Retry or explicit Stop instruction | Specific failed guard logged | No |
| `CameraCaptureController.stopRecording` | Movie output is not recording | Typed `notRecording` maps to blocked banner + Retry | Failed guard logged | No |
| Recording-action watchdog | Start/stop/policy read exceeds 8 seconds | `Recording action took too long. Tap Retry.`; late-start cleanup submits Stop | Watchdog and cleanup result logged | No |
| Recording delegate | Output error or descriptor absent | Blocked banner via callback; finished state only on successful sidecar path | Callback transition logged | No |

Zero audited rows are silent. Advisory ARKit unavailability remains non-blocking by design, preserving the camfix setup-pass ordering and the North Star rule that advisory work cannot stall recording.
