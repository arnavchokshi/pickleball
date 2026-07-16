# Record-tab loud-state and per-tap guard audit

| State / chain location | Guard or route | Persistent user-visible consequence | Per-tap visible reaction | Reduced-motion behavior | Silent? |
|---|---|---|---|---|---|
| Record control, idle | Cold-launch preparation has not started yet | Raised yellow control is fully opaque and enabled | Immediate press/depress, then existing bounded prepare/start route | Same press/depress without ambient breathing | No |
| Record control, requestingAccess | Preparation is already active | High-contrast `Setting up camera…` guidance remains visible | Immediate press/depress, warning haptic, button wobble + guidance highlight; tap then joins the one preparation | No wobble/pulse; thick static red/yellow emphasis remains for 650 ms | No |
| Record control, blocked (any reason) | Capture correctness prevents start | Persistent ink-on-cream blocker card plus prominent bordered banner and Retry | Immediate press/depress, warning haptic, button wobble + card/banner highlight; retry remains bounded | No wobble/pulse; thick static red/yellow emphasis remains for 650 ms | No |
| Record control, portrait blocker | Landscape policy fails before camera configuration | Persistent rotate glyph, `Rotate to landscape`, exact blocked-reason banner, and Retry | Same warning reaction as every blocked tap; landscape enforcement is not weakened | Same static emphasis; no motion | No |
| Record control, ready | Start is allowed | Ready state and live preview remain visible | Immediate press/depress, impact haptic, then existing bounded start flow | Press/depress only | No |
| Record control, recording | Stop is allowed | Red stop control and elapsed pill remain visible | Immediate shadow/depress response, impact haptic, then existing bounded stop flow | Static pressed shadow, no start wobble | No |
| Record control, finished | A new start is allowed | Saved state remains visible | Immediate press/depress, impact haptic, then existing bounded start flow | Press/depress only | No |
| SwiftUI button enablement | Former `.disabled(!canRecordFromTab)` gate | `DinkVisionRecordButton` always exists, is enabled, and remains hittable by contract in all five tested states | SwiftUI can no longer swallow requesting/blocked taps at the control boundary | Identical hit target and enabled semantics | No |
| `noteRecordControlTap` | Every record-control action enters synchronously | Published monotonically increasing feedback sequence | Every state maps to a reaction whose `hasVisibleConsequence` is true | Policy requires `.staticHighlight` for blocked/preparing | No |
| `handleRecordTap`, idle or blocked | Needs prepare before toggle | Preparing surface appears, then Ready/Recording or persistent blocked UI | Control feedback is published before the async route starts | Static feedback already visible before async route | No |
| `handleRecordTap`, requestingAccess | Existing preparation task | Setting-up surface remains until terminal state | Feedback publishes before coalescing with the bounded attempt | Static setting-up emphasis | No |
| `handleRecordTap`, ready/recording/finished | No preparation needed | Existing start/stop/save UI stays visible | Control press/depress precedes bounded recording toggle | Static press state | No |
| `beginPreparation` | Existing preparation task | Setting-up surface remains visible; watchdog is still armed | Any additional tap gets its own feedback event while joining the one task | Static emphasis on each tap | No |
| Permission denial | Camera and/or microphone unavailable | Persistent actionable blocker card + banner + Retry | Each tap warns and visibly emphasizes before retrying permissions | Static emphasis | No |
| Preparation watchdog | Any prepare await exceeds 8 seconds | Persistent `Camera setup took too long. Tap Retry.` card/banner | Each later tap visibly reacts and retries through the same bounded route | Static emphasis | No |
| Late preparation completion | Result arrives after timeout/replacement | Existing loud terminal state is retained | Any tap reacts against the current visible state | Static emphasis when blocked/preparing | No |
| Configure/setup/preview failure | Typed capture error | Specific persistent blocker card, banner, Retry, and VoiceOver announcement | Each tap visibly reacts before bounded retry | Static emphasis | No |
| ARKit advisory unavailable | Setup seed unavailable but recording may proceed | Existing unavailable alignment chip remains visible; AVCapture path continues | Record tap still depresses and follows the startable route | Press/depress only | No |
| `runRecordingToggle`, duplicate action | Existing start/stop task | Current setting-up/recording/saving surface remains | Each control tap already reacted before joining the bounded action | Corresponding static control state | No |
| Record-start readiness guard | State is not ready/finished | Persistent `Camera is not ready. Tap Retry.` card/banner | Originating tap already reacted; future taps repeat the warning reaction | Static emphasis | No |
| Recording-action watchdog | Start/stop/policy exceeds 8 seconds | Persistent timeout card/banner; late-start cleanup still submits Stop | Future taps are enabled and visibly react before retry | Static emphasis | No |
| Recording delegate failure | Output or descriptor fails | Persistent callback-specific blocker card/banner + Retry | Future taps remain enabled and visibly react | Static emphasis | No |
| Blocked-state entry | Status changes into a new blocked reason | Card/banner render and `UIAccessibility.post(.announcement, ...)` fires | Tap feedback and announcement are independent, so neither suppresses the other | VoiceOver announcement and static UI are unchanged | No |
| Blocker clears | Status leaves blocked | Blocker card/banner disappear immediately; the current ready/recording/preparing surface replaces them | The tap that initiated the change already produced visible feedback | No exit animation when Reduce Motion is enabled | No |

Zero audited rows are silent. The landscape gate, ARKit setup-pass ordering, preparation/record
watchdogs, coalescing, and typed capture errors remain intact.

