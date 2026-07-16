# Owner 60-second morning test — record button (iPhone 14 Pro B03696B6)

Build: manager installs the freshest staged build first (wave-2 visibility build if adopted
overnight, else the wave-1 fixed build already on the phone from 2026-07-16 00:34).

Install/launch (manager or owner, phone plugged in + unlocked):
```bash
xcrun devicectl device install app --device B03696B6-6481-5FCD-A79C-105DA3F08F98 /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ios_recordpath_20260715/device_build/Pickleball.app
xcrun devicectl device process launch --device B03696B6-6481-5FCD-A79C-105DA3F08F98 --terminate-existing com.arnavchokshi.pickleball
```
(Optional but ideal — capture diagnostics during the test, run in a second terminal:)
```bash
xcrun devicectl device process launch --device B03696B6-6481-5FCD-A79C-105DA3F08F98 --terminate-existing --console com.arnavchokshi.pickleball 2>&1 | tee /tmp/owner_morning_recordpath.log
```

## The 60 seconds (wave-2 build expectations — staged build includes both waves)
1. Cold-launch lands on the Record tab, phone in your hand PORTRAIT first.
   EXPECT within ~2 s: a large cream "Rotate to landscape — Turn your iPhone sideways to unlock
   recording." card with a Retry button, PLUS the top banner "Rotate to landscape to record.
   Tap to retry" (sim-verified screenshot: runs/lanes/ios_recordvis_20260716/after_portrait_cold.png).
2. Tap the yellow button once while still portrait.
   EXPECT: an immediate visible reaction — button press+wobble, guidance card emphasis pulse, and
   a warning haptic (with Reduce Motion on: a static thick highlight instead of motion). Never
   silence.
3. Rotate the phone to LANDSCAPE.
   EXPECT: blocker clears; camera preview live; yellow button ready (no banner). Allow up to 8 s;
   if anything fails you will see a banner with words — read/screenshot the words.
4. Tap the yellow button once.
   EXPECT within ~1-2 s: button turns RED stop state + elapsed time pill counting up.
5. Record ~10 seconds. Tap the red stop.
   EXPECT: saving/loader card, then capture saved; the new clip appears in the Replays tab list.
6. Done. If ANY step shows a banner instead: the banner text IS the diagnosis — screenshot it.
   If ANY tap produces literally no visible change: screenshot the screen + note the step number
   (this would falsify the wave-2 contract; logs at /tmp/owner_morning_recordpath.log).

## What we already know going in (2026-07-16 00:43 device evidence)
Camera+mic authorized; record path healthy; portrait was the blocker; total time from launch to a
correct loud blocked state: 1.4 s. Step 4 (landscape tap → recording) is the only unproven hop.
