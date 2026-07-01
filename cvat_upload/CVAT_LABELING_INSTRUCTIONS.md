# CVAT Labeling Instructions

Upload these four videos as four separate CVAT tasks. Keep frame step at `1` and do not resize or downsample the videos.

## Labels

Create exactly these labels:

- `player`
- `paddle`
- `ball`

## What To Label

## Fast CVAT Workflow

### Active players

Use `Rectangle` -> `Track` for each active `player`.

1. Go to the first frame where the player is visible.
2. Select rectangle drawing.
3. Choose label `player`.
4. Choose `Track`, not `Shape`.
5. Draw the box.
6. Move forward until the interpolated box is noticeably wrong.
7. Fix the box there. CVAT will make that frame a keyframe and interpolate between keyframes.
8. Repeat until the clip ends.

### Paddles

Use shorter `paddle` tracks only while the paddle is visible.

1. Go to a frame where the paddle is clearly visible.
2. Select rectangle drawing.
3. Choose label `paddle`.
4. Choose `Track`.
5. Draw a tight box.
6. Step forward a few frames.
7. If the paddle is still visible, adjust the box when needed.
8. When the paddle disappears, is too occluded, or you would be guessing, select that paddle track and turn on `Outside`.
9. When the paddle appears again, start a new `paddle` track unless you are very sure it is the same continuous object.

### Ball

For `ball`, use either short tracks or individual shapes. Short tracks are fastest only when the ball stays visible for several adjacent frames.

1. If the ball is visible for several frames in a row, use `Rectangle` -> label `ball` -> `Track`.
2. Draw a very tight box around the ball or visible blur.
3. Adjust every few frames while it remains visible.
4. As soon as it disappears or becomes a guess, select the ball track and turn on `Outside`.
5. If the ball only appears for one frame or is very jumpy, use `Rectangle` -> label `ball` -> `Shape` instead of a track.

CVAT's useful shortcuts are `K` for keyframe, `O` for Outside, and `M` for Merge in current CVAT track-mode docs. If your UI labels shortcuts differently, trust the visible button name over the shortcut.

### player

Use rectangle tracks for the four active on-court players.

- One CVAT track per physical player.
- Tight box around the visible body only.
- Include head, torso, arms, legs, and feet when visible.
- Do not include paddle, shadow, reflection, or another person.
- If a player is partly occluded, box only the visible part.
- If a player leaves and re-enters, continue the same track if you are sure it is the same person; otherwise start a new track.

Do not label spectators, coaches, refs, bystanders, people on other courts, or other background humans. Unlabeled people are treated as background. Only the four active on-court players get `player` labels.

### paddle

Use rectangles for visible paddles.

- If the full paddle face is visible, draw a tight box around the full visible paddle.
- If only part of the paddle is visible, draw a tight box around only the visible part.
- If the paddle is motion-blurred but still clearly a paddle, box the visible blur/object extent.
- If the paddle is fully hidden, too occluded to localize, or you are only inferring where it should be, do not label that frame.
- Do not include the player's hand, arm, body, or shadow.
- Add boxes more densely around swings and contact moments. If possible, label every visible paddle frame; at minimum add keyframes every 1-3 frames while the paddle is moving quickly.
- If using CVAT tracks for paddles, mark the object outside/end the track when it disappears or you are no longer sure it is the same paddle.

### ball

Use rectangles for the visible ball when you have time after players and paddles.

- If the ball is clearly visible, draw a very tight box around it.
- If the ball is motion-blurred but still clearly visible, box the visible blur streak.
- If the ball is behind a player, behind a paddle, lost in compression, or only inferable from its trajectory, do not label that frame.
- Prioritize frames around paddle contact, bounces, fast direction changes, and ball-tracker mistakes.
- If full-frame ball labeling is too slow, label every visible ball frame near contacts and otherwise every 5-10 frames.

## Export

Export each task as `CVAT for video 1.1`.

Put the exported zip files in:

`/Users/arnavchokshi/Desktop/pickleball/cvat_upload/exports/`

If CVAT also offers a COCO export, that is useful as a secondary export, but the primary export should be `CVAT for video 1.1` because it preserves frame indices and tracks.
