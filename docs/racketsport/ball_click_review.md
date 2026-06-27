# Ball Click Review

This is a prototype-only human review loop for bad TrackNet output in videos
with nearby courts, background balls, or occlusions. It does not create final
ground truth.

Export 30 frames for one clip:

```bash
python scripts/racketsport/export_ball_click_review.py \
  --video runs/eval0/prototype_gate_h100_v2/<clip>/tracknet_smoke_0000_0010/input_0000_0010.mp4 \
  --clip <clip> \
  --out runs/eval0/prototype_gate_h100_v2/ball_click_review_30/<clip>
```

Open `review.html`, click the target-court ball once per frame, and use
`Mark missing` or `Mark occluded` when the target-court ball is not visible.
Use `A` or the left arrow for the previous frame, and `D` or the right arrow
for the next frame.
The browser downloads `ball_points.json`; put that file back in the same clip
folder or send it to the lead.

The JSON coordinate frame is original video pixels, not display-scaled pixels:

```json
{
  "artifact_type": "racketsport_ball_click_review",
  "coordinate_frame": "image_pixels_video_space",
  "items": [
    {
      "frame_index": 0,
      "t": 0.0,
      "image": "images/frame_000000.jpg",
      "ball_xy": null,
      "visible": null,
      "notes": ""
    }
  ]
}
```

Current `prototype_gate_h100_v2` review bundles:

| Clip | Review HTML |
| --- | --- |
| `burlington_gold_0300_low_steep_corner` | `runs/eval0/prototype_gate_h100_v2/ball_click_review_30/burlington_gold_0300_low_steep_corner/review.html` |
| `wolverine_mixed_0200_mid_steep_corner` | `runs/eval0/prototype_gate_h100_v2/ball_click_review_30/wolverine_mixed_0200_mid_steep_corner/review.html` |
| `outdoor_webcam_iynbd_1500_long_high_baseline` | `runs/eval0/prototype_gate_h100_v2/ball_click_review_30/outdoor_webcam_iynbd_1500_long_high_baseline/review.html` |
| `indoor_doubles_fwuks_0500_long_mid_baseline` | `runs/eval0/prototype_gate_h100_v2/ball_click_review_30/indoor_doubles_fwuks_0500_long_mid_baseline/review.html` |

The same four clips also have target-court filtered TrackNet smoke overlays
under each clip's `tracknet_smoke_0000_0010/` directory:

- `ball_track_target_court_120px.json`
- `ball_track_target_court_120px_summary.json`
- `ball_track_target_court_120px_overlay_h264.mp4`

The 120 px margin is a prototype setting to tolerate imperfect fisheye and
corner placement while still suppressing obvious balls from neighboring courts.
