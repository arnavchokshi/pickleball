# Tier-A owner spot-check pack (50 deterministic random labels)

Status: `VERIFIED=0`; these are bootstrap pseudo-labels, not truth.

Regenerate exactly:

```bash
.venv/bin/python runs/lanes/event_bootstrap_20260713/build_dataset.py --root . --seed 20260713
```

For each row, scrub roughly 0.25 seconds around the PTS and mark `true`, `false`, or `unclear`. Training spend clears only at >=47/50 apparent true contacts with no systematic source failure.

01. [x] clip=73VurrTKCZ8_rally_0002 | pts=40.967s | audio=0.688 | track=0.705 (dir=139.0deg, speed_change=0.44, near_player=None) | margin=0.073s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0002.mp4 | decision=false
02. [x] clip=73VurrTKCZ8_rally_0002 | pts=41.400s | audio=1.000 | track=0.726 (dir=149.1deg, speed_change=0.33, near_player=None) | margin=0.073s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0002.mp4 | decision=true(paddle) x=0.6081 y=0.4446 dt=-0.067s
03. [x] clip=73VurrTKCZ8_rally_0002 | pts=51.900s | audio=1.000 | track=0.756 (dir=110.0deg, speed_change=0.63, near_player=None) | margin=0.072s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0002.mp4 | decision=true(ground) x=0.4212 y=0.5587 dt=0s
04. [x] clip=73VurrTKCZ8_rally_0002 | pts=64.767s | audio=1.000 | track=0.802 (dir=137.0deg, speed_change=0.48, near_player=None) | margin=0.039s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0002.mp4 | decision=true(ground) x=0.2015 y=0.5827 dt=0.033s
05. [x] clip=73VurrTKCZ8_rally_0005 | pts=17.200s | audio=1.000 | track=0.794 (dir=159.1deg, speed_change=0.13, near_player=None) | margin=0.070s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0005.mp4 | decision=false
06. [x] clip=73VurrTKCZ8_rally_0008 | pts=1.133s | audio=0.982 | track=0.887 (dir=117.4deg, speed_change=0.10, near_player=None) | margin=0.005s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4 | decision=true(ground) x=0.518 y=0.5467 dt=-0.3s
07. [x] clip=73VurrTKCZ8_rally_0008 | pts=16.667s | audio=0.632 | track=0.837 (dir=134.2deg, speed_change=0.44, near_player=None) | margin=0.032s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4 | decision=true(ground) x=0.3254 y=0.5827 dt=-0.033s
08. [x] clip=73VurrTKCZ8_rally_0008 | pts=21.200s | audio=1.000 | track=0.774 (dir=156.4deg, speed_change=0.59, near_player=None) | margin=0.041s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4 | decision=true(paddle) x=0.5957 y=0.4365 dt=0.033s
09. [x] clip=73VurrTKCZ8_rally_0008 | pts=40.433s | audio=1.000 | track=0.722 (dir=173.8deg, speed_change=0.06, near_player=None) | margin=0.070s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4 | decision=false
10. [x] clip=73VurrTKCZ8_rally_0008 | pts=41.800s | audio=0.553 | track=0.794 (dir=174.1deg, speed_change=0.02, near_player=None) | margin=0.050s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4 | decision=false
11. [x] clip=73VurrTKCZ8_rally_0008 | pts=74.500s | audio=0.738 | track=0.778 (dir=158.9deg, speed_change=0.26, near_player=None) | margin=0.068s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/73VurrTKCZ8/73VurrTKCZ8_rally_0008.mp4 | decision=true(ground) x=0.5912 y=0.5687 dt=-0.233s
12. [x] clip=Ezz6HDNHlnk_rally_0002 | pts=33.325s | audio=1.000 | track=0.725 (dir=123.5deg, speed_change=0.43, near_player=None) | margin=0.039s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0002.mp4 | decision=false
13. [x] clip=Ezz6HDNHlnk_rally_0002 | pts=37.996s | audio=1.000 | track=0.804 (dir=174.5deg, speed_change=0.73, near_player=None) | margin=0.027s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0002.mp4 | decision=false
14. [x] clip=Ezz6HDNHlnk_rally_0002 | pts=108.191s | audio=0.749 | track=0.744 (dir=152.7deg, speed_change=0.23, near_player=None) | margin=0.013s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0002.mp4 | decision=true(paddle) x=0.4887 y=0.6688 dt=0.3s
15. [x] clip=Ezz6HDNHlnk_rally_0004 | pts=182.641s | audio=0.694 | track=0.792 (dir=167.2deg, speed_change=0.21, near_player=None) | margin=0.014s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0004.mp4 | decision=false
16. [x] clip=Ezz6HDNHlnk_rally_0006 | pts=127.044s | audio=1.000 | track=0.785 (dir=91.1deg, speed_change=0.54, near_player=None) | margin=0.016s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0006.mp4 | decision=true(paddle) x=0.4899 y=0.6008 dt=0.233s
17. [x] clip=Ezz6HDNHlnk_rally_0006 | pts=155.238s | audio=1.000 | track=0.930 (dir=98.2deg, speed_change=0.25, near_player=None) | margin=0.004s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0006.mp4 | decision=true(paddle) x=0.4414 y=0.6788 dt=0.3s
18. [x] clip=Ezz6HDNHlnk_rally_0007 | pts=77.286s | audio=1.000 | track=0.879 (dir=85.2deg, speed_change=0.04, near_player=None) | margin=0.049s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0007.mp4 | decision=false
19. [x] clip=HyUqT7zFiwk_rally_0001 | pts=19.400s | audio=0.767 | track=0.751 (dir=150.7deg, speed_change=0.50, near_player=None) | margin=0.028s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=true(ground) x=0.7219 y=0.6468 dt=0.3s
20. [x] clip=HyUqT7zFiwk_rally_0001 | pts=34.867s | audio=0.587 | track=0.824 (dir=127.6deg, speed_change=0.57, near_player=None) | margin=0.020s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=false
21. [x] clip=HyUqT7zFiwk_rally_0001 | pts=37.700s | audio=1.000 | track=0.823 (dir=174.2deg, speed_change=0.43, near_player=None) | margin=0.052s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=true(ground) x=0.7309 y=0.6208 dt=0.3s
22. [x] clip=HyUqT7zFiwk_rally_0001 | pts=109.200s | audio=0.782 | track=0.854 (dir=127.9deg, speed_change=0.49, near_player=None) | margin=0.008s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=false
23. [x] clip=HyUqT7zFiwk_rally_0001 | pts=244.800s | audio=0.626 | track=0.876 (dir=102.0deg, speed_change=0.06, near_player=None) | margin=0.036s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=true(paddle) x=0.2894 y=0.6328 dt=0s
24. [x] clip=HyUqT7zFiwk_rally_0001 | pts=690.133s | audio=1.000 | track=0.749 (dir=133.9deg, speed_change=0.53, near_player=None) | margin=0.069s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=false
25. [x] clip=HyUqT7zFiwk_rally_0001 | pts=694.200s | audio=1.000 | track=0.872 (dir=105.2deg, speed_change=0.66, near_player=None) | margin=0.043s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=false
26. [x] clip=HyUqT7zFiwk_rally_0001 | pts=820.633s | audio=1.000 | track=0.879 (dir=145.1deg, speed_change=0.42, near_player=None) | margin=0.050s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=true(paddle) x=0.7106 y=0.6308 dt=-0.167s
27. [x] clip=HyUqT7zFiwk_rally_0001 | pts=851.133s | audio=0.652 | track=0.788 (dir=172.6deg, speed_change=0.06, near_player=None) | margin=0.057s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=true(paddle) x=0.1587 y=0.6528 dt=-0.167s
28. [x] clip=HyUqT7zFiwk_rally_0001 | pts=961.067s | audio=1.000 | track=0.815 (dir=144.0deg, speed_change=0.61, near_player=None) | margin=0.038s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4 | decision=true(paddle) x=0.7512 y=0.6188 dt=0.267s
29. [x] clip=_L0HVmAlCQI_rally_0001 | pts=36.967s | audio=0.617 | track=0.715 (dir=77.9deg, speed_change=0.63, near_player=None) | margin=0.051s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0001.mp4 | decision=false
30. [x] clip=_L0HVmAlCQI_rally_0003 | pts=8.433s | audio=0.623 | track=0.848 (dir=130.3deg, speed_change=0.36, near_player=None) | margin=0.002s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0003.mp4 | decision=true(paddle) x=0.7106 y=0.5407 dt=-0.167s
31. [x] clip=_L0HVmAlCQI_rally_0007 | pts=24.433s | audio=1.000 | track=0.746 (dir=144.2deg, speed_change=0.84, near_player=None) | margin=0.053s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0007.mp4 | decision=true(paddle) x=0.5383 y=0.5267 dt=-0.233s
32. [x] clip=_L0HVmAlCQI_rally_0007 | pts=38.733s | audio=0.619 | track=0.761 (dir=179.1deg, speed_change=0.32, near_player=None) | margin=0.019s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0007.mp4 | decision=false
33. [x] clip=_L0HVmAlCQI_rally_0011 | pts=22.567s | audio=0.830 | track=0.804 (dir=106.4deg, speed_change=0.26, near_player=None) | margin=0.004s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0011.mp4 | decision=true(ground) x=0.562 y=0.5667 dt=-0.067s
34. [x] clip=_L0HVmAlCQI_rally_0011 | pts=49.433s | audio=1.000 | track=0.843 (dir=105.6deg, speed_change=0.31, near_player=None) | margin=0.029s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0011.mp4 | decision=true(paddle) x=0.4606 y=0.5146 dt=-0.2s
35. [x] clip=_L0HVmAlCQI_rally_0019 | pts=4.733s | audio=1.000 | track=0.875 (dir=112.4deg, speed_change=0.64, near_player=None) | margin=0.030s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/_L0HVmAlCQI/_L0HVmAlCQI_rally_0019.mp4 | decision=true(paddle) x=0.5991 y=0.4886 dt=-0.233s
36. [x] clip=wBu8bC4OfUY_rally_0002 | pts=45.933s | audio=0.601 | track=0.710 (dir=173.7deg, speed_change=0.43, near_player=None) | margin=0.019s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/wBu8bC4OfUY/wBu8bC4OfUY_rally_0002.mp4 | decision=false
37. [x] clip=zwCtH_i1_S4_rally_0001 | pts=44.233s | audio=1.000 | track=0.750 (dir=179.1deg, speed_change=0.38, near_player=None) | margin=0.061s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(paddle) x=0.5867 y=0.5146 dt=0.2s
38. [x] clip=zwCtH_i1_S4_rally_0001 | pts=45.933s | audio=0.807 | track=0.713 (dir=122.9deg, speed_change=0.73, near_player=None) | margin=0.066s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(paddle) x=0.5586 y=0.5227 dt=0.3s
39. [x] clip=zwCtH_i1_S4_rally_0001 | pts=107.600s | audio=0.990 | track=0.870 (dir=95.3deg, speed_change=0.40, near_player=None) | margin=0.065s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(paddle) x=0.5642 y=0.5287 dt=0.133s
40. [x] clip=zwCtH_i1_S4_rally_0001 | pts=137.567s | audio=0.992 | track=0.896 (dir=136.6deg, speed_change=0.23, near_player=None) | margin=0.062s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(other) x=0.5045 y=0.5787 dt=-0.167s
41. [x] clip=zwCtH_i1_S4_rally_0001 | pts=144.267s | audio=0.854 | track=0.820 (dir=147.3deg, speed_change=0.57, near_player=None) | margin=0.065s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=false
42. [x] clip=zwCtH_i1_S4_rally_0001 | pts=224.233s | audio=0.736 | track=0.833 (dir=157.2deg, speed_change=0.34, near_player=None) | margin=0.037s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(ground) x=0.5631 y=0.6008 dt=0.3s
43. [x] clip=zwCtH_i1_S4_rally_0001 | pts=270.900s | audio=1.000 | track=0.842 (dir=136.4deg, speed_change=0.38, near_player=None) | margin=0.048s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(ground) x=0.491 y=0.6008 dt=-0.133s
44. [x] clip=zwCtH_i1_S4_rally_0001 | pts=307.567s | audio=0.560 | track=0.763 (dir=149.1deg, speed_change=0.32, near_player=None) | margin=0.039s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=false
45. [x] clip=zwCtH_i1_S4_rally_0001 | pts=424.200s | audio=0.856 | track=0.799 (dir=115.8deg, speed_change=0.21, near_player=None) | margin=0.072s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=false
46. [x] clip=zwCtH_i1_S4_rally_0001 | pts=439.233s | audio=0.655 | track=0.859 (dir=140.7deg, speed_change=0.32, near_player=None) | margin=0.022s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=false
47. [x] clip=zwCtH_i1_S4_rally_0001 | pts=519.167s | audio=0.859 | track=0.836 (dir=131.9deg, speed_change=0.43, near_player=None) | margin=0.022s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=false
48. [x] clip=zwCtH_i1_S4_rally_0001 | pts=524.200s | audio=0.849 | track=0.907 (dir=116.8deg, speed_change=0.21, near_player=None) | margin=0.038s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(ground) x=0.4842 y=0.5967 dt=-0.3s
49. [x] clip=zwCtH_i1_S4_rally_0001 | pts=564.200s | audio=1.000 | track=0.889 (dir=88.0deg, speed_change=0.13, near_player=None) | margin=0.038s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=true(paddle) x=0.5597 y=0.5587 dt=-0.033s
50. [x] clip=zwCtH_i1_S4_rally_0001 | pts=567.500s | audio=1.000 | track=0.825 (dir=146.7deg, speed_change=0.58, near_player=None) | margin=0.023s | video=/Users/arnavchokshi/Desktop/pickleball/data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4 | decision=false

---

## 2026-07-15 owner review ruling (Track C manager transcription)

Provenance: owner-labeled 2026-07-15 via the half-speed clip review page (clips +/-0.6s
around each labeled PTS; richer v2 flow: typed decision paddle|ground|other|none|unclear +
2D contact click normalized to the displayed video + frame-offset nudge recorded as dt
seconds in SOURCE time relative to the labeled PTS). Raw bytes preserved at
`owner_spot_check_results_20260715.json` (sha256 dcb0c00e46e943fd3a4d7b214551869fb88e516219eb6e0067d5fceb7c21f7a1);
manager re-tallied the file independently before transcription.

Tally (manager-verified from the file): paddle 17, ground 11, other 1, none 21, unclear 0
-> apparent true contacts 29/50 (58%).

**GATE: FAIL.** Pre-registered bar was >=47/50 with no systematic source failure. 29/50
misses decisively, and every source fails broadly (73Vur 7/11, Ezz6H 3/7, HyUqT 6/10,
_L0HV 5/7, wBu8b 0/1, zwCtH 8/14) — a systematic auto-labeler failure, not one bad source.
Timing is also poor inside the true subset: 15/29 true contacts have |dt| >= 0.2s (8 pinned
near the +/-0.3s window edge), so even true windows are badly mistimed.

Consequences (Track C ruling, 2026-07-15):
- Training spend on the Tier-A bootstrap labels REMAINS BLOCKED.
- The audio-x-track two-signal auto-labeler is REJECTED as a training-label source at
  current thresholds; its honest 0.274 chance-excess caveat proved out.
- The public-data pretrain leg (real GT: jhong93/spot, ShuttleSet, OpenTTGames, etc.) is
  NOT affected by this failure.
- The 50 owner-reviewed rows (29 typed contacts with 2D click locations and timing offsets
  + 21 hard negatives) are the FIRST owner-verified pickleball event labels. They are
  reserved as a protected eval seed and must NEVER become training data.
- Proven cheap labeling channel: the owner produced 50 rich rows in ~20 minutes via the
  clip-review page; scaling that flow is the obvious label-supply lane.
