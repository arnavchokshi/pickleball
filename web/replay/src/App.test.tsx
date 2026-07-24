import { describe, expect, it, vi } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import * as AppModule from "./App";
import {
  bodyJointSkeletonForFrame,
  bodyMeshOpacityFromBlendWeight,
  bodyMeshMaterialForTrustBadge,
  adjacentBodyMeshWindow,
  cameraPresetPose,
  cameraTransitionDurationMs,
  cocoWholeBodyCoreBoneNames,
  createSolidBodyMeshGeometryCache,
  defaultReviewStartTime,
  contactReadoutText,
  courtBounds,
  createFollowCameraPoseBuffer,
  geometryForSolidBodyMeshFrame,
  handJointPointsForFrame,
  hasExplicitReviewStartTime,
  manifestUrlFromSearch,
  loadCameraPreference,
  persistCameraPreference,
  entityLayerEmptyStates,
  degradedReasonsFromNotes,
  fittedCourtCameraPose,
  loadOptionalArtifact,
  paddleRenderGeometryForFrame,
  shouldRenderReplayScenePointClouds,
  TimelineStrip,
  ReplayAuxiliarySections,
  shotsEmptyText,
  timelineAbsenceText,
  updateFpsSample,
  updateFollowCameraPoseBuffer,
  vertexDebugPointsForFrame,
} from "./App";
import { parseBodyMesh, solidBodyMeshFramesForTime } from "./viewerData";
import { timelineChaptersFromMarkers } from "./viewerData";
import type { ActivePaddleFrame, BodyMesh, TimelineChapter, TimelineMarker, VirtualWorld, VirtualWorldPaddleFrame, VirtualWorldPlayer } from "./viewerData";
import { DEFAULT_VIEW_STATE, applyViewPreset } from "./viewState";

describe("manifestUrlFromSearch", () => {
  it("uses the manifest query parameter when present", () => {
    expect(manifestUrlFromSearch("?manifest=/@fs/tmp/replay_viewer_manifest.json")).toBe(
      "/@fs/tmp/replay_viewer_manifest.json",
    );
  });

  it("does not fall back to a checkout-specific absolute path", () => {
    expect(manifestUrlFromSearch("")).toBeNull();
  });
});

describe("camera preference and motion contract", () => {
  it("persists the last product camera preset and selected follow player", () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    };

    persistCameraPreference({ preset: "follow_player", playerId: 2 }, storage);
    expect(loadCameraPreference(storage)).toEqual({ preset: "follow_player", playerId: 2 });
    expect(cameraTransitionDurationMs(false)).toBeGreaterThan(0);
    expect(cameraTransitionDurationMs(true)).toBe(0);
  });

  it("keeps canonical playback time out of Follow camera React effect inputs", () => {
    const source = readFileSync(resolve(process.cwd(), "src/App.tsx"), "utf8");
    const orbitRig = source.slice(source.indexOf("function OrbitRig("), source.indexOf("const COURT_RENDER_COLORS"));

    expect(orbitRig).not.toContain("currentTime");
    expect(orbitRig).toContain("playbackClock.getTime()");
    expect(orbitRig).toContain("updateFollowCameraPoseBuffer");
  });
});

describe("RecentRunSwitcher", () => {
  it("renders the three current review videos and keeps the selected replay view query", () => {
    const RecentRunSwitcher = (AppModule as any).RecentRunSwitcher;
    const recentRuns = (AppModule as any).RECENT_REPLAY_RUNS;
    expect(typeof RecentRunSwitcher).toBe("function");
    expect(recentRuns).toHaveLength(3);

    const html = renderToStaticMarkup(
      React.createElement(RecentRunSwitcher, {
        currentManifestUrl: "/@fs//Users/arnavchokshi/Desktop/pickleball/runs/visual1_wolverine_20260705T220517Z/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json",
        replayViewMode: "courtmap",
      }),
    );

    expect(html).toContain("Latest video runs");
    expect(html).toContain("Burlington");
    expect(html).toContain("Wolverine");
    expect(html).toContain("Outdoor");
    expect(html).toContain("recent-run-chip active");
    expect(html).toContain("aria-current=\"page\"");
    expect(html).toContain("view=courtmap");
    expect(html).toContain("runs%2Fvisual1_wolverine_20260705T220517Z");
    expect(html).toContain("runs%2Flanes%2Fball_f1_three_clip_runs_20260705%2Fburlington_gold_0300_low_steep_corner");
    expect(html).toContain("runs%2Flanes%2Fball_f1_three_clip_runs_20260705%2Foutdoor_webcam_iynbd_1500_long_high_baseline");
  });
});

describe("loaded replay hierarchy", () => {
  it("keeps intake sections collapsed and after the replay workspace", () => {
    const html = renderToStaticMarkup(
      <ReplayAuxiliarySections replayLoaded currentManifestUrl={null} replayViewMode="world" />,
    );
    const source = readFileSync(resolve(process.cwd(), "src/App.tsx"), "utf8");
    const workspaceIndex = source.indexOf('<section className="review-layout">');
    const layerControlsIndex = source.indexOf('<section className="layer-control-dock"');
    const auxiliaryIndex = source.indexOf("<ReplayAuxiliarySections");

    expect(html).toContain("Upload and process another video");
    expect(html).toContain("Latest video runs");
    expect(html).not.toContain("<details open=\"\"");
    expect(workspaceIndex).toBeGreaterThan(0);
    expect(layerControlsIndex).toBeGreaterThan(workspaceIndex);
    expect(auxiliaryIndex).toBeGreaterThan(layerControlsIndex);
  });

  it("expands intake sections when no manifest is loaded", () => {
    const html = renderToStaticMarkup(
      <ReplayAuxiliarySections replayLoaded={false} currentManifestUrl={null} replayViewMode="world" />,
    );
    expect(html.match(/<details open="">/g)).toHaveLength(2);
  });
});

describe("structured degraded reasons", () => {
  const reasons = degradedReasonsFromNotes([
    'degraded_reasons_json=[{"reasons":["requires ball_track.json"],"stage":"ball_fill","status":"blocked"}]',
  ]);

  it("explains an empty shared timeline using the producer reason verbatim", () => {
    expect(timelineAbsenceText([], reasons)).toBe(
      "No contact/bounce/inflection/shot markers: ball_fill blocked — requires ball_track.json",
    );
    expect(timelineAbsenceText([{ t: 1 } as TimelineMarker], reasons)).toBeNull();
  });

  it("keeps the zero-shot state to one honest producer-backed line", () => {
    expect(shotsEmptyText(reasons)).toBe(
      "No classified shots in this bundle — ball_fill blocked: requires ball_track.json",
    );
  });
});

describe("coachingFactsUrlFromSearch", () => {
  it("uses the explicit coaching query parameter before the manifest pointer", () => {
    const coachingFactsUrlFromSearch = (AppModule as any).coachingFactsUrlFromSearch;
    expect(typeof coachingFactsUrlFromSearch).toBe("function");

    expect(
      coachingFactsUrlFromSearch("?coaching=/@fs/tmp/override/coaching_card_facts.json", {
        coaching_card_facts_url: "/@fs/tmp/manifest/coaching_card_facts.json",
      }),
    ).toBe("/@fs/tmp/override/coaching_card_facts.json");
    expect(
      coachingFactsUrlFromSearch("", {
        coaching_card_facts_url: "/@fs/tmp/manifest/coaching_card_facts.json",
      }),
    ).toBe("/@fs/tmp/manifest/coaching_card_facts.json");
    expect(coachingFactsUrlFromSearch("", null)).toBeNull();
  });
});

describe("bodyMeshOpacityFromBlendWeight", () => {
  it("keeps solid mesh endpoints visible while still scaling by the body_mesh blend weight", () => {
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 0 })).toBeGreaterThan(0.2);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 0.5 })).toBeCloseTo(0.47);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 })).toBeCloseTo(0.68);
  });

  it("multiplies mesh presence fade at window edges instead of popping between visible and hidden", () => {
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 }, 0)).toBe(0);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 }, 0.5)).toBeCloseTo(0.34);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 }, 1)).toBeCloseTo(0.68);
  });
});

describe("bodyMeshMaterialForTrustBadge", () => {
  it("fails missing per-frame trust_badge safe to preview material", () => {
    expect(bodyMeshMaterialForTrustBadge(undefined)).toEqual({
      fillColor: "#ffb454",
      emissiveColor: "#5a3500",
      opacityScale: 0.62,
      label: "estimated",
    });
  });

  it("maps trust-band preview and low-confidence mesh frames to translucent estimated styling", () => {
    expect(bodyMeshMaterialForTrustBadge("verified")).toMatchObject({
      fillColor: "#b4f2bf",
      emissiveColor: "#102d18",
      opacityScale: 1,
      label: "solid",
    });
    expect(bodyMeshMaterialForTrustBadge("preview")).toMatchObject({
      fillColor: "#ffb454",
      emissiveColor: "#5a3500",
      label: "estimated",
    });
    expect(bodyMeshMaterialForTrustBadge("preview").opacityScale).toBeGreaterThan(0);
    expect(bodyMeshMaterialForTrustBadge("preview").opacityScale).toBeLessThan(1);
    expect(bodyMeshMaterialForTrustBadge("low_confidence")).toMatchObject({
      fillColor: "#8a8f98",
      label: "estimated",
    });
    expect(bodyMeshMaterialForTrustBadge("low_confidence").opacityScale).toBeGreaterThan(0);
    expect(bodyMeshMaterialForTrustBadge("low_confidence").opacityScale).toBeLessThan(
      bodyMeshMaterialForTrustBadge("preview").opacityScale,
    );
  });
});

describe("optional capability isolation", () => {
  it("returns null and records the capability without rejecting the core load", async () => {
    const failures: string[] = [];
    await expect(
      loadOptionalArtifact("contacts", async () => { throw new Error("404"); }, (capability) => failures.push(capability)),
    ).resolves.toBeNull();
    expect(failures).toEqual(["contacts"]);
  });
});

describe("viewer truth wiring", () => {
  it("shows three synchronized evidence views and never fabricates a fallback pose", () => {
    const source = readFileSync(resolve(process.cwd(), "src/App.tsx"), "utf8");

    expect(source).toContain('aria-label="Base video"');
    expect(source).toContain('title="Court evidence"');
    expect(source).toContain('title="BODY evidence"');
    expect(source).toContain("new OrbitControls(camera, gl.domElement)");
    expect(source).toContain("Right-drag move");
    expect(source).not.toContain("function skeletonForFrame");
    expect(source).not.toContain("CameraDragPads");
  });

  it("mounts missing-evidence, trust, marker-empty, and review-only paddle-normal surfaces", () => {
    const source = readFileSync(resolve(process.cwd(), "src/App.tsx"), "utf8");
    expect(source).toContain("<NoDetectionPlaceholder");
    expect(source).toContain('aria-label="Player coverage gaps"');
    expect(source).toContain('aria-label="Entity trust badges"');
    expect(source).toContain("Events: no marker evidence at this time");
    expect(source).toContain("showNormals={viewState.layers.paddleNormals}");
    expect(source).toContain('onPause={(event) => { setIsPlaying(false); syncVideoTime(event.currentTarget); }}');
    expect(source).toContain('onRateChange={(event) => { setPlaybackRate(event.currentTarget.playbackRate); syncVideoTime(event.currentTarget); }}');
    expect(source).toContain('className="trust-band-card compact"');
    expect(source).not.toContain("function Ball(");
    expect(source).not.toContain("function BallGhostMarkerRing(");
  });

  it("uses one explicit playback surface and keeps trust and absence inside the pane", () => {
    const source = readFileSync(resolve(process.cwd(), "src/App.tsx"), "utf8");

    expect(source).not.toMatch(/<video[\s\S]{0,180}\bcontrols\b/);
    expect(source).toContain('aria-label={isPlaying ? "Pause replay" : "Play replay"}');
    expect(source).toContain('aria-label={`Set playback speed to ${rate}x`}');
    expect(source).toContain('className="in-pane-entity-trust-strip"');
    expect(source).toContain('className="layer-empty-strip"');
    expect(source).toContain("layer{entityEmptyStates.length === 1");
    expect(source).toContain("Player {playerId}</option>");
    expect(source).toContain("degraded.map((entry)");
    expect(source).toContain("resolved manifest-relative — original absolute paths unreachable (VM-written manifest)");
  });

  it("styles marker provenance separately from authority and reuses the existing render loop for camera easing", () => {
    const source = readFileSync(resolve(process.cwd(), "src/App.tsx"), "utf8");
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");
    expect(source).toContain("marker.provenance");
    expect(source).toContain("camera.position.lerpVectors");
    expect(source.match(/useFrame\(/g)?.length).toBe(3);
    expect(source).toContain("Localizes high-frequency scene renders to the R3F subtree");
    expect(styles).toContain(".timeline-marker.measured");
    expect(styles).toContain(".timeline-marker.model_estimated");
    expect(styles).toContain(".timeline-marker.physics_predicted");
  });

  it("keeps passive player-gap notices below camera controls and out of hit testing", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");
    const playerGapRule = styles.match(/\.player-gap-strip\s*\{([^}]*)\}/)?.[1] ?? "";
    const cameraRule = styles.match(/\.camera-preset-bar\s*\{([^}]*)\}/)?.[1] ?? "";

    expect(playerGapRule).toMatch(/pointer-events:\s*none/);
    expect(playerGapRule).toMatch(/z-index:\s*1/);
    expect(cameraRule).toMatch(/z-index:\s*2/);
  });
});

describe("TimelineStrip", () => {
  it("seeks proportionally through a single dense chapter while preserving exact marker seeks and labels", () => {
    const duration = 10;
    const markers: TimelineMarker[] = Array.from({ length: 56 }, (_, index) => {
      const t = 0.37 + (9.83 - 0.37) * (index / 55);
      return {
        kind: "ball_inflection",
        t,
        t0: t,
        t1: t,
        confidence: 0.8,
        badge: "low_confidence",
        provenance: "model_estimated",
        label: `ball inflection @ ${t.toFixed(2)}s`,
      };
    });
    const chapters = timelineChaptersFromMarkers(markers, duration);
    const seeks: number[] = [];
    let nextEventCalls = 0;
    const strip = TimelineStrip({
      durationSeconds: duration,
      currentTime: 0,
      markers,
      chapters,
      onSeek: (seconds) => seeks.push(seconds),
      onPreviousEvent: () => undefined,
      onNextEvent: () => { nextEventCalls += 1; },
    });
    const track = React.Children.toArray(strip.props.children)[1] as React.ReactElement<any>;

    expect(chapters).toHaveLength(1);
    const html = renderToStaticMarkup(strip);
    expect(html).toContain("Rally 1");
    expect(html).toContain("timeline-marker-glyph");
    expect(html).toContain("ball_inflection · 0.37s · model_estimated · authority low_confidence");
    track.props.onClick({
      clientX: 510,
      currentTarget: { getBoundingClientRect: () => ({ left: 10, width: 1000 }) },
    });
    expect(seeks.at(-1)).toBeCloseTo(duration * 0.5, 6);

    const trackChildren = React.Children.toArray(track.props.children) as React.ReactElement<any>[];
    const chapter = trackChildren.find((child) => String(child.props.className ?? "").startsWith("timeline-chapter "));
    expect(chapter?.props.onClick).toBeUndefined();
    const marker = trackChildren.find((child) => String(child.props.className ?? "").startsWith("timeline-marker "));
    expect(marker).toBeDefined();
    const stopPropagation = vi.fn();
    marker?.props.onClick({ stopPropagation });
    expect(stopPropagation).toHaveBeenCalledOnce();
    expect(seeks.at(-1)).toBeCloseTo(markers[0].t, 9);

    const preventDefault = vi.fn();
    track.props.onKeyDown({ key: "ArrowRight", preventDefault });
    expect(preventDefault).toHaveBeenCalledOnce();
    expect(nextEventCalls).toBe(1);
  });
});

describe("mesh chunk playback prefetch", () => {
  it("selects the next or previous time-ordered window from playback direction", () => {
    const index = { windows: [
      { source_window_index: 2, t0: 2 },
      { source_window_index: 0, t0: 0 },
      { source_window_index: 1, t0: 1 },
    ] } as any;
    expect(adjacentBodyMeshWindow(index, 1, 1)?.source_window_index).toBe(2);
    expect(adjacentBodyMeshWindow(index, 1, -1)?.source_window_index).toBe(0);
  });
});

describe("ViewLayerPanel", () => {
  it("renders all layer buttons with explicit pressed state", () => {
    const ViewLayerPanel = (AppModule as any).ViewLayerPanel;
    expect(typeof ViewLayerPanel).toBe("function");

    const html = renderToStaticMarkup(
      React.createElement(ViewLayerPanel, {
        viewState: DEFAULT_VIEW_STATE,
        playerIds: [1, 2],
        displayFps: {
          enabled: true,
          processing: false,
          readout: "60fps display: computed 30 + interpolated 12 skeletons, 4 mesh",
          onToggle: () => undefined,
        },
        onToggleLayer: () => undefined,
        onBallFocus: () => undefined,
        onPlayerFocus: () => undefined,
        onClearFocus: () => undefined,
        onResetView: () => undefined,
      }),
    );

    expect(html).toContain("Layer controls");
    expect(html).toContain("aria-pressed=\"true\"");
    expect(html).toContain("Ball trail");
    expect(html).toContain("Paddles");
    expect(html).toContain("Skeletons");
    expect(html).toContain("Solid meshes");
    expect(html).toContain("Contact surfaces");
    expect(html).toContain("Target zones");
    expect(html).toContain("Ghost positioning");
    expect(html).toContain("data-layer-key=\"paddles\"");
    expect(html).toContain("data-layer-key=\"playerSkeletons\"");
    expect(html).toContain("data-layer-key=\"floorContactMarkers\"");
    expect(html).toContain("Hand points");
    expect(html).toContain("2x FPS (interpolated)");
    expect(html).toContain("60fps display");
    expect(html).toContain("Debug");
    expect(html).toContain("aria-pressed=\"false\"");
    expect(html).toContain("<details class=\"layer-group debug-layer-group\">");
    expect(html).toContain("Implausible skeletons");
    expect(html).toContain("Point clouds");
  });

  it("names every enabled entity layer that has no renderable data", () => {
    expect(entityLayerEmptyStates(DEFAULT_VIEW_STATE.layers, {
      playerMeshCount: 0,
      playerSkeletonCount: 0,
      ballTrailCount: 0,
      paddleCount: 0,
      contactSurfaceCount: 0,
      targetZoneCount: 0,
      ghostPositionCount: 0,
    })).toEqual(expect.arrayContaining([
      "Player meshes: no data at this time",
      "Player skeletons: no data at this time",
      "Ball trail: no data at this time",
      "Paddles: no data at this time",
    ]));
  });

  it("marks focus presets and player chips as active when selected", () => {
    const ViewLayerPanel = (AppModule as any).ViewLayerPanel;
    const focused = applyViewPreset(DEFAULT_VIEW_STATE, "playerFocus", { playerId: 2 });

    const html = renderToStaticMarkup(
      React.createElement(ViewLayerPanel, {
        viewState: focused,
        playerIds: [1, 2],
        onToggleLayer: () => undefined,
        onBallFocus: () => undefined,
        onPlayerFocus: () => undefined,
        onClearFocus: () => undefined,
        onResetView: () => undefined,
      }),
    );

    expect(html).toContain("Player focus");
    expect(html).toContain("Player 2");
    expect(html).toContain("player-chip active");
    expect(html).toContain("aria-pressed=\"true\"");
  });
});

describe("DisplayFpsControl", () => {
  it("renders compactly inside the layer panel where the owner already looks", () => {
    const DisplayFpsControl = (AppModule as any).DisplayFpsControl;
    expect(typeof DisplayFpsControl).toBe("function");

    const html = renderToStaticMarkup(
      React.createElement(DisplayFpsControl, {
        enabled: true,
        processing: false,
        readout: "60fps display: computed 30 + interpolated 12 skeleton, 4 mesh",
        onToggle: () => undefined,
      }),
    );

    expect(html).toContain("2x FPS (interpolated)");
    expect(html).toContain("aria-pressed=\"true\"");
    expect(html).toContain("layer-fps-control");
    expect(html).toContain("display-fps-badge");
    expect(html).toContain("60fps display: computed 30 + interpolated 12 skeleton, 4 mesh");
  });
});

describe("default world visibility", () => {
  it("starts the 3D world with BODY meshes visible and leaves debug layers opt-in", () => {
    expect(DEFAULT_VIEW_STATE.layers.ballDot).toBe(true);
    expect(DEFAULT_VIEW_STATE.layers.ballTrail).toBe(true);
    expect(DEFAULT_VIEW_STATE.layers.playerSkeletons).toBe(true);
    expect(DEFAULT_VIEW_STATE.layers.playerSolidMeshes).toBe(true);
    expect(DEFAULT_VIEW_STATE.layers.handJointPoints).toBe(false);
    expect(DEFAULT_VIEW_STATE.layers.debugPointClouds).toBe(false);
  });

  it("keeps manifest replay point clouds behind the point-cloud debug layer", () => {
    const replayScene = {
      schema_version: 1 as const,
      world_frame: "court_Z0" as const,
      fps: 30,
      court_glb: "replay_review/court.glb",
      players: [1],
      points: [{ id: 1, t0: 0, t1: 1, glb_url: "replay_review/points/point_001_review.glb", size_mb: 1.2 }],
    };

    expect(shouldRenderReplayScenePointClouds(DEFAULT_VIEW_STATE, replayScene, 0.5)).toBe(false);
    expect(shouldRenderReplayScenePointClouds(applyViewPreset(DEFAULT_VIEW_STATE, "default"), replayScene, 0.5)).toBe(false);
    expect(shouldRenderReplayScenePointClouds({ ...DEFAULT_VIEW_STATE, layers: { ...DEFAULT_VIEW_STATE.layers, debugPointClouds: true } }, replayScene, 0.5)).toBe(true);
  });
});

describe("MeshDebugReadout", () => {
  it("renders an inspectable mesh-chain snapshot behind the debug layer", () => {
    const MeshDebugReadout = (AppModule as any).MeshDebugReadout;
    expect(typeof MeshDebugReadout).toBe("function");

    const html = renderToStaticMarkup(
      React.createElement(MeshDebugReadout, {
        snapshot: {
          current_time: 0.266,
          active_window_id: 0,
          active_window_url: "body_mesh_chunks/window_000.bin.gz",
          load_state: "loaded",
          load_stage: "chunk",
          load_url: "body_mesh_chunks/window_000.bin.gz",
          load_message: null,
          rendered_player_count: 2,
          alignment_floor_guard_count: 0,
          players: [
            {
              world_player_id: 1,
              world_frame_t: 0.26666666666666666,
              mesh_ref_player_id: 1,
              mesh_ref_frame_idx: 8,
              normalized_mesh_player_id: 1,
              mesh_player_present: true,
              mesh_frame_present: true,
              mesh_frame_idx: 8,
            },
          ],
        },
      }),
    );

    expect(html).toContain("data-mesh-debug=");
    expect(html).toContain("window=0");
    expect(html).toContain("load=loaded");
    expect(html).toContain("P1-&gt;M1:frame 8");
  });
});

describe("solid mesh geometry cache", () => {
  const meshPayload = {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh",
    clip: "clip_a",
    model: "sam3dbody_world_joints",
    fps: 30,
    world_frame: "court_Z0",
    faces_ref: "mhr_faces_static",
    mesh_faces: [[0, 1, 2]],
    joint_names: [],
    players: [
      {
        id: 4,
        frames: [
          {
            frame_idx: 42,
            t: 1.4,
            source_window_index: 0,
            blend_weight: 1,
            joints_world: [],
            joint_conf: [],
            mesh_vertices_world: [
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
            ],
            smplx_params: {},
            reasons: ["contact_window"],
          },
        ],
      },
    ],
    summary: {
      mesh_frame_count: 1,
      player_count: 1,
      contact_window_count: 1,
    },
  };

  it("precomputes indexed geometry once per player frame and reuses it across render swaps", () => {
    const bodyMesh = parseBodyMesh(meshPayload) as BodyMesh;
    const cache = createSolidBodyMeshGeometryCache(bodyMesh);
    const frame = bodyMesh.players[0].frames[0];
    const first = geometryForSolidBodyMeshFrame(cache, 4, frame);
    const second = geometryForSolidBodyMeshFrame(cache, 4, frame);

    expect(first).toBe(second);
    expect(cache.geometryCount).toBe(1);
    expect(cache.normalComputeCount).toBe(1);
  });

  it("reuses one BufferAttribute while interpolated mesh vertices advance", () => {
    const bodyMesh = parseBodyMesh(meshPayload) as BodyMesh;
    const cache = createSolidBodyMeshGeometryCache(null);
    const base = bodyMesh.players[0].frames[0];
    const firstFrame = {
      ...base,
      mesh_interpolated: true,
      interpolation: { from_frame_idx: 42, to_frame_idx: 43, alpha: 0.25, max_gap_s: 0.1 },
    };
    const secondFrame = {
      ...firstFrame,
      mesh_vertices_world: base.mesh_vertices_world.map(([x, y, z]) => [x + 0.5, y, z] as [number, number, number]),
      interpolation: { ...firstFrame.interpolation, alpha: 0.75 },
    };
    const geometry = geometryForSolidBodyMeshFrame(cache, 4, firstFrame);
    const position = geometry.getAttribute("position");
    const reused = geometryForSolidBodyMeshFrame(cache, 4, secondFrame);

    expect(reused).toBe(geometry);
    expect(reused.getAttribute("position")).toBe(position);
    expect(position.getX(0)).toBeCloseTo(0.5);
    expect(cache.geometryCount).toBe(1);
  });

  it("does not rewrite cached mesh geometry while playback reuses a held aligned frame", () => {
    const bodyMesh = parseBodyMesh({
      ...meshPayload,
      joint_names: ["left_hip", "right_hip"],
      players: [
        {
          id: 4,
          frames: [
            {
              ...meshPayload.players[0].frames[0],
              t: 1,
              joints_world: [
                [0, 0, 1],
                [1, 0, 1],
              ],
              joint_conf: [1, 1],
            },
          ],
        },
      ],
    }) as BodyMesh;
    const world = makeWorld([
      {
        id: 4,
        side: "near",
        role: "left",
        representation: "mesh",
        frames: Array.from({ length: 100 }, (_, index) => ({
          t: 1 + index / 1000,
          track_world_xy: [index / 1000, 1],
          floor_world_xyz: [index / 1000, 1, 0],
          floor_source: "track_footpoint",
          contact_locked: false,
          floor_penetration_m: 0,
          joints_world: [
            [index / 1000, 0, 1],
            [1 + index / 1000, 0, 1],
          ],
          joint_conf: [1, 1],
          mesh_vertices_world: [],
          joint_count: 2,
          mesh_vertex_count: 0,
          mesh_ref: { artifact: "body_mesh.json", player_id: 4, frame_idx: 42, t: 1 },
        })),
      },
    ]);
    world.joint_names = ["left_hip", "right_hip"];
    const cache = createSolidBodyMeshGeometryCache(null);

    const geometries = Array.from({ length: 100 }, (_, index) => {
      const active = solidBodyMeshFramesForTime(bodyMesh, null, 1 + index / 1000, world);
      expect(active).toHaveLength(1);
      expect(active[0].renderTranslation).toEqual([index / 1000, 0, 0]);
      return geometryForSolidBodyMeshFrame(cache, active[0].meshPlayerId, active[0].frame);
    });

    expect(new Set(geometries).size).toBe(1);
    expect(cache.geometryCount).toBe(1);
    expect(cache.normalComputeCount).toBe(1);
  });

  it("builds geometry for a truncated real Wolverine body_mesh frame and keeps zero-blend opacity visible", () => {
    const realShapedPayload = {
      schema_version: 1,
      artifact_type: "racketsport_body_mesh",
      clip: "wolverine_mixed_0200_mid_steep_corner",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      // Remapped from body_mesh.json player 2 frame 41, source face [3997, 3895, 5229].
      mesh_faces: [[0, 1, 2]],
      joint_names: ["sam3dbody_joint_000", "sam3dbody_joint_001"],
      players: [
        {
          id: 2,
          frames: [
            {
              frame_idx: 41,
              t: 1.3666666666666667,
              source_window_index: 3,
              blend_weight: 0,
              joints_world: [
                [-0.4609871280688127, -2.86639920139313, 1.4727504227208539],
                [-0.44476821007905043, -2.89331748700701, 1.519577850798574],
              ],
              joint_conf: [0.9318317174911499, 0.9318317174911499],
              mesh_vertices_world: [
                [-0.42600738197685617, -2.859783359847331, 1.4446742849885692],
                [-0.42402143790016034, -2.8583956249317484, 1.444022971085586],
                [-0.4242655178571626, -2.858508900821761, 1.4441960016626272],
              ],
              smplx_params: {
                global_orient: [2.114689826965332, -1.3829503059387207, -2.1891682147979736],
                body_pose: [0.021708069369196892, 0.07678874582052231, 0.04448043555021286],
                left_hand_pose: [0.017201418057084084],
                right_hand_pose: [0.058121927082538605],
                betas: [-1.1666346788406372],
                transl_world: [-0.32252942875560225, -3.0421257901883143, 0],
              },
              reasons: ["contact_window", "low_track_confidence"],
            },
          ],
        },
      ],
      summary: {
        mesh_frame_count: 1,
        player_count: 1,
        contact_window_count: 8,
      },
    };
    const bodyMesh = parseBodyMesh(realShapedPayload) as BodyMesh;
    const cache = createSolidBodyMeshGeometryCache(bodyMesh);
    const frame = bodyMesh.players[0].frames[0];

    expect(cache.geometryCount).toBe(1);
    expect(geometryForSolidBodyMeshFrame(cache, 2, frame).attributes.position.count).toBe(3);
    expect(bodyMeshOpacityFromBlendWeight(frame)).toBeGreaterThan(0.2);
  });
});

describe("vertexDebugPointsForFrame", () => {
  it("defaults vertex debug clouds off and only samples the supplied current frame when enabled", () => {
    const frame = {
      t: 1,
      joints_world: [],
      joint_conf: [],
      mesh_vertices_world: [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
      ],
      joint_count: 0,
      mesh_vertex_count: 3,
    } as VirtualWorldPlayer["frames"][number];
    const previousFrame = {
      ...frame,
      t: 0,
      mesh_vertices_world: [[99, 99, 99]],
      mesh_vertex_count: 1,
    } as VirtualWorldPlayer["frames"][number];

    expect(vertexDebugPointsForFrame(frame, false, 10)).toEqual([]);
    expect(vertexDebugPointsForFrame(frame, true, 2)).toEqual([
      [0, 0, 0],
      [2, 0, 0],
    ]);
    expect(vertexDebugPointsForFrame(frame, true, 10)).not.toContainEqual(previousFrame.mesh_vertices_world[0]);
  });
});

describe("bodyJointSkeletonForFrame", () => {
  it("uses the real staged COCO-WholeBody 133-joint names for body bones only", () => {
    const stagedSkeleton = JSON.parse(
      readFileSync(
        resolve(process.cwd(), "../../runs/manager_rebuild_wolverine_20260702T23Z/skeleton3d.json"),
        "utf8",
      ),
    );
    const jointNames = stagedSkeleton.joint_names as string[];
    const stagedFrame = stagedSkeleton.players[0].frames[0] as VirtualWorldPlayer["frames"][number];

    const skeleton = bodyJointSkeletonForFrame(stagedFrame, jointNames);

    expect(skeleton).not.toBeNull();
    if (!skeleton) throw new Error("expected staged COCO-WholeBody skeleton");
    expect(skeleton.boneNames).toEqual(cocoWholeBodyCoreBoneNames);
    expect(skeleton.bones).toHaveLength(cocoWholeBodyCoreBoneNames.length * 2);
    expect(skeleton.boneNames.some(([a, b]) => `${a} ${b}`.includes("face-") || `${a} ${b}`.includes("hand_"))).toBe(false);
    expect(skeleton.boneNames).not.toContainEqual(["left_hip", "right_shoulder"]);
    expect(skeleton.boneNames).not.toContainEqual(["right_hip", "left_shoulder"]);
  });

  it("hides implausible skeleton frames unless debug explicitly includes them", () => {
    const frame = {
      t: 1,
      skeleton_implausible: true,
      joints_world: [
        [0, 0, 0.9],
        [0, 0, 1.45],
      ],
      joint_conf: [0.9, 0.9],
      mesh_vertices_world: [],
      joint_count: 2,
      mesh_vertex_count: 0,
    } as VirtualWorldPlayer["frames"][number];

    expect(bodyJointSkeletonForFrame(frame, ["left_hip", "left_shoulder"])).toBeNull();
    expect(bodyJointSkeletonForFrame(frame, ["left_hip", "left_shoulder"], { includeImplausible: true })).not.toBeNull();
  });

  it("turns BODY joints_world into canonical bone line pairs instead of dots only", () => {
    const frame = {
      t: 1,
      joints_world: [
        [0, 0, 1.7],
        [-0.22, 0, 1.38],
        [0.22, 0, 1.38],
        [-0.48, 0, 1.12],
        [0.48, 0, 1.12],
      ],
      joint_conf: [0.9, 0.9, 0.9, 0.9, 0.9],
      mesh_vertices_world: [],
      joint_count: 5,
      mesh_vertex_count: 0,
    } as VirtualWorldPlayer["frames"][number];

    const skeleton = bodyJointSkeletonForFrame(frame, ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]);

    expect(skeleton).not.toBeNull();
    if (!skeleton) throw new Error("expected BODY joint skeleton");
    expect(skeleton.joints).toHaveLength(5);
    expect(skeleton.bones.length).toBeGreaterThan(0);
    expect(skeleton.bones.length % 2).toBe(0);
  });

  it("keeps whole-body hand joints as optional dots and never body bones", () => {
    const jointNames = [
      "nose",
      "left_shoulder",
      "right_shoulder",
      "left_elbow",
      "right_elbow",
      "left_wrist",
      "right_wrist",
      "left_hand_root",
      "left_thumb1",
      "right_hand_root",
    ];
    const frame = {
      t: 1,
      joints_world: jointNames.map((_, index) => [index * 0.01, 0, 1] as [number, number, number]),
      joint_conf: jointNames.map(() => 0.9),
      mesh_vertices_world: [],
      joint_count: jointNames.length,
      mesh_vertex_count: 0,
    } as VirtualWorldPlayer["frames"][number];

    const skeleton = bodyJointSkeletonForFrame(frame, jointNames);
    const handPoints = handJointPointsForFrame(frame, jointNames);

    expect(skeleton?.boneNames.some(([a, b]) => `${a} ${b}`.includes("hand") || `${a} ${b}`.includes("thumb"))).toBe(false);
    expect(handPoints).toHaveLength(3);
  });
});

describe("paddleRenderGeometryForFrame", () => {
  it("builds an estimated rounded face-plane paddle from a real wrist-proxy frame pose", () => {
    const stagedWorld = JSON.parse(
      readFileSync(
        resolve(process.cwd(), "../../runs/manager_rebuild_wolverine_20260702T23Z/virtual_world.json"),
        "utf8",
      ),
    );
    const paddle = stagedWorld.paddles[0];
    const frame = paddle.frames[0];

    const geometry = paddleRenderGeometryForFrame(frame, paddle.paddle_dims_in);

    expect(frame.source).toBe("wrist_proxy:court_Z0");
    expect(geometry.estimated).toBe(true);
    expect(geometry.vertices.length).toBeGreaterThan(4);
    expect(geometry.faces.length).toBeGreaterThan(2);
    expect(geometry.vertices[1]).not.toEqual(frame.pose_se3.t);
  });

  it("replaces an estimated box mesh with the rounded beveled dimensions proxy", () => {
    const frame: VirtualWorldPaddleFrame = {
      t: 0,
      pose_se3: {
        R: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        t: [0, 0, 0],
      },
      mesh_vertices_world: [
        [-1, 1, 0],
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-0.2, -1, 0],
        [0.2, -1, 0],
        [0.2, -1.8, 0],
        [-0.2, -1.8, 0],
      ],
      mesh_faces: [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
      ],
      conf: 0.5,
      world_frame: "court_Z0",
      translation_unit: "m",
      source: "wrist_proxy:court_Z0",
      ambiguous: false,
      render_only: true,
      not_for_detection_metrics: true,
      reprojection_error_px: null,
      confidence_provenance: null,
      trust_band: null,
    };

    const geometry = paddleRenderGeometryForFrame(frame, { length: 15.5, width: 7.5 });

    expect(geometry.vertices).not.toEqual(frame.mesh_vertices_world);
    expect(geometry.vertices.length).toBeGreaterThan(frame.mesh_vertices_world.length);
    expect(geometry.faces.length).toBeGreaterThan(frame.mesh_faces.length);
    expect(geometry.estimated).toBe(true);
    expect(geometry.edgeSegments.length).toBeGreaterThan(16);
    expect(geometry.material.fillOpacity).toBeLessThan(0.7);
    expect(geometry.material.edgeOpacity).toBeGreaterThanOrEqual(0.9);
    expect(geometry.material.edgeRadiusM).toBeLessThan(0.012);
  });

  it("exposes a face-normal review segment from the paddle pose rotation", () => {
    const frame: VirtualWorldPaddleFrame = {
      t: 0,
      pose_se3: {
        R: [
          [1, 0, 0],
          [0, 1, -0.707106781],
          [0, 0, 0.707106781],
        ],
        t: [0.2, 0.4, 1.1],
      },
      mesh_vertices_world: [],
      mesh_faces: [],
      conf: 0.5,
      world_frame: "court_Z0",
      translation_unit: "m",
      source: "wrist_proxy:court_Z0",
      ambiguous: false,
      render_only: true,
      not_for_detection_metrics: true,
      reprojection_error_px: null,
      confidence_provenance: null,
      trust_band: null,
    };

    const geometry = paddleRenderGeometryForFrame(frame, { length: 15.5, width: 7.5 });

    expect(geometry.normalSegment[0]).toEqual([0.2, 0.4, 1.1]);
    expect(geometry.normalSegment[1]).toEqual([
      expect.closeTo(0.2),
      expect.closeTo(0.4 - 0.707106781 * 0.52),
      expect.closeTo(1.1 + 0.707106781 * 0.52),
    ]);
    expect(geometry.material.normalColor).toBe("#23c9ff");
    expect(geometry.material.normalRadiusM).toBeGreaterThanOrEqual(0.014);
    expect(geometry.material.normalTipRadiusM).toBeGreaterThanOrEqual(0.05);
    expect(geometry.normalTip).toEqual(geometry.normalSegment[1]);
    expect(geometry.material.normalOverlay).toBe(true);
  });
});

function makeWorld(players: VirtualWorldPlayer[]): VirtualWorld {
  return {
    schema_version: 1,
    artifact_type: "racketsport_virtual_world",
    world_frame: "court_Z0",
    fps: 30,
    court: {
      sport: "pickleball",
      coordinate_frame: "origin_net_center_x_width_y_length_z_up_m",
      length_m: 13.41,
      width_m: 6.1,
      line_segments: {
        near_baseline: [
          [-3.05, 0, 0],
          [3.05, 0, 0],
        ],
        far_baseline: [
          [-3.05, 13.41, 0],
          [3.05, 13.41, 0],
        ],
      },
      net: {
        endpoints: [
          [-3.05, 6.705, 0.91],
          [3.05, 6.705, 0.91],
        ],
        center_height_m: 0.86,
        post_height_m: 0.91,
      },
      trust_band: null,
    },
    players,
    ball: { source: null, frames: [] },
    paddles: [],
    summary: {
      player_count: players.length,
      mesh_player_count: 0,
      mesh_player_frame_count: 0,
      joint_player_frame_count: 0,
      track_only_player_frame_count: 0,
      floor_placed_player_frame_count: 0,
      floor_contact_player_frame_count: 0,
      max_floor_penetration_m: 0,
      max_abs_floor_offset_m: 0,
      physics_modes: [],
      ball_frame_count: 0,
      approx_ball_frame_count: 0,
      paddle_player_count: 0,
      paddle_frame_count: 0,
      ambiguous_paddle_frame_count: 0,
      warnings: [],
    },
  };
}

describe("updateFpsSample", () => {
  it("accumulates frames without reporting inside a ~500ms window", () => {
    const first = updateFpsSample({ framesSinceReport: 0, windowStartMs: 0, fps: 0 }, 100);
    expect(first).toEqual({ framesSinceReport: 1, windowStartMs: 0, fps: 0 });
    const second = updateFpsSample(first, 200);
    expect(second).toEqual({ framesSinceReport: 2, windowStartMs: 0, fps: 0 });
  });

  it("reports an fps reading and resets the window once >=500ms elapse", () => {
    const midway = { framesSinceReport: 29, windowStartMs: 0, fps: 0 };
    const next = updateFpsSample(midway, 500);
    expect(next.framesSinceReport).toBe(0);
    expect(next.windowStartMs).toBe(500);
    expect(next.fps).toBeCloseTo(60, 0); // 30 frames / 500ms = 60fps
  });
});

describe("courtBounds and cameraPresetPose", () => {
  const world = makeWorld([]);

  it("computes court bounds from line segments and the template width/length", () => {
    const bounds = courtBounds(world);
    expect(bounds.width).toBeCloseTo(6.1);
    expect(bounds.length).toBeCloseTo(13.41);
    expect(bounds.centerX).toBeCloseTo(0);
  });

  it("does not add an extra apron when court coordinates are centered on the net", () => {
    const centered = makeWorld([]);
    centered.court.line_segments = {
      near_baseline: [
        [-3.05, -6.705, 0],
        [3.05, -6.705, 0],
      ],
      far_baseline: [
        [-3.05, 6.705, 0],
        [3.05, 6.705, 0],
      ],
      left_sideline: [
        [-3.05, -6.705, 0],
        [-3.05, 6.705, 0],
      ],
      right_sideline: [
        [3.05, -6.705, 0],
        [3.05, 6.705, 0],
      ],
      net: [
        [-3.35, 0, 0],
        [3.35, 0, 0],
      ],
    };

    const bounds = courtBounds(centered);

    expect(bounds.length).toBeCloseTo(13.41);
    expect(bounds.width).toBeCloseTo(6.1);
    expect(bounds.centerY).toBeCloseTo(0);
  });

  it("produces three distinct camera poses for broadcast/behind_baseline/top_down", () => {
    const broadcast = cameraPresetPose(world, "broadcast");
    const behindBaseline = cameraPresetPose(world, "behind_baseline");
    const topDown = cameraPresetPose(world, "top_down");
    expect(broadcast.position).not.toEqual(behindBaseline.position);
    expect(broadcast.position).not.toEqual(topDown.position);
    // top_down looks straight down at court level.
    expect(topDown.target[2]).toBe(0);
    expect(topDown.position[2]).toBeGreaterThan(broadcast.position[2]);
  });

  it("supports court, selectable follow-player, and free-orbit product presets", () => {
    const player: VirtualWorldPlayer = {
      id: 7,
      representation: "track_only",
      frames: [{
        t: 0,
        track_world_xy: [1.2, 4.4],
        floor_world_xyz: [1.2, 4.4, 0],
        joints_world: [],
        joint_conf: [],
        mesh_vertices_world: [],
        joint_count: 0,
        mesh_vertex_count: 0,
      }],
    };
    const followed = cameraPresetPose(makeWorld([player]), "follow_player", [], 7, 0);
    const court = cameraPresetPose(world, "court");
    const orbit = cameraPresetPose(world, "free_orbit");

    expect(followed.target[0]).toBeCloseTo(1.2);
    expect(followed.target[1]).toBeCloseTo(4.4);
    expect(court).not.toEqual(followed);
    expect(orbit.position).not.toEqual(followed.position);
  });

  it("reuses the Follow pose buffer and skips steady-state work within one source frame", () => {
    const player: VirtualWorldPlayer = {
      id: 7,
      representation: "track_only",
      frames: [{
        t: 0,
        track_world_xy: [1.2, 4.4],
        floor_world_xyz: [1.2, 4.4, 0],
        joints_world: [],
        joint_conf: [],
        mesh_vertices_world: [],
        joint_count: 0,
        mesh_vertex_count: 0,
      }],
    };
    const followWorld = makeWorld([player]);
    const buffer = createFollowCameraPoseBuffer();
    const position = buffer.position;
    const target = buffer.target;

    expect(updateFollowCameraPoseBuffer(buffer, followWorld, 7, 0)).toBe(true);
    expect(updateFollowCameraPoseBuffer(buffer, followWorld, 7, 0.01)).toBe(false);
    expect(buffer.position).toBe(position);
    expect(buffer.target).toBe(target);
    expect(buffer.target[0]).toBeCloseTo(1.2);
    expect(buffer.target[1]).toBeCloseTo(4.4);
  });

  it("updates Follow framing when switching players whose active frames have the same timestamp", () => {
    const player = (id: number, x: number): VirtualWorldPlayer => ({
      id,
      representation: "track_only",
      frames: [{
        t: 0,
        track_world_xy: [x, 4.4],
        floor_world_xyz: [x, 4.4, 0],
        joints_world: [],
        joint_conf: [],
        mesh_vertices_world: [],
        joint_count: 0,
        mesh_vertex_count: 0,
      }],
    });
    const followWorld = makeWorld([player(7, 1.2), player(8, -1.4)]);
    const buffer = createFollowCameraPoseBuffer();

    expect(updateFollowCameraPoseBuffer(buffer, followWorld, 7, 0)).toBe(true);
    expect(updateFollowCameraPoseBuffer(buffer, followWorld, 8, 0)).toBe(true);
    expect(buffer.target[0]).toBeCloseTo(-1.4);
  });

  it("fits court framing to active entities outside the playable rectangle", () => {
    const player: VirtualWorldPlayer = {
      id: 8,
      representation: "track_only",
      frames: [{
        t: 0,
        track_world_xy: [0, -9],
        floor_world_xyz: [0, -9, 0],
        joints_world: [],
        joint_conf: [],
        mesh_vertices_world: [],
        joint_count: 0,
        mesh_vertex_count: 0,
      }],
    };
    const entityFit = fittedCourtCameraPose(makeWorld([player]), 0);
    const courtOnly = fittedCourtCameraPose(world, 0);

    expect(entityFit.target[1]).toBeLessThan(courtOnly.target[1]);
    expect(entityFit.position[2]).toBeGreaterThan(courtOnly.position[2]);
  });

  it("frames the selected player's active paddle with a close review camera", () => {
    const playerOne = makeActivePaddle(1, [0, 1, 1]);
    const playerTwo = makeActivePaddle(2, [1.2, 4.5, 1.1]);

    const pose = cameraPresetPose(world, "paddle_review", [playerOne, playerTwo], 2);

    expect(pose.target[0]).toBeCloseTo(1.2);
    expect(pose.target[1]).toBeCloseTo(4.5);
    expect(pose.target[2]).toBeCloseTo(1.1);
    expect(pose.position[1]).toBeLessThan(pose.target[1]);
    expect(pose.position[2]).toBeGreaterThan(pose.target[2]);
    expect(Math.hypot(pose.position[0] - pose.target[0], pose.position[1] - pose.target[1], pose.position[2] - pose.target[2])).toBeLessThan(2);
  });

  it("falls back to the top-down court camera when no paddle is active", () => {
    expect(cameraPresetPose(world, "paddle_review")).toEqual(cameraPresetPose(world, "top_down"));
  });
});

function makeActivePaddle(playerId: number, center: [number, number, number]): ActivePaddleFrame {
  const [x, y, z] = center;
  const frame: VirtualWorldPaddleFrame = {
    t: 0,
    pose_se3: {
      R: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      t: center,
    },
    mesh_vertices_world: [
      [x - 0.1, y - 0.2, z],
      [x + 0.1, y - 0.2, z],
      [x + 0.1, y + 0.2, z],
      [x - 0.1, y + 0.2, z],
    ],
    mesh_faces: [[0, 1, 2]],
    conf: 0.5,
    world_frame: "court_Z0",
    translation_unit: "m",
    source: "wrist_proxy:court_Z0",
    ambiguous: false,
    render_only: true,
    not_for_detection_metrics: true,
    reprojection_error_px: null,
    confidence_provenance: null,
    trust_band: null,
  };
  return {
    playerId,
    paddle: {
      player_id: playerId,
      paddle_dims_in: { length: 15.5, width: 7.5 },
      frames: [frame],
      coverage_fraction: 1,
      min_t: 0,
      max_t: 0,
      trust_band: null,
    },
    frame,
    estimated: true,
    staleAgeSeconds: 0,
    opacity: 1,
  };
}

describe("review start time defaults", () => {
  it("opens manifest review links on the first renderable artifact time", () => {
    const world = makeWorld([
      {
        id: 1,
        representation: "joints",
        frames: [{ t: 2.4, floor_world_xyz: [0, 0, 0], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 }],
      },
      {
        id: 2,
        representation: "joints",
        frames: [{ t: 3.1, floor_world_xyz: [1, 1, 0], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 }],
      },
    ]);
    expect(defaultReviewStartTime(world)).toBeCloseTo(2.4);
  });

  it("skips placeholder player frames and opens where the most players are visible", () => {
    const placeholderFrame = {
      t: 2.0,
      joints_world: [],
      joint_conf: [],
      mesh_vertices_world: [],
      joint_count: 0,
      mesh_vertex_count: 0,
    };
    const visibleFrameA = {
      t: 4.0,
      floor_world_xyz: [0, 0, 0] as [number, number, number],
      joints_world: [],
      joint_conf: [],
      mesh_vertices_world: [],
      joint_count: 0,
      mesh_vertex_count: 0,
    };
    const visibleFrameB = {
      ...visibleFrameA,
      floor_world_xyz: [1, 1, 0] as [number, number, number],
    };
    const world = makeWorld([
      { id: 1, representation: "joints", frames: [placeholderFrame, visibleFrameA] },
      { id: 2, representation: "joints", frames: [placeholderFrame, visibleFrameB] },
    ]);

    expect(defaultReviewStartTime(world)).toBeCloseTo(4.0);
  });

  it("distinguishes an explicit zero start time from an omitted start time", () => {
    expect(hasExplicitReviewStartTime("?manifest=/tmp/replay.json")).toBe(false);
    expect(hasExplicitReviewStartTime("?manifest=/tmp/replay.json&t=0")).toBe(true);
    expect(hasExplicitReviewStartTime("?time=1.25")).toBe(true);
  });
});

describe("coaching card facts UI", () => {
  const coachingFacts = {
    artifact_type: "coaching_card_facts",
    rally_scope: "rally_spans",
    priority_rule: ["contact_count_when_present"],
    facts: [
      {
        coverage_fraction: 0.9666666666666667,
        frames_total: 300,
        frames_used: 290,
        metric: "contact_count",
        player_id: "1",
        rally_id: "rally_000",
        rally_scope: "rally_spans",
        trust: "ok",
        unit: "count",
        value: 3,
      },
      {
        coverage_fraction: 0.42,
        frames_total: 100,
        frames_used: 42,
        metric: "kitchen_time_s",
        player_id: "1",
        rally_id: "rally_001",
        rally_scope: "rally_spans",
        trust: "estimated",
        unit: "s",
        value: 1.25,
      },
    ],
  };

  const chapters: TimelineChapter[] = [
    { index: 1, t0: 0, t1: 1, label: "Rally 1", badge: "preview" },
    { index: 2, t0: 3, t1: 4, label: "Rally 2", badge: "preview" },
  ];

  it("renders missing player facts as not measured rather than zero", () => {
    const buildCard = (AppModule as any).coachingCardForTimeline;
    const CoachingCardPanel = (AppModule as any).CoachingCardPanel;
    expect(typeof buildCard).toBe("function");
    expect(typeof CoachingCardPanel).toBe("function");

    const card = buildCard(coachingFacts, chapters, 0.5, ["1", "2"]);
    const html = renderToStaticMarkup(React.createElement(CoachingCardPanel, { card }));

    expect(html).toContain("Per-rally coaching card");
    expect(html).toContain("contact count");
    expect(html).toContain("3 count");
    expect(html).toContain("ok");
    expect(html).toContain("96.7% coverage");
    expect(html).toContain("not measured");
    expect(html).not.toContain(">0<");
    expect(html).not.toContain("0 count");
  });

  it("switches the rendered facts when the active timeline chapter changes", () => {
    const buildCard = (AppModule as any).coachingCardForTimeline;
    expect(typeof buildCard).toBe("function");

    const first = buildCard(coachingFacts, chapters, 0.5, ["1"]);
    const second = buildCard(coachingFacts, chapters, 3.5, ["1"]);

    expect(first.rallyId).toBe("rally_000");
    expect(first.rows[0].fact.metric).toBe("contact_count");
    expect(first.rows[0].fact.value).toBe(3);
    expect(second.rallyId).toBe("rally_001");
    expect(second.rows[0].fact.metric).toBe("kitchen_time_s");
    expect(second.rows[0].fact.value).toBe(1.25);
  });

  it("joins coaching facts through the chapter rally id instead of the chapter index", () => {
    const buildCard = (AppModule as any).coachingCardForTimeline;
    expect(typeof buildCard).toBe("function");

    const chapters: TimelineChapter[] = [
      { index: 1, rallyId: "rally_042", t0: 0, t1: 2, label: "Rally 42", badge: "preview" },
    ];
    const facts = {
      ...coachingFacts,
      facts: [
        {
          coverage_fraction: 0.5,
          frames_total: 100,
          frames_used: 50,
          metric: "contact_count",
          player_id: "1",
          rally_id: "rally_000",
          rally_scope: "rally_spans",
          trust: "ok",
          unit: "count",
          value: 99,
        },
        {
          coverage_fraction: 0.75,
          frames_total: 100,
          frames_used: 75,
          metric: "distance_covered_m",
          player_id: "1",
          rally_id: "rally_042",
          rally_scope: "rally_spans",
          trust: "estimated",
          unit: "m",
          value: 12.5,
        },
      ],
    };

    const card = buildCard(facts, chapters, 1.0, ["1"]);

    expect(card.rallyId).toBe("rally_042");
    expect(card.rows[0].fact.metric).toBe("distance_covered_m");
    expect(card.rows[0].fact.value).toBe(12.5);
  });
});

describe("contactReadoutText", () => {
  it("uses active body mesh frames as contact evidence when contact_windows is absent", () => {
    expect(
      contactReadoutText(new Set(), [
        {
          playerId: 1,
          meshPlayerId: 1,
          presenceOpacity: 1,
          renderTranslation: [0, 0, 0],
          frame: {
            frame_idx: 150,
            t: 2.5,
            source_window_index: 0,
            blend_weight: 1,
            joints_world: [],
            joint_conf: [],
            mesh_vertices_world: [],
            mesh_faces: [],
            smplx_params: {},
            reasons: ["contact_window"],
            mesh_interpolated: false,
            interpolation: null,
          },
        },
      ]),
    ).toBe("3D contact: p1");
  });
});
