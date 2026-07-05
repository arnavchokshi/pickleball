export type Vec3 = [number, number, number];

export type BallBand = "anchored_measured" | "arc_interpolated" | "arc_extrapolated" | "arc_weak" | "hidden" | "unknown";

export type BallHudState = "measured" | "predicted" | "not_visible" | "solver_off";
export type BallLinePattern = "solid" | "dashed" | "gap";

/**
 * Authoritative trusted-status allowlist for the ball-arc-solver artifact
 * (ball_track_arc_solved.json). The producer only ever writes one of three
 * top-level `status` values:
 *   - "ran"                      -> threed/racketsport/ball_arc_solver.py:1329 (success)
 *   - "experimental_off"         -> threed/racketsport/ball_arc_solver.py:1333, 1339
 *                                   (self-kill: leave-one-out median regression, or
 *                                   physical-sanity-violation fraction over threshold)
 *   - "degenerate_zero_segments" -> threed/racketsport/ball_arc_solver.py:1348 (zero
 *                                   accepted segments with a rally anchor present)
 * `status` is unconditionally present on every artifact the solver writes
 * (threed/racketsport/ball_arc_solver.py:1365).
 *
 * Downstream Python consumers already treat anything other than "ran" as
 * untrusted/killed:
 *   - scripts/racketsport/run_ball_chain.py:438 `_solver_killed`:
 *     `str(artifact.get("status") or "") != "ran"` => killed.
 *   - scripts/racketsport/solve_ball_arcs.py:363-372 `build_product_ball_track_view`:
 *     `solver_status = str(arc_solved.get("status") or "ran")`; if not "ran", falls
 *     back to the fused-only product view (no measured bands survive).
 * This viewer parser gates the same way: only "ran" is trusted for
 * measured/anchored styling anywhere downstream. A missing `status` key (only
 * ever seen in synthetic/legacy fixtures, never in a real producer artifact)
 * defaults to "ran", mirroring solve_ball_arcs.py's precedent for the
 * closest analogous "build a render/product view from this artifact" call site.
 */
export const TRUSTED_BALL_ARC_SOLVER_STATUSES: ReadonlySet<string> = new Set(["ran"]);

export type BallBandStyle = {
  band: BallBand;
  hudState: BallHudState;
  label: string;
  color: string;
  emissiveColor: string;
  linePattern: BallLinePattern;
  opacity: number;
  lineWidth: number;
  ballRadius: number;
  pulsingBall: boolean;
  rendersTrail: boolean;
  rendersBall: boolean;
  lowConfidence: boolean;
};

export type BallTrailSample = {
  t: number;
  frame?: number | null;
  band: BallBand;
  conf: number;
  visible: boolean;
  world_xyz: Vec3 | null;
  renderOnly?: boolean;
  segmentId?: number | string | null;
  sigmaM?: number | null;
  source?: string | null;
};

export type BallArcSegment = {
  segmentId: number | string;
  t0: number;
  t1: number;
  netClearanceM: number | null;
  netClearanceOk: boolean | null;
};

export type BallTrailArtifact = {
  samples: BallTrailSample[];
  segments: BallArcSegment[];
  status: string;
  trusted: boolean;
  killReasons: string[];
};

export type BallTrailSegment = {
  from: BallTrailSample & { world_xyz: Vec3 };
  to: BallTrailSample & { world_xyz: Vec3 };
  style: BallBandStyle;
  ageOpacityScale: number;
};

export type BallTrailBuild = {
  segments: BallTrailSegment[];
  current: BallHudReadout;
  hiddenGapCount: number;
};

export type BallHudReadout = {
  state: BallHudState;
  label: string;
  lowConfidence: boolean;
  sample: (BallTrailSample & { world_xyz: Vec3 }) | null;
  style: BallBandStyle;
  reason?: string | null;
};

export type BounceCandidate = {
  t: number;
  frame: number | null;
  position: Vec3;
  confidence: number;
  source: string;
  humanReviewed?: boolean;
};

export type ContactImpactEvent = {
  type: "contact" | "bounce" | "net_cross";
  t: number;
  frame: number | null;
  player_id: number | null;
  confidence: number;
  window: { t0: number; t1: number; importance: number };
};

export type ContactWindowsForImpacts = {
  events: ContactImpactEvent[];
};

export type NetPlane = {
  point: Vec3;
  normal: Vec3;
  topHeightM: number;
};

export type ImpactMarkerKind = "floor_bounce" | "landing_spot" | "paddle_contact" | "net_hit";

export type ImpactMarker = {
  kind: ImpactMarkerKind;
  t: number;
  frame: number | null;
  position: Vec3;
  confidence: number;
  label: string;
  playerId?: number | null;
  source?: string;
  positionSource?: "bounce_candidate" | "nearest_ball_sample" | "net_clearance_flag" | "derived_net_crossing";
  derived?: boolean;
  derivation?: "artifact_net_clearance" | "derived_crossing_below_net_top";
  persistent?: boolean;
  fadeRank?: number;
};

export type BuildBallTrailOptions = {
  windowSeconds?: number;
  confidenceThreshold?: number;
  maxGapSeconds?: number;
  currentSampleToleranceSeconds?: number;
};

const DEFAULT_CONFIDENCE_THRESHOLD = 0.5;
const DEFAULT_WINDOW_SECONDS = 1.75;
const DEFAULT_CURRENT_SAMPLE_TOLERANCE_SECONDS = 0.12;

export function styleForBand(
  rawBand: BallBand | string | null | undefined,
  conf = 1,
  options: { confidenceThreshold?: number } = {},
): BallBandStyle {
  const band = normalizeBand(rawBand);
  const lowConfidenceByScore = conf < (options.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD);
  if (band === "hidden") {
    return {
      band,
      hudState: "not_visible",
      label: "not visible",
      color: "#7a7f87",
      emissiveColor: "#000000",
      linePattern: "gap",
      opacity: 0,
      lineWidth: 0,
      ballRadius: 0,
      pulsingBall: false,
      rendersTrail: false,
      rendersBall: false,
      lowConfidence: false,
    };
  }
  if (band === "anchored_measured") {
    return {
      band,
      hudState: "measured",
      label: "measured",
      color: "#e8ff34",
      emissiveColor: "#526000",
      linePattern: "solid",
      opacity: lowConfidenceByScore ? 0.68 : 0.96,
      lineWidth: 1.45,
      ballRadius: 0.064,
      pulsingBall: false,
      rendersTrail: true,
      rendersBall: true,
      lowConfidence: lowConfidenceByScore,
    };
  }
  if (band === "arc_weak") {
    return {
      band,
      hudState: "predicted",
      label: "predicted weak",
      color: "#ffb454",
      emissiveColor: "#4a2500",
      linePattern: "dashed",
      opacity: 0.3,
      lineWidth: 0.68,
      ballRadius: 0.034,
      pulsingBall: true,
      rendersTrail: true,
      rendersBall: true,
      lowConfidence: true,
    };
  }
  if (band === "arc_extrapolated") {
    return {
      band,
      hudState: "predicted",
      label: "predicted extrapolated",
      color: "#78b7ff",
      emissiveColor: "#0e2d52",
      linePattern: "dashed",
      opacity: lowConfidenceByScore ? 0.28 : 0.36,
      lineWidth: 0.78,
      ballRadius: 0.038,
      pulsingBall: true,
      rendersTrail: true,
      rendersBall: true,
      lowConfidence: lowConfidenceByScore,
    };
  }
  if (band === "arc_interpolated") {
    return {
      band,
      hudState: "predicted",
      label: "predicted interpolated",
      color: "#63d9ff",
      emissiveColor: "#10384a",
      linePattern: "dashed",
      opacity: lowConfidenceByScore ? 0.34 : 0.48,
      lineWidth: 0.92,
      ballRadius: 0.043,
      pulsingBall: true,
      rendersTrail: true,
      rendersBall: true,
      lowConfidence: lowConfidenceByScore,
    };
  }
  return {
    band: "unknown",
    hudState: "predicted",
    label: "predicted unknown",
    color: "#adb3bb",
    emissiveColor: "#20242a",
    linePattern: "dashed",
    opacity: lowConfidenceByScore ? 0.28 : 0.4,
    lineWidth: 0.72,
    ballRadius: 0.038,
    pulsingBall: true,
    rendersTrail: true,
    rendersBall: true,
    lowConfidence: lowConfidenceByScore,
  };
}

export function buildBallTrail(
  samples: BallTrailSample[],
  currentTime: number,
  options: BuildBallTrailOptions = {},
): BallTrailBuild {
  const windowSeconds = options.windowSeconds ?? DEFAULT_WINDOW_SECONDS;
  const confidenceThreshold = options.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD;
  const maxGapSeconds = options.maxGapSeconds ?? Math.max(0.18, windowSeconds);
  const t0 = currentTime - Math.max(0, windowSeconds);
  const windowed = samples
    .filter((sample) => Number.isFinite(sample.t) && t0 <= sample.t && sample.t <= currentTime)
    .slice()
    .sort((left, right) => left.t - right.t);
  const segments: BallTrailSegment[] = [];
  let previous: (BallTrailSample & { world_xyz: Vec3 }) | null = null;
  let hiddenGapCount = 0;

  for (const sample of windowed) {
    const renderable = renderableSample(sample);
    if (!renderable) {
      if (previous) hiddenGapCount += 1;
      previous = null;
      continue;
    }
    if (previous && renderable.t - previous.t <= maxGapSeconds) {
      const band = leastCertainBand(previous.band, renderable.band);
      const conf = Math.min(previous.conf, renderable.conf);
      const baseStyle = styleForBand(band, conf, { confidenceThreshold });
      const midpointT = (previous.t + renderable.t) / 2;
      const ageOpacityScale = clamp((midpointT - t0) / Math.max(0.001, windowSeconds), 0.18, 1);
      segments.push({
        from: previous,
        to: renderable,
        style: {
          ...baseStyle,
          opacity: baseStyle.opacity * ageOpacityScale,
          lowConfidence: baseStyle.lowConfidence || previous.conf < confidenceThreshold || renderable.conf < confidenceThreshold,
        },
        ageOpacityScale,
      });
    } else if (previous) {
      hiddenGapCount += 1;
    }
    previous = renderable;
  }

  return {
    segments,
    hiddenGapCount,
    current: ballHudStateForTime(samples, currentTime, options),
  };
}

export function ballHudStateForTime(
  samples: BallTrailSample[],
  currentTime: number,
  options: Pick<BuildBallTrailOptions, "confidenceThreshold" | "currentSampleToleranceSeconds"> = {},
): BallHudReadout {
  const hiddenStyle = styleForBand("hidden", 0, options);
  if (!samples.length) {
    return { state: "not_visible", label: "ball: not visible", lowConfidence: false, sample: null, style: hiddenStyle };
  }
  const tolerance = options.currentSampleToleranceSeconds ?? DEFAULT_CURRENT_SAMPLE_TOLERANCE_SECONDS;
  const nearest = samples.reduce((best, sample) =>
    Math.abs(sample.t - currentTime) < Math.abs(best.t - currentTime) ? sample : best,
  );
  if (Math.abs(nearest.t - currentTime) > tolerance) {
    return { state: "not_visible", label: "ball: not visible", lowConfidence: false, sample: null, style: hiddenStyle };
  }
  const renderable = renderableSample(nearest);
  if (!renderable) {
    return { state: "not_visible", label: "ball: not visible", lowConfidence: false, sample: null, style: hiddenStyle };
  }
  const style = styleForBand(renderable.band, renderable.conf, options);
  const state = style.hudState;
  return {
    state,
    label: state === "measured" ? "ball: measured" : "ball: predicted",
    lowConfidence: style.lowConfidence,
    sample: renderable,
    style,
  };
}

/**
 * Explicit honest fail-closed HUD readout for when the ball-arc-solver
 * artifact reports an untrusted `status` (self-kill values such as
 * experimental_off/degenerate_zero_segments, or any unknown/future value).
 * Names the reason from the artifact's kill_reasons so the surface is
 * honest rather than silently falling back to a generic "not visible".
 */
export function solverOffReadout(killReasons: string[]): BallHudReadout {
  const reason = killReasons.length ? killReasons.join("; ") : "solver self-killed (untrusted status)";
  return {
    state: "solver_off",
    label: `ball: solver off — ${reason}`,
    lowConfidence: false,
    sample: null,
    style: styleForBand("hidden", 0),
    reason,
  };
}

export function parseBallTrailArtifact(input: unknown): BallTrailArtifact {
  const value = parseMaybeJson(input);
  assertRecord(value, "ball_track_arc_solved");
  const status = readArcSolverStatus(value.status, "ball_track_arc_solved.status");
  const trusted = TRUSTED_BALL_ARC_SOLVER_STATUSES.has(status);
  const killReasons = readKillReasons(value.kill_reasons, "ball_track_arc_solved.kill_reasons");
  const frames = readArray(value.frames, "ball_track_arc_solved.frames");
  return {
    samples: trusted ? frames.map((frame, index) => readBallTrailSample(frame, index)) : [],
    segments:
      trusted && value.segments !== undefined
        ? readArray(value.segments, "ball_track_arc_solved.segments").map(readBallArcSegment)
        : [],
    status,
    trusted,
    killReasons,
  };
}

export function samplesFromVirtualWorld(world: {
  ball: {
    frames: Array<{
      t: number;
      conf: number;
      visible: boolean;
      world_xyz?: Vec3 | null;
      approx?: boolean;
      render_only?: boolean;
      not_for_detection_metrics?: boolean;
      arc_segment_id?: number | string | null;
      confidence_provenance?: {
        band?: string | null;
        display_band?: string | null;
        predicted_sigma_m?: number | null;
      } | null;
      physics_fill?: { uncertainty_m?: number | null; render_only?: boolean | null } | null;
    }>;
  };
}): BallTrailSample[] {
  return world.ball.frames.map((frame) => {
    const band =
      frame.visible === false || !isVec3(frame.world_xyz)
        ? "hidden"
        : normalizeBand(frame.confidence_provenance?.band ?? frame.confidence_provenance?.display_band ?? null) === "unknown"
          ? frame.approx || frame.render_only || frame.not_for_detection_metrics || frame.physics_fill?.render_only
            ? "arc_interpolated"
            : "anchored_measured"
          : normalizeBand(frame.confidence_provenance?.band ?? frame.confidence_provenance?.display_band ?? null);
    return {
      t: frame.t,
      band,
      conf: frame.conf,
      visible: frame.visible,
      world_xyz: isVec3(frame.world_xyz) ? frame.world_xyz : null,
      renderOnly: Boolean(frame.render_only || frame.not_for_detection_metrics || frame.physics_fill?.render_only),
      segmentId: frame.arc_segment_id ?? null,
      sigmaM: readOptionalNumber(frame.confidence_provenance?.predicted_sigma_m) ?? readOptionalNumber(frame.physics_fill?.uncertainty_m),
    };
  });
}

export function parseAutoBounceCandidates(input: unknown): BounceCandidate[] {
  const value = parseMaybeJson(input);
  assertRecord(value, "auto_bounce_candidates");
  return readArray(value.candidates, "auto_bounce_candidates.candidates")
    .map((candidate, index) => readBounceCandidate(candidate, index))
    .filter((candidate): candidate is BounceCandidate => candidate !== null);
}

export function parseContactWindowsForImpacts(input: unknown): ContactWindowsForImpacts {
  const value = parseMaybeJson(input);
  assertRecord(value, "contact_windows");
  return {
    events: readArray(value.events, "contact_windows.events").map(readContactImpactEvent),
  };
}

export function parseNetPlane(input: unknown): NetPlane {
  const value = parseMaybeJson(input);
  assertRecord(value, "net_plane");
  const plane = value.plane;
  const point = plane && typeof plane === "object" && !Array.isArray(plane) ? readVec3((plane as Record<string, unknown>).point, "net_plane.plane.point") : null;
  const normal =
    plane && typeof plane === "object" && !Array.isArray(plane) ? readVec3((plane as Record<string, unknown>).normal, "net_plane.plane.normal") : null;
  const endpoints =
    value.endpoints === undefined
      ? []
      : readArray(value.endpoints, "net_plane.endpoints")
          .slice(0, 2)
          .map((entry, index) => readVec3(entry, `net_plane.endpoints[${index}]`));
  const endpointHeight = endpoints.length ? Math.max(...endpoints.map((endpoint) => endpoint[2])) : null;
  const topHeightM =
    readOptionalNumber(value.post_height_m) ??
    inchesToMeters(readOptionalNumber(value.post_height_in)) ??
    endpointHeight ??
    readOptionalNumber(value.center_height_m) ??
    inchesToMeters(readOptionalNumber(value.center_height_in)) ??
    0.91;
  return {
    point: point ?? endpoints[0] ?? [0, 0, 0],
    normal: normalizeVector(normal ?? [0, 1, 0]),
    topHeightM,
  };
}

export function netPlaneFromCourt(court: {
  net?: { endpoints?: [Vec3, Vec3]; center_height_m?: number; post_height_m?: number } | null;
}): NetPlane | null {
  const endpoints = court.net?.endpoints;
  if (!endpoints) return null;
  return {
    point: [0, endpoints[0][1], 0],
    normal: [0, 1, 0],
    topHeightM: court.net?.post_height_m ?? Math.max(endpoints[0][2], endpoints[1][2], court.net?.center_height_m ?? 0.86),
  };
}

export function extractImpactMarkers({
  samples,
  bounceCandidates = [],
  contactWindows = null,
  netPlane = null,
  arcSegments = [],
  landingLimit = 5,
  netTopMarginM = 0.03,
}: {
  samples: BallTrailSample[];
  bounceCandidates?: BounceCandidate[];
  contactWindows?: ContactWindowsForImpacts | null;
  netPlane?: NetPlane | null;
  arcSegments?: BallArcSegment[];
  landingLimit?: number;
  netTopMarginM?: number;
}): ImpactMarker[] {
  const markers: ImpactMarker[] = [];

  for (const bounce of bounceCandidates) {
    markers.push({
      kind: "floor_bounce",
      t: bounce.t,
      frame: bounce.frame,
      position: [bounce.position[0], bounce.position[1], Math.max(0.012, bounce.position[2])],
      confidence: bounce.confidence,
      label: bounce.humanReviewed ? "reviewed floor bounce" : "predicted floor bounce",
      source: bounce.source,
      positionSource: "bounce_candidate",
      derived: !bounce.humanReviewed,
    });
  }

  const landingCandidates = bounceCandidates
    .slice()
    .sort((left, right) => right.t - left.t)
    .slice(0, Math.max(0, landingLimit))
    .sort((left, right) => left.t - right.t);
  landingCandidates.forEach((bounce, index) => {
    markers.push({
      kind: "landing_spot",
      t: bounce.t,
      frame: bounce.frame,
      position: [bounce.position[0], bounce.position[1], 0.018],
      confidence: bounce.confidence,
      label: "landing spot",
      source: bounce.source,
      positionSource: "bounce_candidate",
      persistent: true,
      fadeRank: landingCandidates.length - index,
      derived: !bounce.humanReviewed,
    });
  });

  for (const event of contactWindows?.events ?? []) {
    const nearest = nearestRenderableSample(samples, event.t, 0.25);
    if (!nearest) continue;
    if (event.type === "contact") {
      markers.push({
        kind: "paddle_contact",
        t: event.t,
        frame: event.frame,
        position: nearest.world_xyz,
        confidence: event.confidence,
        label: "paddle contact",
        playerId: event.player_id,
        positionSource: "nearest_ball_sample",
        source: "contact_windows",
      });
    } else if (event.type === "bounce") {
      markers.push({
        kind: "floor_bounce",
        t: event.t,
        frame: event.frame,
        position: [nearest.world_xyz[0], nearest.world_xyz[1], Math.max(0.012, nearest.world_xyz[2])],
        confidence: event.confidence,
        label: "contact-window bounce",
        positionSource: "nearest_ball_sample",
        source: "contact_windows",
      });
    } else if (event.type === "net_cross") {
      markers.push({
        kind: "net_hit",
        t: event.t,
        frame: event.frame,
        position: nearest.world_xyz,
        confidence: event.confidence,
        label: "net event",
        positionSource: "nearest_ball_sample",
        source: "contact_windows",
        derived: false,
      });
    }
  }

  for (const segment of arcSegments) {
    if (segment.netClearanceM === null) continue;
    if (!(segment.netClearanceM <= 0 || segment.netClearanceOk === false)) continue;
    const crossing = netPlane ? crossingForTimeRange(samples, segment.t0, segment.t1, netPlane) : null;
    const fallback = nearestRenderableSample(samples, (segment.t0 + segment.t1) / 2, Math.max(0.25, segment.t1 - segment.t0));
    const position = crossing?.position ?? fallback?.world_xyz;
    if (!position) continue;
    markers.push({
      kind: "net_hit",
      t: crossing?.t ?? (segment.t0 + segment.t1) / 2,
      frame: fallback?.frame ?? null,
      position,
      confidence: 1,
      label: "net hit",
      positionSource: "net_clearance_flag",
      derivation: "artifact_net_clearance",
      derived: false,
    });
  }

  if (netPlane) {
    const derivedNetHits = derivedNetCrossings(samples, netPlane, netTopMarginM);
    for (const hit of derivedNetHits) {
      if (markers.some((marker) => marker.kind === "net_hit" && Math.abs(marker.t - hit.t) < 0.18)) continue;
      markers.push(hit);
    }
  }

  return markers;
}

function derivedNetCrossings(samples: BallTrailSample[], netPlane: NetPlane, netTopMarginM: number): ImpactMarker[] {
  const renderable = samples
    .map(renderableSample)
    .filter((sample): sample is BallTrailSample & { world_xyz: Vec3 } => sample !== null)
    .sort((left, right) => left.t - right.t);
  const markers: ImpactMarker[] = [];
  for (let index = 1; index < renderable.length; index += 1) {
    const from = renderable[index - 1];
    const to = renderable[index];
    const crossing = crossingBetweenSamples(from, to, netPlane);
    if (!crossing) continue;
    if (crossing.position[2] > netPlane.topHeightM + netTopMarginM) continue;
    if (markers.some((marker) => Math.abs(marker.t - crossing.t) < 0.18)) continue;
    markers.push({
      kind: "net_hit",
      t: crossing.t,
      frame: to.frame ?? from.frame ?? null,
      position: crossing.position,
      confidence: Math.min(from.conf, to.conf),
      label: "derived net hit",
      positionSource: "derived_net_crossing",
      derivation: "derived_crossing_below_net_top",
      derived: true,
    });
  }
  return markers;
}

function crossingForTimeRange(samples: BallTrailSample[], t0: number, t1: number, netPlane: NetPlane): { t: number; position: Vec3 } | null {
  const low = Math.min(t0, t1);
  const high = Math.max(t0, t1);
  const renderable = samples
    .filter((sample) => low <= sample.t && sample.t <= high)
    .map(renderableSample)
    .filter((sample): sample is BallTrailSample & { world_xyz: Vec3 } => sample !== null)
    .sort((left, right) => left.t - right.t);
  for (let index = 1; index < renderable.length; index += 1) {
    const crossing = crossingBetweenSamples(renderable[index - 1], renderable[index], netPlane);
    if (crossing) return crossing;
  }
  return null;
}

function crossingBetweenSamples(
  from: BallTrailSample & { world_xyz: Vec3 },
  to: BallTrailSample & { world_xyz: Vec3 },
  netPlane: NetPlane,
): { t: number; position: Vec3 } | null {
  const fromDistance = signedDistanceToPlane(from.world_xyz, netPlane);
  const toDistance = signedDistanceToPlane(to.world_xyz, netPlane);
  if (fromDistance === 0) return { t: from.t, position: from.world_xyz };
  if (toDistance === 0) return { t: to.t, position: to.world_xyz };
  if (Math.sign(fromDistance) === Math.sign(toDistance)) return null;
  const ratio = Math.abs(fromDistance) / Math.max(1e-9, Math.abs(fromDistance) + Math.abs(toDistance));
  return {
    t: from.t + (to.t - from.t) * ratio,
    position: interpolateVec3(from.world_xyz, to.world_xyz, ratio),
  };
}

function renderableSample(sample: BallTrailSample): (BallTrailSample & { world_xyz: Vec3 }) | null {
  const band = bandForSample(sample);
  if (band === "hidden" || sample.visible === false || !isVec3(sample.world_xyz)) return null;
  return { ...sample, band, world_xyz: sample.world_xyz };
}

function nearestRenderableSample(
  samples: BallTrailSample[],
  timeSeconds: number,
  toleranceSeconds: number,
): (BallTrailSample & { world_xyz: Vec3 }) | null {
  const candidates = samples
    .map(renderableSample)
    .filter((sample): sample is BallTrailSample & { world_xyz: Vec3 } => sample !== null)
    .map((sample) => ({ sample, dt: Math.abs(sample.t - timeSeconds) }))
    .filter((entry) => entry.dt <= toleranceSeconds)
    .sort((left, right) => left.dt - right.dt);
  return candidates[0]?.sample ?? null;
}

function bandForSample(sample: BallTrailSample): BallBand {
  if (sample.visible === false || !isVec3(sample.world_xyz)) return "hidden";
  return normalizeBand(sample.band);
}

function readBallTrailSample(input: unknown, index: number): BallTrailSample {
  const path = `ball_track_arc_solved.frames[${index}]`;
  assertRecord(input, path);
  const explicitBand = normalizeBand(readOptionalString(input.band));
  const visible = input.visible === undefined ? true : readBoolean(input.visible, `${path}.visible`);
  const world = isVec3(input.world_xyz) ? input.world_xyz : null;
  const arcSolver = readOptionalRecord(input.arc_solver);
  const weakSegment = arcSolver ? readOptionalBoolean(arcSolver.weak_segment) === true : false;
  const inferredBand = !visible || world === null ? "hidden" : explicitBand === "unknown" && weakSegment ? "arc_weak" : explicitBand;
  return {
    t: readNumber(input.t, `${path}.t`),
    frame: input.frame === undefined || input.frame === null ? null : readNumber(input.frame, `${path}.frame`, true),
    band: inferredBand,
    conf: input.conf === undefined || input.conf === null ? 0 : readNumber(input.conf, `${path}.conf`),
    visible,
    world_xyz: world,
    renderOnly: readOptionalBoolean(input.render_only) ?? (arcSolver ? readOptionalBoolean(arcSolver.render_only) ?? undefined : undefined),
    segmentId: readSegmentId(input.arc_segment_id ?? input.segment_id ?? arcSolver?.segment_id),
    sigmaM: readOptionalNumber(input.sigma_m),
    source: readOptionalString(input.source),
  };
}

function readBallArcSegment(input: unknown, index: number): BallArcSegment {
  const path = `ball_track_arc_solved.segments[${index}]`;
  assertRecord(input, path);
  return {
    segmentId: readRequiredSegmentId(input.segment_id, `${path}.segment_id`),
    t0: readNumber(input.t0, `${path}.t0`),
    t1: readNumber(input.t1, `${path}.t1`),
    netClearanceM: readOptionalNumber(input.net_clearance_m),
    netClearanceOk: readOptionalBoolean(input.net_clearance_ok) ?? null,
  };
}

function readBounceCandidate(input: unknown, index: number): BounceCandidate | null {
  const path = `auto_bounce_candidates.candidates[${index}]`;
  assertRecord(input, path);
  const worldPosition = isVec3(input.world_xyz)
    ? input.world_xyz
    : isVec2(input.world_xy_at_ball_radius)
      ? ([input.world_xy_at_ball_radius[0], input.world_xy_at_ball_radius[1], 0.0371] as Vec3)
      : null;
  if (!worldPosition) return null;
  return {
    t: readNumber(input.t, `${path}.t`),
    frame: input.frame === undefined || input.frame === null ? null : readNumber(input.frame, `${path}.frame`, true),
    position: worldPosition,
    confidence: input.conf === undefined || input.conf === null ? 1 : readNumber(input.conf, `${path}.conf`),
    source: readOptionalString(input.source) ?? readOptionalString(input.method) ?? "auto_bounce_candidate",
    humanReviewed: readOptionalBoolean(input.human_reviewed) ?? false,
  };
}

function readContactImpactEvent(input: unknown, index: number): ContactImpactEvent {
  const path = `contact_windows.events[${index}]`;
  assertRecord(input, path);
  assertRecord(input.window, `${path}.window`);
  return {
    type: readStringUnion(input.type, `${path}.type`, ["contact", "bounce", "net_cross"] as const),
    t: readNumber(input.t, `${path}.t`),
    frame: input.frame === undefined || input.frame === null ? null : readNumber(input.frame, `${path}.frame`, true),
    player_id: input.player_id === undefined || input.player_id === null ? null : readNumber(input.player_id, `${path}.player_id`, true),
    confidence: input.confidence === undefined || input.confidence === null ? 0 : readNumber(input.confidence, `${path}.confidence`),
    window: {
      t0: readNumber(input.window.t0, `${path}.window.t0`),
      t1: readNumber(input.window.t1, `${path}.window.t1`),
      importance:
        input.window.importance === undefined || input.window.importance === null
          ? 1
          : readNumber(input.window.importance, `${path}.window.importance`),
    },
  };
}

function leastCertainBand(left: BallBand, right: BallBand): BallBand {
  const rank: Record<BallBand, number> = {
    hidden: 5,
    arc_weak: 4,
    arc_extrapolated: 3,
    arc_interpolated: 2,
    unknown: 1,
    anchored_measured: 0,
  };
  return rank[left] >= rank[right] ? left : right;
}

function normalizeBand(input: BallBand | string | null | undefined): BallBand {
  if (
    input === "anchored_measured" ||
    input === "arc_interpolated" ||
    input === "arc_extrapolated" ||
    input === "arc_weak" ||
    input === "hidden"
  ) {
    return input;
  }
  return "unknown";
}

function signedDistanceToPlane(point: Vec3, netPlane: NetPlane): number {
  return dot([point[0] - netPlane.point[0], point[1] - netPlane.point[1], point[2] - netPlane.point[2]], netPlane.normal);
}

function dot(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function normalizeVector(vector: Vec3): Vec3 {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (length <= 1e-9) return [0, 1, 0];
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function interpolateVec3(from: Vec3, to: Vec3, t: number): Vec3 {
  return [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, from[2] + (to[2] - from[2]) * t];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function inchesToMeters(value: number | null): number | null {
  return value === null ? null : value * 0.0254;
}

function parseMaybeJson(input: unknown): unknown {
  return typeof input === "string" ? JSON.parse(input) : input;
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${path} must be an object`);
}

function readOptionalRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function readArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function readNumber(value: unknown, path: string, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a finite number`);
  if (integer && !Number.isInteger(value)) throw new Error(`${path} must be an integer`);
  return value;
}

function readOptionalNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${path} must be boolean`);
  return value;
}

function readOptionalBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readOptionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function readStringUnion<const T extends readonly string[]>(value: unknown, path: string, allowed: T): T[number] {
  if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
    throw new Error(`${path} must be one of ${allowed.join(", ")}`);
  }
  return value as T[number];
}

function readSegmentId(value: unknown): number | string | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) return value;
  return null;
}

function readRequiredSegmentId(value: unknown, path: string): number | string {
  const segmentId = readSegmentId(value);
  if (segmentId === null) throw new Error(`${path} must be a segment id`);
  return segmentId;
}

/**
 * Read the ball-arc-solver artifact's top-level `status`. A missing key
 * (undefined/null) defaults to "ran" (see TRUSTED_BALL_ARC_SOLVER_STATUSES
 * doc comment above for why); a present-but-non-string value is a parse
 * error like the rest of this file's strict readers.
 */
export function readArcSolverStatus(value: unknown, path = "status"): string {
  if (value === undefined || value === null) return "ran";
  if (typeof value !== "string") throw new Error(`${path} must be a string`);
  return value;
}

/** Read the ball-arc-solver artifact's top-level `kill_reasons` array of strings. */
export function readKillReasons(value: unknown, path = "kill_reasons"): string[] {
  if (value === undefined || value === null) return [];
  return readArray(value, path).map((item, index) => {
    if (typeof item !== "string") throw new Error(`${path}[${index}] must be a string`);
    return item;
  });
}

function isVec2(value: unknown): value is [number, number] {
  return Array.isArray(value) && value.length >= 2 && value.slice(0, 2).every((entry) => typeof entry === "number" && Number.isFinite(entry));
}

function isVec3(value: unknown): value is Vec3 {
  return Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every((entry) => typeof entry === "number" && Number.isFinite(entry));
}

function readVec3(value: unknown, path: string): Vec3 {
  if (!isVec3(value)) throw new Error(`${path} must be a finite [x,y,z] vector`);
  return [value[0], value[1], value[2]];
}
