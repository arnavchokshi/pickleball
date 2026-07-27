import { Canvas, useFrame, useThree } from "@react-three/fiber";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  Vector3,
} from "three";

import {
  COACHING_COMPARISON_PHASES,
  mapUserTimeToReferenceTime,
  type CoachingComparisonPhase,
} from "./comparisonData";
import {
  loadMeshCompareBundle,
  meshCompareRouteFromSearch,
  type MeshCompareBundle,
  type MeshCompareFetch,
  type NativeMeshSource,
} from "./meshCompareData";
import {
  bodyMeshIndexWindowForTime,
  fetchBodyMeshChunk,
  solidBodyMeshFramesForTime,
  type ActiveBodyMeshFrame,
  type BodyMesh,
  type Vec3,
} from "../viewerData";
import "./meshCompare.css";

export type MeshComparePageProps = {
  search?: string;
  hostname?: string;
  fetchImpl?: MeshCompareFetch;
};

type LoadState =
  | { status: "loading" }
  | { status: "blocked"; reason: string }
  | { status: "ready"; bundle: MeshCompareBundle };

type StudioView = "front" | "side" | "rear";
type CompareLayout = "overlay" | "side_by_side";

const PHASE_LABELS: Record<CoachingComparisonPhase, string> = {
  ready: "Ready",
  load: "Load",
  forward: "Forward",
  strike_window: "Strike",
  finish: "Finish",
};

export function MeshComparePage({ search, hostname, fetchImpl }: MeshComparePageProps) {
  const resolvedSearch = search ?? (typeof window === "undefined" ? "" : window.location.search);
  const resolvedHostname = hostname ?? (typeof window === "undefined" ? "" : window.location.hostname);
  const route = useMemo(() => meshCompareRouteFromSearch(resolvedSearch), [resolvedSearch]);
  const [loadState, setLoadState] = useState<LoadState>(() =>
    route ? { status: "loading" } : { status: "blocked", reason: "This comparison link is incomplete." },
  );

  useEffect(() => {
    if (!route) {
      setLoadState({ status: "blocked", reason: "This comparison link is incomplete." });
      return;
    }
    let cancelled = false;
    setLoadState({ status: "loading" });
    void loadMeshCompareBundle({ route, hostname: resolvedHostname, fetchImpl })
      .then((bundle) => {
        if (!cancelled) setLoadState({ status: "ready", bundle });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLoadState({ status: "blocked", reason: error instanceof Error ? error.message : String(error) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [fetchImpl, resolvedHostname, route]);

  if (loadState.status === "loading") return <MeshCompareLoading />;
  if (loadState.status === "blocked") return <MeshCompareBlocked reason={loadState.reason} />;
  return <MeshCompareExperience bundle={loadState.bundle} />;
}

export function MeshCompareExperience({
  bundle,
}: {
  bundle: MeshCompareBundle;
}) {
  const { comparison } = bundle;
  const firstCue = comparison.cues[0] ?? null;
  const [selectedCueId, setSelectedCueId] = useState<string | null>(firstCue?.id ?? null);
  const selectedCue = comparison.cues.find((cue) => cue.id === selectedCueId) ?? firstCue;
  const [time, setTime] = useState(comparison.user.interval.t0);
  const [playing, setPlaying] = useState(false);
  const [layout, setLayout] = useState<CompareLayout>("overlay");
  const [view, setView] = useState<StudioView>(() => preferredStudioView(firstCue?.visual.preferred_view));
  const duration = comparison.user.interval.t1 - comparison.user.interval.t0;
  const referenceTime = mapUserTimeToReferenceTime(comparison.alignment, time);

  useEffect(() => {
    if (!playing || duration <= 0) return;
    let animationFrame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const elapsed = Math.max(0, (now - previous) / 1000);
      previous = now;
      setTime((current) => {
        const next = current + elapsed;
        return next > comparison.user.interval.t1 ? comparison.user.interval.t0 : next;
      });
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrame);
  }, [comparison.user.interval.t0, comparison.user.interval.t1, duration, playing]);

  const seekPhase = (phase: CoachingComparisonPhase) => {
    const anchor = comparison.user_phase_anchors.find((candidate) => candidate.phase === phase);
    if (!anchor) return;
    setTime(anchor.user_t);
    setPlaying(false);
  };

  return (
    <main
      className="mesh-compare-page"
      aria-label={bundle.referencePresentation.publicDisplayReady ? "Professional mesh comparison" : "Local reference player mesh comparison"}
    >
      <header className="mesh-compare-header">
        <div className="mesh-compare-wordmark" aria-label="Form Study">
          <span>FORM</span>
          <strong>STUDY</strong>
        </div>
        <div className="mesh-compare-title">
          <p>{comparison.shot.label}</p>
          <h1>
            {bundle.referencePresentation.publicDisplayReady
              ? "Your motion, next to the pro."
              : "Your motion, overlaid frame by frame."}
          </h1>
        </div>
        <div className="mesh-compare-reference-id">
          <span>{bundle.referencePresentation.badge}</span>
          <strong>{bundle.referencePresentation.displayName ?? comparison.reference.athlete_name}</strong>
          {!bundle.referencePresentation.publicDisplayReady ? (
            <small>
              Local R&amp;D preview · not cleared for product or derivative use
              {!comparison.reference.expert_reviewed ? " · not coach-reviewed" : ""}
            </small>
          ) : null}
        </div>
      </header>

      <section className="mesh-compare-stage" aria-label="Real native mesh overlay">
        <NativeMeshComparisonStage
          bundle={bundle}
          userTime={time}
          referenceTime={referenceTime}
          layout={layout}
          view={view}
          cueJointName={selectedCue?.visual.joint_names[0] ?? null}
        />
        <div className="mesh-compare-legend" aria-label="Mesh colors">
          <span><i className="mesh-legend-user" />You</span>
          <span>
            <i className="mesh-legend-reference" />
            {bundle.referencePresentation.badge === "PRO" ? comparison.reference.athlete_name : "Reference player"}
          </span>
        </div>
        <div className="mesh-compare-mode" role="group" aria-label="Comparison layout">
          <button type="button" className={layout === "overlay" ? "active" : ""} onClick={() => setLayout("overlay")}>Overlay</button>
          <button type="button" className={layout === "side_by_side" ? "active" : ""} onClick={() => setLayout("side_by_side")}>Side by side</button>
        </div>
      </section>

      <aside className="mesh-compare-coach" aria-label="Primary coaching correction">
        <p className="mesh-compare-step">
          {comparison.reference.expert_reviewed ? "Change 01" : "Preview cue 01"}
        </p>
        <h2>{selectedCue?.headline ?? "No reliable correction"}</h2>
        <p className="mesh-compare-instruction">
          {selectedCue?.instruction ?? "The motion evidence did not support a specific change."}
        </p>
        {selectedCue ? (
          <div className="mesh-compare-delta">
            <span>Difference</span>
            <strong>{formatCueDelta(selectedCue.measurement.delta, selectedCue.measurement.unit)}</strong>
          </div>
        ) : null}
        {comparison.cues.length > 1 ? (
          <div className="mesh-compare-cue-tabs" role="group" aria-label="Coaching changes">
            {comparison.cues.map((cue) => (
              <button
                type="button"
                key={cue.id}
                className={cue.id === selectedCue?.id ? "active" : ""}
                onClick={() => {
                  setSelectedCueId(cue.id);
                  setView(preferredStudioView(cue.visual.preferred_view));
                  seekPhase(cue.phase);
                }}
              >
                {cue.rank}
              </button>
            ))}
          </div>
        ) : null}
        <p className="mesh-compare-boundary">Motion comparison only. Ball contact and outcome are not inferred here.</p>
      </aside>

      <footer className="mesh-compare-controls">
        <button type="button" className="mesh-play" onClick={() => setPlaying((value) => !value)}>
          {playing ? "Pause" : "Play"}
        </button>
        <div className="mesh-phase-track" role="group" aria-label="Shot phases">
          {COACHING_COMPARISON_PHASES.map((phase) => {
            const anchor = comparison.user_phase_anchors.find((candidate) => candidate.phase === phase);
            const active = anchor ? Math.abs(anchor.user_t - time) <= Math.max(0.055, duration / 18) : false;
            return (
              <button type="button" key={phase} className={active ? "active" : ""} onClick={() => seekPhase(phase)}>
                <i />
                <span>{PHASE_LABELS[phase]}</span>
              </button>
            );
          })}
        </div>
        <input
          className="mesh-time-slider"
          aria-label="Motion time"
          type="range"
          min={comparison.user.interval.t0}
          max={comparison.user.interval.t1}
          step={1 / 60}
          value={time}
          onChange={(event) => {
            setTime(Number(event.currentTarget.value));
            setPlaying(false);
          }}
        />
        <div className="mesh-view-buttons" role="group" aria-label="Camera view">
          {(["front", "side", "rear"] as const).map((candidate) => (
            <button type="button" key={candidate} className={view === candidate ? "active" : ""} onClick={() => setView(candidate)}>
              {candidate}
            </button>
          ))}
        </div>
      </footer>
    </main>
  );
}

function NativeMeshComparisonStage({
  bundle,
  userTime,
  referenceTime,
  layout,
  view,
  cueJointName,
}: {
  bundle: MeshCompareBundle;
  userTime: number;
  referenceTime: number;
  layout: CompareLayout;
  view: StudioView;
  cueJointName: string | null;
}) {
  const userMesh = useActiveNativeMesh(bundle.user, userTime);
  const referenceMesh = useActiveNativeMesh(bundle.reference, referenceTime);
  const userRoot = useMemo(() => meshRoot(userMesh.frame, bundle.user.world.joint_names), [bundle.user.world.joint_names, userMesh.frame]);
  const referenceRoot = useMemo(
    () => meshRoot(referenceMesh.frame, bundle.reference.world.joint_names),
    [bundle.reference.world.joint_names, referenceMesh.frame],
  );
  const ready = userMesh.frame !== null && referenceMesh.frame !== null && userRoot !== null && referenceRoot !== null;
  const loadFailure = userMesh.error ?? referenceMesh.error;

  if (loadFailure) return <MeshStageMessage title="Native mesh unavailable" detail={loadFailure} />;
  if (!ready) return <MeshStageMessage title="Loading native surfaces" detail="No skeleton fallback will be shown." />;

  const spatial = bundle.comparison.alignment.spatial;
  const userOffset: Vec3 = layout === "side_by_side" ? [-0.95, 0, 0] : [0, 0, 0];
  const referenceOffset: Vec3 = layout === "side_by_side" ? [0.95, 0, 0] : [0, 0, 0];
  return (
    <Canvas
      dpr={[1.25, 2]}
      gl={{ antialias: true, powerPreference: "high-performance", alpha: false }}
      camera={{ position: [0, -4.6, 0.15], fov: 29, near: 0.05, far: 30 }}
    >
      <color attach="background" args={["#eeeee9"]} />
      <fog attach="fog" args={["#eeeee9", 7, 13]} />
      <ambientLight intensity={2.4} />
      <directionalLight position={[-2.5, -3.5, 5]} intensity={2.8} color={new Color("#fff9e8")} />
      <directionalLight position={[3, 2, 2.5]} intensity={1.8} color={new Color("#a9eaff")} />
      <StudioCamera view={view} />
      <mesh position={[0, 0, -1.02]} rotation={[0, 0, 0]}>
        <circleGeometry args={[2.6, 96]} />
        <meshStandardMaterial color="#deded7" roughness={0.95} metalness={0} />
      </mesh>
      <DenseBodySurface
        active={referenceMesh.frame!}
        root={referenceRoot!}
        offset={referenceOffset}
        color="#42d4f4"
        opacity={layout === "overlay" ? 0.3 : 0.5}
        yawDegrees={spatial.facing_yaw_deg}
        scale={spatial.uniform_scale}
        mirrored={spatial.mirrored}
        wireframe={layout === "overlay"}
        renderOrder={1}
      />
      <DenseBodySurface
        active={userMesh.frame!}
        root={userRoot!}
        offset={userOffset}
        color="#cbed35"
        opacity={layout === "overlay" ? 0.68 : 0.58}
        yawDegrees={0}
        scale={1}
        mirrored={false}
        wireframe={false}
        renderOrder={2}
      />
      {layout === "overlay" && cueJointName ? (
        <CueDifferenceLine
          user={alignedJoint(userMesh.frame!, bundle.user.world.joint_names, cueJointName, userRoot!, 0, 1, false)}
          reference={alignedJoint(
            referenceMesh.frame!,
            bundle.reference.world.joint_names,
            cueJointName,
            referenceRoot!,
            spatial.facing_yaw_deg,
            spatial.uniform_scale,
            spatial.mirrored,
          )}
        />
      ) : null}
    </Canvas>
  );
}

function DenseBodySurface({
  active,
  root,
  offset,
  color,
  opacity,
  yawDegrees,
  scale,
  mirrored,
  wireframe,
  renderOrder,
}: {
  active: ActiveBodyMeshFrame;
  root: Vec3;
  offset: Vec3;
  color: string;
  opacity: number;
  yawDegrees: number;
  scale: number;
  mirrored: boolean;
  wireframe: boolean;
  renderOrder: number;
}) {
  const geometry = useMemo(() => denseGeometry(active.frame.mesh_vertices_world, active.frame.mesh_faces), [active.frame]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const signedScale = mirrored ? -scale : scale;
  return (
    <group position={offset} rotation={[0, 0, (yawDegrees * Math.PI) / 180]} scale={[signedScale, scale, scale]}>
      <mesh geometry={geometry} position={[-root[0], -root[1], -root[2]]} renderOrder={renderOrder}>
        <meshPhysicalMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.08}
          roughness={0.56}
          metalness={0}
          clearcoat={0.12}
          transparent
          opacity={opacity}
          depthWrite={layoutDepthWrite(opacity)}
          side={DoubleSide}
          wireframe={wireframe}
        />
      </mesh>
    </group>
  );
}

function CueDifferenceLine({ user, reference }: { user: Vec3 | null; reference: Vec3 | null }) {
  const geometry = useMemo(() => {
    if (!user || !reference) return null;
    const value = new BufferGeometry();
    value.setAttribute("position", new BufferAttribute(new Float32Array([...user, ...reference]), 3));
    return value;
  }, [reference, user]);
  useEffect(() => () => geometry?.dispose(), [geometry]);
  if (!geometry || !user || !reference) return null;
  return (
    <group>
      <lineSegments geometry={geometry} renderOrder={8}>
        <lineBasicMaterial color="#ff684a" transparent opacity={0.95} depthTest={false} />
      </lineSegments>
      <mesh position={user} renderOrder={9}>
        <sphereGeometry args={[0.035, 18, 18]} />
        <meshBasicMaterial color="#ff684a" depthTest={false} />
      </mesh>
      <mesh position={reference} renderOrder={9}>
        <sphereGeometry args={[0.035, 18, 18]} />
        <meshBasicMaterial color="#42d4f4" depthTest={false} />
      </mesh>
    </group>
  );
}

function StudioCamera({ view }: { view: StudioView }) {
  const { camera } = useThree();
  const target = useRef(new Vector3(0, 0, -0.05));
  useFrame(() => {
    const desired = view === "front" ? new Vector3(0, -4.6, 0.05) : view === "rear" ? new Vector3(0, 4.6, 0.05) : new Vector3(4.6, 0, 0.05);
    camera.position.lerp(desired, 0.12);
    camera.lookAt(target.current);
  });
  return null;
}

function useActiveNativeMesh(source: NativeMeshSource, time: number) {
  const window = bodyMeshIndexWindowForTime(source.index, time);
  const [chunk, setChunk] = useState<BodyMesh | null>(null);
  const [error, setError] = useState<string | null>(null);
  const verifiedChunkFetch = useMemo<MeshCompareFetch>(
    () => async (url) => {
      const bytes = source.verifiedChunkBytes.get(url);
      if (!bytes) return new Response("unverified native mesh chunk", { status: 403 });
      return new Response(bytes.slice().buffer, { status: 200 });
    },
    [source.verifiedChunkBytes],
  );
  useEffect(() => {
    if (!window) {
      setChunk(null);
      setError(`${source.role} native mesh has no window at this phase`);
      return;
    }
    let cancelled = false;
    setError(null);
    void fetchBodyMeshChunk(source.indexUrl, source.index, window, source.faces, verifiedChunkFetch)
      .then((value) => {
        if (!cancelled) setChunk(value);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [source, verifiedChunkFetch, window]);
  const frame = useMemo(
    // The comparison is pelvis-centered, so the replay viewer's extra
    // court-world alignment is unnecessary here. Keeping the immutable BODY
    // frame also lets React reuse its dense geometry between 60 Hz display
    // ticks instead of rebuilding 36,874 face normals on every tick.
    () => solidBodyMeshFramesForTime(chunk, null, time).find((candidate) => candidate.playerId === source.playerId) ?? null,
    [chunk, source.playerId, time],
  );
  return { frame, error };
}

export function meshRoot(active: ActiveBodyMeshFrame | null, jointNames: string[] | undefined): Vec3 | null {
  if (!active || !jointNames?.length) return null;
  const left = jointIndex(jointNames, "left_hip");
  const right = jointIndex(jointNames, "right_hip");
  if (left >= 0 && right >= 0 && active.frame.joints_world[left] && active.frame.joints_world[right]) {
    const a = active.frame.joints_world[left];
    const b = active.frame.joints_world[right];
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  }
  for (const name of ["pelvis", "hips", "root", "smpl_root"]) {
    const index = jointIndex(jointNames, name);
    if (index >= 0 && active.frame.joints_world[index]) return active.frame.joints_world[index];
  }
  return null;
}

export function alignedJoint(
  active: ActiveBodyMeshFrame,
  jointNames: string[] | undefined,
  jointName: string,
  root: Vec3,
  yawDegrees: number,
  scale: number,
  mirrored: boolean,
): Vec3 | null {
  if (!jointNames?.length) return null;
  const index = jointIndex(jointNames, jointName);
  const point = index >= 0 ? active.frame.joints_world[index] : undefined;
  if (!point) return null;
  const mirror = mirrored ? -1 : 1;
  const localX = (point[0] - root[0]) * scale * mirror;
  const localY = (point[1] - root[1]) * scale;
  const radians = (yawDegrees * Math.PI) / 180;
  return [
    localX * Math.cos(radians) - localY * Math.sin(radians),
    localX * Math.sin(radians) + localY * Math.cos(radians),
    (point[2] - root[2]) * scale,
  ];
}

function jointIndex(names: string[], target: string): number {
  const normalized = target.trim().toLowerCase();
  return names.findIndex((name) => name.trim().toLowerCase() === normalized);
}

function denseGeometry(vertices: Vec3[], faces: Array<[number, number, number]>): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(vertices.flat()), 3));
  geometry.setIndex(faces.flat());
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function layoutDepthWrite(opacity: number): boolean {
  return opacity >= 0.5;
}

function MeshCompareLoading() {
  return (
    <main className="mesh-compare-state" aria-label="Loading mesh comparison">
      <div className="mesh-state-mark"><span /></div>
      <p>FORM STUDY</p>
      <h1>Loading both native surfaces…</h1>
      <small>No skeleton fallback.</small>
    </main>
  );
}

function MeshCompareBlocked({ reason }: { reason: string }) {
  return (
    <main className="mesh-compare-state blocked" aria-label="Mesh comparison unavailable">
      <div className="mesh-state-mark"><span /></div>
      <p>REAL MESH REQUIRED</p>
      <h1>Comparison unavailable.</h1>
      <small>{reason}</small>
    </main>
  );
}

function MeshStageMessage({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="mesh-stage-message" role="status">
      <span />
      <strong>{title}</strong>
      <small>{detail}</small>
    </div>
  );
}

function preferredStudioView(value: "front" | "side" | "rear" | undefined): StudioView {
  return value ?? "front";
}

function formatCueDelta(delta: number | null, unit: string): string {
  if (delta === null) return "Phase-matched movement";
  const sign = delta > 0 ? "+" : "";
  return `${sign}${Number(delta.toFixed(2))} ${unit}`;
}

export default MeshComparePage;
