import { useFrame } from "@react-three/fiber";
import React, { useMemo, useRef } from "react";
import { Quaternion, Vector3 as ThreeVector3, type Mesh } from "three";

import {
  ballHudStateForTime,
  buildBallTrail,
  samplesFromVirtualWorld,
  type BallTrailArtifact,
  type BallTrailSample,
  type BallTrailSegment,
  type Vec3,
} from "./ballTrail";

type FocusStyle = {
  dimmed?: boolean;
  highlighted?: boolean;
};

export function BallTrailLayer({
  samples,
  arcSolved,
  world,
  currentTime,
  windowSeconds = 1.75,
  confidenceThreshold = 0.5,
  showTrail = true,
  showBall = true,
  focusStyle = {},
}: {
  samples?: BallTrailSample[] | null;
  arcSolved?: BallTrailArtifact | null;
  world?: Parameters<typeof samplesFromVirtualWorld>[0] | null;
  currentTime: number;
  windowSeconds?: number;
  confidenceThreshold?: number;
  showTrail?: boolean;
  showBall?: boolean;
  focusStyle?: FocusStyle;
}) {
  const resolvedSamples = useMemo(() => {
    if (samples) return samples;
    if (arcSolved?.samples.length) return arcSolved.samples;
    return world ? samplesFromVirtualWorld(world) : [];
  }, [arcSolved, samples, world]);
  const trail = useMemo(
    () => buildBallTrail(resolvedSamples, currentTime, { windowSeconds, confidenceThreshold }),
    [confidenceThreshold, currentTime, resolvedSamples, windowSeconds],
  );
  const current = useMemo(
    () => ballHudStateForTime(resolvedSamples, currentTime, { confidenceThreshold }),
    [confidenceThreshold, currentTime, resolvedSamples],
  );
  const focusOpacity = focusStyle.dimmed ? 0.34 : focusStyle.highlighted ? 1.22 : 1;
  const focusWidth = focusStyle.highlighted ? 1.5 : focusStyle.dimmed ? 0.75 : 1;
  return (
    <group userData={{ layer: "ball-trail-v1", ballState: current.state }}>
      {showTrail
        ? trail.segments.map((segment, index) => (
            <StyledTrailSegment
              key={`${segment.from.t.toFixed(3)}-${segment.to.t.toFixed(3)}-${index}`}
              segment={segment}
              opacityScale={focusOpacity}
              widthScale={focusWidth}
            />
          ))
        : null}
      {showBall && current.sample ? <CurrentBallSphere readout={current} focusStyle={focusStyle} /> : null}
    </group>
  );
}

function StyledTrailSegment({
  segment,
  opacityScale,
  widthScale,
}: {
  segment: BallTrailSegment;
  opacityScale: number;
  widthScale: number;
}) {
  const radius = Math.max(0.005, segment.style.lineWidth * widthScale * 0.0075);
  const opacity = Math.min(0.95, segment.style.opacity * opacityScale);
  if (segment.style.linePattern === "dashed") {
    return (
      <>
        {dashedPairs(segment.from.world_xyz, segment.to.world_xyz, 7).map(([from, to], index) => (
          <TubeSegment
            key={index}
            from={from}
            to={to}
            color={segment.style.color}
            opacity={opacity}
            radius={radius}
            renderOrder={31}
          />
        ))}
      </>
    );
  }
  return (
    <TubeSegment
      from={segment.from.world_xyz}
      to={segment.to.world_xyz}
      color={segment.style.color}
      opacity={opacity}
      radius={radius}
      renderOrder={30}
    />
  );
}

function CurrentBallSphere({
  readout,
  focusStyle,
}: {
  readout: ReturnType<typeof ballHudStateForTime>;
  focusStyle: FocusStyle;
}) {
  const style = readout.style;
  const radius = style.ballRadius * (focusStyle.highlighted ? 1.35 : 1);
  const opacity = focusStyle.dimmed ? Math.min(style.opacity, 0.24) : style.lowConfidence ? Math.min(style.opacity, 0.72) : style.opacity;
  return (
    <group position={readout.sample?.world_xyz ?? [0, 0, 0]} userData={{ ballState: readout.state, lowConfidence: readout.lowConfidence }}>
      {focusStyle.highlighted ? (
        <mesh renderOrder={36}>
          <sphereGeometry args={[radius * 2.25, 24, 24]} />
          <meshStandardMaterial color="#dfff3d" emissive="#526000" transparent opacity={0.2} depthWrite={false} />
        </mesh>
      ) : null}
      {style.pulsingBall ? <PulsingShell radius={radius * 1.9} color={style.color} dimmed={Boolean(focusStyle.dimmed)} /> : null}
      <mesh renderOrder={37}>
        <sphereGeometry args={[radius, 20, 20]} />
        <meshStandardMaterial color={style.color} emissive={style.emissiveColor} transparent opacity={opacity} depthWrite={false} />
      </mesh>
    </group>
  );
}

function PulsingShell({ radius, color, dimmed }: { radius: number; color: string; dimmed: boolean }) {
  const ref = useRef<Mesh | null>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const pulse = 1 + Math.sin(clock.elapsedTime * 7) * 0.16;
    ref.current.scale.setScalar(pulse);
  });
  return (
    <mesh ref={ref} renderOrder={35}>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial color={color} transparent opacity={dimmed ? 0.08 : 0.18} depthWrite={false} wireframe />
    </mesh>
  );
}

function TubeSegment({
  from,
  to,
  color,
  opacity,
  radius,
  renderOrder,
}: {
  from: Vec3;
  to: Vec3;
  color: string;
  opacity: number;
  radius: number;
  renderOrder: number;
}) {
  const transform = useMemo(() => {
    const start = new ThreeVector3(...from);
    const end = new ThreeVector3(...to);
    const direction = end.clone().sub(start);
    const length = direction.length();
    const midpoint = start.clone().add(end).multiplyScalar(0.5);
    const quaternion =
      length > 0
        ? new Quaternion().setFromUnitVectors(new ThreeVector3(0, 1, 0), direction.clone().normalize())
        : new Quaternion();
    return { length, position: [midpoint.x, midpoint.y, midpoint.z] as Vec3, quaternion };
  }, [from, to]);
  if (transform.length <= 1e-6) return null;
  return (
    <mesh position={transform.position} quaternion={transform.quaternion} renderOrder={renderOrder}>
      <cylinderGeometry args={[radius, radius, transform.length, 8]} />
      <meshStandardMaterial color={color} transparent opacity={opacity} roughness={0.5} metalness={0.02} depthWrite={false} />
    </mesh>
  );
}

function dashedPairs(from: Vec3, to: Vec3, dashCount: number): Array<[Vec3, Vec3]> {
  return Array.from({ length: dashCount }, (_, index) => {
    const start = index / dashCount;
    const end = Math.min(1, start + 0.5 / dashCount);
    return [interpolateVec3(from, to, start), interpolateVec3(from, to, end)] as [Vec3, Vec3];
  });
}

function interpolateVec3(from: Vec3, to: Vec3, t: number): Vec3 {
  return [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, from[2] + (to[2] - from[2]) * t];
}
