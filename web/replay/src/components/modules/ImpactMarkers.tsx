import React, { useMemo } from "react";
import { DoubleSide } from "three";

import {
  extractImpactMarkers,
  netPlaneFromCourt,
  samplesFromVirtualWorld,
  type BallArcSegment,
  type BallTrailArtifact,
  type BallTrailSample,
  type BounceCandidate,
  type ContactWindowsForImpacts,
  type ImpactMarker,
  type NetPlane,
} from "./ballTrail";

export function ImpactMarkers({
  samples,
  arcSolved,
  world,
  bounceCandidates = [],
  contactWindows = null,
  netPlane = null,
  currentTime,
  landingLimit = 5,
}: {
  samples?: BallTrailSample[] | null;
  arcSolved?: BallTrailArtifact | null;
  world?: (Parameters<typeof samplesFromVirtualWorld>[0] & { court?: Parameters<typeof netPlaneFromCourt>[0] }) | null;
  bounceCandidates?: BounceCandidate[];
  contactWindows?: ContactWindowsForImpacts | null;
  netPlane?: NetPlane | null;
  currentTime: number;
  landingLimit?: number;
}) {
  const resolvedSamples = useMemo(() => {
    if (samples) return samples;
    if (arcSolved?.samples.length) return arcSolved.samples;
    return world ? samplesFromVirtualWorld(world) : [];
  }, [arcSolved, samples, world]);
  const resolvedNetPlane = netPlane ?? (world?.court ? netPlaneFromCourt(world.court) : null);
  const arcSegments: BallArcSegment[] = arcSolved?.segments ?? [];
  const markers = useMemo(
    () =>
      extractImpactMarkers({
        samples: resolvedSamples,
        bounceCandidates,
        contactWindows,
        netPlane: resolvedNetPlane,
        arcSegments,
        landingLimit,
      }),
    [arcSegments, bounceCandidates, contactWindows, landingLimit, resolvedNetPlane, resolvedSamples],
  );
  return (
    <group userData={{ layer: "ball-impact-markers-v1" }}>
      {markers.map((marker, index) => (
        <ImpactMarkerMesh key={`${marker.kind}-${marker.t.toFixed(3)}-${index}`} marker={marker} currentTime={currentTime} />
      ))}
    </group>
  );
}

function ImpactMarkerMesh({ marker, currentTime }: { marker: ImpactMarker; currentTime: number }) {
  if (marker.kind === "landing_spot") return <LandingSpot marker={marker} />;
  if (marker.kind === "floor_bounce") return <BounceRing marker={marker} currentTime={currentTime} />;
  if (marker.kind === "paddle_contact") return <ContactBurst marker={marker} currentTime={currentTime} />;
  return <NetHitMarker marker={marker} currentTime={currentTime} />;
}

function BounceRing({ marker, currentTime }: { marker: ImpactMarker; currentTime: number }) {
  const age = currentTime - marker.t;
  if (age < -0.18 || age > 0.75) return null;
  const progress = Math.max(0, Math.min(1, age / 0.75));
  const opacity = (1 - progress) * 0.72;
  const radius = 0.12 + progress * 0.46;
  return (
    <mesh position={[marker.position[0], marker.position[1], 0.024]} rotation={[0, 0, 0]} scale={[radius, radius, 1]} renderOrder={40}>
      <ringGeometry args={[0.82, 1, 48]} />
      <meshBasicMaterial color="#e8ff34" transparent opacity={opacity} depthWrite={false} side={DoubleSide} />
    </mesh>
  );
}

function LandingSpot({ marker }: { marker: ImpactMarker }) {
  const opacity = Math.max(0.18, Math.min(0.64, 0.18 + (marker.fadeRank ?? 1) * 0.08));
  return (
    <mesh position={marker.position} rotation={[0, 0, 0]} renderOrder={39}>
      <circleGeometry args={[0.06, 24]} />
      <meshBasicMaterial color="#dfff3d" transparent opacity={opacity} depthWrite={false} side={DoubleSide} />
    </mesh>
  );
}

function ContactBurst({ marker, currentTime }: { marker: ImpactMarker; currentTime: number }) {
  const age = Math.abs(currentTime - marker.t);
  if (age > 0.28) return null;
  const opacity = (1 - age / 0.28) * 0.9;
  return (
    <group position={marker.position} userData={{ impact: "paddle_contact", playerId: marker.playerId ?? null }}>
      <mesh renderOrder={44}>
        <sphereGeometry args={[0.095, 18, 18]} />
        <meshStandardMaterial color="#ffb454" emissive="#4a2500" transparent opacity={opacity} depthWrite={false} />
      </mesh>
      <mesh renderOrder={43}>
        <torusGeometry args={[0.18, 0.01, 8, 36]} />
        <meshBasicMaterial color="#ffb454" transparent opacity={opacity * 0.72} depthWrite={false} />
      </mesh>
    </group>
  );
}

function NetHitMarker({ marker, currentTime }: { marker: ImpactMarker; currentTime: number }) {
  const age = Math.abs(currentTime - marker.t);
  if (age > 0.6) return null;
  const opacity = marker.derived ? 0.55 : 0.82;
  return (
    <group position={marker.position} userData={{ impact: "net_hit", derived: marker.derived ?? false }}>
      <mesh renderOrder={45}>
        <octahedronGeometry args={[0.13, 0]} />
        <meshStandardMaterial color="#63d9ff" emissive="#10384a" transparent opacity={opacity} depthWrite={false} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} renderOrder={44}>
        <torusGeometry args={[0.24, 0.012, 8, 36]} />
        <meshBasicMaterial color="#63d9ff" transparent opacity={opacity * 0.4} depthWrite={false} />
      </mesh>
    </group>
  );
}
