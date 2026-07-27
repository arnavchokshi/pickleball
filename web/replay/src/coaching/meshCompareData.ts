import {
  bindCoachingComparisonToManifest,
  comparisonHasReferenceMotion,
  parseCoachingComparison,
  sha256Utf8,
  type CoachingComparison,
  type CoachingComparisonAlignment,
  type ComparisonMotionReferenceClip,
} from "./comparisonData";
import {
  parseBodyMeshFaces,
  parseBodyMeshIndex,
  parseViewerManifest,
  parseVirtualWorld,
  resolveBodyMeshAssetUrl,
  resolveManifestChildUrl,
  resolveViewerManifestUrls,
  type BodyMeshFaces,
  type BodyMeshIndex,
  type BodyMeshIndexFrame,
  type ViewerManifest,
  type VirtualWorld,
} from "../viewerData";

export const MIN_DENSE_MESH_VERTICES = 1_000;
export const MIN_DENSE_MESH_FACES = 1_000;

export type MeshCompareRoute = {
  manifestUrl: string;
  comparisonUrl: string;
};

export type MeshReferencePresentation = {
  kind: "professional" | "local_reference_player";
  badge: "PRO" | "REFERENCE PLAYER";
  displayName: string | null;
  publicDisplayReady: boolean;
};

export type DenseMeshCoverage = {
  frameCount: number;
  vertexCount: number;
  faceCount: number;
  firstTime: number;
  lastTime: number;
  maxGapSeconds: number;
};

export type NativeMeshSource = {
  role: "user" | "reference";
  manifestUrl: string;
  manifest: ViewerManifest;
  world: VirtualWorld;
  indexUrl: string;
  index: BodyMeshIndex;
  faces: BodyMeshFaces;
  playerId: number;
  interval: { t0: number; t1: number };
  coverage: DenseMeshCoverage;
  verifiedChunkBytes: ReadonlyMap<string, Uint8Array>;
};

export type MeshCompareBundle = {
  comparison: CoachingComparison & {
    reference: ComparisonMotionReferenceClip;
    alignment: CoachingComparisonAlignment;
  };
  referencePresentation: MeshReferencePresentation;
  user: NativeMeshSource;
  reference: NativeMeshSource;
};

export type MeshCompareFetch = (url: string) => Promise<Response>;

export function meshCompareRouteFromSearch(search: string): MeshCompareRoute | null {
  const params = new URLSearchParams(search);
  if (params.get("view")?.trim().toLowerCase() !== "mesh_compare") return null;
  const manifestUrl = params.get("manifest")?.trim() ?? "";
  const comparisonUrl = params.get("comparison")?.trim() ?? "";
  return manifestUrl && comparisonUrl
    ? { manifestUrl, comparisonUrl }
    : null;
}

export function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.trim().toLowerCase().replace(/^\[/, "").replace(/\]$/, "");
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1";
}

export function referencePresentationForRights(
  displayRights: ComparisonMotionReferenceClip["display_rights"],
  hostname: string,
  trustedPublicClearance = false,
): MeshReferencePresentation {
  if (displayRights === "cleared" && trustedPublicClearance) {
    return { kind: "professional", badge: "PRO", displayName: null, publicDisplayReady: true };
  }
  if (displayRights === "cleared") {
    throw new Error("PRO display requires trusted server-issued reference clearance");
  }
  if (displayRights === "local_review_only" && isLoopbackHostname(hostname)) {
    return {
      kind: "local_reference_player",
      badge: "REFERENCE PLAYER",
      displayName: "Senior Pro reference — local review only",
      publicDisplayReady: false,
    };
  }
  throw new Error(
    "The reference motion is not cleared for this display. Local-review-only motion may be viewed only on localhost and is never labeled PRO.",
  );
}

export async function loadMeshCompareBundle({
  route,
  hostname,
  fetchImpl = (url) => fetch(url),
}: {
  route: MeshCompareRoute;
  hostname: string;
  fetchImpl?: MeshCompareFetch;
}): Promise<MeshCompareBundle> {
  const comparison = parseCoachingComparison(await fetchJson(route.comparisonUrl, fetchImpl));
  if (!comparisonHasReferenceMotion(comparison)) {
    throw new Error(
      "A real professional motion mesh has not been supplied. An official lesson link or published rubric cannot be rendered as a professional body mesh.",
    );
  }
  const referencePresentation = referencePresentationForRights(comparison.reference.display_rights, hostname);

  const [userManifestResult, referenceManifestResult] = await Promise.all([
    fetchManifest(route.manifestUrl, fetchImpl),
    fetchManifest(comparison.reference.replay_manifest_url, fetchImpl),
  ]);
  bindCoachingComparisonToManifest(comparison, userManifestResult.manifest, userManifestResult.sha256);
  if (referenceManifestResult.sha256 !== comparison.reference.replay_manifest_sha256) {
    throw new Error("professional reference manifest hash mismatch");
  }

  const [user, reference] = await Promise.all([
    loadNativeMeshSource({
      role: "user",
      manifestUrl: route.manifestUrl,
      manifest: userManifestResult.manifest,
      playerId: comparison.user.player_id,
      interval: comparison.user.interval,
      phaseTimes: comparison.user_phase_anchors.map((anchor) => anchor.user_t),
      comparison,
      fetchImpl,
    }),
    loadNativeMeshSource({
      role: "reference",
      manifestUrl: comparison.reference.replay_manifest_url,
      manifest: referenceManifestResult.manifest,
      playerId: comparison.reference.player_id,
      interval: comparison.reference.interval,
      phaseTimes: comparison.alignment.phase_anchors.map((anchor) => anchor.reference_t),
      comparison,
      fetchImpl,
    }),
  ]);

  return {
    comparison,
    referencePresentation,
    user,
    reference,
  };
}

export function validateDenseMeshCoverage({
  sourceLabel,
  index,
  faces,
  playerId,
  interval,
  phaseTimes,
}: {
  sourceLabel: string;
  index: BodyMeshIndex;
  faces: BodyMeshFaces;
  playerId: number;
  interval: { t0: number; t1: number };
  phaseTimes: number[];
}): DenseMeshCoverage {
  if (faces.mesh_faces.length < MIN_DENSE_MESH_FACES) {
    throw new Error(`${sourceLabel} has no native dense mesh topology`);
  }
  const frames = index.windows
    .flatMap((window) => window.players)
    .filter((player) => player.id === playerId)
    .flatMap((player) => player.frames)
    .filter((frame) => interval.t0 - 1e-6 <= frame.t && frame.t <= interval.t1 + 1e-6)
    .sort((left, right) => left.t - right.t);
  if (!frames.length) throw new Error(`${sourceLabel} has no native dense mesh frames for player ${playerId}`);
  const vertexCounts = new Set(frames.map((frame) => frame.vertex_count));
  if (vertexCounts.size !== 1 || Math.min(...vertexCounts) < MIN_DENSE_MESH_VERTICES) {
    throw new Error(`${sourceLabel} does not contain one stable native dense mesh surface`);
  }
  const frameTolerance = Math.max(0.05, 1.75 / Math.max(index.fps, 1));
  const firstTime = frames[0].t;
  const lastTime = frames[frames.length - 1].t;
  if (firstTime > interval.t0 + frameTolerance || lastTime < interval.t1 - frameTolerance) {
    throw new Error(`${sourceLabel} native mesh does not cover the complete compared motion`);
  }
  let maxGapSeconds = 0;
  for (let indexValue = 1; indexValue < frames.length; indexValue += 1) {
    maxGapSeconds = Math.max(maxGapSeconds, frames[indexValue].t - frames[indexValue - 1].t);
  }
  if (maxGapSeconds > frameTolerance + 1e-9) {
    throw new Error(`${sourceLabel} native mesh contains an internal coverage gap`);
  }
  for (const phaseTime of phaseTimes) {
    if (!frames.some((frame) => Math.abs(frame.t - phaseTime) <= frameTolerance)) {
      throw new Error(`${sourceLabel} is missing native mesh evidence at a required shot phase`);
    }
  }
  return {
    frameCount: frames.length,
    vertexCount: frames[0].vertex_count,
    faceCount: faces.mesh_faces.length,
    firstTime,
    lastTime,
    maxGapSeconds,
  };
}

async function loadNativeMeshSource({
  role,
  manifestUrl,
  manifest,
  playerId,
  interval,
  phaseTimes,
  comparison,
  fetchImpl,
}: {
  role: NativeMeshSource["role"];
  manifestUrl: string;
  manifest: ViewerManifest;
  playerId: number;
  interval: { t0: number; t1: number };
  phaseTimes: number[];
  comparison: CoachingComparison;
  fetchImpl: MeshCompareFetch;
}): Promise<NativeMeshSource> {
  if (!manifest.body_mesh_index_url) {
    throw new Error(`${role} replay has no native dense body-mesh index`);
  }
  if (manifest.mesh_status !== "windowed_index") {
    throw new Error(`${role} replay does not declare a windowed native mesh index`);
  }
  const evidenceRole = role === "user" ? "user_motion" : "reference_motion";
  const indexUrl = manifest.body_mesh_index_url;
  const [worldAsset, indexAsset] = await Promise.all([
    fetchTextAsset(manifest.virtual_world_url, fetchImpl),
    fetchTextAsset(indexUrl, fetchImpl),
  ]);
  assertEvidenceSha256(comparison, evidenceRole, "racketsport_virtual_world", manifest.virtual_world_url, worldAsset.sha256);
  assertEvidenceSha256(comparison, evidenceRole, "racketsport_body_mesh_index", indexUrl, indexAsset.sha256);
  const world = parseVirtualWorld(parseJsonAsset(worldAsset.text, manifest.virtual_world_url));
  const parsedIndex = parseBodyMeshIndex(parseJsonAsset(indexAsset.text, indexUrl));
  if (parsedIndex.clip !== manifest.clip) throw new Error(`${role} body-mesh index clip mismatch`);
  if (!world.players.some((player) => player.id === playerId)) {
    throw new Error(`${role} player ${playerId} is absent from the synchronized world`);
  }
  const facesUrl = resolveManifestChildUrl(indexUrl, parsedIndex.faces_url);
  const facesAsset = await fetchTextAsset(facesUrl, fetchImpl);
  assertEvidenceSha256(comparison, evidenceRole, "racketsport_body_mesh_faces", facesUrl, facesAsset.sha256);
  const faces = parseBodyMeshFaces(parseJsonAsset(facesAsset.text, facesUrl));
  if (faces.faces_ref !== parsedIndex.faces_ref) throw new Error(`${role} native mesh topology reference mismatch`);
  const verifiedChunkBytes = new Map<string, Uint8Array>();
  const windows = [];
  for (const window of parsedIndex.windows) {
    const overlaps = window.t1 >= interval.t0 - 1e-6 && window.t0 <= interval.t1 + 1e-6;
    if (!overlaps) {
      windows.push(window);
      continue;
    }
    const chunkUrl = resolveManifestChildUrl(indexUrl, window.url);
    const chunkAsset = await fetchBinaryAsset(chunkUrl, fetchImpl);
    assertEvidenceSha256(
      comparison,
      evidenceRole,
      "racketsport_body_mesh_chunk_decoded",
      chunkUrl,
      chunkAsset.sha256,
    );
    const integrityUrl = appendIntegrityQuery(window.url, chunkAsset.sha256);
    const resolvedIntegrityUrl = resolveBodyMeshAssetUrl(indexUrl, integrityUrl);
    verifiedChunkBytes.set(resolvedIntegrityUrl, chunkAsset.bytes);
    windows.push({ ...window, url: integrityUrl });
  }
  const index: BodyMeshIndex = { ...parsedIndex, windows };
  const coverage = validateDenseMeshCoverage({
    sourceLabel: role,
    index,
    faces,
    playerId,
    interval,
    phaseTimes,
  });
  return { role, manifestUrl, manifest, world, indexUrl, index, faces, playerId, interval, coverage, verifiedChunkBytes };
}

async function fetchManifest(
  manifestUrl: string,
  fetchImpl: MeshCompareFetch,
): Promise<{ manifest: ViewerManifest; sha256: string }> {
  const text = await fetchText(manifestUrl, fetchImpl);
  let payload: unknown;
  try {
    payload = JSON.parse(text) as unknown;
  } catch {
    throw new Error(`manifest is not valid JSON: ${manifestUrl}`);
  }
  return {
    manifest: resolveViewerManifestUrls(parseViewerManifest(payload), manifestUrl),
    sha256: await sha256Utf8(text),
  };
}

async function fetchJson(url: string, fetchImpl: MeshCompareFetch): Promise<unknown> {
  const text = await fetchText(url, fetchImpl);
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`asset is not valid JSON: ${url}`);
  }
}

function parseJsonAsset(text: string, url: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`asset is not valid JSON: ${url}`);
  }
}

async function fetchTextAsset(url: string, fetchImpl: MeshCompareFetch): Promise<{ text: string; sha256: string }> {
  const text = await fetchText(url, fetchImpl);
  return { text, sha256: await sha256Utf8(text) };
}

async function fetchBinaryAsset(
  url: string,
  fetchImpl: MeshCompareFetch,
): Promise<{ bytes: Uint8Array; sha256: string }> {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`asset request failed (${response.status}): ${url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  // Browsers transparently decode a `.gz` response when the server declares
  // `Content-Encoding: gzip`, while file-backed tests receive the compressed
  // bytes. Bind integrity to the canonical decoded chunk so both transports
  // verify the same immutable BODY payload.
  const canonicalBytes = await canonicalBodyMeshChunkBytes(bytes);
  const digest = await crypto.subtle.digest("SHA-256", canonicalBytes);
  const sha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  return { bytes, sha256 };
}

async function canonicalBodyMeshChunkBytes(bytes: Uint8Array): Promise<Uint8Array> {
  if (bytes.length < 2 || bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  if (typeof DecompressionStream === "undefined") {
    throw new Error("gzip mesh integrity verification is unavailable in this runtime");
  }
  const stream = new Blob([bytes.slice().buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function assertEvidenceSha256(
  comparison: CoachingComparison,
  role: "user_motion" | "reference_motion",
  artifactType: string,
  url: string,
  actualSha256: string,
): void {
  const candidates = comparison.source_artifacts.filter(
    (source) => source.role === role && source.artifact_type === artifactType,
  );
  const matching = candidates.filter((source) => evidenceUriMatchesUrl(source.uri, url));
  const evidence = matching.length === 1 ? matching[0] : candidates.length === 1 ? candidates[0] : null;
  if (!evidence) throw new Error(`${role} ${artifactType} has no unique integrity evidence`);
  if (evidence.sha256 !== actualSha256) throw new Error(`${role} ${artifactType} hash mismatch`);
}

function evidenceUriMatchesUrl(uri: string, url: string): boolean {
  const cleanUrl = decodeURIComponent(url.split(/[?#]/, 1)[0]);
  const filePath = cleanUrl.replace(/^\/@fs\//, "/");
  return filePath === uri || filePath.endsWith(uri) || uri.endsWith(filePath);
}

function appendIntegrityQuery(url: string, sha256: string): string {
  return `${url}${url.includes("?") ? "&" : "?"}sha256=${sha256}`;
}

async function fetchText(url: string, fetchImpl: MeshCompareFetch): Promise<string> {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`asset request failed (${response.status}): ${url}`);
  return response.text();
}

export function denseMeshFramesForPlayer(index: BodyMeshIndex, playerId: number): BodyMeshIndexFrame[] {
  return index.windows
    .flatMap((window) => window.players)
    .filter((player) => player.id === playerId)
    .flatMap((player) => player.frames)
    .sort((left, right) => left.t - right.t);
}
