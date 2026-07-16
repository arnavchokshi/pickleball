export type RecoveryFetch = (url: string) => Promise<Response>;
export type RecoveryResponseKind = "binary" | "json";

export function manifestRelativeRecoveryUrl(assetUrl: string, manifestUrl: string): string | null {
  const cleanAsset = fsPath(assetUrl);
  const cleanManifest = fsPath(manifestUrl) ?? manifestUrl.split(/[?#]/, 1)[0];
  if (!cleanAsset || !cleanAsset.startsWith("/@fs/") || !cleanManifest || cleanAsset === cleanManifest) return null;
  const basename = cleanAsset.slice(cleanAsset.lastIndexOf("/") + 1);
  if (!basename) return null;
  const manifestDirectory = cleanManifest.slice(0, cleanManifest.lastIndexOf("/") + 1);
  const declaredSubdirectory = cleanAsset.includes("/body_mesh_index/") ? "body_mesh_index/" : "";
  const candidate = `${manifestDirectory}${declaredSubdirectory}${basename}`;
  return candidate === cleanAsset ? null : candidate;
}

function fsPath(url: string): string | null {
  const clean = url.split(/[?#]/, 1)[0];
  if (clean.startsWith("/@fs/")) return clean;
  try {
    const path = new URL(url).pathname;
    return path.startsWith("/@fs/") ? path : null;
  } catch {
    return null;
  }
}

/** Retry one same-basename local candidate after an unreachable VM-written /@fs URL. */
export async function fetchWithManifestRelativeRecovery(
  assetUrl: string,
  manifestUrl: string,
  fetchImpl: RecoveryFetch = (url) => fetch(url),
  onRecovered?: (originalUrl: string, recoveredUrl: string) => void,
  responseKind: RecoveryResponseKind = "binary",
): Promise<Response> {
  let originalResponse: Response | null = null;
  try {
    originalResponse = await fetchImpl(assetUrl);
    if (await responseIsUsable(originalResponse, responseKind)) return originalResponse;
  } catch {
    // Network, status, and invalid-content failures share one recovery path.
  }
  const recoveryUrl = manifestRelativeRecoveryUrl(assetUrl, manifestUrl);
  if (!recoveryUrl) {
    throw assetUnreachableError(assetUrl, responseKind);
  }
  try {
    const recoveredResponse = await fetchImpl(recoveryUrl);
    if (await responseIsUsable(recoveredResponse, responseKind)) {
      onRecovered?.(assetUrl, recoveryUrl);
      return recoveredResponse;
    }
  } catch {
    // Surface the original producer URL; that is the actionable manifest defect.
  }
  throw assetUnreachableError(assetUrl, responseKind);
}

export async function fetchJsonWithManifestRelativeRecovery(
  assetUrl: string,
  manifestUrl: string,
  fetchImpl: RecoveryFetch = (url) => fetch(url),
  onRecovered?: (originalUrl: string, recoveredUrl: string) => void,
): Promise<unknown> {
  const response = await fetchWithManifestRelativeRecovery(assetUrl, manifestUrl, fetchImpl, onRecovered, "json");
  try {
    return await response.json() as unknown;
  } catch {
    throw new Error(`asset unreadable: ${assetUrl} — the response was not valid JSON`);
  }
}

async function responseIsUsable(response: Response, responseKind: RecoveryResponseKind): Promise<boolean> {
  if (!response.ok) return false;
  if (responseKind === "binary") return true;
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.includes("application/json") && !contentType.includes("+json")) return false;
  try {
    return !(await response.clone().text()).trimStart().startsWith("<");
  } catch {
    return false;
  }
}

function assetUnreachableError(assetUrl: string, responseKind: RecoveryResponseKind): Error {
  const detail = responseKind === "json"
    ? "expected JSON but received an unavailable or non-JSON response"
    : "the asset could not be fetched";
  return new Error(`asset unreachable: ${assetUrl} — ${detail}; this manifest appears to have been written on another machine`);
}
