const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";

export type UploadJobStatus = "queued" | "running" | "complete" | "submitted" | "failed";

export type UploadJob = {
  id: string;
  clip?: string;
  status: UploadJobStatus;
  error?: string | null;
  result?: {
    manifest_url?: string | null;
    notes?: string[];
    remote_run_dir?: string | null;
  } | null;
  links: {
    status: string;
    manifest?: string;
  };
};

export type UploadVideoInput = {
  video: File;
  captureSidecar?: File | null;
  courtCorners?: File | null;
  courtCalibration?: File | null;
  clip?: string;
  maxFrames?: number;
};

export type UploadApiOptions = {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
};

export function apiUrl(path: string, baseUrl = DEFAULT_API_BASE_URL): string {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  const cleanBase = baseUrl.trim().replace(/\/+$/, "");
  return cleanBase ? `${cleanBase}${cleanPath}` : cleanPath;
}

export async function uploadVideoJob(input: UploadVideoInput, options: UploadApiOptions = {}): Promise<UploadJob> {
  const body = new FormData();
  body.append("video", input.video);
  if (input.captureSidecar) body.append("capture_sidecar", input.captureSidecar);
  if (input.courtCorners) body.append("court_corners", input.courtCorners);
  if (input.courtCalibration) body.append("court_calibration", input.courtCalibration);
  if (input.clip?.trim()) body.append("clip", input.clip.trim());
  if (input.maxFrames !== undefined) body.append("max_frames", String(input.maxFrames));

  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/jobs", options.baseUrl), {
    method: "POST",
    body,
  });
  return parseJobResponse(response);
}

export async function fetchJobStatus(statusUrl: string, options: UploadApiOptions = {}): Promise<UploadJob> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl(statusUrl, options.baseUrl));
  return parseJobResponse(response);
}

async function parseJobResponse(response: Response): Promise<UploadJob> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : `request failed with ${response.status}`;
    throw new Error(detail);
  }
  return payload as UploadJob;
}
