const DEFAULT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";

export type UploadJobStatus = "queued" | "running" | "complete" | "submitted" | "failed";

export type UploadJobStep = {
  id: string;
  label: string;
  status: "pending" | "active" | "complete" | "failed";
};

export type UploadJobProgress = {
  percent: number;
  stage: string;
  message?: string;
  eta_seconds?: number | null;
  steps?: UploadJobStep[];
};

export type UploadJob = {
  id: string;
  clip?: string;
  status: UploadJobStatus;
  progress?: UploadJobProgress | null;
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
  courtReview?: File | null;
  clip?: string;
  maxFrames?: number;
};

export type CourtPredictionPoint = { xy: [number, number]; confidence: number };
export type CourtPrediction = {
  schema_version: 1;
  artifact_type: "racketsport_court_layout_prediction";
  clip: string;
  image_size: [number, number];
  frame_index: number;
  frame_time_s: number;
  prediction_source: string;
  verified: false;
  not_cal3_verified: true;
  points: Record<string, CourtPredictionPoint>;
  lines?: Array<{ id: string; points: [[number, number], [number, number]] }>;
  warnings?: string[];
  video: {
    id: string;
    filename: string;
    path: string;
    sha256: string;
    size_bytes: number;
  };
};

export type SaveCourtReviewResponse = {
  review: Record<string, unknown>;
  court_calibration: Record<string, unknown>;
  saved: {
    review_path: string;
    court_calibration_path: string;
    index_path: string;
  };
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

export function jobProgressPercent(job: UploadJob | null): number {
  if (!job) return 0;
  if (job.status === "complete") return 100;
  const rawPercent = job.progress?.percent;
  if (typeof rawPercent !== "number" || Number.isNaN(rawPercent)) {
    if (job.status === "queued") return 0;
    if (job.status === "running" || job.status === "submitted") return 35;
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(rawPercent)));
}

export function formatEta(etaSeconds?: number | null): string {
  if (etaSeconds === null || etaSeconds === undefined || Number.isNaN(etaSeconds)) return "ETA calculating";
  if (etaSeconds < 60) return "less than 1 min";
  return `about ${Math.max(1, Math.round(etaSeconds / 60))} min`;
}

export async function uploadVideoJob(input: UploadVideoInput, options: UploadApiOptions = {}): Promise<UploadJob> {
  const body = new FormData();
  body.append("video", input.video);
  if (input.captureSidecar) body.append("capture_sidecar", input.captureSidecar);
  if (input.courtCorners) body.append("court_corners", input.courtCorners);
  if (input.courtCalibration) body.append("court_calibration", input.courtCalibration);
  if (input.courtReview) body.append("court_review", input.courtReview);
  if (input.clip?.trim()) body.append("clip", input.clip.trim());
  if (input.maxFrames !== undefined) body.append("max_frames", String(input.maxFrames));

  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/jobs", options.baseUrl), {
    method: "POST",
    body,
  });
  return parseJobResponse(response);
}

export async function predictCourtLayout(
  input: { video: File; clip?: string; frameIndex?: number },
  options: UploadApiOptions = {},
): Promise<CourtPrediction> {
  const body = new FormData();
  body.append("video", input.video);
  if (input.clip?.trim()) body.append("clip", input.clip.trim());
  if (input.frameIndex !== undefined) body.append("frame_index", String(input.frameIndex));
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/court/predict", options.baseUrl), { method: "POST", body });
  return parseJsonResponse<CourtPrediction>(response);
}

export async function saveCourtReview(
  payload: Record<string, unknown>,
  options: UploadApiOptions = {},
): Promise<SaveCourtReviewResponse> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/court/reviews", options.baseUrl), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJsonResponse<SaveCourtReviewResponse>(response);
}

export async function fetchJobStatus(statusUrl: string, options: UploadApiOptions = {}): Promise<UploadJob> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl(statusUrl, options.baseUrl));
  return parseJobResponse(response);
}

async function parseJobResponse(response: Response): Promise<UploadJob> {
  return parseJsonResponse<UploadJob>(response);
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json().catch(() => ({})) : {};
  if (!response.ok) {
    if (response.status === 404 && !isJson) {
      throw new Error("API server not found. Start the Render gateway or set VITE_API_BASE_URL.");
    }
    if (response.status >= 500 && !isJson) {
      throw new Error("API server not reachable. Start the Render gateway or set VITE_API_BASE_URL.");
    }
    const detail = typeof payload.detail === "string" ? payload.detail : `request failed with ${response.status}`;
    throw new Error(detail);
  }
  return payload as T;
}
