import React, { useCallback, useEffect, useMemo, useState } from "react";

import { authedFetch, type AuthApiOptions } from "./authApi";
import { apiUrl, formatEta, jobProgressPercent, parseJsonResponse, type UploadJob } from "./uploadApi";
import { planParts, type UploadPlan } from "./uploadPlan";

// ---------------------------------------------------------------------------
// API shapes -- must match server/routes/clips.py + server/routes/jobs_v2.py
// exactly (do not invent fields).
// ---------------------------------------------------------------------------

export type ClipRecord = {
  id: string;
  filename: string;
  status: string;
  size_bytes: number;
  key: string;
  job_id?: string | null;
  created_at?: string | null;
};

export type ClipsListResponse = {
  clips: ClipRecord[];
};

export type PresignedPart = {
  part_number: number;
  url: string;
};

export type CreateClipResponse = {
  id: string;
  filename: string;
  key: string;
  upload_id: string;
  part_count: number;
  part_urls: PresignedPart[];
  sidecar_upload_url: string;
};

export type CompletedPart = {
  part_number: number;
  etag: string;
};

export type CompleteClipResponse = {
  id: string;
  status: string;
  key: string;
};

export const DEFAULT_PART_SIZE_BYTES = 8 * 1024 * 1024;
export const DEFAULT_POLL_DELAY_MS = 2500;

// ---------------------------------------------------------------------------
// Pure API calls (all authed; testable directly against a fake fetch without
// mounting React -- this repo's test setup has no jsdom/interaction harness,
// only vitest's node environment + renderToStaticMarkup, matching the
// existing uploadApi.ts / UploadPanel.tsx convention).
// ---------------------------------------------------------------------------

export async function fetchClips(options: AuthApiOptions = {}): Promise<ClipRecord[]> {
  const response = await authedFetch("/api/clips", { method: "GET" }, options);
  const payload = await parseJsonResponse<ClipsListResponse>(response);
  return payload.clips;
}

export async function createClip(
  file: { name: string; size: number },
  partSizeBytes: number,
  options: AuthApiOptions = {},
): Promise<CreateClipResponse> {
  const response = await authedFetch(
    "/api/clips",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ filename: file.name, size_bytes: file.size, part_size_bytes: partSizeBytes }),
    },
    options,
  );
  return parseJsonResponse<CreateClipResponse>(response);
}

/**
 * PUTs every part directly to S3 via its presigned URL (plain `fetchImpl`,
 * NOT `authedFetch`/`apiUrl` -- these are absolute S3 URLs that must not
 * carry our own bearer token, and `apiUrl` would mangle an absolute URL).
 * Captures the `ETag` response header S3 returns per part.
 */
export async function uploadPartsToS3(
  file: Blob,
  plan: UploadPlan,
  partUrls: PresignedPart[],
  fetchImpl: typeof fetch,
): Promise<CompletedPart[]> {
  const urlByPart = new Map(partUrls.map((part) => [part.part_number, part.url]));
  const parts: CompletedPart[] = [];
  for (const range of plan.ranges) {
    const url = urlByPart.get(range.partNumber);
    if (!url) {
      throw new Error(`missing presigned URL for part ${range.partNumber}`);
    }
    const body = file.slice(range.offset, range.offset + range.length);
    const response = await fetchImpl(url, { method: "PUT", body });
    if (!response.ok) {
      throw new Error(`upload of part ${range.partNumber} failed with status ${response.status}`);
    }
    const etag = response.headers.get("ETag") ?? response.headers.get("etag");
    if (!etag) {
      throw new Error(`missing ETag response header for part ${range.partNumber} (check S3 CORS ExposeHeaders)`);
    }
    parts.push({ part_number: range.partNumber, etag });
  }
  return parts;
}

export async function completeClip(
  clipId: string,
  uploadId: string,
  parts: CompletedPart[],
  options: AuthApiOptions = {},
): Promise<CompleteClipResponse> {
  const response = await authedFetch(
    `/api/clips/${clipId}/complete`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId, parts }),
    },
    options,
  );
  return parseJsonResponse<CompleteClipResponse>(response);
}

export async function createJob(clipId: string, options: AuthApiOptions = {}): Promise<UploadJob> {
  const response = await authedFetch(
    "/api/jobs",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ clip_id: clipId, pipeline_preset: "court_skeletons" }),
    },
    options,
  );
  return parseJsonResponse<UploadJob>(response);
}

const TERMINAL_JOB_STATUSES: ReadonlySet<UploadJob["status"]> = new Set(["complete", "failed"]);

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * Polls `GET /api/jobs/{id}` until the job settles (complete/failed).
 * `maxAttempts` is a defensive cap so a server bug (job stuck non-terminal)
 * fails loudly instead of polling forever.
 */
export async function pollJobUntilSettled(
  statusPath: string,
  options: AuthApiOptions,
  pollDelayMs: number,
  onUpdate?: (job: UploadJob) => void,
  maxAttempts = 200,
): Promise<UploadJob> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const response = await authedFetch(statusPath, { method: "GET" }, options);
    const job = await parseJsonResponse<UploadJob>(response);
    onUpdate?.(job);
    if (TERMINAL_JOB_STATUSES.has(job.status)) {
      return job;
    }
    if (pollDelayMs > 0) {
      await delay(pollDelayMs);
    }
  }
  throw new Error(`job at ${statusPath} did not settle after ${maxAttempts} polls`);
}

export type UploadStage = "creating_clip" | "uploading_parts" | "completing_clip" | "creating_job" | "polling_job";

export type UploadCallbacks = {
  onStage?: (stage: UploadStage) => void;
  onClipCreated?: (clip: CreateClipResponse) => void;
  onJobUpdate?: (job: UploadJob) => void;
};

/**
 * Drives the full presigned-upload sequence in order: (a) POST /api/clips,
 * (b) PUT every part to S3, (c) POST /api/clips/{id}/complete, (d) POST
 * /api/jobs, (e) poll GET /api/jobs/{id} until settled.
 */
export async function uploadClipAndQueueJob(
  file: File,
  options: AuthApiOptions,
  partSizeBytes: number = DEFAULT_PART_SIZE_BYTES,
  pollDelayMs: number = DEFAULT_POLL_DELAY_MS,
  callbacks: UploadCallbacks = {},
): Promise<{ clip: CreateClipResponse; job: UploadJob }> {
  callbacks.onStage?.("creating_clip");
  const clip = await createClip(file, partSizeBytes, options);
  callbacks.onClipCreated?.(clip);

  callbacks.onStage?.("uploading_parts");
  const plan = planParts(file.size, partSizeBytes);
  const parts = await uploadPartsToS3(file, plan, clip.part_urls, options.fetchImpl ?? fetch);

  callbacks.onStage?.("completing_clip");
  await completeClip(clip.id, clip.upload_id, parts, options);

  callbacks.onStage?.("creating_job");
  const createdJob = await createJob(clip.id, options);

  callbacks.onStage?.("polling_job");
  const job = await pollJobUntilSettled(createdJob.links.status, options, pollDelayMs, callbacks.onJobUpdate);

  return { clip, job };
}

// ---------------------------------------------------------------------------
// Presentation helpers
// ---------------------------------------------------------------------------

export type CombinedClipStatus = "uploading" | "uploaded" | "queued" | "running" | "submitted" | "complete" | "failed";

export function combinedClipStatus(clip: ClipRecord, job?: UploadJob | null): CombinedClipStatus {
  if (job) return job.status;
  if (clip.status === "uploaded") return "uploaded";
  return "uploading";
}

export function clipStatusLabel(status: CombinedClipStatus): string {
  switch (status) {
    case "uploading":
      return "Uploading";
    case "uploaded":
      return "Uploaded, waiting to process";
    case "queued":
      return "Queued";
    case "running":
    case "submitted":
      return "Processing on GPU";
    case "complete":
      return "Ready";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

export function uploadStageLabel(stage: UploadStage): string {
  switch (stage) {
    case "creating_clip":
      return "Preparing upload...";
    case "uploading_parts":
      return "Uploading video...";
    case "completing_clip":
      return "Finishing upload...";
    case "creating_job":
      return "Queuing pipeline job...";
    case "polling_job":
      return "Processing on GPU...";
    default:
      return "Working...";
  }
}

export function manifestUrlForJob(job: UploadJob | undefined, baseUrl?: string): string | null {
  if (!job) return null;
  const manifestPath = job.result?.manifest_url ?? job.links.manifest;
  if (!manifestPath) return null;
  return apiUrl(manifestPath, baseUrl);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export type LibraryScreenProps = AuthApiOptions & {
  onOpenViewer: (manifestUrl: string) => void;
  onLogout: () => void;
  partSizeBytes?: number;
  pollDelayMs?: number;
};

export function LibraryScreen({
  onOpenViewer,
  onLogout,
  fetchImpl,
  baseUrl,
  partSizeBytes = DEFAULT_PART_SIZE_BYTES,
  pollDelayMs = DEFAULT_POLL_DELAY_MS,
}: LibraryScreenProps) {
  const options = useMemo<AuthApiOptions>(() => ({ fetchImpl, baseUrl }), [fetchImpl, baseUrl]);
  const [clips, setClips] = useState<ClipRecord[]>([]);
  const [jobsByClipId, setJobsByClipId] = useState<Record<string, UploadJob>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadStage, setUploadStage] = useState<UploadStage | null>(null);

  const refreshClips = useCallback(async () => {
    try {
      const nextClips = await fetchClips(options);
      setClips(nextClips);
      setLoadError(null);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, [options]);

  useEffect(() => {
    void refreshClips();
  }, [refreshClips]);

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setUploadError(null);
    let createdClipId = "";
    try {
      const { clip, job } = await uploadClipAndQueueJob(file, options, partSizeBytes, pollDelayMs, {
        onStage: setUploadStage,
        onClipCreated: (created) => {
          createdClipId = created.id;
          void refreshClips();
        },
        onJobUpdate: (nextJob) => {
          if (createdClipId) {
            setJobsByClipId((prev) => ({ ...prev, [createdClipId]: nextJob }));
          }
        },
      });
      setJobsByClipId((prev) => ({ ...prev, [clip.id]: job }));
      await refreshClips();
      const manifestHref = manifestUrlForJob(job, baseUrl);
      if (job.status === "complete" && manifestHref) {
        onOpenViewer(manifestHref);
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : String(error));
    } finally {
      setUploadStage(null);
    }
  }

  async function checkClipStatus(clip: ClipRecord) {
    if (!clip.job_id) return;
    try {
      const response = await authedFetch(`/api/jobs/${clip.job_id}`, { method: "GET" }, options);
      const job = await parseJsonResponse<UploadJob>(response);
      setJobsByClipId((prev) => ({ ...prev, [clip.id]: job }));
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <section className="library-screen" aria-label="Video library">
      <header className="library-header">
        <h1>Library</h1>
        <button type="button" className="library-logout" onClick={onLogout}>
          Log out
        </button>
      </header>

      <div className="library-upload">
        <label htmlFor="library-upload-input">
          <span>Upload a clip</span>
          <input
            id="library-upload-input"
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
            disabled={uploadStage !== null}
            onChange={(event) => void handleUpload(event)}
          />
        </label>
        {uploadStage ? <p className="library-upload-stage">{uploadStageLabel(uploadStage)}</p> : null}
        {uploadError ? (
          <p role="alert" className="library-upload-error">
            {uploadError}
          </p>
        ) : null}
      </div>

      {loadError ? (
        <p role="alert" className="library-load-error">
          {loadError}
        </p>
      ) : null}

      {loading ? (
        <p className="library-loading">Loading clips...</p>
      ) : clips.length === 0 ? (
        <p className="library-empty">No clips yet. Upload one above.</p>
      ) : (
        <ul className="library-clip-list">
          {clips.map((clip) => {
            const job = jobsByClipId[clip.id];
            const status = combinedClipStatus(clip, job);
            const manifestHref = manifestUrlForJob(job, baseUrl);
            const showProgress = job && (status === "running" || status === "queued" || status === "submitted");
            const showCheckStatus = !job && Boolean(clip.job_id) && status !== "complete";
            return (
              <li key={clip.id} className={`library-clip library-clip-${status}`}>
                <span className="library-clip-name">{clip.filename}</span>
                <span className="library-clip-status">{clipStatusLabel(status)}</span>
                {showProgress ? (
                  <span className="library-clip-progress">
                    {jobProgressPercent(job)}% - {formatEta(job?.progress?.eta_seconds)}
                  </span>
                ) : null}
                {status === "complete" && manifestHref ? (
                  <button type="button" className="library-clip-open" onClick={() => onOpenViewer(manifestHref)}>
                    Open replay
                  </button>
                ) : null}
                {showCheckStatus ? (
                  <button type="button" className="library-clip-check" onClick={() => void checkClipStatus(clip)}>
                    Check status
                  </button>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

export default LibraryScreen;
