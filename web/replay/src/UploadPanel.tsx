import React, { useEffect, useMemo, useState } from "react";

import { apiUrl, fetchJobStatus, uploadVideoJob, type UploadJob } from "./uploadApi";

type UploadPanelProps = {
  apiBaseUrl?: string;
};

export function jobStatusText(job: UploadJob | null): string {
  if (!job) return "No upload queued";
  if (job.status === "queued") return "Queued";
  if (job.status === "running" || job.status === "submitted") return "Processing on GPU";
  if (job.status === "complete") return "Replay ready";
  if (job.status === "failed") return "Failed";
  return job.status;
}

export function UploadPanel({ apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() ?? "" }: UploadPanelProps) {
  const [video, setVideo] = useState<File | null>(null);
  const [captureSidecar, setCaptureSidecar] = useState<File | null>(null);
  const [courtCalibration, setCourtCalibration] = useState<File | null>(null);
  const [clip, setClip] = useState("");
  const [maxFrames, setMaxFrames] = useState("");
  const [job, setJob] = useState<UploadJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const replayManifestUrl = useMemo(() => {
    const manifestUrl = job?.result?.manifest_url;
    return manifestUrl ? apiUrl(manifestUrl, apiBaseUrl) : null;
  }, [apiBaseUrl, job?.result?.manifest_url]);

  useEffect(() => {
    if (!job || !["queued", "running", "submitted"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      void fetchJobStatus(job.links.status, { baseUrl: apiBaseUrl })
        .then((nextJob) => {
          setJob(nextJob);
          setError(nextJob.error ?? null);
        })
        .catch((nextError) => setError(nextError instanceof Error ? nextError.message : String(nextError)));
    }, 2500);
    return () => window.clearInterval(timer);
  }, [apiBaseUrl, job]);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!video) {
      setError("Choose a video first.");
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const nextJob = await uploadVideoJob(
        {
          video,
          captureSidecar,
          courtCalibration,
          clip,
          maxFrames: maxFrames ? Number(maxFrames) : undefined,
        },
        { baseUrl: apiBaseUrl },
      );
      setJob(nextJob);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <section className="upload-band" aria-label="Video upload">
      <form className="upload-form" onSubmit={submit}>
        <label>
          <span>Video</span>
          <input type="file" accept="video/mp4,video/quicktime,.mp4,.mov,.m4v" onChange={(event) => setVideo(event.target.files?.[0] ?? null)} />
        </label>
        <label>
          <span>Capture sidecar</span>
          <input type="file" accept="application/json,.json" onChange={(event) => setCaptureSidecar(event.target.files?.[0] ?? null)} />
        </label>
        <label>
          <span>Court calibration</span>
          <input type="file" accept="application/json,.json" onChange={(event) => setCourtCalibration(event.target.files?.[0] ?? null)} />
        </label>
        <label>
          <span>Clip ID</span>
          <input type="text" value={clip} placeholder="drill_01" onChange={(event) => setClip(event.target.value)} />
        </label>
        <label className="short-input">
          <span>Frame cap</span>
          <input type="number" min="1" value={maxFrames} onChange={(event) => setMaxFrames(event.target.value)} />
        </label>
        <button type="submit" className="upload-submit" disabled={isUploading || !video}>
          {isUploading ? "Uploading" : "Upload and process"}
        </button>
      </form>
      <div className="upload-status">
        <span className={`upload-state ${job?.status ?? "idle"}`}>{jobStatusText(job)}</span>
        {job?.id ? <span>{job.id}</span> : null}
        {replayManifestUrl ? <a href={`/?manifest=${encodeURIComponent(replayManifestUrl)}`}>Open replay</a> : null}
        {error ? <span className="upload-error">{error}</span> : null}
      </div>
    </section>
  );
}
