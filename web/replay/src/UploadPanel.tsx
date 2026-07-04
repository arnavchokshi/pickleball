import React, { useEffect, useMemo, useState } from "react";

import {
  buildCourtCornersPayload,
  buildReviewedCourtCorrection,
  type CourtReviewPointMap,
} from "./courtReview";
import {
  apiUrl,
  fetchJobStatus,
  formatEta,
  jobProgressPercent,
  predictCourtLayout,
  saveCourtReview,
  uploadVideoJob,
  type CourtPrediction,
  type UploadJob,
} from "./uploadApi";

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

export function pipelineProgressLabel(job: UploadJob | null): string {
  if (!job) return "Waiting for upload";
  const stage = job.progress?.stage ?? jobStatusText(job);
  if (job.status === "complete") return stage;
  if (job.status === "failed") return stage;
  return `${stage} · ${formatEta(job.progress?.eta_seconds)} left`;
}

export function uploadErrorText(error: string | null | undefined): string | null {
  if (!error) return null;
  if (error.includes("intrinsics.source") && error.includes("not a trusted external calibration")) {
    return "Pipeline rejected an untrusted court calibration. The court prediction was saved as an unverified preview seed, not a trusted calibration.";
  }
  if (error.includes("local process_video failed") || error.includes("GPU pipeline failed")) {
    return "Pipeline failed while processing this video. Check the job logs for the full stage output.";
  }
  if (error.length > 260) return `${error.slice(0, 240).trim()}...`;
  return error;
}

export function UploadPanel({ apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() ?? "" }: UploadPanelProps) {
  const [video, setVideo] = useState<File | null>(null);
  const [courtPrediction, setCourtPrediction] = useState<CourtPrediction | null>(null);
  const [job, setJob] = useState<UploadJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [flowStage, setFlowStage] = useState<"idle" | "predicting" | "saving" | "submitting">("idle");
  const isBusy = flowStage !== "idle";
  const visibleProgress = isBusy && !job ? flowProgress(flowStage) : jobProgressPercent(job);
  const visibleProgressLabel = isBusy && !job ? flowProgressLabel(flowStage) : pipelineProgressLabel(job);

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
          setError(uploadErrorText(nextJob.error));
        })
        .catch((nextError) => setError(uploadErrorText(nextError instanceof Error ? nextError.message : String(nextError))));
    }, 2500);
    return () => window.clearInterval(timer);
  }, [apiBaseUrl, job]);

  async function predictAndProcess(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!video) {
      setError("Choose a video first.");
      return;
    }
    setFlowStage("predicting");
    setError(null);
    setJob(null);
    try {
      const prediction = await predictCourtLayout({ video }, { baseUrl: apiBaseUrl });
      setCourtPrediction(prediction);

      setFlowStage("saving");
      const adjusted = prediction.points as CourtReviewPointMap;
      const artifact = buildReviewedCourtCorrection({
        videoId: prediction.video.id,
        videoPath: prediction.video.path,
        videoSha256: prediction.video.sha256,
        imageSize: prediction.image_size,
        frameIndex: prediction.frame_index,
        frameTimeSeconds: prediction.frame_time_s,
        autoPredictionSource: prediction.prediction_source,
        predicted: prediction.points as CourtReviewPointMap,
        adjusted,
        createdAt: new Date().toISOString(),
        reviewStatus: "auto_predicted_unreviewed",
      });
      const cornerSeed = buildCourtCornersPayload({
        adjusted,
        imageSize: prediction.image_size,
        frameIndex: prediction.frame_index,
        source: prediction.prediction_source,
        reviewStatus: "auto_predicted_unreviewed",
      });
      const reviewResponse = await saveCourtReview(artifact as unknown as Record<string, unknown>, { baseUrl: apiBaseUrl });
      const reviewFile = jsonFile(reviewResponse.review, "reviewed_court_calibration.json");
      const cornerSeedFile = jsonFile(cornerSeed, "court_corners.json");

      setFlowStage("submitting");
      const nextJob = await uploadVideoJob(
        {
          video,
          courtCorners: cornerSeedFile,
          courtReview: reviewFile,
        },
        { baseUrl: apiBaseUrl },
      );
      setJob(nextJob);
    } catch (nextError) {
      setError(uploadErrorText(nextError instanceof Error ? nextError.message : String(nextError)));
    } finally {
      setFlowStage("idle");
    }
  }

  function updateVideo(nextVideo: File | null) {
    setVideo(nextVideo);
    setCourtPrediction(null);
    setJob(null);
  }

  return (
    <section className="upload-band" aria-label="Video upload">
      <form className="upload-form simple-upload-form" onSubmit={predictAndProcess}>
        <label>
          <span>Video</span>
          <input type="file" accept="video/mp4,video/quicktime,.mp4,.mov,.m4v" onChange={(event) => updateVideo(event.target.files?.[0] ?? null)} />
        </label>
        <button type="submit" className="upload-submit" disabled={isBusy || !video}>
          {isBusy ? flowButtonLabel(flowStage) : "Predict Court"}
        </button>
      </form>
      <div className="upload-progress" aria-label="Pipeline progress">
        <div className="upload-progress-head">
          <span>Pipeline progress</span>
          <strong>{visibleProgress}%</strong>
        </div>
        <div
          className="upload-progress-track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={visibleProgress}
          aria-label="Pipeline progress"
        >
          <div className="upload-progress-fill" style={{ width: `${visibleProgress}%` }} />
        </div>
        <div className="upload-progress-meta">{visibleProgressLabel}</div>
        {job?.progress?.steps?.length ? (
          <div className="upload-steps" aria-label="Pipeline steps">
            {job.progress.steps.map((step) => (
              <span key={step.id} className={`upload-step ${step.status}`}>
                {step.label}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      <div className="upload-status">
        <span className={`upload-state ${job?.status ?? "idle"}`}>{jobStatusText(job)}</span>
        {job?.id ? <span>{job.id}</span> : null}
        {courtPrediction ? <span className="upload-state submitted">Court preview saved</span> : null}
        {replayManifestUrl ? <a href={`/?manifest=${encodeURIComponent(replayManifestUrl)}`}>Open replay</a> : null}
        {error ? <span className="upload-error">{error}</span> : null}
      </div>
    </section>
  );
}

function jsonFile(payload: unknown, name: string): File {
  return new File([JSON.stringify(payload, null, 2)], name, { type: "application/json" });
}

function flowProgress(stage: "idle" | "predicting" | "saving" | "submitting"): number {
  if (stage === "predicting") return 18;
  if (stage === "saving") return 30;
  if (stage === "submitting") return 42;
  return 0;
}

function flowProgressLabel(stage: "idle" | "predicting" | "saving" | "submitting"): string {
  if (stage === "predicting") return "Predicting court";
  if (stage === "saving") return "Saving court seed";
  if (stage === "submitting") return "Submitting pipeline job";
  return "Waiting for upload";
}

function flowButtonLabel(stage: "idle" | "predicting" | "saving" | "submitting"): string {
  if (stage === "predicting") return "Predicting";
  if (stage === "saving") return "Saving";
  if (stage === "submitting") return "Submitting";
  return "Predict Court";
}
