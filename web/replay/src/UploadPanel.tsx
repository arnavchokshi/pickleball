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
  type PipelineStageSummary,
  type ResourceUsageSummary,
  type UploadJob,
} from "./uploadApi";
import "./uploadTelemetry.css";

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
      {job?.status === "complete" ? (
        <ResourceUsagePanel
          resourceSummary={job.result?.resource_summary}
          stageSummary={job.result?.pipeline_stage_summary}
          resourceUsageUrl={job.result?.resource_usage_url}
          pipelineSummaryUrl={job.result?.pipeline_summary_url}
          apiBaseUrl={apiBaseUrl}
        />
      ) : null}
    </section>
  );
}

export type ResourceUsagePanelProps = {
  resourceSummary?: ResourceUsageSummary | null;
  stageSummary?: PipelineStageSummary[] | null;
  resourceUsageUrl?: string | null;
  pipelineSummaryUrl?: string | null;
  apiBaseUrl?: string;
};

export function ResourceUsagePanel({
  resourceSummary,
  stageSummary,
  resourceUsageUrl,
  pipelineSummaryUrl,
  apiBaseUrl = "",
}: ResourceUsagePanelProps) {
  const stages = stageSummary ?? [];
  const hasSummary = resourceSummary && Object.keys(resourceSummary).length > 0;
  const hasStages = stages.length > 0;
  const stageMaxSeconds = Math.max(1, ...stages.map((stage) => stage.wall_seconds ?? 0));
  const resourceHref = resourceUsageUrl ? apiUrl(resourceUsageUrl, apiBaseUrl) : null;
  const pipelineHref = pipelineSummaryUrl ? apiUrl(pipelineSummaryUrl, apiBaseUrl) : null;

  if (!hasSummary && !hasStages) {
    return (
      <div className="resource-panel" aria-label="Pipeline resource results">
        <div className="resource-panel-head">
          <span>Run results</span>
          <strong>Telemetry unavailable</strong>
        </div>
      </div>
    );
  }

  const vramPercent =
    typeof resourceSummary?.gpu_memory_used_max_mb === "number" &&
    typeof resourceSummary?.gpu_memory_total_mb === "number" &&
    resourceSummary.gpu_memory_total_mb > 0
      ? (resourceSummary.gpu_memory_used_max_mb / resourceSummary.gpu_memory_total_mb) * 100
      : undefined;
  const systemRamPercent =
    typeof resourceSummary?.system_memory_used_max_mb === "number" &&
    typeof resourceSummary?.system_memory_total_mb === "number" &&
    resourceSummary.system_memory_total_mb > 0
      ? (resourceSummary.system_memory_used_max_mb / resourceSummary.system_memory_total_mb) * 100
      : undefined;

  const metrics = [
    {
      label: "GPU utilization",
      value: percentText(resourceSummary?.gpu_utilization_avg_pct),
      detail: peakText(resourceSummary?.gpu_utilization_max_pct),
      percent: resourceSummary?.gpu_utilization_avg_pct,
    },
    {
      label: "VRAM peak",
      value: bytesText(resourceSummary?.gpu_memory_used_max_mb),
      detail: resourceSummary?.gpu_memory_total_mb ? `of ${bytesText(resourceSummary.gpu_memory_total_mb)}` : "peak allocation",
      percent: vramPercent,
    },
    {
      label: "CPU avg",
      value: percentText(resourceSummary?.cpu_utilization_avg_pct),
      detail: peakText(resourceSummary?.cpu_utilization_max_pct),
      percent: resourceSummary?.cpu_utilization_avg_pct,
    },
    {
      label: "GPU power",
      value: wattsText(resourceSummary?.gpu_power_avg_w),
      detail: peakWattsText(resourceSummary?.gpu_power_max_w),
      percent: undefined,
    },
    {
      label: "System RAM",
      value: bytesText(resourceSummary?.system_memory_used_max_mb),
      detail: resourceSummary?.system_memory_total_mb ? `of ${bytesText(resourceSummary.system_memory_total_mb)}` : "peak system use",
      percent: systemRamPercent,
    },
    {
      label: "Run time",
      value: durationText(resourceSummary?.duration_s),
      detail: resourceSummary?.sample_count ? `${resourceSummary.sample_count} samples` : "resource samples",
      percent: 100,
    },
  ];

  return (
    <div className="resource-panel" aria-label="Pipeline resource results">
      <div className="resource-panel-head">
        <span>Run results</span>
        <strong>GPU telemetry</strong>
      </div>
      <div className="resource-metrics">
        {metrics.map((metric) => (
          <div className="resource-metric" key={metric.label}>
            <div className="resource-metric-row">
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
            <div className="resource-bar" aria-hidden="true">
              <span style={{ width: `${clampedPercent(metric.percent)}%` }} />
            </div>
            <small>{metric.detail}</small>
          </div>
        ))}
      </div>
      {hasStages ? (
        <div className="resource-stages" aria-label="Pipeline stage timing">
          {stages.map((stage) => (
            <div className="resource-stage" key={`${stage.stage}-${stage.wall_seconds ?? "unknown"}`}>
              <span>{stage.stage}</span>
              <div className="resource-stage-track" aria-hidden="true">
                <span style={{ width: `${clampedPercent(((stage.wall_seconds ?? 0) / stageMaxSeconds) * 100)}%` }} />
              </div>
              <strong>{durationText(stage.wall_seconds)}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {(resourceHref || pipelineHref) && (
        <div className="resource-links">
          {resourceHref ? <a href={resourceHref}>Resource JSON</a> : null}
          {pipelineHref ? <a href={pipelineHref}>Stage JSON</a> : null}
        </div>
      )}
    </div>
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

function clampedPercent(value: number | undefined): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function percentText(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${roundOne(value)}%`;
}

function peakText(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "peak n/a";
  return `${roundOne(value)}% peak`;
}

function bytesText(valueMb: number | undefined): string {
  if (typeof valueMb !== "number" || Number.isNaN(valueMb)) return "n/a";
  if (valueMb >= 1024) return `${roundOne(valueMb / 1024)} GB`;
  return `${Math.round(valueMb)} MB`;
}

function wattsText(valueWatts: number | undefined): string {
  if (typeof valueWatts !== "number" || Number.isNaN(valueWatts)) return "n/a";
  return `${roundOne(valueWatts)} W`;
}

function peakWattsText(valueWatts: number | undefined): string {
  if (typeof valueWatts !== "number" || Number.isNaN(valueWatts)) return "peak n/a";
  return `${roundOne(valueWatts)} W peak`;
}

function durationText(valueSeconds: number | undefined): string {
  if (typeof valueSeconds !== "number" || Number.isNaN(valueSeconds)) return "n/a";
  if (valueSeconds < 60) return `${roundOne(valueSeconds)}s`;
  const minutes = Math.floor(valueSeconds / 60);
  const seconds = Math.round(valueSeconds % 60);
  if (minutes < 60) return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
}

function roundOne(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
