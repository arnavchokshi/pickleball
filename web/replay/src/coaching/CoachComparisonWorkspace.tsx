import React, { useMemo, useState } from "react";

import {
  COACHING_COMPARISON_PHASES,
  type CoachingComparison,
  type CoachingComparisonCue,
  type CoachingComparisonPhase,
  type ComparisonReferenceClip,
  type ComparisonUserPhaseAnchor,
} from "./comparisonData";

export type CoachComparisonWorkspaceProps = {
  comparison: CoachingComparison;
  currentTime: number;
  onSeek: (timeSeconds: number) => void;
};

const PHASE_LABELS: Record<CoachingComparisonPhase, string> = {
  ready: "Ready",
  load: "Load",
  forward: "Forward",
  strike_window: "Strike window",
  finish: "Finish",
};

export function CoachComparisonWorkspace({ comparison, currentTime, onSeek }: CoachComparisonWorkspaceProps) {
  const [selectedCueId, setSelectedCueId] = useState<string | null>(comparison.cues[0]?.id ?? null);
  const selectedCue = comparison.cues.find((cue) => cue.id === selectedCueId) ?? comparison.cues[0] ?? null;
  const activePhase = useMemo(
    () =>
      currentTime >= comparison.user.interval.t0 && currentTime <= comparison.user.interval.t1
        ? nearestUserPhase(comparison.user_phase_anchors, currentTime)
        : null,
    [comparison.user.interval.t0, comparison.user.interval.t1, comparison.user_phase_anchors, currentTime],
  );
  const contactVerified = comparison.cues.some((cue) => cue.claim_boundary.contact === "supported");
  const referenceMotionReady = isMotionReference(comparison.reference) && comparison.reference.display_rights === "cleared";

  const chooseCue = (cue: CoachingComparisonCue) => {
    setSelectedCueId(cue.id);
    const anchor = comparison.user_phase_anchors.find((candidate) => candidate.phase === cue.phase);
    if (anchor) onSeek(anchor.user_t);
  };

  return (
    <section className="coach-compare" aria-label="Coach comparison">
      <header className="coach-compare-header">
        <div>
          <p className="coach-kicker">Coach compare · motion preview</p>
          <h2>{comparison.shot.label}</h2>
          <p className="coach-subtitle">
            Player {comparison.user.player_id} · {selectedCue
              ? "one clear change, tied to the exact motion below"
              : "the evidence did not support an actionable change"}
          </p>
        </div>
        <div className="coach-trust-row" aria-label="Comparison trust">
          <span className="coach-trust-chip preview">BODY estimate</span>
          <span className={`coach-trust-chip ${contactVerified ? "supported" : "unverified"}`}>
            {contactVerified ? "Contact verified" : "Ball contact not verified"}
          </span>
          <span className="coach-trust-chip neutral">No causal claim</span>
        </div>
      </header>

      <div className="coach-compare-grid">
        <article className="coach-reference-card">
          <p className="coach-card-label">Professional reference</p>
          <div className="coach-reference-number" aria-hidden="true">01</div>
          <div>
            <h3>{comparison.reference.athlete_name}</h3>
            <p>{referenceTitle(comparison.reference)}</p>
          </div>
          <div className="coach-reference-rights">
            <span>{referenceMotionReady ? "Reference motion cleared" : "Official lesson only"}</span>
            <small>
              {referenceMotionReady
                ? "A permissioned motion reference may be phase-aligned and overlaid."
                : "Linked unmodified. No pro footage or ghost mesh is copied into this result."}
            </small>
          </div>
          <a href={comparison.reference.source_url} target="_blank" rel="noreferrer" className="coach-reference-link">
            Watch official lesson <span aria-hidden="true">↗</span>
          </a>
        </article>

        <article className="coach-visual-card">
          <div className="coach-visual-heading">
            <div>
              <p className="coach-card-label">Selected correction</p>
              <h3>{selectedCue?.headline ?? "No actionable cue"}</h3>
            </div>
            <span>{selectedCue ? PHASE_LABELS[selectedCue.phase] : "Abstained"}</span>
          </div>
          {selectedCue ? (
            <>
              <TechniqueCueDiagram cue={selectedCue} referenceMotionReady={referenceMotionReady} />
              <p className="coach-diagram-caption">
                {referenceMotionReady
                  ? `Lime is your modeled motion. Cyan is the permissioned ${comparison.reference.athlete_name} reference.`
                  : `Lime is your modeled motion. Cyan is a technique guide—not ${comparison.reference.athlete_name} joint data.`}
              </p>
            </>
          ) : (
            <div className="coach-abstention-visual">No reliable comparison to draw.</div>
          )}
        </article>

        <article className="coach-action-card">
          <p className="coach-card-label">Do this next</p>
          {selectedCue ? (
            <>
              <div className="coach-action-rank">{String(selectedCue.rank).padStart(2, "0")}</div>
              <h3>{selectedCue.instruction}</h3>
              <div className="coach-measurement">
                <span>Your motion</span>
                <strong>{formatMeasurement(selectedCue)}</strong>
                <small>{measurementContext(selectedCue)}</small>
              </div>
              <details className="coach-measured-details">
                <summary>How measured</summary>
                <p>{selectedCue.claim_boundary.reason}</p>
                <p>
                  {Math.round(selectedCue.measurement.confidence * 100)}% model confidence · {selectedCue.measurement.provenance.replaceAll("_", " ")}
                </p>
              </details>
            </>
          ) : (
            <p className="coach-abstention">No comparison survived the evidence gates.</p>
          )}
        </article>
      </div>

      <div className="coach-control-deck">
        <nav className="coach-phase-rail" aria-label="Shot phases">
          {COACHING_COMPARISON_PHASES.map((phase, index) => {
            const anchor = comparison.user_phase_anchors.find((candidate) => candidate.phase === phase);
            const active = activePhase?.phase === phase;
            return (
              <button
                type="button"
                key={phase}
                className={active ? "coach-phase active" : "coach-phase"}
                aria-current={active ? "step" : undefined}
                disabled={!anchor}
                onClick={() => anchor && onSeek(anchor.user_t)}
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{PHASE_LABELS[phase]}</strong>
                <small>{anchor ? formatTime(anchor.user_t) : "—"}</small>
              </button>
            );
          })}
        </nav>

        {comparison.cues.length > 1 ? (
          <div className="coach-cue-tabs" aria-label="Priority corrections">
            {comparison.cues.map((cue) => (
              <button
                type="button"
                key={cue.id}
                className={cue.id === selectedCue?.id ? "coach-cue-tab active" : "coach-cue-tab"}
                aria-pressed={cue.id === selectedCue?.id}
                onClick={() => chooseCue(cue)}
              >
                <span>{cue.rank}</span>
                {cue.headline}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <p className="coach-pane-handoff">
        <span aria-hidden="true">↓</span> Your source video and 3D BODY motion stay synchronized below.
      </p>
    </section>
  );
}

export function nearestUserPhase(
  anchors: ComparisonUserPhaseAnchor[],
  currentTime: number,
): ComparisonUserPhaseAnchor | null {
  if (!anchors.length || !Number.isFinite(currentTime)) return null;
  return anchors.reduce((best, anchor) =>
    Math.abs(anchor.user_t - currentTime) < Math.abs(best.user_t - currentTime) ? anchor : best,
  );
}

function isMotionReference(reference: ComparisonReferenceClip): reference is Extract<ComparisonReferenceClip, { replay_manifest_url: string }> {
  return "replay_manifest_url" in reference;
}

function referenceTitle(reference: ComparisonReferenceClip): string {
  if (reference.reference_set_id === "ben_johns_joola_dink_lesson") return "Mastering the dink · JOOLA";
  return reference.exemplar_id.replaceAll("_", " ");
}

function formatMeasurement(cue: CoachingComparisonCue): string {
  const value = Number.isInteger(cue.measurement.user_value)
    ? String(cue.measurement.user_value)
    : cue.measurement.user_value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return `${value} ${cue.measurement.unit}`;
}

function measurementContext(cue: CoachingComparisonCue): string {
  if (cue.measurement.reference_value === null) return "Compared with a published technique rubric; no fabricated pro number.";
  const delta = cue.measurement.delta ?? 0;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(2)} ${cue.measurement.unit} versus the permissioned reference.`;
}

function formatTime(seconds: number): string {
  return `${seconds.toFixed(2)}s`;
}

function TechniqueCueDiagram({
  cue,
  referenceMotionReady,
}: {
  cue: CoachingComparisonCue | null;
  referenceMotionReady: boolean;
}) {
  const kind = cue?.visual.kind ?? "joint_arrow";
  return (
    <svg className={`coach-cue-diagram cue-${kind}`} viewBox="0 0 560 300" role="img" aria-label="Selected technique cue diagram">
      <defs>
        <marker id="coach-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
        </marker>
        <linearGradient id="coach-floor" x1="0" x2="1">
          <stop offset="0" stopColor="#dfff3d" stopOpacity="0.05" />
          <stop offset="1" stopColor="#63d9ff" stopOpacity="0.16" />
        </linearGradient>
      </defs>
      <path className="coach-diagram-floor" d="M70 251 L490 251 L540 286 L18 286 Z" fill="url(#coach-floor)" />
      <path className="coach-guide-zone" d="M228 64 C282 43 347 68 365 121 L347 218 C322 252 263 252 239 218 L211 126 Z" />
      <g className="coach-guide-figure" aria-label={referenceMotionReady ? "permissioned reference outline" : "published technique guide"}>
        <circle cx="286" cy="60" r="23" />
        <path d="M286 84 L281 150 M281 112 L232 155 M281 112 L333 147 M281 150 L249 236 M281 150 L320 236" />
      </g>
      <g className="coach-user-figure">
        <circle cx="258" cy="65" r="23" />
        <path d="M258 89 L244 154 M246 118 L188 165 M246 118 L350 172 M244 154 L210 239 M244 154 L285 240" />
        <circle cx="350" cy="172" r="8" />
      </g>
      {kind === "joint_arrow" ? (
        <>
          <path className="coach-correction-arrow" d="M350 171 C329 151 308 142 291 137" markerEnd="url(#coach-arrow)" />
          <text x="356" y="159">less reach</text>
        </>
      ) : null}
      {kind === "angle_arc" ? <path className="coach-correction-arc" d="M219 197 A52 52 0 0 1 270 183" /> : null}
      {kind === "stance_outline" ? <path className="coach-stance-outline" d="M225 249 L310 249" /> : null}
      {kind === "phase_timing" ? <path className="coach-timing-path" d="M165 184 C235 85 356 86 412 181" markerEnd="url(#coach-arrow)" /> : null}
      <g className="coach-diagram-key">
        <circle cx="36" cy="27" r="5" className="you" />
        <text x="49" y="31">you</text>
        <circle cx="105" cy="27" r="5" className="guide" />
        <text x="118" y="31">guide</text>
      </g>
    </svg>
  );
}
