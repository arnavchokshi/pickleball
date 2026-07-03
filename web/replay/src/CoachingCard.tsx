import React from "react";

import { coachingTrustChipClass, type CoachingCardFact, type CoachingCardFacts, type TimelineChapter } from "./viewerData";

export type CoachingCardRow = {
  playerId: string;
  fact: CoachingCardFact | null;
};

export type CoachingCardView = {
  status: "missing_artifact" | "no_rally_chapters" | "no_active_rally" | "ready";
  rallyId: string | null;
  rallyLabel: string | null;
  rows: CoachingCardRow[];
};

export function coachingCardForTimeline(
  facts: CoachingCardFacts | null,
  chapters: TimelineChapter[],
  currentTime: number,
  playerIds: string[],
): CoachingCardView {
  if (!facts) return { status: "missing_artifact", rallyId: null, rallyLabel: null, rows: [] };
  if (!chapters.length) return { status: "no_rally_chapters", rallyId: null, rallyLabel: null, rows: [] };

  const activeChapter = activeTimelineChapter(chapters, currentTime);
  if (!activeChapter) return { status: "no_active_rally", rallyId: null, rallyLabel: null, rows: [] };

  const rallyId = rallyIdForTimelineChapter(activeChapter);
  const factsForRally = facts.facts.filter((fact) => fact.rally_id === rallyId);
  const factByPlayer = new Map(factsForRally.map((fact) => [fact.player_id, fact]));
  const rowPlayerIds = uniqueSortedPlayerIds([...playerIds, ...factsForRally.map((fact) => fact.player_id)]);

  return {
    status: "ready",
    rallyId,
    rallyLabel: activeChapter.label,
    rows: rowPlayerIds.map((playerId) => ({ playerId, fact: factByPlayer.get(playerId) ?? null })),
  };
}

export function CoachingCardPanel({ card }: { card: CoachingCardView }) {
  if (card.status === "missing_artifact") {
    return (
      <section className="coaching-card" aria-label="Per-rally coaching card">
        <h2>Per-rally coaching card</h2>
        <p className="coaching-card-caveat">Coaching facts artifact not available.</p>
      </section>
    );
  }
  if (card.status === "no_rally_chapters") {
    return (
      <section className="coaching-card" aria-label="Per-rally coaching card">
        <h2>Per-rally coaching card</h2>
        <p className="coaching-card-caveat">No rally chapters available from the timeline.</p>
      </section>
    );
  }
  if (card.status === "no_active_rally") {
    return (
      <section className="coaching-card" aria-label="Per-rally coaching card">
        <h2>Per-rally coaching card</h2>
        <p className="coaching-card-caveat">Scrub into a rally chapter to view measured coaching facts.</p>
      </section>
    );
  }

  return (
    <section className="coaching-card" aria-label="Per-rally coaching card">
      <h2>
        Per-rally coaching card - {card.rallyLabel} ({card.rallyId})
      </h2>
      <ul className="coaching-card-list">
        {card.rows.map((row) => (
          <li key={row.playerId} className={row.fact ? "coaching-card-row" : "coaching-card-row missing"}>
            <span className="coaching-card-player">p{row.playerId}</span>
            {row.fact ? (
              <>
                <span className="coaching-card-metric">{metricLabel(row.fact.metric)}</span>
                <strong>{formatFactValue(row.fact)}</strong>
                <span className={`trust-badge-chip ${coachingTrustChipClass(row.fact.trust)}`}>{row.fact.trust}</span>
                <span className="coaching-card-coverage">{formatCoverage(row.fact.coverage_fraction)} coverage</span>
              </>
            ) : (
              <span className="trust-badge-chip none">not measured</span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function activeTimelineChapter(chapters: TimelineChapter[], currentTime: number): TimelineChapter | null {
  return chapters.find((chapter) => chapter.t0 <= currentTime && currentTime <= chapter.t1) ?? null;
}

function rallyIdForTimelineChapter(chapter: TimelineChapter): string {
  if (chapter.rallyId) return chapter.rallyId;
  return `rally_${String(Math.max(0, chapter.index - 1)).padStart(3, "0")}`;
}

function uniqueSortedPlayerIds(playerIds: string[]): string[] {
  return Array.from(new Set(playerIds.filter((id) => id.trim().length > 0))).sort(comparePlayerIds);
}

function comparePlayerIds(left: string, right: string): number {
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber) && leftNumber !== rightNumber) {
    return leftNumber - rightNumber;
  }
  return left.localeCompare(right);
}

function metricLabel(metric: string): string {
  return metric.replaceAll("_", " ");
}

function formatFactValue(fact: CoachingCardFact): string {
  const value = Number.isInteger(fact.value) ? String(fact.value) : fact.value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return `${value} ${fact.unit}`.trim();
}

function formatCoverage(coverageFraction: number): string {
  return `${(coverageFraction * 100).toFixed(1)}%`;
}
