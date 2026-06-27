"""Habit report data assembly."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from threed.racketsport.schemas import Habit, HabitReport, RacketSportMetrics


REPORT_SPORT = "pickleball"
REPORT_PLACEHOLDER_DRILL_MINUTES = 5.0
REPORT_CLIP_PADDING_SEC = 1.0


@dataclass(frozen=True)
class MetricFact:
    player_id: int
    shot_index: int
    shot_time_s: float
    shot_type: str
    metric: str
    value: Any
    confidence: float
    gated: bool

    @property
    def habit_id(self) -> str:
        return f"p{self.player_id}_s{self.shot_index:03d}_{_slug(self.metric)}"

    @property
    def correction_paths(self) -> set[str]:
        return {
            f"/players/{self.player_id}/shots/{self.shot_index}/metrics/{self.metric}",
            f"/players/{self.player_id}/shots/{self.shot_index}/metrics/{_escape_pointer(self.metric)}",
        }


@dataclass(frozen=True)
class ReportExclusion:
    path: str
    reason: str


def build_habit_report(
    metrics: RacketSportMetrics,
    *,
    exclusions: list[ReportExclusion] | None = None,
) -> HabitReport:
    facts = list(iter_metric_facts(metrics))
    excluded_paths = {exclusion.path for exclusion in exclusions or []}
    skipped: Counter[str] = Counter()
    habits: list[Habit] = []

    for fact in facts:
        if fact.gated:
            skipped["gated_metric"] += 1
            continue
        if fact.correction_paths & excluded_paths:
            skipped["manual_exclusion"] += 1
            continue
        habits.append(habit_from_metric_fact(fact))

    total = len(facts)
    coverage = (len(habits) / total) if total else 0.0
    priority_habit_id = habits[0].id if habits else ""

    return HabitReport(
        schema_version=1,
        sport=REPORT_SPORT,
        coverage={"overall": coverage, "skipped_reason_counts": dict(sorted(skipped.items()))},
        priority_habit_id=priority_habit_id,
        replay_ref=None,
        habits=habits,
    )


def iter_metric_facts(metrics: RacketSportMetrics) -> list[MetricFact]:
    facts: list[MetricFact] = []
    for player in metrics.players:
        for shot_index, shot in enumerate(player.shots):
            for metric_name in sorted(shot.metrics):
                metric_value = shot.metrics[metric_name]
                facts.append(
                    MetricFact(
                        player_id=player.id,
                        shot_index=shot_index,
                        shot_time_s=float(shot.t),
                        shot_type=shot.type,
                        metric=metric_name,
                        value=metric_value.value,
                        confidence=float(metric_value.conf),
                        gated=metric_value.gated is True,
                    )
                )
    return facts


def habit_from_metric_fact(fact: MetricFact) -> Habit:
    metric = _slug(fact.metric)
    return Habit(
        id=fact.habit_id,
        title=f"P{fact.player_id} {fact.shot_type} {metric}",
        summary=f"Measured {fact.metric}={_format_value(fact.value)} at t={fact.shot_time_s:.3f}s.",
        confidence=fact.confidence,
        clip_ref={"t0_sec": max(0.0, fact.shot_time_s - 0.25), "t1_sec": fact.shot_time_s + REPORT_CLIP_PADDING_SEC},
        cue=f"Placeholder only: inspect measured {fact.metric} before coaching copy.",
        drill={"name": f"Metric review: {fact.metric}", "duration_min": REPORT_PLACEHOLDER_DRILL_MINUTES},
        source={
            "player_id": fact.player_id,
            "shot_index": fact.shot_index,
            "shot_type": fact.shot_type,
            "shot_time_s": fact.shot_time_s,
            "metric": fact.metric,
            "value": fact.value,
        },
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
