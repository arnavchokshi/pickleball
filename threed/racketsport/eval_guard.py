"""Central registry and enforcement for the protected held-out eval clips.

This module is the single code-level source of truth for which clips are
eval-only and how they may (or may not) be touched by anything that fits a
model. It exists because prior sessions repeatedly, and independently,
re-derived this policy from prose in ``BUILD_CHECKLIST.md`` /
``runs/manager/outdoor_eval_ledger.md`` and then still leaked eval clips into
training or checkpoint-selection inputs (see the ledger's "seeded history"
rows for the concrete incidents this module is meant to make structurally
impossible going forward).

Policy (binding, enforced here, not just documented):

* ``outdoor_webcam_iynbd_1500_long_high_baseline`` and
  ``indoor_doubles_fwuks_0500_long_mid_baseline`` are **strict holdout**
  clips. They may never appear in training inputs *or* in a
  validation-during-fitting split (checkpoint selection, early stopping,
  threshold sweeps run as part of a training loop, etc). There is no
  override for these two clips -- that is the entire point of holding them
  out.
* ``burlington_gold_0300_low_steep_corner`` and
  ``wolverine_mixed_0200_mid_steep_corner`` are **internal-val-only** clips.
  They must never be used as actual training data, but a caller may use them
  as a validation-during-fitting signal if it explicitly passes
  ``allow_internal_val=True`` to :func:`assert_not_training_on_eval_clip`.
  That flag is a deliberate, auditable opt-in: the guard logs every such use
  in the summary dict it returns so a reviewer can see exactly when and
  where a training entry point relied on it.

Callers typically invoke this guard twice per training entry point: once
for the rows/paths that actually feed gradient updates
(``allow_internal_val=False`` -- refuses all four protected clips
unconditionally), and once for the rows/paths used only to compute a
validation-during-fitting metric (``allow_internal_val=True`` -- permits
Burlington/Wolverine, still refuses Outdoor/Indoor).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

ClipRole = Literal["strict_holdout", "internal_val_only"]


@dataclass(frozen=True)
class ProtectedEvalClip:
    clip_id: str
    role: ClipRole
    description: str


PROTECTED_EVAL_CLIPS: tuple[ProtectedEvalClip, ...] = (
    ProtectedEvalClip(
        clip_id="outdoor_webcam_iynbd_1500_long_high_baseline",
        role="strict_holdout",
        description=(
            "Outdoor held-out eval clip. Never a training or validation-during-fitting "
            "input; no override exists. It is the one honest held-out gate signal for BALL "
            "and the primary strict clip for CAL/TRK/BODY/RKT."
        ),
    ),
    ProtectedEvalClip(
        clip_id="indoor_doubles_fwuks_0500_long_mid_baseline",
        role="strict_holdout",
        description=(
            "Indoor held-out eval clip (CVAT labels landing). Never a training or "
            "validation-during-fitting input; no override exists."
        ),
    ),
    ProtectedEvalClip(
        clip_id="burlington_gold_0300_low_steep_corner",
        role="internal_val_only",
        description=(
            "Burlington train/internal-val clip. May be used as a validation-during-"
            "fitting signal only when the caller explicitly passes allow_internal_val=True; "
            "never usable as actual training data."
        ),
    ),
    ProtectedEvalClip(
        clip_id="wolverine_mixed_0200_mid_steep_corner",
        role="internal_val_only",
        description=(
            "Wolverine train/internal-val clip. May be used as a validation-during-"
            "fitting signal only when the caller explicitly passes allow_internal_val=True; "
            "never usable as actual training data."
        ),
    ),
)

PROTECTED_EVAL_CLIP_IDS: tuple[str, ...] = tuple(clip.clip_id for clip in PROTECTED_EVAL_CLIPS)
STRICT_HOLDOUT_CLIP_IDS: tuple[str, ...] = tuple(
    clip.clip_id for clip in PROTECTED_EVAL_CLIPS if clip.role == "strict_holdout"
)
INTERNAL_VAL_ONLY_CLIP_IDS: tuple[str, ...] = tuple(
    clip.clip_id for clip in PROTECTED_EVAL_CLIPS if clip.role == "internal_val_only"
)

_CLIPS_BY_ID: dict[str, ProtectedEvalClip] = {clip.clip_id: clip for clip in PROTECTED_EVAL_CLIPS}


class EvalClipLeakError(ValueError):
    """Raised when training/validation-during-fitting inputs reference a protected eval clip."""


def assert_not_training_on_eval_clip(
    paths_or_ids: Iterable[Any],
    *,
    allow_internal_val: bool = False,
    clip_ids: Sequence[str] = PROTECTED_EVAL_CLIP_IDS,
) -> dict[str, Any]:
    """Fail closed if ``paths_or_ids`` references a protected eval clip.

    ``paths_or_ids`` may contain file paths, bare clip ids/slugs, or nested
    manifest-like structures (dicts/lists that themselves contain clip ids or
    paths, e.g. a dataset manifest's ``splits`` or ``clip_counts`` mapping).
    Every string found anywhere inside ``paths_or_ids`` is checked as a
    substring match against each protected clip id -- this mirrors how these
    clips actually show up in the repo (e.g.
    ``cvat_upload/02_wolverine_mixed_0200_mid_steep_corner_10s.mp4``, where
    the clip id is embedded with extra prefix/suffix tokens rather than
    appearing as an isolated path segment).

    Raises :class:`EvalClipLeakError` naming the violating clip id and the
    exact matched value:

    * Immediately, regardless of ``allow_internal_val``, if a strict-holdout
      clip (Outdoor or Indoor) is matched.
    * If an internal-val-only clip (Burlington or Wolverine) is matched and
      ``allow_internal_val`` is not ``True``.

    On success, returns a summary dict recording what was checked and, when
    ``allow_internal_val=True`` legitimately permitted a match, which clip(s)
    and matched value(s) were used -- this is the audit log a caller should
    fold into its own training-run summary artifact.
    """

    try:
        registry = [_CLIPS_BY_ID[clip_id] for clip_id in clip_ids]
    except KeyError as exc:
        raise ValueError(f"unknown protected eval clip id passed to clip_ids: {exc}") from exc

    checked_item_count = 0
    internal_val_uses: list[dict[str, str]] = []
    for item in paths_or_ids:
        for text in _iter_strings(item):
            checked_item_count += 1
            lowered = text.lower()
            for clip in registry:
                if clip.clip_id not in lowered:
                    continue
                if clip.role == "strict_holdout":
                    raise EvalClipLeakError(
                        f"refusing to use strict held-out eval clip {clip.clip_id!r} as a training or "
                        f"validation-during-fitting input -- no override exists for this clip; "
                        f"violating value: {text!r}"
                    )
                if not allow_internal_val:
                    raise EvalClipLeakError(
                        f"refusing to use eval clip {clip.clip_id!r} without allow_internal_val=True -- "
                        f"it may only be used as an internal validation-during-fitting signal with that "
                        f"explicit flag, never as training data; violating value: {text!r}"
                    )
                internal_val_uses.append({"clip_id": clip.clip_id, "matched_value": text})

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_eval_clip_guard_check",
        "status": "internal_val_used" if internal_val_uses else "clean",
        "checked_item_count": checked_item_count,
        "allow_internal_val": allow_internal_val,
        "clip_ids_checked": list(clip_ids),
        "internal_val_uses": internal_val_uses,
    }


def _iter_strings(value: Any) -> Iterable[str]:
    """Recursively yield every string found in ``value``.

    Handles bare strings/Paths, and dict/list/tuple/set containers so callers
    can pass whole manifest fragments (e.g. a ``clip_counts`` mapping or a
    ``splits`` list of row dicts) without having to flatten them by hand.
    Non-string leaves (ints, floats, bools, ``None``) are skipped -- they
    cannot contain a clip slug.
    """

    if isinstance(value, (str, Path)):
        yield str(value)
        return
    if isinstance(value, Mapping):
        for key, val in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(val)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_strings(item)
        return
    # int/float/bool/None/other scalars: nothing to check.
    return


__all__ = [
    "ClipRole",
    "EvalClipLeakError",
    "INTERNAL_VAL_ONLY_CLIP_IDS",
    "PROTECTED_EVAL_CLIPS",
    "PROTECTED_EVAL_CLIP_IDS",
    "ProtectedEvalClip",
    "STRICT_HOLDOUT_CLIP_IDS",
    "assert_not_training_on_eval_clip",
]
