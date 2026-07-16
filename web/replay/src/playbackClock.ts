export type PlaybackClockListener = (timeSeconds: number) => void;

export type PlaybackClock = {
  getTime: () => number;
  setTime: (timeSeconds: number) => void;
  subscribe: (listener: PlaybackClockListener) => () => void;
};

/**
 * Mutable video-PTS clock. High-frequency consumers (the 3D camera and the
 * timeline playhead) subscribe without forcing the top-level App to render.
 */
export function createPlaybackClock(initialTime = 0): PlaybackClock {
  let time = finiteTime(initialTime);
  const listeners = new Set<PlaybackClockListener>();
  return {
    getTime: () => time,
    setTime: (nextTime) => {
      const next = finiteTime(nextTime);
      if (Math.abs(next - time) < 1e-6) return;
      time = next;
      for (const listener of listeners) listener(time);
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export type ThrottledTimePublisher = {
  publish: (timeSeconds: number, nowMs: number) => boolean;
  reset: () => void;
};

/** Limits React-facing time updates while the canonical mutable clock runs at rAF speed. */
export function createThrottledTimePublisher(
  onPublish: PlaybackClockListener,
  intervalMs = 200,
): ThrottledTimePublisher {
  let lastPublishedAtMs = Number.NEGATIVE_INFINITY;
  return {
    publish: (timeSeconds, nowMs) => {
      if (nowMs - lastPublishedAtMs < Math.max(0, intervalMs)) return false;
      lastPublishedAtMs = nowMs;
      onPublish(finiteTime(timeSeconds));
      return true;
    },
    reset: () => {
      lastPublishedAtMs = Number.NEGATIVE_INFINITY;
    },
  };
}

function finiteTime(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}
