import { describe, expect, it, vi } from "vitest";

import { createPlaybackClock, createThrottledTimePublisher } from "./playbackClock";

describe("playback clock", () => {
  it("notifies high-frequency subscribers without owning React state", () => {
    const clock = createPlaybackClock(1);
    const listener = vi.fn();
    const unsubscribe = clock.subscribe(listener);

    clock.setTime(1.016);
    clock.setTime(1.032);
    unsubscribe();
    clock.setTime(1.048);

    expect(clock.getTime()).toBeCloseTo(1.048);
    expect(listener.mock.calls.map(([time]) => time)).toEqual([1.016, 1.032]);
  });

  it("limits React-facing publications to five per second", () => {
    const published: number[] = [];
    const publisher = createThrottledTimePublisher((time) => published.push(time), 200);

    expect(publisher.publish(0, 0)).toBe(true);
    for (let now = 16; now < 1000; now += 16) publisher.publish(now / 1000, now);

    expect(published).toEqual([0, 0.208, 0.416, 0.624, 0.832]);
    expect(published).toHaveLength(5);
  });
});
