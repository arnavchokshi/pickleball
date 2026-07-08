import { describe, expect, it } from "vitest";

import { planParts } from "./uploadPlan";

describe("planParts", () => {
  it("splits a size that does not land on a part boundary into a smaller final part", () => {
    const plan = planParts(25, 10);

    expect(plan.partCount).toBe(3);
    expect(plan.ranges).toEqual([
      { partNumber: 1, offset: 0, length: 10 },
      { partNumber: 2, offset: 10, length: 10 },
      { partNumber: 3, offset: 20, length: 5 },
    ]);
  });

  it("handles the size % partSize == 0 boundary with no trailing empty part", () => {
    const plan = planParts(20, 10);

    expect(plan.partCount).toBe(2);
    expect(plan.ranges).toEqual([
      { partNumber: 1, offset: 0, length: 10 },
      { partNumber: 2, offset: 10, length: 10 },
    ]);
    // No third, zero-length part dangling off the boundary.
    expect(plan.ranges.some((range) => range.length === 0)).toBe(false);
  });

  it("returns zero parts for a zero-byte file instead of throwing", () => {
    const plan = planParts(0, 10);

    expect(plan).toEqual({ partCount: 0, ranges: [] });
  });

  it("produces exactly one part when the file is smaller than the part size", () => {
    const plan = planParts(3, 10);

    expect(plan.partCount).toBe(1);
    expect(plan.ranges).toEqual([{ partNumber: 1, offset: 0, length: 3 }]);
  });

  it("rejects a non-positive part size", () => {
    expect(() => planParts(100, 0)).toThrow("partSizeBytes must be positive");
    expect(() => planParts(100, -5)).toThrow("partSizeBytes must be positive");
  });

  it("rejects a negative file size", () => {
    expect(() => planParts(-1, 10)).toThrow("sizeBytes must not be negative");
  });

  it("matches server ceil-division math for a large multi-part clip", () => {
    const partSize = 8 * 1024 * 1024;
    const size = 45 * 1024 * 1024 + 17;
    const plan = planParts(size, partSize);

    expect(plan.partCount).toBe(6);
    const lastRange = plan.ranges[plan.ranges.length - 1];
    expect(lastRange.offset + lastRange.length).toBe(size);
    const totalLength = plan.ranges.reduce((sum, range) => sum + range.length, 0);
    expect(totalLength).toBe(size);
  });
});
