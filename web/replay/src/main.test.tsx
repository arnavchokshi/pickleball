import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";
import { transformWithEsbuild } from "vite";

describe("main entrypoint", () => {
  it("keeps React bound for the JSX emitted by the local Vite transform", async () => {
    const source = readFileSync(resolve(__dirname, "main.tsx"), "utf8");
    const result = await transformWithEsbuild(source, "main.tsx", { loader: "tsx" });

    expect(result.code).toContain('import React from "react"');
    expect(result.code).toContain("React.createElement(App");
  });
});
