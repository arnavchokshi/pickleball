import { realpathSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

const apiTarget = process.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";
const repoRoot = resolve(__dirname, "../..");
const repoRootRealpath = realpathSync.native(repoRoot);
const repoRootTmpAlias = repoRootRealpath.startsWith("/private/tmp/")
  ? `/tmp/${repoRootRealpath.slice("/private/tmp/".length)}`
  : null;
const devFsAllow = Array.from(new Set([repoRoot, repoRootRealpath, repoRootTmpAlias].filter(Boolean) as string[]));

export default defineConfig(({ command }) => ({
  // Replay bundles are manifest-addressed, never production public assets.
  // Keep local public/ aliases available to the dev server without copying
  // large or VM-written symlink trees into production builds.
  publicDir: command === "build" ? false : "public",
  server: {
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
    fs: {
      // Dev-only: macOS /tmp worktrees resolve through /private/tmp, while replay
      // manifests may still use /tmp absolute paths in /@fs/ URLs.
      allow: devFsAllow,
    },
  },
}));
