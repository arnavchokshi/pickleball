import { resolve } from "node:path";
import { defineConfig } from "vite";

const apiTarget = process.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
    fs: {
      allow: [resolve(__dirname, "../..")],
    },
  },
});
