import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  server: {
    fs: {
      allow: [resolve(__dirname, "../..")],
    },
  },
});
