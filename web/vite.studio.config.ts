import { defineConfig } from "vite";

// Wake Word Studio page. In development the API runs separately: python -m studio.app (port 8765).
const api = "http://127.0.0.1:8765";

export default defineConfig({
  root: "studio",
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: { port: 5174, proxy: { "/api": api, "/vad": api } },
  build: { outDir: "../dist-studio", emptyOutDir: true },
});
