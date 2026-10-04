import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { defineConfig, type Plugin } from "vitest/config";

const repo = resolve(import.meta.dirname, "..");

/** Serves the repo's model folders to the demo: /models/... and /smoke/... (the test package). */
function serveModels(): Plugin {
  const roots: Record<string, string> = { "/models/": join(repo, "models"), "/smoke/": join(repo, "testdata", "models", "smoke") };
  const types: Record<string, string> = { ".json": "application/json", ".onnx": "application/octet-stream" };
  return {
    name: "serve-models",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url ?? "").split("?")[0]!);
        const prefix = Object.keys(roots).find((p) => url.startsWith(p));
        if (!prefix) return next();
        const file = normalize(join(roots[prefix]!, url.slice(prefix.length)));
        if (!file.startsWith(roots[prefix]!) || !existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          return res.end();
        }
        res.setHeader("Content-Type", types[extname(file)] ?? "application/octet-stream");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  root: "demo",
  plugins: [serveModels()],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { outDir: "../dist-demo", emptyOutDir: true },
  test: {
    root: ".",
    include: ["test/**/*.test.ts"],
    environment: "node",
    testTimeout: 60000,
  },
});
