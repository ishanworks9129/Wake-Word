// The WASM-only build: half the download of the default (WebGPU) build, and all these small models need.
import * as ort from "onnxruntime-web/wasm";

export { ort };

export interface OnnxRuntimeConfig {
  /**
   * Where onnxruntime-web's .wasm files are served from, e.g. "/ort/" or a CDN such as
   * "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/". Defaults to next to the bundle.
   */
  wasmPaths?: string;
}

/**
 * Single-threaded WASM with SIMD (plan 7.4): the models are small enough, and it needs no cross-origin
 * isolation (COOP/COEP) headers.
 */
export function configureOnnxRuntime(config: OnnxRuntimeConfig = {}): void {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  if (config.wasmPaths) ort.env.wasm.wasmPaths = config.wasmPaths;
}

export function createSession(model: Uint8Array): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
}
