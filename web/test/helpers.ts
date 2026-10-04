import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { configureOnnxRuntime } from "../src/onnx.js";
import { ModelPackage } from "../src/package.js";

configureOnnxRuntime();

export const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
export const SMOKE = join(REPO, "testdata", "models", "smoke");

export const loadSmoke = () => ModelPackage.load(async (name) => new Uint8Array(readFileSync(join(SMOKE, name))));

export const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));

export function readWav(path: string): Int16Array {
  const bytes = readFileSync(path);
  let pos = 12;
  while (bytes.toString("ascii", pos, pos + 4) !== "data") pos += 8 + bytes.readInt32LE(pos + 4);
  const length = bytes.readInt32LE(pos + 4);
  return new Int16Array(bytes.buffer.slice(bytes.byteOffset + pos + 8, bytes.byteOffset + pos + 8 + length));
}
