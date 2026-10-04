import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { OnnxWakeWordModel } from "../src/model.js";
import { SMOKE, json, loadSmoke } from "./helpers.js";

const golden = json(join(SMOKE, "golden.json"));
const audio = Int16Array.from(golden.audio_int16 as number[]);
const chunk = (i: number) => audio.subarray(i * 1280, (i + 1) * 1280);

describe("OnnxWakeWordModel", () => {
  it("matches the Python streaming reference chunk by chunk", async () => {
    const model = await OnnxWakeWordModel.create(await loadSmoke());
    const scores = new Float64Array(model.keywords.length);
    let compared = 0;
    for (const c of golden.chunks) {
      await model.appendFrame(chunk(c.chunk));
      const scored = await model.score(scores);
      expect(scored).toBe("scores" in c);
      if (c.embedding_first8) {
        const actual = Array.from(model.latestEmbedding.subarray(0, 8));
        c.embedding_first8.forEach((v: number, d: number) => expect(Math.abs(v - actual[d]!)).toBeLessThan(2e-3));
      }
      if (c.scores) {
        model.keywords.forEach((k, i) => expect(Math.abs(c.scores[k] - scores[i]!)).toBeLessThan(1e-3));
        compared += model.keywords.length;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(40);
    await model.dispose();
  });

  it("back-fills after a closed VAD gate and matches continuous scoring", async () => {
    const pkg = await loadSmoke();
    const continuous = await OnnxWakeWordModel.create(pkg);
    const gated = await OnnxWakeWordModel.create(pkg);
    const a = new Float64Array(2);
    const b = new Float64Array(2);
    for (let i = 0; i < audio.length / 1280; i++) {
      await continuous.appendFrame(chunk(i));
      const expected = await continuous.score(a);
      await gated.appendFrame(chunk(i));
      if (i < 35) continue;
      expect(await gated.score(b)).toBe(expected);
      expect(b[0]).toBeCloseTo(a[0]!, 5);
      expect(b[1]).toBeCloseTo(a[1]!, 5);
    }
  });

  it("rejects frames of the wrong size", async () => {
    const model = await OnnxWakeWordModel.create(await loadSmoke(), ["hey_uno"]);
    expect(model.keywords).toEqual(["hey_uno"]);
    await expect(model.appendFrame(new Int16Array(512))).rejects.toThrow(RangeError);
  });
});
