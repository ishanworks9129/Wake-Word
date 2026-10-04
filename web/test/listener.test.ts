import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { WakeWordListener, type SessionTelemetry } from "../src/listener.js";
import { OnnxWakeWordModel } from "../src/model.js";
import { SileroVad } from "../src/vad.js";
import { KeywordSpotter } from "../src/spotter.js";
import { WakeWordDetector, defaultDetectorOptions } from "../src/detector.js";
import type { KeywordOptions } from "../src/package.js";
import { FakeTransport, ScriptedModel, ScriptedVad, StaticTokens } from "./fakes.js";
import { REPO, SMOKE, json, loadSmoke, readWav } from "./helpers.js";

const kw = (id: string): KeywordOptions => ({ id, phrase: "Hey UNO", detector: defaultDetectorOptions(), transcriptVariants: [] });
const silence = new Int16Array(1280);
const speech = readWav(join(REPO, "testdata", "audio", "hey_uno_tts.wav"));
const loadVad = () => SileroVad.create(new Uint8Array(readFileSync(join(REPO, "models", "vad", "silero_vad.onnx"))));

describe("WakeWordListener", () => {
  it("starts one session with pre-roll, rejects triggers while busy, and reports telemetry", async () => {
    const transports: FakeTransport[] = [];
    const tokens = new StaticTokens();
    const listener = new WakeWordListener(new ScriptedModel([0.9]), new ScriptedVad([1]), {
      keywords: [kw("hey_uno")],
      tokens,
      transportFactory: () => {
        const t = new FakeTransport();
        transports.push(t);
        return t;
      },
      limits: { noTranscriptCutoffMs: 60_000, hardTimeoutMs: 60_000 },
    });
    const rejections: string[] = [];
    const ended = new Promise<SessionTelemetry>((r) => { listener.onsessionend = r; });
    listener.onreject = (r) => rejections.push(r);

    for (let i = 0; i < 38; i++) await listener.enqueue(silence); // fires at frame 2, then again after the 1.5 s refractory
    await new Promise((r) => setTimeout(r, 20));
    await listener.enqueue(silence); // one more frame pumps everything to the socket

    expect(tokens.prefetches).toBe(1);
    expect(listener.isStreaming).toBe(true);
    expect(rejections).toContain("busy");
    expect(transports).toHaveLength(1);
    expect(transports[0]!.audio.length).toBe(1280 * 39); // first fire at 3,840 samples: the 2 s pre-roll clamps to 0

    listener.stopSession();
    const t = await ended;
    expect(t.result.reason).toBe("cancelled");
    expect([t.trigger, t.keyword, t.score]).toEqual(["wake-word", "hey_uno", 0.9]);
  });

  it("rejects keyword options that don't match the model", () => {
    const model = new ScriptedModel([0], ["hey_uno", "hello_uno"]);
    expect(() => new WakeWordListener(model, new ScriptedVad([0]), { keywords: [kw("hello_uno"), kw("hey_uno")] })).toThrow(/must match/);
  });

  it("runs the real model behind the real Silero VAD", async () => {
    const pkg = await loadSmoke();
    const tokens = new StaticTokens();
    const listener = new WakeWordListener(await OnnxWakeWordModel.create(pkg), await loadVad(), { keywords: pkg.keywordOptions(), tokens });
    const audio = new Int16Array(16000 * 3 + speech.length);
    audio.set(speech, 16000 * 3);
    for (let i = 0; i + 1280 <= audio.length; i += 1280) await listener.enqueue(audio.slice(i, i + 1280));

    expect(listener.keywords).toEqual(["hey_uno", "hello_uno"]);
    expect(listener.framesScored).toBeGreaterThan(0);
    expect(listener.framesScored).toBeLessThan(listener.framesProcessed - 30); // the 3 s of silence was skipped
    expect(tokens.prefetches).toBe(1);
    await listener.dispose();
  });
});

describe("SileroVad", () => {
  it("hears speech but not silence or white noise", async () => {
    const vad = await loadVad();
    const p: number[] = [];
    for (let i = 0; i + 1280 <= speech.length; i += 1280) p.push(await vad.speechProbability(speech.subarray(i, i + 1280)));
    expect(Math.max(...p.slice(0, 10))).toBeLessThan(0.2);
    expect(Math.max(...p)).toBeGreaterThan(0.6);

    vad.reset();
    let noiseMax = 0;
    let seed = 1;
    for (let f = 0; f < 12; f++) {
      const noise = Int16Array.from({ length: 1280 }, () => ((seed = (seed * 16807) % 2147483647) % 600) - 300);
      noiseMax = Math.max(noiseMax, await vad.speechProbability(noise));
    }
    expect(noiseMax).toBeLessThan(0.3);
  });
});

describe("KeywordSpotter", () => {
  it("fires exactly where the shared detector rules say", async () => {
    const golden = json(join(SMOKE, "golden.json"));
    const audio = Int16Array.from(golden.audio_int16 as number[]);
    const chunk = (i: number) => audio.subarray(i * 1280, (i + 1) * 1280);
    const pkg = await loadSmoke();
    const model = await OnnxWakeWordModel.create(pkg, ["hey_uno"]);
    const scores: number[] = [];
    const s = new Float64Array(1);
    for (let i = 0; i < audio.length / 1280; i++) {
      await model.appendFrame(chunk(i));
      scores.push((await model.score(s)) ? s[0]! : 0);
    }
    const threshold = Math.max(...scores) * 0.5;
    const reference = new WakeWordDetector({ baseThreshold: threshold, consecutiveFrames: 2, refractorySeconds: 1.5, adaptive: { points: [], minThreshold: 0, maxThreshold: 1 } });
    const expected = scores.map((v, i) => reference.process(v, -100, (i + 1) * 1280) !== null);

    // Pin the calibrated threshold so the spotter applies the same rule.
    const k = pkg.keyword("hey_uno");
    k.sensitivity_table = [[0.5, threshold]];
    k.detector = { consecutive_frames: 2, refractory_seconds: 1.5, min_threshold: 0, max_threshold: 1 };
    const spotter = await KeywordSpotter.create(pkg, ["hey_uno"]);
    const actual: boolean[] = [];
    for (let i = 0; i < audio.length / 1280; i++) actual.push((await spotter.process(chunk(i))) === 0);

    expect(actual).toEqual(expected);
    expect(actual).toContain(true);
    expect([spotter.sampleRate, spotter.frameLength]).toEqual([16000, 1280]);
  });
});
