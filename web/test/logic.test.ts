import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { SensitivityTable, VadGate, WakeWordDetector } from "../src/detector.js";
import { PcmOverrunError, PcmRingBuffer } from "../src/ring.js";
import { WakePhraseStripper } from "../src/stripper.js";
import { StreamingResampler } from "../src/resampler.js";
import { CachingTokenProvider, brokerTokenSource } from "../src/tokens.js";
import { buildListenUrl, defaultStreamingOptions } from "../src/deepgram.js";
import { REPO, json } from "./helpers.js";

describe("detector golden cases (shared with C# and Python)", () => {
  const golden = json(join(REPO, "testdata", "golden", "detector_cases.json"));
  for (const c of golden.cases) {
    it(c.name, () => {
      const o = c.options;
      const d = new WakeWordDetector({
        baseThreshold: o.base_threshold,
        consecutiveFrames: o.consecutive_frames,
        refractorySeconds: o.refractory_seconds,
        adaptive: {
          points: o.adaptive.points.map(([db, off]: number[]) => ({ noiseFloorDbfs: db!, offset: off! })),
          minThreshold: o.adaptive.min_threshold,
          maxThreshold: o.adaptive.max_threshold,
        },
      }, golden.sample_rate);
      const fires: number[] = [];
      c.scores.forEach((s: number, i: number) => {
        const floor = Array.isArray(c.noise_floor_dbfs) ? c.noise_floor_dbfs[i] : c.noise_floor_dbfs;
        if (d.process(s, floor, (i + 1) * golden.frame_samples)) fires.push(i);
      });
      expect(fires).toEqual(c.expected_fire_frames);
    });
  }

  it("sensitivity interpolates like the training pipeline", () => {
    const t = new SensitivityTable([[0, 0.9], [0.5, 0.6], [1, 0.2]]);
    expect(t.thresholdFor(-1)).toBe(0.9);
    expect(t.thresholdFor(0.25)).toBeCloseTo(0.75);
    expect(t.thresholdFor(2)).toBe(0.2);
  });

  it("VAD gate opens on onset and closes after the hangover", () => {
    const g = new VadGate({ onsetProbability: 0.5, sustainProbability: 0.3, hangoverSeconds: 0.2 }, 1000);
    expect([g.update(0.4, 100), g.update(0.6, 200), g.update(0.35, 300), g.update(0.1, 500), g.update(0.1, 501)])
      .toEqual(["none", "opened", "none", "none", "closed"]);
  });
});

describe("WakePhraseStripper (same cases as C#)", () => {
  const s = new WakePhraseStripper(["Hey UNO", "hey you know", "hey you no", "hey juno"]);
  it.each([
    ["Hey UNO, what's the weather?", "what's the weather?"],
    ["hey you know what's the weather", "what's the weather"],
    ["Hey Juno. Set a timer for ten minutes.", "Set a timer for ten minutes."],
    ["Um, hey UNO play some music", "play some music"],
    ["Hey, Uno.", ""],
    ["Hey you know", ""],
    ["What's the weather", "What's the weather"],
    ["", ""],
  ])("%s", (input, expected) => expect(s.strip(input)).toBe(expected));
});

describe("PcmRingBuffer", () => {
  it("reads across the wrap and refuses overwritten audio", () => {
    const r = new PcmRingBuffer(10, 1);
    r.write(Int16Array.from({ length: 14 }, (_, i) => i));
    const dest = new Int16Array(10);
    expect(r.read(4, dest)).toBe(10);
    expect(Array.from(dest)).toEqual([4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    expect(r.read(14, dest)).toBe(0);
    expect(() => r.read(3, dest)).toThrow(PcmOverrunError);
  });
});

describe("StreamingResampler", () => {
  for (const from of [44100, 48000]) {
    it(`${from} Hz -> 16 kHz keeps a 1 kHz tone and removes a 9 kHz one`, () => {
      const r = new StreamingResampler(from, 16000);
      const tone = (f: number) => Float32Array.from({ length: from }, (_, i) => Math.sin((2 * Math.PI * f * i) / from) * 0.5);
      const chunks = [];
      const input = tone(1000);
      for (let i = 0; i < input.length; i += 128) chunks.push(r.push(input.subarray(i, i + 128))); // worklet-sized pushes
      const out = Float32Array.from(chunks.flatMap((c) => Array.from(c)));
      expect(Math.abs(out.length - 16000)).toBeLessThan(200);
      let err = 0;
      const delay = 0; // linear-phase kernel centred on the output time: no delay
      for (let i = 2000; i < 14000; i++) err = Math.max(err, Math.abs(out[i]! - Math.sin((2 * Math.PI * 1000 * (i - delay)) / 16000) * 0.5));
      expect(err).toBeLessThan(0.01);

      const alias = new StreamingResampler(from, 16000).push(tone(9000));
      const rms = Math.sqrt(alias.slice(2000, 12000).reduce((s, v) => s + v * v, 0) / 10000);
      expect(rms).toBeLessThan(0.005); // 9 kHz is above the 8 kHz Nyquist and must not fold back
    });
  }
});

describe("tokens", () => {
  it("reuses a token until 80% of its life, sharing one fetch", async () => {
    let now = 0;
    let fetches = 0;
    const p = new CachingTokenProvider(async () => ({ accessToken: `t${++fetches}`, expiresAt: now + 300_000 }), () => now);
    p.prefetch();
    expect((await Promise.all([p.getToken(), p.getToken()])).map((t) => t.accessToken)).toEqual(["t1", "t1"]);
    now = 239_000;
    expect((await p.getToken()).accessToken).toBe("t1");
    now = 240_000;
    expect((await p.getToken()).accessToken).toBe("t2");
  });

  it("brokerTokenSource posts with the app's auth and converts expiry", async () => {
    let seen: RequestInit | undefined;
    const fakeFetch = (async (_url: string, init?: RequestInit) => {
      seen = init;
      return new Response(JSON.stringify({ accessToken: "jwt", expiresIn: 300 }), { status: 200 });
    }) as typeof fetch;
    const t = await brokerTokenSource("/v1/deepgram/token", () => ({ Authorization: "Bearer id" }), fakeFetch, () => 1000)();
    expect(t).toEqual({ accessToken: "jwt", expiresAt: 301_000 });
    expect(seen?.method).toBe("POST");
    expect(seen?.headers).toEqual({ Authorization: "Bearer id" });
  });

  it("builds the listen URL with cost and privacy settings", () => {
    const url = buildListenUrl({ ...defaultStreamingOptions(), keyterms: ["UNO"] });
    expect(url).toContain("model=nova-3");
    expect(url).toContain("encoding=linear16&sample_rate=16000&channels=1");
    expect(url).toContain("mip_opt_out=true");
    expect(url).toContain("keyterm=UNO");
  });
});
