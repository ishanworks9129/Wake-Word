import { describe, expect, it } from "vitest";
import { DeepgramSession, defaultSessionLimits, defaultStreamingOptions, isErrorReason, type SessionLimits } from "../src/deepgram.js";
import { PcmRingBuffer } from "../src/ring.js";
import { WakePhraseStripper } from "../src/stripper.js";
import { FakeTransport, StaticTokens, ramp } from "./fakes.js";

const RATE = 16000;
const stripper = new WakePhraseStripper(["Hey UNO", "hey you know"]);

async function until(cond: () => boolean) {
  for (let i = 0; i < 500 && !cond(); i++) await new Promise((r) => setTimeout(r, 5));
}

function session(ring: PcmRingBuffer, t: FakeTransport, limits: Partial<SessionLimits> = {}, tokens = new StaticTokens()) {
  return new DeepgramSession(ring, tokens, () => t, defaultStreamingOptions(), { ...defaultSessionLimits(), ...limits }, stripper);
}

describe("DeepgramSession", () => {
  it("streams pre-roll and live audio with no gap, even while the socket opens", async () => {
    const ring = new PcmRingBuffer(RATE, 10);
    ring.write(ramp(0, 3 * RATE));
    const t = new FakeTransport(() => ring.write(ramp(3 * RATE, RATE / 2)));
    const s = session(ring, t);
    const run = s.run(RATE); // trigger at 3 s, pre-roll from 1 s

    await until(() => t.audio.length > 0);
    ring.write(ramp(3.5 * RATE, RATE / 2));
    s.pump();
    t.push(FakeTransport.results("Hey UNO, what's the weather?", true));
    t.push({ type: "UtteranceEnd" });
    const r = await run;

    expect(r.reason).toBe("utterance-end");
    expect(r.transcript).toBe("what's the weather?");
    expect(t.audio).toEqual(Array.from(ramp(RATE, 3 * RATE)));
    expect(r.audioSentSeconds).toBe(3);
    expect(t.text.some((m) => m.includes("CloseStream"))).toBe(true);
    expect(t.token).toBe("test-token");
    expect(t.url).toContain("mip_opt_out=true");
  });

  it("closes quickly when only the wake phrase is heard", async () => {
    const t = new FakeTransport();
    const run = session(new PcmRingBuffer(RATE, 10), t, { noTranscriptCutoffMs: 200 }).run(0);
    await until(() => t.connected);
    t.push(FakeTransport.results("Hey you know", true));
    const r = await run;
    expect(r.reason).toBe("no-transcript");
    expect(isErrorReason(r.reason)).toBe(false);
  });

  it("interim speech defers the cutoff but not the hard timeout", async () => {
    const t = new FakeTransport();
    const run = session(new PcmRingBuffer(RATE, 10), t, { noTranscriptCutoffMs: 100, hardTimeoutMs: 400 }).run(0);
    await until(() => t.connected);
    t.push(FakeTransport.results("Hey UNO tell me a", false));
    expect((await run).reason).toBe("hard-timeout");
  });

  it("reports connect failures and broker errors as visible errors", async () => {
    const ring = new PcmRingBuffer(RATE, 10);
    expect((await session(ring, new FakeTransport(undefined, true)).run(0)).reason).toBe("connect-failed");
    const r = await session(ring, new FakeTransport(), {}, new StaticTokens(new Error("broker down"))).run(0);
    expect(r.reason).toBe("connect-failed");
    expect(isErrorReason(r.reason)).toBe(true);
  });

  it("ends with an overrun rather than sending audio with a hole", async () => {
    const ring = new PcmRingBuffer(RATE, 1);
    ring.write(ramp(0, RATE));
    const t = new FakeTransport();
    t.bufferedAmount = 10_000_000; // network stalled: nothing more can be queued
    const s = session(ring, t);
    const run = s.run(0);
    await until(() => t.connected);
    ring.write(ramp(RATE, 2 * RATE)); // 2 s more; the ring only holds 1 s
    s.pump();
    expect((await run).reason).toBe("audio-overrun");
  });

  it("ends on server close and on cancel", async () => {
    const ring = new PcmRingBuffer(RATE, 10);
    const closed = new FakeTransport();
    const a = session(ring, closed).run(0);
    await until(() => closed.connected);
    await new Promise((r) => setTimeout(r, 5));
    closed.onclose?.();
    expect((await a).reason).toBe("server-closed");

    const t = new FakeTransport();
    const s = session(ring, t);
    const b = s.run(0);
    await until(() => t.connected);
    s.cancel();
    expect((await b).reason).toBe("cancelled");
  });
});
