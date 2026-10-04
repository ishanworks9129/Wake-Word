import type { DeepgramTransport } from "../src/deepgram.js";
import type { WakeWordModel } from "../src/model.js";
import type { TokenProvider } from "../src/tokens.js";
import type { VoiceActivityDetector } from "../src/vad.js";

export class FakeTransport implements DeepgramTransport {
  audio: number[] = [];
  text: string[] = [];
  url = "";
  token = "";
  bufferedAmount = 0;
  connected = false;
  onmessage: ((text: string) => void) | null = null;
  onclose: ((error?: unknown) => void) | null = null;

  constructor(private readonly duringConnect?: () => void, private readonly fail = false) {}

  async connect(url: string, token: string): Promise<void> {
    this.url = url;
    this.token = token;
    this.duringConnect?.();
    await Promise.resolve();
    if (this.fail) throw new Error("connection refused");
    this.connected = true;
  }

  sendAudio(pcm: Int16Array): void {
    for (const s of pcm) this.audio.push(s);
  }

  sendText(message: string): void {
    this.text.push(message);
    if (message.includes("CloseStream")) queueMicrotask(() => this.onclose?.()); // Deepgram flushes and closes
  }

  close(): void {}

  push(message: object): void {
    this.onmessage?.(JSON.stringify(message));
  }

  static results(transcript: string, isFinal: boolean) {
    return { type: "Results", channel: { alternatives: [{ transcript }] }, is_final: isFinal };
  }
}

export class StaticTokens implements TokenProvider {
  prefetches = 0;

  constructor(private readonly error?: Error) {}

  getToken() {
    return this.error ? Promise.reject(this.error) : Promise.resolve({ accessToken: "test-token", expiresAt: Date.now() + 300_000 });
  }

  prefetch() {
    this.prefetches++;
  }
}

/** Score for frame i is scores[i] (last repeats), for every keyword. */
export class ScriptedModel implements WakeWordModel {
  appended = 0;
  scored = 0;

  constructor(private readonly scores: number[], readonly keywords = ["hey_uno"]) {}

  async appendFrame() {
    this.appended++;
  }

  async score(out: Float64Array) {
    this.scored++;
    out.fill(this.scores[Math.min(this.appended - 1, this.scores.length - 1)]!);
    return true;
  }

  async dispose() {}
}

export class ScriptedVad implements VoiceActivityDetector {
  private i = 0;

  constructor(private readonly p: number[]) {}

  async speechProbability() {
    return this.p[Math.min(this.i++, this.p.length - 1)]!;
  }

  async dispose() {}
}

export const ramp = (start: number, count: number) => Int16Array.from({ length: count }, (_, i) => (start + i) % 32767);
