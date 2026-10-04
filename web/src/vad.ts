import { createSession, ort } from "./onnx.js";

const WINDOW = 512;
const CONTEXT = 64;

export interface VoiceActivityDetector {
  /** Speech probability in [0, 1] for one frame of 16 kHz PCM. */
  speechProbability(frame: Int16Array): Promise<number>;
  dispose(): Promise<void>;
}

/**
 * Silero VAD (MIT) at 16 kHz, same behaviour as WakeWord.Onnx.SileroVad: 512-sample windows with the
 * previous 64 samples as context and a recurrent state. A frame's probability is the highest of the
 * windows completed during it.
 */
export class SileroVad implements VoiceActivityDetector {
  private readonly input = new Float32Array(CONTEXT + WINDOW);
  private readonly pending = new Float32Array(WINDOW);
  private state = new Float32Array(2 * 128);
  private pendingCount = 0;
  private last = 0;
  private readonly sr = new ort.Tensor("int64", BigInt64Array.from([16000n]), []);

  private constructor(private readonly session: ort.InferenceSession) {}

  static async create(model: Uint8Array): Promise<SileroVad> {
    return new SileroVad(await createSession(model));
  }

  static async fromUrl(url: string, fetchImpl: typeof fetch = fetch): Promise<SileroVad> {
    const res = await fetchImpl(url);
    if (!res.ok) throw new Error(`Could not load ${url}: HTTP ${res.status}`);
    return SileroVad.create(new Uint8Array(await res.arrayBuffer()));
  }

  async speechProbability(frame: Int16Array): Promise<number> {
    let max = -1;
    for (const sample of frame) {
      this.pending[this.pendingCount++] = sample / 32768;
      if (this.pendingCount === WINDOW) {
        max = Math.max(max, await this.runWindow());
        this.pendingCount = 0;
      }
    }
    this.last = max >= 0 ? max : this.last;
    return this.last;
  }

  reset(): void {
    this.input.fill(0);
    this.state = new Float32Array(2 * 128);
    this.pendingCount = 0;
    this.last = 0;
  }

  private async runWindow(): Promise<number> {
    this.input.set(this.pending, CONTEXT);
    const out = await this.session.run({
      input: new ort.Tensor("float32", this.input.slice(), [1, CONTEXT + WINDOW]),
      state: new ort.Tensor("float32", this.state, [2, 1, 128]),
      sr: this.sr,
    });
    this.state = new Float32Array(out.stateN!.data as Float32Array);
    this.input.copyWithin(0, WINDOW, WINDOW + CONTEXT); // last 64 samples become the next context
    return (out.output!.data as Float32Array)[0]!;
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}
