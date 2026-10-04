import { SILENCE_DBFS, WakeWordDetector } from "./detector.js";
import { FRAME_LENGTH, OnnxWakeWordModel } from "./model.js";
import type { ModelPackage } from "./package.js";

/**
 * Porcupine-style API: choose keywords and sensitivities, feed 1280-sample frames, get back the index of
 * the keyword heard or -1.
 *
 *   const spotter = await KeywordSpotter.create(pkg, ["hey_uno", "hello_uno"], [0.5, 0.5]);
 *   const keyword = await spotter.process(frame);
 *
 * Runs every frame through the model; for VAD gating, pre-roll and Deepgram use WakeWordListener.
 */
export class KeywordSpotter {
  readonly sampleRate = 16000;
  readonly frameLength = FRAME_LENGTH;
  readonly lastScores: Float64Array;
  private samples = 0;

  private constructor(private readonly model: OnnxWakeWordModel, private readonly detectors: WakeWordDetector[]) {
    this.lastScores = new Float64Array(detectors.length);
  }

  static async create(pkg: ModelPackage, keywords?: string[], sensitivities?: number[]): Promise<KeywordSpotter> {
    const chosen = pkg.keywordOptions(keywords, sensitivities);
    const model = await OnnxWakeWordModel.create(pkg, chosen.map((k) => k.id));
    return new KeywordSpotter(model, chosen.map((k) => new WakeWordDetector(k.detector)));
  }

  get keywords(): string[] {
    return this.model.keywords;
  }

  /** Index into keywords of the keyword that fired on this frame, or -1. Highest score wins a tie. */
  async process(frame: Int16Array): Promise<number> {
    await this.model.appendFrame(frame);
    this.samples += frame.length;
    if (!(await this.model.score(this.lastScores))) return -1;
    let fired = -1;
    this.detectors.forEach((d, k) => {
      if (d.process(this.lastScores[k]!, SILENCE_DBFS, this.samples) && (fired < 0 || this.lastScores[k]! > this.lastScores[fired]!)) fired = k;
    });
    return fired;
  }

  dispose(): Promise<void> {
    return this.model.dispose();
  }
}
