import { createSession, ort } from "./onnx.js";
import type { ModelPackage } from "./package.js";

export const FRAME_LENGTH = 1280;
const MEL_BINS = 32;
const MEL_WINDOW = 76;
const MEL_CONTEXT = 480;
const EMBEDDING_DIM = 96;
const CLASSIFIER_FRAMES = 16;
const FIRST_EMBEDDABLE_CHUNK = 9; // 5 + 8k >= 77 mel frames
const MEL_CAPACITY = MEL_WINDOW + 1 + 8 * (CLASSIFIER_FRAMES - 1) + 8;

/** What the listener needs from a wake word model; lets tests substitute a scripted one. */
export interface WakeWordModel {
  readonly keywords: string[];
  appendFrame(frame: Int16Array): Promise<void>;
  /** Writes one score per keyword; false while warming up (the first ~2 s). */
  score(out: Float64Array): Promise<boolean>;
  dispose(): Promise<void>;
}

/**
 * Streaming openWakeWord features plus one classifier per keyword. A line-for-line port of
 * WakeWord.Onnx.OnnxWakeWordModel, checked against the training pipeline's golden.json:
 * mel over each 1280-sample chunk plus the previous 480 samples (x / 10 + 2), one embedding per chunk
 * from the 76 mel frames before the newest, and a score from the 16 most recent embeddings.
 * appendFrame only runs the mel model; score() back-fills every chunk since the last call (up to 16),
 * so skipping score() while the VAD gate is closed costs no accuracy.
 */
export class OnnxWakeWordModel implements WakeWordModel {
  private readonly context = new Int16Array(MEL_CONTEXT);
  private readonly melRing = new Float32Array(MEL_CAPACITY * MEL_BINS);
  private readonly embeddingRing = new Float32Array(CLASSIFIER_FRAMES * EMBEDDING_DIM);
  private chunks = 0;
  private melFrames = 0;
  private embedded = 0;

  private constructor(
    readonly keywords: string[],
    private readonly mel: ort.InferenceSession,
    private readonly embedding: ort.InferenceSession,
    private readonly classifiers: { session: ort.InferenceSession; input: string }[],
  ) {}

  static async create(pkg: ModelPackage, keywords?: string[]): Promise<OnnxWakeWordModel> {
    const chosen = (keywords ?? pkg.keywords.map((k) => k.id)).map((id) => pkg.keyword(id));
    const [mel, embedding, ...classifierSessions] = await Promise.all([
      createSession(pkg.file(pkg.manifest.features.melspectrogram)),
      createSession(pkg.file(pkg.manifest.features.embedding)),
      ...chosen.map((k) => createSession(pkg.file(k.model))),
    ]);
    const classifiers = classifierSessions.map((session, i) => ({
      session,
      input: session.inputNames.includes(chosen[i]!.input) ? chosen[i]!.input : session.inputNames[0]!,
    }));
    return new OnnxWakeWordModel(chosen.map((k) => k.id), mel!, embedding!, classifiers);
  }

  /** The newest embedding, for golden-vector checks. */
  get latestEmbedding(): Float32Array {
    if (this.embedded === 0) return new Float32Array(0);
    const slot = (this.embedded - 1) % CLASSIFIER_FRAMES;
    return this.embeddingRing.slice(slot * EMBEDDING_DIM, (slot + 1) * EMBEDDING_DIM);
  }

  async appendFrame(frame: Int16Array): Promise<void> {
    if (frame.length !== FRAME_LENGTH) throw new RangeError(`Frames must be ${FRAME_LENGTH} samples; got ${frame.length}.`);
    const contextLength = this.chunks === 0 ? 0 : MEL_CONTEXT;
    const audio = new Float32Array(contextLength + FRAME_LENGTH);
    for (let i = 0; i < contextLength; i++) audio[i] = this.context[i]!;
    for (let i = 0; i < FRAME_LENGTH; i++) audio[contextLength + i] = frame[i]!;
    this.context.set(frame.subarray(FRAME_LENGTH - MEL_CONTEXT));
    this.chunks++;

    const out = await this.mel.run({ [this.mel.inputNames[0]!]: new ort.Tensor("float32", audio, [1, audio.length]) });
    const mel = out[this.mel.outputNames[0]!]!.data as Float32Array; // [1, 1, frames, 32]
    for (let f = 0; f < mel.length / MEL_BINS; f++) {
      const slot = (this.melFrames % MEL_CAPACITY) * MEL_BINS;
      for (let b = 0; b < MEL_BINS; b++) this.melRing[slot + b] = mel[f * MEL_BINS + b]! / 10 + 2;
      this.melFrames++;
    }
  }

  async score(out: Float64Array): Promise<boolean> {
    if (out.length !== this.classifiers.length) throw new RangeError(`Expected room for ${this.classifiers.length} scores.`);
    const newest = this.chunks - 1;
    if (newest < FIRST_EMBEDDABLE_CHUNK) return false;

    const from = Math.max(this.embedded + FIRST_EMBEDDABLE_CHUNK, newest - CLASSIFIER_FRAMES + 1, FIRST_EMBEDDABLE_CHUNK);
    if (this.embedded + FIRST_EMBEDDABLE_CHUNK < from) this.embedded = from - FIRST_EMBEDDABLE_CHUNK; // older chunks fell out of the window
    await this.embed(from, newest);
    if (this.embedded < CLASSIFIER_FRAMES) return false;

    const input = new Float32Array(CLASSIFIER_FRAMES * EMBEDDING_DIM);
    for (let i = 0; i < CLASSIFIER_FRAMES; i++) {
      const slot = (this.embedded - CLASSIFIER_FRAMES + i) % CLASSIFIER_FRAMES;
      input.set(this.embeddingRing.subarray(slot * EMBEDDING_DIM, (slot + 1) * EMBEDDING_DIM), i * EMBEDDING_DIM);
    }
    const tensor = new ort.Tensor("float32", input, [1, CLASSIFIER_FRAMES, EMBEDDING_DIM]);
    for (let k = 0; k < this.classifiers.length; k++) {
      const { session, input: name } = this.classifiers[k]!;
      const result = await session.run({ [name]: tensor });
      out[k] = (result[session.outputNames[0]!]!.data as Float32Array)[0]!;
    }
    return true;
  }

  private async embed(first: number, last: number): Promise<void> {
    const count = last - first + 1;
    if (count <= 0) return;
    const batch = new Float32Array(count * MEL_WINDOW * MEL_BINS);
    for (let i = 0; i < count; i++) {
      const end = 8 * (first + i) + 4; // newest mel frame after this chunk; the window ends just before it
      const start = end - MEL_WINDOW;
      for (let f = 0; f < MEL_WINDOW; f++) {
        const slot = ((start + f) % MEL_CAPACITY) * MEL_BINS;
        batch.set(this.melRing.subarray(slot, slot + MEL_BINS), (i * MEL_WINDOW + f) * MEL_BINS);
      }
    }
    const out = await this.embedding.run({
      [this.embedding.inputNames[0]!]: new ort.Tensor("float32", batch, [count, MEL_WINDOW, MEL_BINS, 1]),
    });
    const embeddings = out[this.embedding.outputNames[0]!]!.data as Float32Array; // [count, 1, 1, 96]
    for (let i = 0; i < count; i++) {
      const slot = (this.embedded % CLASSIFIER_FRAMES) * EMBEDDING_DIM;
      this.embeddingRing.set(embeddings.subarray(i * EMBEDDING_DIM, (i + 1) * EMBEDDING_DIM), slot);
      this.embedded++;
    }
  }

  async dispose(): Promise<void> {
    await Promise.all([this.mel.release(), this.embedding.release(), ...this.classifiers.map((c) => c.session.release())]);
  }
}
