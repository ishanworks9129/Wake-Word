import { MicCapture, type MicCaptureOptions } from "./capture.js";
import {
  DeepgramSession,
  BrowserWebSocketTransport,
  defaultSessionLimits,
  defaultStreamingOptions,
  type DeepgramStreamingOptions,
  type DeepgramTransport,
  type SessionLimits,
  type SessionResult,
  type TranscriptUpdate,
} from "./deepgram.js";
import { NoiseFloorEstimator, VadGate, WakeWordDetector, dbfs, defaultVadGateOptions, type VadGateOptions } from "./detector.js";
import { FRAME_LENGTH, OnnxWakeWordModel, type WakeWordModel } from "./model.js";
import { ModelPackage, type KeywordOptions } from "./package.js";
import { PcmRingBuffer } from "./ring.js";
import { WakePhraseStripper } from "./stripper.js";
import type { TokenProvider } from "./tokens.js";
import { SileroVad, type VoiceActivityDetector } from "./vad.js";

export interface Detection {
  keywordIndex: number;
  keyword: string;
  score: number;
  threshold: number;
  frameEndSample: number;
}

export type SessionTrigger = "wake-word" | "manual";
export type SessionRejection = "busy" | "rate-limited" | "no-token-provider";

export interface SessionTelemetry {
  trigger: SessionTrigger;
  keyword: string | null;
  score: number | null;
  result: SessionResult;
}

export interface ListenerOptions {
  keywords: KeywordOptions[];
  tokens?: TokenProvider;
  preRollSeconds?: number;
  ringSeconds?: number;
  maxSessionsPerHour?: number;
  vad?: Partial<VadGateOptions>;
  deepgram?: Partial<DeepgramStreamingOptions>;
  limits?: Partial<SessionLimits>;
  transportFactory?: () => DeepgramTransport;
  now?: () => number;
}

export interface CreateListenerOptions extends Omit<ListenerOptions, "keywords"> {
  /** Base URL of the trained package (models.json and .onnx files). */
  modelBaseUrl: string;
  /** URL of silero_vad.onnx. */
  vadModelUrl: string;
  keywords?: string[];
  sensitivities?: number[];
}

type Handler<T> = ((value: T) => void) | null;

/**
 * The always-on loop with Deepgram handoff (plan 8): every frame goes to the pre-roll ring, the mel stage
 * runs every frame, Silero VAD gates the rest, each keyword has its own detector, and a fire starts one
 * Deepgram session that streams from 2 s before the trigger. Same behaviour as WakeWordController in C#.
 */
export class WakeWordListener {
  ondetect: Handler<Detection> = null;
  onspeechstart: Handler<void> = null;
  onscores: Handler<{ scores: readonly number[]; vadOpen: boolean; noiseFloorDbfs: number }> = null;
  onsessionstart: Handler<SessionTrigger> = null;
  ontranscript: Handler<TranscriptUpdate> = null;
  onsessionend: Handler<SessionTelemetry> = null;
  onreject: Handler<SessionRejection> = null;

  /** Pauses detection, e.g. while the app plays its own audio (plan 6.1); audio still reaches the ring. */
  paused = false;
  framesProcessed = 0;
  framesScored = 0;

  readonly ring: PcmRingBuffer;
  private readonly detectors: WakeWordDetector[];
  private readonly scores: Float64Array;
  private readonly gate: VadGate;
  private readonly noise = new NoiseFloorEstimator();
  private readonly stripper: WakePhraseStripper;
  private readonly sessionStarts: number[] = [];
  private readonly now: () => number;
  private queue: Promise<void> = Promise.resolve();
  private session: DeepgramSession | null = null;
  private capture: MicCapture | null = null;

  constructor(private readonly model: WakeWordModel, private readonly vad: VoiceActivityDetector, private readonly options: ListenerOptions) {
    const ids = options.keywords.map((k) => k.id);
    if (ids.join() !== model.keywords.join()) {
      throw new Error(`Keyword options [${ids.join(", ")}] must match the model's [${model.keywords.join(", ")}], in order.`);
    }
    this.ring = new PcmRingBuffer(16000, options.ringSeconds ?? 10);
    this.detectors = options.keywords.map((k) => new WakeWordDetector(k.detector));
    this.scores = new Float64Array(this.detectors.length);
    this.gate = new VadGate({ ...defaultVadGateOptions(), ...options.vad });
    this.stripper = new WakePhraseStripper(options.keywords.flatMap((k) => [k.phrase, ...k.transcriptVariants]));
    this.now = options.now ?? Date.now;
  }

  /** Loads the trained package and Silero VAD from URLs. */
  static async create(options: CreateListenerOptions): Promise<WakeWordListener> {
    const pkg = await ModelPackage.fromUrl(options.modelBaseUrl);
    const keywords = pkg.keywordOptions(options.keywords, options.sensitivities);
    const [model, vad] = await Promise.all([
      OnnxWakeWordModel.create(pkg, keywords.map((k) => k.id)),
      SileroVad.fromUrl(options.vadModelUrl),
    ]);
    return new WakeWordListener(model, vad, { ...options, keywords });
  }

  get keywords(): string[] {
    return this.model.keywords;
  }

  get isStreaming(): boolean {
    return this.session !== null;
  }

  /** Starts the microphone. Call from a user gesture (browsers require it for audio). */
  async start(capture: MicCaptureOptions = {}): Promise<void> {
    if (this.capture) return;
    this.capture = await MicCapture.start((frame) => { void this.enqueue(frame); }, capture);
  }

  async stop(): Promise<void> {
    await this.capture?.stop();
    this.capture = null;
    this.session?.cancel();
    await this.queue;
  }

  /** Feeds one 1280-sample frame. Frames are processed strictly in order. */
  enqueue(frame: Int16Array): Promise<void> {
    if (frame.length !== FRAME_LENGTH) throw new RangeError(`Frames must be ${FRAME_LENGTH} samples.`);
    this.queue = this.queue.then(() => this.processFrame(frame));
    return this.queue;
  }

  /** Tap-to-talk (plan 9.2): streams from now, no pre-roll. */
  startManualSession(): boolean {
    return this.tryStart("manual", this.ring.totalWritten, null);
  }

  stopSession(): void {
    this.session?.cancel();
  }

  async dispose(): Promise<void> {
    await this.stop();
    await Promise.all([this.model.dispose(), this.vad.dispose()]);
  }

  private async processFrame(frame: Int16Array): Promise<void> {
    this.ring.write(frame);
    this.session?.pump();
    const frameEnd = this.ring.totalWritten;
    this.framesProcessed++;
    this.noise.update(dbfs(frame), FRAME_LENGTH / 16000);
    await this.model.appendFrame(frame);

    const transition = this.gate.update(await this.vad.speechProbability(frame), frameEnd);
    if (transition === "opened") {
      this.options.tokens?.prefetch();
      this.onspeechstart?.();
    }

    if (!this.gate.isOpen || this.paused) {
      this.detectors.forEach((d) => d.reset());
      this.onscores?.({ scores: Array(this.detectors.length).fill(0), vadOpen: this.gate.isOpen, noiseFloorDbfs: this.noise.floorDbfs });
      return;
    }

    this.framesScored++;
    if (!(await this.model.score(this.scores))) return;
    this.onscores?.({ scores: Array.from(this.scores), vadOpen: true, noiseFloorDbfs: this.noise.floorDbfs });
    for (let k = 0; k < this.detectors.length; k++) {
      const fire = this.detectors[k]!.process(this.scores[k]!, this.noise.floorDbfs, frameEnd);
      if (fire) {
        const detection = { ...fire, keywordIndex: k, keyword: this.model.keywords[k]! };
        this.ondetect?.(detection);
        this.tryStart("wake-word", frameEnd - this.ring.samplesFor(this.options.preRollSeconds ?? 2), detection);
      }
    }
  }

  private tryStart(trigger: SessionTrigger, fromSample: number, detection: Detection | null): boolean {
    if (this.session) {
      this.onreject?.("busy");
      return false;
    }
    const tokens = this.options.tokens;
    if (!tokens) {
      this.onreject?.("no-token-provider");
      return false;
    }
    const now = this.now();
    while (this.sessionStarts.length && now - this.sessionStarts[0]! >= 3_600_000) this.sessionStarts.shift();
    if (this.sessionStarts.length >= (this.options.maxSessionsPerHour ?? 60)) {
      this.onreject?.("rate-limited");
      return false;
    }
    this.sessionStarts.push(now);

    const session = new DeepgramSession(
      this.ring,
      tokens,
      this.options.transportFactory ?? (() => new BrowserWebSocketTransport()),
      { ...defaultStreamingOptions(), ...this.options.deepgram },
      { ...defaultSessionLimits(), ...this.options.limits },
      this.stripper,
    );
    session.ontranscript = (u) => this.ontranscript?.(u);
    this.session = session;
    this.onsessionstart?.(trigger);
    void session.run(fromSample).then((result) => {
      this.session = null;
      this.onsessionend?.({ trigger, keyword: detection?.keyword ?? null, score: detection?.score ?? null, result });
    });
    return true;
  }
}
