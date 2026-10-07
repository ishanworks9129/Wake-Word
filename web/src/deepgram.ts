import { PcmOverrunError, type PcmRingBuffer } from "./ring.js";
import { countWords, type WakePhraseStripper } from "./stripper.js";
import type { TokenProvider } from "./tokens.js";

/** Query parameters for Deepgram's live endpoint (plan 8.3). */
export interface DeepgramStreamingOptions {
  endpoint: string;
  model: string;
  language: string;
  endpointingMs: number;
  utteranceEndMs: number;
  smartFormat: boolean;
  /** Opts out of Deepgram's Model Improvement Program (plan 4.2). */
  mipOptOut: boolean;
  keyterms: string[];
}

export function defaultStreamingOptions(): DeepgramStreamingOptions {
  return {
    endpoint: "wss://api.deepgram.com/v1/listen",
    model: "nova-3",
    language: "en",
    endpointingMs: 300,
    utteranceEndMs: 1000,
    smartFormat: true,
    mipOptOut: true,
    keyterms: [],
  };
}

export function buildListenUrl(o: DeepgramStreamingOptions, sampleRate = 16000): string {
  const q = new URLSearchParams({
    model: o.model,
    language: o.language,
    encoding: "linear16",
    sample_rate: String(sampleRate),
    channels: "1",
    interim_results: "true",
    endpointing: String(o.endpointingMs),
    utterance_end_ms: String(o.utteranceEndMs),
    smart_format: String(o.smartFormat),
    mip_opt_out: String(o.mipOptOut),
  });
  for (const k of o.keyterms) q.append("keyterm", k);
  return `${o.endpoint}?${q}`;
}

/** Cost and safety limits for one session (plan 8.5). */
export interface SessionLimits {
  /** Close if no non-wake-word speech is transcribed within this long of connecting. */
  noTranscriptCutoffMs: number;
  hardTimeoutMs: number;
  /** How long to wait for final results after CloseStream. */
  closeGraceMs: number;
  /** Stop sending while this much is queued in the socket (a stalled network then ends in an overrun). */
  maxBufferedBytes: number;
}

export function defaultSessionLimits(): SessionLimits {
  return { noTranscriptCutoffMs: 4000, hardTimeoutMs: 30000, closeGraceMs: 2000, maxBufferedBytes: 256 * 1024 };
}

export type SessionEndReason =
  | "utterance-end"
  | "no-transcript"
  /** Deepgram's transcript did not contain the wake phrase, so the detector fired on something else. */
  | "not-confirmed"
  | "hard-timeout"
  | "cancelled"
  | "server-closed"
  | "connect-failed"
  | "transport-error"
  | "audio-overrun";

export interface SessionResult {
  reason: SessionEndReason;
  transcript: string;
  audioSentSeconds: number;
  connectLatencyMs: number;
  error?: unknown;
}

export const isErrorReason = (r: SessionEndReason) => r === "connect-failed" || r === "transport-error" || r === "audio-overrun";

export interface TranscriptUpdate {
  text: string;
  isFinal: boolean;
}

/** The socket under a session; abstracted so sessions can be tested without a network. */
export interface DeepgramTransport {
  connect(url: string, accessToken: string): Promise<void>;
  sendAudio(pcm: Int16Array): void;
  sendText(message: string): void;
  readonly bufferedAmount: number;
  onmessage: ((text: string) => void) | null;
  onclose: ((error?: unknown) => void) | null;
  close(): void;
}

/** Browsers cannot set an Authorization header on a WebSocket; Deepgram accepts the token as subprotocols ["bearer", token]. */
export class BrowserWebSocketTransport implements DeepgramTransport {
  private socket: WebSocket | null = null;
  onmessage: ((text: string) => void) | null = null;
  onclose: ((error?: unknown) => void) | null = null;

  get bufferedAmount(): number {
    return this.socket?.bufferedAmount ?? 0;
  }

  connect(url: string, accessToken: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url, ["bearer", accessToken]);
      socket.binaryType = "arraybuffer";
      let open = false;
      socket.onopen = () => { open = true; resolve(); };
      socket.onmessage = (e) => { if (typeof e.data === "string") this.onmessage?.(e.data); };
      socket.onerror = () => { if (!open) reject(new Error("WebSocket connection failed")); };
      socket.onclose = (e) => {
        if (!open) return reject(new Error(`WebSocket closed before opening (${e.code} ${e.reason})`));
        this.onclose?.(e.code === 1000 ? undefined : new Error(`WebSocket closed: ${e.code} ${e.reason}`));
      };
      this.socket = socket;
    });
  }

  sendAudio(pcm: Int16Array): void {
    this.socket?.send(new Int16Array(pcm)); // copy: linear16 little-endian, as every browser platform is
  }

  sendText(message: string): void {
    this.socket?.send(message);
  }

  close(): void {
    this.socket?.close(1000);
  }
}

/**
 * One Deepgram streaming session. Streams from a position in the pre-roll ring and keeps following the
 * write head from the same cursor, so pre-roll and live audio form one gapless stream. The owner calls
 * pump() after writing new audio to the ring.
 */
export class DeepgramSession {
  ontranscript: ((u: TranscriptUpdate) => void) | null = null;

  private transport: DeepgramTransport | null = null;
  private cursor = 0;
  private sent = 0;
  private connected = false;
  private ended = false;
  private hasSpeech = false;
  private hasFinalSpeech = false;
  private confirmed = true;
  private readonly finals: string[] = [];
  private readonly timers: ReturnType<typeof setTimeout>[] = [];
  private finish: ((r: { reason: SessionEndReason; error?: unknown }) => void) | null = null;
  private closed: (() => void) | null = null;

  constructor(
    private readonly ring: PcmRingBuffer,
    private readonly tokens: TokenProvider,
    private readonly transportFactory: () => DeepgramTransport,
    private readonly streaming: DeepgramStreamingOptions,
    private readonly limits: SessionLimits,
    private readonly stripper: WakePhraseStripper,
  ) {}

  /**
   * Runs the session; resolves when it ends. `fromSample` is clamped to what the ring still holds.
   * With `confirmWakePhrase` (wake-word sessions), it ends as "not-confirmed" unless the transcript contains the
   * wake phrase, and passes no text on until it does.
   */
  async run(fromSample: number, confirmWakePhrase = false): Promise<SessionResult> {
    this.confirmed = !confirmWakePhrase;
    this.cursor = Math.max(fromSample, this.ring.oldestAvailable, 0);
    const started = performance.now();
    const outcome = new Promise<{ reason: SessionEndReason; error?: unknown }>((resolve) => { this.finish = resolve; });

    try {
      const token = await this.tokens.getToken();
      if (this.ended) return this.result("cancelled", started, started);
      this.transport = this.transportFactory();
      this.transport.onmessage = (text) => this.onMessage(text);
      this.transport.onclose = (error) => {
        this.closed?.();
        this.end(error ? "transport-error" : "server-closed", error);
      };
      await this.transport.connect(buildListenUrl(this.streaming, this.ring.sampleRate), token.accessToken);
    } catch (error) {
      this.transport?.close();
      return this.result(this.ended ? "cancelled" : "connect-failed", started, performance.now(), error);
    }
    const connectedAt = performance.now();
    if (this.ended) {
      this.transport.close();
      return this.result("cancelled", started, connectedAt);
    }

    this.connected = true;
    this.timers.push(setTimeout(() => { if (!this.hasSpeech) this.end("no-transcript"); }, this.limits.noTranscriptCutoffMs));
    this.timers.push(setTimeout(() => this.end("hard-timeout"), this.limits.hardTimeoutMs));
    this.pump();

    const { reason, error } = await outcome;
    this.timers.forEach(clearTimeout);
    if (reason !== "server-closed" && reason !== "transport-error") {
      // Ask Deepgram to flush final results, then give it a moment to send them and close.
      const closedByServer = new Promise<void>((resolve) => { this.closed = resolve; });
      try { this.transport.sendText(JSON.stringify({ type: "CloseStream" })); } catch { /* socket already gone */ }
      await Promise.race([closedByServer, new Promise((r) => setTimeout(r, this.limits.closeGraceMs))]);
    }
    this.transport.onmessage = null;
    this.transport.onclose = null;
    this.transport.close();
    return this.result(reason, started, connectedAt, error);
  }

  /** Sends everything from the cursor to the ring's write head. */
  pump(): void {
    if (!this.connected || this.ended || !this.transport) return;
    try {
      while (this.transport.bufferedAmount < this.limits.maxBufferedBytes) {
        const chunk = new Int16Array(1600);
        const n = this.ring.read(this.cursor, chunk);
        if (n === 0) return;
        this.transport.sendAudio(chunk.subarray(0, n));
        this.cursor += n;
        this.sent += n;
      }
      this.ring.read(this.cursor, new Int16Array(1)); // throws if the stalled cursor has been overwritten
    } catch (error) {
      this.end(error instanceof PcmOverrunError ? "audio-overrun" : "transport-error", error);
    }
  }

  /** Ends the session early (e.g. the user tapped stop). */
  cancel(): void {
    this.end("cancelled");
  }

  private end(reason: SessionEndReason, error?: unknown): void {
    if (this.ended) return;
    this.ended = true;
    this.finish?.({ reason, error });
  }

  private onMessage(text: string): void {
    let msg: { type?: string; is_final?: boolean; channel?: { alternatives?: { transcript?: string }[] } };
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.type === "UtteranceEnd") {
      if (!this.confirmed && this.finals.length > 0) this.end("not-confirmed");
      else if (this.hasFinalSpeech) this.end("utterance-end");
      return;
    }
    if (msg.type !== "Results") return;
    const transcript = msg.channel?.alternatives?.[0]?.transcript ?? "";
    if (!transcript) return;
    const isFinal = msg.is_final === true;
    const heard = [...this.finals, transcript].join(" ");
    if (isFinal) this.finals.push(transcript);
    if (!this.confirmed) {
      this.confirmed = this.stripper.contains(heard);
      if (!this.confirmed) {
        // Interim text may still be catching up; enough final words without the phrase settle it.
        if (isFinal && countWords(heard) >= this.stripper.wordsToDecide) this.end("not-confirmed");
        return;
      }
    }
    const stripped = this.stripper.strip(heard);
    if (stripped.length > 0) {
      this.hasSpeech = true; // interim words count, so a slow speaker is not cut off
      if (isFinal) this.hasFinalSpeech = true;
    }
    this.ontranscript?.({ text: stripped, isFinal });
  }

  private result(reason: SessionEndReason, started: number, connectedAt: number, error?: unknown): SessionResult {
    this.ended = true;
    return {
      reason,
      transcript: this.confirmed ? this.stripper.strip(this.finals.join(" ")) : "",
      audioSentSeconds: this.sent / this.ring.sampleRate,
      connectLatencyMs: Math.round(connectedAt - started),
      error,
    };
  }
}
