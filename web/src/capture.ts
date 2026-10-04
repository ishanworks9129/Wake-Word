import { FRAME_LENGTH } from "./model.js";
import { FrameAssembler, StreamingResampler, floatToInt16 } from "./resampler.js";

// Inline so the library needs no separate worklet file from the host app's bundler.
const WORKLET = `
class WakeWordCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("wake-word-capture", WakeWordCapture);
`;

export interface MicCaptureOptions {
  deviceId?: string;
}

/**
 * Microphone capture as plan 6.1 requires: echo cancellation, noise suppression and auto gain off, so the
 * signal matches training. Delivers 16 kHz mono 1280-sample frames. Requires a secure context (https).
 */
export class MicCapture {
  private constructor(
    private readonly stream: MediaStream,
    private readonly context: AudioContext,
    private readonly node: AudioWorkletNode,
  ) {}

  get inputSampleRate(): number {
    return this.context.sampleRate;
  }

  static async start(onFrame: (frame: Int16Array) => void, options: MicCaptureOptions = {}): Promise<MicCapture> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: options.deviceId ? { exact: options.deviceId } : undefined,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const context = new AudioContext();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    try {
      await context.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "wake-word-capture");
    const resampler = new StreamingResampler(context.sampleRate, 16000);
    const frames = new FrameAssembler(FRAME_LENGTH);
    node.port.onmessage = (e: MessageEvent<Float32Array>) => frames.push(floatToInt16(resampler.push(e.data)), onFrame);
    source.connect(node);
    if (context.state === "suspended") await context.resume();
    return new MicCapture(stream, context, node);
  }

  async stop(): Promise<void> {
    this.node.port.onmessage = null;
    this.node.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    await this.context.close();
  }
}
