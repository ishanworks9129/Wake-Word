/**
 * Streaming windowed-sinc resampler (Blackman window, 16 zero crossings), so 44.1 or 48 kHz microphone
 * audio becomes clean 16 kHz without aliasing. Browsers' own AudioContext({ sampleRate: 16000 }) is not
 * reliable everywhere (notably Safari), so the library resamples itself.
 */
export class StreamingResampler {
  private readonly ratio: number; // input samples per output sample
  private readonly cutoff: number; // cycles per input sample
  private readonly halfWidth: number; // taps either side, in input samples
  private buffer = new Float32Array(0);
  private bufferStart = 0; // absolute index of buffer[0]
  private inputCount = 0;
  private outputCount = 0;

  constructor(readonly fromRate: number, readonly toRate: number, zeroCrossings = 16) {
    this.ratio = fromRate / toRate;
    this.cutoff = 0.5 * Math.min(1, toRate / fromRate) * 0.95;
    this.halfWidth = Math.ceil(zeroCrossings / (2 * this.cutoff));
  }

  /** Feeds input samples; returns every output sample that can now be computed. */
  push(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate) return input.slice();
    const merged = new Float32Array(this.buffer.length + input.length);
    merged.set(this.buffer);
    merged.set(input, this.buffer.length);
    this.buffer = merged;
    this.inputCount += input.length;

    const out: number[] = [];
    for (;;) {
      const t = this.outputCount * this.ratio;
      const center = Math.floor(t);
      if (center + this.halfWidth >= this.inputCount) break;
      let sum = 0;
      let weight = 0;
      for (let k = center - this.halfWidth + 1; k <= center + this.halfWidth; k++) {
        const x = t - k;
        const w = this.kernel(x);
        weight += w;
        const idx = k - this.bufferStart;
        if (idx >= 0) sum += this.buffer[idx]! * w; // before the first sample counts as silence
      }
      out.push(weight !== 0 ? sum / weight : 0);
      this.outputCount++;
    }

    // Keep only what the next output still needs.
    const keepFrom = Math.max(0, Math.floor(this.outputCount * this.ratio) - this.halfWidth);
    if (keepFrom > this.bufferStart) {
      this.buffer = this.buffer.slice(keepFrom - this.bufferStart);
      this.bufferStart = keepFrom;
    }
    return Float32Array.from(out);
  }

  private table: Float32Array | null = null;

  /** Kernel value at offset x (input samples), from a table interpolated linearly: no trig per tap. */
  private kernel(x: number): number {
    const a = Math.abs(x);
    if (a >= this.halfWidth) return 0;
    const table = (this.table ??= this.buildTable());
    const pos = a * TABLE_STEPS;
    const i = Math.floor(pos);
    const frac = pos - i;
    return table[i]! + (table[i + 1]! - table[i]!) * frac;
  }

  private buildTable(): Float32Array {
    const table = new Float32Array(this.halfWidth * TABLE_STEPS + 2);
    for (let i = 0; i < table.length; i++) {
      const x = i / TABLE_STEPS;
      if (x >= this.halfWidth) break;
      const arg = 2 * this.cutoff * x;
      const sinc = arg === 0 ? 1 : Math.sin(Math.PI * arg) / (Math.PI * arg);
      const n = (x / this.halfWidth + 1) / 2; // 0.5..1 across the right half of the window
      table[i] = sinc * (0.42 - 0.5 * Math.cos(2 * Math.PI * n) + 0.08 * Math.cos(4 * Math.PI * n));
    }
    return table;
  }
}

const TABLE_STEPS = 512;

/** Float [-1, 1] to 16-bit PCM. */
export function floatToInt16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return out;
}

/** Collects 16 kHz samples into fixed-size frames. */
export class FrameAssembler {
  private readonly frame: Int16Array;
  private fill = 0;

  constructor(readonly frameLength: number) {
    this.frame = new Int16Array(frameLength);
  }

  push(samples: Int16Array, onFrame: (frame: Int16Array) => void): void {
    let offset = 0;
    while (offset < samples.length) {
      const take = Math.min(samples.length - offset, this.frameLength - this.fill);
      this.frame.set(samples.subarray(offset, offset + take), this.fill);
      this.fill += take;
      offset += take;
      if (this.fill === this.frameLength) {
        onFrame(this.frame.slice());
        this.fill = 0;
      }
    }
  }
}
