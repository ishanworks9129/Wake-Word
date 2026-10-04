/** A reader fell behind the ring's capacity; the audio it needed is gone. */
export class PcmOverrunError extends Error {
  constructor(readonly requested: number, readonly oldestAvailable: number) {
    super(`Sample ${requested} was overwritten; oldest available is ${oldestAvailable}.`);
    this.name = "PcmOverrunError";
  }
}

/**
 * Fixed-capacity ring of 16-bit PCM addressed by absolute sample position. Every captured frame is
 * written here first; a Deepgram session reads from any position still held, so the pre-roll and live
 * audio form one gapless stream (plan 8.4).
 */
export class PcmRingBuffer {
  private readonly buffer: Int16Array;
  totalWritten = 0;

  constructor(readonly sampleRate: number, capacitySeconds: number) {
    this.buffer = new Int16Array(Math.round(sampleRate * capacitySeconds));
  }

  get capacity(): number {
    return this.buffer.length;
  }

  get oldestAvailable(): number {
    return Math.max(0, this.totalWritten - this.buffer.length);
  }

  samplesFor(seconds: number): number {
    return Math.round(seconds * this.sampleRate);
  }

  write(samples: Int16Array): void {
    let src = samples;
    if (src.length > this.buffer.length) {
      this.totalWritten += src.length - this.buffer.length;
      src = src.subarray(src.length - this.buffer.length);
    }
    const start = this.totalWritten % this.buffer.length;
    const first = Math.min(src.length, this.buffer.length - start);
    this.buffer.set(src.subarray(0, first), start);
    this.buffer.set(src.subarray(first), 0);
    this.totalWritten += src.length;
  }

  /** Copies from an absolute position; returns the count copied (0 at the write head). */
  read(position: number, destination: Int16Array): number {
    const oldest = this.oldestAvailable;
    if (position < oldest) throw new PcmOverrunError(position, oldest);
    const count = Math.min(destination.length, this.totalWritten - position);
    if (count <= 0) return 0;
    const start = position % this.buffer.length;
    const first = Math.min(count, this.buffer.length - start);
    destination.set(this.buffer.subarray(start, start + first), 0);
    destination.set(this.buffer.subarray(0, count - first), first);
    return count;
  }
}
