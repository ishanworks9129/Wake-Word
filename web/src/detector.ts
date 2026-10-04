// Fire rules shared with C# (WakeWord.Core.Detection) and Python (training/eval/detector.py).
// testdata/golden/detector_cases.json keeps all three identical.

export const SAMPLE_RATE = 16000;
export const SILENCE_DBFS = -100;

export interface ThresholdPoint {
  noiseFloorDbfs: number;
  offset: number;
}

export interface AdaptiveThresholdOptions {
  points: ThresholdPoint[];
  minThreshold: number;
  maxThreshold: number;
}

export interface DetectorOptions {
  baseThreshold: number;
  /** Frames in a row at or above threshold before firing. */
  consecutiveFrames: number;
  /** New triggers are ignored for this long after a fire. */
  refractorySeconds: number;
  adaptive: AdaptiveThresholdOptions;
}

export function defaultDetectorOptions(): DetectorOptions {
  return {
    baseThreshold: 0.5,
    consecutiveFrames: 3,
    refractorySeconds: 1.5,
    adaptive: { points: [], minThreshold: 0.2, maxThreshold: 0.95 },
  };
}

/** Base threshold plus a piecewise-linear noise offset, clamped. */
export function adaptiveThreshold(base: number, noiseFloorDbfs: number, options: AdaptiveThresholdOptions): number {
  const p = options.points;
  let offset = 0;
  const first = p[0];
  const last = p[p.length - 1];
  if (first && last) {
    if (noiseFloorDbfs <= first.noiseFloorDbfs) {
      offset = first.offset;
    } else if (noiseFloorDbfs >= last.noiseFloorDbfs) {
      offset = last.offset;
    } else {
      for (let i = 1; i < p.length; i++) {
        const lo = p[i - 1]!;
        const hi = p[i]!;
        if (noiseFloorDbfs <= hi.noiseFloorDbfs) {
          const t = (noiseFloorDbfs - lo.noiseFloorDbfs) / (hi.noiseFloorDbfs - lo.noiseFloorDbfs);
          offset = lo.offset + t * (hi.offset - lo.offset);
          break;
        }
      }
    }
  }
  return Math.min(Math.max(base + offset, options.minThreshold), options.maxThreshold);
}

export interface DetectorFire {
  frameEndSample: number;
  score: number;
  threshold: number;
}

/** Adaptive threshold, N consecutive frames, then a refractory period. */
export class WakeWordDetector {
  private run = 0;
  private lastFire: number | null = null;
  private readonly refractorySamples: number;
  currentThreshold = 0;

  constructor(private readonly options: DetectorOptions, sampleRate = SAMPLE_RATE) {
    if (options.consecutiveFrames < 1) throw new RangeError("consecutiveFrames must be at least 1");
    this.refractorySamples = Math.round(options.refractorySeconds * sampleRate);
  }

  process(score: number, noiseFloorDbfs: number, frameEndSample: number): DetectorFire | null {
    const threshold = adaptiveThreshold(this.options.baseThreshold, noiseFloorDbfs, this.options.adaptive);
    this.currentThreshold = threshold;
    if (this.lastFire !== null && frameEndSample - this.lastFire < this.refractorySamples) {
      this.run = 0;
      return null;
    }
    if (score < threshold) {
      this.run = 0;
      return null;
    }
    if (++this.run < this.options.consecutiveFrames) return null;
    this.run = 0;
    this.lastFire = frameEndSample;
    return { frameEndSample, score, threshold };
  }

  /** Breaks the current run, e.g. when the VAD gate closes. */
  reset(): void {
    this.run = 0;
  }
}

/** Picovoice-style sensitivity (0 = fewest false accepts, 1 = fewest misses) to threshold, via the trained calibration table. */
export class SensitivityTable {
  private readonly points: [number, number][];

  constructor(points: [number, number][]) {
    if (points.length === 0) throw new RangeError("A sensitivity table needs at least one point.");
    this.points = [...points].sort((a, b) => a[0] - b[0]);
  }

  thresholdFor(sensitivity: number): number {
    const s = Math.min(Math.max(sensitivity, 0), 1);
    for (let i = 1; i < this.points.length; i++) {
      const [s0, t0] = this.points[i - 1]!;
      const [s1, t1] = this.points[i]!;
      if (s <= s1) return s1 === s0 ? t1 : t0 + ((t1 - t0) * (s - s0)) / (s1 - s0);
    }
    return this.points[this.points.length - 1]![1];
  }
}

export interface VadGateOptions {
  onsetProbability: number;
  /** While open, frames at or above this keep it open. */
  sustainProbability: number;
  /** Inference continues this long after speech drops. */
  hangoverSeconds: number;
}

export function defaultVadGateOptions(): VadGateOptions {
  return { onsetProbability: 0.5, sustainProbability: 0.35, hangoverSeconds: 1 };
}

export type VadTransition = "none" | "opened" | "closed";

export class VadGate {
  isOpen = false;
  private lastSpeech = 0;
  private readonly hangoverSamples: number;

  constructor(private readonly options: VadGateOptions, sampleRate = SAMPLE_RATE) {
    this.hangoverSamples = Math.round(options.hangoverSeconds * sampleRate);
  }

  update(probability: number, frameEndSample: number): VadTransition {
    if (!this.isOpen) {
      if (probability < this.options.onsetProbability) return "none";
      this.isOpen = true;
      this.lastSpeech = frameEndSample;
      return "opened";
    }
    if (probability >= this.options.sustainProbability) {
      this.lastSpeech = frameEndSample;
      return "none";
    }
    if (frameEndSample - this.lastSpeech > this.hangoverSamples) {
      this.isOpen = false;
      return "closed";
    }
    return "none";
  }
}

/** Follows quiet frames down quickly and rises slowly, so speech bursts barely lift it. */
export class NoiseFloorEstimator {
  floorDbfs = SILENCE_DBFS;
  private initialised = false;

  constructor(private readonly fallFactor = 0.3, private readonly riseDbPerSecond = 2) {}

  update(frameDbfs: number, frameSeconds: number): number {
    if (!this.initialised) {
      this.floorDbfs = frameDbfs;
      this.initialised = true;
    } else if (frameDbfs < this.floorDbfs) {
      this.floorDbfs += (frameDbfs - this.floorDbfs) * this.fallFactor;
    } else {
      this.floorDbfs += Math.min(frameDbfs - this.floorDbfs, this.riseDbPerSecond * frameSeconds);
    }
    return this.floorDbfs;
  }
}

/** RMS level of 16-bit PCM in dB relative to full scale. */
export function dbfs(samples: Int16Array): number {
  if (samples.length === 0) return SILENCE_DBFS;
  let sum = 0;
  for (const s of samples) sum += s * s;
  const rms = Math.sqrt(sum / samples.length) / 32767;
  return rms <= 0 ? SILENCE_DBFS : Math.max(SILENCE_DBFS, 20 * Math.log10(rms));
}
