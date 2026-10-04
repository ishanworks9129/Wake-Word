// Pure helpers, shared by the page and the Node tests (tests/recorder-js/wav.test.mjs).

/** Float samples in [-1, 1] -> 16-bit PCM mono WAV bytes. */
export function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

/**
 * Trims leading/trailing quiet audio (20 ms blocks below -45 dB of the peak), keeping `padSeconds` either side.
 * Keeps the clip short enough for upload limits without clipping the phrase.
 */
export function trimSilence(samples, sampleRate, padSeconds = 0.25) {
  const block = Math.round(sampleRate / 50);
  let peak = 0;
  for (const s of samples) peak = Math.max(peak, Math.abs(s));
  if (peak === 0) return samples;
  const threshold = peak * Math.pow(10, -45 / 20);
  const loud = [];
  for (let b = 0; b * block < samples.length; b++) {
    let sum = 0;
    const end = Math.min(samples.length, (b + 1) * block);
    for (let i = b * block; i < end; i++) sum += samples[i] * samples[i];
    if (Math.sqrt(sum / (end - b * block)) > threshold) loud.push(b);
  }
  if (loud.length === 0) return samples;
  const pad = Math.round(padSeconds * sampleRate);
  const start = Math.max(0, loud[0] * block - pad);
  const stop = Math.min(samples.length, (loud[loud.length - 1] + 1) * block + pad);
  return samples.subarray(start, stop);
}

/** Peak level of float samples, 0..1. */
export function peak(samples) {
  let p = 0;
  for (const s of samples) p = Math.max(p, Math.abs(s));
  return p;
}
