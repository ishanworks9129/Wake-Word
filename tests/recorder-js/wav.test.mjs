// node --test tests/recorder-js
import test from "node:test";
import assert from "node:assert/strict";
import { encodeWav, trimSilence, peak } from "../../src/WakeWord.Recorder/wwwroot/wav.js";

test("encodeWav writes a 16 kHz mono 16-bit PCM header and clamps samples", () => {
  const wav = encodeWav(new Float32Array([0, 0.5, -1, 2]), 16000);
  const v = new DataView(wav.buffer);
  const text = (o) => String.fromCharCode(...wav.slice(o, o + 4));
  assert.equal(text(0), "RIFF");
  assert.equal(text(8), "WAVE");
  assert.equal(v.getUint16(20, true), 1);
  assert.equal(v.getUint16(22, true), 1);
  assert.equal(v.getUint32(24, true), 16000);
  assert.equal(v.getUint16(34, true), 16);
  assert.equal(v.getUint32(40, true), 8);
  assert.deepEqual([0, 1, 2, 3].map((i) => v.getInt16(44 + i * 2, true)), [0, 16383, -32768, 32767]);
});

test("trimSilence keeps the loud part plus padding", () => {
  const rate = 16000;
  const x = new Float32Array(rate * 3);
  for (let i = rate; i < 2 * rate; i++) x[i] = Math.sin(i / 5) * 0.5;
  const t = trimSilence(x, rate, 0.25);
  assert.ok(Math.abs(t.length - 1.5 * rate) <= rate * 0.05, `length ${t.length}`);
  assert.equal(peak(t), peak(x));
  assert.equal(trimSilence(new Float32Array(100), rate).length, 100);
});
