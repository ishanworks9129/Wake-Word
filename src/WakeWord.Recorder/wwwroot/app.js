import { encodeWav, trimSilence, peak } from "./wav.js";

const TARGET_RATE = 16000;
const MAX_RECORD_MS = 5000;
const $ = (id) => document.getElementById(id);

const state = { config: null, prompts: [], done: new Set(), index: 0, participant: null, chunks: [], recording: false, timer: null, wav: null };
const audio = { ctx: null, stream: null, node: null, analyser: null };

// Per-device convenience only: lets a refresh resume. The server is the record of what was saved.
const saved = {
  get() { try { return JSON.parse(localStorage.getItem("recorder.participant")); } catch { return null; } },
  set(v) { try { localStorage.setItem("recorder.participant", JSON.stringify(v)); } catch { /* private mode */ } },
};

function show(id) {
  for (const s of ["welcome", "mic", "record", "done"]) $(s).hidden = s !== id;
  window.scrollTo(0, 0);
}

async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, options);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(body?.detail || `Request failed (${res.status})`);
  return body;
}

async function init() {
  [state.config, state.prompts] = await Promise.all([api("/config"), api("/prompts")]);
  const consent = await fetch(`consent/${state.config.consentVersion}.html`);
  $("consent-text").innerHTML = consent.ok ? await consent.text() : "Consent form missing; please contact the organiser.";

  const previous = saved.get();
  if (previous?.participantId) {
    try {
      const recorded = await api(`/participants/${previous.participantId}/recordings`);
      state.participant = previous;
      recorded.forEach((p) => state.done.add(p));
      show("mic");
      return;
    } catch { /* unknown participant (withdrawn or new server): start over */ }
  }
  show("welcome");
}

$("join-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("join-error").textContent = "";
  try {
    state.participant = await api("/participants", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        inviteCode: $("invite").value,
        consentVersion: state.config.consentVersion,
        agreed: $("agree").checked,
        ageBand: $("age").value || null,
        accent: $("accent").value || null,
        gender: $("gender").value || null,
        device: navigator.userAgent,
      }),
    });
    saved.set(state.participant);
    show("mic");
  } catch (err) {
    $("join-error").textContent = err.message;
  }
});

$("mic-start").addEventListener("click", async () => {
  $("mic-error").textContent = "";
  try {
    // Raw audio, as the app captures it (plan 6.1): no echo cancellation, noise suppression or auto gain.
    audio.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    audio.ctx = new AudioContext();
    await audio.ctx.audioWorklet.addModule("capture-worklet.js");
    const source = audio.ctx.createMediaStreamSource(audio.stream);
    audio.node = new AudioWorkletNode(audio.ctx, "capture-processor");
    audio.node.port.onmessage = (e) => { if (state.recording) state.chunks.push(e.data); };
    audio.analyser = audio.ctx.createAnalyser();
    audio.analyser.fftSize = 1024;
    source.connect(audio.node);
    source.connect(audio.analyser);
    meter();
    $("mic-start").disabled = true;
    $("mic-ok").disabled = false;
  } catch (err) {
    $("mic-error").textContent = window.isSecureContext
      ? `Microphone unavailable: ${err.message}. Check the browser's microphone permission.`
      : "This page must be opened over https for the microphone to work.";
  }
});

function meter() {
  const data = new Float32Array(audio.analyser.fftSize);
  const tick = () => {
    audio.analyser.getFloatTimeDomainData(data);
    $("meter-bar").style.width = `${Math.min(100, peak(data) * 140)}%`;
    requestAnimationFrame(tick);
  };
  tick();
}

$("mic-ok").addEventListener("click", () => {
  state.index = state.prompts.findIndex((p) => !state.done.has(p.id));
  if (state.index < 0) return finish();
  show("record");
  renderPrompt();
});

function renderPrompt() {
  const p = state.prompts[state.index];
  $("phrase").textContent = p.text;
  $("instruction").textContent = p.instruction;
  $("progress-text").textContent = `${state.done.size + 1} of ${state.prompts.length}`;
  $("progress-bar").style.width = `${(100 * state.done.size) / state.prompts.length}%`;
  $("review").hidden = true;
  $("rec").hidden = false;
  $("rec-hint").hidden = false;
  $("rec-error").textContent = "";
}

$("rec").addEventListener("click", async () => {
  if (audio.ctx.state === "suspended") await audio.ctx.resume();
  if (state.recording) return stopRecording();
  state.chunks = [];
  state.recording = true;
  audio.node.port.postMessage("start");
  $("rec").classList.add("on");
  $("rec").textContent = "Recording… tap to stop";
  state.timer = setTimeout(stopRecording, MAX_RECORD_MS);
});

async function stopRecording() {
  if (!state.recording) return;
  clearTimeout(state.timer);
  state.recording = false;
  audio.node.port.postMessage("stop");
  $("rec").classList.remove("on");
  $("rec").textContent = "Tap to record";

  const raw = concat(state.chunks);
  const resampled = await resample(raw, audio.ctx.sampleRate, TARGET_RATE);
  const trimmed = trimSilence(resampled, TARGET_RATE).slice(0, Math.floor(state.config.maxSeconds * TARGET_RATE));
  if (peak(trimmed) < 0.01) {
    $("rec-error").textContent = "We couldn't hear anything. Speak a little closer and try again.";
    return;
  }
  if (trimmed.length < state.config.minSeconds * TARGET_RATE) {
    $("rec-error").textContent = "That was very short. Tap, say the whole phrase, then tap again.";
    return;
  }
  state.wav = encodeWav(trimmed, TARGET_RATE);
  $("playback").src = URL.createObjectURL(new Blob([state.wav], { type: "audio/wav" }));
  $("review").hidden = false;
  $("rec").hidden = true;
  $("rec-hint").hidden = true;
}

$("redo").addEventListener("click", renderPrompt);

$("next").addEventListener("click", async () => {
  const p = state.prompts[state.index];
  $("next").disabled = true;
  $("rec-error").textContent = "";
  try {
    await api(`/participants/${state.participant.participantId}/recordings/${encodeURIComponent(p.id)}`, {
      method: "PUT",
      headers: { "Content-Type": "audio/wav" },
      body: state.wav,
    });
    state.done.add(p.id);
    state.index = state.prompts.findIndex((q) => !state.done.has(q.id));
    if (state.index < 0) finish(); else renderPrompt();
  } catch (err) {
    $("rec-error").textContent = `Upload failed: ${err.message}. Check your connection and tap again.`;
  } finally {
    $("next").disabled = false;
  }
});

function finish() {
  $("done-id").textContent = state.participant.participantId;
  $("done-code").textContent = state.participant.withdrawalCode;
  audio.stream?.getTracks().forEach((t) => t.stop());
  show("done");
}

$("withdraw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api(`/participants/${$("w-id").value.trim()}/withdraw`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ withdrawalCode: $("w-code").value }),
    });
    $("w-msg").textContent = "Deleted. Thank you for taking part.";
    saved.set(null);
  } catch (err) {
    $("w-msg").textContent = err.message;
  }
});

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}

/** The browser's own resampler via OfflineAudioContext; a simple filtered fallback where 16 kHz contexts are refused. */
async function resample(samples, from, to) {
  if (from === to) return samples;
  try {
    const offline = new OfflineAudioContext(1, Math.ceil((samples.length * to) / from), to);
    const buffer = offline.createBuffer(1, samples.length, from);
    buffer.copyToChannel(samples, 0);
    const src = offline.createBufferSource();
    src.buffer = buffer;
    src.connect(offline.destination);
    src.start();
    return (await offline.startRendering()).getChannelData(0);
  } catch {
    const ratio = from / to;
    const out = new Float32Array(Math.floor(samples.length / ratio));
    const width = Math.max(1, Math.round(ratio));
    for (let i = 0; i < out.length; i++) {
      const center = Math.floor(i * ratio);
      let sum = 0, n = 0;
      for (let k = center - width; k <= center + width; k++) {
        if (k >= 0 && k < samples.length) { sum += samples[k]; n++; }
      }
      out[i] = sum / n;
    }
    return out;
  }
}

init().catch((err) => { $("join-error").textContent = `Could not load: ${err.message}`; show("welcome"); });
