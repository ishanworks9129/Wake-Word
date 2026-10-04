import { ModelPackage, OnnxWakeWordModel, SileroVad, WakeWordListener, configureOnnxRuntime } from "../src/index.js";

configureOnnxRuntime();

interface Job {
  id: string;
  phrase: string;
  phrase_id: string;
  profile: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  step: string;
  progress: number;
  message: string;
  created: number;
  started: number | null;
  finished: number | null;
  error: string | null;
  metrics: { recall?: number; fa_per_hour?: number; threshold?: number; negative_hours?: number };
  queue_position: number;
  package_url: string | null;
  log?: string[];
}

const MIC = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z"/></svg>`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

let current: Job | null = null;
let poll: ReturnType<typeof setTimeout> | null = null;
let listener: WakeWordListener | null = null;
let pkg: ModelPackage | null = null;
let sensitivity = 0.5;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(body?.detail ?? `HTTP ${res.status}`);
  return body as T;
}

// ---- form

const phraseInput = $<HTMLInputElement>("phrase");
phraseInput.oninput = () => {
  const ok = phraseInput.value.trim().length >= 3;
  $<HTMLButtonElement>("train").disabled = !ok;
  $<HTMLButtonElement>("hear").disabled = !ok;
};

$("hear").onclick = async () => {
  const button = $<HTMLButtonElement>("hear");
  button.disabled = true;
  try {
    const res = await fetch(`/api/preview?phrase=${encodeURIComponent(phraseInput.value.trim())}`);
    if (!res.ok) throw new Error((await res.json()).detail);
    const audio = new Audio(URL.createObjectURL(await res.blob()));
    await audio.play();
  } catch (err) {
    notice(`Couldn't play a preview: ${(err as Error).message}`);
  } finally {
    button.disabled = false;
  }
};

$<HTMLFormElement>("train-form").onsubmit = async (e) => {
  e.preventDefault();
  $<HTMLButtonElement>("train").disabled = true;
  try {
    const job = await api<Job>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase: phraseInput.value, profile: $<HTMLSelectElement>("profile").value }),
    });
    await open(job.id);
    await refreshJobs();
  } catch (err) {
    notice((err as Error).message);
  } finally {
    $<HTMLButtonElement>("train").disabled = phraseInput.value.trim().length < 3;
  }
};

// ---- job list and selection

async function refreshJobs(): Promise<void> {
  const jobs = await api<Job[]>("/api/jobs");
  const list = $("jobs");
  if (jobs.length === 0) {
    list.innerHTML = `<li class="muted">None yet.</li>`;
    return;
  }
  list.innerHTML = jobs
    .map((j) => `<li><button data-id="${j.id}" class="${j.id === current?.id ? "active" : ""}">
      <span>${esc(j.phrase)}</span><span class="badge ${j.status}">${label(j)}</span></button></li>`)
    .join("");
  list.querySelectorAll<HTMLButtonElement>("button[data-id]").forEach((b) => (b.onclick = () => void open(b.dataset.id!)));
}

function label(j: Job): string {
  if (j.status === "running") return `${Math.round(j.progress * 100)}%`;
  if (j.status === "queued") return j.queue_position > 1 ? `queued #${j.queue_position}` : "starting";
  return j.status === "done" ? "ready" : j.status;
}

async function open(id: string): Promise<void> {
  await stopListening();
  pkg = null;
  location.hash = `job=${id}`;
  await update(id);
}

async function update(id: string): Promise<void> {
  if (poll) clearTimeout(poll);
  const job = await api<Job>(`/api/jobs/${id}`);
  const changed = !current || current.id !== job.id || current.status !== job.status;
  current = job;
  if (changed || job.status === "running" || job.status === "queued") render(job);
  if (job.status === "queued" || job.status === "running") poll = setTimeout(() => void update(id).then(refreshJobs), 2000);
  if (changed) void refreshJobs();
}

// ---- the stage under the form

function render(job: Job): void {
  const stage = $("stage");
  const elapsed = job.started ? Math.round(((job.finished ?? Date.now() / 1000) - job.started) / 60) : 0;
  if (job.status === "queued" || job.status === "running") {
    stage.innerHTML = `
      <div class="mic">${MIC}</div>
      <div><b>Training "${esc(job.phrase)}"</b></div>
      <div class="progress"><div style="width:${(job.progress * 100).toFixed(1)}%"></div></div>
      <div class="muted">${esc(job.status === "queued" && job.queue_position > 1 ? `Waiting: ${job.queue_position - 1} ahead` : job.message)} · ${Math.round(job.progress * 100)}% · ${elapsed} min</div>
      <button class="link" id="cancel">Cancel</button>`;
    $("cancel").onclick = () => void api(`/api/jobs/${job.id}`, { method: "DELETE" }).then(() => update(job.id));
  } else if (job.status === "done") {
    const m = job.metrics;
    stage.innerHTML = `
      <button class="mic ready" id="mic" aria-label="Test with your microphone">${MIC}</button>
      <div class="heard-text" id="heard" aria-live="polite"></div>
      <div id="hint" class="muted">Click the microphone and say "${esc(job.phrase)}".</div>
      <div class="score" hidden id="score"><div class="bar"><div class="fill" id="fill"></div><div class="mark" id="mark"></div></div></div>
      <label class="slider">Sensitivity: <output id="sens-out">${sensitivity.toFixed(2)}</output>
        <input type="range" id="sens" min="0" max="1" step="0.05" value="${sensitivity}"></label>
      <div class="stats">
        <span><b>${m.recall !== undefined ? `${Math.round(m.recall * 100)}%` : "–"}</b><span class="muted">caught (test voices)</span></span>
        <span><b>${m.fa_per_hour !== undefined ? m.fa_per_hour.toFixed(2) : "–"}</b><span class="muted">false alarms / hour</span></span>
        <span><b>${elapsed} min</b><span class="muted">to train</span></span>
      </div>
      <div class="actions">
        <a href="/api/jobs/${job.id}/download"><button class="primary">Download</button></a>
        <button id="again">Train another</button>
      </div>`;
    $("mic").onclick = () => void toggleListening(job);
    $("sens").oninput = async (e) => {
      sensitivity = Number((e.target as HTMLInputElement).value);
      $("sens-out").textContent = sensitivity.toFixed(2);
      if (listener) {
        await stopListening();
        await toggleListening(job);
      }
    };
    $("again").onclick = () => {
      phraseInput.value = "";
      phraseInput.focus();
      phraseInput.dispatchEvent(new Event("input"));
    };
  } else {
    stage.innerHTML = `
      <div class="mic">${MIC}</div>
      <div class="${job.status === "failed" ? "error" : "muted"}"><b>${job.status === "failed" ? "Training failed" : "Cancelled"}</b></div>
      ${job.error ? `<pre class="log">${esc(job.error)}</pre>` : ""}`;
  }
}

// ---- in-browser test

async function toggleListening(job: Job): Promise<void> {
  if (listener) return stopListening();
  const mic = $("mic");
  try {
    $("hint").textContent = "Loading the model…";
    pkg ??= await ModelPackage.fromUrl(job.package_url!);
    const keywords = pkg.keywordOptions([job.phrase_id], [sensitivity]);
    const [model, vad] = await Promise.all([OnnxWakeWordModel.create(pkg, [job.phrase_id]), SileroVad.fromUrl("/vad/silero_vad.onnx")]);
    listener = new WakeWordListener(model, vad, { keywords });
    $("mark").style.left = `${(keywords[0]!.detector.baseThreshold * 100).toFixed(1)}%`;
    $("score").hidden = false;
    listener.onscores = ({ scores, vadOpen }) => {
      $("fill").style.width = `${(scores[0]! * 100).toFixed(1)}%`;
      mic.style.setProperty("--level", vadOpen ? "1" : "0");
    };
    listener.ondetect = () => {
      $("heard").textContent = `Heard "${job.phrase}"`;
      mic.classList.remove("heard");
      void mic.offsetWidth; // restart the animation
      mic.classList.add("heard");
      setTimeout(() => { $("heard").textContent = ""; }, 1500);
    };
    await listener.start();
    mic.classList.add("listening");
    $("hint").textContent = `Listening. Say "${job.phrase}". The first two seconds warm up. Click again to stop.`;
  } catch (err) {
    listener = null;
    $("hint").textContent = `Couldn't start the microphone: ${(err as Error).message}`;
  }
}

async function stopListening(): Promise<void> {
  const l = listener;
  listener = null;
  await l?.dispose();
  document.getElementById("mic")?.classList.remove("listening");
  const hint = document.getElementById("hint");
  if (hint && current) hint.textContent = `Click the microphone and say "${current.phrase}".`;
}

// ---- start

function notice(text: string): void {
  const n = $("notice");
  n.textContent = text;
  n.hidden = false;
  setTimeout(() => { n.hidden = true; }, 8000);
}

async function init(): Promise<void> {
  try {
    const status = await api<{ ready: boolean; missing: string[]; negative_hours: Record<string, number> }>("/api/status");
    if (!status.ready) {
      const n = $("notice");
      n.innerHTML = `<b>Studio isn't ready:</b> the base folder is missing ${status.missing.map(esc).join(", ")}. Point <code>--base</code> at a finished training run.`;
      n.hidden = false;
    }
  } catch {
    notice("Can't reach the Studio server. Start it with: python -m studio.app --base <folder>");
    return;
  }
  $("stage").innerHTML = `<div class="mic">${MIC}</div><div class="muted">Type a wake phrase and press Train.</div>`;
  await refreshJobs();
  const id = /job=([\w-]+)/.exec(location.hash)?.[1];
  if (id) await open(id).catch(() => undefined);
}

void init();
