import {
  CachingTokenProvider,
  ModelPackage,
  OnnxWakeWordModel,
  SileroVad,
  WakeWordListener,
  brokerTokenSource,
  configureOnnxRuntime,
  isErrorReason,
} from "../src/index.js";

configureOnnxRuntime(); // Vite bundles the .wasm next to the app; no CDN needed

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const log = (line: string) => {
  const el = $("log");
  el.textContent = `${new Date().toLocaleTimeString()}  ${line}\n${el.textContent}`;
};

let pkg: ModelPackage | null = null;
let listener: WakeWordListener | null = null;

async function loadPackage(): Promise<void> {
  const url = $<HTMLSelectElement>("package").value;
  $("model-status").textContent = `Loading ${url}…`;
  try {
    pkg = await ModelPackage.fromUrl(url);
    const notes = pkg.manifest.notes.join(" ");
    $("model-status").innerHTML = `Loaded <strong>${pkg.manifest.run}</strong>: ${pkg.keywords.map((k) => k.phrase).join(", ")}.` +
      (url.includes("smoke") ? ' <span class="warn">This package is a tiny test model and will not detect real speech.</span>' : ` ${notes}`);
    $("sensitivities").innerHTML = pkg.keywords.map((k) => `
      <label>${k.phrase} sensitivity: <output id="out-${k.id}">${k.default_sensitivity}</output>
        <input type="range" id="sens-${k.id}" min="0" max="1" step="0.05" value="${k.default_sensitivity}">
      </label>`).join("");
    for (const k of pkg.keywords) {
      $<HTMLInputElement>(`sens-${k.id}`).oninput = (e) => { $(`out-${k.id}`).textContent = (e.target as HTMLInputElement).value; };
    }
  } catch (err) {
    pkg = null;
    $("model-status").innerHTML = `<span class="warn">${(err as Error).message}. Unzip the Colab download into models/wakeword, or pick the smoke-test package.</span>`;
  }
}

$<HTMLSelectElement>("package").onchange = () => void loadPackage();

$("start").onclick = async () => {
  if (!pkg) return;
  $<HTMLButtonElement>("start").disabled = true;
  try {
    const keywords = pkg.keywords.map((k) => k.id);
    const sensitivities = pkg.keywords.map((k) => Number($<HTMLInputElement>(`sens-${k.id}`).value));
    const options = pkg.keywordOptions(keywords, sensitivities);
    const broker = $<HTMLInputElement>("broker").value.trim();
    const devUser = $<HTMLInputElement>("devuser").value.trim();
    const tokens = broker
      ? new CachingTokenProvider(brokerTokenSource(broker, (): Record<string, string> => (devUser ? { "X-Dev-User": devUser } : {})))
      : undefined;

    const [model, vad] = await Promise.all([OnnxWakeWordModel.create(pkg, keywords), SileroVad.fromUrl("/models/vad/silero_vad.onnx")]);
    listener = new WakeWordListener(model, vad, { keywords: options, tokens, deepgram: { keyterms: ["UNO"] } });

    $("meters").innerHTML = options.map((k) => `
      <div class="kw" id="kw-${k.id}">
        <div class="kw-head"><span>${k.phrase}</span><span id="score-${k.id}" class="muted">0.000</span></div>
        <div class="bar"><div class="fill" id="fill-${k.id}"></div><div class="mark" style="left:${(k.detector.baseThreshold * 100).toFixed(1)}%"></div></div>
      </div>`).join("");

    listener.onscores = ({ scores, vadOpen, noiseFloorDbfs }) => {
      $("vad").textContent = vadOpen ? "speech" : "silence";
      $("vad").classList.toggle("on", vadOpen);
      $("noise").textContent = `noise floor ${noiseFloorDbfs.toFixed(0)} dBFS`;
      options.forEach((k, i) => {
        $(`fill-${k.id}`).style.width = `${(scores[i]! * 100).toFixed(1)}%`;
        $(`score-${k.id}`).textContent = scores[i]!.toFixed(3);
      });
    };
    listener.ondetect = (d) => {
      log(`Heard "${options[d.keywordIndex]!.phrase}" (score ${d.score.toFixed(3)}, threshold ${d.threshold.toFixed(3)})`);
      const el = $(`kw-${d.keyword}`);
      el.classList.add("hit");
      setTimeout(() => el.classList.remove("hit"), 1200);
    };
    listener.onspeechstart = () => log("Speech started");
    listener.onreject = (r) => log(r === "no-token-provider" ? "Wake word heard; add a token broker to stream to Deepgram." : `Session not started: ${r}`);
    listener.onsessionstart = (t) => { $("session").textContent = `Streaming to Deepgram (${t})…`; $("transcript").textContent = ""; };
    listener.ontranscript = (u) => { $("transcript").textContent = u.text; };
    listener.onsessionend = ({ result }) => {
      $("session").textContent = "";
      log(`Session ended: ${result.reason}, ${result.audioSentSeconds.toFixed(1)} s sent, connected in ${result.connectLatencyMs} ms` +
        (result.transcript ? `: "${result.transcript}"` : ""));
      if (result.reason === "not-confirmed") log("False wake: Deepgram did not hear the wake phrase, so the session was closed early.");
      if (isErrorReason(result.reason)) $("session").innerHTML = `<span class="warn">Speech service error (${result.reason}); try again.</span>`;
    };

    await listener.start();
    log("Listening");
    $<HTMLButtonElement>("stop").disabled = false;
    $<HTMLButtonElement>("talk").disabled = !tokens;
  } catch (err) {
    log(`Could not start: ${(err as Error).message}`);
    $<HTMLButtonElement>("start").disabled = false;
  }
};

$("stop").onclick = async () => {
  await listener?.dispose();
  listener = null;
  log("Stopped");
  $<HTMLButtonElement>("start").disabled = false;
  $<HTMLButtonElement>("stop").disabled = true;
  $<HTMLButtonElement>("talk").disabled = true;
};

$("talk").onclick = () => {
  if (listener?.isStreaming) listener.stopSession();
  else listener?.startManualSession();
};

void loadPackage();
