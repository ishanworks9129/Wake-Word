import { SensitivityTable, type DetectorOptions } from "./detector.js";

export const PACKAGE_FORMAT = "wakeword-models/1";

/** models.json, as written by training/wakeword_train/export.py. */
export interface PackageManifest {
  format: string;
  run: string;
  sample_rate: number;
  frame_samples: number;
  features: {
    melspectrogram: string;
    embedding: string;
    mel_context_samples: number;
    embedding_window: number;
    embedding_step: number;
    classifier_frames: number;
  };
  keywords: KeywordManifest[];
  notes: string[];
}

export interface KeywordManifest {
  id: string;
  phrase: string;
  model: string;
  input: string;
  output: string;
  default_sensitivity: number;
  sensitivity_table: [number, number][];
  detector: { consecutive_frames: number; refractory_seconds: number; min_threshold?: number; max_threshold?: number };
  transcript_variants: string[];
}

export interface KeywordOptions {
  id: string;
  phrase: string;
  detector: DetectorOptions;
  transcriptVariants: string[];
}

export type FileLoader = (name: string) => Promise<Uint8Array>;

/** A trained model package: models.json plus the feature models and one classifier per keyword. */
export class ModelPackage {
  private constructor(readonly manifest: PackageManifest, private readonly files: Map<string, Uint8Array>) {}

  /** Loads from a base URL, e.g. "/models/wakeword/". */
  static fromUrl(baseUrl: string, fetchImpl: typeof fetch = fetch): Promise<ModelPackage> {
    const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
    return ModelPackage.load(async (name) => {
      const res = await fetchImpl(base + name);
      if (!res.ok) throw new Error(`Could not load ${base + name}: HTTP ${res.status}`);
      return new Uint8Array(await res.arrayBuffer());
    });
  }

  static async load(loader: FileLoader): Promise<ModelPackage> {
    const manifest = JSON.parse(new TextDecoder().decode(await loader("models.json"))) as PackageManifest;
    if (manifest.format !== PACKAGE_FORMAT) {
      throw new Error(`Unsupported model package format '${manifest.format}'; expected '${PACKAGE_FORMAT}'.`);
    }
    if (manifest.sample_rate !== 16000 || manifest.frame_samples !== 1280) {
      throw new Error(`Package is ${manifest.sample_rate} Hz / ${manifest.frame_samples}-sample frames; this runtime needs 16000 / 1280.`);
    }
    const names = new Set([manifest.features.melspectrogram, manifest.features.embedding, ...manifest.keywords.map((k) => k.model)]);
    const files = new Map<string, Uint8Array>();
    await Promise.all(
      [...names].map(async (name) => {
        if (/[\\/]|\.\./.test(name)) throw new Error(`Model file names must be plain file names; got '${name}'.`);
        files.set(name, await loader(name));
      }),
    );
    return new ModelPackage(manifest, files);
  }

  get keywords(): KeywordManifest[] {
    return this.manifest.keywords;
  }

  file(name: string): Uint8Array {
    const f = this.files.get(name);
    if (!f) throw new Error(`File '${name}' is not in the package.`);
    return f;
  }

  keyword(id: string): KeywordManifest {
    const k = this.manifest.keywords.find((x) => x.id === id);
    if (!k) throw new Error(`Keyword '${id}' is not in this package; it has ${this.manifest.keywords.map((x) => x.id).join(", ")}.`);
    return k;
  }

  /** Detector settings for the chosen keywords at the given sensitivities (0..1; package defaults when omitted). */
  keywordOptions(keywords?: string[], sensitivities?: number[]): KeywordOptions[] {
    const chosen = (keywords ?? this.manifest.keywords.map((k) => k.id)).map((id) => this.keyword(id));
    if (sensitivities && sensitivities.length !== chosen.length) {
      throw new RangeError(`Got ${sensitivities.length} sensitivities for ${chosen.length} keywords.`);
    }
    return chosen.map((k, i) => {
      const sensitivity = sensitivities?.[i] ?? k.default_sensitivity;
      if (!(sensitivity >= 0 && sensitivity <= 1)) throw new RangeError(`Sensitivity must be between 0 and 1; got ${sensitivity}.`);
      return {
        id: k.id,
        phrase: k.phrase,
        transcriptVariants: k.transcript_variants,
        detector: {
          baseThreshold: new SensitivityTable(k.sensitivity_table).thresholdFor(sensitivity),
          consecutiveFrames: k.detector.consecutive_frames,
          refractorySeconds: k.detector.refractory_seconds,
          adaptive: { points: [], minThreshold: k.detector.min_threshold ?? 0.01, maxThreshold: k.detector.max_threshold ?? 0.9999 },
        },
      };
    });
  }
}
