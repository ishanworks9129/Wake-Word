const WORDS = /[\p{L}\p{N}']+/gu;
const FILLERS = new Set(["um", "uh", "er", "ah", "oh", "so"]);
const TRAILING = /^[\s,.!?;:\-—]+/;

/**
 * Removes the wake phrase (or a known mis-hearing such as "hey you know") from the start of a transcript.
 * Same rules as WakeWord.Core.Transcripts.WakePhraseStripper.
 */
export class WakePhraseStripper {
  private readonly targets: string[];
  private readonly maxTokens: number;

  constructor(phrases: string[], private readonly maxDistanceRatio = 0.25) {
    if (phrases.length === 0) throw new RangeError("At least one phrase is required.");
    this.targets = [...new Set(phrases.map(squash).filter((t) => t.length > 0))];
    this.maxTokens = Math.max(...phrases.map((p) => (p.match(WORDS) ?? []).length)) + 2;
  }

  strip(transcript: string): string {
    const words = [...transcript.matchAll(WORDS)];
    if (words.length === 0) return transcript.trim();
    let skip = 0;
    let cut = this.match(words, 0);
    if (cut < 0 && FILLERS.has(words[0]![0].toLowerCase())) {
      skip = 1;
      cut = this.match(words, 1);
    }
    if (cut < 0) return transcript.trim();
    const last = words[skip + cut - 1]!;
    return transcript.slice(last.index! + last[0].length).replace(TRAILING, "").trimEnd();
  }

  private match(words: RegExpMatchArray[], start: number): number {
    let bestCount = -1;
    let bestRatio = Number.POSITIVE_INFINITY;
    let candidate = "";
    for (let k = 1; k <= this.maxTokens && start + k <= words.length; k++) {
      candidate += squash(words[start + k - 1]![0]);
      for (const target of this.targets) {
        const ratio = levenshtein(candidate, target) / target.length;
        if (ratio < bestRatio) {
          bestRatio = ratio;
          bestCount = k;
        }
      }
    }
    return bestRatio <= this.maxDistanceRatio ? bestCount : -1;
  }
}

function squash(text: string): string {
  return [...text.toLowerCase()].filter((c) => /[\p{L}\p{N}]/u.test(c)).join("");
}

function levenshtein(a: string, b: string): number {
  let previous = Array.from({ length: b.length + 1 }, (_, j) => j);
  let current = new Array<number>(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    current[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(current[j - 1]! + 1, previous[j]! + 1, previous[j - 1]! + cost);
    }
    [previous, current] = [current, previous];
  }
  return previous[b.length]!;
}
