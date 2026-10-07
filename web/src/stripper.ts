const WORDS = /[\p{L}\p{N}']+/gu;
const TRAILING = /^[\s,.!?;:\-—]+/;

export const countWords = (transcript: string) => (transcript.match(WORDS) ?? []).length;

/**
 * Finds and removes the wake phrase (or a known mis-hearing such as "hey you know") near the start of a
 * transcript, with up to maxLeadingWords before it (a filler, or speech just before it caught by the pre-roll).
 * Same rules as WakeWord.Core.Transcripts.WakePhraseStripper.
 */
export class WakePhraseStripper {
  private readonly targets: string[];
  private readonly maxTokens: number;

  constructor(phrases: string[], private readonly maxDistanceRatio = 0.25, private readonly maxLeadingWords = 3) {
    if (phrases.length === 0) throw new RangeError("At least one phrase is required.");
    this.targets = [...new Set(phrases.map(squash).filter((t) => t.length > 0))];
    this.maxTokens = Math.max(...phrases.map((p) => (p.match(WORDS) ?? []).length)) + 2;
  }

  /** Once a transcript has this many words without the phrase, the phrase is not coming. */
  get wordsToDecide(): number {
    return this.maxLeadingWords + this.maxTokens;
  }

  strip(transcript: string): string {
    const end = this.find(transcript);
    return end < 0 ? transcript.trim() : transcript.slice(end).replace(TRAILING, "").trimEnd();
  }

  /** True if the wake phrase (or a known mis-hearing) is among the first words. */
  contains(transcript: string): boolean {
    return this.find(transcript) >= 0;
  }

  /** Index just after the phrase, or -1. */
  private find(transcript: string): number {
    const words = [...transcript.matchAll(WORDS)];
    for (let skip = 0; skip <= this.maxLeadingWords && skip < words.length; skip++) {
      const cut = this.match(words, skip);
      if (cut >= 0) {
        const last = words[skip + cut - 1]!;
        return last.index! + last[0].length;
      }
    }
    return -1;
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
