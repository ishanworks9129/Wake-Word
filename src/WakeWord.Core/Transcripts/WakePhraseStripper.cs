using System.Text.RegularExpressions;

namespace WakeWord.Core.Transcripts;

/// <summary>
/// Finds and removes the wake phrase near the start of a transcript (plan 8.4). The phrase audio is sent
/// to Deepgram on purpose, so it is matched as text, fuzzily, because Deepgram may hear "Hey UNO" as
/// "hey you know" or "hey Juno". A few words before it (a filler, or the end of what was said just before,
/// caught by the pre-roll) are removed with it.
/// </summary>
public sealed partial class WakePhraseStripper
{
    private readonly string[] _targets;
    private readonly int _maxTokens;
    private readonly int _maxLeadingWords;
    private readonly double _maxDistanceRatio;

    /// <param name="phrase">The wake phrase, e.g. "Hey UNO".</param>
    /// <param name="variants">Known mis-hearings, e.g. "hey you know", "hey juno".</param>
    /// <param name="maxDistanceRatio">Edit distance allowed, as a fraction of the target's length.</param>
    /// <param name="maxLeadingWords">Words allowed before the phrase.</param>
    public WakePhraseStripper(string phrase, IEnumerable<string>? variants = null, double maxDistanceRatio = 0.25, int maxLeadingWords = 3)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(phrase);
        var all = new[] { phrase }.Concat(variants ?? []).ToArray();
        _targets = all.Select(Squash).Where(t => t.Length > 0).Distinct().ToArray();
        _maxTokens = all.Max(p => Words().Count(p)) + 2;
        _maxLeadingWords = maxLeadingWords;
        _maxDistanceRatio = maxDistanceRatio;
    }

    /// <summary>Once a transcript has this many words without the phrase, the phrase is not coming.</summary>
    public int WordsToDecide => _maxLeadingWords + _maxTokens;

    public string Strip(string transcript)
    {
        var end = Find(transcript);
        return end < 0 ? transcript.Trim() : transcript[end..].TrimStart(' ', ',', '.', '!', '?', ';', ':', '-', '—').TrimEnd();
    }

    /// <summary>True if the wake phrase (or a known mis-hearing) is among the first words.</summary>
    public bool Contains(string transcript) => Find(transcript) >= 0;

    public static int CountWords(string transcript) => Words().Count(transcript);

    /// <summary>Index just after the phrase, or -1.</summary>
    private int Find(string transcript)
    {
        var words = Words().Matches(transcript);
        for (var skip = 0; skip <= _maxLeadingWords && skip < words.Count; skip++)
        {
            var cut = Match(words, skip);
            if (cut >= 0)
            {
                var last = words[skip + cut - 1];
                return last.Index + last.Length;
            }
        }

        return -1;
    }

    /// <summary>Number of words from <paramref name="start"/> that best match the phrase, or -1.</summary>
    private int Match(MatchCollection words, int start)
    {
        var bestCount = -1;
        var bestRatio = double.MaxValue;
        var candidate = string.Empty;
        for (var k = 1; k <= _maxTokens && start + k <= words.Count; k++)
        {
            candidate += Squash(words[start + k - 1].Value);
            foreach (var target in _targets)
            {
                var ratio = (double)Levenshtein(candidate, target) / target.Length;
                if (ratio < bestRatio)
                {
                    (bestRatio, bestCount) = (ratio, k);
                }
            }
        }

        return bestRatio <= _maxDistanceRatio ? bestCount : -1;
    }

    private static string Squash(string text) =>
        string.Concat(text.ToLowerInvariant().Where(char.IsLetterOrDigit));

    private static int Levenshtein(string a, string b)
    {
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];
        for (var j = 0; j <= b.Length; j++)
        {
            previous[j] = j;
        }

        for (var i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (var j = 1; j <= b.Length; j++)
            {
                var cost = a[i - 1] == b[j - 1] ? 0 : 1;
                current[j] = Math.Min(Math.Min(current[j - 1] + 1, previous[j] + 1), previous[j - 1] + cost);
            }

            (previous, current) = (current, previous);
        }

        return previous[b.Length];
    }

    [GeneratedRegex(@"[\p{L}\p{N}']+")]
    private static partial Regex Words();
}
