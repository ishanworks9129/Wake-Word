namespace WakeWord.Core.Audio;

public static class AudioLevel
{
    /// <summary>Level reported for digital silence, instead of negative infinity.</summary>
    public const double SilenceDbfs = -100;

    /// <summary>RMS level of 16-bit PCM in dB relative to full scale.</summary>
    public static double Dbfs(ReadOnlySpan<short> samples)
    {
        if (samples.IsEmpty)
        {
            return SilenceDbfs;
        }

        double sumSquares = 0;
        foreach (var s in samples)
        {
            sumSquares += (double)s * s;
        }

        var rms = Math.Sqrt(sumSquares / samples.Length) / short.MaxValue;
        return rms <= 0 ? SilenceDbfs : Math.Max(SilenceDbfs, 20 * Math.Log10(rms));
    }
}
