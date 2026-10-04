namespace WakeWord.Core.Detection;

public sealed record ThresholdPoint(double NoiseFloorDbfs, double Offset);

/// <summary>
/// Maps the ambient noise floor to a threshold offset (plan 6.4). With no points the
/// threshold is fixed. Points are calibrated per noise band in Phase 3.
/// </summary>
public sealed record AdaptiveThresholdOptions
{
    public IReadOnlyList<ThresholdPoint> Points { get; init; } = [];

    public double MinThreshold { get; init; } = 0.2;

    public double MaxThreshold { get; init; } = 0.95;
}

public sealed record WakeWordDetectorOptions
{
    public double BaseThreshold { get; init; } = 0.5;

    /// <summary>Frames in a row above threshold before firing (N in plan 6.4).</summary>
    public int ConsecutiveFrames { get; init; } = 3;

    /// <summary>New triggers are ignored for this long after a fire.</summary>
    public TimeSpan Refractory { get; init; } = TimeSpan.FromSeconds(1.5);

    public AdaptiveThresholdOptions Adaptive { get; init; } = new();
}

public sealed record WakeWordDetection(long FrameEndSample, double Score, double Threshold);

public static class AdaptiveThreshold
{
    /// <summary>
    /// Base threshold plus a piecewise-linear offset, clamped. Below the first point the first
    /// offset applies; above the last point the last offset applies.
    /// Mirrored exactly by training/eval/detector.py.
    /// </summary>
    public static double For(double baseThreshold, double noiseFloorDbfs, AdaptiveThresholdOptions options)
    {
        var points = options.Points;
        double offset = 0;
        if (points.Count > 0)
        {
            if (noiseFloorDbfs <= points[0].NoiseFloorDbfs)
            {
                offset = points[0].Offset;
            }
            else if (noiseFloorDbfs >= points[^1].NoiseFloorDbfs)
            {
                offset = points[^1].Offset;
            }
            else
            {
                for (var i = 1; i < points.Count; i++)
                {
                    var (lo, hi) = (points[i - 1], points[i]);
                    if (noiseFloorDbfs <= hi.NoiseFloorDbfs)
                    {
                        var t = (noiseFloorDbfs - lo.NoiseFloorDbfs) / (hi.NoiseFloorDbfs - lo.NoiseFloorDbfs);
                        offset = lo.Offset + t * (hi.Offset - lo.Offset);
                        break;
                    }
                }
            }
        }

        return Math.Clamp(baseThreshold + offset, options.MinThreshold, options.MaxThreshold);
    }
}

/// <summary>
/// Turns per-frame classifier scores into wake-word fires: adaptive threshold,
/// N consecutive frames, then a refractory period. Not thread-safe; called from the capture thread.
/// </summary>
public sealed class WakeWordDetector
{
    private readonly WakeWordDetectorOptions _options;
    private readonly long _refractorySamples;
    private int _run;
    private long? _lastFireSample;

    public WakeWordDetector(WakeWordDetectorOptions options, int sampleRate)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(options.ConsecutiveFrames, 1);
        _options = options;
        _refractorySamples = (long)Math.Round(options.Refractory.TotalSeconds * sampleRate);
    }

    public double CurrentThreshold { get; private set; }

    public WakeWordDetection? Process(double score, double noiseFloorDbfs, long frameEndSample)
    {
        var threshold = AdaptiveThreshold.For(_options.BaseThreshold, noiseFloorDbfs, _options.Adaptive);
        CurrentThreshold = threshold;

        if (_lastFireSample is { } last && frameEndSample - last < _refractorySamples)
        {
            _run = 0;
            return null;
        }

        if (score < threshold)
        {
            _run = 0;
            return null;
        }

        if (++_run < _options.ConsecutiveFrames)
        {
            return null;
        }

        _run = 0;
        _lastFireSample = frameEndSample;
        return new WakeWordDetection(frameEndSample, score, threshold);
    }

    /// <summary>Breaks the current run, e.g. when the VAD gate closes.</summary>
    public void Reset() => _run = 0;
}
