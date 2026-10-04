namespace WakeWord.Core.Detection;

public sealed record VadGateOptions
{
    public double OnsetProbability { get; init; } = 0.5;

    /// <summary>While open, frames at or above this keep the gate open (hysteresis).</summary>
    public double SustainProbability { get; init; } = 0.35;

    /// <summary>Inference continues this long after speech drops (plan 6.2).</summary>
    public TimeSpan Hangover { get; init; } = TimeSpan.FromSeconds(1);
}

public enum VadTransition
{
    None,
    Opened,
    Closed,
}

/// <summary>Speech gate with hysteresis and hangover over per-frame VAD probabilities.</summary>
public sealed class VadGate(VadGateOptions options, int sampleRate)
{
    private readonly long _hangoverSamples = (long)Math.Round(options.Hangover.TotalSeconds * sampleRate);
    private long _lastSpeechSample;

    public bool IsOpen { get; private set; }

    public VadTransition Update(double speechProbability, long frameEndSample)
    {
        if (!IsOpen)
        {
            if (speechProbability < options.OnsetProbability)
            {
                return VadTransition.None;
            }

            IsOpen = true;
            _lastSpeechSample = frameEndSample;
            return VadTransition.Opened;
        }

        if (speechProbability >= options.SustainProbability)
        {
            _lastSpeechSample = frameEndSample;
            return VadTransition.None;
        }

        if (frameEndSample - _lastSpeechSample > _hangoverSamples)
        {
            IsOpen = false;
            return VadTransition.Closed;
        }

        return VadTransition.None;
    }
}
