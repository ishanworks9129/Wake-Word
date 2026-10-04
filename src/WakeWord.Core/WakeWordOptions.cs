using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;

namespace WakeWord.Core;

/// <summary>Everything tunable in the pipeline, bindable from configuration or a downloaded model manifest.</summary>
public sealed record WakeWordOptions
{
    public const int SampleRate = 16_000;

    /// <summary>Must match the phrase the model was trained on (plan 2).</summary>
    public string WakePhrase { get; init; } = "Hey UNO";

    /// <summary>Known mis-hearings, stripped from transcripts like the phrase itself.</summary>
    public IReadOnlyList<string> WakePhraseVariants { get; init; } = [];

    public TimeSpan PreRoll { get; init; } = TimeSpan.FromSeconds(2);

    public TimeSpan RingCapacity { get; init; } = TimeSpan.FromSeconds(10);

    public int MaxSessionsPerHour { get; init; } = 60;

    public WakeWordDetectorOptions Detector { get; init; } = new();

    public VadGateOptions Vad { get; init; } = new();

    public DeepgramStreamingOptions Deepgram { get; init; } = new();

    public DeepgramSessionLimits Session { get; init; } = new();
}
