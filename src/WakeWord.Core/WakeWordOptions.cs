using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;

namespace WakeWord.Core;

/// <summary>One wake phrase and how it fires. Usually built from a model package (WakeWord.Onnx.ModelPackage).</summary>
public sealed record KeywordOptions
{
    public required string Id { get; init; }

    /// <summary>What users say, e.g. "Hey UNO". Stripped from transcripts.</summary>
    public required string Phrase { get; init; }

    public WakeWordDetectorOptions Detector { get; init; } = new();

    /// <summary>Known Deepgram mis-hearings of the phrase, stripped like the phrase itself.</summary>
    public IReadOnlyList<string> TranscriptVariants { get; init; } = [];
}

/// <summary>Everything tunable in the pipeline, bindable from configuration or a downloaded model manifest.</summary>
public sealed record WakeWordOptions
{
    public const int SampleRate = 16_000;

    /// <summary>One entry per keyword, in the model's keyword order.</summary>
    public required IReadOnlyList<KeywordOptions> Keywords { get; init; }

    public TimeSpan PreRoll { get; init; } = TimeSpan.FromSeconds(2);

    public TimeSpan RingCapacity { get; init; } = TimeSpan.FromSeconds(10);

    public int MaxSessionsPerHour { get; init; } = 60;

    /// <summary>
    /// End a wake-word session within about a second when Deepgram's transcript does not contain the wake phrase
    /// (the detector fired on something else), and show no text until it does. Tap-to-talk sessions are never checked.
    /// </summary>
    public bool ConfirmWakePhrase { get; init; } = true;

    public VadGateOptions Vad { get; init; } = new();

    public DeepgramStreamingOptions Deepgram { get; init; } = new();

    public DeepgramSessionLimits Session { get; init; } = new();
}
