namespace WakeWord.Core.Detection;

/// <summary>
/// The three-stage openWakeWord-style model: melspectrogram, embedding, then one classifier per keyword.
/// Split in two calls so the cheap mel stage runs on every frame and the rest only behind the VAD gate.
/// </summary>
public interface IWakeWordModel : IDisposable
{
    /// <summary>Samples per model frame; 1,280 (80 ms at 16 kHz) for openWakeWord.</summary>
    int FrameSamples { get; }

    /// <summary>Keyword ids, in the order <see cref="Score"/> writes them.</summary>
    IReadOnlyList<string> Keywords { get; }

    /// <summary>Runs the melspectrogram stage. Called for every frame so context stays warm.</summary>
    void AppendFrame(ReadOnlySpan<short> frame);

    /// <summary>
    /// Runs the embedding over every frame not yet embedded, then each keyword's classifier, writing one
    /// score in [0, 1] per keyword. After the gate has been closed this back-fills embeddings from buffered
    /// mel frames, so the start of the wake phrase is not lost to VAD onset lag.
    /// Returns false (scores untouched) while the model is still warming up after start.
    /// </summary>
    bool Score(Span<double> scores);
}

/// <summary>Voice activity detector such as Silero VAD.</summary>
public interface IVoiceActivityDetector : IDisposable
{
    /// <summary>Speech probability in [0, 1] for one model frame.</summary>
    double SpeechProbability(ReadOnlySpan<short> frame);
}
