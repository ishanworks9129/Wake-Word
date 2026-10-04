namespace WakeWord.Core.Detection;

/// <summary>
/// The three-stage openWakeWord-style model: melspectrogram, embedding, classifier.
/// Split in two calls so the cheap mel stage runs on every frame and the rest only behind the VAD gate.
/// </summary>
public interface IWakeWordModel : IDisposable
{
    /// <summary>Samples per model frame; 1,280 (80 ms at 16 kHz) for openWakeWord.</summary>
    int FrameSamples { get; }

    /// <summary>Runs the melspectrogram stage. Called for every frame so context stays warm.</summary>
    void AppendFrame(ReadOnlySpan<short> frame);

    /// <summary>
    /// Runs embedding and classifier over every mel frame not yet embedded and returns the
    /// latest score in [0, 1]. After the gate has been closed this back-fills embeddings from
    /// buffered mel frames, so the start of the wake phrase is not lost to VAD onset lag.
    /// </summary>
    double Score();
}

/// <summary>Voice activity detector such as Silero VAD.</summary>
public interface IVoiceActivityDetector : IDisposable
{
    /// <summary>Speech probability in [0, 1] for one model frame.</summary>
    double SpeechProbability(ReadOnlySpan<short> frame);
}
