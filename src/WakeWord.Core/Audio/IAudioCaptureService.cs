namespace WakeWord.Core.Audio;

/// <summary>Receives captured samples. The span is only valid for the duration of the call.</summary>
public delegate void AudioSamplesHandler(ReadOnlySpan<short> samples);

/// <summary>
/// Continuous microphone capture, implemented once per platform.
/// Delivers 16 kHz mono 16-bit PCM with OS voice processing (AEC, NS, AGC) turned off,
/// using the per-platform settings in plan 6.1, so the signal matches training.
/// </summary>
public interface IAudioCaptureService : IAsyncDisposable
{
    event AudioSamplesHandler? SamplesCaptured;

    bool IsCapturing { get; }

    Task StartAsync(CancellationToken cancellationToken = default);

    Task StopAsync();
}
