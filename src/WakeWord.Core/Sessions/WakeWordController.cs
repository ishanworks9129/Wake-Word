using WakeWord.Core.Audio;
using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;
using WakeWord.Core.Transcripts;

namespace WakeWord.Core.Sessions;

public enum SessionTrigger
{
    WakeWord,

    /// <summary>The mandatory tap-to-talk fallback (plan 9.2).</summary>
    Manual,
}

public enum SessionRejection
{
    /// <summary>A session is already streaming.</summary>
    Busy,

    RateLimited,
}

public sealed record SessionTelemetry(
    SessionTrigger Trigger,
    string? Keyword,
    double? DetectionScore,
    SessionEndReason Reason,
    TimeSpan AudioSent,
    TimeSpan ConnectLatency);

/// <summary>Receives one record per session (plan 8.6). Sessions closed by <see cref="SessionEndReason.NoTranscript"/> are the false-accept proxy.</summary>
public interface ISessionTelemetrySink
{
    void SessionCompleted(SessionTelemetry record);
}

/// <summary>
/// Wires the always-on engine to Deepgram sessions: prefetches a token on speech onset, starts at most
/// one session at a time on a wake-word fire or a tap, applies the per-device rate limit, and reports telemetry.
/// Feed captured audio to <see cref="Engine"/>.
/// </summary>
public sealed class WakeWordController : IAsyncDisposable
{
    private readonly WakeWordOptions _options;
    private readonly PcmRingBuffer _ring;
    private readonly ITokenProvider _tokens;
    private readonly Func<IDeepgramTransport> _transportFactory;
    private readonly SessionRateLimiter _limiter;
    private readonly ISessionTelemetrySink? _telemetry;
    private readonly WakePhraseStripper _stripper;
    private readonly TimeProvider _time;
    private readonly CancellationTokenSource _disposed = new();
    private readonly Lock _gate = new();
    private Task? _current;
    private CancellationTokenSource? _currentCts;

    public WakeWordController(
        WakeWordOptions options,
        IWakeWordModel model,
        IVoiceActivityDetector vad,
        ITokenProvider tokens,
        Func<IDeepgramTransport>? transportFactory = null,
        ISessionTelemetrySink? telemetry = null,
        TimeProvider? time = null)
    {
        _options = options;
        _tokens = tokens;
        _transportFactory = transportFactory ?? (() => new ClientWebSocketTransport());
        _telemetry = telemetry;
        _time = time ?? TimeProvider.System;
        _ring = new PcmRingBuffer(WakeWordOptions.SampleRate, options.RingCapacity);
        _limiter = new SessionRateLimiter(options.MaxSessionsPerHour, TimeSpan.FromHours(1), _time);
        if (!options.Keywords.Select(k => k.Id).SequenceEqual(model.Keywords))
        {
            throw new ArgumentException(
                $"Options keywords [{string.Join(", ", options.Keywords.Select(k => k.Id))}] must match the model's [{string.Join(", ", model.Keywords)}], in order.",
                nameof(options));
        }

        var phrases = options.Keywords.SelectMany(k => k.TranscriptVariants.Prepend(k.Phrase)).ToArray();
        _stripper = new WakePhraseStripper(phrases[0], phrases[1..]);

        Engine = new WakeWordEngine(model, vad, _ring, options.Keywords.Select(k => k.Detector).ToList(), options.Vad);
        Engine.SpeechStarted += (_, _) => _tokens.Prefetch();
        Engine.Detected += (_, detection) => TryStart(
            SessionTrigger.WakeWord, detection.FrameEndSample - _ring.SamplesFor(_options.PreRoll), detection);
    }

    public WakeWordEngine Engine { get; }

    public bool IsStreaming
    {
        get { lock (_gate) return _current is { IsCompleted: false }; }
    }

    public event EventHandler<SessionTrigger>? SessionStarted;

    public event EventHandler<TranscriptUpdate>? TranscriptUpdated;

    public event EventHandler<DeepgramSessionResult>? SessionEnded;

    public event EventHandler<SessionRejection>? SessionRejected;

    /// <summary>Tap-to-talk: streams from now, with no pre-roll.</summary>
    public bool StartManualSession() => TryStart(SessionTrigger.Manual, _ring.TotalWritten, detection: null);

    /// <summary>Ends the current session early, e.g. when the user taps stop.</summary>
    public void StopSession()
    {
        lock (_gate)
        {
            _currentCts?.Cancel();
        }
    }

    public async ValueTask DisposeAsync()
    {
        await _disposed.CancelAsync().ConfigureAwait(false);
        Task? current;
        lock (_gate)
        {
            current = _current;
        }

        if (current is not null)
        {
            await current.ConfigureAwait(false);
        }

        _disposed.Dispose();
    }

    private bool TryStart(SessionTrigger trigger, long streamFromSample, WakeWordDetection? detection)
    {
        lock (_gate)
        {
            if (_disposed.IsCancellationRequested)
            {
                return false;
            }

            if (_current is { IsCompleted: false })
            {
                SessionRejected?.Invoke(this, SessionRejection.Busy);
                return false;
            }

            if (!_limiter.TryAcquire())
            {
                SessionRejected?.Invoke(this, SessionRejection.RateLimited);
                return false;
            }

            _currentCts?.Dispose();
            _currentCts = CancellationTokenSource.CreateLinkedTokenSource(_disposed.Token);
            var token = _currentCts.Token;
            _current = Task.Run(() => RunAsync(trigger, streamFromSample, detection, token), CancellationToken.None);
            return true;
        }
    }

    private async Task RunAsync(SessionTrigger trigger, long streamFromSample, WakeWordDetection? detection, CancellationToken cancellationToken)
    {
        var session = new DeepgramSession(_ring, _tokens, _transportFactory, _options.Deepgram, _options.Session, _stripper, _time);
        session.TranscriptUpdated += (_, update) => TranscriptUpdated?.Invoke(this, update);
        SessionStarted?.Invoke(this, trigger);

        var result = await session.RunAsync(streamFromSample, cancellationToken).ConfigureAwait(false);

        _telemetry?.SessionCompleted(new SessionTelemetry(
            trigger, detection?.Keyword, detection?.Score, result.Reason, result.AudioSent, result.ConnectLatency));
        SessionEnded?.Invoke(this, result);
    }
}
