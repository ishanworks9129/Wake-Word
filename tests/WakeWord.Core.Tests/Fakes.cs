using System.Runtime.InteropServices;
using System.Threading.Channels;
using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;

namespace WakeWord.Core.Tests;

internal sealed class FakeTransport : IDeepgramTransport
{
    private readonly Channel<string?> _incoming = Channel.CreateUnbounded<string?>();
    private readonly List<short> _audio = [];
    private readonly List<string> _text = [];

    public TaskCompletionSource<(Uri Uri, string Token)> Connected { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public Func<Task>? DuringConnect { get; init; }

    public SemaphoreSlim? SendGate { get; init; }

    public bool Disposed { get; private set; }

    public short[] Audio
    {
        get { lock (_audio) return [.. _audio]; }
    }

    public string[] TextSent
    {
        get { lock (_text) return [.. _text]; }
    }

    public async Task ConnectAsync(Uri uri, string accessToken, CancellationToken cancellationToken)
    {
        if (DuringConnect is not null)
        {
            await DuringConnect();
        }

        Connected.TrySetResult((uri, accessToken));
    }

    public async Task SendAudioAsync(ReadOnlyMemory<byte> pcm, CancellationToken cancellationToken)
    {
        if (SendGate is not null)
        {
            await SendGate.WaitAsync(cancellationToken);
        }

        lock (_audio)
        {
            _audio.AddRange(MemoryMarshal.Cast<byte, short>(pcm.Span).ToArray());
        }
    }

    public Task SendTextAsync(string message, CancellationToken cancellationToken)
    {
        lock (_text)
        {
            _text.Add(message);
        }

        if (message.Contains("CloseStream"))
        {
            _incoming.Writer.TryWrite(null); // Deepgram flushes and closes
        }

        return Task.CompletedTask;
    }

    public async Task<string?> ReceiveTextAsync(CancellationToken cancellationToken) =>
        await _incoming.Reader.ReadAsync(cancellationToken);

    public void Push(string message) => _incoming.Writer.TryWrite(message);

    public void CloseFromServer() => _incoming.Writer.TryWrite(null);

    public async Task WaitForAudioAsync(int samples)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        while (Audio.Length < samples)
        {
            if (DateTime.UtcNow > deadline)
            {
                throw new TimeoutException($"Only {Audio.Length} of {samples} samples arrived.");
            }

            await Task.Delay(10);
        }
    }

    public ValueTask DisposeAsync()
    {
        Disposed = true;
        return ValueTask.CompletedTask;
    }

    public static string Results(string transcript, bool isFinal) =>
        $$"""{"type":"Results","channel":{"alternatives":[{"transcript":"{{transcript}}"}]},"is_final":{{(isFinal ? "true" : "false")}},"speech_final":false}""";

    public const string UtteranceEnd = """{"type":"UtteranceEnd","last_word_end":2.1}""";
}

internal sealed class StaticTokenProvider(Exception? error = null) : ITokenProvider
{
    public int Prefetches;

    public ValueTask<DeepgramToken> GetTokenAsync(CancellationToken cancellationToken) =>
        error is null
            ? ValueTask.FromResult(new DeepgramToken("test-token", DateTimeOffset.UtcNow.AddMinutes(5)))
            : ValueTask.FromException<DeepgramToken>(error);

    public void Prefetch() => Interlocked.Increment(ref Prefetches);
}

/// <summary>Model whose score for frame i is scores[i] (last value repeats).</summary>
internal sealed class ScriptedModel(params double[] scores) : IWakeWordModel
{
    public int FrameSamples => 1280;

    public int Appended { get; private set; }

    public int Scored { get; private set; }

    public void AppendFrame(ReadOnlySpan<short> frame) => Appended++;

    public double Score()
    {
        Scored++;
        return scores[Math.Min(Appended - 1, scores.Length - 1)];
    }

    public void Dispose()
    {
    }
}

/// <summary>VAD whose probability for frame i is probabilities[i] (last value repeats).</summary>
internal sealed class ScriptedVad(params double[] probabilities) : IVoiceActivityDetector
{
    private int _frame;

    public double SpeechProbability(ReadOnlySpan<short> frame) =>
        probabilities[Math.Min(_frame++, probabilities.Length - 1)];

    public void Dispose()
    {
    }
}
