using WakeWord.Core.Audio;
using WakeWord.Core.Deepgram;
using WakeWord.Core.Transcripts;

namespace WakeWord.Core.Tests;

public class DeepgramSessionTests
{
    private const int Rate = 16000;
    private static readonly WakePhraseStripper Stripper = new("Hey UNO", ["hey you know"]);

    private static short[] Ramp(int start, int count) =>
        Enumerable.Range(start, count).Select(i => (short)(i % short.MaxValue)).ToArray();

    private static DeepgramSession Session(PcmRingBuffer ring, FakeTransport transport, DeepgramSessionLimits? limits = null, ITokenProvider? tokens = null) =>
        new(ring, tokens ?? new StaticTokenProvider(), () => transport, new DeepgramStreamingOptions(), limits ?? new DeepgramSessionLimits(), Stripper);

    [Fact]
    public async Task Streams_pre_roll_and_live_audio_with_no_gap_even_while_the_socket_opens()
    {
        var ring = new PcmRingBuffer(Rate, TimeSpan.FromSeconds(10));
        ring.Write(Ramp(0, 3 * Rate)); // 3 s captured; trigger at 3 s, pre-roll from 1 s

        // Audio keeps arriving while the socket is still opening.
        var transport = new FakeTransport { DuringConnect = () => { ring.Write(Ramp(3 * Rate, Rate / 2)); return Task.CompletedTask; } };
        var run = Session(ring, transport).RunAsync(Rate, CancellationToken.None);

        await transport.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        ring.Write(Ramp(3 * Rate + Rate / 2, Rate / 2)); // live audio after connect
        await transport.WaitForAudioAsync(3 * Rate);

        transport.Push(FakeTransport.Results("Hey UNO, what's the weather?", isFinal: true));
        transport.Push(FakeTransport.UtteranceEnd);
        var result = await run.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(SessionEndReason.UtteranceEnd, result.Reason);
        Assert.Equal("what's the weather?", result.Transcript);
        Assert.Equal(Ramp(Rate, 3 * Rate), transport.Audio); // every sample from 1 s to 4 s, in order
        Assert.Equal(TimeSpan.FromSeconds(3), result.AudioSent);
        Assert.Contains(transport.TextSent, m => m.Contains("CloseStream"));
        Assert.Equal("test-token", (await transport.Connected.Task).Token);
    }

    [Fact]
    public async Task Closes_quickly_when_only_the_wake_phrase_is_heard()
    {
        var ring = new PcmRingBuffer(Rate, TimeSpan.FromSeconds(10));
        ring.Write(Ramp(0, Rate));
        var transport = new FakeTransport();
        var limits = new DeepgramSessionLimits { NoTranscriptCutoff = TimeSpan.FromMilliseconds(300) };
        var run = Session(ring, transport, limits).RunAsync(0, CancellationToken.None);

        await transport.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        transport.Push(FakeTransport.Results("Hey you know", isFinal: true)); // TV said something close to the phrase
        var result = await run.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(SessionEndReason.NoTranscript, result.Reason);
        Assert.Equal(string.Empty, result.Transcript);
        Assert.False(result.IsError);
    }

    [Fact]
    public async Task Interim_speech_defers_the_cutoff_but_not_the_hard_timeout()
    {
        var ring = new PcmRingBuffer(Rate, TimeSpan.FromSeconds(10));
        var transport = new FakeTransport();
        var limits = new DeepgramSessionLimits { NoTranscriptCutoff = TimeSpan.FromMilliseconds(200), HardTimeout = TimeSpan.FromMilliseconds(600) };
        var run = Session(ring, transport, limits).RunAsync(0, CancellationToken.None);

        await transport.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        transport.Push(FakeTransport.Results("Hey UNO tell me a", isFinal: false));
        var result = await run.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(SessionEndReason.HardTimeout, result.Reason);
    }

    [Fact]
    public async Task Reports_connect_failure_as_a_visible_error()
    {
        var ring = new PcmRingBuffer(Rate, TimeSpan.FromSeconds(10));
        var transport = new FakeTransport();
        var result = await Session(ring, transport, tokens: new StaticTokenProvider(new HttpRequestException("broker down")))
            .RunAsync(0, CancellationToken.None);

        Assert.Equal(SessionEndReason.ConnectFailed, result.Reason);
        Assert.True(result.IsError);
        Assert.IsType<HttpRequestException>(result.Error);
    }

    [Fact]
    public async Task Ends_with_overrun_rather_than_sending_audio_with_a_hole()
    {
        var ring = new PcmRingBuffer(Rate, TimeSpan.FromSeconds(1));
        ring.Write(Ramp(0, Rate));
        var gate = new SemaphoreSlim(0);
        var transport = new FakeTransport { SendGate = gate };
        var run = Session(ring, transport).RunAsync(0, CancellationToken.None);

        await transport.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        ring.Write(Ramp(Rate, 2 * Rate)); // network stalled for 2 s; the ring only holds 1 s
        gate.Release(100);
        var result = await run.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(SessionEndReason.AudioOverrun, result.Reason);
        Assert.True(result.IsError);
    }

    [Fact]
    public async Task Server_close_and_cancellation_end_the_session()
    {
        var ring = new PcmRingBuffer(Rate, TimeSpan.FromSeconds(10));
        var closed = new FakeTransport();
        var run = Session(ring, closed).RunAsync(0, CancellationToken.None);
        await closed.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        closed.CloseFromServer();
        Assert.Equal(SessionEndReason.ServerClosed, (await run.WaitAsync(TimeSpan.FromSeconds(10))).Reason);

        using var cts = new CancellationTokenSource();
        var cancelled = new FakeTransport();
        run = Session(ring, cancelled).RunAsync(0, cts.Token);
        await cancelled.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        await cts.CancelAsync();
        Assert.Equal(SessionEndReason.Cancelled, (await run.WaitAsync(TimeSpan.FromSeconds(10))).Reason);
        Assert.True(cancelled.Disposed);
    }
}
