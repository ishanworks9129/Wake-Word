using System.Runtime.InteropServices;
using System.Text.Json;
using WakeWord.Core.Audio;
using WakeWord.Core.Transcripts;

namespace WakeWord.Core.Deepgram;

public enum SessionEndReason
{
    /// <summary>Deepgram reported the end of the utterance after real speech.</summary>
    UtteranceEnd,

    /// <summary>Nothing but the wake phrase was heard in time; most likely a false accept.</summary>
    NoTranscript,

    HardTimeout,
    Cancelled,
    ServerClosed,

    /// <summary>The token or socket could not be obtained. Show a visible error (plan 8.5).</summary>
    ConnectFailed,

    TransportError,

    /// <summary>The network stalled long enough that the needed audio left the ring; the session ends rather than send a hole.</summary>
    AudioOverrun,
}

public sealed record TranscriptUpdate(string Text, bool IsFinal);

public sealed record DeepgramSessionResult(
    SessionEndReason Reason,
    string Transcript,
    TimeSpan AudioSent,
    TimeSpan ConnectLatency,
    Exception? Error = null)
{
    public bool IsError => Reason is SessionEndReason.ConnectFailed or SessionEndReason.TransportError or SessionEndReason.AudioOverrun;
}

/// <summary>
/// One Deepgram streaming session (plan 8.3). Streams from a position in the pre-roll ring and keeps
/// following the write head from the same cursor, so there is no gap between pre-roll and live audio.
/// Ends on UtteranceEnd, the no-transcript cutoff, the hard timeout, cancellation or an error.
/// </summary>
public sealed class DeepgramSession(
    PcmRingBuffer ring,
    ITokenProvider tokens,
    Func<IDeepgramTransport> transportFactory,
    DeepgramStreamingOptions streaming,
    DeepgramSessionLimits limits,
    WakePhraseStripper stripper,
    TimeProvider? time = null)
{
    private static readonly string CloseStream = """{"type":"CloseStream"}""";

    private readonly TimeProvider _time = time ?? TimeProvider.System;

    /// <summary>Raised from a background thread with the transcript so far, wake phrase removed.</summary>
    public event EventHandler<TranscriptUpdate>? TranscriptUpdated;

    /// <param name="streamFromSample">First sample to send, normally the trigger position minus the pre-roll.
    /// Clamped to the oldest sample the ring still holds.</param>
    public async Task<DeepgramSessionResult> RunAsync(long streamFromSample, CancellationToken cancellationToken)
    {
        var position = Math.Max(Math.Max(0, streamFromSample), ring.OldestAvailable);
        var started = _time.GetTimestamp();
        var transport = transportFactory();
        await using var _ = transport.ConfigureAwait(false);

        try
        {
            var token = await tokens.GetTokenAsync(cancellationToken).ConfigureAwait(false);
            await transport.ConnectAsync(streaming.BuildUri(ring.SampleRate), token.AccessToken, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return new DeepgramSessionResult(SessionEndReason.Cancelled, string.Empty, TimeSpan.Zero, _time.GetElapsedTime(started));
        }
        catch (Exception ex)
        {
            return new DeepgramSessionResult(SessionEndReason.ConnectFailed, string.Empty, TimeSpan.Zero, _time.GetElapsedTime(started), ex);
        }

        var connectLatency = _time.GetElapsedTime(started);
        var state = new TranscriptState(stripper);
        var end = new TaskCompletionSource<(SessionEndReason Reason, Exception? Error)>(TaskCreationOptions.RunContinuationsAsynchronously);

        using var sendCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        using var receiveCts = new CancellationTokenSource();
        using var cutoff = new CancellationTokenSource(limits.NoTranscriptCutoff, _time);
        using var hard = new CancellationTokenSource(limits.HardTimeout, _time);
        using var cutoffReg = cutoff.Token.Register(() =>
        {
            if (!state.HasSpeech)
            {
                end.TrySetResult((SessionEndReason.NoTranscript, null));
            }
        });
        using var hardReg = hard.Token.Register(() => end.TrySetResult((SessionEndReason.HardTimeout, null)));
        using var cancelReg = cancellationToken.Register(() => end.TrySetResult((SessionEndReason.Cancelled, null)));

        long samplesSent = 0;
        var sendTask = SendLoopAsync(transport, position, end, n => Interlocked.Add(ref samplesSent, n), sendCts.Token);
        var receiveTask = ReceiveLoopAsync(transport, state, end, receiveCts.Token);

        var (reason, error) = await end.Task.ConfigureAwait(false);

        await sendCts.CancelAsync().ConfigureAwait(false);
        await sendTask.ConfigureAwait(false);

        if (reason is not (SessionEndReason.ServerClosed or SessionEndReason.TransportError))
        {
            // Ask Deepgram to flush final results, then give it a moment to send them and close.
            try
            {
                using var closeTimeout = new CancellationTokenSource(limits.CloseGrace, _time);
                await transport.SendTextAsync(CloseStream, closeTimeout.Token).ConfigureAwait(false);
                await receiveTask.WaitAsync(limits.CloseGrace, _time).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is TimeoutException or OperationCanceledException or System.Net.WebSockets.WebSocketException)
            {
                // Results already received still count; the socket is disposed below.
            }
        }

        await receiveCts.CancelAsync().ConfigureAwait(false);
        await receiveTask.ConfigureAwait(false);

        return new DeepgramSessionResult(
            reason,
            state.FinalText,
            ring.DurationOf(Interlocked.Read(ref samplesSent)),
            connectLatency,
            error);
    }

    private async Task SendLoopAsync(
        IDeepgramTransport transport,
        long position,
        TaskCompletionSource<(SessionEndReason, Exception?)> end,
        Action<int> onSent,
        CancellationToken cancellationToken)
    {
        var samples = new short[limits.MaxChunkSamples];
        var bytes = new byte[limits.MaxChunkSamples * sizeof(short)];
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var count = ring.Read(position, samples);
                if (count == 0)
                {
                    await ring.WaitForDataAsync(position, cancellationToken).ConfigureAwait(false);
                    continue;
                }

                // linear16 is little-endian; every supported target (ARM64, x64, WASM) is little-endian.
                MemoryMarshal.AsBytes(samples.AsSpan(0, count)).CopyTo(bytes);
                await transport.SendAudioAsync(bytes.AsMemory(0, count * sizeof(short)), cancellationToken).ConfigureAwait(false);
                position += count;
                onSent(count);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        catch (PcmOverrunException ex)
        {
            end.TrySetResult((SessionEndReason.AudioOverrun, ex));
        }
        catch (Exception ex)
        {
            end.TrySetResult((SessionEndReason.TransportError, ex));
        }
    }

    private async Task ReceiveLoopAsync(
        IDeepgramTransport transport,
        TranscriptState state,
        TaskCompletionSource<(SessionEndReason, Exception?)> end,
        CancellationToken cancellationToken)
    {
        try
        {
            while (await transport.ReceiveTextAsync(cancellationToken).ConfigureAwait(false) is { } message)
            {
                switch (state.Apply(message))
                {
                    case MessageEffect.TranscriptChanged(var update):
                        TranscriptUpdated?.Invoke(this, update);
                        break;
                    case MessageEffect.UtteranceEnded when state.HasFinalSpeech:
                        end.TrySetResult((SessionEndReason.UtteranceEnd, null));
                        break;
                }
            }

            end.TrySetResult((SessionEndReason.ServerClosed, null));
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            end.TrySetResult((SessionEndReason.TransportError, ex));
        }
    }

    private abstract record MessageEffect
    {
        public sealed record None : MessageEffect;

        public sealed record TranscriptChanged(TranscriptUpdate Update) : MessageEffect;

        public sealed record UtteranceEnded : MessageEffect;
    }

    /// <summary>Accumulates Deepgram results. Touched only by the receive loop until it completes.</summary>
    private sealed class TranscriptState(WakePhraseStripper stripper)
    {
        private readonly List<string> _finals = [];
        private volatile bool _hasSpeech;

        /// <summary>Any non-wake-word words, interim or final. Interim results count so a slow speaker is not cut off.</summary>
        public bool HasSpeech => _hasSpeech;

        public bool HasFinalSpeech { get; private set; }

        public string FinalText => stripper.Strip(string.Join(' ', _finals));

        public MessageEffect Apply(string message)
        {
            using var doc = JsonDocument.Parse(message);
            var root = doc.RootElement;
            var type = root.TryGetProperty("type", out var t) ? t.GetString() : null;

            if (type == "UtteranceEnd")
            {
                return new MessageEffect.UtteranceEnded();
            }

            if (type != "Results")
            {
                return new MessageEffect.None();
            }

            var transcript = root.GetProperty("channel").GetProperty("alternatives")[0].GetProperty("transcript").GetString() ?? string.Empty;
            var isFinal = root.TryGetProperty("is_final", out var f) && f.GetBoolean();
            if (transcript.Length == 0)
            {
                return new MessageEffect.None();
            }

            var text = string.Join(' ', _finals.Append(transcript));
            if (isFinal)
            {
                _finals.Add(transcript);
            }

            var stripped = stripper.Strip(text);
            if (stripped.Length > 0)
            {
                _hasSpeech = true;
                HasFinalSpeech |= isFinal;
            }

            return new MessageEffect.TranscriptChanged(new TranscriptUpdate(stripped, isFinal));
        }
    }
}
