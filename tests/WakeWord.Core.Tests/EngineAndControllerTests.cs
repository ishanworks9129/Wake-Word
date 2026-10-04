using WakeWord.Core.Audio;
using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;
using WakeWord.Core.Sessions;

namespace WakeWord.Core.Tests;

public class WakeWordEngineTests
{
    private static void Feed(WakeWordEngine engine, int samples, int chunk = 1000)
    {
        var buffer = new short[chunk];
        for (var fed = 0; fed < samples; fed += chunk)
        {
            engine.Process(buffer.AsSpan(0, Math.Min(chunk, samples - fed)));
        }
    }

    [Fact]
    public void Mel_runs_on_every_frame_but_scoring_only_behind_the_vad_gate()
    {
        var ring = new PcmRingBuffer(16000, TimeSpan.FromSeconds(10));
        var model = new ScriptedModel(0.1);
        // Frames 0-4 silent, 5-9 speech, then silence; 1 s hangover = 12.5 frames.
        var vad = new ScriptedVad([0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0]);
        var engine = new WakeWordEngine(model, vad, ring, new WakeWordDetectorOptions(), new VadGateOptions());
        var speechStarts = 0;
        engine.SpeechStarted += (_, _) => speechStarts++;

        Feed(engine, 1280 * 30, chunk: 1000);

        Assert.Equal(1280 * 30, ring.TotalWritten); // ring gets every sample regardless of gating
        Assert.Equal(30, model.Appended);
        Assert.Equal(1, speechStarts);
        Assert.Equal(5 + 12, model.Scored); // 5 speech frames + 12 hangover frames
        Assert.Equal(engine.FramesScored, model.Scored);
    }

    [Fact]
    public void Fires_with_the_frame_end_position_and_respects_pause()
    {
        var ring = new PcmRingBuffer(16000, TimeSpan.FromSeconds(10));
        var engine = new WakeWordEngine(new ScriptedModel(0.9), new ScriptedVad(1.0), ring, new WakeWordDetectorOptions(), new VadGateOptions());
        var fires = new List<WakeWordDetection>();
        engine.Detected += (_, d) => fires.Add(d);

        engine.Paused = true;
        Feed(engine, 1280 * 5);
        Assert.Empty(fires);

        engine.Paused = false;
        Feed(engine, 1280 * 3);
        Assert.Equal(1280 * 8, Assert.Single(fires).FrameEndSample);
    }
}

public class WakeWordControllerTests
{
    private sealed class Telemetry : ISessionTelemetrySink
    {
        public List<SessionTelemetry> Records { get; } = [];

        public void SessionCompleted(SessionTelemetry record)
        {
            lock (Records)
            {
                Records.Add(record);
            }
        }
    }

    [Fact]
    public async Task Starts_one_session_with_pre_roll_and_rejects_triggers_while_busy()
    {
        var transports = new List<FakeTransport>();
        var tokens = new StaticTokenProvider();
        var telemetry = new Telemetry();
        var options = new WakeWordOptions { Session = new DeepgramSessionLimits { NoTranscriptCutoff = TimeSpan.FromMinutes(1), HardTimeout = TimeSpan.FromMinutes(1) } };
        await using var controller = new WakeWordController(
            options, new ScriptedModel(0.9), new ScriptedVad(1.0), tokens,
            () => { var t = new FakeTransport(); lock (transports) transports.Add(t); return t; },
            telemetry);

        var rejections = new List<SessionRejection>();
        var ended = new TaskCompletionSource<DeepgramSessionResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        controller.SessionRejected += (_, r) => rejections.Add(r);
        controller.SessionEnded += (_, r) => ended.TrySetResult(r);

        // 3 s of audio: fires at frame 2 (0.24 s); refractory 1.5 s, so it fires again while the first session streams.
        var silence = new short[1280];
        for (var i = 0; i < 38; i++)
        {
            controller.Engine.Process(silence);
        }

        Assert.Equal(1, tokens.Prefetches); // VAD onset prefetched a token
        Assert.True(controller.IsStreaming);
        Assert.Contains(SessionRejection.Busy, rejections);

        // The first fire was at 3,840 samples, so the 2 s pre-roll clamps to sample 0: everything is sent.
        var transport = transports.Single();
        await transport.WaitForAudioAsync(1280 * 38);

        controller.StopSession();
        var result = await ended.Task.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(SessionEndReason.Cancelled, result.Reason);
        var record = Assert.Single(telemetry.Records);
        Assert.Equal(SessionTrigger.WakeWord, record.Trigger);
        Assert.Equal(0.9, record.DetectionScore);
    }

    [Fact]
    public async Task Manual_session_streams_from_now()
    {
        var transport = new FakeTransport();
        await using var controller = new WakeWordController(
            new WakeWordOptions(), new ScriptedModel(0.0), new ScriptedVad(0.0), new StaticTokenProvider(), () => transport);

        controller.Engine.Process(new short[16000]); // 1 s that must not be sent
        Assert.True(controller.StartManualSession());
        await transport.Connected.Task.WaitAsync(TimeSpan.FromSeconds(5));
        controller.Engine.Process(new short[1600]);
        await transport.WaitForAudioAsync(1600);

        Assert.Equal(1600, transport.Audio.Length);
    }
}
