using System.Text.Json;
using WakeWord.Core;
using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;
using WakeWord.Core.Sessions;
using WakeWord.Onnx;

namespace WakeWord.Onnx.Tests;

/// <summary>
/// testdata/models/smoke is a real (tiny, untrained-quality) package exported by the training pipeline,
/// including golden.json: the Python streaming reference's per-chunk embeddings and scores. The golden
/// checks also run against the shipped package in models/wakeword, so a new model can't ship unchecked.
/// </summary>
public class OnnxRuntimeTests
{
    private static readonly string Smoke = Path.Combine(AppContext.BaseDirectory, "testdata", "models", "smoke");
    private static readonly ModelPackage Package = ModelPackage.LoadFromDirectory(Smoke);
    private static readonly JsonElement Golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(Smoke, "golden.json"))).RootElement;

    private static short[] GoldenAudio() => GoldenAudio(Golden);

    private static short[] GoldenAudio(JsonElement golden) =>
        golden.GetProperty("audio_int16").EnumerateArray().Select(v => (short)v.GetInt32()).ToArray();

    private static ReadOnlySpan<short> Chunk(short[] audio, int i) => audio.AsSpan(i * 1280, 1280);

    [Theory]
    [InlineData("testdata/models/smoke")]
    [InlineData("models/wakeword")]
    public void Streaming_scores_and_embeddings_match_the_python_reference(string packageDir)
    {
        var dir = Path.Combine(AppContext.BaseDirectory, packageDir);
        var golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "golden.json"))).RootElement;
        using var model = new OnnxWakeWordModel(ModelPackage.LoadFromDirectory(dir));
        var audio = GoldenAudio(golden);
        var scores = new double[model.Keywords.Count];
        var checkedScores = 0;

        foreach (var chunk in golden.GetProperty("chunks").EnumerateArray())
        {
            var i = chunk.GetProperty("chunk").GetInt32();
            model.AppendFrame(Chunk(audio, i));
            var scored = model.Score(scores);

            Assert.Equal(chunk.TryGetProperty("scores", out _), scored);
            if (chunk.TryGetProperty("embedding_first8", out var first8))
            {
                var expected = first8.EnumerateArray().Select(v => v.GetSingle()).ToArray();
                var actual = model.LatestEmbedding[..8].ToArray();
                for (var d = 0; d < 8; d++)
                {
                    Assert.True(Math.Abs(expected[d] - actual[d]) < 2e-3, $"chunk {i} dim {d}: {expected[d]} vs {actual[d]}");
                }
            }

            if (chunk.TryGetProperty("scores", out var expectedScores))
            {
                for (var k = 0; k < model.Keywords.Count; k++)
                {
                    var expected = expectedScores.GetProperty(model.Keywords[k]).GetDouble();
                    Assert.True(Math.Abs(expected - scores[k]) < 1e-3, $"chunk {i} {model.Keywords[k]}: {expected} vs {scores[k]}");
                    checkedScores++;
                }
            }
        }

        Assert.True(checkedScores >= 40, $"only {checkedScores} scores compared");
    }

    [Fact]
    public void Scoring_after_a_closed_gate_back_fills_and_matches_continuous_scoring()
    {
        var audio = GoldenAudio();
        var chunks = audio.Length / 1280;
        using var continuous = new OnnxWakeWordModel(Package);
        using var gated = new OnnxWakeWordModel(Package);
        var a = new double[2];
        var b = new double[2];

        for (var i = 0; i < chunks; i++)
        {
            continuous.AppendFrame(Chunk(audio, i));
            var expected = continuous.Score(a);

            gated.AppendFrame(Chunk(audio, i));
            if (i < 35)
            {
                continue; // VAD gate closed: mel only
            }

            Assert.Equal(expected, gated.Score(b));
            Assert.Equal(a[0], b[0], 5);
            Assert.Equal(a[1], b[1], 5);
        }
    }

    [Fact]
    public void Spotter_fires_exactly_where_the_shared_detector_rules_say()
    {
        var audio = GoldenAudio();
        using var model = new OnnxWakeWordModel(Package);
        var scores = new List<double[]>();
        for (var i = 0; i < audio.Length / 1280; i++)
        {
            model.AppendFrame(Chunk(audio, i));
            var s = new double[2];
            scores.Add(model.Score(s) ? s : [0, 0]);
        }

        // Pick a threshold the golden clip actually crosses, so the comparison is meaningful.
        var threshold = scores.Max(s => s[0]) * 0.5;
        var options = new WakeWordDetectorOptions
        {
            BaseThreshold = threshold,
            ConsecutiveFrames = 2,
            Adaptive = new AdaptiveThresholdOptions { MinThreshold = 0, MaxThreshold = 1 },
        };
        var reference = new WakeWordDetector(options, 16000);
        var expected = scores.Select((s, i) => reference.Process(s[0], -100, (i + 1) * 1280L) is not null).ToList();

        var package = Package.KeywordOptions(["hey_uno"]);
        using var spotter = SpotterWith(options);
        var actual = Enumerable.Range(0, audio.Length / 1280).Select(i => spotter.Process(Chunk(audio, i)) == 0).ToList();

        Assert.Equal(expected, actual);
        Assert.Contains(true, actual);
        Assert.Equal("hey_uno", package.Single().Id);
    }

    private static KeywordSpotter SpotterWith(WakeWordDetectorOptions options)
    {
        // Calibrated tables come from training; for this rule check, pin the threshold through a one-point table.
        var dir = Directory.CreateTempSubdirectory("spotter").FullName;
        foreach (var f in Directory.GetFiles(Smoke))
        {
            File.Copy(f, Path.Combine(dir, Path.GetFileName(f)));
        }

        var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(Smoke, "models.json"))).RootElement;
        using var stream = File.Create(Path.Combine(dir, "models.json"));
        using (var w = new Utf8JsonWriter(stream))
        {
            w.WriteStartObject();
            foreach (var p in manifest.EnumerateObject())
            {
                if (p.Name != "keywords")
                {
                    p.WriteTo(w);
                    continue;
                }

                w.WriteStartArray("keywords");
                foreach (var k in p.Value.EnumerateArray())
                {
                    w.WriteStartObject();
                    foreach (var kp in k.EnumerateObject())
                    {
                        if (kp.Name == "sensitivity_table")
                        {
                            w.WriteStartArray(kp.Name);
                            w.WriteStartArray();
                            w.WriteNumberValue(0.5);
                            w.WriteNumberValue(options.BaseThreshold);
                            w.WriteEndArray();
                            w.WriteEndArray();
                        }
                        else if (kp.Name == "detector")
                        {
                            w.WriteStartObject(kp.Name);
                            w.WriteNumber("consecutive_frames", options.ConsecutiveFrames);
                            w.WriteNumber("refractory_seconds", options.Refractory.TotalSeconds);
                            w.WriteNumber("min_threshold", 0);
                            w.WriteNumber("max_threshold", 1);
                            w.WriteEndObject();
                        }
                        else
                        {
                            kp.WriteTo(w);
                        }
                    }

                    w.WriteEndObject();
                }

                w.WriteEndArray();
            }

            w.WriteEndObject();
        }

        stream.Close();
        return KeywordSpotter.Create(dir, ["hey_uno"]);
    }

    [Fact]
    public void Package_maps_sensitivity_to_thresholds_and_validates_input()
    {
        var options = Package.KeywordOptions(["hello_uno", "hey_uno"], [0.2, 0.9]);
        Assert.Equal(["hello_uno", "hey_uno"], options.Select(o => o.Id));
        Assert.Equal("Hello UNO", options[0].Phrase);
        Assert.Equal(3, options[0].Detector.ConsecutiveFrames);
        Assert.Equal(0.9999, options[0].Detector.Adaptive.MaxThreshold);

        Assert.Throws<ArgumentException>(() => Package.KeywordOptions(["hey_google"]));
        Assert.Throws<ArgumentException>(() => Package.KeywordOptions(["hey_uno"], [0.5, 0.5]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Package.KeywordOptions(["hey_uno"], [1.5]));

        using var spotter = KeywordSpotter.Create(Package);
        Assert.Equal((16000, 1280), (spotter.SampleRate, spotter.FrameLength));
        Assert.Equal(["hey_uno", "hello_uno"], spotter.Keywords);
        Assert.Throws<ArgumentException>(() => spotter.Process(new short[512]));
    }

    [Fact]
    public void Silero_hears_speech_but_not_silence_or_noise()
    {
        using var vad = SileroVad.Load(Path.Combine(AppContext.BaseDirectory, "models", "vad", "silero_vad.onnx"));
        var speech = ReadWav(Path.Combine(AppContext.BaseDirectory, "testdata", "audio", "hey_uno_tts.wav"));
        var probabilities = new List<double>();
        for (var i = 0; i + 1280 <= speech.Length; i += 1280)
        {
            probabilities.Add(vad.SpeechProbability(speech.AsSpan(i, 1280)));
        }

        // The file is 1 s of silence, the phrase, then 1 s of silence.
        Assert.True(probabilities.Take(10).Max() < 0.2, $"silence: {probabilities.Take(10).Max()}");
        Assert.True(probabilities.Max() > 0.6, $"speech: {probabilities.Max()}");

        vad.Reset();
        var rng = new Random(1);
        var noise = Enumerable.Range(0, 16000).Select(_ => (short)rng.Next(-300, 300)).ToArray();
        var noiseMax = Enumerable.Range(0, 12).Max(i => vad.SpeechProbability(noise.AsSpan(i * 1280, 1280)));
        Assert.True(noiseMax < 0.3, $"white noise: {noiseMax}");
    }

    [Fact]
    public async Task Controller_runs_the_real_model_behind_the_real_vad()
    {
        var options = new WakeWordOptions { Keywords = Package.KeywordOptions() };
        using var model = new OnnxWakeWordModel(Package);
        using var vad = SileroVad.Load(Path.Combine(AppContext.BaseDirectory, "models", "vad", "silero_vad.onnx"));
        var tokens = new PrefetchCounter();
        await using var controller = new WakeWordController(options, model, vad, tokens);

        var speech = ReadWav(Path.Combine(AppContext.BaseDirectory, "testdata", "audio", "hey_uno_tts.wav"));
        var audio = new short[16000 * 3].Concat(speech).ToArray(); // 3 s of silence, then speech
        for (var i = 0; i + 1000 <= audio.Length; i += 1000)
        {
            controller.Engine.Process(audio.AsSpan(i, 1000)); // capture callbacks need not match the 1280-sample frame
        }

        var engine = controller.Engine;
        Assert.Equal(["hey_uno", "hello_uno"], engine.Keywords);
        Assert.Equal(audio.Length / 1000 * 1000 / 1280, engine.FramesProcessed);
        Assert.InRange(engine.FramesScored, 1, engine.FramesProcessed - 30); // silence was skipped
        Assert.Equal(1, tokens.Prefetches); // speech onset prefetched a Deepgram token
    }

    private sealed class PrefetchCounter : ITokenProvider
    {
        public int Prefetches;

        public ValueTask<DeepgramToken> GetTokenAsync(CancellationToken cancellationToken) =>
            ValueTask.FromResult(new DeepgramToken("t", DateTimeOffset.UtcNow.AddMinutes(5)));

        public void Prefetch() => Prefetches++;
    }

    private static short[] ReadWav(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var data = 12;
        while (System.Text.Encoding.ASCII.GetString(bytes, data, 4) != "data")
        {
            data += 8 + BitConverter.ToInt32(bytes, data + 4);
        }

        var length = BitConverter.ToInt32(bytes, data + 4);
        return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, short>(bytes.AsSpan(data + 8, length)).ToArray();
    }
}
