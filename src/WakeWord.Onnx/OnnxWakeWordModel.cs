using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using WakeWord.Core.Detection;

[assembly: InternalsVisibleTo("WakeWord.Onnx.Tests")]

namespace WakeWord.Onnx;

public sealed record OnnxModelOptions
{
    /// <summary>Threads per ONNX session. One keeps an always-on detector cheap on phones.</summary>
    public int IntraOpThreads { get; init; } = 1;
}

/// <summary>
/// Streaming openWakeWord features plus one classifier per keyword, running on ONNX Runtime.
/// Behaviour is defined by training/wakeword_train/features.py StreamingFeatures and checked against
/// the package's golden.json:
/// <list type="bullet">
/// <item>Each 1280-sample chunk runs the mel model over the chunk plus the previous 480 samples
/// (8 frames; the first chunk has no context and yields 5). Mel values are transformed x / 10 + 2.</item>
/// <item>The embedding for chunk k uses the 76 mel frames before the newest one, which keeps runtime
/// windows on the same grid as training.</item>
/// <item>A score needs the 16 most recent embeddings.</item>
/// </list>
/// <see cref="AppendFrame"/> only runs the mel model; <see cref="Score"/> embeds every chunk since the last
/// call (up to the 16 needed), so skipping Score while the VAD gate is closed costs no accuracy.
/// </summary>
public sealed class OnnxWakeWordModel : IWakeWordModel
{
    public const int FrameLength = 1280;
    private const int MelBins = 32;
    private const int MelWindow = 76;
    private const int MelContext = 480;
    private const int EmbeddingDim = 96;
    private const int ClassifierFrames = 16;

    // Enough mel frames to back-fill all 16 embeddings: the oldest needs frames from 8k - 192 onward.
    private const int MelCapacity = MelWindow + 1 + 8 * (ClassifierFrames - 1) + 8;

    private readonly InferenceSession _mel;
    private readonly InferenceSession _embedding;
    private readonly InferenceSession[] _classifiers;
    private readonly string _melInput;
    private readonly string _embeddingInput;
    private readonly string[] _classifierInputs;
    private readonly RunOptions _run = new();

    private readonly short[] _context = new short[MelContext];
    private readonly float[] _melAudio = new float[MelContext + FrameLength];
    private readonly float[] _melRing = new float[MelCapacity * MelBins];
    private readonly float[] _embeddingRing = new float[ClassifierFrames * EmbeddingDim];
    private readonly float[] _embeddingBatch = new float[ClassifierFrames * MelWindow * MelBins];
    private readonly float[] _classifierInput = new float[ClassifierFrames * EmbeddingDim];

    private long _chunks;        // chunks appended
    private long _melFrames;     // mel frames produced
    private long _embeddedChunks; // chunks whose embedding is in the ring (always the newest ones)

    public OnnxWakeWordModel(ModelPackage package, IReadOnlyList<string>? keywords = null, OnnxModelOptions? options = null)
    {
        options ??= new OnnxModelOptions();
        using var sessionOptions = new SessionOptions
        {
            IntraOpNumThreads = options.IntraOpThreads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };

        var chosen = (keywords ?? package.Keywords.Select(k => k.Id).ToList()).Select(package.Keyword).ToList();
        Keywords = chosen.Select(k => k.Id).ToList();

        _mel = new InferenceSession(package.File(package.Manifest.Features.Melspectrogram), sessionOptions);
        _embedding = new InferenceSession(package.File(package.Manifest.Features.Embedding), sessionOptions);
        _classifiers = chosen.Select(k => new InferenceSession(package.File(k.Model), sessionOptions)).ToArray();
        _melInput = _mel.InputNames[0];
        _embeddingInput = _embedding.InputNames[0];
        _classifierInputs = chosen.Select((k, i) => _classifiers[i].InputNames.Contains(k.Input) ? k.Input : _classifiers[i].InputNames[0]).ToArray();
    }

    public int FrameSamples => FrameLength;

    public IReadOnlyList<string> Keywords { get; }

    /// <summary>The newest embedding, for golden-vector checks.</summary>
    internal ReadOnlySpan<float> LatestEmbedding =>
        _embeddedChunks == 0 ? [] : _embeddingRing.AsSpan(EmbeddingSlot(_embeddedChunks - 1) * EmbeddingDim, EmbeddingDim);

    public void AppendFrame(ReadOnlySpan<short> frame)
    {
        if (frame.Length != FrameLength)
        {
            throw new ArgumentException($"Frames must be {FrameLength} samples; got {frame.Length}.", nameof(frame));
        }

        var contextLength = _chunks == 0 ? 0 : MelContext;
        for (var i = 0; i < contextLength; i++)
        {
            _melAudio[i] = _context[i];
        }

        for (var i = 0; i < FrameLength; i++)
        {
            _melAudio[contextLength + i] = frame[i];
        }

        frame[^MelContext..].CopyTo(_context);
        _chunks++;

        var samples = contextLength + FrameLength;
        using var input = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _melAudio.AsMemory(0, samples), [1, samples]);
        using var outputs = _mel.Run(_run, [_melInput], [input], _mel.OutputNames);
        var mel = outputs[0].GetTensorDataAsSpan<float>(); // [1, 1, frames, 32]
        var frames = mel.Length / MelBins;
        for (var f = 0; f < frames; f++)
        {
            var dest = _melRing.AsSpan(MelSlot(_melFrames) * MelBins, MelBins);
            var src = mel.Slice(f * MelBins, MelBins);
            for (var b = 0; b < MelBins; b++)
            {
                dest[b] = src[b] / 10f + 2f;
            }

            _melFrames++;
        }
    }

    public bool Score(Span<double> scores)
    {
        if (scores.Length != _classifiers.Length)
        {
            throw new ArgumentException($"Expected room for {_classifiers.Length} scores.", nameof(scores));
        }

        // Chunk k (0-based) can be embedded once 77 mel frames exist after it: 5 + 8k >= 77, so k >= 9.
        const long firstEmbeddableChunk = 9;
        var newest = _chunks - 1;
        if (newest < firstEmbeddableChunk)
        {
            return false;
        }

        var from = Math.Max(Math.Max(_embeddedChunks + firstEmbeddableChunk, newest - ClassifierFrames + 1), firstEmbeddableChunk);
        if (_embeddedChunks + firstEmbeddableChunk < from)
        {
            _embeddedChunks = from - firstEmbeddableChunk; // older chunks fell out of the 16-embedding window
        }

        Embed(from, newest);

        if (_embeddedChunks < ClassifierFrames)
        {
            return false;
        }

        for (var i = 0; i < ClassifierFrames; i++)
        {
            var slot = EmbeddingSlot(_embeddedChunks - ClassifierFrames + i);
            _embeddingRing.AsSpan(slot * EmbeddingDim, EmbeddingDim).CopyTo(_classifierInput.AsSpan(i * EmbeddingDim));
        }

        using var input = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _classifierInput.AsMemory(), [1, ClassifierFrames, EmbeddingDim]);
        for (var k = 0; k < _classifiers.Length; k++)
        {
            using var outputs = _classifiers[k].Run(_run, [_classifierInputs[k]], [input], _classifiers[k].OutputNames);
            scores[k] = outputs[0].GetTensorDataAsSpan<float>()[0];
        }

        return true;
    }

    /// <summary>Embeds chunks first..last (inclusive) in one batch.</summary>
    private void Embed(long first, long last)
    {
        var count = (int)(last - first + 1);
        if (count <= 0)
        {
            return;
        }

        for (var i = 0; i < count; i++)
        {
            // After chunk k the buffer holds frames 0..8k+4; its window is the 76 frames ending before the newest.
            var end = 8 * (first + i) + 4; // exclusive end = newest frame index
            var start = end - MelWindow;
            for (var f = 0; f < MelWindow; f++)
            {
                _melRing.AsSpan(MelSlot(start + f) * MelBins, MelBins).CopyTo(_embeddingBatch.AsSpan((i * MelWindow + f) * MelBins));
            }
        }

        using var input = OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance, _embeddingBatch.AsMemory(0, count * MelWindow * MelBins), [count, MelWindow, MelBins, 1]);
        using var outputs = _embedding.Run(_run, [_embeddingInput], [input], _embedding.OutputNames);
        var embeddings = outputs[0].GetTensorDataAsSpan<float>(); // [count, 1, 1, 96]
        for (var i = 0; i < count; i++)
        {
            embeddings.Slice(i * EmbeddingDim, EmbeddingDim).CopyTo(_embeddingRing.AsSpan(EmbeddingSlot(_embeddedChunks) * EmbeddingDim));
            _embeddedChunks++;
        }
    }

    private static int MelSlot(long frame) => (int)(frame % MelCapacity);

    private static int EmbeddingSlot(long embedding) => (int)(embedding % ClassifierFrames);

    public void Dispose()
    {
        _mel.Dispose();
        _embedding.Dispose();
        foreach (var c in _classifiers)
        {
            c.Dispose();
        }

        _run.Dispose();
    }
}
