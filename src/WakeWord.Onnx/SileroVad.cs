using Microsoft.ML.OnnxRuntime;
using WakeWord.Core.Detection;

namespace WakeWord.Onnx;

/// <summary>
/// Silero VAD (MIT, models/vad/silero_vad.onnx) at 16 kHz. The model takes 512-sample windows plus the
/// previous 64 samples of context and carries a recurrent state, so any frame size works: a frame's
/// probability is the highest of the windows completed during it.
/// </summary>
public sealed class SileroVad : IVoiceActivityDetector
{
    private const int Window = 512;
    private const int Context = 64;
    private const int StateSize = 2 * 1 * 128;

    private readonly InferenceSession _session;
    private readonly RunOptions _run = new();
    private readonly float[] _input = new float[Context + Window];
    private readonly float[] _pending = new float[Window];
    private readonly long[] _sampleRate = [16000];
    private float[] _state = new float[StateSize];
    private int _pendingCount;
    private double _lastProbability;

    public SileroVad(byte[] model, OnnxModelOptions? options = null)
    {
        using var sessionOptions = new SessionOptions
        {
            IntraOpNumThreads = (options ?? new OnnxModelOptions()).IntraOpThreads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
        _session = new InferenceSession(model, sessionOptions);
    }

    public static SileroVad Load(string path, OnnxModelOptions? options = null) => new(File.ReadAllBytes(path), options);

    public double SpeechProbability(ReadOnlySpan<short> frame)
    {
        double max = -1;
        foreach (var sample in frame)
        {
            _pending[_pendingCount++] = sample / 32768f;
            if (_pendingCount == Window)
            {
                max = Math.Max(max, RunWindow());
                _pendingCount = 0;
            }
        }

        // A frame shorter than one window carries the last known probability.
        _lastProbability = max >= 0 ? max : _lastProbability;
        return _lastProbability;
    }

    public void Reset()
    {
        Array.Clear(_input);
        _state = new float[StateSize];
        _pendingCount = 0;
        _lastProbability = 0;
    }

    private double RunWindow()
    {
        _pending.CopyTo(_input.AsSpan(Context)); // context occupies the first 64 samples, from the previous window
        using var audio = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _input.AsMemory(), [1, Context + Window]);
        using var state = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _state.AsMemory(), [2, 1, 128]);
        using var sr = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _sampleRate.AsMemory(), []);
        using var outputs = _session.Run(_run, ["input", "state", "sr"], [audio, state, sr], ["output", "stateN"]);

        var probability = outputs[0].GetTensorDataAsSpan<float>()[0];
        outputs[1].GetTensorDataAsSpan<float>().CopyTo(_state);
        _input.AsSpan(Window, Context).CopyTo(_input); // last 64 samples become the next context
        return probability;
    }

    public void Dispose()
    {
        _session.Dispose();
        _run.Dispose();
    }
}
