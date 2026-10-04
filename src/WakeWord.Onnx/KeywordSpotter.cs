using WakeWord.Core.Detection;

namespace WakeWord.Onnx;

/// <summary>
/// Porcupine-style wake word API: pick keywords and sensitivities, feed 1280-sample frames, get back the
/// index of the keyword that fired or -1.
/// <code>
/// using var spotter = KeywordSpotter.Create("models", ["hey_uno", "hello_uno"], [0.5, 0.5]);
/// int keyword = spotter.Process(frame); // frame: spotter.FrameLength samples of 16 kHz mono PCM
/// </code>
/// Runs every frame through the model. For an always-on app with VAD gating, pre-roll and Deepgram
/// handoff, use WakeWordController with <see cref="OnnxWakeWordModel"/> and <see cref="SileroVad"/> instead.
/// Not thread-safe.
/// </summary>
public sealed class KeywordSpotter : IDisposable
{
    private readonly OnnxWakeWordModel _model;
    private readonly WakeWordDetector[] _detectors;
    private readonly double[] _scores;
    private long _samples;

    private KeywordSpotter(OnnxWakeWordModel model, IReadOnlyList<WakeWordDetectorOptions> detectors)
    {
        _model = model;
        _detectors = detectors.Select(o => new WakeWordDetector(o, SampleRate)).ToArray();
        _scores = new double[_detectors.Length];
    }

    public static KeywordSpotter Create(
        string modelDirectory,
        IReadOnlyList<string>? keywords = null,
        IReadOnlyList<double>? sensitivities = null,
        OnnxModelOptions? options = null) =>
        Create(ModelPackage.LoadFromDirectory(modelDirectory), keywords, sensitivities, options);

    /// <param name="keywords">Keyword ids from models.json; all of them when null.</param>
    /// <param name="sensitivities">0 (fewest false accepts) to 1 (fewest misses), one per keyword; package defaults when null.</param>
    public static KeywordSpotter Create(
        ModelPackage package,
        IReadOnlyList<string>? keywords = null,
        IReadOnlyList<double>? sensitivities = null,
        OnnxModelOptions? options = null)
    {
        var chosen = package.KeywordOptions(keywords, sensitivities);
        var model = new OnnxWakeWordModel(package, chosen.Select(k => k.Id).ToList(), options);
        return new KeywordSpotter(model, chosen.Select(k => k.Detector).ToList());
    }

    public int SampleRate => 16000;

    public int FrameLength => OnnxWakeWordModel.FrameLength;

    public IReadOnlyList<string> Keywords => _model.Keywords;

    /// <summary>Scores from the last frame, one per keyword (0 during the first ~2 s warm-up).</summary>
    public IReadOnlyList<double> LastScores => _scores;

    /// <returns>Index into <see cref="Keywords"/> of the keyword that fired on this frame, or -1.
    /// If several fire on the same frame, the highest score wins.</returns>
    public int Process(ReadOnlySpan<short> frame)
    {
        _model.AppendFrame(frame);
        _samples += frame.Length;
        if (!_model.Score(_scores))
        {
            return -1;
        }

        var fired = -1;
        for (var k = 0; k < _detectors.Length; k++)
        {
            if (_detectors[k].Process(_scores[k], Core.Audio.AudioLevel.SilenceDbfs, _samples) is not null
                && (fired < 0 || _scores[k] > _scores[fired]))
            {
                fired = k;
            }
        }

        return fired;
    }

    public void Dispose() => _model.Dispose();
}
