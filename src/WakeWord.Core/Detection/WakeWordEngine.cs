using WakeWord.Core.Audio;

namespace WakeWord.Core.Detection;

/// <summary>
/// The always-on loop (plan 8.1). Feed it captured audio in any chunk size; it writes every
/// sample to the pre-roll ring first, then runs mel features, VAD, and, behind the gate,
/// the embedding and each keyword's classifier. Not thread-safe; call <see cref="Process"/> from one capture thread.
/// </summary>
public sealed class WakeWordEngine
{
    private readonly IWakeWordModel _model;
    private readonly IVoiceActivityDetector _vad;
    private readonly PcmRingBuffer _ring;
    private readonly VadGate _gate;
    private readonly WakeWordDetector[] _detectors;
    private readonly double[] _scores;
    private readonly NoiseFloorEstimator _noise = new();
    private readonly short[] _frame;
    private readonly TimeSpan _frameDuration;
    private int _frameFill;
    private long _framedSamples;

    /// <param name="detectorOptions">One per keyword, in <see cref="IWakeWordModel.Keywords"/> order.</param>
    public WakeWordEngine(
        IWakeWordModel model,
        IVoiceActivityDetector vad,
        PcmRingBuffer ring,
        IReadOnlyList<WakeWordDetectorOptions> detectorOptions,
        VadGateOptions vadOptions)
    {
        if (detectorOptions.Count != model.Keywords.Count)
        {
            throw new ArgumentException($"Expected {model.Keywords.Count} detector options, one per keyword; got {detectorOptions.Count}.", nameof(detectorOptions));
        }

        _model = model;
        _vad = vad;
        _ring = ring;
        _gate = new VadGate(vadOptions, ring.SampleRate);
        _detectors = detectorOptions.Select(o => new WakeWordDetector(o, ring.SampleRate)).ToArray();
        _scores = new double[_detectors.Length];
        _frame = new short[model.FrameSamples];
        _frameDuration = ring.DurationOf(model.FrameSamples);
    }

    /// <summary>Raised on the capture thread when a keyword fires. Handlers must not block.</summary>
    public event EventHandler<WakeWordDetection>? Detected;

    /// <summary>Raised when the VAD gate opens; used to prefetch a Deepgram token.</summary>
    public event EventHandler? SpeechStarted;

    /// <summary>
    /// Pauses detection, for example while the app plays its own audio or TTS (plan 6.1 barge-in policy).
    /// Audio still reaches the pre-roll ring.
    /// </summary>
    public bool Paused { get; set; }

    public IReadOnlyList<string> Keywords => _model.Keywords;

    public long FramesProcessed { get; private set; }

    /// <summary>Frames that ran the embedding and classifiers; the ratio to <see cref="FramesProcessed"/> is the VAD-gating saving.</summary>
    public long FramesScored { get; private set; }

    public double NoiseFloorDbfs => _noise.FloorDbfs;

    /// <summary>Latest score per keyword.</summary>
    public IReadOnlyList<double> LastScores => _scores;

    public void Process(ReadOnlySpan<short> samples)
    {
        _ring.Write(samples);

        while (!samples.IsEmpty)
        {
            var take = Math.Min(samples.Length, _frame.Length - _frameFill);
            samples[..take].CopyTo(_frame.AsSpan(_frameFill));
            samples = samples[take..];
            _frameFill += take;

            if (_frameFill == _frame.Length)
            {
                _frameFill = 0;
                _framedSamples += _frame.Length;
                ProcessFrame(_frame, _framedSamples);
            }
        }
    }

    private void ProcessFrame(ReadOnlySpan<short> frame, long frameEndSample)
    {
        FramesProcessed++;
        _noise.Update(AudioLevel.Dbfs(frame), _frameDuration);
        _model.AppendFrame(frame);

        var transition = _gate.Update(_vad.SpeechProbability(frame), frameEndSample);
        if (transition == VadTransition.Opened)
        {
            SpeechStarted?.Invoke(this, EventArgs.Empty);
        }

        if (!_gate.IsOpen || Paused)
        {
            foreach (var d in _detectors)
            {
                d.Reset();
            }

            return;
        }

        FramesScored++;
        if (!_model.Score(_scores))
        {
            return;
        }

        for (var k = 0; k < _detectors.Length; k++)
        {
            if (_detectors[k].Process(_scores[k], _noise.FloorDbfs, frameEndSample) is { } detection)
            {
                Detected?.Invoke(this, detection with { KeywordIndex = k, Keyword = _model.Keywords[k] });
            }
        }
    }
}
