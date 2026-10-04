using WakeWord.Core.Audio;

namespace WakeWord.Core.Detection;

/// <summary>
/// The always-on loop (plan 8.1). Feed it captured audio in any chunk size; it writes every
/// sample to the pre-roll ring first, then runs mel features, VAD, and, behind the gate,
/// the embedding and classifier. Not thread-safe; call <see cref="Process"/> from one capture thread.
/// </summary>
public sealed class WakeWordEngine
{
    private readonly IWakeWordModel _model;
    private readonly IVoiceActivityDetector _vad;
    private readonly PcmRingBuffer _ring;
    private readonly VadGate _gate;
    private readonly WakeWordDetector _detector;
    private readonly NoiseFloorEstimator _noise = new();
    private readonly short[] _frame;
    private readonly TimeSpan _frameDuration;
    private int _frameFill;
    private long _framedSamples;

    public WakeWordEngine(
        IWakeWordModel model,
        IVoiceActivityDetector vad,
        PcmRingBuffer ring,
        WakeWordDetectorOptions detectorOptions,
        VadGateOptions vadOptions)
    {
        _model = model;
        _vad = vad;
        _ring = ring;
        _gate = new VadGate(vadOptions, ring.SampleRate);
        _detector = new WakeWordDetector(detectorOptions, ring.SampleRate);
        _frame = new short[model.FrameSamples];
        _frameDuration = ring.DurationOf(model.FrameSamples);
    }

    /// <summary>Raised on the capture thread when the wake word fires. Handlers must not block.</summary>
    public event EventHandler<WakeWordDetection>? Detected;

    /// <summary>Raised when the VAD gate opens; used to prefetch a Deepgram token.</summary>
    public event EventHandler? SpeechStarted;

    /// <summary>
    /// Pauses detection, for example while the app plays its own audio or TTS (plan 6.1 barge-in policy).
    /// Audio still reaches the pre-roll ring.
    /// </summary>
    public bool Paused { get; set; }

    public long FramesProcessed { get; private set; }

    /// <summary>Frames that ran the embedding and classifier; the ratio to <see cref="FramesProcessed"/> is the VAD-gating saving.</summary>
    public long FramesScored { get; private set; }

    public double NoiseFloorDbfs => _noise.FloorDbfs;

    public double LastScore { get; private set; }

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
            _detector.Reset();
            return;
        }

        FramesScored++;
        LastScore = _model.Score();
        var detection = _detector.Process(LastScore, _noise.FloorDbfs, frameEndSample);
        if (detection is not null)
        {
            Detected?.Invoke(this, detection);
        }
    }
}
