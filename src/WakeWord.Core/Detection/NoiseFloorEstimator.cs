namespace WakeWord.Core.Detection;

/// <summary>
/// Rolling ambient noise estimate: follows quiet frames down quickly and rises slowly,
/// so speech bursts barely lift it but a TV switched on is tracked within seconds.
/// </summary>
public sealed class NoiseFloorEstimator(double fallFactor = 0.3, double riseDbPerSecond = 2.0)
{
    private bool _initialised;

    public double FloorDbfs { get; private set; } = Audio.AudioLevel.SilenceDbfs;

    public double Update(double frameDbfs, TimeSpan frameDuration)
    {
        if (!_initialised)
        {
            FloorDbfs = frameDbfs;
            _initialised = true;
        }
        else if (frameDbfs < FloorDbfs)
        {
            FloorDbfs += (frameDbfs - FloorDbfs) * fallFactor;
        }
        else
        {
            FloorDbfs += Math.Min(frameDbfs - FloorDbfs, riseDbPerSecond * frameDuration.TotalSeconds);
        }

        return FloorDbfs;
    }
}
