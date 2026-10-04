using WakeWord.Core.Audio;

namespace WakeWord.Core.Tests;

public class PcmRingBufferTests
{
    private static short[] Ramp(int start, int count) =>
        Enumerable.Range(start, count).Select(i => (short)(i % short.MaxValue)).ToArray();

    [Fact]
    public void Reads_across_the_wrap_point_in_order()
    {
        var ring = new PcmRingBuffer(10, TimeSpan.FromSeconds(1)); // 10 samples
        ring.Write(Ramp(0, 7));
        ring.Write(Ramp(7, 7)); // wraps; oldest is now 4

        Assert.Equal(14, ring.TotalWritten);
        Assert.Equal(4, ring.OldestAvailable);

        var dest = new short[10];
        Assert.Equal(10, ring.Read(4, dest));
        Assert.Equal(Ramp(4, 10), dest);
    }

    [Fact]
    public void Reading_overwritten_audio_throws_instead_of_returning_a_hole()
    {
        var ring = new PcmRingBuffer(10, TimeSpan.FromSeconds(1));
        ring.Write(Ramp(0, 25));

        var ex = Assert.Throws<PcmOverrunException>(() => ring.Read(3, new short[4]));
        Assert.Equal(15, ex.OldestAvailable);
    }

    [Fact]
    public void Write_larger_than_capacity_keeps_the_newest_samples()
    {
        var ring = new PcmRingBuffer(10, TimeSpan.FromSeconds(1));
        ring.Write(Ramp(0, 23));

        var dest = new short[10];
        Assert.Equal(10, ring.Read(13, dest));
        Assert.Equal(Ramp(13, 10), dest);
    }

    [Fact]
    public void Reading_at_the_write_head_returns_nothing()
    {
        var ring = new PcmRingBuffer(10, TimeSpan.FromSeconds(1));
        ring.Write(Ramp(0, 5));
        Assert.Equal(0, ring.Read(5, new short[4]));
        Assert.Equal(2, ring.Read(3, new short[4]));
    }

    [Fact]
    public async Task WaitForData_completes_when_the_position_is_written()
    {
        var ring = new PcmRingBuffer(10, TimeSpan.FromSeconds(1));
        ring.Write(Ramp(0, 5));

        var wait = ring.WaitForDataAsync(6, CancellationToken.None);
        Assert.False(wait.IsCompleted);

        ring.Write(Ramp(5, 1)); // position 5 only
        await Task.Delay(20, CancellationToken.None);
        Assert.False(wait.IsCompleted);

        ring.Write(Ramp(6, 1));
        await wait.WaitAsync(TimeSpan.FromSeconds(5), CancellationToken.None);
    }
}
