namespace WakeWord.Core.Audio;

/// <summary>
/// Fixed-capacity ring of 16-bit mono PCM, addressed by absolute sample position.
/// The capture callback writes every frame unconditionally; a Deepgram session reads from
/// any position the ring still holds, so audio captured while the socket opens is never
/// dropped (plan 8.4: gapless pre-roll queue).
/// </summary>
public sealed class PcmRingBuffer
{
    private readonly short[] _buffer;
    private readonly Lock _gate = new();
    private long _written;
    private TaskCompletionSource _dataAvailable = NewSignal();

    public PcmRingBuffer(int sampleRate, TimeSpan capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        var samples = checked((int)Math.Round(sampleRate * capacity.TotalSeconds));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(samples, nameof(capacity));
        SampleRate = sampleRate;
        _buffer = new short[samples];
    }

    public int SampleRate { get; }

    public int Capacity => _buffer.Length;

    /// <summary>Total samples ever written; the position the next sample will occupy.</summary>
    public long TotalWritten
    {
        get { lock (_gate) return _written; }
    }

    /// <summary>Oldest absolute position still readable.</summary>
    public long OldestAvailable
    {
        get { lock (_gate) return Math.Max(0, _written - _buffer.Length); }
    }

    public long SamplesFor(TimeSpan duration) => (long)Math.Round(duration.TotalSeconds * SampleRate);

    public TimeSpan DurationOf(long samples) => TimeSpan.FromSeconds((double)samples / SampleRate);

    public void Write(ReadOnlySpan<short> samples)
    {
        if (samples.IsEmpty)
        {
            return;
        }

        TaskCompletionSource signal;
        lock (_gate)
        {
            if (samples.Length > _buffer.Length)
            {
                _written += samples.Length - _buffer.Length;
                samples = samples[^_buffer.Length..];
            }

            var start = (int)(_written % _buffer.Length);
            var first = Math.Min(samples.Length, _buffer.Length - start);
            samples[..first].CopyTo(_buffer.AsSpan(start));
            samples[first..].CopyTo(_buffer);
            _written += samples.Length;

            signal = _dataAvailable;
            _dataAvailable = NewSignal();
        }

        signal.TrySetResult();
    }

    /// <summary>
    /// Copies samples starting at <paramref name="position"/> into <paramref name="destination"/>.
    /// Returns the number copied, which is 0 when nothing has been written past the position yet.
    /// </summary>
    /// <exception cref="PcmOverrunException">The position has already been overwritten.</exception>
    public int Read(long position, Span<short> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        lock (_gate)
        {
            var oldest = Math.Max(0, _written - _buffer.Length);
            if (position < oldest)
            {
                throw new PcmOverrunException(position, oldest);
            }

            var count = (int)Math.Min(destination.Length, _written - position);
            if (count <= 0)
            {
                return 0;
            }

            var start = (int)(position % _buffer.Length);
            var first = Math.Min(count, _buffer.Length - start);
            _buffer.AsSpan(start, first).CopyTo(destination);
            _buffer.AsSpan(0, count - first).CopyTo(destination[first..]);
            return count;
        }
    }

    /// <summary>Completes once a sample exists at <paramref name="position"/>.</summary>
    public async Task WaitForDataAsync(long position, CancellationToken cancellationToken)
    {
        while (true)
        {
            Task signal;
            lock (_gate)
            {
                if (_written > position)
                {
                    return;
                }

                signal = _dataAvailable.Task;
            }

            await signal.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    private static TaskCompletionSource NewSignal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
}

/// <summary>A reader fell behind the ring's capacity; the audio it needed is gone.</summary>
public sealed class PcmOverrunException(long requested, long oldestAvailable)
    : Exception($"Sample {requested} was overwritten; oldest available is {oldestAvailable}.")
{
    public long Requested { get; } = requested;

    public long OldestAvailable { get; } = oldestAvailable;
}
