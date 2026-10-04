namespace WakeWord.Core.Sessions;

/// <summary>On-device ceiling on Deepgram sessions per rolling window (plan 8.5). The broker enforces its own limit too.</summary>
public sealed class SessionRateLimiter(int maxSessions, TimeSpan window, TimeProvider? time = null)
{
    private readonly TimeProvider _time = time ?? TimeProvider.System;
    private readonly Queue<DateTimeOffset> _starts = new();
    private readonly Lock _gate = new();

    public bool TryAcquire()
    {
        var now = _time.GetUtcNow();
        lock (_gate)
        {
            while (_starts.TryPeek(out var oldest) && now - oldest >= window)
            {
                _starts.Dequeue();
            }

            if (_starts.Count >= maxSessions)
            {
                return false;
            }

            _starts.Enqueue(now);
            return true;
        }
    }
}
