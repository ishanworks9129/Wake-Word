using Microsoft.Extensions.Time.Testing;
using WakeWord.Core.Deepgram;
using WakeWord.Core.Detection;
using WakeWord.Core.Sessions;
using WakeWord.Core.Transcripts;

namespace WakeWord.Core.Tests;

public class WakePhraseStripperTests
{
    private readonly WakePhraseStripper _stripper = new("Hey UNO", ["hey you know", "hey you no", "hey juno"]);

    [Theory]
    [InlineData("Hey UNO, what's the weather?", "what's the weather?")]
    [InlineData("hey you know what's the weather", "what's the weather")]
    [InlineData("Hey Juno. Set a timer for ten minutes.", "Set a timer for ten minutes.")]
    [InlineData("Um, hey UNO play some music", "play some music")]
    [InlineData("Hey, Uno.", "")]
    [InlineData("Hey you know", "")]
    [InlineData("What's the weather", "What's the weather")]
    [InlineData("", "")]
    public void Strips_the_phrase_and_common_mishearings(string transcript, string expected) =>
        Assert.Equal(expected, _stripper.Strip(transcript));
}

public class VadGateTests
{
    [Fact]
    public void Opens_on_onset_and_closes_after_the_hangover()
    {
        var gate = new VadGate(new VadGateOptions { OnsetProbability = 0.5, SustainProbability = 0.3, Hangover = TimeSpan.FromSeconds(0.2) }, 1000);

        Assert.Equal(VadTransition.None, gate.Update(0.4, 100));
        Assert.Equal(VadTransition.Opened, gate.Update(0.6, 200));
        Assert.Equal(VadTransition.None, gate.Update(0.35, 300)); // sustain keeps it open
        Assert.Equal(VadTransition.None, gate.Update(0.1, 400));
        Assert.Equal(VadTransition.None, gate.Update(0.1, 500)); // 200 ms since last speech: still inside hangover
        Assert.Equal(VadTransition.Closed, gate.Update(0.1, 501));
        Assert.False(gate.IsOpen);
    }
}

public class SessionRateLimiterTests
{
    [Fact]
    public void Allows_at_most_n_per_rolling_window()
    {
        var time = new FakeTimeProvider();
        var limiter = new SessionRateLimiter(2, TimeSpan.FromHours(1), time);

        Assert.True(limiter.TryAcquire());
        time.Advance(TimeSpan.FromMinutes(30));
        Assert.True(limiter.TryAcquire());
        Assert.False(limiter.TryAcquire());

        time.Advance(TimeSpan.FromMinutes(30));
        Assert.True(limiter.TryAcquire());
    }
}

public class CachingTokenProviderTests
{
    private sealed class CountingSource(FakeTimeProvider time) : ITokenSource
    {
        public int Fetches;
        public TaskCompletionSource? Gate;

        public async Task<DeepgramToken> FetchAsync(CancellationToken cancellationToken)
        {
            var n = Interlocked.Increment(ref Fetches);
            if (Gate is not null)
            {
                await Gate.Task;
            }

            return new DeepgramToken($"token-{n}", time.GetUtcNow() + TimeSpan.FromSeconds(300));
        }
    }

    [Fact]
    public async Task Reuses_a_token_until_80_percent_of_its_life_has_passed()
    {
        var time = new FakeTimeProvider();
        var source = new CountingSource(time);
        var provider = new CachingTokenProvider(source, time);
        var ct = CancellationToken.None;

        Assert.Equal("token-1", (await provider.GetTokenAsync(ct)).AccessToken);
        time.Advance(TimeSpan.FromSeconds(239));
        Assert.Equal("token-1", (await provider.GetTokenAsync(ct)).AccessToken);
        time.Advance(TimeSpan.FromSeconds(1));
        Assert.Equal("token-2", (await provider.GetTokenAsync(ct)).AccessToken);
        Assert.Equal(2, source.Fetches);
    }

    [Fact]
    public async Task Concurrent_callers_and_prefetch_share_one_fetch()
    {
        var time = new FakeTimeProvider();
        var source = new CountingSource(time) { Gate = new TaskCompletionSource() };
        var provider = new CachingTokenProvider(source, time);
        var ct = CancellationToken.None;

        provider.Prefetch();
        var a = provider.GetTokenAsync(ct).AsTask();
        var b = provider.GetTokenAsync(ct).AsTask();
        source.Gate.SetResult();

        Assert.Equal("token-1", (await a).AccessToken);
        Assert.Equal("token-1", (await b).AccessToken);
        provider.Prefetch(); // fresh: no new fetch
        Assert.Equal(1, source.Fetches);
    }
}

public class DeepgramStreamingOptionsTests
{
    [Fact]
    public void Builds_the_listen_uri_with_cost_and_privacy_settings()
    {
        var uri = new DeepgramStreamingOptions { Keyterms = ["UNO"] }.BuildUri(16000).ToString();

        Assert.StartsWith("wss://api.deepgram.com/v1/listen?model=nova-3", uri);
        Assert.Contains("encoding=linear16&sample_rate=16000&channels=1", uri);
        Assert.Contains("utterance_end_ms=1000", uri);
        Assert.Contains("mip_opt_out=true", uri);
        Assert.Contains("keyterm=UNO", uri);
    }
}
