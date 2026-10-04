using System.Net.Http.Json;

namespace WakeWord.Core.Deepgram;

/// <summary>A short-lived Deepgram access token. Only needs to be valid when the socket opens.</summary>
public sealed record DeepgramToken(string AccessToken, DateTimeOffset ExpiresAt);

/// <summary>Fetches a fresh token, normally from the token broker.</summary>
public interface ITokenSource
{
    Task<DeepgramToken> FetchAsync(CancellationToken cancellationToken);
}

public interface ITokenProvider
{
    ValueTask<DeepgramToken> GetTokenAsync(CancellationToken cancellationToken);

    /// <summary>Starts a background refresh if the cached token is stale. Called on VAD onset.</summary>
    void Prefetch();
}

/// <summary>
/// Keeps a token ready so a fetch is never on the wake-word trigger path (plan 8.2).
/// A cached token is reused until 80% of its lifetime has passed; concurrent callers share one fetch.
/// </summary>
public sealed class CachingTokenProvider(ITokenSource source, TimeProvider? time = null, double refreshAtFraction = 0.8)
    : ITokenProvider
{
    private readonly TimeProvider _time = time ?? TimeProvider.System;
    private readonly Lock _gate = new();
    private DeepgramToken? _token;
    private DateTimeOffset _refreshAt;
    private Task<DeepgramToken>? _inflight;

    public async ValueTask<DeepgramToken> GetTokenAsync(CancellationToken cancellationToken)
    {
        Task<DeepgramToken> fetch;
        lock (_gate)
        {
            if (_token is { } token && _time.GetUtcNow() < _refreshAt)
            {
                return token;
            }

            fetch = StartFetchLocked();
        }

        return await fetch.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    public void Prefetch()
    {
        Task<DeepgramToken> fetch;
        lock (_gate)
        {
            if (_token is not null && _time.GetUtcNow() < _refreshAt)
            {
                return;
            }

            fetch = StartFetchLocked();
        }

        // Failures surface on the next GetTokenAsync; observe them here so they are not unobserved.
        fetch.ContinueWith(static t => _ = t.Exception, TaskContinuationOptions.OnlyOnFaulted);
    }

    private Task<DeepgramToken> StartFetchLocked()
    {
        if (_inflight is { IsCompleted: false } running)
        {
            return running;
        }

        _inflight = FetchAndStoreAsync();
        return _inflight;
    }

    private async Task<DeepgramToken> FetchAndStoreAsync()
    {
        var issuedAt = _time.GetUtcNow();
        var token = await source.FetchAsync(CancellationToken.None).ConfigureAwait(false);
        lock (_gate)
        {
            _token = token;
            _refreshAt = issuedAt + (token.ExpiresAt - issuedAt) * refreshAtFraction;
        }

        return token;
    }
}

/// <summary>
/// Gets tokens from the token broker. Configure <paramref name="http"/> with the app's own user
/// authentication (for example a bearer handler); the broker never sees a Deepgram key from the client.
/// </summary>
public sealed class BrokerTokenSource(HttpClient http, Uri tokenEndpoint, TimeProvider? time = null) : ITokenSource
{
    private readonly TimeProvider _time = time ?? TimeProvider.System;

    public async Task<DeepgramToken> FetchAsync(CancellationToken cancellationToken)
    {
        var requestedAt = _time.GetUtcNow();
        using var response = await http.PostAsync(tokenEndpoint, content: null, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<BrokerTokenResponse>(cancellationToken).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Token broker returned an empty body.");
        return new DeepgramToken(body.AccessToken, requestedAt + TimeSpan.FromSeconds(body.ExpiresIn));
    }

    private sealed record BrokerTokenResponse(string AccessToken, int ExpiresIn);
}
