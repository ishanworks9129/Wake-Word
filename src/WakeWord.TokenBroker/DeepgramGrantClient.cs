using System.ComponentModel.DataAnnotations;
using System.Net;
using System.Net.Http.Headers;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Options;

namespace WakeWord.TokenBroker;

public sealed class DeepgramGrantOptions
{
    /// <summary>Server-side Deepgram key (Member role or higher). Set via user-secrets or the Deepgram__ApiKey environment variable; never in source.</summary>
    [Required]
    public string ApiKey { get; set; } = string.Empty;

    public Uri GrantUrl { get; set; } = new("https://api.deepgram.com/v1/auth/grant");

    /// <summary>Clients refresh at 80% of this, so 300 s means about 15 broker calls per listening user per hour.</summary>
    [Range(1, 3600)]
    public int TokenTtlSeconds { get; set; } = 300;
}

public sealed record GrantedToken(string AccessToken, int ExpiresIn);

public sealed class DeepgramGrantException(HttpStatusCode status)
    : Exception($"Deepgram token grant failed with {(int)status}.")
{
    public HttpStatusCode Status { get; } = status;
}

/// <summary>Calls Deepgram's token grant endpoint, which issues short-lived JWTs for the streaming API.</summary>
public sealed class DeepgramGrantClient(HttpClient http, IOptions<DeepgramGrantOptions> options)
{
    public async Task<GrantedToken> GrantAsync(CancellationToken cancellationToken)
    {
        var o = options.Value;
        using var request = new HttpRequestMessage(HttpMethod.Post, o.GrantUrl)
        {
            Content = JsonContent.Create(new GrantRequest(o.TokenTtlSeconds)),
        };
        request.Headers.Authorization = new AuthenticationHeaderValue("Token", o.ApiKey);

        using var response = await http.SendAsync(request, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            throw new DeepgramGrantException(response.StatusCode);
        }

        var body = await response.Content.ReadFromJsonAsync<GrantResponse>(cancellationToken)
            ?? throw new DeepgramGrantException(HttpStatusCode.BadGateway);
        return new GrantedToken(body.AccessToken, body.ExpiresIn);
    }

    private sealed record GrantRequest([property: JsonPropertyName("ttl_seconds")] int TtlSeconds);

    private sealed record GrantResponse(
        [property: JsonPropertyName("access_token")] string AccessToken,
        [property: JsonPropertyName("expires_in")] int ExpiresIn);
}
