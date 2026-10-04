using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using WakeWord.TokenBroker;

namespace WakeWord.TokenBroker.Tests;

public class TokenEndpointTests
{
    private sealed class FakeDeepgram : HttpMessageHandler
    {
        public HttpStatusCode Status { get; set; } = HttpStatusCode.OK;

        public List<(string? Auth, string Body)> Requests { get; } = [];

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add((request.Headers.Authorization?.ToString(), await request.Content!.ReadAsStringAsync(cancellationToken)));
            return new HttpResponseMessage(Status)
            {
                Content = JsonContent.Create(new { access_token = "jwt-from-deepgram", expires_in = 300 }),
            };
        }
    }

    private static (WebApplicationFactory<Program> Factory, FakeDeepgram Deepgram) Create(int permitsPerHour = 3)
    {
        var deepgram = new FakeDeepgram();
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseEnvironment("Development");
            b.UseSetting("Deepgram:ApiKey", "server-secret-key");
            b.UseSetting("Auth:DevHeader", "true");
            b.UseSetting("TokenRateLimit:PermitsPerHour", permitsPerHour.ToString());
            b.ConfigureTestServices(s => s.AddHttpClient<DeepgramGrantClient>().ConfigurePrimaryHttpMessageHandler(() => deepgram));
        });
        return (factory, deepgram);
    }

    private static HttpRequestMessage TokenRequest(string? user)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/v1/deepgram/token");
        if (user is not null)
        {
            request.Headers.Add(DevHeaderAuthenticationHandler.Header, user);
        }

        return request;
    }

    [Fact]
    public async Task Rejects_unauthenticated_callers()
    {
        var (factory, deepgram) = Create();
        using var client = factory.CreateClient();

        var response = await client.SendAsync(TokenRequest(null));

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
        Assert.Empty(deepgram.Requests);
    }

    [Fact]
    public async Task Issues_a_short_lived_token_without_exposing_the_api_key()
    {
        var (factory, deepgram) = Create();
        using var client = factory.CreateClient();

        var response = await client.SendAsync(TokenRequest("user-1"));
        var body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var json = JsonDocument.Parse(body);
        Assert.Equal("jwt-from-deepgram", json.RootElement.GetProperty("accessToken").GetString());
        Assert.Equal(300, json.RootElement.GetProperty("expiresIn").GetInt32());
        Assert.DoesNotContain("server-secret-key", body);

        var (auth, grantBody) = Assert.Single(deepgram.Requests);
        Assert.Equal("Token server-secret-key", auth);
        Assert.Contains("\"ttl_seconds\":300", grantBody);
    }

    [Fact]
    public async Task Rate_limits_each_user_separately()
    {
        var (factory, _) = Create(permitsPerHour: 2);
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, (await client.SendAsync(TokenRequest("user-1"))).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.SendAsync(TokenRequest("user-1"))).StatusCode);
        Assert.Equal(HttpStatusCode.TooManyRequests, (await client.SendAsync(TokenRequest("user-1"))).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.SendAsync(TokenRequest("user-2"))).StatusCode);
    }

    [Fact]
    public async Task Returns_502_when_deepgram_refuses()
    {
        var (factory, deepgram) = Create();
        deepgram.Status = HttpStatusCode.Forbidden;
        using var client = factory.CreateClient();

        var response = await client.SendAsync(TokenRequest("user-1"));

        Assert.Equal(HttpStatusCode.BadGateway, response.StatusCode);
    }

    [Fact]
    public void Refuses_to_start_with_the_dev_header_outside_development()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseEnvironment("Production");
            b.UseSetting("Deepgram:ApiKey", "k");
            b.UseSetting("Auth:DevHeader", "true");
        });

        var ex = Assert.ThrowsAny<Exception>(() => factory.CreateClient());
        Assert.Contains("Development", ex.ToString());
    }
}
