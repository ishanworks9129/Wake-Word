using System.Security.Claims;
using System.Threading.RateLimiting;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using WakeWord.TokenBroker;

// Token broker (plan 8.2): authenticated users get a short-lived Deepgram token; the API key never leaves the server.
var builder = WebApplication.CreateBuilder(args);
var config = builder.Configuration;

builder.Services.AddOptions<DeepgramGrantOptions>()
    .Bind(config.GetSection("Deepgram"))
    .ValidateDataAnnotations()
    .ValidateOnStart();
builder.Services.AddHttpClient<DeepgramGrantClient>(http => http.Timeout = TimeSpan.FromSeconds(5));
builder.Services.AddProblemDetails();

// Authentication fails closed: either a real JWT authority, or the dev header in Development only.
var authority = config["Auth:Authority"];
var devHeader = config.GetValue<bool>("Auth:DevHeader");
if (devHeader && !builder.Environment.IsDevelopment())
{
    throw new InvalidOperationException("Auth:DevHeader may only be enabled in the Development environment.");
}

if (string.IsNullOrWhiteSpace(authority) && !devHeader)
{
    throw new InvalidOperationException("Configure Auth:Authority (and Auth:Audience) for the app's identity provider.");
}

var auth = builder.Services.AddAuthentication(devHeader ? DevHeaderAuthenticationHandler.SchemeName : JwtBearerDefaults.AuthenticationScheme);
if (!string.IsNullOrWhiteSpace(authority))
{
    auth.AddJwtBearer(o =>
    {
        o.Authority = authority;
        o.Audience = config["Auth:Audience"];
        o.MapInboundClaims = false;
    });
}

if (devHeader)
{
    auth.AddScheme<AuthenticationSchemeOptions, DevHeaderAuthenticationHandler>(DevHeaderAuthenticationHandler.SchemeName, null);
}

builder.Services.AddAuthorization();

// Per-user ceiling so one compromised client cannot run up the Deepgram bill (plan 8.5).
var permitsPerHour = config.GetValue("TokenRateLimit:PermitsPerHour", 120);
builder.Services.AddRateLimiter(o =>
{
    o.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
    o.AddPolicy("per-user-token", context => RateLimitPartition.GetSlidingWindowLimiter(
        UserId(context.User) ?? "anonymous",
        _ => new SlidingWindowRateLimiterOptions
        {
            PermitLimit = permitsPerHour,
            Window = TimeSpan.FromHours(1),
            SegmentsPerWindow = 6,
            QueueLimit = 0,
        }));
});

var corsOrigins = config.GetSection("Cors:AllowedOrigins").Get<string[]>() ?? [];
string[] corsHeaders = devHeader ? ["Authorization", "Content-Type", DevHeaderAuthenticationHandler.Header] : ["Authorization", "Content-Type"];
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p.WithOrigins(corsOrigins).WithMethods("POST").WithHeaders(corsHeaders)));

var app = builder.Build();

app.UseExceptionHandler();
app.UseCors();
app.UseAuthentication();
app.UseAuthorization();
app.UseRateLimiter();

app.MapGet("/healthz", () => Results.Ok());

app.MapPost("/v1/deepgram/token", async (DeepgramGrantClient deepgram, ClaimsPrincipal user, ILogger<Program> log, CancellationToken ct) =>
    {
        try
        {
            var token = await deepgram.GrantAsync(ct);
            log.LogInformation("Issued Deepgram token to {User} for {Seconds} s", UserId(user), token.ExpiresIn);
            return Results.Ok(new { accessToken = token.AccessToken, expiresIn = token.ExpiresIn });
        }
        catch (Exception ex) when (ex is DeepgramGrantException or HttpRequestException or TaskCanceledException && !ct.IsCancellationRequested)
        {
            log.LogWarning(ex, "Deepgram token grant failed for {User}", UserId(user));
            return Results.Problem("Speech service unavailable.", statusCode: StatusCodes.Status502BadGateway);
        }
    })
    .RequireAuthorization()
    .RequireRateLimiting("per-user-token")
    .RequireCors();

app.Run();

static string? UserId(ClaimsPrincipal user) =>
    user.FindFirstValue(ClaimTypes.NameIdentifier) ?? user.FindFirstValue("sub");

public partial class Program;
