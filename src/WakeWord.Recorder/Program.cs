using System.Security.Cryptography;
using System.Text;
using System.Threading.RateLimiting;
using Microsoft.AspNetCore.Http.Features;
using WakeWord.Recorder;

// Collects real wake-phrase recordings with consent (plan 5.2). Contributors open the page on their own phone.
var builder = WebApplication.CreateBuilder(args);
var options = builder.Configuration.GetSection("Recorder").Get<RecorderOptions>() ?? new RecorderOptions();
if (string.IsNullOrWhiteSpace(options.InviteCode) || string.IsNullOrWhiteSpace(options.AdminKey))
{
    throw new InvalidOperationException("Set Recorder:InviteCode and Recorder:AdminKey (user-secrets or environment variables).");
}

builder.Services.AddSingleton(options);
builder.Services.AddSingleton(TimeProvider.System);
builder.Services.AddSingleton<RecordingStore>();
builder.Services.AddProblemDetails();
builder.Services.AddRateLimiter(o =>
{
    o.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
    o.AddPolicy("per-ip", ctx => RateLimitPartition.GetFixedWindowLimiter(
        ctx.Connection.RemoteIpAddress?.ToString() ?? "unknown",
        _ => new FixedWindowRateLimiterOptions { PermitLimit = 300, Window = TimeSpan.FromHours(1) }));
});

var app = builder.Build();
app.UseExceptionHandler();
app.UseDefaultFiles();
app.UseStaticFiles();
app.UseRateLimiter();

var api = app.MapGroup("/api").RequireRateLimiting("per-ip");

api.MapGet("/config", () => new { consentVersion = options.ConsentVersion, minSeconds = options.MinSeconds, maxSeconds = options.MaxSeconds });

api.MapGet("/prompts", () => Prompts.All.Select(p => new { p.Id, p.Text, p.Instruction, kind = p.Kind.ToString() }));

api.MapPost("/participants", (JoinRequest request, RecordingStore store) =>
{
    if (!FixedTimeEquals(request.InviteCode?.Trim() ?? string.Empty, options.InviteCode))
    {
        return Results.Problem("That invite code isn't right.", statusCode: StatusCodes.Status403Forbidden);
    }

    if (!request.Agreed || request.ConsentVersion != options.ConsentVersion)
    {
        return Results.Problem("Please read and accept the current consent form.", statusCode: StatusCodes.Status400BadRequest);
    }

    var (participant, code) = store.CreateParticipant(new ParticipantProfile(
        Trim(request.AgeBand), Trim(request.Accent), Trim(request.Gender), Trim(request.Device, 200)));
    return Results.Ok(new { participantId = participant.Id, withdrawalCode = code });
});

api.MapGet("/participants/{id}/recordings", (string id, RecordingStore store) =>
    store.Find(id) is null ? Results.NotFound() : Results.Ok(store.RecordedPrompts(id)));

api.MapPut("/participants/{id}/recordings/{promptId}", async (string id, string promptId, HttpRequest http, RecordingStore store, CancellationToken ct) =>
{
    var participant = store.Find(id);
    var prompt = Prompts.Find(promptId);
    if (participant is null || prompt is null)
    {
        return Results.NotFound();
    }

    const int maxBytes = 512 * 1024; // 6 s of 16 kHz 16-bit audio is under 200 KB
    if (http.HttpContext.Features.Get<IHttpMaxRequestBodySizeFeature>() is { IsReadOnly: false } limit)
    {
        limit.MaxRequestBodySize = maxBytes;
    }

    var buffer = new byte[maxBytes + 1];
    var length = 0;
    int read;
    while (length < buffer.Length && (read = await http.Body.ReadAsync(buffer.AsMemory(length), ct)) > 0)
    {
        length += read;
    }

    if (length > maxBytes)
    {
        return Results.Problem("Recording too large.", statusCode: StatusCodes.Status413PayloadTooLarge);
    }

    var wav = buffer[..length];
    var (info, error) = WavValidator.Validate(wav, TimeSpan.FromSeconds(options.MinSeconds), TimeSpan.FromSeconds(options.MaxSeconds), options.MinPeak);
    if (info is null)
    {
        return Results.Problem(error, statusCode: StatusCodes.Status422UnprocessableEntity);
    }

    store.SaveRecording(participant, prompt, wav, info);
    return Results.Ok(new { seconds = Math.Round(info.Duration.TotalSeconds, 2) });
});

api.MapPost("/participants/{id}/withdraw", (string id, WithdrawRequest request, RecordingStore store) =>
    store.Withdraw(id, request.WithdrawalCode ?? string.Empty)
        ? Results.Ok(new { deleted = true })
        : Results.Problem("Participant or withdrawal code not recognised.", statusCode: StatusCodes.Status403Forbidden));

var admin = api.MapGroup("/admin").AddEndpointFilter(async (ctx, next) =>
    FixedTimeEquals(ctx.HttpContext.Request.Headers["X-Admin-Key"].ToString(), options.AdminKey)
        ? await next(ctx)
        : Results.StatusCode(StatusCodes.Status401Unauthorized));

admin.MapGet("/stats", (RecordingStore store) => store.Stats());

admin.MapGet("/export", async (RecordingStore store, CancellationToken ct) =>
{
    // Built in a temp file (deleted when sent) so a large collection never sits in memory.
    var tmp = Path.GetTempFileName();
    await using (var file = File.Create(tmp))
    {
        await store.ExportAsync(file, ct);
    }

    var stream = new FileStream(tmp, FileMode.Open, FileAccess.Read, FileShare.None, 81920, FileOptions.DeleteOnClose);
    return Results.File(stream, "application/zip", $"recordings-{DateTime.UtcNow:yyyyMMdd-HHmm}.zip");
});

app.Run();

static bool FixedTimeEquals(string a, string b) =>
    b.Length > 0 && CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(a), Encoding.UTF8.GetBytes(b));

static string? Trim(string? s, int max = 60) => string.IsNullOrWhiteSpace(s) ? null : s.Trim()[..Math.Min(s.Trim().Length, max)];

public sealed record JoinRequest(string? InviteCode, string? ConsentVersion, bool Agreed, string? AgeBand, string? Accent, string? Gender, string? Device);

public sealed record WithdrawRequest(string? WithdrawalCode);

public partial class Program;
