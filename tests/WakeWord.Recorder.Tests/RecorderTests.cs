using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;

namespace WakeWord.Recorder.Tests;

public sealed class RecorderTests : IDisposable
{
    private readonly string _storage = Directory.CreateTempSubdirectory("recorder").FullName;
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public RecorderTests()
    {
        _factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseSetting("Recorder:StoragePath", _storage);
            b.UseSetting("Recorder:InviteCode", "uno-2026");
            b.UseSetting("Recorder:AdminKey", "admin-secret");
        });
        _client = _factory.CreateClient();
    }

    public void Dispose()
    {
        _client.Dispose();
        _factory.Dispose();
        Directory.Delete(_storage, recursive: true);
    }

    private static byte[] Wav(double seconds, int rate = 16000, short amplitude = 8000, int channels = 1)
    {
        var samples = (int)(seconds * rate) * channels;
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8);
        w.Write(36 + samples * 2);
        w.Write("WAVEfmt "u8);
        w.Write(16);
        w.Write((short)1);
        w.Write((short)channels);
        w.Write(rate);
        w.Write(rate * 2 * channels);
        w.Write((short)(2 * channels));
        w.Write((short)16);
        w.Write("data"u8);
        w.Write(samples * 2);
        for (var i = 0; i < samples; i++)
        {
            w.Write((short)(Math.Sin(i / 5.0) * amplitude));
        }

        return ms.ToArray();
    }

    private async Task<(string Id, string Code)> Join()
    {
        var res = await _client.PostAsJsonAsync("/api/participants", new { inviteCode = "uno-2026", consentVersion = "v1-draft", agreed = true, accent = "Indian English" });
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        return (body.GetProperty("participantId").GetString()!, body.GetProperty("withdrawalCode").GetString()!);
    }

    private Task<HttpResponseMessage> Upload(string id, string prompt, byte[] wav) =>
        _client.PutAsync($"/api/participants/{id}/recordings/{prompt}", new ByteArrayContent(wav));

    [Fact]
    public async Task Serves_the_page_prompts_and_consent()
    {
        Assert.Contains("Help train", await _client.GetStringAsync("/"));
        Assert.Contains("DRAFT", await _client.GetStringAsync("/consent/v1-draft.html"));
        var prompts = await _client.GetFromJsonAsync<JsonElement[]>("/api/prompts");
        Assert.Equal(26, prompts!.Length);
        Assert.Equal(10, prompts.Count(p => p.GetProperty("text").GetString() == "Hey UNO"));
    }

    [Fact]
    public async Task Joining_needs_the_invite_code_and_current_consent()
    {
        var wrongCode = await _client.PostAsJsonAsync("/api/participants", new { inviteCode = "guess", consentVersion = "v1-draft", agreed = true });
        Assert.Equal(HttpStatusCode.Forbidden, wrongCode.StatusCode);

        var oldConsent = await _client.PostAsJsonAsync("/api/participants", new { inviteCode = "uno-2026", consentVersion = "v0", agreed = true });
        Assert.Equal(HttpStatusCode.BadRequest, oldConsent.StatusCode);

        var notAgreed = await _client.PostAsJsonAsync("/api/participants", new { inviteCode = "uno-2026", consentVersion = "v1-draft", agreed = false });
        Assert.Equal(HttpStatusCode.BadRequest, notAgreed.StatusCode);
    }

    [Fact]
    public async Task Stores_valid_recordings_with_manifest_rows_and_replaces_retakes()
    {
        var (id, _) = await Join();

        Assert.Equal(HttpStatusCode.OK, (await Upload(id, "hey_uno.normal", Wav(1.2))).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await Upload(id, "near.play_uno", Wav(1.5))).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await Upload(id, "hey_uno.normal", Wav(1.0))).StatusCode); // retake

        Assert.Equal(["hey_uno.normal", "near.play_uno"], (await _client.GetFromJsonAsync<string[]>($"/api/participants/{id}/recordings"))!);

        var manifest = File.ReadAllLines(Path.Combine(_storage, "manifest.csv"));
        Assert.Equal("path,kind,split,source,url,license,license_url,retrieved,duration_seconds,speaker_id,release_id", manifest[0]);
        Assert.Equal(3, manifest.Length);
        var heyRow = manifest.Single(l => l.Contains("hey_uno.normal")).Split(',');
        Assert.Equal(["positive", "Contributor-Release", "1", id, $"v1-draft:{id}"], [heyRow[1], heyRow[5], heyRow[8], heyRow[9], heyRow[10]]);
        Assert.Contains(heyRow[2], new[] { "train", "dev", "test" });
        Assert.Contains("hard_negative", manifest.Single(l => l.Contains("near.play_uno")));
    }

    [Theory]
    [InlineData(1.0, 44100, 8000, 1)] // wrong sample rate
    [InlineData(1.0, 16000, 8000, 2)] // stereo
    [InlineData(0.2, 16000, 8000, 1)] // too short
    [InlineData(7.0, 16000, 8000, 1)] // too long
    [InlineData(1.0, 16000, 50, 1)]   // silent
    public async Task Rejects_audio_training_cannot_use(double seconds, int rate, short amplitude, int channels)
    {
        var (id, _) = await Join();
        var res = await Upload(id, "hey_uno.normal", Wav(seconds, rate, amplitude, channels));
        Assert.Equal(HttpStatusCode.UnprocessableEntity, res.StatusCode);
        Assert.False(File.Exists(Path.Combine(_storage, "participants", id, "hey_uno.normal.wav")));
    }

    [Fact]
    public async Task Unknown_participant_or_prompt_is_not_found()
    {
        var (id, _) = await Join();
        Assert.Equal(HttpStatusCode.NotFound, (await Upload(new string('a', 32), "hey_uno.normal", Wav(1))).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await Upload(id, "../../evil", Wav(1))).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await Upload("..%2F..%2Fx", "hey_uno.normal", Wav(1))).StatusCode);
    }

    [Fact]
    public async Task Withdrawal_deletes_audio_and_manifest_rows_only_with_the_right_code()
    {
        var (id, code) = await Join();
        var (other, _) = await Join();
        await Upload(id, "hey_uno.normal", Wav(1));
        await Upload(other, "hey_uno.normal", Wav(1));

        var wrong = await _client.PostAsJsonAsync($"/api/participants/{id}/withdraw", new { withdrawalCode = "WRONG" });
        Assert.Equal(HttpStatusCode.Forbidden, wrong.StatusCode);

        var ok = await _client.PostAsJsonAsync($"/api/participants/{id}/withdraw", new { withdrawalCode = code.ToLowerInvariant() });
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        Assert.False(Directory.Exists(Path.Combine(_storage, "participants", id)));
        var manifest = File.ReadAllText(Path.Combine(_storage, "manifest.csv"));
        Assert.DoesNotContain(id, manifest);
        Assert.Contains(other, manifest);
        Assert.Contains(id, File.ReadAllText(Path.Combine(_storage, "withdrawals.log")));
    }

    [Fact]
    public async Task Admin_endpoints_need_the_key()
    {
        var (id, _) = await Join();
        await Upload(id, "hello_uno.slow", Wav(1.5));

        Assert.Equal(HttpStatusCode.Unauthorized, (await _client.GetAsync("/api/admin/stats")).StatusCode);

        var request = new HttpRequestMessage(HttpMethod.Get, "/api/admin/stats");
        request.Headers.Add("X-Admin-Key", "admin-secret");
        var stats = await (await _client.SendAsync(request)).Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(1, stats.GetProperty("recordings").GetInt32());

        var export = new HttpRequestMessage(HttpMethod.Get, "/api/admin/export");
        export.Headers.Add("X-Admin-Key", "admin-secret");
        var zipResponse = await _client.SendAsync(export);
        Assert.Equal("application/zip", zipResponse.Content.Headers.ContentType?.MediaType);
        using var zip = new System.IO.Compression.ZipArchive(await zipResponse.Content.ReadAsStreamAsync());
        Assert.Contains(zip.Entries, e => e.FullName == $"participants/{id}/hello_uno.slow.wav");
        Assert.Contains(zip.Entries, e => e.FullName == "manifest.csv");
    }
}
