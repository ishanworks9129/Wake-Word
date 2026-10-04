using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace WakeWord.Recorder;

public sealed class RecorderOptions
{
    /// <summary>Folder that holds everything collected. Back it up; it is the only copy.</summary>
    public string StoragePath { get; set; } = "recordings";

    /// <summary>Shared with invited contributors; stops strangers uploading.</summary>
    public string InviteCode { get; set; } = string.Empty;

    /// <summary>Required for export and stats. Keep it secret.</summary>
    public string AdminKey { get; set; } = string.Empty;

    /// <summary>Must match the consent text the page shows (wwwroot/consent/{ConsentVersion}.html).</summary>
    public string ConsentVersion { get; set; } = "v1-draft";

    public double MinSeconds { get; set; } = 0.5;

    public double MaxSeconds { get; set; } = 6.0;

    public int MinPeak { get; set; } = 300;
}

public sealed record ParticipantProfile(string? AgeBand, string? Accent, string? Gender, string? Device);

public sealed record Participant(
    string Id,
    string ConsentVersion,
    DateTimeOffset ConsentedAt,
    string WithdrawalCodeHash,
    string Split,
    ParticipantProfile Profile);

/// <summary>
/// File-system store. Layout under StoragePath:
///   participants/{id}/consent.json, participants/{id}/{promptId}.wav, manifest.csv (DATASET_MANIFEST schema),
///   withdrawals.log (ids and times only).
/// </summary>
public sealed class RecordingStore
{
    private static readonly string[] ManifestColumns =
        ["path", "kind", "split", "source", "url", "license", "license_url", "retrieved", "duration_seconds", "speaker_id", "release_id"];

    private readonly RecorderOptions _options;
    private readonly Lock _manifestLock = new();
    private readonly TimeProvider _time;

    public RecordingStore(RecorderOptions options, TimeProvider time)
    {
        _options = options;
        _time = time;
        Directory.CreateDirectory(ParticipantsRoot);
    }

    private string Root => Path.GetFullPath(_options.StoragePath);

    private string ParticipantsRoot => Path.Combine(Root, "participants");

    private string ManifestPath => Path.Combine(Root, "manifest.csv");

    public (Participant Participant, string WithdrawalCode) CreateParticipant(ParticipantProfile profile)
    {
        var id = Guid.NewGuid().ToString("N");
        var code = Convert.ToHexString(RandomNumberGenerator.GetBytes(5)); // 10 hex characters
        // Held-out speakers (plan 5.2): the split is fixed per person, never per clip.
        var bucket = BitConverter.ToUInt32(SHA256.HashData(Encoding.UTF8.GetBytes(id))) % 10;
        var split = bucket switch { 0 => "test", 1 => "dev", _ => "train" };
        var participant = new Participant(id, _options.ConsentVersion, _time.GetUtcNow(), Hash(code), split, profile);

        var dir = ParticipantDir(id);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "consent.json"), JsonSerializer.Serialize(participant, JsonOptions));
        return (participant, code);
    }

    public Participant? Find(string id)
    {
        if (!IsValidId(id))
        {
            return null;
        }

        var path = Path.Combine(ParticipantDir(id), "consent.json");
        return File.Exists(path) ? JsonSerializer.Deserialize<Participant>(File.ReadAllText(path), JsonOptions) : null;
    }

    public void SaveRecording(Participant participant, Prompt prompt, byte[] wav, WavInfo info)
    {
        var relative = $"participants/{participant.Id}/{prompt.Id}.wav";
        File.WriteAllBytes(Path.Combine(Root, relative), wav);

        lock (_manifestLock)
        {
            var rows = ReadManifest().Where(r => r[0] != relative).ToList(); // a re-record replaces the old take
            rows.Add([
                relative,
                prompt.Kind == PromptKind.Positive ? "positive" : "hard_negative",
                participant.Split,
                "Contributor recordings (WakeWord.Recorder)",
                "urn:wakeword-recorder",
                "Contributor-Release",
                $"consent/{participant.ConsentVersion}.html",
                _time.GetUtcNow().ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                info.Duration.TotalSeconds.ToString("0.###", CultureInfo.InvariantCulture),
                participant.Id,
                $"{participant.ConsentVersion}:{participant.Id}",
            ]);
            WriteManifest(rows);
        }
    }

    public IReadOnlyList<string> RecordedPrompts(string participantId) =>
        Directory.EnumerateFiles(ParticipantDir(participantId), "*.wav").Select(f => Path.GetFileNameWithoutExtension(f)).Order().ToList();

    /// <summary>Deletes everything a contributor gave, if the withdrawal code matches.</summary>
    public bool Withdraw(string participantId, string withdrawalCode)
    {
        var participant = Find(participantId);
        if (participant is null || !CryptographicOperations.FixedTimeEquals(
                Encoding.ASCII.GetBytes(participant.WithdrawalCodeHash), Encoding.ASCII.GetBytes(Hash(withdrawalCode.Trim().ToUpperInvariant()))))
        {
            return false;
        }

        lock (_manifestLock)
        {
            Directory.Delete(ParticipantDir(participantId), recursive: true);
            WriteManifest(ReadManifest().Where(r => r[9] != participantId).ToList());
            File.AppendAllText(Path.Combine(Root, "withdrawals.log"), $"{_time.GetUtcNow():O} {participantId}{Environment.NewLine}");
        }

        return true;
    }

    public object Stats()
    {
        lock (_manifestLock)
        {
            var rows = ReadManifest();
            return new
            {
                participants = Directory.EnumerateDirectories(ParticipantsRoot).Count(),
                recordings = rows.Count,
                byKind = rows.GroupBy(r => r[1]).ToDictionary(g => g.Key, g => g.Count()),
                bySplit = rows.GroupBy(r => r[2]).ToDictionary(g => g.Key, g => g.Select(r => r[9]).Distinct().Count()),
                minutes = Math.Round(rows.Sum(r => double.Parse(r[8], CultureInfo.InvariantCulture)) / 60, 1),
            };
        }
    }

    /// <summary>Zip of the manifest, consent records and audio.</summary>
    public async Task ExportAsync(Stream output, CancellationToken cancellationToken)
    {
        using var zip = new ZipArchive(output, ZipArchiveMode.Create, leaveOpen: true);
        string[] files;
        lock (_manifestLock)
        {
            files = Directory.EnumerateFiles(Root, "*", SearchOption.AllDirectories).ToArray();
        }

        foreach (var file in files)
        {
            var entry = zip.CreateEntry(Path.GetRelativePath(Root, file).Replace('\\', '/'), CompressionLevel.Fastest);
            await using var target = await entry.OpenAsync(cancellationToken);
            await using var source = File.OpenRead(file);
            await source.CopyToAsync(target, cancellationToken);
        }
    }

    private string ParticipantDir(string id) => Path.Combine(ParticipantsRoot, id);

    private static bool IsValidId(string id) => id.Length == 32 && id.All(Uri.IsHexDigit);

    private static string Hash(string code) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(code)));

    private List<string[]> ReadManifest() =>
        File.Exists(ManifestPath)
            ? File.ReadAllLines(ManifestPath).Skip(1).Where(l => l.Length > 0).Select(l => l.Split(',')).ToList()
            : [];

    private void WriteManifest(List<string[]> rows)
    {
        var tmp = ManifestPath + ".tmp";
        File.WriteAllLines(tmp, rows.Select(r => string.Join(',', r)).Prepend(string.Join(',', ManifestColumns)));
        File.Move(tmp, ManifestPath, overwrite: true);
    }

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web) { WriteIndented = true };
}
