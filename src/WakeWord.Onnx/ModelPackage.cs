using System.Text.Json;
using System.Text.Json.Serialization;
using WakeWord.Core;
using WakeWord.Core.Detection;

namespace WakeWord.Onnx;

/// <summary>
/// A trained model package (training/wakeword_train/export.py, format "wakeword-models/1"):
/// models.json plus the feature models and one classifier per keyword.
/// </summary>
public sealed class ModelPackage
{
    public const string Format = "wakeword-models/1";

    private readonly Dictionary<string, byte[]> _files;

    private ModelPackage(PackageManifest manifest, Dictionary<string, byte[]> files)
    {
        Manifest = manifest;
        _files = files;
    }

    public PackageManifest Manifest { get; }

    public IReadOnlyList<KeywordManifest> Keywords => Manifest.Keywords;

    public static ModelPackage LoadFromDirectory(string directory) =>
        LoadAsync(name => Task.FromResult<Stream>(System.IO.File.OpenRead(Path.Combine(directory, name)))).GetAwaiter().GetResult();

    /// <summary>Loads through any file opener, e.g. MAUI's FileSystem.OpenAppPackageFileAsync.</summary>
    public static async Task<ModelPackage> LoadAsync(Func<string, Task<Stream>> open, CancellationToken cancellationToken = default)
    {
        var manifestBytes = await ReadAllAsync(open, "models.json", cancellationToken).ConfigureAwait(false);
        var manifest = JsonSerializer.Deserialize<PackageManifest>(manifestBytes, JsonOptions)
            ?? throw new InvalidDataException("models.json is empty.");
        if (manifest.Format != Format)
        {
            throw new InvalidDataException($"Unsupported model package format '{manifest.Format}'; expected '{Format}'.");
        }

        if (manifest.SampleRate != WakeWordOptions.SampleRate || manifest.FrameSamples != OnnxWakeWordModel.FrameLength)
        {
            throw new InvalidDataException($"Package is {manifest.SampleRate} Hz / {manifest.FrameSamples}-sample frames; this runtime needs 16000 Hz / 1280.");
        }

        var names = new[] { manifest.Features.Melspectrogram, manifest.Features.Embedding }.Concat(manifest.Keywords.Select(k => k.Model));
        var files = new Dictionary<string, byte[]>();
        foreach (var name in names.Distinct())
        {
            files[name] = await ReadAllAsync(open, name, cancellationToken).ConfigureAwait(false);
        }

        return new ModelPackage(manifest, files);
    }

    public byte[] File(string name) => _files[name];

    public KeywordManifest Keyword(string id) =>
        Manifest.Keywords.FirstOrDefault(k => k.Id == id)
        ?? throw new ArgumentException($"Keyword '{id}' is not in this package; it has {string.Join(", ", Manifest.Keywords.Select(k => k.Id))}.", nameof(id));

    /// <summary>
    /// Detector settings for the chosen keywords at the given sensitivities (0..1, default from the package).
    /// The calibrated threshold becomes the base threshold; the package's bounds keep the clamp from cutting it off.
    /// </summary>
    public IReadOnlyList<KeywordOptions> KeywordOptions(IReadOnlyList<string>? keywords = null, IReadOnlyList<double>? sensitivities = null)
    {
        var chosen = (keywords ?? Manifest.Keywords.Select(k => k.Id).ToList()).Select(Keyword).ToList();
        if (sensitivities is not null && sensitivities.Count != chosen.Count)
        {
            throw new ArgumentException($"Got {sensitivities.Count} sensitivities for {chosen.Count} keywords.", nameof(sensitivities));
        }

        return chosen.Select((k, i) =>
        {
            var sensitivity = sensitivities?[i] ?? k.DefaultSensitivity;
            if (sensitivity is < 0 or > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(sensitivities), $"Sensitivity must be between 0 and 1; got {sensitivity}.");
            }

            var table = new SensitivityTable(k.SensitivityTable.Select(p => (p[0], p[1])));
            return new KeywordOptions
            {
                Id = k.Id,
                Phrase = k.Phrase,
                TranscriptVariants = k.TranscriptVariants,
                Detector = new WakeWordDetectorOptions
                {
                    BaseThreshold = table.ThresholdFor(sensitivity),
                    ConsecutiveFrames = k.Detector.ConsecutiveFrames,
                    Refractory = TimeSpan.FromSeconds(k.Detector.RefractorySeconds),
                    Adaptive = new AdaptiveThresholdOptions { MinThreshold = k.Detector.MinThreshold, MaxThreshold = k.Detector.MaxThreshold },
                },
            };
        }).ToList();
    }

    private static async Task<byte[]> ReadAllAsync(Func<string, Task<Stream>> open, string name, CancellationToken cancellationToken)
    {
        if (name.Contains('/') || name.Contains('\\') || name.Contains(".."))
        {
            throw new InvalidDataException($"Model file names must be plain file names; got '{name}'.");
        }

        await using var stream = await open(name).ConfigureAwait(false);
        using var buffer = new MemoryStream();
        await stream.CopyToAsync(buffer, cancellationToken).ConfigureAwait(false);
        return buffer.ToArray();
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };
}

public sealed record PackageManifest(
    string Format,
    string Run,
    int SampleRate,
    int FrameSamples,
    FeatureManifest Features,
    IReadOnlyList<KeywordManifest> Keywords,
    IReadOnlyList<string> Notes);

public sealed record FeatureManifest(
    string Melspectrogram,
    string Embedding,
    int MelContextSamples,
    int EmbeddingWindow,
    int EmbeddingStep,
    int ClassifierFrames);

public sealed record KeywordManifest(
    string Id,
    string Phrase,
    string Model,
    string Input,
    string Output,
    double DefaultSensitivity,
    IReadOnlyList<double[]> SensitivityTable,
    DetectorManifest Detector,
    IReadOnlyList<string> TranscriptVariants);

public sealed record DetectorManifest(
    int ConsecutiveFrames,
    double RefractorySeconds,
    double MinThreshold = 0.01,
    double MaxThreshold = 0.9999);
