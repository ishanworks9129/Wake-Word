using System.Text.Json;
using WakeWord.Core.Detection;

namespace WakeWord.Core.Tests;

/// <summary>
/// Runs testdata/golden/detector_cases.json. training/eval/detector.py runs the same file,
/// so the shipped detector and the one used to measure Section 3 metrics cannot drift apart.
/// </summary>
public class DetectorGoldenTests
{
    private static readonly JsonElement Golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "golden", "detector_cases.json"))).RootElement;

    public static TheoryData<string> CaseNames()
    {
        var data = new TheoryData<string>();
        foreach (var c in Golden.GetProperty("cases").EnumerateArray())
        {
            data.Add(c.GetProperty("name").GetString()!);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Fires_on_exactly_the_expected_frames(string name)
    {
        var sampleRate = Golden.GetProperty("sample_rate").GetInt32();
        var frameSamples = Golden.GetProperty("frame_samples").GetInt32();
        var c = Golden.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("name").GetString() == name);
        var o = c.GetProperty("options");
        var adaptive = o.GetProperty("adaptive");

        var detector = new WakeWordDetector(
            new WakeWordDetectorOptions
            {
                BaseThreshold = o.GetProperty("base_threshold").GetDouble(),
                ConsecutiveFrames = o.GetProperty("consecutive_frames").GetInt32(),
                Refractory = TimeSpan.FromSeconds(o.GetProperty("refractory_seconds").GetDouble()),
                Adaptive = new AdaptiveThresholdOptions
                {
                    Points = adaptive.GetProperty("points").EnumerateArray()
                        .Select(p => new ThresholdPoint(p[0].GetDouble(), p[1].GetDouble())).ToList(),
                    MinThreshold = adaptive.GetProperty("min_threshold").GetDouble(),
                    MaxThreshold = adaptive.GetProperty("max_threshold").GetDouble(),
                },
            },
            sampleRate);

        var scores = c.GetProperty("scores").EnumerateArray().Select(s => s.GetDouble()).ToArray();
        var noise = c.GetProperty("noise_floor_dbfs");
        var fires = new List<int>();
        for (var i = 0; i < scores.Length; i++)
        {
            var floor = noise.ValueKind == JsonValueKind.Array ? noise[i].GetDouble() : noise.GetDouble();
            if (detector.Process(scores[i], floor, (long)(i + 1) * frameSamples) is not null)
            {
                fires.Add(i);
            }
        }

        var expected = c.GetProperty("expected_fire_frames").EnumerateArray().Select(e => e.GetInt32());
        Assert.Equal(expected, fires);
    }
}
