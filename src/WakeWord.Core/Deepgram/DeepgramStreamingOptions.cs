using System.Globalization;
using System.Text;

namespace WakeWord.Core.Deepgram;

/// <summary>Query parameters for Deepgram's live endpoint (plan 8.3).</summary>
public sealed record DeepgramStreamingOptions
{
    public Uri Endpoint { get; init; } = new("wss://api.deepgram.com/v1/listen");

    public string Model { get; init; } = "nova-3";

    public string Language { get; init; } = "en";

    public int EndpointingMs { get; init; } = 300;

    /// <summary>Requires interim results.</summary>
    public int UtteranceEndMs { get; init; } = 1000;

    public bool SmartFormat { get; init; } = true;

    /// <summary>Opts out of Deepgram's Model Improvement Program (plan 4.2).</summary>
    public bool MipOptOut { get; init; } = true;

    /// <summary>Words to bias recognition toward, typically the brand word in the wake phrase.</summary>
    public IReadOnlyList<string> Keyterms { get; init; } = [];

    public Uri BuildUri(int sampleRate)
    {
        var query = new StringBuilder()
            .Append("model=").Append(Uri.EscapeDataString(Model))
            .Append("&language=").Append(Uri.EscapeDataString(Language))
            .Append("&encoding=linear16")
            .Append("&sample_rate=").Append(sampleRate.ToString(CultureInfo.InvariantCulture))
            .Append("&channels=1")
            .Append("&interim_results=true")
            .Append("&endpointing=").Append(EndpointingMs.ToString(CultureInfo.InvariantCulture))
            .Append("&utterance_end_ms=").Append(UtteranceEndMs.ToString(CultureInfo.InvariantCulture))
            .Append("&smart_format=").Append(SmartFormat ? "true" : "false")
            .Append("&mip_opt_out=").Append(MipOptOut ? "true" : "false");
        foreach (var term in Keyterms)
        {
            query.Append("&keyterm=").Append(Uri.EscapeDataString(term));
        }

        return new UriBuilder(Endpoint) { Query = query.ToString() }.Uri;
    }
}

/// <summary>Cost and safety limits for one session (plan 8.5).</summary>
public sealed record DeepgramSessionLimits
{
    /// <summary>Close if no non-wake-word speech is transcribed within this time of streaming starting.</summary>
    public TimeSpan NoTranscriptCutoff { get; init; } = TimeSpan.FromSeconds(4);

    public TimeSpan HardTimeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>How long to wait for final results after sending CloseStream.</summary>
    public TimeSpan CloseGrace { get; init; } = TimeSpan.FromSeconds(2);

    /// <summary>Largest audio message, in samples (100 ms at 16 kHz).</summary>
    public int MaxChunkSamples { get; init; } = 1600;
}
