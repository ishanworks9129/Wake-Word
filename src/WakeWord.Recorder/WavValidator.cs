using System.Buffers.Binary;
using System.Text;

namespace WakeWord.Recorder;

public sealed record WavInfo(TimeSpan Duration, int PeakAbs);

/// <summary>Accepts only what training needs: RIFF/WAVE, PCM, mono, 16 kHz, 16-bit, a sane length, and actual sound.</summary>
public static class WavValidator
{
    public static (WavInfo? Info, string? Error) Validate(ReadOnlySpan<byte> wav, TimeSpan min, TimeSpan max, int minPeak)
    {
        if (wav.Length < 44 || !wav[..4].SequenceEqual("RIFF"u8) || !wav.Slice(8, 4).SequenceEqual("WAVE"u8))
        {
            return (null, "Not a WAV file.");
        }

        int? format = null, channels = null, rate = null, bits = null;
        ReadOnlySpan<byte> data = default;
        var pos = 12;
        while (pos + 8 <= wav.Length)
        {
            var id = Encoding.ASCII.GetString(wav.Slice(pos, 4));
            var size = BinaryPrimitives.ReadInt32LittleEndian(wav.Slice(pos + 4, 4));
            if (size < 0 || pos + 8 + size > wav.Length)
            {
                return (null, "Truncated WAV file.");
            }

            var body = wav.Slice(pos + 8, size);
            if (id == "fmt " && size >= 16)
            {
                format = BinaryPrimitives.ReadInt16LittleEndian(body);
                channels = BinaryPrimitives.ReadInt16LittleEndian(body[2..]);
                rate = BinaryPrimitives.ReadInt32LittleEndian(body[4..]);
                bits = BinaryPrimitives.ReadInt16LittleEndian(body[14..]);
            }
            else if (id == "data")
            {
                data = body;
            }

            pos += 8 + size + (size & 1);
        }

        if (format != 1 || channels != 1 || rate != 16000 || bits != 16)
        {
            return (null, $"Expected 16 kHz mono 16-bit PCM; got format {format}, {channels} channel(s), {rate} Hz, {bits}-bit.");
        }

        if (data.IsEmpty || data.Length % 2 != 0)
        {
            return (null, "No audio data.");
        }

        var duration = TimeSpan.FromSeconds(data.Length / 2 / 16000.0);
        if (duration < min || duration > max)
        {
            return (null, $"Recording must be {min.TotalSeconds:0.#}-{max.TotalSeconds:0.#} s; got {duration.TotalSeconds:0.##} s.");
        }

        var peak = 0;
        for (var i = 0; i < data.Length; i += 2)
        {
            peak = Math.Max(peak, Math.Abs((int)BinaryPrimitives.ReadInt16LittleEndian(data[i..])));
        }

        return peak < minPeak
            ? (null, "We couldn't hear anything. Check the microphone and try again.")
            : (new WavInfo(duration, peak), null);
    }
}
