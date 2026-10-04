namespace WakeWord.Recorder;

public enum PromptKind
{
    /// <summary>The wake phrase itself.</summary>
    Positive,

    /// <summary>A near-miss that must not fire; real voices saying these are the most useful negatives.</summary>
    HardNegative,
}

public sealed record Prompt(string Id, string Text, string Instruction, PromptKind Kind, string? Keyword);

/// <summary>About 26 clips, roughly five minutes per person. Varied delivery gives the model varied examples.</summary>
public static class Prompts
{
    private static readonly (string Id, string Instruction)[] Deliveries =
    [
        ("normal", "Say it in your normal voice."),
        ("quick", "Say it quickly, like you're in a hurry."),
        ("slow", "Say it slowly and clearly."),
        ("quiet", "Say it quietly, as if someone nearby is sleeping."),
        ("loud", "Say it a little louder, as if from across the room."),
        ("far", "Hold the phone at arm's length and say it."),
        ("question", "Say it like you're asking a question."),
        ("tired", "Say it casually, like you're tired."),
        ("normal2", "Say it in your normal voice again."),
        ("walking", "Say it while turning your head away slightly."),
    ];

    private static readonly (string Id, string Text)[] NearMisses =
    [
        ("hey_you_know", "Hey, you know what I mean?"),
        ("hello_you_know", "Hello, you know him?"),
        ("you_know", "You know, I think so."),
        ("play_uno", "Let's play Uno tonight."),
        ("hey_juno", "Hey Juno, come here."),
        ("hello_there", "Hello there, how are you?"),
    ];

    public static IReadOnlyList<Prompt> All { get; } =
    [
        .. Deliveries.Select(d => new Prompt($"hey_uno.{d.Id}", "Hey UNO", d.Instruction, PromptKind.Positive, "hey_uno")),
        .. Deliveries.Select(d => new Prompt($"hello_uno.{d.Id}", "Hello UNO", d.Instruction, PromptKind.Positive, "hello_uno")),
        .. NearMisses.Select(n => new Prompt($"near.{n.Id}", n.Text, "Say this naturally.", PromptKind.HardNegative, null)),
    ];

    public static Prompt? Find(string id) => All.FirstOrDefault(p => p.Id == id);
}
