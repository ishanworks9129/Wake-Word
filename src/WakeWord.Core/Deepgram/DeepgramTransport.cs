using System.Net.WebSockets;
using System.Text;

namespace WakeWord.Core.Deepgram;

/// <summary>The socket under a <see cref="DeepgramSession"/>; abstracted so sessions can be tested without a network.</summary>
public interface IDeepgramTransport : IAsyncDisposable
{
    Task ConnectAsync(Uri uri, string accessToken, CancellationToken cancellationToken);

    Task SendAudioAsync(ReadOnlyMemory<byte> pcm, CancellationToken cancellationToken);

    Task SendTextAsync(string message, CancellationToken cancellationToken);

    /// <summary>Next text message, or null once the server has closed the socket.</summary>
    Task<string?> ReceiveTextAsync(CancellationToken cancellationToken);
}

public sealed class ClientWebSocketTransport : IDeepgramTransport
{
    private readonly ClientWebSocket _socket = new();
    private readonly SemaphoreSlim _sendLock = new(1, 1);

    public Task ConnectAsync(Uri uri, string accessToken, CancellationToken cancellationToken)
    {
        _socket.Options.SetRequestHeader("Authorization", $"Bearer {accessToken}");
        return _socket.ConnectAsync(uri, cancellationToken);
    }

    public Task SendAudioAsync(ReadOnlyMemory<byte> pcm, CancellationToken cancellationToken) =>
        SendAsync(pcm, WebSocketMessageType.Binary, cancellationToken);

    public Task SendTextAsync(string message, CancellationToken cancellationToken) =>
        SendAsync(Encoding.UTF8.GetBytes(message), WebSocketMessageType.Text, cancellationToken);

    public async Task<string?> ReceiveTextAsync(CancellationToken cancellationToken)
    {
        var buffer = new byte[8192];
        using var message = new MemoryStream();
        while (true)
        {
            var result = await _socket.ReceiveAsync(buffer, cancellationToken).ConfigureAwait(false);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                CloseDescription = result.CloseStatusDescription;
                return null;
            }

            message.Write(buffer, 0, result.Count);
            if (result.EndOfMessage)
            {
                return result.MessageType == WebSocketMessageType.Text
                    ? Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length)
                    : string.Empty;
            }
        }
    }

    /// <summary>Deepgram puts error details in the close frame's description.</summary>
    public string? CloseDescription { get; private set; }

    public async ValueTask DisposeAsync()
    {
        if (_socket.State is WebSocketState.Open or WebSocketState.CloseReceived)
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(1));
            try
            {
                await _socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, null, timeout.Token).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is WebSocketException or OperationCanceledException)
            {
                // Best effort; the socket is torn down below either way.
            }
        }

        _socket.Dispose();
        _sendLock.Dispose();
    }

    private async Task SendAsync(ReadOnlyMemory<byte> data, WebSocketMessageType type, CancellationToken cancellationToken)
    {
        await _sendLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await _socket.SendAsync(data, type, endOfMessage: true, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _sendLock.Release();
        }
    }
}
