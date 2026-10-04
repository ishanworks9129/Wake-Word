/** A short-lived Deepgram access token. Only needs to be valid when the socket opens. */
export interface DeepgramToken {
  accessToken: string;
  /** Epoch milliseconds. */
  expiresAt: number;
}

export type TokenSource = () => Promise<DeepgramToken>;

export interface TokenProvider {
  getToken(): Promise<DeepgramToken>;
  /** Starts a background refresh if the cached token is stale. Called on speech onset. */
  prefetch(): void;
}

/**
 * Gets tokens from the token broker (src/WakeWord.TokenBroker). `headers` supplies the app's own user
 * authentication, e.g. `() => ({ Authorization: \`Bearer ${idToken}\` })`.
 */
export function brokerTokenSource(
  url: string,
  headers: () => Record<string, string> | Promise<Record<string, string>> = () => ({}),
  fetchImpl: typeof fetch = fetch,
  now: () => number = Date.now,
): TokenSource {
  return async () => {
    const requestedAt = now();
    const res = await fetchImpl(url, { method: "POST", headers: await headers() });
    if (!res.ok) throw new Error(`Token broker returned HTTP ${res.status}`);
    const body = (await res.json()) as { accessToken: string; expiresIn: number };
    return { accessToken: body.accessToken, expiresAt: requestedAt + body.expiresIn * 1000 };
  };
}

/**
 * Keeps a token ready so fetching is never on the wake-word path (plan 8.2). Reuses a token until 80% of
 * its lifetime has passed; concurrent callers share one fetch.
 */
export class CachingTokenProvider implements TokenProvider {
  private token: DeepgramToken | null = null;
  private refreshAt = 0;
  private inflight: Promise<DeepgramToken> | null = null;

  constructor(private readonly source: TokenSource, private readonly now: () => number = Date.now, private readonly refreshAtFraction = 0.8) {}

  getToken(): Promise<DeepgramToken> {
    if (this.token && this.now() < this.refreshAt) return Promise.resolve(this.token);
    return this.fetch();
  }

  prefetch(): void {
    if (this.token && this.now() < this.refreshAt) return;
    this.fetch().catch(() => { /* surfaces on the next getToken */ });
  }

  private fetch(): Promise<DeepgramToken> {
    if (this.inflight) return this.inflight;
    const issuedAt = this.now();
    this.inflight = this.source()
      .then((token) => {
        this.token = token;
        this.refreshAt = issuedAt + (token.expiresAt - issuedAt) * this.refreshAtFraction;
        return token;
      })
      .finally(() => { this.inflight = null; });
    return this.inflight;
  }
}
