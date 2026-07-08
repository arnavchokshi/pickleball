import { apiUrl, parseJsonResponse } from "./uploadApi";

/**
 * Auth client (INFRA-3) for the INFRA-1 accounts API: `server/routes/auth.py`.
 *
 * The access token is kept in an in-memory module variable ONLY (never
 * localStorage/sessionStorage) so it disappears on full page reload; the
 * refresh token lives in an httpOnly SameSite=Lax Secure cookie the browser
 * manages for us (`credentials: "include"` so it flows on same-origin
 * requests in both dev, via the Vite proxy, and prod, via StaticFiles).
 */

export type AuthApiOptions = {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
};

export type AuthSession = {
  access_token: string;
  token_type: string;
  expires_in: number;
};

export type RegisteredUser = {
  id: string;
  email: string;
};

export type RegisterInput = {
  email: string;
  password: string;
  inviteCode: string;
};

export type LoginInput = {
  email: string;
  password: string;
  deviceLabel?: string;
};

let accessToken: string | null = null;

export function getAccessToken(): string | null {
  return accessToken;
}

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export async function register(input: RegisterInput, options: AuthApiOptions = {}): Promise<RegisteredUser> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/auth/register", options.baseUrl), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      email: input.email,
      password: input.password,
      invite_code: input.inviteCode,
    }),
  });
  return parseJsonResponse<RegisteredUser>(response);
}

export async function login(input: LoginInput, options: AuthApiOptions = {}): Promise<AuthSession> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/auth/login", options.baseUrl), {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      email: input.email,
      password: input.password,
      ...(input.deviceLabel ? { device_label: input.deviceLabel } : {}),
    }),
  });
  const session = await parseJsonResponse<AuthSession>(response);
  setAccessToken(session.access_token);
  return session;
}

/**
 * Presents the httpOnly refresh cookie to `/api/auth/refresh` and stores the
 * newly-rotated access token. Returns `null` (and clears any stale access
 * token) instead of throwing on a non-OK response, since callers use this in
 * a "try silently, fall back to sign-in" position (`authedFetch`'s one retry,
 * and app-boot session restore).
 */
export async function refresh(options: AuthApiOptions = {}): Promise<AuthSession | null> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(apiUrl("/api/auth/refresh", options.baseUrl), {
    method: "POST",
    credentials: "include",
  });
  if (!response.ok) {
    setAccessToken(null);
    return null;
  }
  const session = await parseJsonResponse<AuthSession>(response);
  setAccessToken(session.access_token);
  return session;
}

export async function logout(options: AuthApiOptions = {}): Promise<void> {
  const fetchImpl = options.fetchImpl ?? fetch;
  try {
    await fetchImpl(apiUrl("/api/auth/logout", options.baseUrl), {
      method: "POST",
      credentials: "include",
    });
  } finally {
    setAccessToken(null);
  }
}

/**
 * Fetch wrapper for authenticated API calls: attaches
 * `Authorization: Bearer <access token>` and, on a 401, tries exactly one
 * `refresh()` and retries the request exactly once with the refreshed token.
 * If refresh itself fails, the original 401 response is returned so callers
 * see a normal failed-auth response (and can route back to sign-in).
 */
export async function authedFetch(path: string, init: RequestInit = {}, options: AuthApiOptions = {}): Promise<Response> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const url = apiUrl(path, options.baseUrl);

  const attempt = async (): Promise<Response> => {
    const headers = new Headers(init.headers);
    if (accessToken) {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }
    return fetchImpl(url, {
      ...init,
      headers,
      credentials: init.credentials ?? "include",
    });
  };

  const firstResponse = await attempt();
  if (firstResponse.status !== 401) {
    return firstResponse;
  }
  const session = await refresh(options);
  if (!session) {
    return firstResponse;
  }
  return attempt();
}
