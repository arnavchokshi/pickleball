import { beforeEach, describe, expect, it } from "vitest";

import { authedFetch, getAccessToken, login, logout, refresh, register, setAccessToken } from "./authApi";

beforeEach(() => {
  setAccessToken(null);
});

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("register", () => {
  it("posts email/password/invite_code and does not store a token (register issues no session)", async () => {
    let postedUrl = "";
    let postedBody: unknown = null;
    const fetchImpl: typeof fetch = async (url, init) => {
      postedUrl = String(url);
      postedBody = JSON.parse(String(init?.body));
      return jsonResponse({ id: "user_1", email: "a@example.com" }, 201);
    };

    const result = await register(
      { email: "a@example.com", password: "hunter22", inviteCode: "dink-abc" },
      { baseUrl: "https://pb.example.com", fetchImpl },
    );

    expect(postedUrl).toBe("https://pb.example.com/api/auth/register");
    expect(postedBody).toEqual({ email: "a@example.com", password: "hunter22", invite_code: "dink-abc" });
    expect(result).toEqual({ id: "user_1", email: "a@example.com" });
    expect(getAccessToken()).toBeNull();
  });

  it("surfaces a bad-invite 403 as a readable error", async () => {
    const fetchImpl: typeof fetch = async () => jsonResponse({ detail: "invalid invite code" }, 403);

    await expect(
      register({ email: "a@example.com", password: "hunter22", inviteCode: "wrong" }, { fetchImpl }),
    ).rejects.toThrow("invalid invite code");
  });
});

describe("login", () => {
  it("stores the access token and sends credentials so the refresh cookie is set", async () => {
    let sawCredentials: RequestCredentials | undefined;
    const fetchImpl: typeof fetch = async (_url, init) => {
      sawCredentials = init?.credentials;
      return jsonResponse({ access_token: "tok_123", token_type: "bearer", expires_in: 900 });
    };

    const session = await login({ email: "a@example.com", password: "hunter22" }, { fetchImpl });

    expect(session.access_token).toBe("tok_123");
    expect(getAccessToken()).toBe("tok_123");
    expect(sawCredentials).toBe("include");
  });

  it("surfaces a wrong-password 401 as a readable error", async () => {
    const fetchImpl: typeof fetch = async () => jsonResponse({ detail: "invalid email or password" }, 401);

    await expect(login({ email: "a@example.com", password: "wrong" }, { fetchImpl })).rejects.toThrow(
      "invalid email or password",
    );
    expect(getAccessToken()).toBeNull();
  });
});

describe("refresh", () => {
  it("stores the rotated access token on success", async () => {
    const fetchImpl: typeof fetch = async () => jsonResponse({ access_token: "tok_new", token_type: "bearer", expires_in: 900 });

    const session = await refresh({ fetchImpl });

    expect(session?.access_token).toBe("tok_new");
    expect(getAccessToken()).toBe("tok_new");
  });

  it("clears the token and returns null on a failed refresh instead of throwing", async () => {
    setAccessToken("stale");
    const fetchImpl: typeof fetch = async () => new Response(null, { status: 401 });

    const session = await refresh({ fetchImpl });

    expect(session).toBeNull();
    expect(getAccessToken()).toBeNull();
  });
});

describe("logout", () => {
  it("clears the in-memory access token even though the endpoint returns 204 (no body)", async () => {
    setAccessToken("tok_123");
    let called = false;
    const fetchImpl: typeof fetch = async () => {
      called = true;
      return new Response(null, { status: 204 });
    };

    await logout({ fetchImpl });

    expect(called).toBe(true);
    expect(getAccessToken()).toBeNull();
  });

  it("still clears the token if the logout request itself fails", async () => {
    setAccessToken("tok_123");
    const fetchImpl: typeof fetch = async () => {
      throw new Error("network down");
    };

    await expect(logout({ fetchImpl })).rejects.toThrow("network down");
    expect(getAccessToken()).toBeNull();
  });
});

describe("authedFetch", () => {
  it("attaches the bearer header from the in-memory token", async () => {
    setAccessToken("tok_abc");
    let sawAuth: string | null = null;
    const fetchImpl: typeof fetch = async (_url, init) => {
      sawAuth = new Headers(init?.headers).get("Authorization");
      return jsonResponse({ ok: true });
    };

    await authedFetch("/api/clips", {}, { fetchImpl });

    expect(sawAuth).toBe("Bearer tok_abc");
  });

  it("on a 401 tries exactly one refresh then retries once with the new token", async () => {
    setAccessToken("tok_expired");
    const calls: Array<{ url: string; auth: string | null }> = [];
    const fetchImpl: typeof fetch = async (url, init) => {
      const entry = { url: String(url), auth: new Headers(init?.headers).get("Authorization") };
      calls.push(entry);
      if (entry.url.endsWith("/api/auth/refresh")) {
        return jsonResponse({ access_token: "tok_fresh", token_type: "bearer", expires_in: 900 });
      }
      if (entry.auth === "Bearer tok_expired") {
        return new Response(JSON.stringify({ detail: "invalid or expired token" }), {
          status: 401,
          headers: { "content-type": "application/json" },
        });
      }
      return jsonResponse({ id: "job_1", status: "queued" });
    };

    const response = await authedFetch("/api/jobs/job_1", {}, { fetchImpl });
    const payload = await response.json();

    expect(payload).toEqual({ id: "job_1", status: "queued" });
    expect(calls.map((c) => c.url)).toEqual(["/api/jobs/job_1", "/api/auth/refresh", "/api/jobs/job_1"]);
    expect(calls[2].auth).toBe("Bearer tok_fresh");
    expect(getAccessToken()).toBe("tok_fresh");
  });

  it("does not retry a second time if refresh itself fails; returns the original 401", async () => {
    setAccessToken("tok_expired");
    let callCount = 0;
    const fetchImpl: typeof fetch = async (url) => {
      callCount += 1;
      if (String(url).endsWith("/api/auth/refresh")) {
        return new Response(null, { status: 401 });
      }
      return new Response(JSON.stringify({ detail: "invalid or expired token" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      });
    };

    const response = await authedFetch("/api/jobs/job_1", {}, { fetchImpl });

    expect(response.status).toBe(401);
    expect(callCount).toBe(2); // original attempt + refresh; no second retry
    expect(getAccessToken()).toBeNull();
  });

  it("does not attach a bearer header when there is no access token", async () => {
    let sawAuth: string | null = "unset";
    const fetchImpl: typeof fetch = async (_url, init) => {
      sawAuth = new Headers(init?.headers).get("Authorization");
      return jsonResponse({ clips: [] });
    };

    await authedFetch("/api/clips", {}, { fetchImpl });

    expect(sawAuth).toBeNull();
  });
});
