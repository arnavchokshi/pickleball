import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { performSignIn, signInErrorText, SignInScreen } from "./SignInScreen";
import { getAccessToken, setAccessToken } from "./authApi";

describe("SignInScreen markup", () => {
  it("renders the login form by default (no invite code field)", () => {
    const markup = renderToStaticMarkup(<SignInScreen onAuthed={() => {}} />);

    expect(markup).toContain("Sign in");
    expect(markup).toContain("Email");
    expect(markup).toContain("Password");
    expect(markup).not.toContain("Invite code");
  });

  it("renders the register form with the invite code field via initialMode", () => {
    const markup = renderToStaticMarkup(<SignInScreen onAuthed={() => {}} initialMode="register" />);

    expect(markup).toContain("Create account");
    expect(markup).toContain("Invite code");
  });

  it("renders the explicit loopback manifest bypass hint only when requested", () => {
    const hinted = renderToStaticMarkup(<SignInScreen onAuthed={() => {}} devManifestHint />);
    const normal = renderToStaticMarkup(<SignInScreen onAuthed={() => {}} />);

    expect(hinted).toContain("manifest param detected");
    expect(hinted).toContain("VITE_REPLAY_VERIFY_DEV_BYPASS=1");
    expect(normal).not.toContain("manifest param detected");
  });
});

describe("signInErrorText", () => {
  it("unwraps an Error message and stringifies anything else", () => {
    expect(signInErrorText(new Error("invalid invite code"))).toBe("invalid invite code");
    expect(signInErrorText("plain string")).toBe("plain string");
  });
});

describe("performSignIn", () => {
  it("login mode calls only /api/auth/login", async () => {
    const calls: string[] = [];
    const fetchImpl: typeof fetch = async (url) => {
      calls.push(String(url));
      return new Response(JSON.stringify({ access_token: "tok_1", token_type: "bearer", expires_in: 900 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };

    await performSignIn("login", { email: "a@example.com", password: "hunter22", inviteCode: "" }, { fetchImpl });

    expect(calls).toEqual(["/api/auth/login"]);
    expect(getAccessToken()).toBe("tok_1");
    setAccessToken(null);
  });

  it("register mode calls register then login, in order", async () => {
    const calls: string[] = [];
    const fetchImpl: typeof fetch = async (url) => {
      calls.push(String(url));
      if (String(url).endsWith("/api/auth/register")) {
        return new Response(JSON.stringify({ id: "user_1", email: "a@example.com" }), {
          status: 201,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(JSON.stringify({ access_token: "tok_1", token_type: "bearer", expires_in: 900 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };

    await performSignIn(
      "register",
      { email: "a@example.com", password: "hunter22", inviteCode: "dink-abc" },
      { fetchImpl },
    );

    expect(calls).toEqual(["/api/auth/register", "/api/auth/login"]);
    setAccessToken(null);
  });

  it("surfaces a bad-invite (403) error and never calls login", async () => {
    const calls: string[] = [];
    const fetchImpl: typeof fetch = async (url) => {
      calls.push(String(url));
      return new Response(JSON.stringify({ detail: "invalid invite code" }), {
        status: 403,
        headers: { "content-type": "application/json" },
      });
    };

    await expect(
      performSignIn("register", { email: "a@example.com", password: "hunter22", inviteCode: "wrong" }, { fetchImpl }),
    ).rejects.toThrow("invalid invite code");
    expect(calls).toEqual(["/api/auth/register"]);
  });

  it("surfaces a wrong-password (401) error on login", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response(JSON.stringify({ detail: "invalid email or password" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      });

    await expect(
      performSignIn("login", { email: "a@example.com", password: "wrong", inviteCode: "" }, { fetchImpl }),
    ).rejects.toThrow("invalid email or password");
  });
});
