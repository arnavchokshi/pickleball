import React, { useState } from "react";

import { login, register, type AuthApiOptions } from "./authApi";

export type SignInScreenProps = AuthApiOptions & {
  onAuthed: () => void;
  /** Testability hook only (no react-router / interaction harness in this
   * repo's test setup): lets a static-markup test render the register-mode
   * form without simulating a click on the mode-toggle button. */
  initialMode?: Mode;
  devManifestHint?: boolean;
};

export type Mode = "login" | "register";

export type SignInCredentials = {
  email: string;
  password: string;
  inviteCode: string;
};

export function signInErrorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

/**
 * Pure orchestration, split out of the component's submit handler so it is
 * directly testable against a fake fetch (no DOM/event simulation needed):
 * login mode calls `/api/auth/login`; register mode calls
 * `/api/auth/register` (which returns only `{id, email}`, no session) and
 * then immediately `login()`s with the same credentials so a fresh
 * registration lands the user in the app.
 */
export async function performSignIn(mode: Mode, credentials: SignInCredentials, options: AuthApiOptions = {}): Promise<void> {
  if (mode === "register") {
    await register({ email: credentials.email, password: credentials.password, inviteCode: credentials.inviteCode }, options);
  }
  await login({ email: credentials.email, password: credentials.password }, options);
}

export function SignInScreen({ onAuthed, fetchImpl, baseUrl, initialMode = "login", devManifestHint = false }: SignInScreenProps) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await performSignIn(mode, { email, password, inviteCode }, { fetchImpl, baseUrl });
      onAuthed();
    } catch (nextError) {
      setError(signInErrorText(nextError));
    } finally {
      setBusy(false);
    }
  }

  function toggleMode() {
    setMode((current) => (current === "login" ? "register" : "login"));
    setError(null);
  }

  const submitLabel = mode === "login" ? "Sign in" : "Create account";

  return (
    <section className="signin-screen" aria-label="Sign in">
      <form className="signin-form" onSubmit={handleSubmit}>
        <h1>{mode === "login" ? "Sign in" : "Create account"}</h1>
        {devManifestHint ? (
          <aside className="dev-manifest-hint" role="note">
            manifest param detected; to open without auth run the dev server with <code>VITE_REPLAY_VERIFY_DEV_BYPASS=1</code>
          </aside>
        ) : null}
        <label htmlFor="signin-email">
          <span>Email</span>
          <input
            id="signin-email"
            name="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </label>
        <label htmlFor="signin-password">
          <span>Password</span>
          <input
            id="signin-password"
            name="password"
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            minLength={8}
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {mode === "register" ? (
          <label htmlFor="signin-invite">
            <span>Invite code</span>
            <input
              id="signin-invite"
              name="inviteCode"
              type="text"
              required
              value={inviteCode}
              onChange={(event) => setInviteCode(event.target.value)}
            />
          </label>
        ) : null}
        <button type="submit" className="signin-submit" disabled={busy}>
          {busy ? "Please wait..." : submitLabel}
        </button>
        <button type="button" className="signin-toggle-mode" onClick={toggleMode} disabled={busy}>
          {mode === "login" ? "Need an account? Register" : "Have an account? Sign in"}
        </button>
        {error ? (
          <p role="alert" className="signin-error">
            {error}
          </p>
        ) : null}
      </form>
    </section>
  );
}

export default SignInScreen;
