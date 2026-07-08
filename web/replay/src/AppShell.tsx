import React, { useCallback, useEffect, useState } from "react";

import App, { manifestUrlFromSearch } from "./App";
import { getAccessToken, logout as apiLogout, setAccessToken, type AuthApiOptions } from "./authApi";
import { LibraryScreen } from "./LibraryScreen";
import { SignInScreen } from "./SignInScreen";

export type Screen = "signin" | "library" | "viewer";

/**
 * Pure screen resolver (no react-router): "signin" if there is no in-memory
 * access token; "viewer" if authed and the URL carries `?manifest=` (the
 * convention `App.tsx` already reads via `manifestUrlFromSearch`) or a
 * `?clip=` deep link; otherwise "library". Kept as a standalone export so
 * routing logic is testable without mounting the untouched `<App/>`, which
 * reads `window.location.search` unconditionally during its first render and
 * therefore cannot be exercised under this repo's jsdom-less vitest setup
 * (confirmed: the existing App.test.tsx never renders `<App/>` either, only
 * its exported pure helpers).
 */
export function resolveScreen(hasAccessToken: boolean, search: string): Screen {
  if (!hasAccessToken) return "signin";
  if (manifestUrlFromSearch(search)) return "viewer";
  if (new URLSearchParams(search).get("clip") !== null) return "viewer";
  return "library";
}

export type AppShellProps = AuthApiOptions;

export function AppShell({ fetchImpl, baseUrl }: AppShellProps) {
  const [hasToken, setHasToken] = useState<boolean>(() => Boolean(getAccessToken()));
  const [search, setSearch] = useState<string>(() => (typeof window === "undefined" ? "" : window.location.search));

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onPopState = () => setSearch(window.location.search);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const handleAuthed = useCallback(() => {
    setHasToken(Boolean(getAccessToken()));
  }, []);

  const handleLogout = useCallback(() => {
    void apiLogout({ fetchImpl, baseUrl }).finally(() => {
      setAccessToken(null);
      setHasToken(false);
    });
  }, [fetchImpl, baseUrl]);

  const handleOpenViewer = useCallback((manifestUrl: string) => {
    if (typeof window !== "undefined") {
      const nextSearch = `?manifest=${encodeURIComponent(manifestUrl)}`;
      window.history.pushState(null, "", `${window.location.pathname}${nextSearch}`);
      setSearch(nextSearch);
    }
  }, []);

  const screen = resolveScreen(hasToken, search);

  if (screen === "signin") {
    return <SignInScreen onAuthed={handleAuthed} fetchImpl={fetchImpl} baseUrl={baseUrl} />;
  }
  if (screen === "viewer") {
    // The 3D replay viewer -- untouched, additive-only per the lane fence.
    return <App />;
  }
  return <LibraryScreen onOpenViewer={handleOpenViewer} onLogout={handleLogout} fetchImpl={fetchImpl} baseUrl={baseUrl} />;
}

export default AppShell;
