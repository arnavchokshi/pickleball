import { beforeEach, describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { AppShell, resolveScreen } from "./AppShell";
import { setAccessToken } from "./authApi";

beforeEach(() => {
  setAccessToken(null);
});

describe("resolveScreen", () => {
  it("routes to signin when there is no access token, regardless of search params", () => {
    expect(resolveScreen(false, "")).toBe("signin");
    expect(resolveScreen(false, "?manifest=/tmp/x.json")).toBe("signin");
    expect(resolveScreen(false, "?clip=clip_1")).toBe("signin");
  });

  it("routes to library when authed with no manifest/clip param", () => {
    expect(resolveScreen(true, "")).toBe("library");
    expect(resolveScreen(true, "?coaching=1")).toBe("library");
  });

  it("routes to viewer when authed and ?manifest= is present", () => {
    expect(resolveScreen(true, "?manifest=/@fs/tmp/replay_viewer_manifest.json")).toBe("viewer");
  });

  it("routes to viewer when authed and ?clip= is present", () => {
    expect(resolveScreen(true, "?clip=clip_abc123")).toBe("viewer");
  });
});

describe("AppShell markup", () => {
  // Only the signin/library branches are rendered here: the viewer branch
  // mounts the untouched `<App/>`, which reads `window.location.search`
  // unconditionally at the top of its render (no prop injection, per the
  // "do not modify App.tsx internals" fence) and this repo's vitest setup
  // has no jsdom/window global (confirmed: App.test.tsx never renders
  // `<App/>` either). The viewer branch's *routing decision* is covered
  // above via resolveScreen.

  it("renders SignInScreen when there is no access token", () => {
    const markup = renderToStaticMarkup(<AppShell />);

    expect(markup).toContain("Sign in");
    expect(markup).not.toContain("Library");
  });

  it("renders LibraryScreen when an access token is present and there is no viewer param", () => {
    setAccessToken("tok_abc");

    const markup = renderToStaticMarkup(<AppShell />);

    expect(markup).toContain("Library");
    expect(markup).toContain("Upload a clip");
  });
});
