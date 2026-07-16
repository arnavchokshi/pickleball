import { describe, expect, it, vi } from "vitest";

import {
  fetchJsonWithManifestRelativeRecovery,
  fetchWithManifestRelativeRecovery,
  manifestRelativeRecoveryUrl,
} from "./assetRecovery";

const manifest = "/@fs//Users/reviewer/pulled/run/replay_viewer_manifest.json";

describe("manifest-relative asset recovery", () => {
  it("keeps the exact basename beside the local manifest, including the declared mesh-index subdir", () => {
    expect(manifestRelativeRecoveryUrl("/@fs//home/gpu/run/virtual_world.json", manifest)).toBe(
      "/@fs//Users/reviewer/pulled/run/virtual_world.json",
    );
    expect(manifestRelativeRecoveryUrl("/@fs//home/gpu/run/body_mesh_index/body_mesh_index.json", manifest)).toBe(
      "/@fs//Users/reviewer/pulled/run/body_mesh_index/body_mesh_index.json",
    );
    expect(manifestRelativeRecoveryUrl("https://example.com/virtual_world.json", manifest)).toBeNull();
    expect(manifestRelativeRecoveryUrl("http://127.0.0.1:5173/@fs//home/gpu/run/body_mesh_index/chunk_0.bin", manifest)).toBe(
      "/@fs//Users/reviewer/pulled/run/body_mesh_index/chunk_0.bin",
    );
  });

  it("retries once after the VM path fails and reports the loud recovery source", async () => {
    const fetchImpl = vi.fn(async (url: string) =>
      new Response(JSON.stringify({ url }), {
        status: url.includes("/home/gpu/") ? 404 : 200,
        headers: { "content-type": "application/json" },
      }),
    );
    const recovered = vi.fn();

    const response = await fetchWithManifestRelativeRecovery(
      "/@fs//home/gpu/run/virtual_world.json",
      manifest,
      fetchImpl,
      recovered,
    );

    expect(response.ok).toBe(true);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(recovered).toHaveBeenCalledWith(
      "/@fs//home/gpu/run/virtual_world.json",
      "/@fs//Users/reviewer/pulled/run/virtual_world.json",
    );
  });

  it.each([200, 403])("retries JSON assets when the VM URL returns an HTML document with status %s", async (status) => {
    const fetchImpl = vi.fn(async (url: string) => url.includes("/home/gpu/")
      ? new Response("<!doctype html><title>Vite fallback</title>", {
          status,
          headers: { "content-type": "text/html" },
        })
      : new Response(JSON.stringify({ recovered: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }));
    const recovered = vi.fn();

    await expect(fetchJsonWithManifestRelativeRecovery(
      "/@fs//home/gpu/run/virtual_world.json",
      manifest,
      fetchImpl,
      recovered,
    )).resolves.toEqual({ recovered: true });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(recovered).toHaveBeenCalledOnce();
  });

  it("rejects a 200 HTML manifest response with a plain-language URL error", async () => {
    const fetchImpl = vi.fn(async () => new Response("  <!doctype html><title>Not served</title>", {
      status: 200,
      headers: { "content-type": "text/html" },
    }));

    await expect(fetchJsonWithManifestRelativeRecovery(manifest, manifest, fetchImpl)).rejects.toThrow(
      `asset unreachable: ${manifest} — expected JSON but received an unavailable or non-JSON response`,
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("does not trust an HTML body mislabeled as JSON", async () => {
    const fetchImpl = vi.fn(async (url: string) => url.includes("/home/gpu/")
      ? new Response("<html>fallback</html>", {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      : new Response("{\"recovered\":true}", {
          status: 200,
          headers: { "content-type": "application/json" },
        }));

    await expect(fetchJsonWithManifestRelativeRecovery(
      "/@fs//home/gpu/run/virtual_world.json",
      manifest,
      fetchImpl,
    )).resolves.toEqual({ recovered: true });
  });
});
