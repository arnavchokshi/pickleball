import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  bodyMeshIndexWindowForTime,
  fetchBodyMeshChunk,
  solidBodyMeshFramesForTime,
} from "../viewerData";
import { loadMeshCompareBundle, type MeshCompareFetch } from "./meshCompareData";

const comparisonPath = process.env.MESH_COMPARE_LIVE_COMPARISON?.trim() || null;
const userManifestPath = process.env.MESH_COMPARE_LIVE_USER_MANIFEST?.trim() || null;

function fsUrl(path: string): string {
  return `/@fs/${path}`;
}

function localFileFetch(): MeshCompareFetch {
  return async (url: string) => {
    const pathname = /^https?:/i.test(url) ? new URL(url).pathname : url.split(/[?#]/, 1)[0];
    if (!pathname.startsWith("/@fs/")) return new Response("unsupported test URL", { status: 404 });
    try {
      return new Response(await readFile(pathname.slice("/@fs/".length)), { status: 200 });
    } catch (error) {
      return new Response(error instanceof Error ? error.message : String(error), { status: 404 });
    }
  };
}

describe("live compact mesh comparison evidence", () => {
  const run = comparisonPath && userManifestPath ? it : it.skip;

  run("loads and decodes two real dense surfaces without a skeleton fallback", async () => {
    const fetchImpl = localFileFetch();
    const bundle = await loadMeshCompareBundle({
      route: {
        manifestUrl: fsUrl(userManifestPath!),
        comparisonUrl: fsUrl(comparisonPath!),
      },
      hostname: "127.0.0.1",
      fetchImpl,
    });

    expect(bundle.referencePresentation.badge).toBe("REFERENCE PLAYER");
    expect(bundle.referencePresentation.publicDisplayReady).toBe(false);
    expect(bundle.user.coverage).toMatchObject({ frameCount: 48, vertexCount: 18_439, faceCount: 36_874 });
    expect(bundle.reference.coverage).toMatchObject({ frameCount: 43, vertexCount: 18_439, faceCount: 36_874 });

    for (const [source, time] of [
      [bundle.user, 1.867],
      [bundle.reference, 6.6],
    ] as const) {
      const window = bodyMeshIndexWindowForTime(source.index, time);
      expect(window).not.toBeNull();
      const chunk = await fetchBodyMeshChunk(
        source.indexUrl,
        source.index,
        window!,
        source.faces,
        async (url) => {
          const bytes = source.verifiedChunkBytes.get(url);
          expect(bytes).toBeDefined();
          return new Response(bytes!.slice().buffer, { status: 200 });
        },
      );
      const active = solidBodyMeshFramesForTime(chunk, null, time, source.world).find(
        (candidate) => candidate.playerId === source.playerId,
      );
      expect(active?.frame.mesh_vertices_world).toHaveLength(18_439);
      expect(active?.frame.mesh_faces).toHaveLength(36_874);
      expect(active?.frame.mesh_interpolated).toBe(false);
    }
  }, 15_000);
});
