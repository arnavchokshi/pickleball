import { readFile } from "node:fs/promises";

import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { describe, expect, it } from "vitest";

import { configureGltfLoader } from "./App";

const compressedBurlingtonGlb = new URL(
  "../../../runs/replay_native_20260702T032204Z/burlington/body_mesh_animated_compressed.glb",
  import.meta.url,
);

describe("configureGltfLoader", () => {
  it("loads the real meshopt-compressed Burlington animated body GLB", async () => {
    const data = await readFile(compressedBurlingtonGlb);
    const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const gltf = await configureGltfLoader(new GLTFLoader()).parseAsync(arrayBuffer, "");

    expect(gltf.scene.children.length).toBeGreaterThan(0);
    expect(gltf.animations.length).toBeGreaterThan(0);
  });
});
