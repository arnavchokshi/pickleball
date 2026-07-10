# NS-01.5 server-side handoff

Status: server-side acceptance PASS; `VERIFIED=0` remains binding.

## Evidence

- `wide_suite.xml`: 3,456 collected outcomes; 3,426 passed, 24 skipped, and the
  six unchanged localhost-bind tests failed with sandbox `PermissionError`.
- `tests/server/test_ns015_bundle_policy.py`: complete, four explicit missing-
  capability cases, missing advertised URL, simulated kill, enforced ordering,
  and late-stats fail-closed fixtures. All ten cases passed inside the wide run.
- The repository structure checks found 283 scaffold CLIs with zero missing
  direct-CLI reference tests, zero unknown Python sources, and zero unknown
  large files. The storage audit's overall `fail` is the existing allowlisted-
  source state, not a new unknown file.

## Best-stack delta

(c) NO stack delta. This lane changes product infrastructure only. No model,
checkpoint, selected runtime, confidence threshold, or promotion policy changed.

## Forbidden runner hunk for the integration owner to re-derive

The runner still builds the manifest before `match_stats`, has no production
coaching-facts stage, and does not emit `missing_capabilities`. The server now
packages existing facts in the correct order and fails closed when they are
absent, but the runner should become authoritative rather than relying on that
server compensation.

```diff
diff --git a/scripts/racketsport/process_video.py b/scripts/racketsport/process_video.py
@@ def _build_suffix_stage_fns(self) -> list[tuple[str, Callable[[], StageOutcome]]]:
             ("paddle_pose", self._stage_paddle_pose),
             ("world", self._stage_world),
             ("confidence_gate", self._stage_confidence_gate),
-            ("manifest", self._stage_manifest),
             ("match_stats", self._stage_match_stats),
+            ("coaching_facts", self._stage_coaching_facts),
+            ("manifest", self._stage_manifest),
         ]
@@ def _write_summary(self, *, wall_seconds: float) -> dict[str, Any]:
             "status": status,
+            "missing_capabilities": _minimum_bundle_missing_capabilities(
+                clip_dir=self.clip_dir,
+                stage_outcomes=self.stage_outcomes,
+            ),
             "wall_seconds": round(wall_seconds, 3),
```

Rationale: implement `_stage_coaching_facts` with the existing deterministic
builder, then implement `_minimum_bundle_missing_capabilities` from the exact
North Star §3.2 capability names and explicit stage notes. Do not infer it from
exit code, and do not call the bundle complete when the returned list is
non-empty. The manifest must advertise same-run stats/coaching only after those
stages finish. The integration owner must add runner tests because neither
helper exists in the current forbidden file.

## Forbidden native-app hunk for the integration owner to re-derive

The current Swift decoder rejects the server's honest `partial` status and has
no fields for missing capabilities or trust bands.

```diff
diff --git a/ios/Upload/Sources/PickleballUpload/RenderGatewayClient.swift b/ios/Upload/Sources/PickleballUpload/RenderGatewayClient.swift
@@ public enum RenderGatewayJobStatus: String, Codable, Equatable, Sendable {
     case running
     case complete
+    case partial
     case submitted
     case failed
 }
+
+public struct RenderGatewayMissingCapability: Codable, Equatable, Sendable {
+    public var capability: String
+    public var reason: String
+}
+
+public struct RenderGatewayTrustBand: Codable, Equatable, Sendable {
+    public var badge: String?
+    public var stage: String?
+    public var gateId: String?
+    public var gateStatus: String?
+    public var reason: String?
+    public var reasons: [String]?
+}
@@ public struct RenderGatewayJobResult: Codable, Equatable, Sendable {
     public var remoteRunDir: String?
+    public var missingCapabilities: [RenderGatewayMissingCapability]?
+    public var trustBands: [String: RenderGatewayTrustBand?]?
 }
@@ public struct RenderGatewayJob: Codable, Equatable, Sendable {
     public var result: RenderGatewayJobResult?
+    public var missingCapabilities: [RenderGatewayMissingCapability]?
+    public var trustBands: [String: RenderGatewayTrustBand?]?
     public var links: RenderGatewayJobLinks
@@
     public var isActive: Bool {
         status == .queued || status == .running || status == .submitted
     }
+
+    public var isInspectable: Bool {
+        status == .complete || status == .partial
+    }
```

```diff
diff --git a/ios/Upload/Tests/PickleballUploadTests/UploadManifestTests.swift b/ios/Upload/Tests/PickleballUploadTests/UploadManifestTests.swift
@@ func testRenderGatewayJobDecodesProgressAndReplayManifestURL() throws {
 }
+
+func testPartialJobPreservesMissingCapabilitiesAndTrustBands() throws {
+    let data = Data(#"{"id":"job_1","status":"partial","missing_capabilities":[{"capability":"body","reason":"BODY output missing"}],"trust_bands":{"body":{"badge":"preview","stage":"BODY"}},"result":{"manifest_url":"/api/jobs/job_1/manifest"},"links":{"status":"/api/jobs/job_1"}}"#.utf8)
+    let job = try RenderGatewayJob.decode(data)
+    XCTAssertEqual(job.status, .partial)
+    XCTAssertEqual(job.missingCapabilities?.first?.capability, "body")
+    let bodyBand = job.trustBands?["body"] ?? nil
+    XCTAssertEqual(bodyBand?.badge, "preview")
+    XCTAssertTrue(job.isInspectable)
+    XCTAssertFalse(job.isActive)
+}
```

Rationale: partial is terminal and inspectable, never an alias for complete or
“Replay ready.” Product UI should show the missing list and trust badges before
offering the replay. Preserve unknown server fields at the HTTP boundary; if
the concrete trust-band schema expands, use a lossless JSON value wrapper
rather than dropping keys.
