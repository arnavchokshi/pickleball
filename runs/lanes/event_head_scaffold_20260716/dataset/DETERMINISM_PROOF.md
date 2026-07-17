# Builder determinism proof (replaces duplicate manifest_b.json blob)

The scaffold lane's determinism gate ran build_event_head_dataset.py twice with the same
seed and inputs; the second run (manifest_b.json) was byte-identical to manifest_a.json:

- cmp exit 0 (scaffold lane smoke evidence, 2026-07-16)
- md5(manifest_a.json) = md5(manifest_b.json) = 520325787e0855edacb90bc262e54cdd

The duplicate 5MB blob was originally committed in 40b013ab2 and removed by the G2 manager
per the storage policy's duplicate-tracked-blob check (test_truthful_capabilities);
manifest_a.json remains the single tracked copy and the registered allowlist entry.
