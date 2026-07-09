# VM train/inference tensor-contract check — TRANSCRIPT-RECONSTRUCTED EXCERPT

**HONESTY HEADER: this is NOT a pulled log file.** The on-VM pytest runs below were executed
interactively over SSH (stdout streamed to the lane session, not tee'd to a file on the VM),
so no log artifact existed to pull before the VM was deleted. The commands and outputs below
are reconstructed verbatim from this lane's own session transcript (the exact tool-call
commands and the exact returned stdout). Nothing here is re-run, inferred, or fabricated.
Timestamp bound: both runs executed between 2026-07-09T06:36:22Z (a UTC date check
immediately before them in the transcript) and ~2026-07-09T06:37Z (the ARM3A training
process launched immediately after; a later `ps aux` snapshot shows its start time as 06:37).
The VM was `pickleball-h100-w7ball` (34.143.198.82), repo at committed HEAD
8721f786101bcc8f9634745f63b4d389f49693cc.

## Run 1 — full targeted suite (the w5-precedent "20 passed" code-identity proof)

Command (exact, as issued from the Mac):

```
ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no arnavchokshi@34.143.198.82 \
  "cd ~/coldstart_20260706/repo && touch /tmp/w7ball_heartbeat && \
   MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport/test_roboflow_corpus.py \
   tests/racketsport/test_ball_stage2_training.py -q 2>&1 | tail -40"
```

Output (exact, complete as returned — the `tail -40` window captured the whole summary):

```
..F..................                                                    [100%]
=================================== FAILURES ===================================
_________ test_loader_smoke_reads_real_roboflow_samples_across_sources _________

    def test_loader_smoke_reads_real_roboflow_samples_across_sources() -> None:
        manifest_path = Path("data/roboflow_universe_20260706/manifest.json")
        if not manifest_path.is_file():
            pytest.skip("Roboflow universe manifest is not present in this checkout")

        smoke = load_smoke_samples(manifest_path, repo_root=Path("."), limit=50, min_datasets=5)

>       assert smoke["opened_samples"] >= 50
E       assert 0 >= 50

tests/racketsport/test_roboflow_corpus.py:177: AssertionError
=========================== short test summary info ============================
FAILED tests/racketsport/test_roboflow_corpus.py::test_loader_smoke_reads_real_roboflow_samples_across_sources
1 failed, 20 passed in 6.50s
```

Result line: **`1 failed, 20 passed in 6.50s`** — the 20 passed matches the w5 lane's
20/20 code-identity figure; the single failure is the roboflow manifest.json smoke test
(`opened_samples=0`), a different file than the `aggregated/corpus_index.json` this lane
verified present (61,260 samples) and orthogonal to the stage-2 training path used by ARM3/4.

## Run 2 — the specific tensor/label-geometry contract test, verbose

Command (exact, as issued from the Mac):

```
ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no arnavchokshi@34.143.198.82 \
  "cd ~/coldstart_20260706/repo && MPLBACKEND=Agg .venv/bin/python -m pytest \
   tests/racketsport/test_ball_stage2_training.py::test_stage2_dataset_tensor_and_label_geometry_use_wasb_official_affine \
   -v 2>&1 | tail -15"
```

Output (exact, complete as returned):

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0 -- /home/arnavchokshi/coldstart_20260706/repo/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/arnavchokshi/coldstart_20260706/repo
configfile: pyproject.toml
plugins: hydra-core-1.3.3, anyio-4.14.1
collecting ... collected 1 item

tests/racketsport/test_ball_stage2_training.py::test_stage2_dataset_tensor_and_label_geometry_use_wasb_official_affine PASSED [100%]

============================== 1 passed in 1.42s ===============================
```

Result line: **`1 passed in 1.42s`** — this is the test that pushes a sample through the
stage-2 training dataset path and asserts the tensor + label geometry match the official WASB
affine contract (the wave-4 preprocessing-contract bug class), satisfying the lane brief's
mandatory pre-ARM3 contract check via the repo's stamped contract test rather than an ad-hoc
tensor diff.

## Known reconstruction limits

- No per-command wall-clock timestamps were captured for these two SSH calls; only the
  06:36:22Z-to-~06:37Z bound stated in the header is evidenced by adjacent transcript entries.
- The `tail -40` / `tail -15` filters mean any pytest header lines above the shown windows
  (for Run 1) were not captured in the transcript and are therefore not reproduced here.
  The pass/fail counts and failure detail shown are the complete, unedited returned text.
