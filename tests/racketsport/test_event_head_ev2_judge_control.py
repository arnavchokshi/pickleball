from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import torch

from threed.racketsport.event_head.model import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
E1_CODE_COMMIT = "9bbd8011828631b4cc7df4afdf3b1932e758914a"
JUDGE_PATHS = (
    "scripts/racketsport/eval_event_head.py",
    "threed/racketsport/event_head/matcher.py",
)
EXPECTED_JUDGE_SHA256 = {
    "scripts/racketsport/eval_event_head.py": (
        "a0c172f73231113af3c14bcfb8b91dd83415e5406ab89d0439b697d27848e22f"
    ),
    "threed/racketsport/event_head/matcher.py": (
        "2272a01d94a02d6663764b3fc7018f43b70bec428a8ad7c2c3fc125373149b62"
    ),
}
E1_CHECKPOINTS = {
    "A": (
        "runs/lanes/abc_experiment_20260721/vm_pull/seed_20260720/"
        "A_owner_only/best_event_head_finetuned.pt",
        "e38f7c1be382a4e781f0a96b22383bd8636817304443ddb7ca343f2e51837005",
        "ed6473fc302f11fac978b0ff72f656a3bc8a9bcd66c91fc26b2fc08d0c6b43c0",
    ),
    "B": (
        "runs/lanes/abc_experiment_20260721/vm_pull_v2/"
        "B_pbvision_teacher/best_event_head_finetuned.pt",
        "64559c3836108412cbe8ed085f9547aa557c2196b194d17ecbedead5ce9dc2e1",
        "d255898723b72df4bcf8ff977a7fbd1bd99a946658c6e8bbfb229c93725b150c",
    ),
    "C": (
        "runs/lanes/abc_experiment_20260721/vm_pull_v2/"
        "C_placebo/best_event_head_finetuned.pt",
        "691b0f2b680fb4b266a67778c39298d9223df8036692142516d906fa7fff0f7b",
        "34d9f9d03d240d153e91e6ef80bedccf59af10086969ddf905eaf0dd2f4e0675",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture() -> torch.Tensor:
    return (
        torch.arange(1 * 4 * 3 * 32 * 32, dtype=torch.float32)
        .reshape(1, 4, 3, 32, 32)
        .remainder(257)
        .div(256)
    )


def test_e1_judge_files_are_byte_identical_to_recorded_e1_code_commit() -> None:
    subprocess.run(
        ["git", "diff", "--exit-code", E1_CODE_COMMIT, "--", *JUDGE_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {
        relative: _sha256(ROOT / relative) for relative in JUDGE_PATHS
    } == EXPECTED_JUDGE_SHA256


def test_banked_e1_checkpoints_keep_exact_synthetic_fixture_logits() -> None:
    fixture = _fixture()
    fixture_sha256 = hashlib.sha256(
        fixture.detach().contiguous().numpy().tobytes()
    ).hexdigest()
    assert fixture_sha256 == (
        "2425e016a022466776166aab64d938f942b0f7ec34f9d9e5eaf1a12b4f84cfb2"
    )

    for _, (relative, checkpoint_sha256, expected_logits_sha256) in sorted(
        E1_CHECKPOINTS.items()
    ):
        path = ROOT / relative
        assert _sha256(path) == checkpoint_sha256
        model, _ = load_checkpoint(path, device="cpu")
        with torch.inference_mode():
            logits = model(fixture)
        actual_logits_sha256 = hashlib.sha256(
            logits.detach().contiguous().numpy().tobytes()
        ).hexdigest()
        assert actual_logits_sha256 == expected_logits_sha256
