from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_coreml_manifest_matches_tracked_package_files() -> None:
    manifest_path = Path("models_coreml/MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["runtime_status"] == "candidate_assets_tracked_not_app_bundled"
    assert manifest["app_lookup_root"] == "Documents/models_coreml"
    assert manifest["tracked_asset_policy"]["max_package_size_mb_without_lfs_or_release_artifact"] == 25
    assert manifest["packages"]

    for package in manifest["packages"]:
        package_root = Path(package["tracked_path"])
        assert package_root.is_dir(), package["id"]
        assert package["app_documents_path"].startswith("Documents/models_coreml/")
        assert _dir_size_mb(package_root) <= manifest["tracked_asset_policy"]["max_package_size_mb_without_lfs_or_release_artifact"]
        for relative_path, expected_sha256 in package["files_sha256"].items():
            file_path = package_root / relative_path
            assert file_path.is_file(), f"{package['id']} missing {relative_path}"
            assert _sha256(file_path) == expected_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size_mb(path: Path) -> float:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file()) / 1_000_000
