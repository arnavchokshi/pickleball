from __future__ import annotations

import subprocess
from pathlib import Path


def test_refresh_remote_host_replaces_stale_entries_idempotently(tmp_path: Path) -> None:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(
        "\n".join(
            [
                "# fixture header",
                "203.0.113.9 ssh-ed25519 old-ip-key",
                "a100-fleet ssh-rsa old-alias-key",
                "198.51.100.7 ssh-ed25519 keep-key",
                "",
            ]
        ),
        encoding="utf-8",
    )
    keyscan = tmp_path / "keyscan.out"
    keyscan.write_text(
        "\n".join(
            [
                "# 203.0.113.9:22 SSH-2.0-OpenSSH_fixture",
                "203.0.113.9 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixtureKeyBlob1234567890",
                "203.0.113.9 ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDFixtureKeyBlob1234567890",
                "",
            ]
        ),
        encoding="utf-8",
    )

    command = [
        "bash",
        "scripts/fleet/refresh_remote_host.sh",
        "--host",
        "203.0.113.9",
        "--alias",
        "a100-fleet",
        "--known-hosts",
        str(known_hosts),
        "--keyscan-file",
        str(keyscan),
        "--skip-connectivity-check",
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    after_first = known_hosts.read_text(encoding="utf-8")
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    after_second = known_hosts.read_text(encoding="utf-8")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert after_second == after_first
    assert "old-ip-key" not in after_second
    assert "old-alias-key" not in after_second
    assert "198.51.100.7 ssh-ed25519 keep-key" in after_second
    assert "203.0.113.9,a100-fleet ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixtureKeyBlob1234567890" in after_second
    assert "203.0.113.9,a100-fleet ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDFixtureKeyBlob1234567890" in after_second
    assert "PASS refresh_remote_host host=203.0.113.9" in first.stdout
    assert "connectivity_check=skipped_runs_at_fleet_start" in first.stdout


def test_refresh_remote_host_copies_refreshed_known_hosts_into_pinned_worktree(tmp_path: Path) -> None:
    known_hosts = tmp_path / "repo" / "configs" / "ssh" / "a100_known_hosts"
    known_hosts.parent.mkdir(parents=True)
    known_hosts.write_text("198.51.100.7 ssh-ed25519 keep-key\n", encoding="utf-8")
    keyscan = tmp_path / "keyscan.out"
    keyscan.write_text(
        "203.0.113.9 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixtureKeyBlob1234567890\n",
        encoding="utf-8",
    )
    pinned_worktree = tmp_path / "pinned-worktree"
    pinned_known_hosts = pinned_worktree / "configs" / "ssh" / "a100_known_hosts"
    pinned_known_hosts.parent.mkdir(parents=True)
    pinned_known_hosts.write_text("203.0.113.9 ssh-rsa stale-key\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            "scripts/fleet/refresh_remote_host.sh",
            "--host",
            "203.0.113.9",
            "--alias",
            "h100-lane",
            "--known-hosts",
            str(known_hosts),
            "--keyscan-file",
            str(keyscan),
            "--pinned-worktree",
            str(pinned_worktree),
            "--skip-connectivity-check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert pinned_known_hosts.read_text(encoding="utf-8") == known_hosts.read_text(encoding="utf-8")
    assert "pinned_worktrees=1" in completed.stdout
