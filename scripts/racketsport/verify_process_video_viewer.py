#!/usr/bin/env python3
"""Headless web-viewer load check for a process_video.py run's manifest.

Starts (or reuses) the local `web/replay` Vite dev server, loads the
manifest through Playwright headless Chromium, and reports console/page
errors plus a screenshot -- the same style of check
`runs/scrubber_v0_20260702T015724Z/burlington/verify_viewer.py` used for the
W3-SCRUBBER-V0 build. This is best-effort: `process_video.py`'s `verify`
stage already treats any failure here (including Playwright not being
installed in the current interpreter) as a non-fatal `degraded` stage, never
a pipeline crash.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WEB_REPLAY_DIR = ROOT / "web" / "replay"
DEV_SERVER_HOST = "127.0.0.1"
DEV_SERVER_PORT = 5173
ENTITY_COUNT_LABELS = ("Players", "Mesh Frames", "Solid Mesh Frames", "Floor Frames", "Ball Contacts", "Replay Points")


def viewer_url_for_manifest(manifest_path: Path | str | None) -> str:
    path = _require_manifest_path(manifest_path)
    return f"http://{DEV_SERVER_HOST}:{DEV_SERVER_PORT}/?manifest=/@fs{path}"


def assert_non_empty_entity_counts(loaded_counts: dict[str, Any], *, allow_empty: bool = False) -> None:
    if allow_empty:
        return
    entity_counts = {label: loaded_counts.get(label, 0) for label in ENTITY_COUNT_LABELS}
    if any(_numeric_count(value) > 0 for value in entity_counts.values()):
        return
    raise AssertionError(f"empty viewer: expected at least one nonzero entity count; loaded_counts={entity_counts}")


def write_headless_verify_report(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "headless_verify.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def _require_manifest_path(manifest_path: Path | str | None) -> Path:
    if manifest_path is None:
        raise ValueError("manifest path is required")
    if isinstance(manifest_path, str) and not manifest_path.strip():
        raise ValueError("manifest path is required")
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    return path.resolve()


def _numeric_count(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if value == value else 0.0
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _port_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def _wait_for_port(host: str, port: int, *, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def _collect_loaded_counts(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const counts = {};
          document.querySelectorAll(".status-grid > div").forEach((node) => {
            const label = node.querySelector("dt")?.textContent?.trim();
            const value = node.querySelector("dd")?.textContent?.trim();
            if (label) counts[label] = value ?? "";
          });
          return counts;
        }"""
    )


def _collect_load_errors(page: Any) -> list[str]:
    return page.locator(".load-error").all_inner_texts()


def verify_viewer_loads(
    manifest_path: Path | str | None,
    *,
    out_dir: Path,
    timeout_s: float = 45.0,
    allow_empty: bool = False,
) -> dict[str, Any]:
    manifest_path = _require_manifest_path(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            f"playwright is not installed in this interpreter ({sys.executable}); "
            "pip install playwright && playwright install chromium"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    started_server = False
    dev_process: subprocess.Popen | None = None
    if not _port_open(DEV_SERVER_HOST, DEV_SERVER_PORT):
        dev_process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--host", DEV_SERVER_HOST, "--port", str(DEV_SERVER_PORT)],
            cwd=str(WEB_REPLAY_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started_server = True
        if not _wait_for_port(DEV_SERVER_HOST, DEV_SERVER_PORT, timeout_s=timeout_s):
            dev_process.terminate()
            raise RuntimeError(
                f"vite dev server did not start listening on {DEV_SERVER_HOST}:{DEV_SERVER_PORT} within {timeout_s}s"
            )

    console_messages: list[str] = []
    page_errors: list[str] = []
    notes: list[str] = []
    screenshots: list[str] = []
    assertion_errors: list[str] = []
    load_errors: list[str] = []
    loaded_counts: dict[str, Any] = {}
    trust_chip_count = 0
    url = viewer_url_for_manifest(manifest_path)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 960})
            page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_selector(".world-panel canvas", timeout=15000)
            page.wait_for_timeout(1500)
            screenshot_path = out_dir / "process_video_verify.png"
            page.screenshot(path=str(screenshot_path))
            screenshots.append(str(screenshot_path))
            loaded_counts = _collect_loaded_counts(page)
            load_errors = _collect_load_errors(page)
            trust_chip_count = page.locator(".trust-badge-chip").count()
            browser.close()
        notes.append(f"loaded {url} headless; {trust_chip_count} trust badge chip(s) rendered")
    finally:
        if started_server and dev_process is not None:
            dev_process.terminate()
            try:
                dev_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dev_process.kill()

    console_errors = [m for m in console_messages if m.startswith("[error]")]
    try:
        assert_non_empty_entity_counts(loaded_counts, allow_empty=allow_empty)
    except AssertionError as exc:
        assertion_errors.append(str(exc))
    page_errors.extend(load_errors)
    ok = not page_errors and not console_errors and not assertion_errors
    if page_errors:
        notes.append(f"{len(page_errors)} page error(s): {page_errors[:3]}")
    if console_errors:
        notes.append(f"{len(console_errors)} console error(s): {console_errors[:3]}")
    if assertion_errors:
        notes.append(f"{len(assertion_errors)} assertion error(s): {assertion_errors[:3]}")

    result = {
        "ok": ok,
        "url": url,
        "notes": notes,
        "screenshots": screenshots,
        "headless_verify_json": str(out_dir / "headless_verify.json"),
        "loaded_counts": loaded_counts,
        "page_errors": page_errors,
        "assertion_errors": assertion_errors,
        "console_message_count": len(console_messages),
        "page_error_count": len(page_errors),
        "assertion_error_count": len(assertion_errors),
        "trust_chip_count": trust_chip_count,
        "allow_empty": allow_empty,
    }
    write_headless_verify_report(out_dir, result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Headless web-viewer load check for a process_video.py manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-empty", action="store_true", help="Allow a manifest that intentionally renders zero viewer entities.")
    args = parser.parse_args(argv)
    result = verify_viewer_loads(args.manifest, out_dir=args.out_dir, allow_empty=args.allow_empty)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
