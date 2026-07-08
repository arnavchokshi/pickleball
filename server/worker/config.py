"""Worker daemon configuration (INFRA-2).

`worker_config_from_env(env)` mirrors the `*_from_env(env: Mapping | None)`
convention used everywhere else in the product-infra stack
(`server/gpu_runner.py:runner_from_env`, `server/s3.py:s3_client_from_env`,
`server/db.py:mongo_client_from_env`, `server/security.py:auth_config_from_env`):
real env is read only when `env=None`, so tests inject a literal dict
instead of monkeypatching `os.environ`. Never raises on missing/empty
values -- callers decide how to degrade (the `--check-config` CLI path
prints whatever resolved, even if empty; the real network calls in
`daemon.main()` are what actually fail on a misconfigured box).
"""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from typing import Mapping

DEFAULT_POLL_WAIT_S = 25
DEFAULT_HEARTBEAT_INTERVAL_S = 60
DEFAULT_COMMAND_TIMEOUT_S = 7200
DEFAULT_WORK_DIR = "/tmp/pickleball_worker_jobs"
DEFAULT_S3_REGION = "us-east-1"


@dataclass(frozen=True)
class WorkerConfig:
    api_base_url: str
    worker_bearer_token: str
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    s3_bucket: str
    s3_region: str
    pipeline_python: str
    repo_dir: str
    worker_id: str
    poll_wait_s: int
    heartbeat_interval_s: int
    command_timeout_s: int
    work_dir: str


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def worker_config_from_env(env: Mapping[str, str] | None = None) -> WorkerConfig:
    resolved_env = os.environ if env is None else env
    return WorkerConfig(
        api_base_url=resolved_env.get("PICKLEBALL_WORKER_API_BASE_URL", "").strip(),
        worker_bearer_token=resolved_env.get("PICKLEBALL_WORKER_BEARER_TOKEN", "").strip(),
        aws_access_key_id=resolved_env.get("PICKLEBALL_AWS_ACCESS_KEY_ID", "").strip() or None,
        aws_secret_access_key=resolved_env.get("PICKLEBALL_AWS_SECRET_ACCESS_KEY", "").strip() or None,
        s3_bucket=resolved_env.get("PICKLEBALL_S3_BUCKET", "").strip(),
        s3_region=resolved_env.get("PICKLEBALL_S3_REGION", DEFAULT_S3_REGION).strip() or DEFAULT_S3_REGION,
        pipeline_python=resolved_env.get("PICKLEBALL_WORKER_PIPELINE_PYTHON", "").strip(),
        repo_dir=resolved_env.get("PICKLEBALL_WORKER_REPO_DIR", "").strip(),
        worker_id=resolved_env.get("PICKLEBALL_WORKER_ID", "").strip() or _default_worker_id(),
        poll_wait_s=int(resolved_env.get("PICKLEBALL_WORKER_POLL_WAIT_S", str(DEFAULT_POLL_WAIT_S))),
        heartbeat_interval_s=int(
            resolved_env.get("PICKLEBALL_WORKER_HEARTBEAT_INTERVAL_S", str(DEFAULT_HEARTBEAT_INTERVAL_S))
        ),
        command_timeout_s=int(
            resolved_env.get("PICKLEBALL_WORKER_COMMAND_TIMEOUT_S", str(DEFAULT_COMMAND_TIMEOUT_S))
        ),
        work_dir=resolved_env.get("PICKLEBALL_WORKER_WORK_DIR", DEFAULT_WORK_DIR).strip() or DEFAULT_WORK_DIR,
    )
