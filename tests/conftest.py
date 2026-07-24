from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolved_basetemp(config: pytest.Config) -> Path | None:
    raw_basetemp = config.option.basetemp
    if raw_basetemp is None:
        return None
    path = Path(raw_basetemp)
    if not path.is_absolute():
        path = Path(str(config.invocation_params.dir)) / path
    return path.resolve()


def pytest_configure(config: pytest.Config) -> None:
    basetemp = _resolved_basetemp(config)
    if basetemp is None:
        return
    if basetemp == REPO_ROOT or REPO_ROOT in basetemp.parents:
        raise pytest.UsageError(
            "Refusing an in-repo pytest --basetemp because model tests can create "
            "hundreds of megabytes per case. Omit --basetemp or use a path under "
            "/tmp instead."
        )
