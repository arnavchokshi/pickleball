"""Account-era API routers (INFRA-1), mounted by `create_app` only when
`PICKLEBALL_ACCOUNTS_ENABLED=1` (or `accounts_enabled=True` is injected).

Each module exposes a `build_*_router(...)` factory taking explicit
collaborators (db, s3 client, auth config, runner) so tests inject fakes the
same way `create_app(runner=...)` already works. No module here imports
`server.render_app` — shared job-execution helpers are passed in as kwargs.
"""

from .account import build_account_router
from .auth import build_auth_router
from .clips import build_clips_router
from .jobs_v2 import build_jobs_v2_router
from .profiles_worker import build_profiles_worker_router
from .stripe_webhook import build_stripe_webhook_router

__all__ = [
    "build_account_router",
    "build_auth_router",
    "build_clips_router",
    "build_jobs_v2_router",
    "build_profiles_worker_router",
    "build_stripe_webhook_router",
]
