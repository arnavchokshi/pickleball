# P7-4c Pre-Launch Security Review

Lane: `w7_securityreview_20260709`  
Reviewed state: committed `HEAD` `d51239be853bb9bec2448502b50fdfbcb1ff3493`, with dirty worktree files ignored unless security-relevant.  
Mode: report-only. No repo code edits.

## Executive Verdict

**NO-GO for a public/pre-launch flag flip until the HIGH findings are resolved or explicitly risk-accepted.**

Summary counts:

| Severity | Count |
|---|---:|
| CRITICAL | 0 |
| HIGH | 3 |
| MEDIUM | 4 |
| LOW | 2 |

## Findings

### HIGH PII-1: Non-owner biometric/derived replay artifacts persist despite the standing session-only default

`NORTH_STAR_ROADMAP.md` says the unresolved biometric consent decision blocks persistent non-owner biometric profiles and defaults all non-owner people to **session-only, non-persistent tracking** until consent is answered (`NORTH_STAR_ROADMAP.md:73-77`). The product storage plan persists raw video and derived artifacts under `raw/`, `artifacts/`, and `bundles/`, with `artifacts/` and `bundles/` staying hot (`docs/specs/2026-07-07-product-infra-design.md:90-99`). The worker uploads the full pipeline output directory and replay bundle to S3 without per-person filtering (`server/worker/daemon.py:204-206`).

The pipeline output includes person tracking, BODY frames, offline body joints/mesh, refined placement, world, trust bands, and replay manifest artifacts (`RUNBOOK.md:107-128`). For clips containing friends or harvested/public players, that is persistent biometric-adjacent data even when no `PlayerProfile` is written.

Impact: Launching accounts/storage as-is can retain non-owner skeletons, meshes, placement tracks, and raw video beyond the session-only default.

Required before launch: define and enforce a product retention boundary for non-owner people. Either capture consent before persistent BODY/ReID/profile/bundle artifacts are stored, or store non-owner derived person data only as session-scoped/transient artifacts and redact it from durable bundles.

### HIGH AUTHZ-1: Production can silently boot into the unauthenticated legacy API if the accounts flag is not flipped

The approved product-infra spec says all `/api/*` routes require JWT except health/login/register/Stripe/worker (`docs/specs/2026-07-07-product-infra-design.md:103-116`). In code, `PICKLEBALL_ACCOUNTS_ENABLED` defaults to off (`server/render_app.py:146-150`), and `render.yaml` still sets it to `"0"` with a dark-flag comment (`render.yaml:26-29`). When accounts are off, the legacy unauthenticated routes are mounted: `POST /api/jobs`, `GET /api/jobs/{job_id}`, `GET /api/jobs/{job_id}/manifest`, and `GET /api/jobs/{job_id}/artifacts/{artifact_path}` (`server/render_app.py:321-404`).

Impact: A misconfigured production deploy exposes unauthenticated upload, GPU processing, status, manifest, and artifact retrieval.

Required before launch: make production fail closed. For launch deploys, require `PICKLEBALL_ACCOUNTS_ENABLED=1` and fail startup unless all auth/Mongo/S3 secrets are present. Keep the legacy API only in explicit local/dev mode.

### HIGH AUTHZ-2: Worker profile endpoint can read any account's profile registry by path parameter

The worker profile endpoint accepts `account_id` directly from the URL and returns that account's registry when the machine bearer token is valid (`server/routes/profiles_worker.py:38-41`). The registry can contain player biometric references such as frozen shape betas and ReID galleries (`threed/racketsport/profile_registry.py:166-181`).

Impact: Any worker-token holder, worker bug, or SSRF-like call path with that token can read arbitrary account profile data, not only the account tied to a claimed job.

Required before launch: scope worker profile reads to a claimed job or server-derived user/account id. Do not let the worker choose arbitrary `account_id` from the URL.

### MEDIUM AUTHZ-3: Worker bearer token can mutate any job id, regardless of claimant

Worker heartbeat and completion routes look up jobs by `_id` only and update them after bearer-token auth (`server/routes/worker.py:153-176`). They do not require the request's `X-Worker-Id` to match the job's recorded `worker_id`, nor require a claimed/running status owned by that worker.

Impact: A compromised or misbehaving worker can mark another worker's job running/complete/failed, including cross-account jobs.

Recommended fix: require `{"_id": job_id, "worker_id": header_worker_id, "status": {"$in": [...]}}` in heartbeat/complete updates and return 409/404 on mismatch.

### MEDIUM AUTHZ-4: Durable artifact prefixes are not account-scoped

The product spec stores bundles as `bundles/{clip_id}/` and artifacts as `artifacts/{job_id}/` (`docs/specs/2026-07-07-product-infra-design.md:90-94`). The worker implements the same prefixes (`server/worker/daemon.py:197-206`). Raw uploads are user-scoped, but derived artifacts are only random-id scoped.

Impact: Random ids reduce guessability, but account id is not a structural isolation boundary for derived private media. Prefix-only IAM/lifecycle/delete policies have less defense in depth.

Recommended fix: use `artifacts/{user_id}/{job_id}/` and `bundles/{user_id}/{clip_id}/`, or store a server-side ownership map and never expose bare bundle keys to clients.

### MEDIUM AUTHZ-5: Multipart completion does not verify the submitted upload id matches the stored clip upload id

`create_clip` stores the multipart `upload_id` in the clip document (`server/routes/clips.py:118`). `complete_clip` owner-checks the clip but passes the caller-supplied `body.upload_id` to S3 without comparing it to the stored one (`server/routes/clips.py:141-153`).

Impact: S3 usually rejects an upload id for the wrong key, so this is not an obvious direct cross-account read/write. It is still a missing integrity check on the upload state machine.

Recommended fix: reject completion unless `body.upload_id == clip["upload_id"]`.

### MEDIUM DEP-1: Python dependencies are mostly ranges/unpinned; offline CVE freshness could not be checked

The render/worker requirements use broad ranges such as `fastapi>=0.115,<1`, `python-multipart>=0.0.9,<1`, `boto3>=1.34,<2`, `stripe>=11,<13`, and test doubles in deploy requirements (`requirements-render.txt:3-18`; `requirements-worker.txt:5-6`). The racketsport requirements also leave runtime dependencies unpinned by design (`requirements-racketsport.txt:10-14`).

Impact: Reproducibility and CVE response are weak without a lockfile or image digest. No network was available, so current vulnerability status is **needs-network**.

Recommended fix: generate a production lock/constraints file for Render and worker images, and run `pip-audit`/`npm audit` in a networked lane before launch.

### LOW SEC-1: `.claude/settings.json` grants git add/commit/push

The repo-local agent settings allow `Bash(git add *)`, `Bash(git commit *)`, and `Bash(git push *)` (`.claude/settings.json:1-7`).

Impact: Not a product runtime vulnerability, but it increases accidental supply-chain/change-management risk for local agents.

Recommended fix: keep only if intentional for this repo; otherwise narrow or remove push permission.

### LOW INFO-1: Client-visible errors can expose internal backend details

Several API paths include raw exception type/message in client-visible error states, for example input download failures (`server/routes/jobs_v2.py:127-132`) and GPU/local runner failures (`server/render_app.py:445-459`).

Impact: May leak internal paths, backend exception names, S3 details, or remote command fragments to an authenticated user. This is lower severity than auth/PII issues, but should be sanitized for launch.

Recommended fix: return stable public error codes/messages to clients and log detailed exceptions server-side.

## Positive Checks

- No committed high-confidence AWS keys, Google API keys, GitHub PATs, OpenAI-style `sk-...` keys, MongoDB SRV URIs, or private key PEM blocks were found by a HEAD-only `git grep` scan.
- Secret configuration is mostly environment-backed. `render.yaml` uses `sync: false` for GPU host, worker bearer token, Stripe secrets, MongoDB URI, AWS keys, JWT secret, and invite code (`render.yaml:24-25`, `render.yaml:39-40`, `render.yaml:59-74`).
- `data/credentials/` is gitignored (`.gitignore:22-28`), and the local credential files observed in this workspace are chmod `600`; they are not tracked.
- The dev auth bypass is fail-closed in the reviewed code: exact flag `"1"`, loopback hostname, `import.meta.env.PROD === false`, and non-production mode are required (`web/replay/src/devAuthBypass.ts:17-32`). AppShell only routes unauthenticated `?manifest=` links to the viewer when that runtime check passes (`web/replay/src/AppShell.tsx:27-35`, `web/replay/src/AppShell.tsx:69-76`). The verifier sets the Vite env only on explicit request (`scripts/racketsport/verify_process_video_viewer.py:57-71`, `scripts/racketsport/verify_process_video_viewer.py:419-425`).
- User-controlled filenames and clip ids are sanitized before filesystem use in the upload path (`server/render_app.py:673-695`; `server/routes/clips.py:44-54`). Artifact download paths are resolved and checked to stay under the job artifact root (`server/render_app.py:395-404`, `server/render_app.py:720-725`).
- Subprocess execution uses argv lists locally (`server/gpu_runner.py:139-140`, `server/gpu_runner.py:392-416`; `server/worker/daemon.py:287-294`). The SSH remote command is a shell string, but user-influenced args are slugged or shell-quoted (`server/gpu_runner.py:303-344`).

## PII / Biometric Flow Map

| Flow | Evidence | Persistence |
|---|---|---|
| Client upload video + capture sidecar | `POST /api/clips` mints S3 keys under `raw/{user_id}/{clip_id}/` (`server/routes/clips.py:83-133`) | S3 raw prefix; delete cascade exists. |
| Queue/job execution | job records include user id, clip id, raw S3 keys (`server/routes/jobs_v2.py:158-218`) | Mongo jobs collection. |
| Worker processing | worker downloads raw video/sidecar, runs `process_video.py`, uploads out dir and replay bundle (`server/worker/daemon.py:122-206`) | S3 `artifacts/` and `bundles/`. |
| Person/body/ReID artifacts | pipeline stages include tracking, BODY mesh/joints, placement, world, trust bands (`RUNBOOK.md:107-128`) | Persisted as job artifacts/bundles unless filtered. |
| Profile machinery | `PlayerProfile` can hold height, frozen shape betas, ReID gallery refs, consent status (`threed/racketsport/profile_registry.py:166-181`) | Flat `runs/profiles` or Mongo profile registry; consent enforcement exists for profiles. |
| Delete cascade | account delete removes raw, artifacts, bundles, jobs, clips, entitlements, profiles (`server/routes/account.py:43-75`) | Manual/JWT+password delete; default lifecycle enforcement is not proven in code. |

## Go / No-Go Checklist

- [ ] HIGH PII-1 resolved: no durable non-owner biometric derivatives without consent, or explicit owner risk acceptance recorded.
- [ ] HIGH AUTHZ-1 resolved: production cannot boot with unauthenticated legacy `/api/jobs`/artifact routes.
- [ ] HIGH AUTHZ-2 resolved: worker profile reads are job/account scoped.
- [ ] MEDIUM worker/job ownership checks added or accepted.
- [ ] S3 artifact/bundle prefix ownership model finalized before real multi-account data.
- [ ] Multipart upload state validates stored `upload_id`.
- [ ] Networked dependency audit run against the exact production lock/image.
- [ ] Live/staging auth smoke: anonymous `/api/jobs` and `/api/jobs/{id}/artifacts/...` fail; authenticated owner can only see own clips/jobs; non-owner gets 404/403.
- [ ] Delete-cascade smoke on staging S3/Mongo with a real clip, verifying raw/artifacts/bundles/profile docs are gone.

## Offline Limits

- No network vulnerability database check was possible; dependency risk is a quick-pass only.
- No live Render, Atlas, S3, worker VM, or browser production build was exercised.
- Secret scan was regex/static only; it does not replace a full entropy scanner such as Gitleaks/TruffleHog.
- I reviewed current tracked files and relevant HEAD state; dirty worktree files observed were not security-relevant to the product auth/storage surfaces, except that `scripts/racketsport/process_video.py` was dirty and was not used as a basis for any passing security claim.
