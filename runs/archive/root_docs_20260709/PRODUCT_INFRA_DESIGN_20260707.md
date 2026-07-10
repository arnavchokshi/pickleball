# Product Infrastructure Design — accounts, storage, queue, and client wiring (P7-1 / P0-9 / P7-4b)

**Date:** 2026-07-07 · **Status:** owner-approved design (approach + all sections approved as-is)
**Owner rulings captured here:** auth = own accounts in MongoDB (email+password, JWT); stack =
MongoDB Atlas + AWS S3 + GCP GPU fleet + Render hosting; React web, Swift iOS; Stripe scaffolded
now / live later (web + mobile); budget ≤ $30/mo for always-on services.
**SUPERSEDES:** the TECH_BLUEPRINTS P7-1 ruling "Render Persistent Disk + SQLite + Render Key Value
+ Background Worker" (TECH_BLUEPRINTS.md ~2807-2817). No lane should build against that plan.
The typed purchase-approval STOP on paid tiers is RESOLVED by the owner's $30/mo grant (2026-07-07).

## 1. Goal

Serve the North Star Definition of Done v1 (NORTH_STAR_ROADMAP.md I.7): a user records a game in
our iOS app, uploads it, and with zero human intervention gets back — within ≤2× game duration — a
QA-passed 3D world + coaching card. This spec covers the *product plumbing* that makes that
multi-user, durable, and private: accounts, storage, job queue, client wiring. It does NOT touch
the 17-stage pipeline, accuracy work (P1-P4/PF), the fleet ≤$5/hr policy, or eval/gate rules.

## 2. What exists today (verified 2026-07-07 by repo sweep)

- `server/render_app.py` — FastAPI gateway, tested, deployed via `render.yaml` (free tier).
  Upload → job → SSH push to GCP VM → pipeline → artifacts → viewer manifest all work E2E.
- Gaps (the repo's own words): JobStore = JSON files in ephemeral `/tmp`; jobs run in in-process
  `BackgroundTasks` (die on restart); **zero client-facing auth**; no database; no object storage.
- `web/replay` — React 19 + Three.js viewer with a working `UploadPanel`; built `dist/` is served
  as static files by the same FastAPI service.
- `ios/` — real Swift modules; `RenderGatewayClient` (URLSession, multipart, unit-tested) exists
  but is **dead code**: no call site in the app. `UploadManifest`/`ResumableChunkPlan` primitives
  exist and are tested. Capture sidecar contract is correct and untouched by this spec.
- GCP — spot-VM fleet driven by SSH/rsync; no GCS/managed services. Spot IPs recycle on restart,
  and the Render→VM SSH push requires hand-syncing the IP (documented recurring pain).
- MongoDB / S3 / Firebase / Supabase: zero prior usage or mentions anywhere in the repo.

## 3. Architecture (approved: "A — pull-based worker")

```
iOS app ──┐                          ┌── GCP GPU VM(s) [existing fleet]
React web ─┼──▶ Render API (FastAPI) │    worker daemon (new, small):
           │      │ JWT auth          │    poll /api/worker/next-job → atomic claim
           │      ▼                   │    S3 pull video → run process_video.py →
           │    MongoDB Atlas ◀─poll──┘    S3 push artifacts → heartbeat/progress
           │    (users, jobs=queue,
           │     clips, profiles,
           │     entitlements)
           └──▶ S3 presigned upload/download
                (raw video, artifacts, viewer bundles)
```

Component responsibilities:

- **Render API service** (evolves `server/render_app.py`; Starter plan $7/mo): stateless FastAPI.
  Owns auth, job records, presigned S3 URL minting, worker endpoints, Stripe webhook stub; serves
  the built React SPA. Restart-safe because all state lives in Mongo + S3.
- **MongoDB Atlas** (Flex, ~$8/mo): system of record AND the job queue (atomic
  `findOneAndUpdate` claim — no Redis at this scale).
- **AWS S3** (one private bucket): all durable bytes. Clients transfer video/artifacts directly
  via short-lived presigned URLs — multi-GB game files never transit Render.
- **GPU worker daemon** (new; systemd unit installed by the VM startup path): pull-based job
  consumer on the existing spot VMs. Kills the SSH/IP-sync fragility; spot preemption mid-job →
  stale heartbeat → auto-requeue.
- **Clients**: React SPA (login/library/upload + existing viewer) and iOS (wire the dormant
  `RenderGatewayClient` into capture flow + sign-in + Keychain).

## 4. MongoDB data model

Collections (all documents carry `created_at`/`updated_at`):

- `users` — email (unique, lowercased), `password_hash` (argon2id), `last_login_at`,
  `stripe_customer_id` (null until Stripe goes live), consent flags + retention prefs (P7-4b),
  `deleted_at` tombstone.
- `refresh_tokens` — `token_hash` (never plaintext), `user_id`, device label, `expires_at`,
  `rotated_from`, `revoked_at`. One row per device; rotation revokes the predecessor.
- `jobs` — status machine `queued → claimed → running → succeeded | failed`; `attempts` (max 2),
  `worker_id`, `heartbeat_at`, `progress` (stage-level, same `PIPELINE_SUMMARY` structure the
  UploadPanel already renders), S3 keys (input/artifacts/bundle), structured `error`.
- `clips` — per-user library row: `user_id`, capture metadata + sidecar fingerprint, S3 keys,
  `job_id`, viewer-bundle pointer, retention class.
- `profiles` — the P0-9 registry ported from flat JSON (`runs/profiles/<account>/`) to Mongo:
  same 5 Pydantic schemas (court / device / player / gear / session-cache), same consent
  enforcement, new storage backend. Pipeline consumption semantics unchanged (profile present →
  use it; absent → generic path + trust band).
- `entitlements` — empty Stripe scaffold (see §9).

**Delete-cascade (P7-4b, designed-in):** every derived artifact traces to `user_id` + `clip_id`.
`DELETE /api/account` (and per-clip delete) tombstones the user doc, deletes S3 prefixes
`raw/{user}/…`, `artifacts/{job}/…`, `bundles/{clip}/…`, and drops derived docs. The
cross-account-shared player-profile case follows the P0-9 consent scopes already modeled in
`threed/racketsport/profile_registry.py` (`session_only` / `account_lifetime` / `delete_after_days`).

## 5. S3 layout

- Bucket: single, private, versioning ON, default SSE. Prefixes:
  `raw/{user_id}/{clip_id}/` (video + sidecar) · `artifacts/{job_id}/` (full pipeline output dir)
  · `bundles/{clip_id}/` (viewer-only: manifest, world JSON, mesh chunks, court map).
- **Uploads:** presigned multipart PUT direct from client (iOS `ResumableChunkPlan` finally used).
- **Downloads:** presigned GET minted per-request by the API; nothing public. Viewer manifests are
  rewritten at serve time so artifact URLs are presigned.
- Lifecycle: `raw/` → Glacier Instant Retrieval after 60 days (overridable per user retention
  prefs); `artifacts/`+`bundles/` stay hot. CloudFront deferred until real traffic.
- IAM: one scoped user/role — bucket-only, prefix-scoped where practical; worker VM gets its own
  key pair distinct from the API's.

## 6. Auth

- Register/login with email+password; **argon2id** hashing. Registration invite-gated via a
  single `INVITE_CODE` env var (owner+friends scope; public signup is a later decision).
- Sessions: **15-min JWT access token** (HS256, secret in Render env) + **30-day rotating refresh
  token** (hashed at rest, per-device; reuse of a rotated token revokes the whole chain).
- All `/api/*` routes require JWT except `health`, `login`, `register`, Stripe webhook (verified
  by Stripe signature instead), and `worker/*` (separate long-lived worker bearer token, rotated
  manually; workers are our own VMs, not user devices).
- Web: access token in memory, refresh token in httpOnly SameSite cookie (no localStorage).
  iOS: both in Keychain. Login/register rate-limited per-IP (slowapi).
- Password reset: **deferred** to a follow-up lane (needs an email sender — Resend or SES free
  tier); explicitly not blocking v1 since registration is invite-gated and owner can reset by CLI.
- Account deletion: §4 delete-cascade endpoint, JWT-authenticated + password re-confirmation.

## 7. Job flow end-to-end

1. Client `POST /api/clips` → API creates clip doc + returns presigned multipart upload URLs.
2. Client uploads video+sidecar directly to S3; `POST /api/clips/{id}/complete` verifies parts.
3. Client `POST /api/jobs` (clip id + options) → job doc `queued`.
4. Worker daemon long-polls `GET /api/worker/next-job` → API claims atomically
   (`findOneAndUpdate` on `status: queued`, sets `claimed` + `worker_id` + `heartbeat_at`).
5. Daemon: S3 pull → run `scripts/racketsport/process_video.py` **unchanged** → POST stage
   progress as heartbeats → S3 push `artifacts/` + `bundles/` → `succeeded`.
6. Requeue rule: `heartbeat_at` stale > 5 min → API flips back to `queued`, `attempts += 1`;
   after 2 attempts → `failed` with partial `PIPELINE_SUMMARY` preserved (honest failure surface).
7. No VM up → jobs wait in `queued`; clients show queue position, never a fake spinner. (Fleet
   policy governs when VMs exist; a queued-jobs-waiting signal in the fleet ledger is a follow-up.)

## 8. Clients

- **React (`web/replay` evolves in place):** three new screens — sign-in, library (clips + live
  job status), upload — with the existing 3D viewer as the detail view. Same single-service
  deploy (Vite build served by FastAPI): no new hosting cost, no CORS surface beyond S3's.
- **iOS:** minimal sign-in screen; Keychain token storage; wire `RenderGatewayClient` into the
  capture flow (record → Analyze → presigned upload → job poll → replay deep link). Client gains
  auth headers + presigned-S3 upload path alongside its existing multipart method (which remains
  for local/dev gateways). Capture sidecar contract untouched.

## 9. Stripe (scaffold only — owner has account; live = later, P7-3)

- Now: `stripe_customer_id` on users, `entitlements` collection, `POST /api/stripe/webhook` stub
  (signature-verified, feature-flagged off), price/product IDs as env placeholders.
- Later (P7-3 pricing decision): web = Stripe Checkout. Mobile note recorded for that decision:
  Apple requires IAP for in-app digital purchases, but US apps may link out to external web
  checkout (2025 anti-steering rulings) — this scaffold supports the link-out path; IAP-vs-linkout
  is decided at P7-3, not here.

## 10. Ops, security, observability

- Sentry free tier on API, worker daemon, and React. Structured job errors in Mongo.
- Atlas: daily backups (Flex built-in), IP allowlist = Render egress + fleet VMs, SCRAM user per
  service. S3: versioning + block-public-access. Secrets: Render env vars + worker `.env` on VM
  (never committed). Keys needed from owner at implementation time: Atlas connection string, AWS
  IAM keys (bucket-scoped ×2), Render dashboard/API access, Sentry DSN (optional).
- Health: `/api/health` extends to report Mongo + S3 reachability and queue depth.

## 11. Testing

- Extend `tests/render_service/`: auth flows (register/login/refresh-rotation/revocation/rate
  limit), queue semantics (claim atomicity, heartbeat requeue, attempt cap), presigned-URL flows
  (moto for S3, mongomock or testcontainers for Mongo), delete-cascade.
- Worker daemon: unit tests + one integration test against a local FastAPI + moto.
- iOS: extend existing `PickleballUploadTests` for auth headers + presigned path. Web: Vitest for
  the new screens. E2E gate: owner uploads a real clip from the iOS app through the full flow.

## 12. Rollout (each step = one lane, own gate; current demo flow never breaks)

1. **INFRA-1** Mongo + auth + S3 into the API (old /tmp flow still working behind a flag).
2. **INFRA-2** Worker daemon + queue cutover; retire `SshGpuRunner` push path (kept as fallback
   flag for one wave).
3. **INFRA-3** React sign-in/library/upload screens.
4. **INFRA-4** iOS wiring (sign-in, Keychain, capture→upload→replay).
5. **INFRA-5** Delete-cascade + retention pass + P0-9 profile port (pairs with wave-5 P4-0 work).

Provisioning (Atlas project, S3 bucket + IAM, Render upgrade) happens at INFRA-1 start, with owner
keys. Budget check at each lane exit vs the $30/mo ceiling.

## 13. Cost

Render Starter $7 + Atlas Flex ~$8 + S3 ~$3-5 + Sentry $0 ≈ **$18-20/mo** (ceiling $30; headroom
reserved for Resend/SES + growth). GPU spot unchanged under fleet policy (≤$5/GPU/hr, ≤4 GPUs).

## 14. Out of scope (explicit)

Pipeline accuracy (P1-P4/PF), coaching product (P6), pricing/payments live (P7-3), public signup,
autoscaling, CloudFront, password reset email (follow-up lane), GCS migration, IaC/terraform.
