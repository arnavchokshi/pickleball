# Render Single-Service Gateway

This deploys one public Render Web Service that serves both surfaces from the
same origin:

- React replay/upload UI from `web/replay/dist`
- FastAPI backend under `/api/*`

## API

`POST /api/jobs` accepts `multipart/form-data`:

- `video`: required `.mp4`, `.mov`, or `.m4v`
- `capture_sidecar`: optional `capture_sidecar.json`
- `court_corners`: optional `court_corners.json`
- `court_calibration`: optional `court_calibration.json`
- `clip`: optional safe clip id
- `max_frames`: optional smoke-test frame cap

The response is a queued job:

```json
{
  "id": "job_...",
  "status": "queued",
  "links": {
    "status": "/api/jobs/job_...",
    "manifest": "/api/jobs/job_.../manifest"
  }
}
```

Swift and web clients poll `GET /api/jobs/{job_id}` until `status` is
`complete` or `failed`. When complete, `result.manifest_url` points at the
replay manifest that the React viewer can open with `/?manifest=...`.

## GPU execution

The backend fails closed unless one real GPU runner is configured.

SSH runner:

- `PICKLEBALL_GPU_SSH_HOST`: Render env var, for example `user@host`
- `PICKLEBALL_GPU_SSH_KEY_PATH`: `/etc/secrets/gcp_ssh_key`
- `PICKLEBALL_GPU_KNOWN_HOSTS_PATH`: `/etc/secrets/gcp_known_hosts`
- `PICKLEBALL_GPU_REPO`: GCP checkout, default `/home/arnavchokshi/pickleball_git`
- `PICKLEBALL_GPU_PYTHON`: GCP venv python, default `/home/arnavchokshi/pickleball_git/.venv/bin/python`

The SSH runner rsyncs the upload to the GCP host, runs
`scripts/racketsport/process_video.py --body-local --device cuda:0`, then syncs
the produced clip artifacts back into the Render job directory.

HTTP worker runner:

- `PICKLEBALL_GPU_WORKER_URL`
- `PICKLEBALL_GPU_WORKER_TOKEN` when the worker requires bearer auth

## Render setup

Use the root `render.yaml` Blueprint. Secret files cannot be committed in the
Blueprint; add `gcp_ssh_key` and `gcp_known_hosts` in the Render Dashboard as
secret files for the service.

The service uses Render's `PORT` and exposes `/api/health` as the health check.
The default plan in `render.yaml` is `free` per Render Blueprint defaults; real
uploads will likely need a paid instance or a persistent datastore once job
history must survive restarts.
