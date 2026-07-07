"""Clip routes (INFRA-1): presigned multipart upload + per-user clip library.

Flow per the approved design (Sec 5/7): `POST /api/clips` creates the clip doc
and mints presigned S3 multipart PUT part URLs (plus a simple presigned PUT for
the capture sidecar); the client uploads bytes directly to S3 and then calls
`POST /api/clips/{id}/complete` with the part ETags. S3 keys live under
`raw/{user_id}/{clip_id}/`.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..s3 import complete_multipart, presign_multipart_put, presign_put
from ..security import AuthConfig, bearer_auth_dependency

_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
SIDECAR_FILENAME = "capture_sidecar.json"


class CreateClipBody(BaseModel):
    filename: str
    size_bytes: int
    part_size_bytes: int


class CompletedPart(BaseModel):
    part_number: int
    etag: str


class CompleteClipBody(BaseModel):
    upload_id: str
    parts: list[CompletedPart]


def _sanitized_stem(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")


def _safe_video_filename(filename: str) -> str:
    raw = Path(filename or "video.mp4").name
    stem = _sanitized_stem(Path(raw).stem) or "video"
    suffix = Path(raw).suffix.lower()
    if suffix not in _VIDEO_SUFFIXES:
        suffix = ".mp4"
    return f"{stem}{suffix}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _public_clip(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["_id"],
        "filename": doc["filename"],
        "status": doc["status"],
        "size_bytes": doc["size_bytes"],
        "key": doc["video_key"],
        "job_id": doc.get("job_id"),
        "created_at": doc["created_at"].isoformat() if isinstance(doc.get("created_at"), datetime) else doc.get("created_at"),
    }


def build_clips_router(
    *,
    db: Any,
    s3_client: Any,
    bucket: str,
    auth_config: AuthConfig,
) -> APIRouter:
    router = APIRouter()
    require_user = bearer_auth_dependency(auth_config)

    @router.post("/api/clips", status_code=201)
    def create_clip(body: CreateClipBody, user_id: str = Depends(require_user)) -> dict[str, Any]:
        if body.size_bytes <= 0:
            raise HTTPException(status_code=422, detail="size_bytes must be positive")
        if body.part_size_bytes <= 0:
            raise HTTPException(status_code=422, detail="part_size_bytes must be positive")
        clip_id = f"clip_{uuid.uuid4().hex[:16]}"
        safe_name = _safe_video_filename(body.filename)
        slug = _sanitized_stem(Path(safe_name).stem) or "clip"
        video_key = f"raw/{user_id}/{clip_id}/{safe_name}"
        sidecar_key = f"raw/{user_id}/{clip_id}/{SIDECAR_FILENAME}"
        multipart = presign_multipart_put(
            s3_client,
            bucket=bucket,
            key=video_key,
            size_bytes=body.size_bytes,
            part_size_bytes=body.part_size_bytes,
        )
        sidecar_upload_url = presign_put(
            s3_client,
            bucket=bucket,
            key=sidecar_key,
            content_type="application/json",
        )
        now = _utcnow()
        db.clips.insert_one(
            {
                "_id": clip_id,
                "user_id": user_id,
                "filename": safe_name,
                "slug": slug,
                "size_bytes": body.size_bytes,
                "part_size_bytes": body.part_size_bytes,
                "video_key": video_key,
                "sidecar_key": sidecar_key,
                "upload_id": multipart["upload_id"],
                "status": "uploading",
                "job_id": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return {
            "id": clip_id,
            "filename": safe_name,
            "key": video_key,
            "upload_id": multipart["upload_id"],
            "part_count": multipart["part_count"],
            "part_urls": multipart["part_urls"],
            "sidecar_upload_url": sidecar_upload_url,
        }

    @router.post("/api/clips/{clip_id}/complete")
    def complete_clip(
        clip_id: str,
        body: CompleteClipBody,
        user_id: str = Depends(require_user),
    ) -> dict[str, Any]:
        clip = db.clips.find_one({"_id": clip_id, "user_id": user_id})
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        if not body.parts:
            raise HTTPException(status_code=422, detail="parts must not be empty")
        try:
            complete_multipart(
                s3_client,
                bucket=bucket,
                key=clip["video_key"],
                upload_id=body.upload_id,
                parts=[part.model_dump() for part in body.parts],
            )
        except Exception as exc:  # noqa: BLE001 - S3 rejections are client-visible API state
            raise HTTPException(
                status_code=400,
                detail=f"multipart completion failed: {type(exc).__name__}: {exc}",
            ) from None
        now = _utcnow()
        db.clips.update_one(
            {"_id": clip_id},
            {"$set": {"status": "uploaded", "updated_at": now}},
        )
        return {"id": clip_id, "status": "uploaded", "key": clip["video_key"]}

    @router.get("/api/clips")
    def list_clips(user_id: str = Depends(require_user)) -> dict[str, Any]:
        docs = db.clips.find({"user_id": user_id}).sort("created_at", -1)
        return {"clips": [_public_clip(doc) for doc in docs]}

    return router
