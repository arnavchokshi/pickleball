"""S3 client + presigned-URL helpers (INFRA-1).

Per the approved design (Sec 5): a single private bucket, presigned multipart
PUT for uploads, presigned GET for downloads, prefix delete for the
delete-cascade (wired up in full in INFRA-5). All functions take an injected
boto3 S3 client so tests run against `moto` instead of real AWS.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import boto3
from botocore.client import Config as BotoConfig

DEFAULT_PRESIGN_EXPIRES_S = 3600


def s3_client_from_env(env: Mapping[str, str] | None = None):
    resolved_env = os.environ if env is None else env
    access_key = resolved_env.get("PICKLEBALL_AWS_ACCESS_KEY_ID", "").strip()
    secret_key = resolved_env.get("PICKLEBALL_AWS_SECRET_ACCESS_KEY", "").strip()
    region = resolved_env.get("PICKLEBALL_S3_REGION", "us-east-1").strip() or "us-east-1"
    kwargs: dict[str, Any] = {
        "region_name": region,
        "config": BotoConfig(signature_version="s3v4"),
    }
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def presign_put(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    expires_in: int = DEFAULT_PRESIGN_EXPIRES_S,
    content_type: str | None = None,
) -> str:
    params: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    return s3_client.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)


def presign_get(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    expires_in: int = DEFAULT_PRESIGN_EXPIRES_S,
) -> str:
    return s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
    )


def presign_multipart_put(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    size_bytes: int,
    part_size_bytes: int,
    expires_in: int = DEFAULT_PRESIGN_EXPIRES_S,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Start a multipart upload and presign a PUT URL for every part.

    `part_count` uses ceiling division so `size_bytes % part_size_bytes == 0`
    lands exactly on the boundary (no dangling empty final part).
    """
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    if part_size_bytes <= 0:
        raise ValueError("part_size_bytes must be positive")

    create_kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if content_type:
        create_kwargs["ContentType"] = content_type
    created = s3_client.create_multipart_upload(**create_kwargs)
    upload_id = created["UploadId"]

    part_count = math.ceil(size_bytes / part_size_bytes)
    part_urls = [
        {
            "part_number": part_number,
            "url": s3_client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=expires_in,
            ),
        }
        for part_number in range(1, part_count + 1)
    ]
    return {
        "upload_id": upload_id,
        "key": key,
        "part_count": part_count,
        "part_urls": part_urls,
    }


def complete_multipart(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    upload_id: str,
    parts: list[dict[str, Any]],
) -> dict[str, Any]:
    """`parts` is a list of `{"part_number": int, "etag": str}` (client-facing
    naming); S3's API wants `PartNumber`/`ETag`."""
    formatted_parts = [
        {"ETag": part["etag"], "PartNumber": part["part_number"]} for part in parts
    ]
    return s3_client.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": formatted_parts},
    )


def delete_prefix(s3_client: Any, *, bucket: str, prefix: str) -> int:
    """Delete every object under `prefix`. Returns the count deleted."""
    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        deleted += len(objects)
    return deleted


def s3_health(s3_client: Any | None, bucket: str) -> dict[str, Any]:
    """Cheap reachability probe for the health endpoint. Never raises."""
    if s3_client is None or not bucket:
        return {"ok": False, "detail": "s3 not configured"}
    try:
        s3_client.head_bucket(Bucket=bucket)
    except Exception as exc:  # noqa: BLE001 - health checks must never 500
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"ok": True}
