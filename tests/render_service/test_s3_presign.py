import boto3
import pytest
import requests
from moto import mock_aws

from server.s3 import (
    complete_multipart,
    delete_prefix,
    presign_get,
    presign_multipart_put,
    presign_put,
    s3_client_from_env,
)

BUCKET = "presign-test-bucket"
MB = 1024 * 1024


def _client():
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    client.create_bucket(Bucket=BUCKET)
    return client


def test_s3_client_from_env_uses_region_default_and_override() -> None:
    default_client = s3_client_from_env({})
    override_client = s3_client_from_env(
        {
            "PICKLEBALL_S3_REGION": "eu-west-2",
            "PICKLEBALL_AWS_ACCESS_KEY_ID": "key",
            "PICKLEBALL_AWS_SECRET_ACCESS_KEY": "secret",
        }
    )

    assert default_client.meta.region_name == "us-east-1"
    assert override_client.meta.region_name == "eu-west-2"


def test_presign_multipart_put_part_count_math() -> None:
    with mock_aws():
        client = _client()

        over_boundary = presign_multipart_put(
            client, bucket=BUCKET, key="raw/u/c/a.mp4", size_bytes=10 * MB + 1, part_size_bytes=5 * MB
        )
        exact_boundary = presign_multipart_put(
            client, bucket=BUCKET, key="raw/u/c/b.mp4", size_bytes=10 * MB, part_size_bytes=5 * MB
        )
        single_part = presign_multipart_put(
            client, bucket=BUCKET, key="raw/u/c/c.mp4", size_bytes=3, part_size_bytes=5 * MB
        )

        assert over_boundary["part_count"] == 3
        # size % part_size == 0: exactly size/part_size parts, no empty tail part.
        assert exact_boundary["part_count"] == 2
        assert single_part["part_count"] == 1
        assert [part["part_number"] for part in exact_boundary["part_urls"]] == [1, 2]
        urls = [part["url"] for part in exact_boundary["part_urls"]]
        assert len(set(urls)) == 2
        assert all("partNumber=" in url for url in urls)


def test_presign_multipart_put_rejects_nonpositive_inputs() -> None:
    with mock_aws():
        client = _client()

        with pytest.raises(ValueError):
            presign_multipart_put(
                client, bucket=BUCKET, key="raw/u/c/a.mp4", size_bytes=0, part_size_bytes=5 * MB
            )
        with pytest.raises(ValueError):
            presign_multipart_put(
                client, bucket=BUCKET, key="raw/u/c/a.mp4", size_bytes=10, part_size_bytes=0
            )


def test_complete_multipart_two_parts_lands_full_object() -> None:
    with mock_aws():
        client = _client()
        key = "raw/user_1/clip_1/game.mp4"
        plan = presign_multipart_put(
            client, bucket=BUCKET, key=key, size_bytes=10 * MB, part_size_bytes=5 * MB
        )
        assert plan["part_count"] == 2

        parts = []
        for part in plan["part_urls"]:
            response = requests.put(part["url"], data=b"x" * (5 * MB))
            assert response.status_code == 200, response.text
            parts.append(
                {"part_number": part["part_number"], "etag": response.headers["ETag"].strip('"')}
            )

        complete_multipart(
            client, bucket=BUCKET, key=key, upload_id=plan["upload_id"], parts=parts
        )

        listed = client.list_objects_v2(Bucket=BUCKET, Prefix=key)
        assert listed["KeyCount"] == 1
        assert listed["Contents"][0]["Size"] == 10 * MB


def test_presign_put_and_get_roundtrip() -> None:
    with mock_aws():
        client = _client()
        key = "raw/user_1/clip_1/capture_sidecar.json"

        put_url = presign_put(client, bucket=BUCKET, key=key, content_type="application/json")
        put_response = requests.put(
            put_url, data=b'{"ok":true}', headers={"Content-Type": "application/json"}
        )
        assert put_response.status_code == 200, put_response.text

        get_url = presign_get(client, bucket=BUCKET, key=key)
        get_response = requests.get(get_url)
        assert get_response.status_code == 200
        assert get_response.content == b'{"ok":true}'


def test_delete_prefix_removes_only_that_prefix() -> None:
    with mock_aws():
        client = _client()
        client.put_object(Bucket=BUCKET, Key="raw/u1/c1/a.mp4", Body=b"a")
        client.put_object(Bucket=BUCKET, Key="raw/u1/c1/capture_sidecar.json", Body=b"{}")
        client.put_object(Bucket=BUCKET, Key="raw/u1/c2/keep.mp4", Body=b"k")

        deleted = delete_prefix(client, bucket=BUCKET, prefix="raw/u1/c1/")

        assert deleted == 2
        remaining = client.list_objects_v2(Bucket=BUCKET, Prefix="raw/u1/")
        assert [obj["Key"] for obj in remaining["Contents"]] == ["raw/u1/c2/keep.mp4"]
