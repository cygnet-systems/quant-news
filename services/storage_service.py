"""S3-compatible object storage service.

Works with MinIO (local dev), Railway Object Storage, AWS S3, or Cloudflare R2.
Stores rendered reports (Markdown, PDF, HTML) and any other large blobs.
"""

import io
import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config import STORAGE

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=STORAGE.ENDPOINT_URL,
            aws_access_key_id=STORAGE.ACCESS_KEY,
            aws_secret_access_key=STORAGE.SECRET_KEY,
            region_name=STORAGE.REGION,
            # Bounded, always: an object-store stall without timeouts blocks
            # the archive step, and with it the whole run, indefinitely
            # (prime suspect in the 2026-09-01 54-minute tail hang). Better
            # to fail the archive, mark the run partial, and keep the data
            # that is already in Postgres.
            config=BotoConfig(signature_version="s3v4",
                              connect_timeout=10, read_timeout=60,
                              retries={"max_attempts": 3}),
        )
    return _client


def ensure_bucket():
    """Create the bucket if it doesn't exist (idempotent)."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=STORAGE.BUCKET_NAME)
    except ClientError:
        try:
            client.create_bucket(Bucket=STORAGE.BUCKET_NAME)
            logger.info(f"Created bucket: {STORAGE.BUCKET_NAME}")
        except ClientError as e:
            logger.warning(f"Could not create bucket: {e}")


def upload_report(
    storage_key: str,
    content: str | bytes,
    content_type: str = "text/markdown",
    metadata: Optional[dict] = None,
) -> int:
    """Upload a report to object storage.

    Args:
        storage_key: S3 object key (e.g. 'reports/EIX/2026-07-13/full.md').
        content: File content (str or bytes).
        content_type: MIME type.
        metadata: Optional user-defined metadata dict.

    Returns:
        Size in bytes of the uploaded object.
    """
    client = _get_client()
    if isinstance(content, str):
        content = content.encode("utf-8")

    extra = {}
    if metadata:
        extra["Metadata"] = {k: str(v) for k, v in metadata.items()}

    client.put_object(
        Bucket=STORAGE.BUCKET_NAME,
        Key=storage_key,
        Body=content,
        ContentType=content_type,
        **extra,
    )
    logger.info(f"Uploaded {storage_key} ({len(content)} bytes)")
    return len(content)


def download_report(storage_key: str) -> Optional[bytes]:
    """Download a report from object storage.

    Returns:
        Raw bytes, or None if not found.
    """
    client = _get_client()
    try:
        response = client.get_object(
            Bucket=STORAGE.BUCKET_NAME, Key=storage_key
        )
        return response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def delete_report(storage_key: str) -> bool:
    """Delete a report from object storage."""
    client = _get_client()
    try:
        client.delete_object(Bucket=STORAGE.BUCKET_NAME, Key=storage_key)
        return True
    except ClientError:
        return False


def report_exists(storage_key: str) -> bool:
    """Check if a report exists without downloading it."""
    client = _get_client()
    try:
        client.head_object(Bucket=STORAGE.BUCKET_NAME, Key=storage_key)
        return True
    except ClientError:
        return False


def list_reports(prefix: str = "reports/", max_keys: int = 1000) -> list[dict]:
    """List reports under a prefix.

    Returns:
        List of dicts with 'Key', 'Size', 'LastModified'.
    """
    client = _get_client()
    try:
        response = client.list_objects_v2(
            Bucket=STORAGE.BUCKET_NAME, Prefix=prefix, MaxKeys=max_keys
        )
        return [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            }
            for obj in response.get("Contents", [])
        ]
    except ClientError:
        return []


