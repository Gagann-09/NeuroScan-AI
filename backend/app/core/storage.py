"""
MinIO object storage client and utility functions.
Credentials are sourced from the centralized config module.

Initialization is lazy to allow test monkeypatching of environment variables.
"""
from minio import Minio
from datetime import timedelta

from app.core.config import get_settings

_minio_client = None


def _get_minio_client() -> Minio:
    """Lazily create and return the MinIO client."""
    global _minio_client
    if _minio_client is None:
        settings = get_settings()
        _minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _minio_client


# Backward compatibility: minio_client as a proxy object
class _MinioClientProxy:
    def __getattr__(self, name):
        return getattr(_get_minio_client(), name)


minio_client = _MinioClientProxy()


def init_buckets():
    buckets = ["neuroscan-bucket", "brats-scans", "kaggle-scans"]
    client = _get_minio_client()
    for bucket in buckets:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)


def upload_file_to_minio(file_path: str, object_name: str, bucket_name: str = "neuroscan-bucket"):
    client = _get_minio_client()
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    client.fput_object(bucket_name, object_name, file_path)


def get_presigned_url(object_name: str, bucket_name: str = "neuroscan-bucket"):
    try:
        settings = get_settings()
        return _get_minio_client().presigned_get_object(bucket_name, object_name, expires=settings.presigned_url_ttl)
    except Exception:
        return None