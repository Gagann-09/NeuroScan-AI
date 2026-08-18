"""
MinIO object storage client and utility functions.
Credentials are sourced from the centralized config module.
"""
from minio import Minio
from datetime import timedelta

from app.core.config import get_settings

settings = get_settings()

minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)


def init_buckets():
    buckets = ["neuroscan-bucket", "brats-scans", "kaggle-scans"]
    for bucket in buckets:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)


def upload_file_to_minio(file_path: str, object_name: str, bucket_name: str = "neuroscan-bucket"):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    minio_client.fput_object(bucket_name, object_name, file_path)


def get_presigned_url(object_name: str, bucket_name: str = "neuroscan-bucket"):
    try:
        return minio_client.presigned_get_object(bucket_name, object_name, expires=timedelta(hours=2))
    except Exception:
        return None