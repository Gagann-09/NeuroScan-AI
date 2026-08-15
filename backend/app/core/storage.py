import os
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

minio_client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

def init_buckets():
    """Creates the necessary storage buckets if they don't exist."""
    buckets = ["brats-scans", "kaggle-scans", "segmentation-masks", "xai-visualizations"]
    for bucket in buckets:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
            print(f"[Storage] Bucket '{bucket}' created successfully.")
        else:
            print(f"[Storage] Bucket '{bucket}' already exists.")