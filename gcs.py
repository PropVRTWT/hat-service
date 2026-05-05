"""
GCS upload helpers for the HAT service.

Environment variables:
  GCS_BUCKET_NAME   (required) – bucket to upload results into
  GCS_SIGNED_URLS   (optional) – set to "true" to return 24-hour V4 signed URLs
                                  instead of public object URLs.
                                  Requires the Cloud Run service account to have
                                  the roles/iam.serviceAccountTokenCreator role
                                  on itself (or iam.serviceAccounts.signBlob perm).
"""
import datetime
import os
import uuid
from urllib.parse import quote

import google.auth
import google.auth.transport.requests
from google.cloud import storage

_client: storage.Client | None = None


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def _bucket_name() -> str:
    name = os.environ.get("GCS_BUCKET_NAME", "")
    if not name:
        raise RuntimeError(
            "GCS_BUCKET_NAME environment variable is not set. "
            "Set it to the GCS bucket where upscaled files should be stored."
        )
    return name


def _make_firebase_url(blob: storage.Blob) -> str:
    """Return a Firebase Storage download URL with a persistent token."""
    token = str(uuid.uuid4())
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.patch()

    encoded_path = quote(blob.name, safe="")
    return (
        f"https://firebasestorage.googleapis.com/v0/b/{blob.bucket.name}"
        f"/o/{encoded_path}?alt=media&token={token}"
    )


def _make_url(blob: storage.Blob) -> str:
    """Return a Firebase Storage URL, or a V4 signed URL if configured."""
    if os.getenv("GCS_SIGNED_URLS", "false").lower() == "true":
        credentials, _ = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())
        return blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=24),
            method="GET",
            credentials=credentials,
        )
    return _make_firebase_url(blob)


def upload_bytes(data: bytes, filename: str, content_type: str) -> str:
    """Upload raw bytes to GCS and return a URL to the object."""
    blob_name = f"upscaled/hat_model/{uuid.uuid4()}/{filename}"
    blob = _get_client().bucket(_bucket_name()).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    url = _make_url(blob)
    print(f"[HAT Service] Uploaded to GCS: {blob_name}")
    return url


def upload_file(local_path: str, filename: str, content_type: str) -> str:
    """Upload a local file to GCS and return a URL to the object."""
    blob_name = f"upscaled/hat_model/{uuid.uuid4()}/{filename}"
    blob = _get_client().bucket(_bucket_name()).blob(blob_name)
    blob.upload_from_filename(local_path, content_type=content_type)
    url = _make_url(blob)
    print(f"[HAT Service] Uploaded to GCS: {blob_name}")
    return url
