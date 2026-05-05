"""
HAT GPU Microservice
FastAPI app that receives an image URL, upscales it with HAT, uploads the
result to GCS / Firebase Storage, and returns a URL. Designed to run on GCP
Cloud Run with an NVIDIA L4 GPU.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import gcs
from hat_runner import get_model, upscale_image


def _log_gpu_info() -> None:
    """Log GPU details at startup for debugging architecture mismatches."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            mem = props.total_memory / (1024 ** 3)
            print(
                f"[HAT Service] GPU: {name}, SM {cap[0]}.{cap[1]}, "
                f"{mem:.1f} GB VRAM",
                flush=True,
            )
            print(f"[HAT Service] CUDA version: {torch.version.cuda}", flush=True)
            print(f"[HAT Service] PyTorch version: {torch.__version__}", flush=True)
        else:
            print("[HAT Service] WARNING: No CUDA GPU detected!", flush=True)
    except Exception as e:
        print(f"[HAT Service] GPU info query failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the default model so the first request isn't slow."""
    _log_gpu_info()
    print("[HAT Service] Pre-loading default model at startup...", flush=True)
    try:
        get_model("real_gan")
    except Exception as e:
        print(f"[HAT Service] Pre-load failed (will retry per request): {e}", flush=True)
    print("[HAT Service] Lifespan complete. Web server ready.", flush=True)
    yield


app = FastAPI(
    title="HAT Upscale Service",
    version="1.0.0",
    lifespan=lifespan,
)


class UpscaleRequest(BaseModel):
    image_url: str
    model: Literal["real_gan", "real_gan_sharper", "classical"] = "real_gan"
    scale: float = 4.0
    ext: Literal["auto", "jpg", "png"] = "auto"
    tile_size: int = Field(default=512, ge=16)
    tile_pad: int = Field(default=32, ge=0)


class UpscaleResponse(BaseModel):
    status: str
    output_url: str | None = None
    width: int | None = None
    height: int | None = None
    processing_time_ms: int | None = None
    error: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "hat"}


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(body: UpscaleRequest):
    """
    Download an image from a URL, upscale it with HAT, upload the result
    to GCS / Firebase Storage, and return a URL.
    """
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(body.image_url)
            resp.raise_for_status()
            image_bytes = resp.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    try:
        upscaled_bytes, mime_type = await asyncio.to_thread(
            upscale_image,
            image_bytes,
            body.model,
            body.scale,
            body.tile_size,
            body.tile_pad,
            body.ext,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Upscale failed: {e}")
    except Exception as e:
        print(f"[HAT Service] Upscale failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Upscale failed: {e}")

    np_arr = np.frombuffer(upscaled_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    out_ext = "png" if mime_type == "image/png" else "jpg"
    try:
        output_url = await asyncio.to_thread(
            gcs.upload_bytes, upscaled_bytes, f"upscaled.{out_ext}", mime_type
        )
    except Exception as e:
        print(f"[HAT Service] GCS upload failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="GCS upload failed")

    elapsed_ms = int((time.time() - start) * 1000)
    print(f"[HAT Service] Done in {elapsed_ms}ms — {output_url}", flush=True)

    return UpscaleResponse(
        status="ok",
        output_url=output_url,
        width=w,
        height=h,
        processing_time_ms=elapsed_ms,
    )
