"""
HAT super-resolution runner.

Wraps the `hat.archs.hat_arch.HAT` PyTorch module with:
  - a per-name weight cache so repeated requests stay warm on the GPU
  - input padding to a multiple of `window_size` (HAT requires this)
  - a tile loop adapted from BasicSR / Real-ESRGAN, since a full-image
    forward will OOM on a 24 GB L4 above ~720p
  - convenience encode/decode helpers that round-trip through OpenCV bytes
"""
from __future__ import annotations

import math
import os
import threading
from contextlib import nullcontext
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hat.archs.hat_arch import HAT


# ── Model registry ────────────────────────────────────────────────────────────
#
# All three baked-in models share the same architecture (only the weights
# differ), so we use a single set of constructor kwargs that mirrors the
# `network_g` block from `options/test/HAT_GAN_Real_SRx4.yml` and
# `options/test/HAT_SRx4_ImageNet-LR.yml`.
_HAT_KWARGS = dict(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)

ModelName = Literal["real_gan", "real_gan_sharper", "classical"]

_WEIGHT_FILES: dict[str, str] = {
    "real_gan": "Real_HAT_GAN_SRx4.pth",
    "real_gan_sharper": "Real_HAT_GAN_sharper.pth",
    "classical": "HAT_SRx4_ImageNet-pretrain.pth",
}

_NATIVE_SCALE = 4
_WINDOW_SIZE = _HAT_KWARGS["window_size"]

_models: dict[str, HAT] = {}
_load_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# HAT uses stacked LayerNorm + attention; fp16 autocast often overflows or
# yields NaNs in those blocks. NaNs survive clamp_(0,1) and become 0 when cast
# to uint8 → an all-black output image. So we run inference in full fp32.
#
# Optional: set HAT_INFER_DTYPE=bf16 on Ada/L4+ GPUs for a bit more headroom
# (bfloat16 has the same exponent range as fp32 and is usually stable here).
_infer_dtype = os.environ.get("HAT_INFER_DTYPE", "fp32").strip().lower()


def _bf16_supported() -> bool:
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(fn and fn())


def _amp_context():
    if not torch.cuda.is_available():
        return nullcontext()
    if _infer_dtype == "bf16" and _bf16_supported():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if _infer_dtype in ("fp16", "half"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _sanitize_output(t: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf (broken attention / overflow) before uint8 conversion."""
    if torch.isfinite(t).all():
        return t
    bad = (~torch.isfinite(t)).sum().item()
    print(f"[HAT Service] WARNING: {bad} non-finite pixels in model output; patching.", flush=True)
    return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)


def _weights_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "weights")


def get_model(name: ModelName = "real_gan") -> HAT:
    """Return a cached HAT module on GPU, loading it on first request."""
    if name not in _WEIGHT_FILES:
        raise ValueError(
            f"Unknown model '{name}'. Choices: {list(_WEIGHT_FILES.keys())}"
        )

    if name in _models:
        return _models[name]

    with _load_lock:
        if name in _models:
            return _models[name]

        weight_path = os.path.join(_weights_dir(), _WEIGHT_FILES[name])
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Weight file missing: {weight_path}. "
                "It should have been baked into the Docker image."
            )

        print(f"[HAT Service] Loading model '{name}' from {weight_path}...", flush=True)

        model = HAT(**_HAT_KWARGS)
        state = torch.load(weight_path, map_location="cpu")
        # All three configs in the upstream repo use param_key_g: 'params_ema'.
        # Some community-mirrored checkpoints store the state dict at the top
        # level; fall back to that if the key isn't present.
        if isinstance(state, dict) and "params_ema" in state:
            state = state["params_ema"]
        elif isinstance(state, dict) and "params" in state:
            state = state["params"]
        model.load_state_dict(state, strict=True)

        model.eval()
        model = model.to(_device)
        torch.backends.cudnn.benchmark = True

        _models[name] = model
        if _device.type == "cuda":
            mode = f"cuda infer={_infer_dtype}"
        else:
            mode = "cpu infer=fp32"
        print(f"[HAT Service] Model '{name}' ready on {mode}.", flush=True)
        return model


# ── Inference ─────────────────────────────────────────────────────────────────


def _pad_to_window(x: torch.Tensor, window: int) -> tuple[torch.Tensor, int, int]:
    """Reflect-pad a [N, C, H, W] tensor so H and W are multiples of `window`."""
    _, _, h, w = x.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


@torch.no_grad()
def _forward_full(model: HAT, x: torch.Tensor) -> torch.Tensor:
    with _amp_context():
        out = model(x)
    return _sanitize_output(out.float())


@torch.no_grad()
def _forward_tiled(
    model: HAT,
    img: torch.Tensor,
    tile_size: int,
    tile_pad: int,
) -> torch.Tensor:
    """Run `model` on `img` in tiles to bound peak VRAM.

    Adapted from BasicSR / Real-ESRGAN's RealESRGANer.tile_process: split into
    overlapping tiles, run each through the model, then stitch the unpadded
    centres back together at the upscaled resolution.
    """
    batch, _, height, width = img.shape
    out_h = height * _NATIVE_SCALE
    out_w = width * _NATIVE_SCALE

    output = img.new_zeros((batch, 3, out_h, out_w))

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            ofs_x = tx * tile_size
            ofs_y = ty * tile_size

            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            tile_w = input_end_x - input_start_x
            tile_h = input_end_y - input_start_y

            input_tile = img[
                :,
                :,
                input_start_y_pad:input_end_y_pad,
                input_start_x_pad:input_end_x_pad,
            ]

            input_tile, pad_h, pad_w = _pad_to_window(input_tile, _WINDOW_SIZE)

            try:
                with _amp_context():
                    output_tile = model(input_tile)
                output_tile = _sanitize_output(output_tile.float())
            except RuntimeError as e:
                raise RuntimeError(
                    f"HAT forward failed for tile ({tx},{ty}) "
                    f"size {input_tile.shape[-2:]}: {e}"
                ) from e

            if pad_h or pad_w:
                output_tile = output_tile[
                    :,
                    :,
                    : output_tile.shape[2] - pad_h * _NATIVE_SCALE,
                    : output_tile.shape[3] - pad_w * _NATIVE_SCALE,
                ]

            output_start_x = input_start_x * _NATIVE_SCALE
            output_end_x = input_end_x * _NATIVE_SCALE
            output_start_y = input_start_y * _NATIVE_SCALE
            output_end_y = input_end_y * _NATIVE_SCALE

            crop_start_x = (input_start_x - input_start_x_pad) * _NATIVE_SCALE
            crop_end_x = crop_start_x + tile_w * _NATIVE_SCALE
            crop_start_y = (input_start_y - input_start_y_pad) * _NATIVE_SCALE
            crop_end_y = crop_start_y + tile_h * _NATIVE_SCALE

            output[
                :,
                :,
                output_start_y:output_end_y,
                output_start_x:output_end_x,
            ] = output_tile[
                :,
                :,
                crop_start_y:crop_end_y,
                crop_start_x:crop_end_x,
            ]

    return output


# ── Encode / decode ───────────────────────────────────────────────────────────


def _detect_ext(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "jpg"


def upscale_image(
    image_bytes: bytes,
    model_name: ModelName = "real_gan",
    scale: float = 4.0,
    tile_size: int = 512,
    tile_pad: int = 32,
    ext: str = "auto",
) -> tuple[bytes, str]:
    """
    Upscale an encoded image with HAT.

    Returns (encoded_bytes, mime_type).

    `scale` is the final scale factor. The underlying model is always 4x;
    for `scale != 4` we resize the 4x output with INTER_AREA (downscale) or
    INTER_LANCZOS4 (upscale) so callers get exactly the requested ratio.

    `tile_size` and `tile_pad` are passed through to the tile loop. tile_size
    must be a multiple of HAT's window size (16); we round down silently if not.
    """
    if model_name not in _WEIGHT_FILES:
        raise ValueError(f"Unknown model '{model_name}'")

    resolved_ext = _detect_ext(image_bytes) if ext == "auto" else ext
    if resolved_ext not in {"jpg", "png"}:
        raise ValueError(f"Unsupported output extension '{resolved_ext}'")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")

    in_h, in_w = img_bgr.shape[:2]
    print(
        f"[HAT Service] Input: {in_w}x{in_h}, model={model_name}, "
        f"scale={scale}x, tile={tile_size}, ext={resolved_ext}",
        flush=True,
    )

    model = get_model(model_name)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().div_(255.0)
    # fp32 tensor on device; _forward_* uses fp32 by default (see HAT_INFER_DTYPE).
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(_device)

    # Snap tile_size to a multiple of window_size; clamp so a single tile is
    # never smaller than the window itself.
    eff_tile = max(_WINDOW_SIZE, (tile_size // _WINDOW_SIZE) * _WINDOW_SIZE)

    use_tiles = max(in_h, in_w) > eff_tile
    if use_tiles:
        out_tensor = _forward_tiled(model, tensor, eff_tile, tile_pad)
    else:
        padded, pad_h, pad_w = _pad_to_window(tensor, _WINDOW_SIZE)
        out_tensor = _forward_full(model, padded)
        if pad_h or pad_w:
            out_tensor = out_tensor[
                :,
                :,
                : out_tensor.shape[2] - pad_h * _NATIVE_SCALE,
                : out_tensor.shape[3] - pad_w * _NATIVE_SCALE,
            ]

    out_tensor = _sanitize_output(out_tensor).clamp_(0, 1).squeeze(0).cpu()
    out_rgb = (out_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    target_w = max(1, int(round(in_w * scale)))
    target_h = max(1, int(round(in_h * scale)))
    if (out_bgr.shape[1], out_bgr.shape[0]) != (target_w, target_h):
        interp = cv2.INTER_AREA if scale < _NATIVE_SCALE else cv2.INTER_LANCZOS4
        out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=interp)

    out_h, out_w = out_bgr.shape[:2]
    print(f"[HAT Service] Output: {out_w}x{out_h}", flush=True)

    if resolved_ext == "png":
        ok, encoded = cv2.imencode(".png", out_bgr)
        mime_type = "image/png"
    else:
        ok, encoded = cv2.imencode(".jpg", out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mime_type = "image/jpeg"

    if not ok:
        raise RuntimeError(f"Failed to encode output as {resolved_ext}")

    return encoded.tobytes(), mime_type
