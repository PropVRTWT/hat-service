# HAT Upscale Service

FastAPI microservice that wraps the [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT)
Hybrid Attention Transformer for image super-resolution. Designed to run on GCP
Cloud Run with an NVIDIA L4 GPU.

Mirrors the structure of the sibling `esrgan-service` (image-URL in, GCS/Firebase
URL out) so it can be deployed and called the same way.

## Models baked into the image

| Request key         | Weight file                          | Notes                                  |
|---------------------|--------------------------------------|----------------------------------------|
| `real_gan`          | `Real_HAT_GAN_SRx4.pth`              | Default. Real-world SR, better fidelity. |
| `real_gan_sharper`  | `Real_HAT_GAN_sharper.pth`           | Real-world SR, sharper / more aggressive. |
| `classical`         | `HAT_SRx4_ImageNet-pretrain.pth`     | Classical bicubic SR, best PSNR.       |

All three are 4x super-resolution models. The output is `scale * input` pixels;
fractional `scale` values are achieved by downscaling the 4x output.

## Environment variables

| Variable           | Required | Default | Purpose                                              |
|--------------------|----------|---------|------------------------------------------------------|
| `GCS_BUCKET_NAME`  | yes      | —       | Bucket where upscaled outputs are uploaded.          |
| `GCS_SIGNED_URLS`  | no       | `false` | Set to `true` to return 24h V4 signed URLs instead of Firebase download URLs. |
| `PORT`             | no       | `8080`  | Port uvicorn binds to (Cloud Run sets this).         |

## API

### `GET /health`

```json
{ "status": "ok", "service": "hat" }
```

### `POST /upscale`

Request body:

```json
{
  "image_url": "https://example.com/input.jpg",
  "model": "real_gan",
  "scale": 4,
  "ext": "auto",
  "tile_size": 512,
  "tile_pad": 32
}
```

| Field        | Type    | Default      | Notes                                                                    |
|--------------|---------|--------------|--------------------------------------------------------------------------|
| `image_url`  | string  | (required)   | URL of the source image.                                                 |
| `model`      | string  | `real_gan`   | One of `real_gan`, `real_gan_sharper`, `classical`.                      |
| `scale`      | float   | `4`          | Final scale factor. Underlying model is 4x; other values are resized.    |
| `ext`        | string  | `auto`       | `auto` (sniff from input bytes), `jpg`, or `png`.                        |
| `tile_size`  | int     | `512`        | Tile patch size. Must be a multiple of HAT's window size (16).           |
| `tile_pad`   | int     | `32`         | Overlap between tiles, in pixels.                                        |

Response:

```json
{
  "status": "ok",
  "output_url": "https://firebasestorage.googleapis.com/...",
  "width": 4096,
  "height": 2304,
  "processing_time_ms": 6342
}
```

## Local quick start

```bash
docker build -t hat-service .
docker run --gpus all --rm -p 8080:8080 \
  -e GCS_BUCKET_NAME=your-bucket \
  -v $HOME/.config/gcloud:/root/.config/gcloud \
  hat-service

curl -X POST http://localhost:8080/upscale \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://example.com/photo.jpg"}'
```

## Deploy to Cloud Run

The deploy is driven by [`cloudbuild.yaml`](cloudbuild.yaml), which mirrors the
sibling `esrgan-service` setup (Artifact Registry path
`us-central1-docker.pkg.dev/$PROJECT_ID/propvrtwt/hat-service/$BRANCH_NAME/app`).

```bash
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=hat-service,_REGION_NAME=us-central1,BRANCH_NAME=main
```

Make sure the target Cloud Run service is configured with:
- An NVIDIA L4 GPU (`--gpu 1 --gpu-type nvidia-l4`)
- At least 16Gi memory and 4 CPU
- Concurrency 1 (the model serializes on a single GPU)
- A service account with `roles/storage.objectAdmin` on `$GCS_BUCKET_NAME`
  (and `roles/iam.serviceAccountTokenCreator` on itself if `GCS_SIGNED_URLS=true`)
