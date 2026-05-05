# HAT GPU Microservice
# Runs on NVIDIA L4 GPU via GCP Cloud Run

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PORT=8080

# Install system dependencies.
# Apt-related flags worth knowing:
#   Acquire::Retries=5             — survive transient mirror flakiness
#   Acquire::http::Timeout=30      — don't hang forever on a dead mirror
#   --fix-missing                  — keep going if a single deb fails to fetch
#   set -e + apt-get update -y     — fail the layer loudly instead of silently
#                                     proceeding against a half-broken index
RUN set -eux; \
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries; \
    echo 'Acquire::http::Timeout "30";' > /etc/apt/apt.conf.d/80-timeout; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.1 bindings first to avoid version conflicts
RUN pip install --no-cache-dir \
        torch==2.3.1 \
        torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Install HAT from GitHub. `python setup.py develop` registers the `hat`
# package and pulls in `basicsr`, `einops`, etc.
RUN git clone --depth 1 https://github.com/XPixelGroup/HAT.git /opt/HAT && \
    cd /opt/HAT && \
    pip install --no-cache-dir basicsr==1.4.2 einops && \
    pip install --no-cache-dir -e .

# Patch the well-known basicsr torchvision-compat bug:
# `from torchvision.transforms.functional_tensor import rgb_to_grayscale` was
# removed upstream and now lives at `torchvision.transforms.functional`.
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3*/dist-packages/basicsr/data/degradations.py || true

# Install the rest of our service-specific dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all baked-in HAT weights so the container boots offline-ready.
# Sources are public HuggingFace mirrors (the official weights only live on
# Google Drive / Baidu, neither of which work in unattended Docker builds).
RUN mkdir -p weights && \
    wget -q --show-progress -O weights/Real_HAT_GAN_SRx4.pth \
        https://huggingface.co/jaideepsingh/upscale_models/resolve/867e7c0ad519aba5d36b5bb4ef4a4a91781914c4/HAT/Real_HAT_GAN_SRx4.pth && \
    wget -q --show-progress -O weights/Real_HAT_GAN_sharper.pth \
        https://huggingface.co/Acly/hat/resolve/main/Real_HAT_GAN_sharper.pth && \
    wget -q --show-progress -O weights/HAT_SRx4_ImageNet-pretrain.pth \
        https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
