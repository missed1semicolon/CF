
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/ChangeFormer

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 \
    https://github.com/wgcban/ChangeFormer.git \
    /app/ChangeFormer

WORKDIR /app/ChangeFormer

RUN pip install --no-cache-dir \
    runpod \
    einops==0.8.1 \
    timm==0.9.16 \
    pillow \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    opencv-python-headless

RUN mkdir -p /app/checkpoints && \
    wget -O /tmp/changeformer_levir.zip \
    https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip && \
    unzip -q /tmp/changeformer_levir.zip -d /app/checkpoints && \
    find /app/checkpoints -name "best_ckpt.pt" -exec cp {} /app/checkpoints/best_ckpt.pt \; && \
    test -f /app/checkpoints/best_ckpt.pt && \
    rm /tmp/changeformer_levir.zip

COPY handler.py /app/handler.py

ENV CHECKPOINT_PATH=/app/checkpoints/best_ckpt.pt
ENV IMAGE_SIZE=256

CMD ["python", "-u", "/app/handler.py"]