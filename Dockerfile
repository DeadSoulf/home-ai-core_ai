# Same pinned upstream Ollama release for CPU and GPU variants. This
# image is only a build source; its >2 GiB GPU layers are NOT copied into
# the default final image.
FROM ollama/ollama@sha256:292ee7945dfc3d5840a181f3ab86fedb1e66703e02c8af98b50f4da56b7e278c AS ollama

# Remove the GPU-specific Ollama runner directories BEFORE the final COPY.
# Removing them in a final-stage RUN would NOT shrink the pushed OCI image.
FROM ollama AS ollama-cpu
RUN set -eu; \
    for path in /usr/lib/ollama/cuda* /usr/lib/ollama/rocm* \
                /usr/lib/ollama/vulkan* /usr/lib/ollama/mlx* \
                /usr/lib/ollama/jetpack* /usr/lib/ollama/hip*; do \
      if [ -e "$path" ]; then rm -rf -- "$path"; fi; \
    done; \
    test -x /usr/bin/ollama; \
    test -d /usr/lib/ollama; \
    find /usr/lib/ollama -maxdepth 2 -type f | head -n 20

FROM python:3.12-slim-trixie AS common
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libvulkan1 libopenblas0 ca-certificates \
  && rm -rf /var/lib/apt/lists/*
COPY ai_module/ /app/ai_module/
COPY web/ /app/web/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  AI_DATA_DIR=/data AI_HOST=0.0.0.0 AI_PORT=8080 \
  AI_LOCAL_OLLAMA_ENABLED=1 OLLAMA_HOST=127.0.0.1:11434 \
  OLLAMA_MODELS=/data/ollama/models HOME=/data \
  OLLAMA_TMPDIR=/tmp \
  CUDA_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GGML_VK_VISIBLE_DEVICES=-1
WORKDIR /app
RUN mkdir -p /data && chown 65532:65532 /data
USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).read()" || exit 1
CMD ["python", "-m", "ai_module.app"]

# Optional NVIDIA acceleration variant. This intentionally contains all Ollama
# runtimes and is downloaded only after an administrator selects a supported GPU.
FROM common AS gpu
ENV AI_RUNTIME_FLAVOR=gpu
COPY --from=ollama /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama

# Default module image: CPU-only, small download, works without a GPU driver.
FROM common AS cpu
ENV AI_RUNTIME_FLAVOR=cpu
COPY --from=ollama-cpu /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama-cpu /usr/lib/ollama /usr/lib/ollama
