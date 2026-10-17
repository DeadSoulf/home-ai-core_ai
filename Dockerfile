# Pinned official Ollama multi-architecture OCI image. No install.sh or
# mutable Git branch is executed during the build.
FROM ollama/ollama@sha256:292ee7945dfc3d5840a181f3ab86fedb1e66703e02c8af98b50f4da56b7e278c AS ollama
FROM python:3.12-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libvulkan1 libopenblas0 ca-certificates \
  && rm -rf /var/lib/apt/lists/*
COPY --from=ollama /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  AI_DATA_DIR=/data AI_HOST=0.0.0.0 AI_PORT=8080 \
  AI_LOCAL_OLLAMA_ENABLED=1 OLLAMA_HOST=127.0.0.1:11434 \
  OLLAMA_MODELS=/data/ollama/models HOME=/data \
  OLLAMA_TMPDIR=/tmp \
  CUDA_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GGML_VK_VISIBLE_DEVICES=-1
WORKDIR /app
COPY ai_module/ /app/ai_module/
COPY web/ /app/web/
RUN mkdir -p /data && chown 65532:65532 /data
USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).read()" || exit 1
CMD ["python", "-m", "ai_module.app"]
