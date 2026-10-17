FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AI_DATA_DIR=/data AI_HOST=0.0.0.0 AI_PORT=8080
WORKDIR /app
COPY ai_module/ /app/ai_module/
COPY web/ /app/web/
RUN mkdir -p /data && chown 65532:65532 /data
USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).read()" || exit 1
CMD ["python", "-m", "ai_module.app"]
