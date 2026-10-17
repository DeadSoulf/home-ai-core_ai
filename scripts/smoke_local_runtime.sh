#!/usr/bin/env bash
# Real CPU-only runtime smoke test, using Core's restricted container options.
# Requires Docker, curl and Python 3; no API credentials or host devices exposed.
set -Eeuo pipefail

image="$1"
container="home-ai-local-smoke-$RANDOM-$$"
volume="home-ai-local-smoke-data-$RANDOM-$$"
status_file="$(mktemp)"
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker volume rm "$volume" >/dev/null 2>&1 || true
  rm -f "$status_file"
}
trap cleanup EXIT

docker volume create "$volume" >/dev/null
docker run --name "$container" --detach --rm \
  --user 65532:65532 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --publish 127.0.0.1:8091:8080 \
  --mount "type=volume,src=$volume,dst=/data" \
  "$image" >/dev/null

token=""
for iteration in $(seq 1 60); do
  token="$(docker exec "$container" python -c 'from pathlib import Path; print(Path("/data/access-token").read_text().strip())' 2>/dev/null || true)"
  if [ -n "$token" ]; then
    break
  fi
  sleep 1
done
if [ -z "$token" ]; then
  echo "Bootstrap token was not generated; cannot validate restricted AI container"
  docker logs "$container" --tail 35 2>&1 || true
  exit 1
fi

ready=0
for iteration in $(seq 1 90); do
  if curl --silent --show-error --fail --max-time 3 \
    --header "Authorization: Bearer $token" \
    http://127.0.0.1:8091/api/local/status > "$status_file" 2>/dev/null; then
    if python3 - "$status_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    status = json.load(stream)

if (status.get("enabled") is True and status.get("engine_state") == "ready"
        and status.get("model_state") in ("downloading", "ready")):
    print("Local Ollama launched successfully; first-run model is",
          status.get("active_model"), status.get("model_state"))
    sys.exit(0)
sys.exit(1)
PY
    then
      ready=1
      break
    fi
  fi
  sleep 2
done

if [ "$ready" -ne 1 ]; then
  echo "Bundled Ollama failed to start with Core's rootless/read-only runtime settings"
  docker logs "$container" --tail 35 2>&1 || true
  exit 1
fi

curl --silent --show-error --fail --max-time 10 \
  --header "Authorization: Bearer $token" \
  http://127.0.0.1:8091/api/local/models > "$status_file"
python3 - "$status_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    models = json.load(stream)
assert models.get("enabled") is True, models
presets = {entry["name"] for entry in models.get("recommended_models", [])}
assert "qwen3:0.6b" in presets, presets
print("Local preset catalog is available; CPU model bootstrap was started")
PY
