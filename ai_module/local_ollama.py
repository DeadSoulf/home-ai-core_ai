"""Managed loopback-only Ollama engine and first-run local model bootstrap.

No host services, Docker socket, shell installer or external API keys are used.
The small starter model is downloaded in the background to persistent /data;
the AI HTTP API remains responsive during the first download.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*(?:/[a-z0-9][a-z0-9_.-]*)?:[a-z0-9][a-z0-9_.-]*$")
STARTER_MODEL = "qwen3:0.6b"
INTERNAL_OLLAMA = "http://127.0.0.1:11434"
# This curated catalog is advisory. The requested model is checked against it
# so that an accidental typo cannot cause an arbitrary large download.
PRESETS = (
    {"name": "qwen3:0.6b", "size_gb": 0.6, "minimum_ram_gb": 2, "purpose": "Базовый агент, CPU"},
    {"name": "qwen3:1.7b", "size_gb": 1.5, "minimum_ram_gb": 4, "purpose": "Улучшенный локальный чат"},
    {"name": "qwen3:4b", "size_gb": 2.8, "minimum_ram_gb": 8, "purpose": "Сложные задачи"},
    {"name": "qwen3:8b", "size_gb": 5.5, "minimum_ram_gb": 16, "purpose": "Большая модель"},
)
MODEL_WHITELIST = frozenset(x["name"] for x in PRESETS)
LOCAL_PROVIDER_NAME = "Встроенный Ollama"


def available_memory_bytes() -> int:
    """Use the stricter of RAM and cgroup limits; this is not host GPU VRAM."""
    values: list[int] = []
    try:
        values.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        value = Path("/sys/fs/cgroup/memory.max").read_text(encoding="ascii").strip()
        if value.isdigit() and int(value) > 0:
            values.append(int(value))
    except OSError:
        pass
    return min(values) if values else 0


def model_suggestions(ram_bytes: int) -> list[dict[str, Any]]:
    gb = ram_bytes / 1024**3
    return [{**item, "recommended": gb >= item["minimum_ram_gb"]}
            for item in PRESETS]


class LocalOllama:
    def __init__(self, store: Any, *,
                 enabled: bool = True, executable: str = "/usr/local/bin/ollama",
                 base_url: str = INTERNAL_OLLAMA, models_dir: str = "/data/ollama/models"):
        self.store = store
        self.enabled = enabled
        self.executable = executable
        self.base_url = base_url
        self.models_dir = models_dir
        self.lock = threading.RLock()
        self.process: subprocess.Popen[bytes] | None = None
        self.engine_state = "starting" if enabled else "disabled"
        self.model_state = "waiting" if enabled else "disabled"
        self.active_model = ""
        self.download_status = ""
        self.download_percent: int | None = None
        self.error = ""
        self.worker: threading.Thread | None = None
        self.stopped = threading.Event()
        self.ram_bytes = available_memory_bytes()

    def _request(self, path: str, payload: dict | None = None, timeout: int = 6) -> Any:
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            headers=headers,
            data=json.dumps(payload).encode() if payload is not None else None,
            method="POST" if payload is not None else "GET",
        )
        # Ignore environment http_proxy/https_proxy even in hardened deployments.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("Ollama response exceeds 1 MiB")
            return json.loads(raw)

    def _model_names(self) -> list[str]:
        result = self._request("/api/tags")
        return sorted(x["name"] for x in result.get("models", [])
                      if isinstance(x, dict) and isinstance(x.get("name"), str))

    def installed(self) -> list[str]:
        try:
            return self._model_names() if self.engine_state == "ready" else []
        except (ValueError, OSError, urllib.error.URLError):
            return []

    def start(self) -> None:
        if not self.enabled:
            return
        models = Path(self.models_dir)
        models.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(models, 0o700)
        env = os.environ.copy()
        env.update({
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_MODELS": str(models),
            "HOME": os.environ.get("AI_DATA_DIR", "/data"),
            "OLLAMA_TMPDIR": "/tmp",
            # CPU-first works on servers without a GPU/toolkit. GPU allocation
            # needs a separately reviewed Core runtime change (no auto-privilege).
            # CPU by default; the privileged Core runtime alone may provide
            # a verified list of NVIDIA UUIDs when a user explicitly approves.
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "-1"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", "-1"),
            "GGML_VK_VISIBLE_DEVICES": os.environ.get("GGML_VK_VISIBLE_DEVICES", "-1"),
        })
        try:
            self.process = subprocess.Popen(
                [self.executable, "serve"], env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError):
            with self.lock:
                self.engine_state = "error"
                self.model_state = "error"
                self.error = "Не удалось запустить встроенный Ollama"
            return
        threading.Thread(target=self._bootstrap, name="ollama-bootstrap", daemon=True).start()

    def _bootstrap(self) -> None:
        for _ in range(90):
            if self.stopped.is_set():
                return
            if self.process and self.process.poll() is not None:
                with self.lock:
                    self.engine_state = "error"
                    self.model_state = "error"
                    self.error = "Процесс Ollama завершился"
                return
            try:
                self._model_names()
                break
            except (OSError, ValueError, urllib.error.URLError):
                self.stopped.wait(1)
        else:
            with self.lock:
                self.engine_state = "error"
                self.model_state = "error"
                self.error = "Сервер Ollama не ответил за 90 секунд"
            return
        with self.lock:
            self.engine_state = "ready"
        names = self.installed()
        for name in names:
            if name in MODEL_WHITELIST:
                self.store.ensure_local_provider(LOCAL_PROVIDER_NAME, INTERNAL_OLLAMA, name)
        if STARTER_MODEL in names:
            with self.lock:
                self.model_state = "ready"
                self.active_model = STARTER_MODEL
            return
        self.pull(STARTER_MODEL)

    def pull(self, name: str) -> bool:
        if name not in MODEL_WHITELIST or not MODEL_NAME.fullmatch(name):
            raise ValueError("Выберите модель из проверенного списка")
        if self.engine_state != "ready":
            raise ValueError("Ollama ещё запускается")
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return False
            self.error = ""
            self.active_model = name
            self.model_state = "downloading"
            self.download_status = "Подготовка загрузки"
            self.download_percent = 0
            self.worker = threading.Thread(target=self._download, args=(name,), daemon=True, name="ollama-model-pull")
            self.worker.start()
        return True

    def _download(self, name: str) -> None:
        request = urllib.request.Request(
            self.base_url + "/api/pull",
            data=json.dumps({"name": name, "stream": True}).encode(),
            headers={"Accept": "application/x-ndjson", "Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 30*60
        success = False
        try:
            # Re-pulling an installed model is a no-op. Disk space must be
            # sufficient to avoid filling a server's root storage unexpectedly.
            available = shutil.disk_usage(Path(self.models_dir)).free
            estimated = next(item["size_gb"] for item in PRESETS if item["name"] == name)
            if available < int((estimated + 1) * 1024**3):
                raise ValueError("Недостаточно свободного места для выбранной модели")
            with opener.open(request, timeout=60) as response:
                while not self.stopped.is_set():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Превышено время загрузки модели")
                    line = response.readline(8193)
                    if not line:
                        break
                    if len(line) > 8192:
                        raise ValueError("Слишком большой ответ от Ollama")
                    data = json.loads(line)
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    percent = (min(100, int(completed * 100 / total))
                               if isinstance(total, (int, float)) and total > 0 and
                               isinstance(completed, (int, float)) else None)
                    with self.lock:
                        self.download_status = str(data.get("status", ""))[:120]
                        if percent is not None:
                            self.download_percent = percent
                    if data.get("error"):
                        raise ValueError("Ollama отклонил загрузку модели")
                    if data.get("status") == "success":
                        success = True
                        break
            if not success or name not in self._model_names():
                raise ValueError("Загрузка модели не завершилась")
            self.store.ensure_local_provider(LOCAL_PROVIDER_NAME, INTERNAL_OLLAMA, name)
            with self.lock:
                self.model_state = "ready"
                self.download_status = "Готово"
                self.download_percent = 100
        except (OSError, ValueError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            with self.lock:
                self.model_state = "error"
                self.download_status = ""
                self.error = "Не удалось загрузить локальную модель. Проверьте интернет и свободное место и повторите."
        finally:
            with self.lock:
                self.worker = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "enabled": self.enabled,
                "engine_state": self.engine_state,
                "model_state": self.model_state,
                "active_model": self.active_model,
                "download_status": self.download_status,
                "download_percent": self.download_percent,
                "error": self.error,
                "cpu_first": True,
                "ram_bytes": self.ram_bytes,
                "recommended_models": model_suggestions(self.ram_bytes),
            }

    def stop(self) -> None:
        self.stopped.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
