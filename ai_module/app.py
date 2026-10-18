"""Minimal authenticated HTTP API and browser UI for a standalone AI module."""
from __future__ import annotations

import hmac
import json
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .providers import ProviderError, chat, validate_action, validate_provider
from .store import Store, access_token
from .local_ollama import LocalOllama

SYSTEM_PROMPT = (
    "You are the HomeAICore assistant. Your tools can ONLY prepare a module-action "
    "proposal for human review, not execute it. Never claim any Core operation "
    "was actually performed. Do not request credentials or invent system status. "
    "For dangerous operations, use request_module_action when available."
)
ID = re.compile(r"^[a-f0-9]{32}$")
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def handler_factory(store: Store, token: str, local: LocalOllama | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HomeAICoreAI/" + __version__

        def log_message(self, _format, *_args):
            # Avoid logging prompts, tokens, query strings and other user content.
            return

        def _respond(self, status: int, data: dict | list):
            encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _common_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")

        def _authorized(self):
            header = self.headers.get("Authorization", "")
            candidate = header[7:] if header.startswith("Bearer ") else ""
            if not hmac.compare_digest(candidate.encode("utf-8"), token.encode("utf-8")):
                self._respond(HTTPStatus.UNAUTHORIZED, {"error": "valid bearer token required"})
                return False
            return True

        def _payload(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("JSON body must be 1-65536 bytes")
                obj = json.loads(self.rfile.read(length))
                if not isinstance(obj, dict):
                    raise ValueError("Expected a JSON object")
                return obj
            except (ValueError, json.JSONDecodeError, UnicodeError):
                raise ValueError("Invalid JSON body or size")

        def _static(self, name: str, mime: str):
            content = (WEB_ROOT / name).read_bytes()
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                return self._respond(200, {"status": "ok", "service": "ai.agent", "version": __version__})
            if path in {"/", "/index.html"}:
                return self._static("index.html", "text/html; charset=utf-8")
            if path == "/app.js":
                return self._static("app.js", "text/javascript; charset=utf-8")
            if path == "/style.css":
                return self._static("style.css", "text/css; charset=utf-8")
            if not path.startswith("/api/"):
                return self._respond(404, {"error": "not found"})
            if not self._authorized():
                return
            if path == "/api/status":
                return self._respond(200, {"version": __version__, "providers": len(store.providers()), "approvals_pending": sum(a["status"] == "pending" for a in store.approvals()), "core_actions_enabled": False})
            if path == "/api/local/status":
                return self._respond(200, local.status() if local is not None else {"enabled": False, "engine_state": "disabled"})
            if path == "/api/local/models":
                if local is None or not local.enabled:
                    return self._respond(200, {"installed": [], "recommended_models": [], "enabled": False})
                return self._respond(200, {
                    "installed": local.installed(),
                    "recommended_models": local.status()["recommended_models"],
                    "enabled": True,
                })
            if path == "/api/providers":
                return self._respond(200, {"providers": store.providers()})
            if path == "/api/conversations":
                return self._respond(200, {"conversations": store.conversations()})
            if path == "/api/approvals":
                return self._respond(200, {"approvals": store.approvals()})
            parts = path.split("/")
            if len(parts) == 5 and parts[:3] == ["", "api", "conversations"] and parts[4] == "messages" and ID.fullmatch(parts[3]):
                if not store.conversation_exists(parts[3]):
                    return self._respond(404, {"error": "conversation not found"})
                return self._respond(200, {"messages": store.messages(parts[3], 100)})
            return self._respond(404, {"error": "not found"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if not path.startswith("/api/"):
                return self._respond(404, {"error": "not found"})
            if not self._authorized():
                return
            try:
                data = self._payload()
                if path == "/api/local/pull":
                    if local is None or not local.enabled:
                        return self._respond(503, {"error": "Local Ollama is not enabled"})
                    name = data.get("name")
                    if not isinstance(name, str):
                        raise ValueError("Model name required")
                    if not local.pull(name):
                        return self._respond(409, {"error": "Model download already in progress"})
                    return self._respond(202, {"queued": True, "name": name})
                if path == "/api/providers":
                    result = validate_provider(data)
                    identifier = store.add_provider(result)
                    return self._respond(201, {"id": identifier})
                if path == "/api/chat":
                    provider_id = data.get("provider_id")
                    message = data.get("message")
                    conv_id = data.get("conversation_id")
                    if not isinstance(provider_id, str) or not isinstance(message, str) or not 1 <= len(message.strip()) <= 6000:
                        raise ValueError("A provider and a message of 1-6000 characters are required")
                    provider = store.provider(provider_id)
                    if not provider:
                        return self._respond(404, {"error": "provider not found"})
                    if conv_id is not None and (not isinstance(conv_id, str) or not ID.fullmatch(conv_id) or not store.conversation_exists(conv_id)):
                        return self._respond(404, {"error": "conversation not found"})
                    if conv_id is None:
                        conv_id = store.new_conversation(message.strip()[:100])
                    store.add_message(conv_id, "user", message.strip())
                    try:
                        reply, proposals = chat(provider, [{"role": "system", "content": SYSTEM_PROMPT}] + store.messages(conv_id, 30))
                    except ProviderError as exc:
                        return self._respond(502, {"error": str(exc), "conversation_id": conv_id})
                    approvals = [store.add_approval(proposal) for proposal in proposals]
                    if approvals:
                        reply += "\n\nПредложение ожидает подтверждения. Никакие действия в Core не выполнены."
                    store.add_message(conv_id, "assistant", reply)
                    return self._respond(200, {"conversation_id": conv_id, "reply": reply, "approval_ids": approvals})
                if path == "/api/proposals":
                    proposal = validate_action(data.get("action", ""), data.get("target", ""), data.get("reason", ""))
                    return self._respond(201, {"id": store.add_approval(proposal), "executed": False})
                parts = path.split("/")
                if len(parts) == 5 and parts[:3] == ["", "api", "approvals"] and parts[4] == "decision" and ID.fullmatch(parts[3]):
                    decision = data.get("decision")
                    if decision not in {"approve", "reject"}:
                        raise ValueError("decision must be approve or reject")
                    if not store.decide(parts[3], decision):
                        return self._respond(409, {"error": "approval missing or already decided"})
                    return self._respond(200, {"status": "approved_not_executed" if decision == "approve" else "rejected", "executed": False})
                return self._respond(404, {"error": "not found"})
            except ValueError as exc:
                return self._respond(400, {"error": str(exc)})

        def do_DELETE(self):
            path = urlsplit(self.path).path
            if not self._authorized():
                return
            parts = path.split("/")
            if len(parts) == 4 and parts[:3] == ["", "api", "providers"] and ID.fullmatch(parts[3]):
                if store.delete_provider(parts[3]):
                    return self._respond(200, {"deleted": True})
            return self._respond(404, {"error": "not found"})

    return Handler


def require_compatible_runtime_flavor() -> None:
    """Fail closed if Core tries to attach a GPU to the CPU-only image.

    The legacy Core GPU reconfiguration reused the existing OCI image. That
    cannot provide acceleration once GPU runners are intentionally excluded.
    Failing startup makes the staged helper roll back instead of claiming GPU
    acceleration even though model inference would still run on CPU.
    """
    if os.environ.get("AI_RUNTIME_FLAVOR") != "cpu":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES", "-1") not in ("", "-1", "none", "void"):
        raise RuntimeError("CPU-only AI image cannot accept GPU devices; update Core to use the optional GPU image")
    if os.environ.get("ROCR_VISIBLE_DEVICES", "-1") not in ("", "-1"):
        raise RuntimeError("CPU-only AI image cannot accept ROCm devices")
    if os.environ.get("GGML_VK_VISIBLE_DEVICES", "-1") not in ("", "-1"):
        raise RuntimeError("CPU-only AI image cannot accept Vulkan devices")


def serve(host: str | None = None, port: int | None = None, data_dir: str | None = None):
    require_compatible_runtime_flavor()
    host = host or os.environ.get("AI_HOST", "0.0.0.0")
    port = int(port if port is not None else os.environ.get("AI_PORT", "8080"))
    data_dir = data_dir or os.environ.get("AI_DATA_DIR", "/data")
    os.umask(0o077)  # Restrict SQLite journal files and generated credentials.
    store = Store(data_dir)
    token = access_token(data_dir)
    local_enabled = os.environ.get("AI_LOCAL_OLLAMA_ENABLED", "0") == "1"
    local = LocalOllama(store, enabled=local_enabled, models_dir=os.path.join(data_dir, "ollama", "models"))
    server = ThreadingHTTPServer((host, port), handler_factory(store, token, local))
    if local_enabled:
        local.start()
    print(f"AI module {__version__} listening on {host}:{port}; local Ollama: {local_enabled}; Core actions disabled", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        local.stop()
        server.server_close()


if __name__ == "__main__":
    serve()
