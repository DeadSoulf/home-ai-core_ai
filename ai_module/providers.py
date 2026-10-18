"""Strict, replaceable model-provider adapters. No model executes Core actions."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

KINDS = {"ollama", "openai", "openai_compatible"}
MODULE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
ACTIONS = {"modules.enable", "modules.disable", "modules.restart"}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "request_module_action",
        "description": (
            "Prepare a HomeAICore module operation for human review. "
            "This does NOT execute it; a separate secure Core adapter is needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(ACTIONS)},
                "target": {"type": "string", "description": "Stable module ID"},
                "reason": {"type": "string"},
            },
            "required": ["action", "target", "reason"],
            "additionalProperties": False,
        },
    },
}]

class ProviderError(Exception):
    pass


def validate_provider(data: dict[str, Any]) -> dict[str, Any]:
    kind = data.get("kind")
    name = str(data.get("name", "")).strip()
    model = str(data.get("model", "")).strip()
    url = str(data.get("base_url", "")).strip().rstrip("/")
    key = str(data.get("api_key", ""))
    if not isinstance(kind, str) or kind not in KINDS or not 1 <= len(name) <= 80 or not 1 <= len(model) <= 120:
        raise ValueError("Invalid provider kind, name, or model")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or any(c.isspace() for c in url):
        raise ValueError("Provider URL must be an HTTP(S) address without credentials or fragment")
    if kind == "openai" and parsed.scheme != "https":
        raise ValueError("OpenAI API must use HTTPS")
    if kind == "openai" and not (parsed.hostname == "api.openai.com" and parsed.path.rstrip("/") == "/v1"):
        raise ValueError("OpenAI provider must use https://api.openai.com/v1; use compatible for other endpoints")
    if len(url) > 512 or len(key) > 4096:
        raise ValueError("Provider URL or API key too long")
    return {"kind": kind, "name": name, "model": model, "base_url": url, "api_key": key}


def validate_action(action: str, target: str, reason: str) -> dict[str, str]:
    if not isinstance(action, str) or not isinstance(target, str) or action not in ACTIONS or not MODULE_ID.fullmatch(target) or len(target) > 96:
        raise ValueError("Unsupported module action or target")
    if not isinstance(reason, str):
        raise ValueError("Reason must be text")
    reason = reason.strip()
    if not reason or len(reason) > 1000:
        raise ValueError("Reason is required and must be under 1000 characters")
    return {"action": action, "target": target, "reason": reason}


def _endpoint(provider: dict[str, Any]) -> str:
    if provider["kind"] == "ollama":
        return provider["base_url"] + "/api/chat"
    return provider["base_url"] + "/chat/completions"


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: int, *, direct: bool = False) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        # The embedded loopback Ollama server must not be routed through a
        # HTTP_PROXY inherited from the host or CI environment.
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                  if direct else urllib.request.build_opener())
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ProviderError("Provider response exceeds 1 MiB")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ProviderError("Invalid provider response")
            return value
    except urllib.error.HTTPError as exc:
        # Do not echo provider error bodies: they may contain secrets or prompts.
        raise ProviderError(f"Provider returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError("Provider connection or response failed") from exc


def chat(provider: dict[str, Any], messages: list[dict[str, str]], *, timeout: int | None = None) -> tuple[str, list[dict[str, str]]]:
    is_ollama = provider["kind"] == "ollama"
    # Cold CPU inference can take minutes; external APIs retain a shorter
    # request budget rather than holding the chat indefinitely.
    if timeout is None:
        timeout = 195 if is_ollama else 45
    payload: dict[str, Any] = {"model": provider["model"], "messages": messages, "stream": False, "tools": TOOLS}
    if is_ollama:
        payload["options"] = {"num_predict": 512}
        # Standard chat need not spend minutes in Qwen3 reasoning mode.
        if provider["model"].startswith("qwen3:"):
            payload["think"] = False
    else:
        payload["tool_choice"] = "auto"
    try:
        value = _post_json(_endpoint(provider), payload, provider["api_key"], timeout, direct=is_ollama)
    except ProviderError as exc:
        # Older local models may reject tools. Retry only for HTTP 400.
        if "HTTP 400" not in str(exc):
            raise
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        value = _post_json(_endpoint(provider), payload, provider["api_key"], timeout, direct=is_ollama)
    try:
        message = value["message"] if provider["kind"] == "ollama" else value["choices"][0]["message"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = ""
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
    except (TypeError, KeyError, IndexError) as exc:
        raise ProviderError("Model response has no usable message") from exc
    proposals = []
    for call in tool_calls[:5]:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if fn.get("name") != "request_module_action":
            continue
        arguments = fn.get("arguments", {})
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                continue
            proposals.append(validate_action(arguments.get("action", ""), arguments.get("target", ""), arguments.get("reason", "")))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    if not content and not proposals:
        raise ProviderError("Model returned no answer or valid proposal")
    return content[:20000], proposals
