import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from ai_module.app import handler_factory
from ai_module.providers import ProviderError, chat, validate_action, validate_provider
from ai_module.store import Store, access_token

TOKEN = "secure-test-token-with-32-chars-or-more"


class ProviderValidationTests(unittest.TestCase):
    def test_openai_must_be_https_and_official_host(self):
        for url in ["http://api.openai.com/v1", "https://evil.test/v1", "https://api.openai.com/v1#fragment"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_provider({"kind": "openai", "name": "OpenAI", "model": "gpt-test", "base_url": url})

    def test_local_providers_are_extensible(self):
        for kind in ["ollama", "openai_compatible"]:
            result = validate_provider({"kind": kind, "name": "Local", "model": "test", "base_url": "http://192.168.1.20:11434"})
            self.assertEqual(result["kind"], kind)

    def test_invalid_module_action_rejected(self):
        for action, target in [("shell.run", "ai.agent"), ("modules.restart", "../../root"), ("modules.remove", "ai.agent")]:
            with self.assertRaises(ValueError):
                validate_action(action, target, "test")

    @patch("ai_module.providers._post_json")
    def test_tool_call_only_yields_proposal(self, send):
        send.return_value = {"choices": [{"message": {"content": "", "tool_calls": [{"function": {
            "name": "request_module_action",
            "arguments": json.dumps({"action": "modules.restart", "target": "dvr.module", "reason": "Owner asked"})
        }}]}}]}
        answer, proposals = chat({"kind": "openai_compatible", "model": "test", "base_url": "http://127.0.0.1:8000/v1", "api_key": ""}, [{"role": "user", "content": "restart DVR"}])
        self.assertEqual(answer, "")
        self.assertEqual(proposals[0]["target"], "dvr.module")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(send.call_count, 1)

    @patch("ai_module.providers._post_json")
    def test_local_qwen3_uses_cpu_friendly_generation_and_extended_timeout(self, send):
        send.return_value = {"message": {"content": "Привет!"}}
        answer, proposals = chat({"kind": "ollama", "model": "qwen3:8b",
                                  "base_url": "http://127.0.0.1:11434", "api_key": ""},
                                 [{"role": "user", "content": "Привет"}])
        self.assertEqual(answer, "Привет!")
        self.assertEqual(proposals, [])
        payload = send.call_args.args[1]
        self.assertEqual(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 512)
        self.assertEqual(send.call_args.args[3], 195)
        self.assertTrue(send.call_args.kwargs["direct"])

    @patch("ai_module.providers._post_json")
    def test_cloud_provider_keeps_short_timeout_and_standard_parameters(self, send):
        send.return_value = {"choices": [{"message": {"content": "Hello"}}]}
        chat({"kind": "openai", "model": "gpt-test", "base_url": "https://api.openai.com/v1",
              "api_key": "secret"}, [{"role": "user", "content": "Hello"}])
        payload = send.call_args.args[1]
        self.assertNotIn("think", payload)
        self.assertEqual(send.call_args.args[3], 45)
        self.assertFalse(send.call_args.kwargs["direct"])

    @patch("ai_module.providers._post_json")
    def test_http400_retries_without_tools(self, send):
        send.side_effect = [ProviderError("Provider returned HTTP 400"), {"choices": [{"message": {"content": "Ready"}}]}]
        answer, proposals = chat({"kind": "openai_compatible", "model": "test", "base_url": "http://127.0.0.1:8000/v1", "api_key": ""}, [{"role": "user", "content": "hello"}])
        self.assertEqual(answer, "Ready")
        self.assertEqual(proposals, [])
        self.assertNotIn("tools", send.call_args[0][1])


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(self.store, TOKEN))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, method="GET", data=None, token=TOKEN):
        headers = {"Authorization": "Bearer " + token}
        if data is not None:
            headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.base + path, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_health_public_but_api_requires_bearer(self):
        self.assertEqual(self.request("/health", token="wrong")[0], 200)
        self.assertEqual(self.request("/api/status", token="wrong")[0], 401)
        self.assertEqual(self.request("/api/status")[1]["core_actions_enabled"], False)

    def test_secrets_never_returned_from_provider_list(self):
        status, provider = self.request("/api/providers", "POST", {"kind": "openai", "name": "Cloud", "model": "gpt-test", "base_url": "https://api.openai.com/v1", "api_key": "SECRET_KEY"})
        self.assertEqual(status, 201)
        _, result = self.request("/api/providers")
        self.assertEqual(result["providers"][0]["has_api_key"], 1)
        self.assertNotIn("SECRET_KEY", json.dumps(result))
        self.assertEqual(self.store.provider(provider["id"])["api_key"], "SECRET_KEY")

    @patch("ai_module.app.chat")
    def test_chat_with_model_and_approval_no_execution(self, model_chat):
        model_chat.return_value = ("", [{"action": "modules.restart", "target": "ai.agent", "reason": "Routine"}])
        _, provider = self.request("/api/providers", "POST", {"kind": "ollama", "name": "Home", "model": "qwen", "base_url": "http://127.0.0.1:11434"})
        status, result = self.request("/api/chat", "POST", {"provider_id": provider["id"], "message": "Restart AI module"})
        self.assertEqual(status, 200)
        self.assertEqual(len(result["approval_ids"]), 1)
        self.assertIn("не выполнены", result["reply"])
        approval = result["approval_ids"][0]
        _, decision = self.request("/api/approvals/" + approval + "/decision", "POST", {"decision": "approve"})
        self.assertEqual(decision, {"status": "approved_not_executed", "executed": False})
        self.assertEqual(self.request("/api/approvals/" + approval + "/decision", "POST", {"decision": "approve"})[0], 409)
        _, history = self.request("/api/conversations/" + result["conversation_id"] + "/messages")
        self.assertEqual([m["role"] for m in history["messages"]], ["user", "assistant"])

    def test_invalid_api_json_and_action_are_rejected(self):
        self.assertEqual(self.request("/api/proposals", "POST", {"action": "shell.run", "target": "ai.agent", "reason": "hi"})[0], 400)
        self.assertEqual(self.request("/api/providers", "POST", {"kind": "openai", "name": "A", "model": "m", "base_url": "http://foo/v1"})[0], 400)


class TokenTests(unittest.TestCase):
    def test_bootstrap_token_is_stable_and_private(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            first = access_token(folder)
            self.assertEqual(first, access_token(folder))
            self.assertGreaterEqual(len(first), 24)
            self.assertEqual(os.stat(os.path.join(folder, "access-token")).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
