"""Bundled model bootstrap and local API safety regressions; no model download."""
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import json
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from ai_module.app import handler_factory, require_compatible_runtime_flavor
from ai_module.local_ollama import (
    INTERNAL_OLLAMA, MODEL_WHITELIST, LocalOllama,
    STARTER_MODEL, available_memory_bytes, model_suggestions,
)
from ai_module.store import Store

TOKEN = "secure-test-token-123456789012345678901234567890"


class RuntimeFlavorTests(unittest.TestCase):
    def test_cpu_rejects_accidental_gpu_assignment(self):
        with patch.dict(os.environ, {"AI_RUNTIME_FLAVOR": "cpu", "CUDA_VISIBLE_DEVICES": "-1",
                                     "ROCR_VISIBLE_DEVICES": "-1", "GGML_VK_VISIBLE_DEVICES": "-1"}):
            require_compatible_runtime_flavor()
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-11111111-1111-1111-1111-111111111111"
            with self.assertRaisesRegex(RuntimeError, "CPU-only"):
                require_compatible_runtime_flavor()

    def test_gpu_variant_permits_explicit_driver_assignment(self):
        with patch.dict(os.environ, {"AI_RUNTIME_FLAVOR": "gpu",
                                     "CUDA_VISIBLE_DEVICES": "GPU-11111111-1111-1111-1111-111111111111"}):
            require_compatible_runtime_flavor()


class OllamaBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.engine = LocalOllama(self.store, enabled=True, models_dir=self.temp.name)

    def tearDown(self):
        self.engine.stop()
        self.temp.cleanup()

    def test_curated_small_cpu_starter_and_model_recommendations(self):
        self.assertEqual(STARTER_MODEL, "qwen3:0.6b")
        self.assertTrue(all(x["name"] in MODEL_WHITELIST for x in model_suggestions(16 * 1024**3)))
        self.assertEqual([x["name"] for x in model_suggestions(2 * 1024**3)
                          if x["recommended"]], [STARTER_MODEL])
        self.assertIsInstance(available_memory_bytes(), int)

    def test_builtin_provider_is_idempotent_and_preserves_cloud_credentials(self):
        cloud_id = self.store.add_provider({
            "name": "Cloud", "kind": "openai", "base_url": "https://api.openai.com/v1",
            "model": "test", "api_key": "KEEP_THIS_SECRET",
        })
        local_id = self.store.ensure_local_provider("Встроенный Ollama", INTERNAL_OLLAMA, STARTER_MODEL)
        self.assertEqual(local_id, self.store.ensure_local_provider("Встроенный Ollama", INTERNAL_OLLAMA, STARTER_MODEL))
        self.assertEqual(len(self.store.providers()), 2)
        self.assertEqual(self.store.provider(cloud_id)["api_key"], "KEEP_THIS_SECRET")

    def test_bootstrap_registers_previously_pulled_model_without_network(self):
        with patch.object(self.engine, "_model_names", return_value=[STARTER_MODEL]):
            self.engine._bootstrap()
        self.assertEqual(self.engine.status()["model_state"], "ready")
        self.assertEqual(self.engine.status()["active_model"], STARTER_MODEL)
        self.assertEqual(self.store.providers()[0]["model"], STARTER_MODEL)

    def test_installed_models_outside_presets_appear_in_chat_without_duplicate(self):
        cloud_id = self.store.add_provider({
            "name": "Cloud", "kind": "openai",
            "base_url": "https://api.openai.com/v1", "model": "gpt-test",
            "api_key": "KEEP_ME",
        })
        self.engine.engine_state = "ready"
        names = ["qwen3:0.6b", "qwen3:8b", "deepseek-r1:7b"]
        with patch.object(self.engine, "installed", return_value=names):
            self.assertEqual(self.engine.sync_installed_models(), names)
            self.assertEqual(self.engine.sync_installed_models(), names)
        local = [p for p in self.store.providers()
                 if p["kind"] == "ollama"]
        self.assertEqual({p["model"] for p in local}, set(names))
        self.assertEqual(len(local), 3)
        self.assertEqual(self.store.provider(cloud_id)["api_key"], "KEEP_ME")

    def test_unsafe_installed_name_is_not_registered(self):
        self.engine.engine_state = "ready"
        with patch.object(self.engine, "installed",
                          return_value=["qwen3:4b", "../../invalid", "a" * 150]):
            self.engine.sync_installed_models()
        self.assertEqual([p["model"] for p in self.store.providers()], ["qwen3:4b"])

    def test_invalid_or_outsize_model_pull_rejected(self):
        self.engine.engine_state = "ready"
        for name in ["../../etc/passwd", "qwen3:999b", "qwen3", "x" * 140]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.engine.pull(name)
        self.assertEqual(self.engine.status()["model_state"], "waiting")

    def test_supervised_ollama_inherits_explicit_core_gpu_allocation(self):
        # The container defaults to CPU, but the Core helper can explicitly
        # assign verified NVIDIA UUIDs without rebuilding/re-downloading models.
        cuda = "GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": cuda}):
            with patch("ai_module.local_ollama.subprocess.Popen") as launch:
                with patch("ai_module.local_ollama.threading.Thread"):
                    self.engine.start()
                self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], cuda)
                self.assertEqual(launch.call_args.kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11434")
                launch.return_value.poll.return_value = 1

    def test_failed_engine_spawn_does_not_kill_agent(self):
        engine = LocalOllama(self.store, executable="/missing/binary", models_dir=self.temp.name)
        engine.start()
        self.assertEqual(engine.status()["engine_state"], "error")
        engine.stop()


class LocalAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.engine = LocalOllama(self.store, models_dir=self.temp.name)
        self.engine.engine_state = "ready"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(self.store, TOKEN, self.engine))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        self.engine.stop()
        self.temp.cleanup()

    def fetch(self, path, *, method="GET", payload=None, token=TOKEN):
        headers = {"Authorization": "Bearer " + token}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, method=method,
                                     headers=headers, data=json.dumps(payload).encode() if payload is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_local_status_requires_token_and_never_exposes_process_path(self):
        self.assertEqual(self.fetch("/api/local/status", token="bad")[0], 401)
        status, result = self.fetch("/api/local/status")
        self.assertEqual(status, 200)
        self.assertTrue(result["cpu_first"])
        self.assertNotIn("executable", result)
        self.assertNotIn("models_dir", result)

    def test_installed_and_recommended_lists_are_distinct(self):
        with patch.object(self.engine, "installed", return_value=["qwen3:0.6b"]):
            code, result = self.fetch("/api/local/models")
        self.assertEqual(code, 200)
        self.assertEqual(result["installed"], ["qwen3:0.6b"])
        self.assertGreater(len(result["recommended_models"]), 1)

    def test_status_and_provider_selector_use_actual_installed_models(self):
        self.engine.engine_state = "ready"
        names = ["qwen3:0.6b", "qwen3:8b", "deepseek-r1:7b"]
        with patch.object(self.engine, "installed", return_value=names):
            _, status = self.fetch("/api/status")
            self.assertEqual(status["local_models_count"], 3)
            self.assertEqual(status["providers"], 3)
            _, models = self.fetch("/api/local/models")
            self.assertEqual(models["installed"], names)
            _, providers = self.fetch("/api/providers")
        self.assertEqual({p["model"] for p in providers["providers"]}, set(names))
        self.assertTrue(all("api_key" not in p for p in providers["providers"]))

    def test_offline_ollama_does_not_claim_models_deleted(self):
        self.engine.engine_state = "starting"
        _, status = self.fetch("/api/status")
        self.assertIsNone(status["local_models_count"])

    def test_pull_only_curated_model_with_authorization(self):
        with patch.object(self.engine, "pull", return_value=True) as start:
            self.assertEqual(self.fetch("/api/local/pull", method="POST", payload={"name": STARTER_MODEL}, token="bad")[0], 401)
            self.assertEqual(self.fetch("/api/local/pull", method="POST", payload={"name": STARTER_MODEL})[0], 202)
        start.assert_called_once_with(STARTER_MODEL)
        self.assertEqual(self.fetch("/api/local/pull", method="POST", payload={"name": "qwen3:70b"})[0], 400)


if __name__ == "__main__":
    unittest.main()
