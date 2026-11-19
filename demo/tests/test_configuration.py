import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        model = self.directory / "models/model"
        (model / "assets/token2wav").mkdir(parents=True)
        for name in (
            "config.json",
            "modeling_minicpmo.py",
            "weights.safetensors",
            "system_ref_audio.wav",
        ):
            (model / name).touch()
        for name in launch.ASSETS:
            (model / "assets/token2wav" / name).touch()
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"tts.weight": "weights.safetensors"}})
        )
        self.path = self.directory / "config.local.json"
        self.path.write_text(json.dumps({"checkpoint": "models/model"}))

    def arguments(self, *extra):
        return launch.parse_args(["--config", str(self.path), *extra])

    def test_relative_paths_and_current_interaction_defaults(self):
        cfg = launch.load_config(self.arguments())
        launch.validate_model_files(cfg)
        self.assertEqual(
            Path(cfg["frontend_model"]), (self.directory / "models/model").resolve()
        )
        self.assertFalse(cfg["enable_system_barge_in"])
        self.assertTrue(cfg["enable_scheduled_reminders"])
        self.assertTrue(cfg["enable_input_noise_filter"])

    def test_credentials_never_enter_runtime_configuration(self):
        with patch.dict(os.environ, {"SAGE_BACKEND_API_KEY": "unit-test-only"}):
            cfg = launch.load_config(self.arguments())
        self.assertNotIn("unit-test-only", json.dumps(cfg))
        self.path.write_text(
            json.dumps({"checkpoint": "models/model", "api_key": "unit-test-only"})
        )
        with self.assertRaises(ValueError):
            launch.load_config(self.arguments())

    def test_reject_duplicate_gpus_overlapping_ports_and_empty_worker_pool(self):
        for extra in (("--gpu", "0,0"), ("--port", "19310"), ("--instances", "0")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launch.load_config(self.arguments(*extra))

    def test_reject_credentials_in_url(self):
        for url in (
            "https://user:password@example.invalid/v1",
            "https://example.invalid/v1?key=value",
        ):
            with (
                self.subTest(url=url),
                patch.dict(os.environ, {"SAGE_BACKEND_BASE_URL": url}),
                self.assertRaises(ValueError),
            ):
                launch.load_config(self.arguments())

    def test_missing_speech_asset_fails_before_startup(self):
        (self.directory / "models/model/assets/token2wav/flow.pt").unlink()
        with self.assertRaisesRegex(ValueError, "flow.pt"):
            launch.validate_model_files(launch.load_config(self.arguments()))

    def test_stop_only_owned_processes(self):
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        owned = subprocess.Popen(command, start_new_session=True)
        unrelated = subprocess.Popen(command, start_new_session=True)
        try:
            launch.stop_children([owned])
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (owned, unrelated):
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
