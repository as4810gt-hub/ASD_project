import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.transcription_service import (
    TranscriptionError,
    TranscriptionService,
)


class TranscriptionServiceHealthTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        package_dir = self.root / "faster_whisper"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_disabled_service_does_not_import_or_load_whisper(self):
        service = TranscriptionService(
            source_dir=self.root,
            model="tiny",
            enabled=False,
        )

        with patch(
            "app.services.transcription_service.importlib.import_module"
        ) as import_module:
            health = service.health()

        self.assertEqual(health["status"], "disabled")
        self.assertFalse(health["loaded"])
        import_module.assert_not_called()

        with self.assertRaises(TranscriptionError) as raised:
            service.transcribe(self.root / "missing.wav")
        self.assertEqual(raised.exception.code, "transcription_disabled")

    def test_health_reports_dependencies_without_importing_whisper(self):
        service = TranscriptionService(
            source_dir=self.root,
            model="tiny",
        )

        with (
            patch(
                "app.services.transcription_service.importlib.util.find_spec",
                return_value=None,
            ),
            patch(
                "app.services.transcription_service.importlib.import_module"
            ) as import_module,
        ):
            health = service.health()

        self.assertEqual(health["status"], "dependency_missing")
        self.assertEqual(health["dependency_state"], "missing")
        self.assertEqual(
            set(health["missing_dependencies"]),
            set(TranscriptionService.REQUIRED_DEPENDENCIES),
        )
        import_module.assert_not_called()

    def test_health_recognizes_complete_local_model(self):
        model_dir = self.root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.bin").write_bytes(b"placeholder")
        service = TranscriptionService(
            source_dir=self.root,
            model=model_dir,
            device="cpu",
            compute_type="int8",
        )

        with patch(
            "app.services.transcription_service.importlib.util.find_spec",
            return_value=object(),
        ):
            health = service.health()

        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["model_state"], "local")
        self.assertEqual(health["model_path"], str(model_dir.resolve()))
        self.assertFalse(health["loaded"])

    def test_transcribe_reports_model_and_inference_timing(self):
        audio_path = self.root / "utterance.wav"
        audio_path.write_bytes(b"fake-audio")
        service = TranscriptionService(
            source_dir=self.root,
            model="tiny",
            device="cpu",
            compute_type="int8",
        )

        segment = types.SimpleNamespace(
            id=0,
            start=0.0,
            end=1.2,
            text=" 車車",
        )
        info = types.SimpleNamespace(
            duration=1.2,
            language="zh",
            language_probability=0.98,
        )

        class FakeModel:
            @staticmethod
            def transcribe(_path, **_options):
                return iter([segment]), info

        with patch.object(
            service,
            "_get_or_load_model",
            return_value=FakeModel(),
        ):
            result = service.transcribe(audio_path)

        self.assertEqual(result["text"], "車車")
        self.assertEqual(result["model"], "tiny")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["compute_type"], "int8")
        self.assertGreaterEqual(result["processing_seconds"], 0)
        self.assertGreaterEqual(result["model_load_seconds"], 0)
        self.assertGreaterEqual(result["inference_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
