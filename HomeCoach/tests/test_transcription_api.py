import io
import tempfile
import unittest
from pathlib import Path

from app import create_app


class FakeTranscriptionService:
    def __init__(self, result):
        self.result = result
        self.paths = []
        self.payloads = []

    def health(self):
        return {
            "status": "ready",
            "model": "fake",
            "loaded": False,
        }

    def transcribe(self, path):
        audio_path = Path(path)
        self.paths.append(audio_path)
        self.payloads.append(audio_path.read_bytes())
        return self.result


class TranscriptionRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = str(Path(self.temp_dir.name) / "test.sqlite3")
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": database,
                "WHISPER_ENABLED": False,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_session(self):
        response = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material": "玩具"},
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["session"]["id"]

    @staticmethod
    def _audio_form(**fields):
        return {
            "audio": (io.BytesIO(b"complete-webm-audio"), "utterance.webm"),
            "speaker": "child",
            "pause_before": "3.2",
            "gaze_available": "false",
            "gaze_on_target": "false",
            **fields,
        }

    def test_successful_transcription_creates_event_and_removes_temp_file(self):
        fake = FakeTranscriptionService(
            {
                "text": "紅色球球",
                "language": "zh",
                "language_probability": 0.97,
                "duration": 1.4,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.4,
                        "text": "紅色球球",
                    }
                ],
            }
        )
        self.app.extensions["transcription_service"] = fake
        session_id = self._create_session()

        response = self.client.post(
            f"/api/sessions/{session_id}/transcriptions",
            data=self._audio_form(),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["transcription"]["text"], "紅色球球")
        self.assertEqual(payload["event"]["text"], "紅色球球")
        self.assertFalse(payload["event"]["gaze_available"])
        self.assertFalse(payload["event"]["analysis"]["gaze_available"])
        self.assertEqual(
            payload["event"]["analysis"]["transcription"]["language"],
            "zh",
        )
        self.assertEqual(payload["coach_source"], "rule_engine")
        self.assertFalse(payload["coach_pending"])
        self.assertEqual(fake.payloads, [b"complete-webm-audio"])
        self.assertFalse(fake.paths[0].exists())

    def test_no_speech_does_not_create_an_event(self):
        fake = FakeTranscriptionService(
            {
                "text": "  ",
                "language": "zh",
                "language_probability": 0.0,
                "duration": 0.8,
                "segments": [],
            }
        )
        self.app.extensions["transcription_service"] = fake
        session_id = self._create_session()

        response = self.client.post(
            f"/api/sessions/{session_id}/transcriptions",
            data=self._audio_form(),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "no_speech")
        session = self.client.get(f"/api/sessions/{session_id}").get_json()
        self.assertEqual(session["events"], [])
        self.assertFalse(fake.paths[0].exists())

    def test_audio_size_is_limited_before_transcription(self):
        fake = FakeTranscriptionService({"text": "不應執行"})
        self.app.extensions["transcription_service"] = fake
        self.app.config["WHISPER_MAX_AUDIO_BYTES"] = 4
        session_id = self._create_session()

        response = self.client.post(
            f"/api/sessions/{session_id}/transcriptions",
            data=self._audio_form(),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(fake.paths, [])

    def test_invalid_gaze_boolean_is_rejected(self):
        fake = FakeTranscriptionService({"text": "不應執行"})
        self.app.extensions["transcription_service"] = fake
        session_id = self._create_session()

        response = self.client.post(
            f"/api/sessions/{session_id}/transcriptions",
            data=self._audio_form(gaze_available="unknown"),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(fake.paths, [])

    def test_completed_session_rejects_transcription_with_conflict(self):
        session_id = self._create_session()
        finished = self.client.post(f"/api/sessions/{session_id}/finish")
        self.assertEqual(finished.status_code, 200)

        response = self.client.post(
            f"/api/sessions/{session_id}/transcriptions",
            data=self._audio_form(),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
