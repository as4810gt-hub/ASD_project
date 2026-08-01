import io
import json
import tempfile
import unittest
from pathlib import Path

from app import create_app
from app.services.asd_analysis_service import ASDAnalysisService


class FakeASDAnalysisService:
    def __init__(self):
        self.calls = []
        self.latest = None
        self.finished = []

    def health(self):
        return {
            "status": "ready",
            "severity": {"status": "ready"},
            "emotion": {"status": "ready"},
            "non_diagnostic": True,
        }

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        self.latest = {
            "status": "ready",
            "severity": "輕度ASD",
            "classification_available": True,
            "probabilities": {"td": 0.1, "mild": 0.7, "severe": 0.2},
            "emotion": "happy",
            "emotion_zh": "開心",
            "emotion_available": True,
            "blink_rate_per_min": kwargs["blink_rate_per_min"],
            "blink_available": kwargs["blink_available"],
            "eye_state": "transitional",
            "eye_state_zh": "轉換中",
            "non_diagnostic": True,
            "errors": [],
        }
        return self.latest

    def get_latest(self, _session_id):
        return self.latest

    def finish_session(self, session_id):
        self.finished.append(session_id)
        latest = self.latest
        self.latest = None
        return latest


class ASDAnalysisRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(Path(self.temp_dir.name) / "test.sqlite3"),
                "OLLAMA_ENABLED": False,
                "WHISPER_ENABLED": False,
                "ASD_FRAME_MAX_BYTES": 1024,
            }
        )
        self.fake = FakeASDAnalysisService()
        self.app.extensions["asd_analysis_service"] = self.fake
        self.app.extensions["coaching_service"].asd_analysis_service = self.fake
        self.client = self.app.test_client()
        response = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material_id": "176"},
        )
        self.session_id = response.get_json()["session"]["id"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analysis_upload_parses_signals_and_keeps_frame_in_memory(self):
        response = self.client.post(
            f"/api/sessions/{self.session_id}/asd-analysis",
            data={
                "gaze_samples": json.dumps(
                    [
                        {"x": 100, "y": 200, "at": 10.5},
                        {"x": "bad", "y": 2, "at": 11},
                    ]
                ),
                "viewport_width": "1280",
                "viewport_height": "720",
                "blink_rate_per_min": "18.4",
                "blink_available": "true",
                "face_found": "true",
                "frame": (io.BytesIO(b"encoded-image"), "frame.jpg", "image/jpeg"),
            },
        )

        self.assertEqual(response.status_code, 200)
        analysis = response.get_json()["analysis"]
        self.assertEqual(analysis["severity"], "輕度ASD")
        self.assertTrue(analysis["non_diagnostic"])

        call = self.fake.calls[0]
        self.assertEqual(call["session_id"], self.session_id)
        self.assertEqual(call["gaze_samples"], [{"x": 100.0, "y": 200.0, "at": 10.5}])
        self.assertEqual(call["blink_rate_per_min"], 18.4)
        self.assertTrue(call["blink_available"])
        self.assertTrue(call["face_found"])
        self.assertEqual(call["frame_bytes"], b"encoded-image")

    def test_analysis_rejects_bad_input_and_completed_session(self):
        invalid = self.client.post(
            f"/api/sessions/{self.session_id}/asd-analysis",
            data={
                "gaze_samples": "not-json",
                "blink_available": "maybe",
            },
        )
        self.assertEqual(invalid.status_code, 400)

        self.client.post(f"/api/sessions/{self.session_id}/finish", json={})
        completed = self.client.post(
            f"/api/sessions/{self.session_id}/asd-analysis",
            data={"gaze_samples": "[]"},
        )
        self.assertEqual(completed.status_code, 409)
        self.assertEqual(self.fake.finished, [self.session_id])

    def test_health_exposes_nested_asd_component_status(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        asd = response.get_json()["modules"]["asd_analysis"]
        self.assertEqual(asd["severity"]["status"], "ready")
        self.assertEqual(asd["emotion"]["status"], "ready")
        self.assertTrue(asd["non_diagnostic"])


class ASDAnalysisServiceTests(unittest.TestCase):
    class FakeModel:
        classes_ = [2, 0, 1]

        @staticmethod
        def predict(_samples):
            return [1]

        @staticmethod
        def predict_proba(_samples):
            return [[0.2, 0.5, 0.3]]

    def make_service(self):
        service = ASDAnalysisService(
            model_path="/missing/severity.pkl",
            emotion_module_path="/missing/emotion.py",
            enabled=True,
            emotion_enabled=False,
        )
        service._model = self.FakeModel()
        service._feature_cols = service.FEATURE_COLUMNS
        return service

    def test_fixation_features_and_model_class_order_mapping(self):
        service = self.make_service()
        gazes = [(100.0 + index, 100.0) for index in range(4)]
        gazes += [(500.0 + index, 400.0) for index in range(4)]
        centres, durations = service._extract_fixations(gazes)
        features = service._compute_features(centres, durations)

        self.assertEqual(len(features), 11)
        result = service._classify(features, centres)
        self.assertEqual(result["severity"], "輕度ASD")
        self.assertEqual(
            result["probabilities"],
            {"td": 0.5, "mild": 0.3, "severe": 0.2},
        )

    def test_real_timestamps_drive_duration_and_outside_samples_are_ignored(self):
        service = self.make_service()
        normalized, ignored = service._normalize_gaze_samples(
            [
                {"x": -1, "y": 20, "at": 0},
                {"x": 100, "y": 100, "at": 10},
                {"x": 101, "y": 100, "at": 110},
            ],
            viewport_width=1280,
            viewport_height=720,
        )
        self.assertEqual(ignored, 1)

        centres, durations = service._extract_fixations(normalized)
        self.assertEqual(len(centres), 1)
        self.assertAlmostEqual(durations[0], 200.0)

        separated, separated_durations = service._extract_fixations(
            [(100, 100, 0), (101, 100, 400)]
        )
        self.assertEqual(separated, [])
        self.assertEqual(separated_durations, [])

    def test_analyze_is_json_safe_and_clears_session_state(self):
        service = self.make_service()
        samples = [
            {"x": 100 + index, "y": 100, "at": index * 33.3}
            for index in range(4)
        ]
        result = service.analyze(
            session_id=7,
            gaze_samples=samples,
            viewport_width=1280,
            viewport_height=720,
            blink_rate_per_min=16,
            blink_available=True,
            face_found=True,
        )

        json.dumps(result, ensure_ascii=False)
        self.assertTrue(result["classification_available"])
        self.assertTrue(result["blink_available"])
        self.assertTrue(result["non_diagnostic"])
        self.assertEqual(service.get_latest(7), result)
        self.assertEqual(service.finish_session(7), result)
        self.assertIsNone(service.get_latest(7))


if __name__ == "__main__":
    unittest.main()
