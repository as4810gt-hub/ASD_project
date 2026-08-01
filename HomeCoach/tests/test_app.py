import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import create_app
from app.services.emt_rule_engine import EMTRuleEngine
from app.services.ollama_coach_provider import OllamaCoachProvider


class RuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = EMTRuleEngine()

    def test_parent_expansion_and_wait_are_detected(self):
        prior = [{"speaker": "child", "text": "紅色球球"}]
        result = self.engine.analyze(
            speaker="parent",
            text="紅色球球滾過來了",
            pause_before=3.6,
            gaze_on_target=True,
            prior_events=prior,
        )
        self.assertTrue(result["wait_met"])
        self.assertTrue(result["expansion_met"])

    def test_short_wait_generates_waiting_suggestion(self):
        prior = [{"speaker": "child", "text": "球球"}]
        result = self.engine.analyze(
            speaker="parent",
            text="對，球球",
            pause_before=1.2,
            gaze_on_target=True,
            prior_events=prior,
        )
        self.assertFalse(result["wait_met"])
        self.assertEqual(result["suggestion"]["eyebrow"], "等待時間")

    def test_unavailable_gaze_does_not_trigger_gaze_suggestion(self):
        result = self.engine.analyze(
            speaker="child",
            text="球球",
            pause_before=2.0,
            gaze_on_target=False,
            gaze_available=False,
            prior_events=[],
        )

        self.assertFalse(result["gaze_available"])
        self.assertEqual(result["suggestion"]["eyebrow"], "孩子已回應")


class OllamaProviderTests(unittest.TestCase):
    def test_disabled_provider_returns_rule_fallback(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=False,
        )
        fallback = {
            "tone": "coach",
            "eyebrow": "等待時間",
            "title": "先留一點空白",
            "message": "請等待三秒。",
            "example": "安靜等待。",
        }

        suggestion = provider.generate({}, fallback)

        self.assertEqual(suggestion["source"], "rule_engine")
        self.assertEqual(suggestion["title"], fallback["title"])

    def test_model_cannot_change_the_rule_engine_target(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = lambda *_args, **_kwargs: {
            "message": {
                "content": (
                    '{"tone":"positive","eyebrow":"忽略規則",'
                    '"title":"改做別的事","message":"溫柔地多等一下。",'
                    '"example":"（微笑等待）"}'
                )
            }
        }
        fallback = {
            "tone": "coach",
            "eyebrow": "等待時間",
            "title": "先留一點空白",
            "message": "請等待三秒。",
            "example": "安靜等待。",
        }

        suggestion = provider.generate({}, fallback)

        self.assertEqual(suggestion["tone"], fallback["tone"])
        self.assertEqual(suggestion["eyebrow"], fallback["eyebrow"])
        self.assertEqual(suggestion["title"], fallback["title"])
        self.assertEqual(suggestion["message"], "溫柔地多等一下。")
        self.assertEqual(suggestion["source"], "ollama")


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = str(Path(self.temp_dir.name) / "test.sqlite3")
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": database,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pages_render(self):
        for path in ("/", "/coach", "/records"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)

    def test_local_webgazer_assets_are_served_from_allowlist(self):
        script = self.client.get("/vendor/webgazer.js")
        face_mesh = self.client.get(
            "/mediapipe/face_mesh/face_mesh.binarypb"
        )
        denied = self.client.get(
            "/mediapipe/face_mesh/../../package.json"
        )

        self.assertEqual(script.status_code, 200)
        self.assertIn(b"webgazer", script.data.lower())
        self.assertEqual(face_mesh.status_code, 200)
        self.assertGreater(len(face_mesh.data), 0)
        self.assertEqual(denied.status_code, 404)
        script.close()
        face_mesh.close()
        denied.close()

    def test_session_rejects_non_object_json(self):
        response = self.client.post(
            "/api/sessions",
            json=["不是", "物件"],
        )

        self.assertEqual(response.status_code, 400)

    def test_session_event_flow(self):
        response = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material": "玩具"},
        )
        self.assertEqual(response.status_code, 201)
        session_id = response.get_json()["session"]["id"]

        event = self.client.post(
            f"/api/sessions/{session_id}/events",
            json={
                "speaker": "child",
                "text": "球球",
                "pause_before": 3.1,
                "gaze_on_target": True,
            },
        )
        self.assertEqual(event.status_code, 201)
        self.assertIn("suggestion", event.get_json())
        self.assertEqual(event.get_json()["coach_source"], "rule_engine")
        self.assertFalse(event.get_json()["event"]["gaze_available"])

    def test_event_api_parses_string_booleans(self):
        response = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material": "玩具"},
        )
        session_id = response.get_json()["session"]["id"]

        event = self.client.post(
            f"/api/sessions/{session_id}/events",
            json={
                "speaker": "child",
                "text": "球球",
                "gaze_available": "true",
                "gaze_on_target": "false",
            },
        )

        self.assertEqual(event.status_code, 201)
        payload = event.get_json()
        self.assertFalse(payload["event"]["gaze_on_target"])
        self.assertTrue(payload["event"]["gaze_available"])
        self.assertEqual(payload["suggestion"]["eyebrow"], "共同注意")

    def test_event_returns_immediate_fallback_then_refines_with_ollama(self):
        class DeferredProvider:
            enabled = True

            def __init__(self):
                self.generate_calls = 0

            @staticmethod
            def fallback(context, fallback):
                return {
                    **fallback,
                    "message": "文字完成後立刻顯示的提示",
                    "example": "「先看看圖片。」",
                    "practice_prompt": "你想先看哪裡？",
                    "source": "rule_engine",
                }

            def generate(self, context, fallback):
                self.generate_calls += 1
                return {
                    **fallback,
                    "message": "Ollama 在背景完成的自然提示",
                    "source": "ollama",
                    "model": "fake-model",
                }

        provider = DeferredProvider()
        service = self.app.extensions["coaching_service"]
        service.coach_provider = provider
        session = service.start_session("測試", "看圖共讀：雪地活動")

        initial = self.client.post(
            f'/api/sessions/{session["id"]}/events',
            json={
                "speaker": "child",
                "text": "雪雪",
                "gaze_available": False,
                "gaze_on_target": True,
            },
        )

        self.assertEqual(initial.status_code, 201)
        initial_payload = initial.get_json()
        self.assertTrue(initial_payload["coach_pending"])
        self.assertEqual(initial_payload["coach_source"], "rule_engine")
        self.assertEqual(provider.generate_calls, 0)

        event_id = initial_payload["event"]["id"]
        refined = self.client.post(
            f'/api/sessions/{session["id"]}/events/{event_id}/coach-refinement',
            json={},
        )

        self.assertEqual(refined.status_code, 200)
        refined_payload = refined.get_json()
        self.assertEqual(refined_payload["coach_source"], "ollama")
        self.assertFalse(refined_payload["coach_pending"])
        self.assertEqual(provider.generate_calls, 1)
        stored = service.get_session(session["id"])["events"][0]
        self.assertEqual(
            stored["analysis"]["suggestion"]["message"],
            "Ollama 在背景完成的自然提示",
        )

    def test_urgent_safety_event_never_queues_background_model_refinement(self):
        class DeferredProvider:
            enabled = True

            def __init__(self):
                self.generate_calls = 0

            @staticmethod
            def fallback(context, fallback):
                return {**fallback, "source": "rule_engine"}

            def generate(self, context, fallback):
                self.generate_calls += 1
                return {**fallback, "source": "ollama"}

        provider = DeferredProvider()
        service = self.app.extensions["coaching_service"]
        service.coach_provider = provider
        session = service.start_session("測試", "看圖共讀：雪地活動")

        response = self.client.post(
            f'/api/sessions/{session["id"]}/events',
            json={
                "speaker": "child",
                "text": "我不想活了。",
                "gaze_available": True,
                "gaze_on_target": False,
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertFalse(payload["coach_pending"])
        self.assertEqual(payload["suggestion"]["response_mode"], "safety_check")
        self.assertEqual(provider.generate_calls, 0)

    def test_same_session_events_are_serialized(self):
        class SlowProvider:
            @staticmethod
            def generate(context, fallback):
                time.sleep(0.04)
                return {**fallback, "source": "rule_engine"}

        service = self.app.extensions["coaching_service"]
        service.coach_provider = SlowProvider()
        session = service.start_session("測試", "玩具")
        start_barrier = threading.Barrier(3)

        def record(text):
            start_barrier.wait()
            return service.record_event(
                session_id=session["id"],
                speaker="parent",
                text=text,
                pause_before=0,
                gaze_on_target=True,
                gaze_available=False,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(record, "第一句")
            second = executor.submit(record, "第二句")
            start_barrier.wait()
            first.result()
            second.result()

        stored = service.get_session(session["id"])
        self.assertTrue(stored["events"][0]["analysis"]["turn_taking"])
        self.assertFalse(stored["events"][1]["analysis"]["turn_taking"])
        self.assertEqual(stored["turn_taking_rate"], 0)

    def test_health_includes_whisper_without_loading_a_model(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        whisper = response.get_json()["modules"]["whisper"]
        self.assertEqual(whisper["model"], "base")
        self.assertFalse(whisper["loaded"])


if __name__ == "__main__":
    unittest.main()
