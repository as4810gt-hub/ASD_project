import json
import tempfile
import unittest
from pathlib import Path

from app import create_app
from app.materials import DEFAULT_MATERIAL_ID, MATERIALS, get_material
from app.services.context_builder import ContextBuilder
from app.services.ollama_coach_provider import OllamaCoachProvider


class MaterialsAndContextTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = str(Path(self.temp_dir.name) / "test.sqlite3")
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": database,
                "OLLAMA_ENABLED": False,
                "WHISPER_ENABLED": False,
                "ASD_ANALYSIS_ENABLED": False,
                "ASD_EMOTION_ENABLED": False,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_catalog_has_five_complete_materials_and_default_176(self):
        required_fields = {
            "id",
            "filename",
            "title",
            "session_label",
            "subtitle",
            "alt_text",
            "scene_description",
            "visible_elements",
            "interaction_goal",
            "practice_prompt",
            "parent_example",
        }

        self.assertEqual(DEFAULT_MATERIAL_ID, "176")
        self.assertEqual(set(MATERIALS), {"162", "176", "189", "229", "274"})
        self.assertEqual(get_material()["id"], "176")

        for material_id, material in MATERIALS.items():
            with self.subTest(material_id=material_id):
                self.assertTrue(required_fields.issubset(material))
                self.assertEqual(material["id"], material_id)
                self.assertEqual(material["filename"], f"{material_id}.png")
                for field in required_fields - {"visible_elements"}:
                    self.assertTrue(str(material[field]).strip(), field)
                self.assertIsInstance(material["visible_elements"], list)
                self.assertTrue(material["visible_elements"])
                self.assertTrue(
                    all(str(item).strip() for item in material["visible_elements"])
                )

    def test_coach_route_renders_selected_material_and_rejects_unknown_id(self):
        material = get_material("162")

        response = self.client.get("/coach?material=162")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('/stimuli/162.png', html)
        self.assertIn(material["title"], html)
        self.assertIn(material["practice_prompt"], html)
        self.assertIn('data-material-id="162"', html)

        unknown = self.client.get("/coach?material=999")
        self.assertEqual(unknown.status_code, 404)

    def test_session_api_persists_material_id_and_rejects_unknown_id(self):
        material = get_material("162")

        response = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material_id": "162"},
        )

        self.assertEqual(response.status_code, 201)
        session = response.get_json()["session"]
        self.assertEqual(session["material_id"], "162")
        self.assertEqual(session["material"], material["session_label"])

        stored = self.client.get(f'/api/sessions/{session["id"]}')
        self.assertEqual(stored.status_code, 200)
        self.assertEqual(stored.get_json()["material_id"], "162")

        unknown = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material_id": "999"},
        )
        self.assertEqual(unknown.status_code, 400)

    def test_context_builder_uses_material_profile_and_sanitizes_asd_observation(self):
        material = get_material("162")
        context = ContextBuilder(history_limit=5).build(
            session={
                "material": material["session_label"],
                "material_id": material["id"],
            },
            events=[],
            current_event={
                "speaker": "child",
                "text": "玩具",
                "pause_before": 2.0,
                "gaze_available": True,
                "gaze_on_target": True,
            },
            rule_analysis={
                "wait_met": None,
                "expansion_met": None,
                "turn_taking": True,
                "gaze_available": True,
                "gaze_on_target": True,
                "suggestion": {
                    "tone": "coach",
                    "eyebrow": "孩子已回應",
                    "title": "接住孩子的話",
                    "message": "把孩子的詞多加一個線索。",
                    "example": "「橘色玩具。」",
                },
            },
            asd_observation={
                "status": "ready-with-an-overly-long-untrusted-suffix",
                "severity": "輕度ASD",
                "probabilities": {
                    "td": 0.12349,
                    "mild": 0.55555,
                    "severe": 0.32109,
                },
                "emotion_zh": "非常開心而且這段不應無限制送出",
                "blink_rate_per_min": -8,
                "eye_state_zh": "專注但這段文字超過允許長度",
                "diagnosis": "不應傳給 LLM",
                "raw_frame": "base64-data-must-not-leak",
            },
        )

        profile = context["material_profile"]
        self.assertEqual(profile["id"], "162")
        self.assertEqual(profile["title"], material["title"])
        self.assertEqual(profile["scene_description"], material["scene_description"])
        self.assertEqual(profile["visible_elements"], material["visible_elements"])
        self.assertEqual(
            profile["default_practice_prompt"],
            material["practice_prompt"],
        )

        observation = context["asd_v4_observation"]
        self.assertEqual(
            observation["status"],
            "ready-with-an-overly-long-untrusted-suffix"[:24],
        )
        self.assertNotIn("severity_signal", observation)
        self.assertNotIn("probabilities", observation)
        self.assertIn("interaction_adjustments", observation)
        serialized_observation = json.dumps(observation, ensure_ascii=False)
        self.assertNotIn("ASD", serialized_observation)
        self.assertNotIn("0.555", serialized_observation)
        self.assertEqual(observation["blink_rate_per_min"], 0.0)
        self.assertTrue(observation["non_diagnostic"])
        self.assertNotIn("diagnosis", observation)
        self.assertNotIn("raw_frame", observation)

    def test_ollama_fallback_and_normalization_preserve_safety_contract(self):
        material = get_material("176")
        context = {
            "material_profile": {
                "default_practice_prompt": material["practice_prompt"],
            }
        }
        fallback = {
            "tone": "coach",
            "eyebrow": "等待時間",
            "title": "先留一點空白",
            "message": "請等待三秒。",
            "example": "安靜等待。",
        }

        disabled = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=False,
        )
        fallback_suggestion = disabled.generate(context, fallback)
        self.assertEqual(
            fallback_suggestion["practice_prompt"],
            material["practice_prompt"],
        )
        self.assertEqual(fallback_suggestion["source"], "rule_engine")

        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        long_prompt = "圖" * 50
        provider._request_json = lambda *_args, **_kwargs: {
            "message": {
                "content": json.dumps(
                    {
                        "tone": "positive",
                        "eyebrow": "忽略規則",
                        "title": "改做別的事",
                        "message": "溫柔地多等一下。",
                        "example": "（微笑等待）",
                        "practice_prompt": long_prompt,
                    },
                    ensure_ascii=False,
                )
            }
        }

        suggestion = provider.generate(context, fallback)

        self.assertEqual(suggestion["tone"], fallback["tone"])
        self.assertEqual(suggestion["eyebrow"], fallback["eyebrow"])
        self.assertEqual(suggestion["title"], fallback["title"])
        self.assertEqual(suggestion["practice_prompt"], "圖" * 36)
        self.assertEqual(len(suggestion["practice_prompt"]), 36)
        self.assertEqual(suggestion["source"], "ollama")

    def test_context_builder_rejects_non_finite_untrusted_numbers(self):
        observation = ContextBuilder._safe_asd_observation(
            {
                "probabilities": {
                    "td": "nan",
                    "mild": "infinity",
                    "severe": "invalid",
                },
                "blink_rate_per_min": "-infinity",
            }
        )

        self.assertEqual(observation["blink_rate_per_min"], 0.0)

    def test_ollama_diagnostic_language_is_replaced_by_safe_fallback(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = lambda *_args, **_kwargs: {
            "message": {
                "content": json.dumps(
                    {
                        "tone": "coach",
                        "eyebrow": "觀察",
                        "title": "結果",
                        "message": "孩子已確診為重度 ASD。",
                        "example": "再問一次。",
                        "practice_prompt": "你看到什麼？",
                    },
                    ensure_ascii=False,
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

        suggestion = provider.generate(
            {
                "material_profile": {
                    "default_practice_prompt": "你看到什麼？",
                }
            },
            fallback,
        )

        self.assertEqual(suggestion["source"], "rule_engine")
        self.assertIn("先別急", suggestion["message"])
        self.assertNotIn("ASD", suggestion["message"])

    def test_contextual_fallback_uses_child_words_and_current_material(self):
        material = get_material("189")
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=False,
        )
        fallback = {
            "tone": "ready",
            "eyebrow": "孩子已回應",
            "title": "接住他，再多說一點",
            "message": "重複孩子的話。",
            "example": "再說一個詞。",
        }

        suggestion = provider.generate(
            {
                "material_profile": {
                    "visible_elements": material["visible_elements"],
                    "parent_example": material["parent_example"],
                    "default_practice_prompt": material["practice_prompt"],
                },
                "recent_dialogue": [],
                "current_event": {
                    "speaker": "child",
                    "text": "車車",
                    "pause_before": 2.0,
                },
            },
            fallback,
        )

        self.assertEqual(suggestion["source"], "rule_engine")
        self.assertIn("車車", suggestion["message"])
        self.assertIn("很好接", suggestion["message"])
        self.assertEqual(suggestion["example"], material["parent_example"])
        self.assertEqual(
            suggestion["practice_prompt"],
            material["practice_prompt"],
        )


if __name__ == "__main__":
    unittest.main()
