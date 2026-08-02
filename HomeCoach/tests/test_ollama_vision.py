import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.materials import get_material
from app.services.ollama_coach_provider import OllamaCoachProvider


class OllamaVisionTests(unittest.TestCase):
    TRUSTED_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
        "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stimuli_dir = Path(self.temp_dir.name) / "stimuli"
        self.stimuli_dir.mkdir()
        (self.stimuli_dir / "176.png").write_bytes(self.TRUSTED_PNG)
        material = get_material("176")
        self.context = {
            "material_profile": {
                "id": material["id"],
                "title": material["title"],
                "scene_description": material["scene_description"],
                "visible_elements": material["visible_elements"],
            },
            "recent_dialogue": [],
            "recent_coach_copy": [],
            "current_event": {"speaker": "child", "text": "他們在做什麼？"},
            "interaction_brief": {
                "response_mode": "answer_child_question",
                "image_focus": "雪杖",
            },
            "clinical_rule_suggestion": {},
        }
        self.fallback = {
            "tone": "ready",
            "eyebrow": "孩子已回應",
            "title": "接住孩子的話",
            "message": "先回答孩子。",
            "example": "「他們拿著雪杖。」",
            "practice_prompt": "你覺得他們在做什麼？",
        }
        self.model_copy = {
            **self.fallback,
            "response_mode": "follow_rule",
            "message": "孩子真的在問你，先用圖片裡的動作回答。",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _provider(self, stimuli_dir):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="gemma4:e4b",
            enabled=True,
            stimuli_dir=stimuli_dir,
        )
        provider._request_json = Mock(
            return_value={
                "message": {
                    "content": json.dumps(self.model_copy, ensure_ascii=False)
                }
            }
        )
        return provider

    def test_generate_attaches_catalog_image_to_ollama_user_message(self):
        provider = self._provider(self.stimuli_dir)

        result = provider.generate(self.context, self.fallback)

        payload = provider._request_json.call_args.args[1]
        user_message = payload["messages"][1]
        self.assertEqual(
            user_message["images"],
            [base64.b64encode(self.TRUSTED_PNG).decode("ascii")],
        )
        self.assertEqual(result["source"], "ollama")

    def test_disabled_vision_does_not_add_images(self):
        provider = self._provider(None)

        provider.generate(self.context, self.fallback)

        payload = provider._request_json.call_args.args[1]
        self.assertNotIn("images", payload["messages"][1])

    def test_context_paths_cannot_select_an_untrusted_image(self):
        outside = Path(self.temp_dir.name) / "outside.png"
        outside.write_bytes(b"\x89PNG\r\n\x1a\nUNTRUSTED")
        self.context["path"] = str(outside)
        self.context["material_profile"]["path"] = str(outside)
        provider = self._provider(self.stimuli_dir)

        provider.generate(self.context, self.fallback)

        payload = provider._request_json.call_args.args[1]
        self.assertEqual(
            payload["messages"][1]["images"],
            [base64.b64encode(self.TRUSTED_PNG).decode("ascii")],
        )


if __name__ == "__main__":
    unittest.main()
