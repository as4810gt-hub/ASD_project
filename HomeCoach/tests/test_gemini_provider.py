import base64
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from urllib import error

from app.materials import get_material
from app.services.gemini_coach_provider import GeminiCoachProvider


class RecordingFallbackProvider:
    def __init__(self):
        self.calls = []
        self.result = {
            "tone": "ready",
            "eyebrow": "孩子已回應",
            "title": "先接住孩子",
            "message": "這是 fallback provider 的提示。",
            "example": "「我們一起看圖片。」",
            "practice_prompt": "你想先看哪裡？",
            "source": "fallback-provider",
        }

    def generate(self, context, fallback):
        self.calls.append((context, fallback))
        return dict(self.result)


class GeminiCoachProviderContractTests(unittest.TestCase):
    API_KEY = "gemini-secret-contract-key"
    MODEL = "gemini-3.6-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MATERIAL_ID = "176"
    TRUSTED_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
        "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.stimuli_dir = self.temp_path / "stimuli"
        self.stimuli_dir.mkdir()
        (self.stimuli_dir / f"{self.MATERIAL_ID}.png").write_bytes(
            self.TRUSTED_PNG
        )

        self.material = get_material(self.MATERIAL_ID)
        self.fallback = {
            "tone": "ready",
            "eyebrow": "孩子已回應",
            "title": "接住他，再多說一點",
            "message": "沿著孩子的話多加一個畫面線索。",
            "example": "「兩根雪杖交叉了。」",
            "practice_prompt": "你看到雪杖在哪裡？",
        }
        self.model_copy = {
            "tone": self.fallback["tone"],
            "eyebrow": self.fallback["eyebrow"],
            "title": self.fallback["title"],
            "message": "孩子正在問畫面，很好接，先回答他看得見的動作。",
            "example": "「他們把兩根雪杖交叉在一起。」",
            "practice_prompt": "你覺得雪杖像什麼形狀？",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _context(self):
        return {
            "material": self.material["session_label"],
            "material_profile": {
                "id": self.material["id"],
                "title": self.material["title"],
                "scene_description": self.material["scene_description"],
                "visible_elements": self.material["visible_elements"],
                "interaction_goal": self.material["interaction_goal"],
                "default_practice_prompt": self.material["practice_prompt"],
                "parent_example": self.material["parent_example"],
            },
            "recent_dialogue": [],
            "recent_coach_copy": [],
            "current_event": {
                "speaker": "child",
                "text": "他們在幹嘛？",
                "pause_before": 2.0,
                "gaze_available": True,
                "gaze_on_target": True,
            },
            "interaction_brief": {
                "response_mode": "answer_child_question",
                "child_anchor": "他們在幹嘛",
                "image_focus": "雪杖",
            },
            "clinical_rule_suggestion": dict(self.fallback),
        }

    def _provider(self, **overrides):
        options = {
            "api_key": self.API_KEY,
            "model": self.MODEL,
            "base_url": self.BASE_URL,
            "stimuli_dir": self.stimuli_dir,
            "timeout_seconds": 15,
            "enabled": True,
            "thinking_level": "low",
            "max_image_bytes": 3_145_728,
            "fallback_provider": None,
        }
        options.update(overrides)
        return GeminiCoachProvider(**options)

    def _interaction_response(self):
        return {
            "id": "interaction-contract-test",
            "model": self.MODEL,
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                self.model_copy,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                }
            ],
        }

    @staticmethod
    def _input_content(payload):
        interaction_input = payload["input"]
        if isinstance(interaction_input, dict):
            return interaction_input.get("content", [])
        if (
            isinstance(interaction_input, list)
            and len(interaction_input) == 1
            and isinstance(interaction_input[0], dict)
            and interaction_input[0].get("type") == "user_input"
        ):
            return interaction_input[0].get("content", [])
        return interaction_input

    def test_generate_sends_structured_multimodal_interaction_payload(self):
        provider = self._provider()
        provider._request_interaction = Mock(
            return_value=self._interaction_response()
        )

        result = provider.generate(self._context(), self.fallback)

        provider._request_interaction.assert_called_once()
        payload = provider._request_interaction.call_args.args[0]
        self.assertEqual(payload["model"], "gemini-3.6-flash")
        self.assertIs(payload["store"], False)
        self.assertIsInstance(payload["system_instruction"], str)
        self.assertTrue(payload["system_instruction"].strip())
        self.assertEqual(
            payload["generation_config"]["thinking_level"],
            "low",
        )

        response_format = payload["response_format"]
        if isinstance(response_format, list):
            response_format = next(
                item
                for item in response_format
                if item.get("type") == "text"
            )
        self.assertEqual(response_format["type"], "text")
        self.assertEqual(response_format["mime_type"], "application/json")
        schema = response_format["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertTrue(
            {
                "tone",
                "eyebrow",
                "title",
                "message",
                "example",
                "practice_prompt",
            }.issubset(schema["required"])
        )

        content = self._input_content(payload)
        text_blocks = [item for item in content if item.get("type") == "text"]
        image_blocks = [item for item in content if item.get("type") == "image"]
        self.assertTrue(text_blocks)
        self.assertIn(self.MATERIAL_ID, text_blocks[0]["text"])
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["mime_type"], "image/png")
        self.assertEqual(
            image_blocks[0]["data"],
            base64.b64encode(self.TRUSTED_PNG).decode("ascii"),
        )

        self.assertEqual(result["source"], "gemini")
        self.assertEqual(result["model"], self.MODEL)
        self.assertEqual(result["message"], self.model_copy["message"])

    def test_forged_context_paths_cannot_select_the_uploaded_image(self):
        untrusted_png = b"\x89PNG\r\n\x1a\nUNTRUSTED-CONTEXT-FILE"
        untrusted_path = self.temp_path / "outside.png"
        untrusted_path.write_bytes(untrusted_png)
        context = self._context()
        context["filename"] = "../outside.png"
        context["path"] = str(untrusted_path)
        context["material_profile"]["filename"] = "../outside.png"
        context["material_profile"]["path"] = str(untrusted_path)

        provider = self._provider()
        provider._request_interaction = Mock(
            return_value=self._interaction_response()
        )

        provider.generate(context, self.fallback)

        payload = provider._request_interaction.call_args.args[0]
        image_blocks = [
            item
            for item in self._input_content(payload)
            if item.get("type") == "image"
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(
            image_blocks[0]["data"],
            base64.b64encode(self.TRUSTED_PNG).decode("ascii"),
        )
        self.assertNotEqual(
            image_blocks[0]["data"],
            base64.b64encode(untrusted_png).decode("ascii"),
        )

    def test_missing_material_id_never_uploads_the_default_image(self):
        context = self._context()
        context["material_profile"].pop("id")
        provider = self._provider()
        provider._request_interaction = Mock(
            return_value=self._interaction_response()
        )

        provider.generate(context, self.fallback)

        payload = provider._request_interaction.call_args.args[0]
        image_blocks = [
            item
            for item in self._input_content(payload)
            if item.get("type") == "image"
        ]
        self.assertEqual(image_blocks, [])

    def test_missing_api_key_delegates_to_fallback_provider(self):
        fallback_provider = RecordingFallbackProvider()
        provider = self._provider(
            api_key="",
            fallback_provider=fallback_provider,
        )
        provider._request_interaction = Mock(
            side_effect=AssertionError("API must not be called without a key")
        )
        context = self._context()

        result = provider.generate(context, self.fallback)

        provider._request_interaction.assert_not_called()
        self.assertEqual(len(fallback_provider.calls), 1)
        delegated_context, delegated_fallback = fallback_provider.calls[0]
        self.assertEqual(delegated_context, context)
        for field in ("tone", "eyebrow", "title"):
            self.assertEqual(delegated_fallback[field], self.fallback[field])
        self.assertEqual(result, fallback_provider.result)

    def test_api_exception_delegates_to_fallback_provider(self):
        fallback_provider = RecordingFallbackProvider()
        provider = self._provider(fallback_provider=fallback_provider)
        provider._request_interaction = Mock(
            side_effect=error.URLError("offline")
        )
        context = self._context()

        result = provider.generate(context, self.fallback)

        provider._request_interaction.assert_called_once()
        self.assertEqual(len(fallback_provider.calls), 1)
        delegated_context, delegated_fallback = fallback_provider.calls[0]
        self.assertEqual(delegated_context, context)
        for field in ("tone", "eyebrow", "title"):
            self.assertEqual(delegated_fallback[field], self.fallback[field])
        self.assertEqual(result, fallback_provider.result)

    def test_unsafe_gemini_output_delegates_to_fallback_provider(self):
        fallback_provider = RecordingFallbackProvider()
        provider = self._provider(fallback_provider=fallback_provider)
        unsafe_copy = dict(self.model_copy)
        unsafe_copy["message"] = "孩子已確診為重度 ASD。"
        response = self._interaction_response()
        response["steps"][0]["content"][0]["text"] = json.dumps(
            unsafe_copy,
            ensure_ascii=False,
        )
        provider._request_interaction = Mock(return_value=response)

        result = provider.generate(self._context(), self.fallback)

        self.assertEqual(result, fallback_provider.result)
        self.assertEqual(len(fallback_provider.calls), 1)
        self.assertEqual(provider.health()["status"], "degraded")

    def test_health_never_exposes_the_api_key(self):
        provider = self._provider()

        health = provider.health()

        serialized = json.dumps(health, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(self.API_KEY, serialized)
        self.assertEqual(health["model"], self.MODEL)
        self.assertEqual(health["status"], "configured")

        provider._request_interaction = Mock(
            return_value=self._interaction_response()
        )
        provider.generate(self._context(), self.fallback)
        self.assertEqual(provider.health()["status"], "ready")

    def test_dns_failures_are_reported_distinctly_from_api_errors(self):
        failure = error.URLError(
            socket.gaierror(8, "nodename nor servname provided")
        )

        self.assertEqual(
            GeminiCoachProvider._safe_error_name(failure),
            "dns_unavailable",
        )

    def test_provider_has_a_verified_ca_bundle_in_python_venv(self):
        provider = self._provider()

        self.assertTrue(provider.ssl_context.verify_mode)
        self.assertTrue(provider.ssl_context.check_hostname)
        self.assertTrue(provider.ssl_context.get_ca_certs())


if __name__ == "__main__":
    unittest.main()
