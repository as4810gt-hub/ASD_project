import json
import unittest
from unittest.mock import Mock

from app.services.gemini_coach_provider import GeminiCoachProvider
from app.services.ollama_coach_provider import OllamaCoachProvider


class RepairConnectionProviderTests(unittest.TestCase):
    MATERIAL_PROFILE = {
        "id": "176",
        "title": "雪地裡的兩個人",
        "scene_description": "兩個人站在雪地，手上的雪杖交叉。",
        "visible_elements": ["兩個人", "雪地", "雪杖", "樹林"],
        "default_practice_prompt": "你看到雪杖在哪裡？",
        "parent_example": "「兩個人站在白白的雪地上。」",
    }
    IMAGE_FALLBACK = {
        "tone": "ready",
        "eyebrow": "語句擴展",
        "title": "接住他的詞，再添一點",
        "message": "孩子的話很好接，再補一個圖片裡的小線索。",
        "example": "「兩個人站在白白的雪地上。」",
        "practice_prompt": "你看到雪杖在哪裡？",
    }

    def _context(self, *, repair=True, child_text=None):
        child_text = child_text or "媽媽，你是不是覺得我是笨蛋？"
        brief = {
            "response_mode": (
                "repair_connection" if repair else "answer_child_question"
            ),
            "micro_action": "先回答關係問題，再接住孩子的感受。",
            "child_question": child_text,
            "child_anchor": "",
            "image_focus": "" if repair else "雪杖",
            "candidate_parent_examples": [
                "「我沒有那樣看你。你剛才是不是有點難過？」"
            ],
            "candidate_practice_prompts": [
                "我有聽見；你現在心裡是什麼感覺？"
            ],
        }
        if repair:
            brief.update(
                {
                    "relationship_priority": True,
                    "stay_with_relationship": True,
                    "emotional_category": "self_worth",
                }
            )
        return {
            "material_profile": dict(self.MATERIAL_PROFILE),
            "recent_dialogue": [],
            "recent_coach_copy": [],
            "current_event": {
                "speaker": "child",
                "text": child_text,
                "pause_before": 2.0,
            },
            "interaction_brief": brief,
            "emt_analysis": {
                "relationship_priority": repair,
                "emotional_bid": {
                    "active": repair,
                    "category": "self_worth" if repair else None,
                    "source_speaker": "child",
                },
            },
            "clinical_rule_suggestion": dict(self.IMAGE_FALLBACK),
        }

    @staticmethod
    def _ollama_response(copy):
        return {"message": {"content": json.dumps(copy, ensure_ascii=False)}}

    @staticmethod
    def _gemini_response(copy):
        return {
            "id": "repair-test",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(copy, ensure_ascii=False),
                        }
                    ],
                }
            ],
        }

    def test_rule_repair_fallback_answers_explicit_label_before_feeling(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=False,
        )

        result = provider.generate(self._context(), self.IMAGE_FALLBACK)

        self.assertEqual(result["source"], "rule_engine")
        self.assertEqual(result["response_mode"], "repair_connection")
        self.assertEqual(result["tone"], "notice")
        self.assertEqual(result["eyebrow"], "情緒接住")
        self.assertEqual(result["title"], "先回答孩子心裡的問題")
        self.assertIn("不覺得你笨", result["example"])
        self.assertIn("不覺得你笨", result["practice_prompt"])
        combined = " ".join(
            result[field] for field in ("message", "example", "practice_prompt")
        )
        for forbidden in ("圖片", "教材", "畫面", "雪地", "雪杖", "很好接"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)

    def test_emotional_statement_uses_a_feeling_title_not_question_title(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=False,
        )
        context = self._context(child_text="我真的好難過，我想哭。")
        context["interaction_brief"]["emotional_category"] = "distress"
        context["emt_analysis"]["emotional_bid"][
            "category"
        ] = "distress"

        result = provider.generate(context, self.IMAGE_FALLBACK)

        self.assertEqual(result["response_mode"], "repair_connection")
        self.assertEqual(result["emotional_category"], "distress")
        self.assertEqual(result["title"], "先接住孩子現在的感受")

    def test_ollama_rejects_repair_copy_that_redirects_to_the_picture(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = Mock(
            return_value=self._ollama_response(
                {
                    "response_mode": "repair_connection",
                    "tone": "ready",
                    "eyebrow": "語句擴展",
                    "title": "回到圖片",
                    "message": "這很好接，先請孩子看看圖片裡的兩個人。",
                    "example": "「我們回到雪地裡找雪杖。」",
                    "practice_prompt": "你看到雪杖在哪裡？",
                }
            )
        )

        result = provider.generate(self._context(), self.IMAGE_FALLBACK)

        self.assertEqual(result["source"], "rule_engine")
        self.assertEqual(result["response_mode"], "repair_connection")
        combined = " ".join(
            result[field] for field in ("message", "example", "practice_prompt")
        )
        for forbidden in ("圖片", "雪地", "雪杖", "很好接"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)
        self.assertIn("不覺得你笨", combined)

    def test_model_cannot_downgrade_an_existing_repair_target(self):
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = Mock(
            return_value=self._ollama_response(
                {
                    "response_mode": "follow_rule",
                    "tone": "ready",
                    "eyebrow": "語句擴展",
                    "title": "接著說圖片",
                    "message": "先清楚回答孩子，再問問他是不是有點難過。",
                    "example": "「我不覺得你笨。你是不是有點難過？」",
                    "practice_prompt": "我不覺得你笨；你願意告訴我怎麼了嗎？",
                }
            )
        )

        result = provider.generate(self._context(), self.IMAGE_FALLBACK)

        self.assertEqual(result["source"], "ollama")
        self.assertEqual(result["response_mode"], "repair_connection")
        self.assertEqual(result["eyebrow"], "情緒接住")
        self.assertEqual(result["title"], "先回答孩子心裡的問題")
        self.assertEqual(result["coach_target_source"], "clinical_rule")

    def test_carried_repair_message_acknowledges_current_parent_turn(self):
        context = self._context()
        context["recent_dialogue"] = [
            {
                "speaker": "child",
                "text": "我不想理你了，你走開。",
                "pause_before": 2.0,
            }
        ]
        context["current_event"] = {
            "speaker": "parent",
            "text": "你看雪地這麼白，是不是很想跑？",
            "pause_before": 2.0,
        }
        context["interaction_brief"].update(
            {
                "current_turn_acknowledgement_required": True,
                "relationship_continues_from_previous_turn": True,
                "relationship_response_state": "needs_repair",
            }
        )
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = Mock(
            return_value=self._ollama_response(
                {
                    "response_mode": "repair_connection",
                    "tone": "ready",
                    "eyebrow": "情緒接住",
                    "title": "先接住孩子",
                    "message": "孩子剛才說不想理你，先聽他的感受。",
                    "example": "「我聽見你很不好受，我在這裡陪你。」",
                    "practice_prompt": "我想聽你說，剛才是不是很難受？",
                }
            )
        )

        result = provider.generate(context, self.IMAGE_FALLBACK)

        self.assertEqual(result["source"], "ollama")
        self.assertIn("你這一句", result["message"])
        self.assertIn("上一輪", result["message"])
        self.assertNotIn("message", result["model_generated_fields"])
        self.assertIn("example", result["model_generated_fields"])

    def test_model_can_safely_escalate_a_missed_relational_hurt(self):
        context = self._context(
            repair=False,
            child_text="你都不聽我說，我很難過。",
        )
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = Mock(
            return_value=self._ollama_response(
                {
                    "response_mode": "repair_connection",
                    "tone": "ready",
                    "eyebrow": "語句擴展",
                    "title": "繼續共讀",
                    "message": "先承認剛才沒有聽完，讓孩子知道你現在願意聽。",
                    "example": "「對不起，剛才我沒聽完。你願意再說給我聽嗎？」",
                    "practice_prompt": "剛才我漏聽了；你想從哪裡再告訴我？",
                }
            )
        )

        result = provider.generate(context, self.IMAGE_FALLBACK)

        self.assertEqual(result["source"], "ollama")
        self.assertEqual(result["response_mode"], "repair_connection")
        self.assertEqual(result["coach_target_source"], "llm_escalation")
        self.assertEqual(result["eyebrow"], "情緒接住")
        self.assertNotIn("雪杖", " ".join(result.values()))

    def test_rejected_model_escalation_uses_relationship_message_not_wait_copy(self):
        context = self._context(
            repair=False,
            child_text="你每次都把我放到最後。",
        )
        wait_fallback = {
            **self.IMAGE_FALLBACK,
            "eyebrow": "等待時間",
            "title": "先別急著接下一句",
            "message": "剛才等一秒；看著孩子，心裡數到三。",
        }
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = Mock(
            return_value=self._ollama_response(
                {
                    "response_mode": "repair_connection",
                    "tone": "ready",
                    "eyebrow": "語句擴展",
                    "title": "回圖片",
                    "message": "先看圖片再說。",
                    "example": "「雪地裡有兩個人。」",
                    "practice_prompt": "你看到雪杖嗎？",
                }
            )
        )

        result = provider.generate(context, wait_fallback)

        self.assertEqual(result["response_mode"], "repair_connection")
        self.assertEqual(result["coach_target_source"], "llm_escalation")
        self.assertNotIn("數到三", result["message"])
        self.assertNotRegex(
            " ".join(
                result[field]
                for field in ("message", "example", "practice_prompt")
            ),
            r"圖片|雪地|雪杖",
        )

    def test_whitespace_only_model_fields_fall_back_instead_of_staying_blank(self):
        context = self._context(repair=False, child_text="兩個人。")
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=True,
        )
        provider._request_json = Mock(
            return_value=self._ollama_response(
                {
                    "response_mode": "follow_rule",
                    "tone": "ready",
                    "eyebrow": "",
                    "title": "",
                    "message": "   ",
                    "example": "\n",
                    "practice_prompt": "\t",
                }
            )
        )

        result = provider.generate(context, self.IMAGE_FALLBACK)

        self.assertTrue(result["message"].strip())
        self.assertTrue(result["example"].strip())
        self.assertTrue(result["practice_prompt"].strip())

    def test_gemini_never_uploads_an_urgent_safety_disclosure(self):
        context = self._context(child_text="我不想活了。")
        context["interaction_brief"].update(
            {
                "response_mode": "safety_check",
                "emotional_category": "urgent_safety",
                "safety_priority": True,
            }
        )
        context["emt_analysis"].update(
            {
                "safety_priority": True,
                "emotional_bid": {
                    "active": True,
                    "category": "urgent_safety",
                    "signals": ["urgent_safety", "self_harm"],
                    "source_speaker": "child",
                },
            }
        )
        context["clinical_rule_suggestion"]["response_mode"] = "safety_check"
        provider = GeminiCoachProvider(
            api_key="test-secret",
            model="gemini-test",
            base_url="https://example.invalid/v1beta",
            stimuli_dir=None,
            enabled=True,
            fallback_provider=None,
        )
        provider._request_interaction = Mock()

        result = provider.generate(context, self.IMAGE_FALLBACK)

        provider._request_interaction.assert_not_called()
        self.assertEqual(result["source"], "rule_engine")
        self.assertEqual(result["response_mode"], "safety_check")
        self.assertEqual(result["eyebrow"], "安全優先")
        self.assertIn("不要讓孩子獨處", result["message"])
        self.assertNotRegex(
            " ".join(
                result[field]
                for field in ("message", "example", "practice_prompt")
            ),
            r"圖片|雪地|雪杖",
        )

    def test_gemini_never_uploads_a_possible_abuse_disclosure(self):
        context = self._context(child_text="叔叔逼我脫衣服。")
        context["interaction_brief"].update(
            {
                "response_mode": "safety_check",
                "emotional_category": "urgent_safety",
                "safety_priority": True,
            }
        )
        context["emt_analysis"].update(
            {
                "safety_priority": True,
                "emotional_bid": {
                    "active": True,
                    "category": "urgent_safety",
                    "signals": ["urgent_safety", "possible_abuse"],
                    "source_speaker": "child",
                },
            }
        )
        context["clinical_rule_suggestion"]["response_mode"] = "safety_check"
        provider = GeminiCoachProvider(
            api_key="test-secret",
            model="gemini-test",
            base_url="https://example.invalid/v1beta",
            stimuli_dir=None,
            enabled=True,
            fallback_provider=None,
        )
        provider._request_interaction = Mock()

        result = provider.generate(context, self.IMAGE_FALLBACK)

        provider._request_interaction.assert_not_called()
        self.assertEqual(result["response_mode"], "safety_check")
        self.assertEqual(result["safety_kind"], "possible_abuse")
        self.assertIn("這不是你的錯", result["example"])

    def test_gemini_repair_rejection_fails_closed_without_model_handoff(self):
        provider = GeminiCoachProvider(
            api_key="test-secret",
            model="gemini-test",
            base_url="https://example.invalid/v1beta",
            stimuli_dir=None,
            enabled=True,
            fallback_provider=None,
        )
        provider._request_interaction = Mock(
            return_value=self._gemini_response(
                {
                    "response_mode": "repair_connection",
                    "tone": "ready",
                    "eyebrow": "共同注意",
                    "title": "看回教材",
                    "message": "先回到教材，看看畫面裡的兩個人。",
                    "example": "「雪地裡有兩個人。」",
                    "practice_prompt": "你看到雪杖了嗎？",
                }
            )
        )

        result = provider.generate(self._context(), self.IMAGE_FALLBACK)

        self.assertEqual(result["source"], "rule_engine")
        self.assertEqual(result["response_mode"], "repair_connection")
        self.assertEqual(provider.health()["status"], "degraded")
        combined = " ".join(
            result[field] for field in ("message", "example", "practice_prompt")
        )
        self.assertNotIn("教材", combined)
        self.assertNotIn("雪杖", combined)

    def test_output_schema_exposes_only_bounded_target_correction(self):
        mode_schema = OllamaCoachProvider.OUTPUT_SCHEMA["properties"][
            "response_mode"
        ]

        self.assertEqual(
            set(mode_schema["enum"]),
            {"follow_rule", "repair_connection"},
        )
        self.assertIn(
            "response_mode",
            OllamaCoachProvider.OUTPUT_SCHEMA["required"],
        )


if __name__ == "__main__":
    unittest.main()
