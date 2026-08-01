import json
import re
import unittest

from app.materials import get_material
from app.services.context_builder import ContextBuilder
from app.services.emt_rule_engine import EMTRuleEngine
from app.services.ollama_coach_provider import OllamaCoachProvider


class CoachQualityContractTests(unittest.TestCase):
    MATERIAL_ID = "176"

    def setUp(self):
        self.material = get_material(self.MATERIAL_ID)
        self.engine = EMTRuleEngine()
        self.builder = ContextBuilder(history_limit=5)

    def _child_analysis(self, text):
        return self.engine.analyze(
            speaker="child",
            text=text,
            pause_before=2.0,
            gaze_on_target=True,
            prior_events=[],
            gaze_available=True,
        )

    def _build_child_context(self, text, events=None):
        return self.builder.build(
            session={
                "material": self.material["session_label"],
                "material_id": self.material["id"],
            },
            events=events or [],
            current_event={
                "speaker": "child",
                "text": text,
                "pause_before": 2.0,
                "gaze_available": True,
                "gaze_on_target": True,
            },
            rule_analysis=self._child_analysis(text),
        )

    @staticmethod
    def _compact(text):
        return re.sub(r"[\s，。！？!?、；：:「」『』]", "", str(text))

    def test_long_parent_wait_after_child_still_meets_wait_contract(self):
        result = self.engine.analyze(
            speaker="parent",
            text="對，他們把雪杖交叉了。",
            pause_before=12.0,
            gaze_on_target=True,
            prior_events=[
                {
                    "speaker": "child",
                    "text": "他們在做什麼？",
                }
            ],
            gaze_available=True,
        )

        self.assertIs(result["wait_met"], True)
        self.assertNotEqual(result["suggestion"]["eyebrow"], "等待時間")

    def test_child_question_requests_an_answer_in_interaction_brief(self):
        context = self._build_child_context("對呀，他們在幹嘛？")

        self.assertIn("interaction_brief", context)
        self.assertEqual(
            context["interaction_brief"]["response_mode"],
            "answer_child_question",
        )

    def test_child_statement_is_anchored_for_expansion(self):
        context = self._build_child_context("他們在打架")

        self.assertIn("interaction_brief", context)
        brief = context["interaction_brief"]
        self.assertEqual(brief["response_mode"], "expand_child_idea")
        self.assertIn("打架", brief["child_anchor"])

    def test_disabled_fallback_does_not_echo_a_child_question_as_parent_copy(self):
        child_question = "對呀，他們在幹嘛？"
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="test-model",
            enabled=False,
        )
        context = {
            "material_profile": {
                "scene_description": self.material["scene_description"],
                "visible_elements": self.material["visible_elements"],
                "default_practice_prompt": self.material["practice_prompt"],
                "parent_example": self.material["parent_example"],
            },
            "recent_dialogue": [],
            "recent_coach_copy": [],
            "current_event": {
                "speaker": "child",
                "text": child_question,
                "pause_before": 2.0,
            },
            "interaction_brief": {
                "response_mode": "answer_child_question",
                "child_anchor": "他們在幹嘛",
                "image_focus": "雪杖",
            },
        }

        suggestion = provider.generate(
            context,
            self._child_analysis(child_question)["suggestion"],
        )

        bad_echo = self._compact("對，對呀他們在幹嘛")
        for field in ("message", "example", "practice_prompt"):
            with self.subTest(field=field):
                self.assertNotIn(bad_echo, self._compact(suggestion[field]))

        # The example is the literal copy a parent may say aloud. It must not
        # ask the parent to mirror the child's entire question back verbatim.
        self.assertNotIn(
            self._compact(child_question),
            self._compact(suggestion["example"]),
        )
        combined_guidance = self._compact(
            f'{suggestion["message"]} {suggestion["example"]}'
        )
        for directive in (
            "重複整句問句",
            "把整句問句重複",
            "照著孩子的問句再說一次",
            "重複孩子剛才的整句",
        ):
            with self.subTest(directive=directive):
                self.assertNotIn(self._compact(directive), combined_guidance)

    def test_image_focus_rotates_away_from_elements_in_recent_coach_copy(self):
        events = [
            {
                "speaker": "parent",
                "text": "你看這裡。",
                "pause_before": 3.0,
                "gaze_on_target": True,
                "analysis": {
                    "gaze_available": True,
                    "suggestion": {
                        "message": "先看看兩個人和地上的白雪。",
                        "example": "「兩個人站在白雪裡。」",
                        "practice_prompt": "你看到兩個人了嗎？",
                    },
                },
            },
            {
                "speaker": "child",
                "text": "看到了。",
                "pause_before": 2.0,
                "gaze_on_target": True,
                "analysis": {
                    "gaze_available": True,
                    "suggestion": {
                        "message": "再找找樹林旁邊的雪杖。",
                        "example": "「雪杖在樹林前面。」",
                        "practice_prompt": "雪杖在哪裡？",
                    },
                },
            },
        ]

        context = self._build_child_context("我看到了", events=events)

        self.assertIn("interaction_brief", context)
        image_focus = context["interaction_brief"]["image_focus"]
        used_elements = {"兩個人", "白雪", "樹林", "雪杖"}
        self.assertIn(image_focus, self.material["visible_elements"])
        self.assertNotIn(image_focus, used_elements)
        self.assertEqual(image_focus, "雪鞋")

    def test_model_cannot_tell_parent_to_echo_a_child_question(self):
        question = "對呀，他們在幹嘛？"
        context = self._build_child_context(question)
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="weak-model",
            enabled=True,
        )
        provider._request_json = lambda *_args, **_kwargs: {
            "message": {
                "content": json.dumps(
                    {
                        "tone": "ready",
                        "eyebrow": "孩子已回應",
                        "title": "接住他，再多說一點",
                        "message": "先跟著他說『對呀，他們在幹嘛』。",
                        "example": "「雪杖碰在一起了。」",
                        "practice_prompt": "雪地裡站著兩個人。",
                    },
                    ensure_ascii=False,
                )
            }
        }

        suggestion = provider.generate(
            context,
            self._child_analysis(question)["suggestion"],
        )

        self.assertNotIn("跟著他說", suggestion["message"])
        self.assertIn("先用畫面", suggestion["message"])

    def test_model_cannot_reintroduce_corrected_asr_homophone(self):
        raw_text = "他們在打借"
        context = self._build_child_context(raw_text)
        self.assertEqual(context["interaction_brief"]["child_anchor"], "他們在打架")
        provider = OllamaCoachProvider(
            base_url="http://localhost:11434/api",
            model="weak-model",
            enabled=True,
        )
        provider._request_json = lambda *_args, **_kwargs: {
            "message": {
                "content": json.dumps(
                    {
                        "tone": "ready",
                        "eyebrow": "孩子已回應",
                        "title": "接住他，再多說一點",
                        "message": "孩子說他們在打借。",
                        "example": "「對，他們在打借。」",
                        "practice_prompt": "他們是在打借嗎？",
                    },
                    ensure_ascii=False,
                )
            }
        }

        suggestion = provider.generate(
            context,
            self._child_analysis(raw_text)["suggestion"],
        )

        combined = " ".join(
            suggestion[field]
            for field in ("message", "example", "practice_prompt")
        )
        self.assertNotIn("打借", combined)
        self.assertIn("打架", combined)


if __name__ == "__main__":
    unittest.main()
