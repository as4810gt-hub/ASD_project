import unittest

from app.materials import get_material
from app.services.context_builder import ContextBuilder


class EmotionalContextTests(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder(history_limit=5)
        self.material = get_material("176")

    def _rule_analysis(
        self,
        emotional_bid=None,
        relationship_priority=False,
        suggestion_mode=None,
    ):
        suggestion = {
            "tone": "ready",
            "eyebrow": "孩子已回應",
            "title": "接住孩子的話",
            "message": "沿著孩子的詞多說一點。",
            "example": "「我也看到了。」",
        }
        if suggestion_mode:
            suggestion["response_mode"] = suggestion_mode
        return {
            "wait_met": None,
            "wait_status": None,
            "expansion_met": None,
            "turn_taking": True,
            "gaze_available": True,
            "gaze_on_target": True,
            "emotional_bid": emotional_bid,
            "relationship_priority": relationship_priority,
            "suggestion": suggestion,
        }

    def _build(
        self,
        text,
        emotional_bid=None,
        relationship_priority=False,
        speaker="child",
        events=None,
        suggestion_mode=None,
    ):
        return self.builder.build(
            session={
                "material": self.material["session_label"],
                "material_id": self.material["id"],
            },
            events=events or [],
            current_event={
                "speaker": speaker,
                "text": text,
                "pause_before": 2.0,
                "gaze_available": True,
                "gaze_on_target": True,
            },
            rule_analysis=self._rule_analysis(
                emotional_bid=emotional_bid,
                relationship_priority=relationship_priority,
                suggestion_mode=suggestion_mode,
            ),
        )

    def test_relationship_bid_overrides_picture_question_mode(self):
        context = self._build(
            "媽媽，你是不是覺得我很笨？",
            emotional_bid={
                "active": True,
                "category": "self_worth",
                "signals": ["negative_self_label"],
                "source_speaker": "child",
            },
            relationship_priority=True,
        )

        brief = context["interaction_brief"]
        self.assertEqual(brief["response_mode"], "repair_connection")
        self.assertEqual(brief["emotional_category"], "self_worth")
        self.assertTrue(brief["relationship_priority"])
        self.assertTrue(brief["stay_with_relationship"])
        self.assertEqual(brief["image_focus"], "")
        self.assertIn("承接", brief["micro_action"])
        self.assertIn("停下來聽", brief["micro_action"])
        self.assertIn("不要", brief["micro_action"])

        candidate_text = " ".join(
            brief["candidate_parent_examples"]
            + brief["candidate_practice_prompts"]
        )
        for image_term in self.material["visible_elements"]:
            with self.subTest(image_term=image_term):
                self.assertNotIn(image_term, candidate_text)

        self.assertEqual(
            context["emt_analysis"]["emotional_bid"]["category"],
            "self_worth",
        )

    def test_supported_categories_produce_distinct_relationship_language(self):
        cases = {
            "ability_shame": ("ability", "現在還不會"),
            "rejection_fear": ("belonging", "沒有不要你"),
            "relational_hurt": ("conflict", "可能讓你受傷"),
            "emotional_distress": ("distress", "很不好受"),
            "fear_or_safety": ("safety", "有點害怕"),
        }

        first_examples = set()
        for raw_category, (expected_category, expected_copy) in cases.items():
            with self.subTest(category=raw_category):
                context = self._build(
                    "我現在不想說了。",
                    emotional_bid={
                        "active": True,
                        "category": raw_category,
                        "source_speaker": "child",
                    },
                    relationship_priority=True,
                )
                brief = context["interaction_brief"]
                self.assertEqual(brief["response_mode"], "repair_connection")
                self.assertEqual(
                    brief["emotional_category"],
                    expected_category,
                )
                self.assertIn(
                    expected_copy,
                    brief["candidate_parent_examples"][0],
                )
                first_examples.add(brief["candidate_parent_examples"][0])

        self.assertEqual(len(first_examples), len(cases))

    def test_parent_turn_recovers_bid_from_immediately_previous_child_event(self):
        child_event = {
            "speaker": "child",
            "text": "你是不是不要我了？",
            "pause_before": 2.0,
            "gaze_on_target": False,
            "analysis": {
                "gaze_available": True,
                "emotional_bid": {
                    "active": True,
                    "category": "rejection_fear",
                    "source_speaker": "child",
                },
                "relationship_priority": True,
                "suggestion": {
                    "message": "先接住孩子的擔心。",
                    "example": "「我還在這裡。」",
                    "practice_prompt": "「我想聽你說。」",
                },
            },
        }

        context = self._build(
            "才沒有，你先看圖片。",
            speaker="parent",
            events=[child_event],
            emotional_bid={"active": False},
        )

        brief = context["interaction_brief"]
        self.assertEqual(brief["response_mode"], "repair_connection")
        self.assertEqual(brief["emotional_category"], "belonging")
        self.assertEqual(brief["image_focus"], "")
        self.assertTrue(
            context["emt_analysis"]["emotional_bid"]["from_previous_turn"]
        )

    def test_suggestion_response_mode_alone_activates_safe_general_repair(self):
        context = self._build(
            "我不知道怎麼說。",
            emotional_bid={"active": False},
            suggestion_mode="repair_connection",
        )

        brief = context["interaction_brief"]
        self.assertEqual(brief["response_mode"], "repair_connection")
        self.assertEqual(brief["emotional_category"], "general")
        self.assertEqual(brief["image_focus"], "")
        self.assertTrue(brief["candidate_parent_examples"])

    def test_inactive_bid_preserves_normal_picture_coaching(self):
        context = self._build(
            "他們在做什麼？",
            emotional_bid={
                "active": False,
                "category": "self_worth",
                "source_speaker": "child",
            },
            relationship_priority=False,
        )

        brief = context["interaction_brief"]
        self.assertEqual(brief["response_mode"], "answer_child_question")
        self.assertFalse(brief["relationship_priority"])
        self.assertFalse(brief["stay_with_relationship"])
        self.assertIn(brief["image_focus"], self.material["visible_elements"])
        self.assertIsNone(context["emt_analysis"]["emotional_bid"])

    def test_safety_context_has_no_picture_instruction_or_model_downgrade_hint(self):
        context = self.builder.build(
            session={
                "material": self.material["session_label"],
                "material_id": self.material["id"],
            },
            events=[],
            current_event={
                "speaker": "child",
                "text": "我不想活了。",
                "pause_before": 2.0,
                "gaze_available": True,
                "gaze_on_target": True,
            },
            rule_analysis=self._rule_analysis(
                emotional_bid={
                    "active": True,
                    "category": "urgent_safety",
                    "signals": ["urgent_safety", "self_harm"],
                    "source_speaker": "child",
                },
                relationship_priority=True,
                suggestion_mode="safety_check",
            ),
            asd_observation={
                "status": "ready",
                "emotion_zh": "開心",
                "emotion_available": True,
                "classification_available": True,
                "eye_state_zh": "穩定",
            },
        )

        brief = context["interaction_brief"]
        self.assertEqual(brief["response_mode"], "safety_check")
        self.assertTrue(brief["safety_priority"])
        self.assertEqual(brief["image_focus"], "")
        adjustments = context["asd_v4_observation"][
            "interaction_adjustments"
        ]
        self.assertTrue(any("不要用表情" in item for item in adjustments))
        self.assertFalse(any("畫面細節" in item for item in adjustments))

    def test_relationship_context_replaces_conflicting_asd_picture_adjustment(self):
        context = self.builder.build(
            session={
                "material": self.material["session_label"],
                "material_id": self.material["id"],
            },
            events=[],
            current_event={
                "speaker": "child",
                "text": "我好害怕。",
                "pause_before": 2.0,
                "gaze_available": True,
                "gaze_on_target": False,
            },
            rule_analysis=self._rule_analysis(
                emotional_bid={
                    "active": True,
                    "category": "emotional_distress",
                    "source_speaker": "child",
                },
                relationship_priority=True,
            ),
            asd_observation={
                "status": "ready",
                "emotion_zh": "恐懼",
                "emotion_available": True,
                "classification_available": True,
                "eye_state_zh": "迴避",
            },
        )

        self.assertEqual(
            context["interaction_brief"]["emotional_category"],
            "safety",
        )
        adjustments = context["asd_v4_observation"][
            "interaction_adjustments"
        ]
        self.assertTrue(any("關係回應" in item for item in adjustments))
        self.assertFalse(any("畫面細節" in item for item in adjustments))

    def test_cloud_context_caps_prior_plus_current_transcript_at_five(self):
        events = [
            {
                "speaker": "child" if index % 2 == 0 else "parent",
                "text": f"先前第 {index} 句",
                "pause_before": 1.0,
                "gaze_on_target": True,
                "analysis": {
                    "gaze_available": True,
                    "suggestion": {},
                },
            }
            for index in range(7)
        ]

        context = self._build("目前這一句。", events=events)

        self.assertEqual(len(context["recent_dialogue"]), 4)
        self.assertEqual(context["recent_dialogue"][0]["text"], "先前第 3 句")
        self.assertEqual(context["current_event"]["text"], "目前這一句。")


if __name__ == "__main__":
    unittest.main()
