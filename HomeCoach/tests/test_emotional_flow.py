import tempfile
import unittest
from pathlib import Path

from app import create_app
from app.materials import get_material


class EmotionalCoachingFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(Path(self.temp_dir.name) / "flow.sqlite3"),
            }
        )
        self.client = self.app.test_client()
        self.material = get_material("176")

        response = self.client.post(
            "/api/sessions",
            json={"child_name": "測試", "material_id": "176"},
        )
        self.session_id = response.get_json()["session"]["id"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _event(self, speaker, text, pause, gaze=False):
        return self.client.post(
            f"/api/sessions/{self.session_id}/events",
            json={
                "speaker": speaker,
                "text": text,
                "pause_before": pause,
                "gaze_available": True,
                "gaze_on_target": gaze,
            },
        ).get_json()

    def _assert_stays_off_material(self, suggestion):
        spoken_copy = " ".join(
            str(suggestion.get(field) or "")
            for field in ("example", "practice_prompt")
        )
        for term in (
            "圖片",
            "教材",
            "畫面",
            "繪本",
            *self.material["visible_elements"],
        ):
            with self.subTest(term=term):
                self.assertNotIn(term, spoken_copy)
        self.assertNotIn("這很好接", " ".join(suggestion.values()))

    def test_screenshot_scenario_stays_with_relationship_across_two_turns(self):
        child = self._event(
            "child",
            "媽媽，你是不是覺得我是笨蛋？",
            27.4,
            gaze=False,
        )

        self.assertTrue(child["event"]["analysis"]["emotional_bid"]["active"])
        self.assertEqual(
            child["event"]["analysis"]["emotional_bid"]["category"],
            "self_worth",
        )
        self.assertEqual(child["suggestion"]["response_mode"], "repair_connection")
        self.assertEqual(child["suggestion"]["eyebrow"], "情緒接住")
        self.assertIn(
            "不覺得你笨",
            f'{child["suggestion"]["example"]} '
            f'{child["suggestion"]["practice_prompt"]}',
        )
        self._assert_stays_off_material(child["suggestion"])

        parent = self._event(
            "parent",
            "我才沒有，你看看雪地裡面站著兩個人耶。",
            15.4,
            gaze=True,
        )

        # The underlying EMT measurements remain available, but they no
        # longer steal the coaching target from the relationship repair.
        self.assertIs(parent["event"]["analysis"]["wait_met"], True)
        self.assertIs(parent["event"]["analysis"]["expansion_met"], False)
        self.assertEqual(parent["suggestion"]["response_mode"], "repair_connection")
        self.assertEqual(parent["suggestion"]["eyebrow"], "關係回應")
        self.assertEqual(
            parent["suggestion"]["title"],
            "先留在孩子的感受上",
        )
        self._assert_stays_off_material(parent["suggestion"])

        child_follow_up = self._event(
            "child",
            "嗯。",
            2.0,
            gaze=True,
        )

        self.assertEqual(
            child_follow_up["suggestion"]["response_mode"],
            "repair_connection",
        )
        self.assertEqual(
            child_follow_up["suggestion"]["title"],
            "孩子還在等你回應",
        )
        self._assert_stays_off_material(child_follow_up["suggestion"])

    def test_picture_story_with_a_negative_word_does_not_false_trigger(self):
        ordinary = self._event(
            "child",
            "圖片裡的兩個人互相叫對方笨蛋。",
            2.0,
            gaze=True,
        )

        self.assertFalse(
            ordinary["event"]["analysis"]["emotional_bid"]["active"]
        )
        self.assertNotEqual(
            ordinary["suggestion"].get("response_mode"),
            "repair_connection",
        )


if __name__ == "__main__":
    unittest.main()
