import unittest

from app.services.emt_rule_engine import EMTRuleEngine


class EmotionalPriorityTests(unittest.TestCase):
    def setUp(self):
        self.engine = EMTRuleEngine()

    def _analyze_child(self, text, *, gaze_on_target=False):
        return self.engine.analyze(
            speaker="child",
            text=text,
            pause_before=2.0,
            gaze_on_target=gaze_on_target,
            gaze_available=True,
            prior_events=[],
        )

    def test_structured_categories_cover_more_than_one_trigger_phrase(self):
        cases = (
            ("媽媽，你是不是覺得我是笨蛋？", "self_worth"),
            ("我是不是很笨？", "self_worth"),
            ("我真的很差勁。", "self_worth"),
            ("我覺得自己很沒用。", "self_worth"),
            ("我根本就是垃圾。", "self_worth"),
            ("你覺得我是廢物嗎？", "self_worth"),
            ("我是不是討人厭、沒救了？", "self_worth"),
            ("我不值得被愛。", "self_worth"),
            ("是不是我什麼都做不好？", "ability_shame"),
            ("別人都會，只有我又做錯了。", "ability_shame"),
            ("都是我的錯。", "ability_shame"),
            ("爸爸，你會不會把我丟下？", "rejection_fear"),
            ("你還愛我嗎？", "rejection_fear"),
            ("你根本不在乎我。", "rejection_fear"),
            ("你都不陪我。", "rejection_fear"),
            ("你剛剛都不聽我說。", "relational_hurt"),
            ("你那樣說，我很受傷。", "relational_hurt"),
            ("你不懂我。", "relational_hurt"),
            ("你只會罵我。", "relational_hurt"),
            ("我真的好難過，我想哭。", "emotional_distress"),
            ("我快受不了了。", "emotional_distress"),
            ("我不想跟你說了。", "emotional_distress"),
        )

        for text, expected_category in cases:
            with self.subTest(text=text):
                result = self._analyze_child(text)
                bid = result["emotional_bid"]

                self.assertTrue(bid["active"])
                self.assertEqual(bid["category"], expected_category)
                self.assertGreaterEqual(bid["confidence"], 0.8)
                self.assertEqual(bid["source_speaker"], "child")
                self.assertTrue(result["relationship_priority"])
                self.assertEqual(
                    result["suggestion"]["response_mode"],
                    "repair_connection",
                )
                self.assertNotRegex(
                    result["suggestion"]["message"],
                    r"圖片|教材|畫面",
                )

    def test_child_bid_overrides_gaze_prompt(self):
        result = self._analyze_child(
            "媽媽，你是不是覺得我是笨蛋？",
            gaze_on_target=False,
        )

        self.assertEqual(result["suggestion"]["eyebrow"], "情緒接住")
        self.assertEqual(
            result["suggestion"]["title"],
            "先回答孩子心裡的問題",
        )
        self.assertNotEqual(result["suggestion"]["eyebrow"], "共同注意")
        self.assertNotIn("這很好接", result["suggestion"]["message"])

    def test_parent_turn_after_bid_keeps_relationship_priority_and_metrics(self):
        result = self.engine.analyze(
            speaker="parent",
            text="我才沒有，你看雪地裡有兩個人。",
            pause_before=1.0,
            gaze_on_target=False,
            gaze_available=True,
            prior_events=[
                {
                    "speaker": "child",
                    "text": "媽媽，你是不是覺得我是笨蛋？",
                }
            ],
        )

        # Relationship coaching changes the target, not the underlying EMT
        # measurements.  A short wait and non-expansion stay measurable.
        self.assertIs(result["wait_met"], False)
        self.assertEqual(result["wait_status"], "too_short")
        self.assertIs(result["expansion_met"], False)
        self.assertIs(result["gaze_on_target"], False)
        self.assertTrue(result["relationship_priority"])
        self.assertEqual(result["emotional_bid"]["category"], "self_worth")
        self.assertEqual(
            result["emotional_bid"]["source_speaker"],
            "child",
        )
        self.assertEqual(result["suggestion"]["eyebrow"], "情緒接住")
        self.assertEqual(
            result["suggestion"]["title"],
            "先留在孩子的感受上",
        )
        self.assertEqual(
            result["suggestion"]["response_mode"],
            "repair_connection",
        )

    def test_parent_turn_can_reuse_bid_stored_in_previous_analysis(self):
        result = self.engine.analyze(
            speaker="parent",
            text="我在這裡。",
            pause_before=4.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[
                {
                    "speaker": "child",
                    "text": "嗯。",
                    "analysis": {
                        "emotional_bid": {
                            "active": True,
                            "category": "rejection_fear",
                            "confidence": 0.97,
                            "signals": ["rejection_fear"],
                            "source_speaker": "child",
                        }
                    },
                }
            ],
        )

        self.assertTrue(result["relationship_priority"])
        self.assertEqual(result["emotional_bid"]["category"], "rejection_fear")
        self.assertEqual(result["emotional_bid"]["confidence"], 0.97)
        self.assertIn("安全感", result["suggestion"]["message"])

    def test_parent_turn_keeps_a_safe_llm_escalation_that_rules_missed(self):
        child_text = "你每次都把我放到最後。"
        self.assertFalse(
            self._analyze_child(child_text)["emotional_bid"]["active"]
        )

        result = self.engine.analyze(
            speaker="parent",
            text="我們先看下一張圖。",
            pause_before=4.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[
                {
                    "speaker": "child",
                    "text": child_text,
                    "analysis": {
                        "emotional_bid": {
                            "active": False,
                            "category": None,
                        },
                        "suggestion": {
                            "response_mode": "repair_connection",
                            "emotional_category": "general",
                            "coach_target_source": "llm_escalation",
                        },
                    },
                }
            ],
        )

        self.assertTrue(result["relationship_priority"])
        self.assertTrue(result["emotional_bid"]["active"])
        self.assertEqual(
            result["emotional_bid"]["signals"],
            ["llm_escalation"],
        )
        self.assertEqual(
            result["suggestion"]["response_mode"],
            "repair_connection",
        )
        self.assertEqual(
            result["suggestion"]["title"],
            "先留在孩子的感受上",
        )

    def test_context_is_required_to_avoid_picture_and_skill_false_positives(self):
        ordinary_lines = (
            "圖片裡的兩個笨蛋在雪地裡打架。",
            "我看到圖裡有一個壞人。",
            "我不會念雪杖這個字。",
            "媽媽說圖片裡的人很笨。",
            "那個人不喜歡我手上的雪杖。",
            "我要把垃圾丟掉。",
            "他覺得圖片裡的人很蠢。",
            "都是我的雪杖。",
            "我拿錯雪杖了。",
            "媽媽在圖片裡陪我找雪杖。",
            "你覺得我畫的壞人像誰？",
            "你覺得我手上的垃圾要丟哪裡？",
            "我麻煩你幫我拿雪杖。",
            "我是一個垃圾桶。",
            "你覺得我畫的人不會滑雪嗎？",
            "你覺得我旁邊的人做不到嗎？",
            "媽媽覺得我拿的車做不好嗎？",
            "你不喜歡我畫的人嗎？",
            "你不要我的雪杖嗎？",
            "你討厭我說的那個壞人嗎？",
            "媽媽不想要我手上的積木嗎？",
            "沒有人喜歡我的畫。",
            "你離開我的房間了嗎？",
            "你還愛我的畫嗎？",
            "爸爸打我的玩具鼓。",
            "媽媽推我的娃娃車。",
            "爸爸打我畫的壞人。",
            "你笑我畫的人嗎？",
            "你不理我說的角色嗎？",
            "你罵我做的壞人面具嗎？",
            "我是壞人的好朋友。",
            "我是沒用完的紅紙。",
            "我是失敗組的隊長。",
            "我是不重要角色的配音員。",
            "我總是不會把垃圾丟地上。",
            "我都不會傷害別人。",
            "我永遠不會做壞事。",
            "別人都會亂丟垃圾，只有我不會。",
            "都怪我的鞋子太滑。",
            "全怪我的鉛筆斷掉。",
            "這些都是我的錯字。",
            "都是我的問題集。",
            "小熊說媽媽不愛我了。",
            "故事裡的小孩問你還愛我嗎？",
            "圖片裡弟弟說你是不是覺得我是笨蛋？",
            "我在念台詞：我是壞人。",
            "小兔子說我很怕。",
            "我不想死。",
            "我沒有想過自殺。",
            "我不會傷害自己。",
            "我想去死海玩。",
            "我想死你了。",
            "我要看死亡筆記本。",
            "我想玩消失魔術。",
            "我不想活動。",
            "我想傷害自己畫的角色。",
            "故事裡的人說：我想死。",
            "壞人死掉比較好。",
        )

        for text in ordinary_lines:
            with self.subTest(text=text):
                result = self._analyze_child(text, gaze_on_target=True)
                self.assertFalse(result["emotional_bid"]["active"])
                self.assertIsNone(result["emotional_bid"]["category"])
                self.assertEqual(result["emotional_bid"]["confidence"], 0.0)
                self.assertFalse(result["relationship_priority"])
                self.assertNotIn("response_mode", result["suggestion"])

    def test_fear_about_a_creation_is_emotion_not_rejection_of_the_child(self):
        result = self._analyze_child("我怕你不愛我的作品。")

        self.assertTrue(result["emotional_bid"]["active"])
        self.assertEqual(
            result["emotional_bid"]["category"],
            "emotional_distress",
        )

    def test_natural_phrasings_cover_relationship_and_ability_context(self):
        cases = (
            ("你根本就不在乎我。", "rejection_fear"),
            ("你為什麼不愛我？", "rejection_fear"),
            ("你是不是還在生我的氣？", "rejection_fear"),
            ("你到底愛不愛我？", "rejection_fear"),
            ("你不想跟我玩了嗎？", "rejection_fear"),
            ("你只愛弟弟不愛我。", "rejection_fear"),
            ("你要把我丟掉嗎？", "rejection_fear"),
            ("你剛才罵我。", "relational_hurt"),
            ("你剛剛叫我閉嘴。", "relational_hurt"),
            ("你從來不聽我說。", "relational_hurt"),
            ("你這樣說我很難過。", "relational_hurt"),
            ("你把我當笨蛋嗎？", "self_worth"),
            ("你是不是看不起我？", "self_worth"),
            ("我是不是讓你很失望？", "self_worth"),
            ("我是個廢物。", "self_worth"),
            ("我超沒用。", "self_worth"),
            ("我怎麼那麼沒用。", "self_worth"),
            ("我好麻煩。", "self_worth"),
            ("為什麼只有我做不到？", "ability_shame"),
            ("大家都會，只有我不會。", "ability_shame"),
            ("我怎麼連這個都不會？", "ability_shame"),
            ("我學畫畫很久還是不會。", "ability_shame"),
            ("我覺得很難過。", "emotional_distress"),
            ("我今天有一點不開心。", "emotional_distress"),
            ("好怕喔。", "emotional_distress"),
        )

        for text, category in cases:
            with self.subTest(text=text):
                result = self._analyze_child(text)
                self.assertTrue(result["emotional_bid"]["active"])
                self.assertEqual(result["emotional_bid"]["category"], category)

    def test_unanswered_bid_survives_a_parent_deflection_and_short_child_turn(self):
        child = self.engine.analyze(
            speaker="child",
            text="媽媽，你是不是覺得我是笨蛋？",
            pause_before=2.0,
            gaze_on_target=False,
            gaze_available=True,
            prior_events=[],
        )
        child_event = {
            "speaker": "child",
            "text": "媽媽，你是不是覺得我是笨蛋？",
            "analysis": child,
        }
        parent = self.engine.analyze(
            speaker="parent",
            text="我才沒有，你看雪地裡有兩個人。",
            pause_before=4.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[child_event],
        )
        parent_event = {
            "speaker": "parent",
            "text": "我才沒有，你看雪地裡有兩個人。",
            "analysis": parent,
        }

        continued = self.engine.analyze(
            speaker="child",
            text="嗯。",
            pause_before=2.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[child_event, parent_event],
        )

        self.assertTrue(continued["relationship_priority"])
        self.assertEqual(continued["emotional_bid"]["category"], "self_worth")
        self.assertIn(
            "continued_after_missed_response",
            continued["emotional_bid"]["signals"],
        )
        self.assertEqual(
            continued["suggestion"]["response_mode"],
            "repair_connection",
        )
        self.assertEqual(continued["suggestion"]["title"], "孩子還在等你回應")

    def test_late_llm_refinement_survives_the_parent_event_race(self):
        child_event = {
            "speaker": "child",
            "text": "你每次都把我放到最後。",
            "analysis": {
                "emotional_bid": {"active": False, "category": None},
                "suggestion": {
                    "response_mode": "repair_connection",
                    "emotional_category": "general",
                    "coach_target_source": "llm_escalation",
                },
            },
        }
        parent_event = {
            "speaker": "parent",
            "text": "我們先看下一張圖片。",
            "analysis": {
                "emotional_bid": {"active": False, "category": None},
                "suggestion": {
                    "response_mode": "recast_parent_turn",
                    "eyebrow": "語句擴展",
                },
            },
        }

        result = self.engine.analyze(
            speaker="child",
            text="嗯。",
            pause_before=2.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[child_event, parent_event],
        )

        self.assertTrue(result["relationship_priority"])
        self.assertIn(
            "continued_after_late_refinement",
            result["emotional_bid"]["signals"],
        )
        self.assertEqual(
            result["suggestion"]["response_mode"],
            "repair_connection",
        )

    def test_completed_relationship_answer_releases_the_next_child_turn(self):
        child_event = {
            "speaker": "child",
            "text": "你是不是覺得我是笨蛋？",
            "analysis": self._analyze_child("你是不是覺得我是笨蛋？"),
        }
        parent = self.engine.analyze(
            speaker="parent",
            text="我不覺得你笨。我想聽你說。",
            pause_before=4.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[child_event],
        )
        parent_event = {
            "speaker": "parent",
            "text": "我不覺得你笨。我想聽你說。",
            "analysis": parent,
        }

        next_turn = self.engine.analyze(
            speaker="child",
            text="嗯。",
            pause_before=2.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[child_event, parent_event],
        )

        self.assertFalse(next_turn["relationship_priority"])

    def test_mocking_or_reversed_reassurance_does_not_close_the_bid(self):
        for parent_text in (
            "我愛你才怪。",
            "我沒有不要你才怪。",
            "我在乎你才怪。",
            "我愛你，但是你很煩。",
            "我不覺得你笨，我只是覺得你沒用。",
            "我愛你，但你再這樣我就不要你了。",
            "對不起，但都是你的錯。",
        ):
            with self.subTest(parent_text=parent_text):
                self.assertFalse(
                    self.engine._parent_addressed_emotional_bid(
                        parent_text,
                        "rejection_fear",
                    )
                )

    def test_urgent_safety_language_stops_picture_coaching_and_persists(self):
        disclosures = (
            "我不想活了。媽媽你不要管我。",
            "我想死，你不要跟別人說。",
            "我想傷害自己，剛剛就想了。",
            "我想消失。",
            "我以前想死過。",
            "如果你不要我，我就去死。",
            "我想一睡不醒。",
            "我受不了了，想一睡不醒。",
            "我想睡著後永遠不要再醒來。",
            "我不願意再活下去。",
            "我打算自殺。",
            "我準備自殺。",
            "我想自殘。",
            "我想結束自己的生命。",
            "我想弄傷自己。",
            "我想跳樓。",
            "我已經割腕了。",
            "沒有我你們比較好。",
            "我不值得活著。",
            "老師一直打我。",
            "我被爸爸打。",
            "爸爸昨天用皮帶打我。",
            "叔叔逼我脫衣服。",
            "媽媽說要殺我。",
            "有人摸我身體。",
        )
        for text in disclosures:
            with self.subTest(text=text):
                result = self._analyze_child(text)
                self.assertEqual(
                    result["emotional_bid"]["category"],
                    "urgent_safety",
                )
                self.assertEqual(
                    result["suggestion"]["response_mode"],
                    "safety_check",
                )
                self.assertNotRegex(
                    " ".join(result["suggestion"].values()),
                    r"圖片|教材|畫面",
                )

        initial = self._analyze_child("我不想活了。")
        safety_event = {
            "speaker": "child",
            "text": "我不想活了。",
            "analysis": initial,
        }
        later = self.engine.analyze(
            speaker="child",
            text="嗯。",
            pause_before=2.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[
                safety_event,
                {
                    "speaker": "parent",
                    "text": "我會陪你。",
                    "analysis": self.engine.analyze(
                        speaker="parent",
                        text="我會陪你。",
                        pause_before=2.0,
                        gaze_on_target=True,
                        gaze_available=True,
                        prior_events=[safety_event],
                    ),
                },
            ],
        )
        self.assertEqual(later["suggestion"]["response_mode"], "safety_check")

    def test_parent_using_the_same_words_is_not_mislabeled_as_the_child(self):
        result = self.engine.analyze(
            speaker="parent",
            text="我不想活了。",
            pause_before=2.0,
            gaze_on_target=True,
            gaze_available=True,
            prior_events=[],
        )

        self.assertNotEqual(
            result["suggestion"].get("response_mode"),
            "safety_check",
        )

    def test_off_material_safety_turns_do_not_lower_shared_reading_metrics(self):
        metrics = self.engine.summarize(
            [
                {
                    "speaker": "child",
                    "pause_before": 1.0,
                    "analysis": {
                        "turn_taking": True,
                        "wait_met": None,
                        "expansion_met": None,
                        "suggestion": {},
                    },
                },
                {
                    "speaker": "parent",
                    "pause_before": 4.0,
                    "analysis": {
                        "turn_taking": True,
                        "wait_met": True,
                        "expansion_met": True,
                        "relationship_priority": False,
                        "emotional_bid": {"active": False},
                        "suggestion": {"response_mode": "expand_child_idea"},
                    },
                },
                {
                    "speaker": "parent",
                    "pause_before": 1.0,
                    "analysis": {
                        "turn_taking": True,
                        "wait_met": False,
                        "expansion_met": False,
                        "relationship_priority": True,
                        "emotional_bid": {
                            "active": True,
                            "category": "urgent_safety",
                        },
                        "suggestion": {"response_mode": "safety_check"},
                    },
                },
            ]
        )

        self.assertEqual(metrics["average_wait"], 4.0)
        self.assertEqual(metrics["expansion_rate"], 100)

    def test_unrelated_child_turn_preserves_existing_wait_priority(self):
        result = self.engine.analyze(
            speaker="parent",
            text="對，球球。",
            pause_before=1.2,
            gaze_on_target=False,
            gaze_available=True,
            prior_events=[{"speaker": "child", "text": "球球。"}],
        )

        self.assertFalse(result["relationship_priority"])
        self.assertFalse(result["emotional_bid"]["active"])
        self.assertEqual(result["suggestion"]["eyebrow"], "等待時間")


if __name__ == "__main__":
    unittest.main()
