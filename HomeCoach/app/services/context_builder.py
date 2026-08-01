import re

from ..materials import get_material, get_material_by_title


class ContextBuilder:
    """Build a small, privacy-conscious context for the local coach model."""

    RELATIONSHIP_RESPONSE_MODE = "repair_connection"
    SAFETY_RESPONSE_MODE = "safety_check"
    EMOTIONAL_CATEGORY_ALIASES = {
        "self_worth": "self_worth",
        "negative_self_label": "self_worth",
        "competence": "ability",
        "自我價值": "self_worth",
        "負向自我標籤": "self_worth",
        "能力否定": "ability",
        "belonging": "belonging",
        "attachment": "belonging",
        "attachment_security": "belonging",
        "rejection": "belonging",
        "rejection_fear": "belonging",
        "love_reassurance": "belonging",
        "被愛與接納": "belonging",
        "害怕被拒絕": "belonging",
        "歸屬感": "belonging",
        "safety": "safety",
        "fear": "safety",
        "fear_or_safety": "safety",
        "threat": "safety",
        "害怕與安全": "safety",
        "安全感": "safety",
        "conflict": "conflict",
        "relationship_conflict": "conflict",
        "rupture": "conflict",
        "repair": "conflict",
        "hurt_or_conflict": "conflict",
        "關係衝突": "conflict",
        "衝突修復": "conflict",
        "shame": "shame",
        "ability": "ability",
        "ability_shame": "ability",
        "guilt": "shame",
        "blame": "shame",
        "bad_child": "shame",
        "shame_or_blame": "shame",
        "羞愧與責備": "shame",
        "general": "general",
        "distress": "distress",
        "emotional_distress": "distress",
        "relationship_reassurance": "general",
        "relational_hurt": "conflict",
        "情緒困擾": "distress",
        "urgent_safety": "urgent_safety",
    }
    EMOTIONAL_SUPPORT = {
        "self_worth": {
            "micro_action": (
                "先明確說出你沒有用負面標籤看孩子，再承接他可能的難過或不安；"
                "不要解釋、辯論或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「我沒有那樣看你。你這樣問，是不是剛才有點受傷？」",
                "「這件事很難，不代表你不好；我想聽聽你怎麼了。」",
                "「你很在意我怎麼看你，對嗎？我在這裡聽。」",
                "「我不會用一個標籤決定你是誰。剛才哪裡讓你不好受？」",
            ],
            "prompts": [
                "「我沒有那樣看你。你是不是有點難過？」",
                "「你很在意我怎麼看你，對嗎？我在聽。」",
                "「這件事很難，但不代表你不好。」",
            ],
        },
        "belonging": {
            "micro_action": (
                "先明確確認愛與接納沒有消失，再承接孩子怕被推開的感受；"
                "不要用條件交換或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「我沒有不要你，我還在這裡。你是不是有點擔心？」",
                "「就算我們剛才不開心，我還是愛你，也想聽你說。」",
                "「你是不是怕我不喜歡你了？不會，我們可以慢慢說。」",
                "「你對我很重要；剛才發生什麼，讓你覺得被推開了？」",
            ],
            "prompts": [
                "「我還在這裡，也還是愛你。你是不是有點擔心？」",
                "「就算我們意見不同，你對我還是很重要。」",
                "「你是不是怕我不要你了？我想聽你說。」",
            ],
        },
        "safety": {
            "micro_action": (
                "先明確告訴孩子此刻有人陪、會一起處理，再承接他的害怕或不安；"
                "不要逼問細節或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「你現在有點害怕嗎？我在這裡，我們一起慢慢來。」",
                "「你不用一個人撐著，我會陪你。你想先說哪一點？」",
                "「我聽到你很不安心；現在我在你身邊。」",
                "「先不用急著解決，我陪你。剛才最讓你害怕的是什麼？」",
            ],
            "prompts": [
                "「我在這裡陪你，你現在是不是有點害怕？」",
                "「你不用一個人面對，我們一起慢慢來。」",
                "「我聽到你很不安心，先說給我聽。」",
            ],
        },
        "conflict": {
            "micro_action": (
                "先承認關係剛才卡住或自己的話可能傷人，明確表達仍想理解孩子，"
                "再承接感受；不要爭輸贏或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「剛才我的話可能讓你受傷了，對不起；我想聽你怎麼想。」",
                "「我們剛才都不好受，但我還在這裡，想跟你好好說。」",
                "「我先不爭誰對；你是不是覺得我沒有聽懂你？」",
                "「剛才我們卡住了。我想重新聽一次，你最在意的是什麼？」",
            ],
            "prompts": [
                "「剛才我的話讓你不好受嗎？我想重新聽。」",
                "「我先不爭對錯；你最希望我懂的是什麼？」",
                "「我們剛才卡住了，但我還想理解你。」",
            ],
        },
        "shame": {
            "micro_action": (
                "先把做錯一件事和孩子是壞的分開，明確保住孩子的價值，"
                "再承接羞愧或擔心；不要說教或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「做錯一件事，不代表你是壞孩子；你現在是不是很不好受？」",
                "「我會陪你一起想辦法，不需要先罵自己。你最擔心什麼？」",
                "「我在意的是發生了什麼，不是把你說成不好的人。」",
                "「你可以告訴我，我會先聽；犯錯也不會改變你對我的重要。」",
            ],
            "prompts": [
                "「做錯一件事，不代表你是壞孩子。」",
                "「你不用先責怪自己，我想聽發生了什麼。」",
                "「這件事可以一起處理，你對我仍然很重要。」",
            ],
        },
        "ability": {
            "micro_action": (
                "先把『現在還不會』和『孩子不行』分開，承接他的挫折，再陪他"
                "想下一小步；不要比較、說教或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「現在還不會，不代表你不行；我會陪你慢慢來。」",
                "「你已經試很久了，難怪會挫折。我們一起想下一小步。」",
                "「做不到真的不好受；你想先休息，還是要我陪你試一點？」",
                "「不用拿自己跟別人比。我想先聽你最卡在哪裡。」",
            ],
            "prompts": [
                "「現在還不會，不代表你不行；我陪你慢慢來。」",
                "「試了這麼久很挫折，對嗎？我在聽。」",
                "「你想先休息，還是要我陪你試一小步？」",
            ],
        },
        "distress": {
            "micro_action": (
                "先說出你聽見孩子正不好受，讓他知道你會留在身邊；不要急著"
                "找原因、解決或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「我聽見你現在很不好受。我在這裡陪你，慢慢說。」",
                "「你不用一個人撐著；我先陪你停一下。」",
                "「我有聽見。你想說的時候，我會好好聽。」",
                "「先不用急著解決；我在你身邊。」",
            ],
            "prompts": [
                "「我聽見你很不好受，我在這裡陪你。」",
                "「你不用一個人撐著，我會聽你慢慢說。」",
                "「先不用急著解決；我在你身邊。」",
            ],
        },
        "urgent_safety": {
            "micro_action": (
                "停止共讀、留在孩子身邊並直接確認安全；不要讓孩子獨處，"
                "若有立即危險，立刻聯絡當地緊急服務或專業支援。"
            ),
            "examples": [
                "「我很在意你剛才說的話。你現在有想傷害自己嗎？」",
            ],
            "prompts": [
                "「我會陪著你，我們現在一起找能幫忙的大人。」",
            ],
        },
        "general": {
            "micro_action": (
                "先直接回答孩子真正擔心的關係或自我價值問題，再承接他的感受；"
                "不要分析、說教或轉回教材，說完停下來聽。"
            ),
            "examples": [
                "「我聽到你很在意這件事；我先陪你，你慢慢說。」",
                "「我沒有要把你推開。你剛才是不是有點難受？」",
                "「你對我很重要；我想先聽懂現在的感受。」",
                "「先不用回到剛才的事，我在聽。哪一部分最讓你不好受？」",
            ],
            "prompts": [
                "「你對我很重要；你現在是不是有點難受？」",
                "「我先不急著說別的，你慢慢告訴我。」",
                "「我在這裡聽，哪一部分最讓你不好受？」",
            ],
        },
    }

    QUESTION_PATTERN = re.compile(
        r"[？?]|為什麼|怎麼|幹嘛|什麼|哪裡|哪個|誰|幾(?:個|隻|台|本)?|多少|"
        r"是不是|有沒有|要不要|好不好|嗎(?:[。！!\s]|$)|呢(?:[。！!\s]|$)"
    )
    LEADING_FILLER_PATTERN = re.compile(
        r"^(?:(?:嗯+|呃+|啊+|喔+|哦+|欸+|對呀|對啊|是呀|是啊|好呀|好啊)"
        r"[，,、\s]*)+"
    )
    LEADING_CONNECTOR_PATTERN = re.compile(r"^(?:因為|所以|可是|但是|然後)")
    COMMON_ASR_REPLACEMENTS = {
        "打借": "打架",
        "雪帳": "雪杖",
        "雪藏": "雪杖",
    }

    def __init__(self, history_limit=5):
        self.history_limit = history_limit

    def build(
        self,
        session,
        events,
        current_event,
        rule_analysis,
        asd_observation=None,
    ):
        # current_event is sent separately; cap the combined cloud transcript
        # to history_limit utterances rather than leaking one extra prior turn.
        prior_limit = max(0, self.history_limit - 1)
        recent_events = events[-prior_limit:] if prior_limit else []
        material = get_material(session.get("material_id"))
        if material is None:
            material = get_material_by_title(session.get("material"))

        material_profile = {
            "id": material["id"],
            "title": material["title"],
            "scene_description": material["scene_description"],
            "visible_elements": material["visible_elements"],
            "interaction_goal": material["interaction_goal"],
            "default_practice_prompt": material["practice_prompt"],
            "parent_example": material["parent_example"],
            "prompt_bank": material.get("prompt_bank", []),
            "example_bank": material.get("example_bank", []),
        } if material else {
            "id": session.get("material_id"),
            "title": session["material"],
            "scene_description": "沒有可用的教材圖片描述。",
            "visible_elements": [],
            "interaction_goal": "跟隨孩子注意力並使用簡短語句。",
            "default_practice_prompt": "你在圖片裡看到什麼？",
            "parent_example": "「我看到一個有趣的東西。」",
            "prompt_bank": ["你在圖片裡看到什麼？"],
            "example_bank": ["「我看到一個有趣的東西。」"],
        }

        recent_coach_copy = [
            {
                "message": suggestion.get("message"),
                "example": suggestion.get("example"),
                "practice_prompt": suggestion.get("practice_prompt"),
            }
            for event in recent_events[-3:]
            if isinstance(event.get("analysis"), dict)
            and isinstance(
                suggestion := event["analysis"].get("suggestion"),
                dict,
            )
        ]
        emotional_context = self._resolve_emotional_context(
            rule_analysis=rule_analysis,
            recent_events=recent_events,
            current_event=current_event,
        )
        interaction_brief = self._interaction_brief(
            material_profile=material_profile,
            recent_events=recent_events,
            recent_coach_copy=recent_coach_copy,
            current_event=current_event,
            rule_analysis=rule_analysis,
            emotional_context=emotional_context,
        )

        return {
            "material": session["material"],
            "material_profile": material_profile,
            "recent_dialogue": [
                {
                    "speaker": event["speaker"],
                    "text": event["text"],
                    "pause_before": event["pause_before"],
                    "gaze_available": event["analysis"].get(
                        "gaze_available",
                        True,
                    ),
                    "gaze_on_target": event["gaze_on_target"],
                }
                for event in recent_events
            ],
            "recent_coach_copy": recent_coach_copy,
            "current_event": current_event,
            "interaction_brief": interaction_brief,
            "emt_analysis": {
                "wait_met": rule_analysis["wait_met"],
                "wait_status": rule_analysis.get("wait_status"),
                "expansion_met": rule_analysis["expansion_met"],
                "turn_taking": rule_analysis["turn_taking"],
                "gaze_available": rule_analysis["gaze_available"],
                "gaze_on_target": rule_analysis["gaze_on_target"],
                "relationship_priority": emotional_context["active"],
                "safety_priority": (
                    emotional_context["category"] == "urgent_safety"
                ),
                "emotional_bid": (
                    {
                        "category": emotional_context["category"],
                        "source_speaker": emotional_context["source_speaker"],
                        "signals": emotional_context.get("signals", []),
                        "from_previous_turn": emotional_context[
                            "from_previous_turn"
                        ],
                    }
                    if emotional_context["active"]
                    else None
                ),
            },
            "asd_v4_observation": self._safe_asd_observation(
                asd_observation,
                relationship_mode=emotional_context["active"],
                safety_mode=(
                    emotional_context["category"] == "urgent_safety"
                ),
            ),
            "clinical_rule_suggestion": rule_analysis["suggestion"],
        }

    def _interaction_brief(
        self,
        material_profile,
        recent_events,
        recent_coach_copy,
        current_event,
        rule_analysis,
        emotional_context,
    ):
        """Turn raw telemetry into one clear, child-led coaching task.

        Small local models struggled when asked to infer the conversational
        move from a bag of metrics. This brief makes that decision explicit
        while leaving the wording to the language model.
        """

        current_speaker = str(current_event.get("speaker") or "")
        child_text = ""
        if current_speaker == "child":
            child_text = str(current_event.get("text") or "")
        else:
            for event in reversed(recent_events):
                if event.get("speaker") == "child":
                    child_text = str(event.get("text") or "")
                    break

        child_is_question = self._is_question(child_text)
        child_anchor = self._child_anchor(child_text, child_is_question)
        eyebrow = str(
            (rule_analysis.get("suggestion") or {}).get("eyebrow") or ""
        )

        if emotional_context["category"] == "urgent_safety":
            response_mode = self.SAFETY_RESPONSE_MODE
            micro_action = self.EMOTIONAL_SUPPORT["urgent_safety"][
                "micro_action"
            ]
        elif emotional_context["active"]:
            response_mode = self.RELATIONSHIP_RESPONSE_MODE
            category = emotional_context["category"]
            micro_action = self.EMOTIONAL_SUPPORT[category]["micro_action"]
        elif current_speaker == "child" and child_is_question:
            response_mode = "answer_child_question"
            micro_action = (
                "先用一個畫面裡看得到的線索回答孩子，不要把同一題問回去；"
                "回答後停一下，讓孩子接話。"
            )
        elif current_speaker == "child":
            response_mode = "expand_child_idea"
            micro_action = (
                "先承認孩子的看法，保留他的原意，再補一個看得見的畫面證據；"
                "不要糾正、反駁或替人物編故事。"
            )
        elif eyebrow == "等待時間":
            response_mode = "pause_and_wait"
            micro_action = "只提醒家長說完停三秒；現在不要再加第二個問題。"
        elif eyebrow == "語句擴展":
            response_mode = "recast_parent_turn"
            micro_action = (
                "示範如何接住孩子上一句，再加一個畫面線索；一次只示範一句。"
            )
        elif eyebrow == "共同注意":
            response_mode = "follow_child_attention"
            micro_action = (
                "先跟著孩子看的方向，再用一個短句把注意力接回目前圖片。"
            )
        elif eyebrow == "輪流互動":
            response_mode = "invite_child_turn"
            micro_action = "停止連續提問，留一輪給孩子用話、眼神或動作回應。"
        else:
            response_mode = "keep_natural_turn"
            micro_action = "沿著目前話題說一個短句，再留空白給孩子接下一輪。"

        recent_text = " ".join(
            str(value or "")
            for copy in recent_coach_copy
            for value in copy.values()
        )
        dialogue_text = " ".join(
            str(event.get("text") or "") for event in recent_events
        )
        if emotional_context["active"]:
            support = self.EMOTIONAL_SUPPORT[emotional_context["category"]]
            # A relationship bid temporarily outranks the teaching material.
            # Keeping image_focus empty and selecting from a separate phrase
            # bank prevents a model from forcing the conversation back to the
            # currently displayed picture before the child feels heard.
            image_focus = ""
            prompt_candidates = self._unused_candidates(
                support["prompts"],
                recent_text,
            )
            example_candidates = self._unused_candidates(
                support["examples"],
                recent_text,
            )
        else:
            visible_elements = list(material_profile.get("visible_elements") or [])
            image_focus = self._least_used_element(
                visible_elements,
                f"{recent_text} {dialogue_text}",
            )
            prompt_candidates = self._unused_candidates(
                material_profile.get("prompt_bank")
                or [material_profile.get("default_practice_prompt")],
                recent_text,
            )
            example_candidates = self._unused_candidates(
                material_profile.get("example_bank")
                or [material_profile.get("parent_example")],
                recent_text,
            )

        return {
            "response_mode": response_mode,
            "micro_action": micro_action,
            "relationship_priority": emotional_context["active"],
            "relationship_continues_from_previous_turn": bool(
                emotional_context["active"]
                and emotional_context["from_previous_turn"]
            ),
            "current_turn_acknowledgement_required": bool(
                emotional_context["active"]
                and emotional_context["from_previous_turn"]
                and current_speaker == "parent"
            ),
            "current_parent_utterance": (
                self._clean_utterance(current_event.get("text"))[:48]
                if current_speaker == "parent"
                else ""
            ),
            "relationship_response_state": rule_analysis.get(
                "relationship_response_state"
            ),
            "emotional_category": (
                emotional_context["category"]
                if emotional_context["active"]
                else None
            ),
            "stay_with_relationship": emotional_context["active"],
            "safety_priority": (
                emotional_context["category"] == "urgent_safety"
            ),
            "child_is_question": child_is_question,
            "child_question": self._clean_utterance(child_text)[:36]
            if child_is_question
            else "",
            "child_anchor": child_anchor,
            "image_focus": image_focus,
            "candidate_parent_examples": example_candidates[:3],
            "candidate_practice_prompts": prompt_candidates[:3],
            "avoid_repeating": [
                str(copy.get("practice_prompt") or "")[:36]
                for copy in recent_coach_copy
                if copy.get("practice_prompt")
            ],
        }

    @classmethod
    def _resolve_emotional_context(
        cls,
        rule_analysis,
        recent_events,
        current_event,
    ):
        """Resolve current or immediately previous relationship bids.

        Rule engines may annotate the parent turn directly, or only leave the
        bid on the preceding child event. Supporting both forms makes the
        priority survive persistence/refinement without guessing from broad
        emotional vocabulary in ordinary picture-book dialogue.
        """

        suggestion = rule_analysis.get("suggestion") or {}
        current_bid = cls._coerce_emotional_bid(
            rule_analysis.get("emotional_bid"),
            fallback_category=(
                rule_analysis.get("emotional_category")
                or suggestion.get("emotional_category")
            ),
        )
        previous_bid = cls._coerce_emotional_bid(
            rule_analysis.get("previous_emotional_bid")
        )
        from_previous_turn = False

        current_speaker = str(current_event.get("speaker") or "")
        if current_bid is None and previous_bid is not None:
            current_bid = previous_bid
            from_previous_turn = True

        if (
            current_bid is None
            and current_speaker == "parent"
            and recent_events
            and recent_events[-1].get("speaker") == "child"
        ):
            prior_analysis = recent_events[-1].get("analysis") or {}
            current_bid = cls._coerce_emotional_bid(
                prior_analysis.get("emotional_bid"),
                fallback_category=prior_analysis.get("emotional_category"),
            )
            from_previous_turn = current_bid is not None

        suggestion_mode = str(suggestion.get("response_mode") or "").strip()
        active = bool(
            rule_analysis.get("relationship_priority")
            or suggestion_mode
            in {cls.RELATIONSHIP_RESPONSE_MODE, cls.SAFETY_RESPONSE_MODE}
            or current_bid is not None
        )
        if not active:
            return {
                "active": False,
                "category": "general",
                "source_speaker": "",
                "from_previous_turn": False,
                "signals": [],
            }

        category = cls._normalize_emotional_category(
            (current_bid or {}).get("category")
        )
        current_text = str(current_event.get("text") or "")
        if (
            category == "distress"
            and re.search(r"害怕|好怕|很怕|不安|恐懼", current_text)
        ):
            category = "safety"
        source_speaker = str(
            (current_bid or {}).get("source_speaker") or "child"
        ).strip().lower()
        if source_speaker not in {"child", "parent"}:
            source_speaker = "child"
        if current_speaker == "parent" and source_speaker == "child":
            from_previous_turn = True
        return {
            "active": True,
            "category": category,
            "source_speaker": source_speaker,
            "from_previous_turn": from_previous_turn,
            "signals": list((current_bid or {}).get("signals") or []),
        }

    @classmethod
    def _coerce_emotional_bid(cls, value, fallback_category=None):
        if isinstance(value, dict):
            if value.get("active") is False or value.get("detected") is False:
                return None
            category = value.get("category") or fallback_category
            return {
                "category": cls._normalize_emotional_category(category),
                "source_speaker": value.get("source_speaker") or "child",
                "signals": list(value.get("signals") or []),
            }
        if value is True or fallback_category:
            return {
                "category": cls._normalize_emotional_category(fallback_category),
                "source_speaker": "child",
                "signals": [],
            }
        return None

    @classmethod
    def _normalize_emotional_category(cls, category):
        normalized = str(category or "").strip().lower().replace("-", "_")
        if normalized in cls.EMOTIONAL_CATEGORY_ALIASES:
            return cls.EMOTIONAL_CATEGORY_ALIASES[normalized]

        keyword_groups = {
            "self_worth": ("worth", "label", "價值", "標籤"),
            "ability": ("compet", "ability", "能力", "做不到", "不會"),
            "belonging": (
                "belong",
                "attach",
                "reject",
                "abandon",
                "love",
                "接納",
                "被愛",
                "拒絕",
                "拋棄",
                "歸屬",
            ),
            "safety": ("safe", "fear", "threat", "害怕", "安全", "威脅"),
            "conflict": ("conflict", "rupture", "repair", "衝突", "修復"),
            "shame": ("shame", "guilt", "blame", "羞愧", "罪惡", "責備"),
            "distress": ("distress", "情緒困擾", "難過", "痛苦"),
        }
        for canonical, keywords in keyword_groups.items():
            if any(keyword in normalized for keyword in keywords):
                return canonical
        return "general"

    @classmethod
    def _is_question(cls, text):
        return bool(cls.QUESTION_PATTERN.search(str(text or "")))

    @classmethod
    def _child_anchor(cls, text, is_question=False):
        cleaned = cls._clean_utterance(text)
        cleaned = cls.LEADING_FILLER_PATTERN.sub("", cleaned).strip()
        if is_question:
            return ""
        cleaned = cls.LEADING_CONNECTOR_PATTERN.sub("", cleaned).strip()
        return cleaned[:24]

    @staticmethod
    def _clean_utterance(text):
        cleaned = re.sub(r"\s+", "", str(text or "")).strip(
            "，,。！？!?、 "
        )
        for mistaken, corrected in ContextBuilder.COMMON_ASR_REPLACEMENTS.items():
            cleaned = cleaned.replace(mistaken, corrected)
        return cleaned

    @staticmethod
    def _least_used_element(elements, recent_text):
        normalized = [str(item).strip() for item in elements if str(item).strip()]
        if not normalized:
            return "圖片"
        return min(
            enumerate(normalized),
            key=lambda pair: (str(recent_text).count(pair[1]), pair[0]),
        )[1]

    @classmethod
    def _unused_candidates(cls, candidates, recent_text):
        normalized = [
            str(candidate).strip()
            for candidate in candidates or []
            if str(candidate or "").strip()
        ]
        if not normalized:
            return []
        recent_compact = cls._clean_utterance(recent_text)
        unused = [
            candidate
            for candidate in normalized
            if cls._clean_utterance(candidate) not in recent_compact
        ]
        return unused or normalized

    @staticmethod
    def _safe_asd_observation(
        observation,
        relationship_mode=False,
        safety_mode=False,
    ):
        if not isinstance(observation, dict):
            return {
                "status": "collecting",
                "note": "尚未累積足夠的視覺訊號。",
            }

        def safe_number(value, default=0.0):
            try:
                normalized = float(value)
            except (TypeError, ValueError, OverflowError):
                return default
            if normalized != normalized or normalized in {float("inf"), float("-inf")}:
                return default
            return normalized

        emotion_available = bool(
            observation.get(
                "emotion_available",
                observation.get("emotion_zh") is not None,
            )
        )
        blink_available = bool(
            observation.get(
                "blink_available",
                observation.get("blink_rate_per_min") is not None,
            )
        )
        eye_state_available = bool(
            observation.get(
                "classification_available",
                observation.get("eye_state_zh") is not None,
            )
        )
        emotion = (
            str(observation.get("emotion_zh") or "分析中")[:16]
            if emotion_available
            else "無可用資料"
        )
        eye_state = (
            str(observation.get("eye_state_zh") or "分析中")[:16]
            if eye_state_available
            else "資料累積中"
        )
        lower_stimulation = emotion in {
            "悲傷",
            "憤怒",
            "恐懼",
            "厭惡",
        } or eye_state in {
            "迴避",
            "輕度迴避",
            "過度掃視",
        }
        if safety_mode:
            adjustments = [
                "不要用表情、眨眼或眼動結果降低安全處理層級",
            ]
        elif relationship_mode:
            adjustments = ["一次只做一個情緒承接或關係回應"]
        else:
            adjustments = ["一次只談一個看得見的畫面細節"]
        if lower_stimulation:
            adjustments.append("放慢語速、降低刺激並增加等待時間")
        else:
            adjustments.append("維持溫和節奏並等待孩子主動回應")

        return {
            "status": str(observation.get("status") or "collecting")[:24],
            "observed_emotion": emotion,
            "blink_rate_per_min": (
                round(
                    max(
                        0.0,
                        safe_number(observation.get("blink_rate_per_min")),
                    ),
                    1,
                )
                if blink_available
                else None
            ),
            "eye_state": eye_state,
            "signal_availability": {
                "emotion": emotion_available,
                "blink": blink_available,
                "eye_state": eye_state_available,
            },
            "interaction_adjustments": adjustments,
            "non_diagnostic": True,
        }
