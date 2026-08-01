import json
from urllib import error, request


class OllamaCoachProvider:
    SAFETY_RESPONSE_MODE = "safety_check"
    UNSAFE_OUTPUT_TERMS = (
        "asd",
        "td",
        "自閉",
        "確診",
        "診斷",
        "輕度",
        "重度",
    )

    # The model may only make one bounded coaching-target correction: it can
    # escalate a normal picture-book turn to relationship repair. It cannot
    # select arbitrary clinical targets or downgrade an already detected
    # relationship bid.
    MODEL_RESPONSE_MODES = ("follow_rule", "repair_connection")
    REPAIR_TARGETS = {
        "child": {
            "tone": "notice",
            "eyebrow": "情緒接住",
            "title": "先回答孩子心裡的問題",
        },
        "parent": {
            "tone": "notice",
            "eyebrow": "關係回應",
            "title": "先留在孩子的感受上",
        },
    }
    REPAIR_MECHANICAL_PRAISE = (
        "這很好接",
        "很好接",
        "太棒了",
        "做得很好",
        "說得真好",
    )
    REPAIR_IMAGE_REDIRECT_TERMS = (
        "回到圖片",
        "轉回圖片",
        "拉回圖片",
        "回到畫面",
        "轉回畫面",
        "拉回畫面",
        "回到教材",
        "轉回教材",
        "拉回教材",
        "看看圖片",
        "看著圖片",
        "看圖片",
        "看看畫面",
        "看著畫面",
        "看畫面",
        "看看教材",
        "看教材",
        "看看繪本",
        "看繪本",
        "圖裡",
        "圖中",
        "畫面裡",
        "畫面中",
        "圖片裡",
        "圖片中",
    )

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "response_mode": {
                "type": "string",
                "enum": list(MODEL_RESPONSE_MODES),
            },
            "tone": {
                "type": "string",
                "enum": ["coach", "notice", "positive", "ready"],
            },
            "eyebrow": {"type": "string"},
            "title": {"type": "string"},
            "message": {"type": "string"},
            "example": {"type": "string"},
            "practice_prompt": {"type": "string"},
        },
        "required": [
            "response_mode",
            "tone",
            "eyebrow",
            "title",
            "message",
            "example",
            "practice_prompt",
        ],
        "additionalProperties": False,
    }

    SYSTEM_PROMPT = """
你是一位坐在家長身旁、熟悉加強式情境教學法（EMT）的親子共讀教練。請先真正
聽見孩子這一刻的語氣和心情，再像一位有溫度、有生活感的真人夥伴開口。文字可以
有停頓、疼惜、驚喜或一點輕鬆感；不要像客服、評量報告或規則摘要。

請根據 current_event、recent_dialogue、interaction_brief 和 material_profile，寫出
自然的繁體中文提示。message 是你對家長貼近當下的一小段提醒；example 與
practice_prompt 則是家長此刻真的說得出口的話。可以先回應孩子的感受或興趣，再
自然帶出下一步，不必把每句話壓成僵硬的單一步驟，也不要套固定句型。
material_profile 裡的 parent_example 與候選句只用來理解教材，不要原封不動照抄；
每一輪的 example 要真的接住 current_event。若此刻停下來比再說一句更自然，就用
一個簡短動作示範，不必硬塞教材描述。

請保留以下必要邊界，其餘措辭與情感表達由你自由發揮：
1. response_mode 只能是 follow_rule 或 repair_connection。通常跟隨
   clinical_rule_suggestion；若孩子正在懷疑自己的價值或能力、害怕被拒絕、表達
   難過／害怕／委屈，或親子關係剛受傷，選 repair_connection。已標記的關係修復
   不得降級。follow_rule 時複製 clinical_rule_suggestion 的 tone、eyebrow、title。
2. 只根據輸入中的對話、圖片描述與分析，不猜測孩子的能力、病況或意圖；不做 ASD
   或其他醫療診斷，不責備、恐嚇，也不保證療效。asd_v4_observation 只能用來調整
   節奏、刺激量與等待時間。
3. follow_rule 可使用 material_profile 中確實可見的內容，但不可虛構畫面。
   repair_connection 要留在孩子的感受與關係裡，不把話題拉回圖片或教材；先讓孩子
   感到「你有聽見我」，再溫柔地了解他怎麼了。
4. interaction_brief.current_turn_acknowledgement_required 為 true，表示目前已經是
   家長的新一句，但上一輪孩子的關係訊號仍未被接住。message 必須先描述家長目前
   這句的回應方式（例如已回答一部分、很快換了話題），再提醒尚未接住的孩子感受；
   不可只重述孩子上一句，否則會看起來像慢一輪。example 與 practice_prompt 才是
   建議家長接下來說的話。
5. 孩子問問題時，先給一個自然短答，不把問句當陳述照抄；孩子提出自己的看法時，
   先保留他的觀點，不急著糾正成唯一答案。逐字稿若有明顯同音錯字，可依上下文
   保守修正，不要僵硬引用。
6. 避免「建議進行、目前狀態、執行目標、可嘗試」等公文語氣，也避免空泛稱讚。
   多寫出可被感受到的回應，例如「你有看到他眼睛亮起來」、「先靠近一點，讓這句
   話慢慢落下來」、「嗯，你真的很想知道」。參考 recent_coach_copy 避免連續重複。
7. eyebrow 最多 10 字，title 最多 18 字，message 最多 120 字，example 最多 56 字，
   practice_prompt 最多 48 字。長度是上限，不是要填滿；自然比完整更重要。
8. 輸出必須完全符合指定 JSON schema，不要輸出解釋、Markdown 或額外欄位。

自然度示例（只學情緒和節奏，不可挪用不存在的內容）：
- 孩子說「車車」：message 可像「他一看到車車，聲音都亮起來了。就沿著這份興奮，
  陪他多放進一個小線索吧。」
- 家長太快接話：message 可像「先別急著把空白填滿。看著他、笑一下，讓他知道你
  還在等他的下一句。」
- 孩子問「你是不是覺得我很笨」：選 repair_connection，可說「我不覺得你笨。
  你這樣問，我有點心疼；剛才是不是哪裡讓你很難受？」
""".strip()

    FIELD_LIMITS = {
        "eyebrow": 10,
        "title": 18,
        "message": 120,
        "example": 56,
        "practice_prompt": 48,
    }

    def __init__(
        self,
        base_url,
        model,
        timeout_seconds=30,
        enabled=True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.enabled = bool(enabled)

    def generate(self, context, fallback):
        if self._context_requires_safety(context):
            return self._fallback(self._safety_fallback(context, fallback))
        enriched_fallback = self._contextual_fallback(context, fallback)
        if not self.enabled:
            return self._fallback(enriched_fallback)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "請依照以下 JSON 情境改寫教練提示：\n"
                        + json.dumps(context, ensure_ascii=False)
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "format": self.OUTPUT_SCHEMA,
            "options": {
                "temperature": 0.68,
                "top_p": 0.94,
                "repeat_penalty": 1.04,
                "num_predict": 420,
            },
            "keep_alive": "10m",
        }

        try:
            response = self._request_json("/chat", payload)
            content = response.get("message", {}).get("content", "")
            suggestion = json.loads(content)
            return self._normalize(suggestion, enriched_fallback, context)
        except (
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return self._fallback(enriched_fallback)

    def fallback(self, context, fallback):
        """Return the contextual rule copy without waiting for Ollama."""

        if self._context_requires_safety(context):
            return self._fallback(self._safety_fallback(context, fallback))
        return self._fallback(self._contextual_fallback(context, fallback))

    def health(self):
        if not self.enabled:
            return {
                "status": "disabled",
                "model": self.model,
            }

        try:
            response = self._request_json(
                "/tags",
                payload=None,
                timeout_seconds=min(self.timeout_seconds, 3),
            )
            available_models = {
                model.get("name") or model.get("model")
                for model in response.get("models", [])
            }
            return {
                "status": "ready" if self.model in available_models else "model_missing",
                "model": self.model,
            }
        except (error.URLError, TimeoutError, TypeError, ValueError):
            return {
                "status": "unavailable",
                "model": self.model,
            }

    def _request_json(self, endpoint, payload, timeout_seconds=None):
        body = None
        headers = {"Accept": "application/json"}
        method = "GET"

        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"

        http_request = request.Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers=headers,
            method=method,
        )
        with request.urlopen(
            http_request,
            timeout=timeout_seconds or self.timeout_seconds,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _contextual_fallback(self, context, fallback):
        material = context.get("material_profile") or {}
        current_event = context.get("current_event") or {}
        recent_dialogue = context.get("recent_dialogue") or []
        interaction_brief = context.get("interaction_brief") or {}
        if self._context_requires_safety(context):
            return self._safety_fallback(context, fallback)
        if self._context_requires_repair(context):
            return self._repair_fallback(context, fallback)

        visible_elements = [
            str(item).strip()
            for item in material.get("visible_elements") or []
            if str(item).strip()
        ]
        first_element = visible_elements[0] if visible_elements else "圖片"
        example_candidates = interaction_brief.get(
            "candidate_parent_examples"
        ) or material.get("example_bank") or []
        prompt_candidates = interaction_brief.get(
            "candidate_practice_prompts"
        ) or material.get("prompt_bank") or []
        parent_example = str(
            (example_candidates[0] if example_candidates else None)
            or material.get("parent_example")
            or fallback.get("example")
            or f"「我看到{first_element}了。」"
        ).strip()
        practice_prompt = str(
            (prompt_candidates[0] if prompt_candidates else None)
            or material.get("default_practice_prompt")
            or fallback.get("practice_prompt")
            or "你想先從哪裡看？"
        ).strip()

        child_text = ""
        if current_event.get("speaker") == "child":
            child_text = str(current_event.get("text") or "").strip("。！？!? ")
        if not child_text:
            for event in reversed(recent_dialogue):
                if event.get("speaker") == "child":
                    child_text = str(event.get("text") or "").strip("。！？!? ")
                    if child_text:
                        break
        child_text = child_text[:24]

        eyebrow = str(fallback.get("eyebrow") or "")
        message = str(fallback.get("message") or "")
        example = parent_example
        response_mode = str(interaction_brief.get("response_mode") or "")
        child_anchor = str(interaction_brief.get("child_anchor") or "").strip()

        if response_mode == "answer_child_question":
            message = (
                "孩子是真的在問你，不用把同一題再問回去。先用畫面裡看得到的"
                "線索回答短短一句，再停一下讓他接話。"
            )
            example = parent_example
        elif response_mode == "expand_child_idea" and child_anchor:
            message = (
                f"孩子把「{child_anchor}」說成自己的看法了。先接住這個想法，"
                "不急著糾正，再補一個畫面裡看得到的理由。"
            )
            detail = parent_example.strip("「」『』\" ")
            example = f"「你覺得{child_anchor}。{detail}」"
        elif response_mode == "recast_parent_turn" and child_anchor:
            message = (
                f"你這一句已經往下帶了；孩子上一輪的重點是「{child_anchor}」。"
                "下一句先留住這個意思，"
                "再補一個看得到的小線索，不用立刻換新問題。"
            )
            detail = parent_example.strip("「」『』\" ")
            example = f"「你覺得{child_anchor}。{detail}」"
        elif eyebrow == "等待時間":
            pause = current_event.get("pause_before")
            try:
                pause_copy = f"剛才大約等了 {float(pause):.1f} 秒。"
            except (TypeError, ValueError, OverflowError):
                pause_copy = "剛才接話快了一點。"
            message = (
                f"{pause_copy}先別急著補第二句；看著孩子笑一笑，"
                "心裡慢慢數到三，等他用聲音、眼神或動作接你。"
            )
            example = f"（指一指{first_element}，微笑等孩子先回應）"
        elif eyebrow in {"語句擴展", "孩子已回應"}:
            if child_text:
                message = (
                    f"孩子剛說「{child_text}」，這很好接！沿著他的詞，"
                    "再添一個圖片裡的小線索就好，不用把句子拉得很長。"
                )
            else:
                message = "孩子已經把這一輪接起來了。跟著他的意思，再添一個小線索就好。"
        elif eyebrow == "共同注意":
            message = (
                f"先跟著孩子現在看的方向，不急著拉回來；再指一指{first_element}，"
                "用短短一句把你們的注意力接在一起。"
            )
            example = f"「我先看到{first_element}了，你呢？」"
        elif eyebrow == "節奏很好":
            message = (
                f"剛才這一來一往很自然。就沿著{first_element}再聊一輪，"
                "你說短短一句，接著把空白留給孩子。"
            )
            example = "（看著孩子微笑，停一下等他接話）"
        elif eyebrow == "輪流互動":
            message = (
                "你已經給了不少線索，現在把這一輪留給孩子。安靜看著他，"
                "任何聲音、眼神或小動作都算是在回答。"
            )
            example = f"（停在{first_element}這裡，等孩子先有反應）"

        return {
            **fallback,
            "message": message[: self.FIELD_LIMITS["message"]],
            "example": example[: self.FIELD_LIMITS["example"]],
            "practice_prompt": practice_prompt[
                : self.FIELD_LIMITS["practice_prompt"]
            ],
            "response_mode": str(
                interaction_brief.get("response_mode")
                or fallback.get("response_mode")
                or "follow_rule"
            ),
            "coach_target_source": str(
                fallback.get("coach_target_source") or "clinical_rule"
            ),
        }

    def _repair_fallback(self, context, fallback):
        """Build copy that stays with the relationship, never the material.

        This is also the fail-closed result when a model notices a missed
        relationship bid but produces unsafe or picture-directed wording.
        """

        brief = context.get("interaction_brief") or {}
        current_event = context.get("current_event") or {}
        speaker = str(current_event.get("speaker") or "child")
        category = self._repair_category(context)
        target = self._repair_target(speaker, category)
        if str(fallback.get("title") or "") == "孩子還在等你回應":
            target["title"] = "孩子還在等你回應"
        child_text = self._latest_child_text(context)
        default_example, default_prompt = self._repair_direct_copy(
            category,
            child_text,
        )

        needs_direct_answer = category in {
            "ability",
            "ability_shame",
            "competence",
        } or any(
            term in child_text
            for term in (
                "笨",
                "壞",
                "沒用",
                "不愛",
                "愛我",
                "不要我",
                "討厭我",
                "不喜歡我",
            )
        )
        if needs_direct_answer:
            # A generic "我沒有那樣看你" still evades the child's actual
            # question. Explicit labels and rejection fears need an explicit,
            # immediate answer before a feeling question.
            example = default_example
            practice_prompt = default_prompt
        else:
            example_candidates = [
                *(brief.get("candidate_parent_examples") or []),
                fallback.get("example"),
            ]
            prompt_candidates = [
                *(brief.get("candidate_practice_prompts") or []),
                fallback.get("practice_prompt"),
            ]
            example = self._first_safe_repair_copy(
                example_candidates,
                context,
            ) or default_example
            practice_prompt = self._first_safe_repair_copy(
                prompt_candidates,
                context,
            ) or default_prompt

        fallback_was_repair = bool(
            str(fallback.get("response_mode") or "") == "repair_connection"
            or self._context_requires_repair(context)
        )
        fallback_message = str(fallback.get("message") or "").strip()
        if brief.get("current_turn_acknowledgement_required"):
            response_state = str(
                brief.get("relationship_response_state") or ""
            )
            if response_state == "addressed":
                message = (
                    "你這一句已經在回應孩子了。先別急著往下帶，讓這份回應再多停一會，"
                    "接著問問他最在意的是什麼。"
                )
            else:
                message = (
                    "你這一句很快帶到別的話題了，但孩子上一輪的感受還沒被接住。"
                    "先回頭回應他真正擔心的事，說完停一下，等他接話。"
                )
        elif (
            fallback_was_repair
            and self._repair_copy_is_safe(fallback_message, context)
        ):
            message = fallback_message
        else:
            message = (
                "先停在孩子在意的這件事上。清楚回應他真正擔心的關係問題，"
                "再溫和問問他的感受；說完留點空白，讓他知道你願意聽。"
            )

        return {
            **fallback,
            **target,
            "message": message[: self.FIELD_LIMITS["message"]],
            "example": example[: self.FIELD_LIMITS["example"]],
            "practice_prompt": practice_prompt[
                : self.FIELD_LIMITS["practice_prompt"]
            ],
            "response_mode": "repair_connection",
            "emotional_category": category,
            "coach_target_source": str(
                fallback.get("coach_target_source") or "clinical_rule"
            ),
        }

    def _safety_fallback(self, context, fallback):
        """Return fixed crisis copy without consulting a language model."""

        kind = self._safety_kind(context)
        if kind == "possible_abuse":
            message = (
                "先停止共讀，平靜聽、不責怪、不承諾保密，也不要反覆追問細節。"
                "確認孩子目前安全，並聯絡合適的兒少保護、醫療或緊急支援。"
            )
            example = "「謝謝你告訴我，這不是你的錯。你現在安全嗎？」"
            practice_prompt = "我會陪著你；我們現在一起找能保護你的大人。"
        else:
            message = (
                "先停止共讀，不要讓孩子獨處。平靜直接確認是否有傷害自己的想法、"
                "計畫或可取得的物品；若有立即危險，立刻聯絡當地緊急服務或前往急診。"
            )
            example = (
                "「你剛才說不想活，我很在意。你現在有想傷害自己，"
                "或想好怎麼做嗎？」"
            )
            practice_prompt = (
                "謝謝你告訴我；我會陪著你，我們現在一起找能幫忙的大人。"
            )
        return {
            **fallback,
            "tone": "notice",
            "eyebrow": "安全優先",
            "title": "先停下共讀，確認孩子安全",
            "message": message[: self.FIELD_LIMITS["message"]],
            "example": example[: self.FIELD_LIMITS["example"]],
            "practice_prompt": practice_prompt[
                : self.FIELD_LIMITS["practice_prompt"]
            ],
            "response_mode": self.SAFETY_RESPONSE_MODE,
            "safety_kind": kind,
            "coach_target_source": "clinical_rule",
        }

    @classmethod
    def _context_requires_safety(cls, context):
        brief = context.get("interaction_brief") or {}
        analysis = context.get("emt_analysis") or {}
        clinical = context.get("clinical_rule_suggestion") or {}
        containers = (context, brief, analysis, clinical)
        for container in containers:
            if not isinstance(container, dict):
                continue
            if (
                str(container.get("response_mode") or "")
                == cls.SAFETY_RESPONSE_MODE
                or container.get("safety_priority") is True
                or str(container.get("emotional_category") or "")
                == "urgent_safety"
            ):
                return True
            bid = container.get("emotional_bid")
            if (
                isinstance(bid, dict)
                and str(bid.get("category") or "") == "urgent_safety"
            ):
                return True
        return False

    @classmethod
    def _safety_kind(cls, context):
        brief = context.get("interaction_brief") or {}
        analysis = context.get("emt_analysis") or {}
        bid = analysis.get("emotional_bid") or {}
        signals = {
            str(value or "").strip().lower()
            for value in (
                list(bid.get("signals") or [])
                if isinstance(bid, dict)
                else []
            )
        }
        if "possible_abuse" in signals:
            return "possible_abuse"
        child_text = cls._latest_child_text(context)
        if any(
            term in child_text
            for term in (
                "打我",
                "揍我",
                "踢我",
                "摸我身體",
                "碰我身體",
                "私密處",
            )
        ):
            return "possible_abuse"
        return str(brief.get("safety_kind") or "self_harm")

    @classmethod
    def _repair_target(cls, speaker, category):
        target = dict(
            cls.REPAIR_TARGETS.get(speaker, cls.REPAIR_TARGETS["child"])
        )
        if speaker == "child" and category in {
            "conflict",
            "relational_hurt",
            "safety",
            "distress",
            "emotional_distress",
            "general",
        }:
            target["title"] = "先接住孩子現在的感受"
        elif speaker == "child" and category in {
            "ability",
            "ability_shame",
            "competence",
        }:
            target["title"] = "先接住孩子的挫折"
        return target

    @classmethod
    def _context_requires_repair(cls, context):
        brief = context.get("interaction_brief") or {}
        analysis = context.get("emt_analysis") or {}
        clinical = context.get("clinical_rule_suggestion") or {}
        containers = (context, brief, analysis, clinical)
        if any(
            str(container.get("response_mode") or "")
            == "repair_connection"
            for container in containers
            if isinstance(container, dict)
        ):
            return True
        if any(
            container.get("relationship_priority") is True
            or container.get("stay_with_relationship") is True
            for container in containers
            if isinstance(container, dict)
        ):
            return True

        for container in containers:
            if not isinstance(container, dict):
                continue
            bid = container.get("emotional_bid")
            if isinstance(bid, dict) and bid.get("active") is True:
                return True
        return False

    @classmethod
    def _repair_category(cls, context):
        brief = context.get("interaction_brief") or {}
        analysis = context.get("emt_analysis") or {}
        candidates = [
            brief.get("emotional_category"),
            brief.get("category"),
            (analysis.get("emotional_bid") or {}).get("category")
            if isinstance(analysis.get("emotional_bid"), dict)
            else None,
            analysis.get("emotional_category"),
            context.get("emotional_category"),
        ]
        explicit_category = ""
        for value in candidates:
            normalized = str(value or "").strip().lower()
            if normalized and normalized != "general":
                return normalized
            if normalized:
                explicit_category = normalized

        child_text = cls._latest_child_text(context)
        if any(
            term in child_text
            for term in (
                "做不到",
                "做不好",
                "學不會",
                "弄不好",
                "只有我不會",
            )
        ):
            return "ability"
        if any(term in child_text for term in ("笨", "沒用", "很差", "很煩")):
            return "self_worth"
        if any(
            term in child_text
            for term in (
                "不愛",
                "愛我",
                "不要我",
                "討厭我",
                "不喜歡我",
                "丟下我",
            )
        ):
            return "rejection_fear"
        if any(term in child_text for term in ("生氣", "兇我", "罵我", "傷心")):
            return "relational_hurt"
        if any(term in child_text for term in ("難過", "害怕", "不舒服", "想哭")):
            return "emotional_distress"
        return explicit_category or "general"

    @classmethod
    def _repair_direct_copy(cls, category, child_text):
        if category in {"ability", "ability_shame", "competence"}:
            return (
                "「現在還不會，不代表你不行。我會陪你慢慢來。」",
                "現在還不會，不代表你不行；我陪你慢慢來。",
            )
        if category in {"self_worth", "shame"}:
            if "笨" in child_text:
                return (
                    "「我不覺得你笨。你這樣問，是不是有點難過？」",
                    "我不覺得你笨；你剛才是不是有點難過？",
                )
            if "壞" in child_text:
                return (
                    "「我不覺得你是壞孩子。剛才是不是讓你不好受？」",
                    "我不覺得你壞；剛才是不是讓你不好受？",
                )
            if "沒用" in child_text:
                return (
                    "「我不覺得你沒用。你這樣問，是不是有點難過？」",
                    "我不覺得你沒用；你現在心裡怎麼了？",
                )
            return (
                "「我不是在說你不好。我想聽聽剛才哪裡讓你難過。」",
                "我不是在說你不好；剛才哪裡讓你難過？",
            )
        if category in {"belonging", "rejection_fear"}:
            if "不愛" in child_text or "愛我" in child_text:
                return (
                    "「我愛你，也願意聽你說。你剛才是不是有點擔心？」",
                    "我愛你，也願意聽；你剛才在擔心什麼？",
                )
            if "不要" in child_text:
                return (
                    "「我沒有不要你。我在這裡，想聽你說怎麼了。」",
                    "我沒有不要你；你願意告訴我怎麼了嗎？",
                )
            if "討厭" in child_text or "不喜歡" in child_text:
                return (
                    "「我不討厭你。我想聽聽剛才哪裡讓你受傷了。」",
                    "我不討厭你；剛才哪裡讓你不好受？",
                )
            return (
                "「我在乎你，也願意聽。你剛才是不是有點擔心？」",
                "我在乎你，也願意聽；你剛才在擔心什麼？",
            )
        if category in {"conflict", "relational_hurt"}:
            return (
                "「剛才那件事讓你不好受，對嗎？我想聽你說。」",
                "剛才哪裡讓你不好受？我會聽你說。",
            )
        if category in {"safety", "distress", "emotional_distress"}:
            if any(term in child_text for term in ("害怕", "好怕", "很怕", "不安")):
                return (
                    "「我在這裡陪你。你現在最害怕的是什麼？」",
                    "我在這裡陪你；你現在最害怕的是什麼？",
                )
            return (
                "「我在這裡陪你。剛才哪裡讓你覺得不舒服？」",
                "我在這裡陪你；剛才哪裡讓你不舒服？",
            )
        return (
            "「我有聽見。你這樣說，心裡是不是有點不好受？」",
            "我有聽見；你現在心裡是什麼感覺？",
        )

    @classmethod
    def _latest_child_text(cls, context):
        current = context.get("current_event") or {}
        if current.get("speaker") == "child":
            text = str(current.get("text") or "").strip()
            if text:
                return text
        for event in reversed(context.get("recent_dialogue") or []):
            if event.get("speaker") == "child":
                text = str(event.get("text") or "").strip()
                if text:
                    return text
        return ""

    def _first_safe_repair_copy(self, candidates, context):
        for candidate in candidates:
            value = str(candidate or "").strip()
            if self._repair_copy_is_safe(value, context):
                return value
        return ""

    def _repair_copy_is_safe(self, value, context):
        value = str(value or "").strip()
        if not value:
            return False
        compact = self._compact_copy(value).lower()
        if any(term in compact for term in self.UNSAFE_OUTPUT_TERMS):
            return False
        if any(
            self._compact_copy(term).lower() in compact
            for term in self.REPAIR_MECHANICAL_PRAISE
        ):
            return False
        if any(
            self._compact_copy(term).lower() in compact
            for term in self.REPAIR_IMAGE_REDIRECT_TERMS
        ):
            return False
        if any(term in value for term in ("圖片", "教材", "畫面", "繪本", "插圖")):
            return False

        material = context.get("material_profile") or {}
        visible_elements = [
            self._compact_copy(element).lower()
            for element in material.get("visible_elements") or []
            if len(self._compact_copy(element)) >= 2
        ]
        return not any(element in compact for element in visible_elements)

    def _normalize(
        self,
        suggestion,
        fallback,
        context=None,
        source="ollama",
        model=None,
    ):
        if not isinstance(suggestion, dict):
            raise TypeError("Ollama suggestion must be an object")

        context = context or {}
        if self._context_requires_safety(context):
            return self._fallback(self._safety_fallback(context, fallback))
        brief = context.get("interaction_brief") or {}
        raw_model_mode = suggestion.get("response_mode")
        model_mode = str(raw_model_mode or "follow_rule").strip()
        if model_mode not in self.MODEL_RESPONSE_MODES:
            return self._fallback(self._contextual_fallback(context, fallback))

        context_requires_repair = self._context_requires_repair(context)
        model_escalated_repair = (
            model_mode == "repair_connection" and not context_requires_repair
        )
        if context_requires_repair or model_mode == "repair_connection":
            repair_seed = dict(fallback)
            if model_escalated_repair:
                repair_seed["coach_target_source"] = "llm_escalation"
            repair_fallback = self._repair_fallback(context, repair_seed)
            category = self._repair_category(context)
            target = {
                field: repair_fallback[field]
                for field in ("tone", "eyebrow", "title")
            }
            normalized = {
                **target,
                "response_mode": "repair_connection",
                "emotional_category": category,
                "coach_target_source": (
                    "llm_escalation"
                    if model_escalated_repair
                    else "clinical_rule"
                ),
            }
            model_generated_fields = []
            for field in ("message", "example", "practice_prompt"):
                limit = self.FIELD_LIMITS[field]
                value = str(suggestion.get(field) or "").strip()
                if (
                    field == "message"
                    and brief.get("current_turn_acknowledgement_required")
                ):
                    value = str(repair_fallback[field]).strip()
                elif not value:
                    value = str(repair_fallback[field]).strip()
                else:
                    model_generated_fields.append(field)
                normalized[field] = value[:limit]

            if not all(
                self._repair_copy_is_safe(normalized[field], context)
                for field in ("message", "example", "practice_prompt")
            ):
                return self._fallback(repair_fallback)

            combined_copy = " ".join(
                normalized[field].lower()
                for field in ("message", "example", "practice_prompt")
            )
            if any(term in combined_copy for term in self.UNSAFE_OUTPUT_TERMS):
                return self._fallback(repair_fallback)

            normalized["source"] = str(source or "ollama")
            normalized["model"] = str(model or self.model)
            normalized["model_generated_fields"] = ",".join(
                model_generated_fields
            )
            return normalized

        # The deterministic rule engine owns the coaching target. Even if a
        # transcript contains prompt injection or the model drifts, the only
        # target change it may request is the bounded relationship-repair
        # escalation handled above.
        normalized = {
            "tone": fallback["tone"],
            "eyebrow": fallback["eyebrow"],
            "title": fallback["title"],
            "response_mode": str(
                brief.get("response_mode")
                or fallback.get("response_mode")
                or "follow_rule"
            ),
            "coach_target_source": "clinical_rule",
        }
        model_generated_fields = []
        for field in ("message", "example", "practice_prompt"):
            limit = self.FIELD_LIMITS[field]
            value = str(suggestion.get(field) or "").strip()
            if not value:
                value = str(fallback[field]).strip()
            else:
                model_generated_fields.append(field)
            normalized[field] = value[:limit]

        combined_copy = " ".join(
            normalized[field].lower()
            for field in ("message", "example", "practice_prompt")
        )
        if any(term in combined_copy for term in self.UNSAFE_OUTPUT_TERMS):
            return self._fallback(fallback)

        if brief.get("response_mode") == "answer_child_question":
            child_question = str(
                brief.get("child_question")
                or (context.get("current_event") or {}).get("text")
                or ""
            )
            compact_question = self._compact_copy(child_question)
            compact_message = self._compact_copy(normalized["message"])
            compact_example = self._compact_copy(normalized["example"])
            if compact_question and compact_question in compact_example:
                normalized["example"] = fallback["example"]
                if "example" in model_generated_fields:
                    model_generated_fields.remove("example")
            bad_echo_directives = (
                "跟著他說",
                "跟著孩子說",
                "重複他的話",
                "重複孩子",
                "照著他說",
                "再說一次",
            )
            if (
                compact_question
                and compact_question in compact_message
                and any(
                    self._compact_copy(directive) in compact_message
                    for directive in bad_echo_directives
                )
            ):
                normalized["message"] = fallback["message"]
                if "message" in model_generated_fields:
                    model_generated_fields.remove("message")

        child_anchor = str(brief.get("child_anchor") or "")
        raw_child_text = str(
            (context.get("current_event") or {}).get("text") or ""
        )
        compact_anchor = self._compact_copy(child_anchor)
        compact_raw = self._compact_copy(raw_child_text)
        # ContextBuilder conservatively fixes a few obvious domain-specific
        # ASR homophones (for example 雪帳→雪杖). Do not let a weaker model
        # reintroduce the malformed raw transcript into parent-facing copy.
        if compact_anchor and compact_raw and compact_anchor != compact_raw:
            anchor_bigrams = {
                compact_anchor[index : index + 2]
                for index in range(max(0, len(compact_anchor) - 1))
            }
            malformed_bigrams = {
                compact_raw[index : index + 2]
                for index in range(max(0, len(compact_raw) - 1))
            } - anchor_bigrams
            for field in ("message", "example", "practice_prompt"):
                compact_field = self._compact_copy(normalized[field])
                if compact_raw in compact_field or any(
                    bigram in compact_field for bigram in malformed_bigrams
                ):
                    normalized[field] = fallback[field]
                    if field in model_generated_fields:
                        model_generated_fields.remove(field)

        recent_prompts = {
            self._compact_copy(copy.get("practice_prompt"))
            for copy in context.get("recent_coach_copy") or []
            if isinstance(copy, dict) and copy.get("practice_prompt")
        }
        if (
            self._compact_copy(normalized["practice_prompt"])
            in recent_prompts
        ):
            normalized["practice_prompt"] = fallback["practice_prompt"]
            if "practice_prompt" in model_generated_fields:
                model_generated_fields.remove("practice_prompt")

        # Catalog examples are scaffolding, not live coaching copy. Small
        # models otherwise repeat the first material sentence every session.
        material = context.get("material_profile") or {}
        current_event = context.get("current_event") or {}
        catalog_example = self._compact_copy(material.get("parent_example"))
        if (
            current_event.get("speaker") == "parent"
            and catalog_example
            and self._compact_copy(normalized["example"]) == catalog_example
        ):
            normalized["example"] = str(
                fallback.get("example")
                or "（看著孩子，停一下等他接話）"
            ).strip()[: self.FIELD_LIMITS["example"]]
            if "example" in model_generated_fields:
                model_generated_fields.remove("example")

        recent_examples = {
            self._compact_copy(copy.get("example"))
            for copy in context.get("recent_coach_copy") or []
            if isinstance(copy, dict) and copy.get("example")
        }
        if self._compact_copy(normalized["example"]) in recent_examples:
            replacement = str(fallback.get("example") or "").strip()
            if self._compact_copy(replacement) in recent_examples:
                replacement = "（先停一下，看著孩子等他接話）"
            normalized["example"] = replacement[: self.FIELD_LIMITS["example"]]
            if "example" in model_generated_fields:
                model_generated_fields.remove("example")

        normalized["source"] = str(source or "ollama")
        normalized["model"] = str(model or self.model)
        normalized["model_generated_fields"] = ",".join(
            model_generated_fields
        )
        return normalized

    @staticmethod
    def _compact_copy(value):
        return "".join(
            character
            for character in str(value or "")
            if not character.isspace()
            and character not in "，,。！？!?、；;：:「」『』\"'（）()"
        )

    def _fallback(self, fallback):
        normalized = {
            **fallback,
            "source": "rule_engine",
            "model_generated_fields": "",
        }
        normalized.pop("model", None)
        return normalized
