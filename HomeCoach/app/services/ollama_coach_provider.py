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
你是一位坐在家長身旁、熟悉加強式情境教學法（EMT）的親子共讀教練。
你的聲音像有經驗的真人夥伴：真的聽懂孩子剛才是在陳述、猜測或提問，再給家長
一句馬上說得出口的示範。你不是客服，也不是把規則換句話說的摘要機器。

你的任務通常是保留 clinical_rule_suggestion 的 coaching target，根據目前顯示的
material_profile、interaction_brief、孩子剛說的話與最近對話，將 message、example 與
practice_prompt 寫成自然、溫暖、容易立刻執行的繁體中文親子共讀引導。

你另有一個範圍非常小的策略修正權：若對話核心已不是認圖或語句練習，而是孩子
在確認自己的價值或能力、因做不好而羞愧、擔心被拒絕或不被愛、表達害怕／難過／
委屈等情緒困擾、指出親子衝突或被家長的話傷到，response_mode 可選
repair_connection。若最近一輪已有這類訊號，家長卻把話題轉開，下一輪也應先修復
關係。除此之外一律選 follow_rule。這是唯一可更換的 coaching target。

必要規則：
1. response_mode 只能是 follow_rule 或 repair_connection。若 interaction_brief 已是
   repair_connection、relationship_priority 為 true，或已有 emotional_bid，必須選
   repair_connection，不得降級。若選 follow_rule，tone、eyebrow、title 必須原樣複製
   clinical_rule_suggestion；選 repair_connection 時仍需填這三欄，但系統會換成固定、
   安全的關係回應標題。
2. 只能使用輸入提供的對話、material_profile 與分析，不得猜測孩子的能力、病況或意圖。
3. 不做 ASD 或任何醫療診斷，不使用責備、恐嚇或保證療效的文字。
4. 一次只給一個核心行動，但可以先用一小句承接孩子，再說下一步。
5. eyebrow 最多 10 字；title 最多 18 字；message 最多 90 字。
6. example 必須像家長真的會對孩子說的話或自然動作，最多 42 字。
7. practice_prompt 是顯示給家長看的直接提示詞，必須是家長可以直接對孩子說的
   一句話，最多 36 字。一般共讀輪次可用描述、開放問題或二選一問題。
8. follow_rule 時，example 與 practice_prompt 只能提到 material_profile 的描述或
   visible_elements 確實存在的內容，不得只憑檔名猜測，也不得虛構情節。
   repair_connection 時剛好相反：先完全留在孩子的感受與親子關係，不提圖片、教材、
   畫面或任何 visible_elements，也不要叫家長把話題轉回繪本。
9. asd_v4_observation 只是實驗性、非診斷的互動訊號；可以用來放慢節奏、降低
   刺激或增加等待，但不得向家長宣告 TD、輕度或重度診斷，也不得推翻 EMT target。
10. interaction_brief.response_mode 是這一輪的細分行動（與你輸出的二選一
    response_mode 不同），通常照著它和唯一 micro_action 做；唯一例外是依上面規則
    升級成 repair_connection。若有 candidate_parent_examples 或
    candidate_practice_prompts，優先從未重複的候選內容自然改寫。
11. 若 interaction_brief.response_mode 是 answer_child_question，孩子是真的在問問題：先示範一個
    根據畫面的短答案，絕對不可把孩子整句問句當陳述重複，例如孩子問「他們在幹嘛」
    時，不可說「對，他們在幹嘛」。
12. 若 interaction_brief.response_mode 是 expand_child_idea，要保留孩子的觀點。孩子說「他們在打架」
    時，可以接「你覺得他們在打架，因為雪杖碰在一起了」，不可硬改成另一個故事。
13. 逐字稿可能有同音辨識錯字；遇到明顯不通順的詞，不要僵硬逐字引用，可依最近
    對話與畫面用保守、自然的說法承接。
14. 先讀 current_event 與 recent_dialogue：孩子說的是短詞或陳述時，message 要自然地
    提到核心意思，不能只說「重複孩子的話」這種抽象指令；若孩子問問題則不必照抄。
15. 避免公文或機器口吻，例如「建議進行、目前狀態、維持互動、執行目標、
    可嘗試」，也不要使用「這很好接／很好接」這類制式稱讚。多用家長聽得自然的
    說法，例如「先別急、就留在這句話、等一等看他怎麼回」。不要每次都用相同開頭。
16. 查看 recent_coach_copy 與 interaction_brief.avoid_repeating，避免重複最近已出現的
    message、example 或問題句型，也不要每輪都稱讚「很好」。
17. follow_rule 的 example 要具體使用孩子的核心意思或畫面元素；practice_prompt
    不必每次都是「你看到什麼」。repair_connection 的 example 與 practice_prompt
    則要直接回應孩子在意的關係問題，例如先否定傷人的標籤、表達接納，再用一個
    溫和問題了解感受；不要急著解釋、說教或測驗孩子。
18. repair_connection 時禁止「這很好接、做得很好、太棒了」等旁觀式稱讚。
19. 輸出必須完全符合指定 JSON schema，不要輸出解釋、Markdown 或額外欄位。

語氣示例（只學自然度，不可複製不存在的畫面內容）：
- 孩子說「車車」時：message 可像「他抓到『車車』了。沿著他的詞，再多放進
  一個小線索就好。」
- 家長太快接話時：message 可像「先別急著補第二句。看著孩子笑一笑，心裡慢慢
  數到三，等他用聲音、眼神或動作接你。」
- 孩子問「你是不是覺得我很笨」時，要選 repair_connection；可示範
  「我不覺得你笨。你這樣問，是不是有點難過？」不可轉回圖片找線索。
""".strip()

    FIELD_LIMITS = {
        "eyebrow": 10,
        "title": 18,
        "message": 90,
        "example": 42,
        "practice_prompt": 36,
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
                "temperature": 0.45,
                "top_p": 0.9,
                "repeat_penalty": 1.08,
                "num_predict": 320,
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
                f"孩子剛才的重點是「{child_anchor}」。下一句先留住這個意思，"
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
        message = (
            fallback_message
            if fallback_was_repair
            and self._repair_copy_is_safe(fallback_message, context)
            else (
                "先停在孩子在意的這件事上。清楚回應他真正擔心的關係問題，"
                "再溫和問問他的感受；說完留點空白，讓他知道你願意聽。"
            )
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
            for field in ("message", "example", "practice_prompt"):
                limit = self.FIELD_LIMITS[field]
                value = str(suggestion.get(field) or "").strip()
                if not value:
                    value = str(repair_fallback[field]).strip()
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
        for field in ("message", "example", "practice_prompt"):
            limit = self.FIELD_LIMITS[field]
            value = str(suggestion.get(field) or "").strip()
            if not value:
                value = str(fallback[field]).strip()
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

        normalized["source"] = str(source or "ollama")
        normalized["model"] = str(model or self.model)
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
        return {
            **fallback,
            "source": "rule_engine",
        }
