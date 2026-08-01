import re
import unicodedata


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9']+")


# Emotional bids need more context than a flat list of "bad words".  These
# fragments are deliberately combined into sentence structures below: a word
# such as "笨蛋" in a picture description is not, by itself, a relationship
# bid, while "你是不是覺得我是笨蛋" is.
CAREGIVER = r"(?:你們|妳們|你|妳|媽媽|媽咪|媽|爸爸|爸比|爸|阿公|阿嬤|爺爺|奶奶)"
NEGATIVE_IDENTITY_ATOM = (
    r"(?:笨蛋?|笨笨的|壞孩子|壞人|很壞|沒用(?!完)|無用|差勁|很差|很爛|"
    r"白痴|很蠢|蠢|廢物|垃圾(?!桶|車|袋|場|分類|食物|問題)|"
    r"討人厭|惹人討厭|沒救|多餘|失敗|累贅|不乖|不夠好|沒人要|"
    r"不值得被愛|不值得|沒價值|不重要|很煩|麻煩|丟臉)"
)
NEGATIVE_IDENTITY = (
    rf"(?:{NEGATIVE_IDENTITY_ATOM})"
    rf"(?:(?:還是|又|和|也|而且)?{NEGATIVE_IDENTITY_ATOM})?"
)
DISTRESS_FEELING = (
    r"(?:難過|傷心|害怕|怕|孤單|寂寞|委屈|不安|緊張|想哭|"
    r"不開心|很痛苦|受傷|生氣)"
)
ABILITY_FAILURE = (
    r"(?:不會|做不到|做不好|學不會|弄不好|搞不好|搞砸了?|"
    r"做錯了?|又錯了)"
)
CLAUSE_END = (
    r"(?=(?:了|嗎|嘛|吧|啊|呀|耶|喔|哦|呢|啦)*"
    r"(?:$|可是|但是|因為|所以|才|對不對|是不是))"
)
SAFETY_CLAUSE_END = (
    r"(?=(?:了|嗎|嘛|吧|啊|呀|耶|喔|哦|呢|啦)*"
    r"(?:$|。|可是|但是|因為|所以|才|對不對|是不是))"
)
RELATION_OBJECT_END = (
    r"(?:這個孩子|這樣)?"
    rf"{CLAUSE_END}"
)
POTENTIAL_HARM_ACTOR = (
    rf"(?:{CAREGIVER}|有人|老師|同學|親戚|叔叔|阿姨|哥哥|姐姐|照顧我的人)"
)


# These disclosures stop the picture-book exercise.  They are intentionally
# narrow, first-person structures rather than a list of alarming words: a
# story character saying "想死" must not be treated as the child's own risk.
URGENT_SAFETY_PATTERNS = (
    (
        "self_harm",
        re.compile(
            rf"我(?:現在|真的|已經|有點|一直)?(?:不想活(?:著|了|下去)?|"
            r"不要活(?:了|下去)?|想(?:要|去)?死|想自殺|要自殺|"
            r"想傷害自己|要傷害自己|想把自己弄死|死掉比較好)"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"我(?:現在|真的|已經|一直|再也|就|只|乾脆){0,2}"
            r"(?:想要|想|要|打算|準備|決定)(?:去|直接|乾脆)?"
            r"(?:殺死自己|把自己殺死|跳樓|割腕|"
            r"吞(?:下)?(?:很多|一堆)藥|吃(?:下|了)?(?:很多|一堆)藥)"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"我(?:已經|剛剛|剛才|昨天)?(?:割腕|"
            r"吞(?:了|下)?(?:很多|一堆)藥|"
            r"吃(?:了|下)?(?:很多|一堆)藥)(?:了)?"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"我(?:現在|真的|已經|一直|再也|就|只|乾脆){0,2}"
            r"(?:不想|不要|不願意)(?:再)?活(?:著|下去|在這個世界上)?"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"我(?:現在|真的|已經|一直|再也|就|只|乾脆){0,2}"
            r"(?:想要|想|要|打算|準備|決定)(?:去|直接|乾脆)?"
            r"(?:自殺|自殘|結束(?:自己(?:的)?|我的)?生命|死掉?)"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"我(?:現在|真的|已經|一直|再也|就|只|乾脆){0,2}"
            r"(?:想要|想|要|打算|準備)(?:傷害|弄傷)(?:我)?自己"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            rf"(?:沒有我|我死了)(?:大家|你們|你)?(?:會)?(?:比較|更)?好"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"(?:"
            r"我(?:快|真的)?受不了了[。]?"
            r"(?:我)?(?:想|要)(?:一睡不醒|(?:睡著後)?(?:永遠)?"
            r"(?:不要|不想|不會)?(?:再)?醒(?:來)?)|"
            r"我(?:想|要)睡(?:著|覺)(?:後)?(?:永遠)?"
            r"(?:不要|不想|不會)(?:再)?醒(?:來)?"
            r")"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            rf"我(?:活著)?(?:沒有|沒)(?:意思|意義){SAFETY_CLAUSE_END}|"
            rf"我不值得活著{SAFETY_CLAUSE_END}|"
            rf"我(?:想|要)一睡不醒{SAFETY_CLAUSE_END}|"
            rf"我(?:想|要)(?:永遠|從這個世界上)?消失(?:不見|算了|比較好)?"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"我(?:以前|之前|剛剛|有時候|最近)(?:有)?(?:"
            r"想過(?:死|自殺|傷害自己)|"
            r"想(?:死|自殺|傷害自己)過)"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"(?:如果|要是).{0,24}我(?:就|真的會|會)?"
            rf"(?:去死|自殺|不活(?:了|下去)){SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "possible_abuse",
        re.compile(
            rf"{POTENTIAL_HARM_ACTOR}(?:剛剛|昨天|今天|又|一直|常常|老是|有時候)?"
            r"(?:用.{1,8})?(?:打|揍|踢|推|掐|勒|咬|欺負)(?:了)?"
            r"(?:我|人家)"
            r"(?:(?:的)?(?:頭|臉|手|腳|身體|肚子|背|屁股))?"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "possible_abuse",
        re.compile(
            rf"我被{POTENTIAL_HARM_ACTOR}(?:剛剛|昨天|今天|又|一直|常常|老是)?"
            r"(?:用.{1,8})?(?:打|揍|踢|推|掐|勒|咬|欺負)(?:了)?"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "possible_abuse",
        re.compile(
            rf"{POTENTIAL_HARM_ACTOR}(?:逼|強迫)(?:我|人家)"
            r"(?:脫(?:衣服|褲子)|摸|碰|看)(?:我|人家|他|她|自己)?(?:的)?"
            r"(?:身體|私密處|下面|胸部|屁股)?"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "possible_abuse",
        re.compile(
            rf"{POTENTIAL_HARM_ACTOR}(?:說)?(?:要|想|會)"
            r"(?:殺|殺死|弄死)(?:我|人家)"
            rf"{SAFETY_CLAUSE_END}"
        ),
    ),
    (
        "possible_abuse",
        re.compile(
            r"(?:有人|大人|老師|同學|親戚|叔叔|阿姨|哥哥|姐姐)"
            r"(?:叫我|要我)?(?:摸|碰|看|脫)(?:我|人家)?(?:的)?"
            r"(?:身體|私密處|下面|胸部|屁股)"
        ),
    ),
    (
        "possible_abuse",
        re.compile(
            r"有人(?:摸|碰)我(?:的)?身體(?:還|而且|可是|讓我)?"
            r"(?:叫我不要說|不能說|不准說|很不舒服|很害怕|好害怕)"
        ),
    ),
    (
        "possible_abuse",
        re.compile(rf"有人(?:摸|碰)我(?:的)?身體{SAFETY_CLAUSE_END}"),
    ),
)

REPORTED_STORY_SPEECH = re.compile(
    r"(?:圖片|圖裡|圖中|畫面|故事|書裡|繪本|角色|人物).{0,16}"
    r"(?:說|問|喊|念|讀|台詞|扮演)[。]?.{0,18}$|"
    r"(?:小熊|小兔|小狗|小貓).{0,10}(?:說|問|喊|念|讀)[。]?.{0,18}$|"
    r"(?:念|讀)(?:著|的是)?台詞.{0,4}$|(?:正在|在)?扮演.{0,6}$"
)

ABILITY_NON_SHAME_PHRASES = (
    "不會傷害",
    "不會欺負",
    "不會亂丟",
    "不會把垃圾丟",
    "不會做壞事",
    "不會偷",
    "不會說謊",
    "不會壞掉",
)
COMPARISON_NON_SHAME_PHRASES = (
    "都會亂丟",
    "都會傷害",
    "都會欺負",
    "都會做壞事",
    "都會偷",
    "都會說謊",
)


EMOTIONAL_BID_PATTERNS = {
    "rejection_fear": (
        re.compile(
            rf"{CAREGIVER}(?:(?:是不是|會不會|真的|已經|再也|都|還|根本|根本就|為什麼|到底)){{0,3}}"
            rf"(?:不愛|不要|不想要|討厭|不喜歡|嫌棄|不理|不想理|不在乎|不陪)"
            rf"(?:我|人家){RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:是不是|會不會|要|想|真的會)?"
            rf"(?:離開|丟下|拋棄)(?:我|人家){RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:是不是|會不會|真的會)?把(?:我|人家)"
            rf"(?:丟下|留下|拋棄){RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:還|真的)?(?:愛|喜歡|要)(?:我|人家)"
            rf"{RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:(?:是不是|還在|真的在|是不是還在))?生(?:我|人家)的氣(?!球)"
        ),
        re.compile(
            rf"(?:是不是)?沒有人(?:愛|喜歡|要|在乎)我{RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"我(?:很|真的|有點)?怕{CAREGIVER}.{{0,4}}"
            rf"(?:不要|離開|丟下|討厭|不愛)(?:我|人家)?"
            rf"{RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:到底|還|真的)?(?:愛不愛|喜不喜歡|要不要)"
            rf"(?:我|人家){RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:是不是|真的|已經|再也|都|還|根本|根本就)?"
            rf"不想跟(?:我|人家)(?:玩|說話|講話){CLAUSE_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:只|都)(?:愛|喜歡|陪).{{1,8}}"
            rf"(?:不愛|不喜歡|不陪)(?:我|人家){RELATION_OBJECT_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:會不會|是不是|要|想|真的會)?把(?:我|人家)"
            rf"(?:丟掉|送走){RELATION_OBJECT_END}"
        ),
    ),
    "relational_hurt": (
        re.compile(
            rf"{CAREGIVER}(?:(?:剛剛|剛才|從來|一直|都|老是|又|常常)){{0,2}}"
            rf"(?:"
            rf"(?:罵|笑|兇|不理|騙|推|打|欺負|傷害)(?:我|人家){RELATION_OBJECT_END}"
            rf"|不聽(?:我|人家)(?:說(?:話)?)?{CLAUSE_END}"
            rf")"
        ),
        re.compile(
            rf"{CAREGIVER}(?:(?:根本|根本就|從來|一直|都|老是))?"
            rf"(?:不懂我|不願意聽我說|只會罵我|只會兇我)"
        ),
        re.compile(
            rf"{CAREGIVER}.{{0,8}}(?:讓我|害我)(?:真的|很|好|有點)?"
            rf"{DISTRESS_FEELING}"
        ),
        re.compile(
            rf"(?:你|妳)(?:剛剛|剛才)?(?:這樣|那樣)(?:說|講|做).{{0,4}}"
            rf"我(?:真的|很|好|有點)?"
            rf"{DISTRESS_FEELING}"
        ),
        re.compile(
            rf"我被{CAREGIVER}(?:罵|笑|兇|忽略|騙|推|打|欺負|傷害)"
        ),
        re.compile(rf"{CAREGIVER}(?:說|叫|罵|笑)(?:我|人家)(?:是|像)?{NEGATIVE_IDENTITY}"),
        re.compile(
            rf"{CAREGIVER}(?:剛剛|剛才)?叫(?:我|人家)(?:閉嘴|不要說話|安靜)"
        ),
    ),
    "self_worth": (
        re.compile(
            rf"我(?:(?:是不是|真的|就是|是|很|太|超|這麼|那麼|怎麼這麼|怎麼那麼|根本|好像|好)){{0,2}}"
            rf"(?:一個|個)?{NEGATIVE_IDENTITY}{CLAUSE_END}"
        ),
        re.compile(
            rf"我(?:覺得|感覺)(?:我|自己)?(?:真的|就是|很|太|好)?"
            rf"{NEGATIVE_IDENTITY}{CLAUSE_END}"
        ),
        re.compile(
            rf"{CAREGIVER}(?:(?:是不是|真的|也|都|還|根本|根本就|為什麼|到底)){{0,3}}"
            rf"(?:覺得|認為|以為|把)(?:我|人家)"
            rf"(?:當(?:成)?|看成|是|像|很|真的|就是)?"
            rf"{NEGATIVE_IDENTITY}{CLAUSE_END}"
        ),
        re.compile(r"(?:我是不是|是不是我)(?:永遠|什麼都)?(?:不值得|不重要|沒價值)"),
        re.compile(
            rf"{CAREGIVER}(?:(?:是不是|真的|也|都|還|根本|根本就)){{0,3}}"
            rf"(?:看不起)(?:我|人家){RELATION_OBJECT_END}"
        ),
        re.compile(rf"我(?:是不是|有沒有)?讓{CAREGIVER}(?:很|太)?失望{CLAUSE_END}"),
        re.compile(rf"{CAREGIVER}(?:是不是|有沒有)?對(?:我|人家)(?:很|太)?失望{CLAUSE_END}"),
    ),
    "ability_shame": (
        re.compile(
            rf"我(?:怎麼|為什麼)?(?:總是|老是|一直|每次都|又|什麼都|永遠|就是|都)"
            rf"{ABILITY_FAILURE}"
        ),
        re.compile(rf"我連.{{1,8}}都{ABILITY_FAILURE}"),
        re.compile(rf"別人都(?:會|做得到|做得好).{{0,6}}(?:只有)?我{ABILITY_FAILURE}"),
        re.compile(
            rf"{CAREGIVER}(?:(?:是不是|真的|也|都|還|根本|根本就)){{0,3}}"
            rf"(?:覺得|認為)(?:我|人家)"
            rf"(?:真的|總是|老是|一直|每次都|又|什麼都|永遠|就是|都)?"
            rf"{ABILITY_FAILURE}"
        ),
        re.compile(r"(?:我是不是|是不是我)(?:什麼都|一直|總是|老是)?(?:做不好|做不到|學不會|不會)"),
        re.compile(rf"(?:都|全)(?:是)?我(?:的)?(?:不好|錯誤?|害的|問題){CLAUSE_END}"),
        re.compile(r"(?:都|全)怪我(?=$|了|吧|嗎|不好|做|弄|搞|害|沒|不|太|自己)"),
        re.compile(rf"(?:為什麼|怎麼)(?:只有)?我(?:還是|總是|一直)?{ABILITY_FAILURE}"),
        re.compile(rf"(?:大家|同學|別人)都(?:會|做得到|做得好).{{0,8}}只有我{ABILITY_FAILURE}"),
        re.compile(rf"我(?:學|試|練).{{1,8}}(?:很久|好多次)(?:還是|仍然){ABILITY_FAILURE}"),
        re.compile(rf"我怎麼連.{{1,8}}都{ABILITY_FAILURE}"),
    ),
    "emotional_distress": (
        re.compile(
            rf"我(?:(?:今天|現在|真的|好|很|有點|有一點|有一些|突然|一直|心裡)){{0,3}}"
            rf"{DISTRESS_FEELING}"
        ),
        re.compile(
            rf"我(?:覺得|感覺)(?:自己|心裡)?(?:真的|好|很|有點|有一點)?"
            rf"{DISTRESS_FEELING}"
        ),
        re.compile(r"我(?:快|真的)?受不了了"),
        re.compile(
            rf"我不想(?:再)?(?:跟{CAREGIVER})?(?:說|講|玩|繼續)(?:話)?了"
        ),
        re.compile(r"我(?:好想|想要|快要|要)哭了?"),
        re.compile(rf"^好(?:難過|害怕|怕|傷心|委屈|不安|想哭)(?:了|喔|哦|啊)?$"),
    ),
}


EMOTIONAL_BID_CONFIDENCE = {
    "urgent_safety": 0.99,
    "rejection_fear": 0.96,
    "relational_hurt": 0.94,
    "self_worth": 0.94,
    "ability_shame": 0.9,
    "emotional_distress": 0.86,
}

REPAIR_CATEGORY_ALIASES = {
    "self_worth": "self_worth",
    "ability_shame": "ability_shame",
    "shame": "ability_shame",
    "belonging": "rejection_fear",
    "rejection_fear": "rejection_fear",
    "conflict": "relational_hurt",
    "relational_hurt": "relational_hurt",
    "safety": "emotional_distress",
    "general": "emotional_distress",
    "emotional_distress": "emotional_distress",
}


class EMTRuleEngine:
    TARGET_WAIT_MIN = 3.0
    TARGET_WAIT_MAX = 5.0

    def analyze(
        self,
        speaker,
        text,
        pause_before,
        gaze_on_target,
        prior_events,
        gaze_available=True,
    ):
        previous = prior_events[-1] if prior_events else None
        turn_taking = not previous or previous["speaker"] != speaker

        emotional_bid = self._empty_emotional_bid()
        relationship_phase = None
        if speaker == "child":
            emotional_bid = self._classify_emotional_bid(text)
            if emotional_bid["active"]:
                relationship_phase = "receive_bid"
            elif previous and previous.get("speaker") == "parent":
                emotional_bid = self._unanswered_bid_from_parent(previous)
                if (
                    not emotional_bid["active"]
                    and len(prior_events) >= 2
                    and prior_events[-2].get("speaker") == "child"
                ):
                    emotional_bid = self._late_refined_bid_across_parent(
                        previous,
                        prior_events[-2],
                    )
                if emotional_bid["active"]:
                    relationship_phase = "continued_bid"
        elif previous and previous["speaker"] == "child":
            emotional_bid = self._emotional_bid_from_event(previous)
            if emotional_bid["active"]:
                relationship_phase = "follow_up"

        # Once a child has made an urgent safety disclosure, the current
        # picture-book exercise remains stopped for the rest of this session.
        # A short verbal reassurance is not enough evidence that risk has
        # resolved, so every subsequent event keeps the deterministic safety
        # prompt instead of silently returning to the material.
        historical_safety = self._urgent_safety_from_history(prior_events)
        if (
            historical_safety["active"]
            and emotional_bid.get("category") != "urgent_safety"
        ):
            emotional_bid = historical_safety
            relationship_phase = "safety_follow_up"

        relationship_priority = emotional_bid["active"]
        relationship_response_state = None
        if speaker == "parent" and emotional_bid["active"]:
            if emotional_bid.get("category") == "urgent_safety":
                relationship_response_state = "safety_active"
            else:
                relationship_response_state = (
                    "addressed"
                    if self._parent_addressed_emotional_bid(
                        text,
                        emotional_bid.get("category"),
                    )
                    else "needs_repair"
                )

        wait_met = None
        wait_status = None
        expansion_met = None

        if speaker == "parent" and previous and previous["speaker"] == "child":
            # EMT's 3–5 second guidance is a *minimum opportunity* for the
            # child to respond. A longer pause may feel slow, but it must not
            # be described as the parent "rushing" the child. Keep the
            # boolean metric backwards-compatible and expose the nuance to
            # the coach copy separately.
            wait_met = pause_before >= self.TARGET_WAIT_MIN
            if pause_before < self.TARGET_WAIT_MIN:
                wait_status = "too_short"
            elif pause_before <= self.TARGET_WAIT_MAX:
                wait_status = "target"
            else:
                wait_status = "long"
            expansion_met = self._is_expansion(previous["text"], text)

        suggestion = self._build_suggestion(
            speaker=speaker,
            text=text,
            pause_before=pause_before,
            gaze_available=gaze_available,
            gaze_on_target=gaze_on_target,
            previous=previous,
            wait_met=wait_met,
            expansion_met=expansion_met,
            turn_taking=turn_taking,
            emotional_bid=emotional_bid,
            relationship_phase=relationship_phase,
        )

        return {
            "wait_met": wait_met,
            "wait_status": wait_status,
            "expansion_met": expansion_met,
            "turn_taking": turn_taking,
            "gaze_available": gaze_available,
            "gaze_on_target": gaze_on_target,
            "emotional_bid": emotional_bid,
            "relationship_priority": relationship_priority,
            "relationship_response_state": relationship_response_state,
            "suggestion": suggestion,
        }

    def summarize(self, events):
        def is_shared_reading_event(event):
            analysis = event.get("analysis") or {}
            suggestion = analysis.get("suggestion") or {}
            bid = analysis.get("emotional_bid")
            return not (
                analysis.get("relationship_priority") is True
                or (isinstance(bid, dict) and bid.get("active") is True)
                or suggestion.get("response_mode")
                in {"repair_connection", "safety_check"}
            )

        parent_responses = [
            event
            for event in events
            if is_shared_reading_event(event)
            and event["analysis"].get("wait_met") is not None
        ]
        expansions = [
            event
            for event in events
            if is_shared_reading_event(event)
            and event["analysis"].get("expansion_met") is not None
        ]
        turns = [event for event in events[1:] if is_shared_reading_event(event)]

        average_wait = (
            round(
                sum(event["pause_before"] for event in parent_responses)
                / len(parent_responses),
                1,
            )
            if parent_responses
            else 0
        )
        expansion_rate = (
            round(
                100
                * sum(bool(event["analysis"]["expansion_met"]) for event in expansions)
                / len(expansions)
            )
            if expansions
            else 0
        )
        turn_taking_rate = (
            round(
                100
                * sum(bool(event["analysis"]["turn_taking"]) for event in turns)
                / len(turns)
            )
            if turns
            else 0
        )

        return {
            "average_wait": average_wait,
            "expansion_rate": expansion_rate,
            "turn_taking_rate": turn_taking_rate,
        }

    def _is_expansion(self, child_text, parent_text):
        child_tokens = self._tokens(child_text)
        parent_tokens = self._tokens(parent_text)
        if not child_tokens or len(parent_tokens) <= len(child_tokens):
            return False

        child_set = set(child_tokens)
        overlap = len(child_set.intersection(parent_tokens)) / len(child_set)
        added_length = len(parent_tokens) - len(child_tokens)
        return overlap >= 0.6 and 1 <= added_length <= 10

    @staticmethod
    def _tokens(text):
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _empty_emotional_bid():
        return {
            "active": False,
            "category": None,
            "confidence": 0.0,
            "signals": [],
            "source_speaker": None,
        }

    @staticmethod
    def _normalize_for_emotional_bid(text):
        return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text or "")).lower()

    @staticmethod
    def _normalize_for_safety(text):
        normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
        normalized = re.sub(r"\s+", "", normalized)
        # A comma immediately after the speaker label is usually ASR
        # punctuation inside the same clause; elsewhere it is a useful clause
        # boundary (for example「我想死，你不要跟別人說」).
        normalized = re.sub(
            rf"({CAREGIVER}|我|人家)[，,、：:]+",
            r"\1",
            normalized,
        )
        normalized = re.sub(r"[，,、。！？!?；;…]+", "。", normalized)
        return re.sub(r"[「」『』\"'（）()]", "", normalized)

    def _classify_emotional_bid(self, text):
        normalized = self._normalize_for_emotional_bid(text)
        if not normalized:
            return self._empty_emotional_bid()

        safety_normalized = self._normalize_for_safety(text)
        for risk_kind, pattern in URGENT_SAFETY_PATTERNS:
            match = pattern.search(safety_normalized)
            if match is None or self._is_reported_story_speech(
                safety_normalized,
                match.start(),
            ):
                continue
            return {
                "active": True,
                "category": "urgent_safety",
                "confidence": EMOTIONAL_BID_CONFIDENCE["urgent_safety"],
                "signals": ["urgent_safety", risk_kind],
                "source_speaker": "child",
            }

        for category, patterns in EMOTIONAL_BID_PATTERNS.items():
            for index, pattern in enumerate(patterns):
                match = pattern.search(normalized)
                if match is None:
                    continue
                if self._is_reported_story_speech(normalized, match.start()):
                    continue
                if (
                    category == "ability_shame"
                    and self._is_non_shame_ability_statement(normalized)
                ):
                    continue
                return {
                    "active": True,
                    "category": category,
                    "confidence": EMOTIONAL_BID_CONFIDENCE[category],
                    "signals": [category, f"pattern_{index + 1}"],
                    "source_speaker": "child",
                }

        return self._empty_emotional_bid()

    @staticmethod
    def _is_reported_story_speech(normalized, match_start):
        prefix = normalized[max(0, match_start - 28) : match_start]
        return bool(REPORTED_STORY_SPEECH.search(prefix))

    @staticmethod
    def _is_non_shame_ability_statement(normalized):
        return any(
            phrase in normalized
            for phrase in (
                *ABILITY_NON_SHAME_PHRASES,
                *COMPARISON_NON_SHAME_PHRASES,
            )
        )

    def _emotional_bid_from_event(self, event):
        analysis = event.get("analysis") or {}
        stored = analysis.get("emotional_bid")
        if isinstance(stored, dict) and stored.get("active"):
            category = stored.get("category")
            if category in {*EMOTIONAL_BID_PATTERNS, "urgent_safety"}:
                confidence = stored.get(
                    "confidence",
                    EMOTIONAL_BID_CONFIDENCE[category],
                )
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = EMOTIONAL_BID_CONFIDENCE[category]
                return {
                    "active": True,
                    "category": category,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "signals": list(stored.get("signals") or [category]),
                    "source_speaker": "child",
                }

        # A language model is allowed to make one bounded correction when a
        # nuanced relationship bid falls outside the deterministic patterns.
        # Persist that safe escalation into the immediately following parent
        # turn instead of forgetting it and jumping back to picture coaching.
        suggestion = analysis.get("suggestion") or {}
        if suggestion.get("response_mode") == "repair_connection":
            raw_category = str(
                suggestion.get("emotional_category") or "general"
            ).strip().lower()
            category = REPAIR_CATEGORY_ALIASES.get(
                raw_category,
                "emotional_distress",
            )
            target_source = str(
                suggestion.get("coach_target_source") or "relationship_repair"
            )
            return {
                "active": True,
                "category": category,
                "confidence": 0.78,
                "signals": [target_source],
                "source_speaker": "child",
            }

        # Older stored events do not have the structured field yet.  Re-run
        # the classifier so a parent response immediately after such a child
        # utterance still receives relationship-first coaching.
        return self._classify_emotional_bid(event.get("text", ""))

    def _unanswered_bid_from_parent(self, parent_event):
        analysis = parent_event.get("analysis") or {}
        suggestion = analysis.get("suggestion") or {}
        mode = str(suggestion.get("response_mode") or "")
        if mode != "repair_connection":
            return self._empty_emotional_bid()

        inherited = self._emotional_bid_from_event(parent_event)
        if not inherited["active"] or inherited.get("category") == "urgent_safety":
            return self._empty_emotional_bid()
        if self._parent_addressed_emotional_bid(
            parent_event.get("text", ""),
            inherited.get("category"),
        ):
            return self._empty_emotional_bid()

        inherited["signals"] = list(
            dict.fromkeys(
                [
                    *inherited.get("signals", []),
                    "continued_after_missed_response",
                ]
            )
        )
        inherited["confidence"] = max(
            0.72,
            float(inherited.get("confidence") or 0.0),
        )
        return inherited

    def _late_refined_bid_across_parent(self, parent_event, child_event):
        """Recover an LLM escalation that completed after the parent spoke.

        Background refinement must not create a race where the database knows
        the child's earlier turn needed repair but the already-stored parent
        event still has an ordinary picture target.
        """

        child_analysis = child_event.get("analysis") or {}
        child_suggestion = child_analysis.get("suggestion") or {}
        if child_suggestion.get("response_mode") != "repair_connection":
            return self._empty_emotional_bid()
        inherited = self._emotional_bid_from_event(child_event)
        if not inherited["active"] or inherited.get("category") == "urgent_safety":
            return self._empty_emotional_bid()
        if self._parent_addressed_emotional_bid(
            parent_event.get("text", ""),
            inherited.get("category"),
        ):
            return self._empty_emotional_bid()
        inherited["signals"] = list(
            dict.fromkeys(
                [
                    *inherited.get("signals", []),
                    "continued_after_late_refinement",
                ]
            )
        )
        return inherited

    def _urgent_safety_from_history(self, prior_events):
        for event in reversed(prior_events or []):
            analysis = event.get("analysis") or {}
            stored = analysis.get("emotional_bid")
            if (
                isinstance(stored, dict)
                and stored.get("active")
                and stored.get("category") == "urgent_safety"
            ):
                inherited = dict(stored)
                inherited["source_speaker"] = "child"
                inherited["signals"] = list(
                    dict.fromkeys(
                        [
                            *stored.get("signals", []),
                            "session_safety_hold",
                        ]
                    )
                )
                return inherited
        return self._empty_emotional_bid()

    @classmethod
    def _parent_addressed_emotional_bid(cls, text, category):
        normalized = cls._normalize_for_emotional_bid(text)
        if re.search(
            r"才怪|騙你的|都是你的錯|都怪你|是你害的|"
            r"(?:但是|可是|但|只是|不過|然而).{0,18}"
            r"(?:沒用|很差|很笨|很煩|討厭|不要|不愛|不想理|"
            r"不在乎|都是你的錯|怪你|你害的|閉嘴)|"
            r"(?:如果|再).{0,14}(?:就)?"
            r"(?:不要你|不愛你|離開你|丟下你|打你|罵你)",
            normalized,
        ):
            return False
        response_patterns = {
            "self_worth": (
                r"(?:不|沒有|才沒有)覺得你(?:很|是)?(?:笨|壞|沒用|差)",
                r"你(?:不笨|不壞|不是壞孩子|不是沒用)",
                r"(?:不代表|不表示)你(?:不好|不行|很差)",
                r"你對我(?:很)?重要",
            ),
            "ability_shame": (
                r"(?:現在|暫時|還)不會(?:也)?沒關係",
                r"不代表你(?:不行|不好|很差)",
                r"我(?:會)?陪你(?:一起|慢慢|一步一步)?",
                r"可以慢慢(?:學|練|來)",
            ),
            "rejection_fear": (
                r"我(?:還是)?愛你",
                r"我(?:沒有|不會)不要你",
                r"我(?:沒有|不會|不)討厭你",
                r"我(?:不會)?離開你",
                r"我(?:很)?在乎你",
                r"你對我(?:很)?重要",
            ),
            "relational_hurt": (
                r"對不起",
                r"我(?:有)?聽(?:見|到)你",
                r"我會聽你(?:說)?",
                r"我想(?:重新)?(?:聽|理解)",
                r"剛才我.{0,8}讓你(?:難過|受傷|不好受)",
            ),
            "emotional_distress": (
                r"我在這裡",
                r"我(?:會)?陪你",
                r"我(?:有)?聽(?:見|到)你",
                r"你(?:可以)?慢慢說",
                r"不用一個人",
            ),
        }
        return any(
            re.search(pattern, normalized)
            for pattern in response_patterns.get(
                category,
                response_patterns["emotional_distress"],
            )
        )

    def _build_suggestion(
        self,
        speaker,
        text,
        pause_before,
        gaze_available,
        gaze_on_target,
        previous,
        wait_met,
        expansion_met,
        turn_taking,
        emotional_bid,
        relationship_phase,
    ):
        if emotional_bid.get("category") == "urgent_safety":
            return self._build_urgent_safety_suggestion(
                emotional_bid,
                phase=relationship_phase,
            )

        if emotional_bid["active"]:
            return self._build_relationship_suggestion(
                emotional_bid["category"],
                phase=relationship_phase,
            )

        if speaker == "parent" and wait_met is False:
            return {
                "tone": "coach",
                "eyebrow": "等待時間",
                "title": "先別急著接下一句",
                "message": f"剛才等了 {pause_before:.1f} 秒。這次說完先看著孩子、笑一笑，心裡慢慢數到三。",
                "example": "（保持微笑，把說話空間留給孩子）",
            }

        if speaker == "parent" and expansion_met is False and previous:
            child_words = previous["text"].strip("。！？!? ")
            return {
                "tone": "coach",
                "eyebrow": "語句擴展",
                "title": "接住他的詞，再添一點",
                "message": f"孩子剛才說了「{child_words}」。沿著他的詞，多放進一個畫面線索就好。",
                "example": f"「對，{child_words}，我也看到了。」",
            }

        if gaze_available and not gaze_on_target:
            return {
                "tone": "notice",
                "eyebrow": "共同注意",
                "title": "先跟著孩子看過去",
                "message": "先別急著把孩子拉回來。看看他在注意哪裡，再用手指把圖片線索輕輕接過去。",
                "example": "「我看到這裡有個有趣的東西。」",
            }

        if speaker == "child":
            return {
                "tone": "ready",
                "eyebrow": "孩子已回應",
                "title": "接住他，再多說一點",
                "message": f"他剛說「{text.strip('。！？!? ')}」，這很好接！重複他的詞，再自然添一個小線索。",
                "example": f"「對，{text.strip('。！？!? ')}，我也看到了。」",
            }

        if turn_taking:
            return {
                "tone": "positive",
                "eyebrow": "節奏很好",
                "title": "剛才這個來回很自然",
                "message": "就照這個舒服的節奏繼續：你說短短一句，再留一個空白給孩子接。",
                "example": "（說一句，微笑等孩子接下一輪）",
            }

        return {
            "tone": "coach",
            "eyebrow": "輪流互動",
            "title": "這一輪換孩子來",
            "message": "你已經給了不少線索，現在停一下、看著孩子，任何聲音、手勢或眼神都算回應。",
            "example": "（安靜看著孩子，等他先有動作）",
        }

    @staticmethod
    def _build_relationship_suggestion(category, phase):
        category_copy = {
            "self_worth": {
                "message": (
                    "孩子現在是在確認你怎麼看他。先清楚否定那個負面標籤，"
                    "再接住他可能受傷的感受；此刻先停在這句，說完停下來聽。"
                ),
                "example": "「不會，我不覺得你笨。你這樣問，是不是剛才有點難過？」",
            },
            "ability_shame": {
                "message": (
                    "孩子可能正因為做不到而挫折。先讓他知道『現在不會』不等於"
                    "不夠好；此刻先陪伴，說完停下來聽，不急著給答案。"
                ),
                "example": "「現在不會也沒關係，我會陪你。你是不是有點挫折？」",
            },
            "rejection_fear": {
                "message": (
                    "孩子在確認自己會不會被討厭、離開或不要。先直接給他安全感，"
                    "再接住那份擔心；此刻先停在這句，說完停下來聽。"
                ),
                "example": "「我愛你，也不會不要你。你是不是有點擔心？」",
            },
            "relational_hurt": {
                "message": (
                    "孩子正在告訴你，剛才的互動讓他受傷。先承認你有聽見，"
                    "不要急著解釋或辯解，再問他最在意的是哪一段。"
                ),
                "example": "「我聽到你剛才受傷了。你願意告訴我哪裡最難過嗎？」",
            },
            "emotional_distress": {
                "message": (
                    "孩子正在表達難過、害怕或委屈。先讓他知道你有聽見，"
                    "陪他停一下；此刻先停在這句，說完聽他接下來想說什麼。"
                ),
                "example": "「我聽見你很難過，我在這裡陪你。要不要告訴我發生什麼事？」",
            },
        }
        copy = category_copy.get(category, category_copy["emotional_distress"])
        if phase == "continued_bid":
            title = "孩子還在等你回應"
        elif phase == "follow_up":
            title = "先留在孩子的感受上"
        elif category in {"relational_hurt", "emotional_distress"}:
            title = "先接住孩子現在的感受"
        else:
            title = "先回答孩子心裡的問題"
        return {
            "tone": "notice",
            "eyebrow": "情緒接住",
            "title": title,
            "message": copy["message"],
            "example": copy["example"],
            "response_mode": "repair_connection",
        }

    @staticmethod
    def _build_urgent_safety_suggestion(emotional_bid, phase=None):
        signals = set(emotional_bid.get("signals") or [])
        possible_abuse = "possible_abuse" in signals
        if possible_abuse:
            message = (
                "先停止共讀，平靜地聽，不追問、不責怪，也不要承諾保密。"
                "確認孩子此刻是否安全，並立即找合適的專業或緊急支援。"
            )
            example = "「謝謝你告訴我，這不是你的錯。你現在安全嗎？」"
            safety_kind = "possible_abuse"
        else:
            message = (
                "先停止共讀並留在孩子身邊，用平靜、直接的話確認他現在是否"
                "想傷害自己；若有立即危險，立刻聯絡當地緊急服務或專業支援。"
            )
            example = "「你剛說不想活，我很在意。你現在有想傷害自己嗎？」"
            safety_kind = "self_harm"
        return {
            "tone": "notice",
            "eyebrow": "安全優先",
            "title": "先停下共讀，確認孩子安全",
            "message": message,
            "example": example,
            "response_mode": "safety_check",
            "safety_kind": safety_kind,
        }
