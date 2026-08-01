"""Trusted metadata for the picture-based parent-child practice materials."""

from copy import deepcopy


DEFAULT_MATERIAL_ID = "176"

MATERIALS = {
    "162": {
        "id": "162",
        "filename": "162.png",
        "title": "笑臉寶寶與玩具",
        "session_label": "看圖共讀：寶寶玩玩具",
        "subtitle": "跟著寶寶的表情與手上物品說一說",
        "alt_text": "一位微笑的寶寶坐在遊戲區，雙手拿著橘色玩具。",
        "scene_description": (
            "一位微笑的寶寶坐在遊戲區，雙手拿著橘色玩具；"
            "旁邊有藍色遊戲圍欄與方形積木。"
        ),
        "visible_elements": ["寶寶", "笑臉", "橘色玩具", "藍色遊戲圍欄", "積木"],
        "interaction_goal": "命名寶寶的表情、手上物品與動作，等待孩子回應後再多加一個詞。",
        "practice_prompt": "寶寶手上拿著什麼？",
        "parent_example": "「寶寶笑笑，拿著橘色玩具。」",
        "prompt_bank": [
            "寶寶手上拿著什麼？",
            "寶寶笑咪咪地看過來。",
            "你還看到哪些顏色？",
            "橘色玩具上有什麼？",
            "玩具是橘色還是藍色？",
            "積木在寶寶後面還是前面？",
        ],
        "example_bank": [
            "「寶寶笑笑，拿著橘色玩具。」",
            "「寶寶用兩隻手摸玩具。」",
            "「橘色玩具上有一顆顆圓點。」",
            "「後面有藍色和綠色積木。」",
            "「藍色圍欄在寶寶旁邊。」",
            "「寶寶穿著灰色上衣和紅色褲子。」",
        ],
    },
    "176": {
        "id": "176",
        "filename": "176.png",
        "title": "雪地裡的兩個人",
        "session_label": "看圖共讀：雪地活動",
        "subtitle": "一起找找人物、雪地、樹木與雪杖",
        "alt_text": "兩名成人站在積雪的樹林裡，手持雪杖並將雪杖交叉。",
        "scene_description": (
            "兩名成人站在積雪的樹林裡，兩人手持雪杖並將雪杖交叉；"
            "地面有白雪，四周有高大的樹。"
        ),
        "visible_elements": ["兩個人", "白雪", "樹林", "雪杖", "雪鞋"],
        "interaction_goal": "帶孩子觀察人物正在做的動作、雪地環境與交叉的雪杖。",
        "practice_prompt": "他們在雪地裡做什麼？",
        "parent_example": "「兩根長長的雪杖交叉了。」",
        "prompt_bank": [
            "他們在雪地裡做什麼？",
            "雪地裡站著兩個人。",
            "你在雪地裡還看到什麼？",
            "他們手上拿著什麼？",
            "雪是白色還是綠色？",
            "雪杖是直直的還是交叉的？",
        ],
        "example_bank": [
            "「兩根長長的雪杖交叉了。」",
            "「兩個人站在白白的雪地上。」",
            "「高高的樹長在他們旁邊。」",
            "「左邊的人穿深藍色外套。」",
            "「右邊的人穿綠色外套。」",
            "「他們腳上踩著雪鞋。」",
        ],
    },
    "189": {
        "id": "189",
        "filename": "189.png",
        "title": "入口前的車子與行人",
        "session_label": "看圖共讀：車子去哪裡",
        "subtitle": "從車子、人群和建築入口開始聊",
        "alt_text": "一座大型建築入口前停著數台黑色汽車，附近有多名行人。",
        "scene_description": (
            "大型建築入口前的停車區停著數台黑色汽車，入口附近有多名行人；"
            "屋頂上方可以看見旗幟。"
        ),
        "visible_elements": ["黑色汽車", "行人", "建築入口", "旗幟", "停車區"],
        "interaction_goal": "和孩子輪流找車子與人物，練習位置詞、數量詞和簡短推測。",
        "practice_prompt": "你找到幾台黑色車子？",
        "parent_example": "「車子停在大門前面。」",
        "prompt_bank": [
            "你找到幾台黑色車子？",
            "車子停在建築入口前面。",
            "屋頂上方有幾面旗子？",
            "入口附近還有哪些人？",
            "車子是黑色還是紅色？",
            "你先看到車子還是大門？",
        ],
        "example_bank": [
            "「車子停在大門前面。」",
            "「三台黑色車子排在前面。」",
            "「建築入口在車子的後面。」",
            "「屋頂上方飄著旗子。」",
            "「停車區裡有好幾個人。」",
            "「中間的黑色車子高高的。」",
        ],
    },
    "229": {
        "id": "229",
        "filename": "229.png",
        "title": "街上的兩個人",
        "session_label": "看圖共讀：街上的人物",
        "subtitle": "看看人物的動作、表情和周圍街景",
        "alt_text": "兩名年輕人站在街道旁，其中一人拿著紙板，另一人比出 V 手勢。",
        "scene_description": (
            "兩名年輕人站在街道旁，其中一人拿著寫有文字的紙板並張口大笑，"
            "另一人看向鏡頭、比出 V 手勢；後方有車子、行人與紅綠燈。"
        ),
        "visible_elements": ["兩個人", "紙板", "V 手勢", "車子", "紅綠燈"],
        "interaction_goal": "描述人物可直接看見的表情和動作，再邀請孩子模仿或選擇。",
        "practice_prompt": "你想學哪一個人的動作？",
        "parent_example": "「他舉起兩根手指，像一個 V。」",
        "prompt_bank": [
            "你想學哪一個人的動作？",
            "左邊的人拿著一塊紙板。",
            "右邊的人舉起兩根手指。",
            "街道後面還有什麼？",
            "紙板是棕色還是藍色？",
            "右邊的人穿格子衣還是紅衣？",
        ],
        "example_bank": [
            "「他舉起兩根手指，像一個 V。」",
            "「街上有兩個人站在前面。」",
            "「左邊的人拿著棕色紙板。」",
            "「右邊的人穿著格子外套。」",
            "「紙板上有黑色的大字。」",
            "「後面有車子、行人和紅綠燈。」",
        ],
    },
    "274": {
        "id": "274",
        "filename": "274.png",
        "title": "大建築前的人們",
        "session_label": "看圖共讀：大建築前的人們",
        "subtitle": "找找磚牆、窗戶、陽台與參觀的人",
        "alt_text": "幾個人站在一座大型磚石建築前，建築有窗戶、陽台與幾何裝飾。",
        "scene_description": (
            "幾個人站在大型磚石建築前，建築有不同形狀的窗戶、陽台與裝飾；"
            "天空晴朗，建築前方有綠色樹叢。"
        ),
        "visible_elements": ["幾個人", "磚石建築", "窗戶", "陽台", "樹叢"],
        "interaction_goal": "跟孩子輪流尋找建築細節，使用大小、形狀與位置詞擴展語句。",
        "practice_prompt": "這棟大房子哪裡最特別？",
        "parent_example": "「高高的牆上有好多窗戶。」",
        "prompt_bank": [
            "這棟大房子哪裡最特別？",
            "高高的建築有好多窗戶。",
            "你找到哪些形狀的窗戶？",
            "牆上還有哪些花紋？",
            "建築是高高的還是矮矮的？",
            "牆是磚色還是藍色？",
        ],
        "example_bank": [
            "「高高的牆上有好多窗戶。」",
            "「大建築有高高的磚牆。」",
            "「左邊有一個小陽台。」",
            "「牆上有方形和長長的窗戶。」",
            "「建築前面有綠色樹叢。」",
            "「幾個人站在建築前面。」",
        ],
    },
}


def get_material(material_id=None):
    """Return a copy so request-specific changes cannot mutate the catalog."""
    key = str(material_id or DEFAULT_MATERIAL_ID).strip()
    material = MATERIALS.get(key)
    return deepcopy(material) if material else None


def get_material_by_title(title):
    normalized = str(title or "").strip()
    for material in MATERIALS.values():
        if normalized in {material["title"], material["session_label"]}:
            return deepcopy(material)
    return None


def list_materials():
    return [deepcopy(material) for material in MATERIALS.values()]


def allowed_filenames():
    return {material["filename"] for material in MATERIALS.values()}
