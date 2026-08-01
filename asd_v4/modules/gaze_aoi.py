import numpy as np
from collections import deque, Counter

DEFAULT_AOIS = {
    "social_left":     {"rect": (0.0, 0.0, 0.5, 1.0), "type": "social",    "label": "社交刺激區"},
    "geometric_right": {"rect": (0.5, 0.0, 1.0, 1.0), "type": "geometric", "label": "幾何刺激區"},
}
FREEVIEW_AOIS = {
    "top_left":   {"rect": (0.0,  0.0,  0.33, 0.33), "type": "corner", "label": "左上角"},
    "top_center": {"rect": (0.33, 0.0,  0.67, 0.33), "type": "center", "label": "上方中央"},
    "top_right":  {"rect": (0.67, 0.0,  1.0,  0.33), "type": "corner", "label": "右上角"},
    "mid_left":   {"rect": (0.0,  0.33, 0.33, 0.67), "type": "side",   "label": "左側"},
    "center":     {"rect": (0.33, 0.33, 0.67, 0.67), "type": "center", "label": "中央"},
    "mid_right":  {"rect": (0.67, 0.33, 1.0,  0.67), "type": "side",   "label": "右側"},
    "bot_left":   {"rect": (0.0,  0.67, 0.33, 1.0),  "type": "corner", "label": "左下角"},
    "bot_center": {"rect": (0.33, 0.67, 0.67, 1.0),  "type": "center", "label": "下方中央"},
    "bot_right":  {"rect": (0.67, 0.67, 1.0,  1.0),  "type": "corner", "label": "右下角"},
}

class GazeAOIAnalyzer:
    def __init__(self, screen_w=1280, screen_h=720, mode="geopref", window_sec=5.0, fps=30):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.mode     = mode
        self.aois     = DEFAULT_AOIS if mode == "geopref" else FREEVIEW_AOIS
        maxlen        = int(window_sec * fps)
        self.aoi_history  = deque(maxlen=maxlen)
        self.gaze_history = deque(maxlen=maxlen)
        self.total_frames = 0

    def _find_aoi(self, gx, gy):
        nx, ny = gx / self.screen_w, gy / self.screen_h
        for key, aoi in self.aois.items():
            x1, y1, x2, y2 = aoi["rect"]
            if x1 <= nx < x2 and y1 <= ny < y2:
                return key
        return None

    def update(self, gx, gy):
        self.total_frames += 1
        self.gaze_history.append((gx, gy))
        key = self._find_aoi(gx, gy)
        self.aoi_history.append(key if key else "vacant")

    def get_summary(self):
        if not self.aoi_history:
            return {"status": "no_data"}
        history = list(self.aoi_history)
        total   = len(history)
        counter = Counter(history)
        aoi_ratios    = {k: counter.get(k,0)/total for k in self.aois}
        vacant_ratio  = counter.get("vacant",0)/total
        dominant      = max(aoi_ratios, key=aoi_ratios.get)
        dominant_ratio = aoi_ratios[dominant]
        switches = sum(1 for i in range(1,len(history))
                      if history[i]!=history[i-1]
                      and history[i]!="vacant"
                      and history[i-1]!="vacant")
        gazes = np.array(list(self.gaze_history))
        dispersion = float(np.std(gazes,axis=0).mean()) if len(gazes)>1 else 0.0
        social_ratio   = aoi_ratios.get("social_left",0)
        geometric_ratio = aoi_ratios.get("geometric_right",0)
        parts = []
        label = self.aois[dominant]["label"]
        pct   = int(dominant_ratio*100)
        parts.append(f"患者目前主要注視【{label}】（佔 {pct}% 時間）")
        if self.mode == "geopref":
            if social_ratio > 0.6:
                parts.append("患者對社交刺激有明顯偏好")
            elif geometric_ratio > 0.6:
                parts.append("患者強烈偏好幾何刺激，社交注視極少")
            elif vacant_ratio > 0.4:
                parts.append("患者大量時間注視空白區域")
        else:
            t = self.aois[dominant]["type"]
            if t == "corner":   parts.append("患者視線偏向螢幕邊角，可能在迴避主要內容")
            elif t == "center": parts.append("患者視線集中於螢幕中央")
            elif t == "side":   parts.append("患者視線偏向螢幕側邊")
        if switches > 15:   parts.append(f"注意力切換頻繁（{switches}次）")
        elif switches < 3:  parts.append(f"注意力幾乎不切換（{switches}次），可能固著")
        if vacant_ratio > 0.35: parts.append(f"空白注視比例 {int(vacant_ratio*100)}%，需引導")
        llm_context = "。".join(parts) + "。"
        return {
            "dominant_label":  self.aois[dominant]["label"],
            "dominant_ratio":  round(dominant_ratio,3),
            "social_ratio":    round(social_ratio,3),
            "geometric_ratio": round(geometric_ratio,3),
            "vacant_ratio":    round(vacant_ratio,3),
            "aoi_switches":    switches,
            "dispersion_px":   round(dispersion,1),
            "llm_context":     llm_context,
        }

    def get_llm_prompt_addition(self, severity, eye_state):
        s = self.get_summary()
        gaze_info = s.get("llm_context","目前無法取得患者注視資訊。")
        return f"【患者當前狀態】\n嚴重程度：{severity}\n眼動行為：{eye_state}\n注視分析：{gaze_info}\n\n請根據以上資訊，用繁體中文給予適當的語言治療引導（60字以內，語氣溫和）："

    def reset(self):
        self.aoi_history.clear()
        self.gaze_history.clear()
        self.total_frames = 0
