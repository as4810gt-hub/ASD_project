import os
import time, pickle
from functools import lru_cache
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
from pathlib import Path
from modules.gaze_aoi import GazeAOIAnalyzer
from modules.emotion_analyzer import EmotionAnalyzer

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]

EYE_STATE_ZH = {
    "focused": "專注",
    "avoidant": "迴避",
    "avoidant_mild": "輕度迴避",
    "hyperscanning": "過度掃視",
    "transitional": "轉換中",
    "unknown": "分析中",
}


@lru_cache(maxsize=8)
def _load_cjk_font(size):
    """Load a Traditional-Chinese-capable font on macOS/Windows/Linux."""
    candidates = [
        os.getenv("ASD_CJK_FONT"),
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_unicode_lines(frame, lines):
    """Draw Unicode text with Pillow and return the result as a BGR frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for text, position, size, bgr_color in lines:
        rgb_color = tuple(reversed(bgr_color))
        draw.text(position, text, font=_load_cjk_font(size), fill=rgb_color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _download_model(dest="models/eye/face_landmarker.task"):
    import urllib.request
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if not Path(dest).exists():
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        print(f"[EyeTracker] 下載 face_landmarker.task ...")
        urllib.request.urlretrieve(url, dest)
    return dest


class EyeTracker:
    """
    整合三個分析維度：
    1. 眼動特徵 → 三分類（TD / 輕度ASD / 重度ASD）
    2. 注視區域（AOI）→ 螢幕位置分析
    3. 情緒分析（DeepFace）→ 七種情緒
    """

    def __init__(self, model_path, window_sec=5.0, fps=30,
                 screen_w=1280, screen_h=720,
                 ivt_threshold=5.0, min_fix_ms=30.0,
                 task_model="models/eye/face_landmarker.task",
                 emotion_every_n=15):

        self.window_sec    = window_sec
        self.fps           = fps
        self.screen_w      = screen_w
        self.screen_h      = screen_h
        self.win_frames    = int(window_sec * fps)
        self.ivt_threshold = ivt_threshold
        self.min_fix_ms    = min_fix_ms

        # ── 三分類器 ──
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        self.model        = obj["model"]
        self.feature_cols = obj["feature_cols"]
        self.classes      = obj.get("classes", {0:"TD",1:"輕度ASD",2:"重度ASD"})
        print(f"[EyeTracker] 三分類器載入（{len(self.feature_cols)} 特徵）")
        print(f"[EyeTracker] 類別：{self.classes}")

        # ── Face Landmarker ──
        task_path = _download_model(task_model)
        base_opts = mp_python.BaseOptions(model_asset_path=task_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        print("[EyeTracker] FaceLandmarker 初始化完成")

        # ── AOI 分析器 ──
        self.aoi_analyzer = GazeAOIAnalyzer(
            screen_w=screen_w, screen_h=screen_h,
            mode="freeview", window_sec=window_sec, fps=fps,
        )
        print("[EyeTracker] AOI 分析器初始化完成")

        # ── 情緒分析器 ──
        self.emotion_analyzer = EmotionAnalyzer(
            detector_backend="yunet",
            window_sec=window_sec,
            fps=fps,
            analyze_every_n_frames=emotion_every_n,
        )
        print("[EyeTracker] 情緒分析器初始化完成")

        # ── 緩衝區 ──
        self.gaze_buf  = deque(maxlen=self.win_frames)
        self.dur_buf   = deque(maxlen=self.win_frames)
        self.blink_buf = deque(maxlen=self.win_frames)

        # ── I-VT 狀態 ──
        self._last_gaze      = None
        self._fixation_start = None
        self._fixation_pos   = []
        self._frame_ms       = 1000.0 / fps
        self._frame_count    = 0

        # ── 最新結果 ──
        self.latest_result = {
            "eye_state":      "unknown",
            "severity":       "unknown",
            "asd_prob":       0.0,
            "td_prob":        0.0,
            "mild_prob":      0.0,
            "severe_prob":    0.0,
            "features":       None,
            "face_found":     False,
            "aoi_summary":    {},
            "emotion":        "neutral",
            "emotion_zh":     "平靜",
            "emotion_scores": {},
            "llm_prompt":     "",
        }
        self._last_classify_time = time.time()

    # ── 工具函式 ──────────────────────────────────────────────
    def _iris_center(self, lm, indices, w, h):
        return np.array([
            np.mean([lm[i].x * w for i in indices]),
            np.mean([lm[i].y * h for i in indices])
        ])

    def _ear(self, lm, indices, w, h):
        pts = [(lm[i].x*w, lm[i].y*h) for i in indices]
        v1  = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
        v2  = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
        hz  = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
        return (v1+v2)/(2.0*hz+1e-6)

    def _update_fixation(self, gaze, frame_count):
        if self._last_gaze is None:
            self._last_gaze = gaze
            self._fixation_start = frame_count
            self._fixation_pos = [gaze]
            return
        dist = np.linalg.norm(gaze - self._last_gaze)
        if dist < self.ivt_threshold:
            self._fixation_pos.append(gaze)
        else:
            if self._fixation_start is not None and len(self._fixation_pos) >= 1:
                dur_ms = len(self._fixation_pos) * self._frame_ms
                if dur_ms >= self.min_fix_ms:
                    cx = np.mean([p[0] for p in self._fixation_pos])
                    cy = np.mean([p[1] for p in self._fixation_pos])
                    self.gaze_buf.append(np.array([cx, cy]))
                    self.dur_buf.append(dur_ms)
            self._fixation_start = frame_count
            self._fixation_pos = [gaze]
        self._last_gaze = gaze

    # ── 主要處理 ──────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        now  = time.time()
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        frame_info = {"face_found": False, "gaze": None, "blink": False}

        if result.face_landmarks:
            lm         = result.face_landmarks[0]
            left_iris  = self._iris_center(lm, LEFT_IRIS,  w, h)
            right_iris = self._iris_center(lm, RIGHT_IRIS, w, h)
            gaze       = (left_iris + right_iris) / 2

            ear      = (self._ear(lm, LEFT_EYE,  w, h) +
                        self._ear(lm, RIGHT_EYE, w, h)) / 2
            is_blink = ear < 0.20

            self.blink_buf.append(is_blink)
            self._update_fixation(gaze, self._frame_count)
            self.aoi_analyzer.update(float(gaze[0]), float(gaze[1]))
            self.latest_result["face_found"] = True
            frame_info = {"face_found": True, "gaze": gaze, "blink": is_blink}
        else:
            self.blink_buf.append(False)
            self.latest_result["face_found"] = False

        # ── 情緒分析（每 N 幀） ──
        emotion_result = self.emotion_analyzer.analyze_frame(frame)
        self.latest_result["emotion"]        = emotion_result["dominant_emotion"]
        self.latest_result["emotion_zh"]     = emotion_result["dominant_emotion_zh"]
        self.latest_result["emotion_scores"] = emotion_result.get("scores", {})

        self._frame_count += 1

        # ── 每 window_sec 做眼動分類 ──
        if now - self._last_classify_time >= self.window_sec:
            feats = self._compute_features()
            if feats:
                self._classify(feats)
            self._last_classify_time = now

        return frame_info

    def _compute_features(self):
        # Completed fixations alone can stay empty indefinitely when the user
        # looks steadily at one point. Include the active fixation so that the
        # first window can still produce a classification.
        gaze_samples = list(self.gaze_buf)
        duration_samples = list(self.dur_buf)
        if self._fixation_pos:
            active_duration = len(self._fixation_pos) * self._frame_ms
            if active_duration >= self.min_fix_ms:
                gaze_samples.append(np.mean(self._fixation_pos, axis=0))
                duration_samples.append(active_duration)

        if not gaze_samples:
            return None
        gazes = np.array(gaze_samples)
        durs  = np.array(duration_samples) if duration_samples else np.array([200.0])
        x, y  = gazes[:,0], gazes[:,1]
        amps  = (np.linalg.norm(np.diff(gazes,axis=0),axis=1)
                 if len(gazes)>1 else np.array([0.0]))
        cx, cy      = self.screen_w/2, self.screen_h/2
        dist_centre = np.sqrt((x-cx)**2+(y-cy)**2)
        dist_mean   = np.sqrt((x-x.mean())**2+(y-y.mean())**2)
        return {
            "sp_fix_count":                   len(gazes),
            "sp_fix_duration_ms_total":       float(np.sum(durs)),
            "sp_fix_duration_ms_mean":        float(np.mean(durs)),
            "sp_fix_duration_ms_var":         float(np.var(durs)),
            "sp_len_px_total":                float(np.sum(amps)),
            "sp_saccade_amplitude_px_mean":   float(np.mean(amps)),
            "sp_saccade_amplitude_px_var":    float(np.var(amps)),
            "sp_distance_to_centre_px_mean":  float(np.mean(dist_centre)),
            "sp_distance_to_centre_px_var":   float(np.var(dist_centre)),
            "sp_distance_to_sp_mean_px_mean": float(np.mean(dist_mean)),
            "sp_distance_to_sp_mean_px_var":  float(np.var(dist_mean)),
        }

    def _classify(self, features):
        X     = np.array([[features[c] for c in self.feature_cols]])
        pred  = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        td_p, mild_p, severe_p = float(proba[0]), float(proba[1]), float(proba[2])
        asd_p    = mild_p + severe_p
        severity = self.classes.get(int(pred), "unknown")
        gazes    = np.array(list(self.gaze_buf))
        disp     = float(np.std(gazes,axis=0).mean()) if len(gazes)>1 else 0
        if pred == 2:
            eye_state = "hyperscanning" if disp > 150 else "avoidant"
        elif pred == 1:
            eye_state = "transitional" if disp > 100 else "avoidant_mild"
        else:
            eye_state = "focused"

        aoi_summary  = self.aoi_analyzer.get_summary()
        emotion_ctx  = self.emotion_analyzer.get_llm_context()
        aoi_ctx      = self.aoi_analyzer.get_llm_prompt_addition(severity, eye_state)

        # 組裝完整 LLM Prompt（眼動 + AOI + 情緒）
        llm_prompt = f"""【患者當前完整狀態】
嚴重程度：{severity}
眼動行為：{eye_state}
情緒狀態：{self.latest_result['emotion_zh']}（{self.latest_result['emotion']}）
注視分析：{aoi_summary.get('llm_context', '分析中')}
情緒分析：{emotion_ctx}

請根據以上資訊，用繁體中文給予適當的語言治療引導（60字以內，語氣溫和）："""

        self.latest_result.update({
            "eye_state":   eye_state,
            "severity":    severity,
            "asd_prob":    asd_p,
            "td_prob":     td_p,
            "mild_prob":   mild_p,
            "severe_prob": severe_p,
            "features":    features,
            "aoi_summary": aoi_summary,
            "llm_prompt":  llm_prompt,
        })

        print(f"[分類] {severity:8s} | {eye_state:14s} | "
              f"情緒:{self.latest_result['emotion_zh']:3s} | "
              f"AOI:{aoi_summary.get('dominant_label','?')}")

    def get_latest_result(self):
        return self.latest_result.copy()

    def get_blink_rate(self):
        bl = list(self.blink_buf)
        ev = sum(1 for i in range(1,len(bl)) if bl[i] and not bl[i-1])
        return ev / max(len(bl)/self.fps/60.0, 1/60.0)

    def draw_overlay(self, frame):
        r = self.latest_result
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(650,160),(0,0,0),-1)
        cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
        cmap = {
            "focused":       (0,255,0),
            "avoidant":      (0,100,255),
            "avoidant_mild": (0,165,255),
            "hyperscanning": (0,0,255),
            "transitional":  (0,255,255),
            "unknown":       (128,128,128),
        }
        color = cmap.get(r["eye_state"],(128,128,128))
        classified = r.get("features") is not None
        severity = r["severity"] if classified else "分析中"
        eye_state = EYE_STATE_ZH.get(r["eye_state"], r["eye_state"])
        probability_text = (
            f"TD:{r['td_prob']:.2f}  輕度:{r['mild_prob']:.2f}  重度:{r['severe_prob']:.2f}"
            if classified else "分類機率：正在累積眼動資料"
        )

        aoi_summary = r.get("aoi_summary", {})
        if not aoi_summary.get("dominant_label"):
            aoi_summary = self.aoi_analyzer.get_summary()
        aoi_label = aoi_summary.get("dominant_label", "分析中")
        emotion_zh = r.get("emotion_zh","?")
        emotion_en = r.get("emotion","?")
        lines = [
            (f"{severity}｜{eye_state}", (10, 7), 24, color),
            (probability_text, (10, 40), 19, (255,255,255)),
            (f"注視區域：{aoi_label}", (10, 69), 19, (255,220,0)),
            (f"情緒：{emotion_zh}（{emotion_en}）", (10, 98), 19, (0,220,255)),
            (f"眨眼率：{self.get_blink_rate():.0f} 次／分鐘", (10, 127), 17, (180,180,180)),
        ]
        return _draw_unicode_lines(frame, lines)


    def finalize(self) -> dict:
        """
        Session 結束時強制輸出分類結果
        若 gaze_buf 不足，改用 aoi_analyzer 的 gaze_history 直接計算
        """
        # 先嘗試正常分類
        feats = self._compute_features()
        if feats:
            self._classify(feats)
            return self.latest_result.copy()

        # gaze_buf 不足，改用 aoi_analyzer 的 gaze_history
        gazes = np.array(list(self.aoi_analyzer.gaze_history))
        if len(gazes) < 3:
            print("[finalize] gaze 資料不足，無法分類")
            return self.latest_result.copy()

        durs = np.full(len(gazes), self._frame_ms)
        x, y = gazes[:,0], gazes[:,1]
        amps = (np.linalg.norm(np.diff(gazes,axis=0),axis=1)
                if len(gazes)>1 else np.array([0.0]))
        cx, cy = self.screen_w/2, self.screen_h/2
        dc = np.sqrt((x-cx)**2+(y-cy)**2)
        dm = np.sqrt((x-x.mean())**2+(y-y.mean())**2)

        feats = {
            "sp_fix_count":                   len(gazes),
            "sp_fix_duration_ms_total":       float(np.sum(durs)),
            "sp_fix_duration_ms_mean":        float(np.mean(durs)),
            "sp_fix_duration_ms_var":         float(np.var(durs)),
            "sp_len_px_total":                float(np.sum(amps)),
            "sp_saccade_amplitude_px_mean":   float(np.mean(amps)),
            "sp_saccade_amplitude_px_var":    float(np.var(amps)),
            "sp_distance_to_centre_px_mean":  float(np.mean(dc)),
            "sp_distance_to_centre_px_var":   float(np.var(dc)),
            "sp_distance_to_sp_mean_px_mean": float(np.mean(dm)),
            "sp_distance_to_sp_mean_px_var":  float(np.var(dm)),
        }
        print(f"[finalize] 使用全幀資料（{len(gazes)} 個 gaze 點）分類")
        self._classify(feats)
        return self.latest_result.copy()

    def release(self):
        self.landmarker.close()
