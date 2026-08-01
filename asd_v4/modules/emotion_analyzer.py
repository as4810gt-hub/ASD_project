import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from collections import deque, Counter

# 情緒中文對照
EMOTION_ZH = {
    'happy':    '開心',
    'neutral':  '平靜',
    'sad':      '悲傷',
    'angry':    '憤怒',
    'surprise': '驚訝',
    'fear':     '恐懼',
    'disgust':  '厭惡',
}

# 情緒對 ASD 治療的臨床意義
EMOTION_STRATEGY = {
    'happy':    '患者情緒正向，可趁機推進互動任務或正向強化',
    'neutral':  '患者情緒平穩，適合進行結構化語言引導',
    'sad':      '患者出現悲傷情緒，需給予情感支持與安撫',
    'angry':    '患者出現憤怒情緒，需先降低刺激並穩定情緒',
    'surprise': '患者對刺激有反應，可善用此時機引導注意力',
    'fear':     '患者出現恐懼情緒，需立即減少刺激並給予安全感',
    'disgust':  '患者出現厭惡反應，需調整互動方式或刺激內容',
}


class EmotionAnalyzer:
    def __init__(self, detector_backend='yunet',
                 window_sec=5.0, fps=30,
                 analyze_every_n_frames=15):
        self.detector_backend     = detector_backend
        self.analyze_every        = analyze_every_n_frames
        self.window_sec           = window_sec
        self.fps                  = fps
        maxlen                    = int(window_sec * fps / analyze_every_n_frames)
        self.emotion_history      = deque(maxlen=maxlen)
        self.score_history        = deque(maxlen=maxlen)
        self._frame_count         = 0
        self._deepface_loaded     = False
        self._DeepFace            = None
        self.latest_emotion       = 'neutral'
        self.latest_scores        = {e: 0.0 for e in EMOTION_ZH}
        self.latest_scores['neutral'] = 100.0
        self.analysis_available   = False
        self.latest_error         = None

    def _lazy_load(self):
        if not self._deepface_loaded:
            from deepface import DeepFace
            self._DeepFace        = DeepFace
            self._deepface_loaded = True

    def analyze_frame(self, frame) -> dict:
        self._frame_count += 1
        if self._frame_count % self.analyze_every != 0:
            return self._current_result()

        self._lazy_load()
        try:
            result = self._DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend=self.detector_backend,
                silent=True,
            )
            r = result[0]
            emotion = r['dominant_emotion']
            scores  = r['emotion']
            self.latest_emotion = emotion
            self.latest_scores  = scores
            self.emotion_history.append(emotion)
            self.score_history.append(scores)
            self.analysis_available = True
            self.latest_error = None
        except Exception as e:
            # A failed/no-face frame is not evidence of a neutral emotion.
            # Keep the last valid result, but expose availability so callers
            # can display "no data" instead of a fabricated neutral result.
            self.analysis_available = False
            self.latest_error = str(e).strip() or e.__class__.__name__

        return self._current_result()

    def _current_result(self) -> dict:
        zh = EMOTION_ZH.get(self.latest_emotion, self.latest_emotion)
        strategy = EMOTION_STRATEGY.get(self.latest_emotion, '')
        return {
            'dominant_emotion':    self.latest_emotion,
            'dominant_emotion_zh': zh,
            'scores':              self.latest_scores,
            'strategy':            strategy,
            'llm_context':         f'患者當前情緒：{zh}（{self.latest_emotion}）。{strategy}。',
            'available':           self.analysis_available,
            'error':               self.latest_error,
        }

    def get_summary(self) -> dict:
        if not self.emotion_history:
            strategy = EMOTION_STRATEGY['neutral']
            return {
                'dominant_emotion': 'neutral',
                'dominant_emotion_zh': '平靜',
                'emotion_ratios': {
                    emotion: 1.0 if emotion == 'neutral' else 0.0
                    for emotion in EMOTION_ZH
                },
                'strategy': strategy,
                'llm_context': f'患者當前情緒：平靜（neutral）。{strategy}。',
            }
        counter   = Counter(self.emotion_history)
        dominant  = counter.most_common(1)[0][0]
        zh        = EMOTION_ZH.get(dominant, dominant)
        strategy  = EMOTION_STRATEGY.get(dominant, '')
        ratios    = {e: counter.get(e,0)/len(self.emotion_history) for e in EMOTION_ZH}
        return {
            'dominant_emotion':    dominant,
            'dominant_emotion_zh': zh,
            'emotion_ratios':      {k: round(v,3) for k,v in ratios.items()},
            'strategy':            strategy,
            'llm_context':         f'患者當前情緒：{zh}（{dominant}）。{strategy}。',
        }

    def get_llm_context(self) -> str:
        s = self.get_summary()
        return s['llm_context']
