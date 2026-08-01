# 🧠 ASD 眼動追蹤 + 情緒辨識系統 v4

## 📋 概述

這是一套整合式系統，用於**自閉症譜系障礙 (ASD) 患者**的眼動追蹤與情緒分析。系統在實時視頻中檢測患者的眼部運動、分類 ASD 嚴重程度，並辨識患者的情緒狀態，最終生成臨床治療建議。

### 系統特色
- 🎥 **實時眼動追蹤**：使用 MediaPipe Face Landmarker 進行高精度面部檢測
- 🧪 **三分類 ASD 診斷**：區分 TD（正常發展）、輕度 ASD、重度 ASD
- 😊 **七種情緒辨識**：使用 DeepFace 進行實時情緒分類
- 👁️ **注視區域分析 (AOI)**：追蹤患者在螢幕上的注意力位置
- 📊 **LLM 友善輸出**：自動生成臨床治療建議提示詞

---

## 🗂️ 資料夾結構

```
asd_v4/
├── run.py                  # 主啟動程式
├── run.bat                 # Windows 快速啟動批次檔
├── requirements.txt        # Python 依賴清單
├── README.md              # 本檔案
│
├── modules/               # 核心模組
│   ├── eye_tracker.py     # 眼動追蹤與 ASD 分類核心
│   ├── emotion_analyzer.py # 情緒辨識與治療策略
│   ├── gaze_aoi.py        # 注視區域分析
│   └── __pycache__/       # Python 快取（自動生成）
│
└── models/                # 預訓練模型
    └── eye/
        ├── face_landmarker.task      # MediaPipe 人臉檢測模型 (3.7 MB)
        └── severity_classifier.pkl   # ASD 分類模型 (scikit-learn)
```

---

## 📦 系統需求

### Python 版本
- **Python 3.10+**（推薦 3.13）
- Windows / macOS / Linux

### 硬體需求
- 📷 **攝影機**：支援 USB 攝影機或內建鏡頭
- 💻 **CPU**：Intel i5 或同級以上
- 📌 **RAM**：4GB 最低，8GB 推薦
- 🎮 **GPU**：可選（會自動加速 TensorFlow）

---

## 🛠️ 依賴套件

所有依賴已列在 `requirements.txt` 中。主要套件包括：

| 套件 | 版本 | 用途 |
|------|------|------|
| `opencv-python` | 最新 | 視頻捕捉與影像處理 |
| `numpy` | 最新 | 數值運算 |
| `mediapipe` | 1.0.0+ | 人臉與虹膜偵測 |
| `scikit-learn` | 1.9.0+ | ASD 分類模型 |
| `xgboost` | 最新 | 提升模型效能 |
| `deepface` | 0.0.100+ | 情緒辨識 |
| `tensorflow` | 2.21.0+ | DeepFace 後端 |
| `tf-keras` | 最新 | TensorFlow Keras API |

### 完整安裝指令

```bash
pip install -r requirements.txt
```

或單獨安裝：
```bash
pip install --user opencv-python numpy mediapipe scikit-learn xgboost deepface tensorflow tf-keras
```

---

## 🚀 快速開始

### 方式 1：使用批次檔（推薦 Windows）
直接**雙擊** `run.bat`：
- 自動檢查環境與相機
- 環境無誤後自動啟動程式

### 方式 2：命令行

**檢查環境與相機：**
```bash
python run.py --check
```

輸出範例：
```
[檢查] 開始檢查執行環境...
[檢查] 依賴與模型檔皆已就緒。
[檢查] 相機可正常開啟，已準備好啟動主程式。
```

**啟動主程式：**
```bash
python run.py
```

---

## 👁️ 使用說明

### 程式啟動流程

1. **環境驗證** → 檢查 7 個必需模組 + 2 個模型檔
2. **模型初始化** → 加載 ASD 分類器、Face Landmarker、情緒分析器
3. **相機開啟** → 啟用視頻捲軸（1280×720 @ 30fps）
4. **實時分析** → 每幀處理眼動、AOI、情緒

### 螢幕顯示說明

程式視窗顯示以下資訊（實時更新）：

```
【患者資訊面板】
嚴重程度：輕度ASD  │ 眼動狀態：注視中  │ 眨眼率：12次/分
───────────────────────────────────────────────
TD 機率：15%  │  輕度ASD 機率：65%  │  重度ASD 機率：20%
───────────────────────────────────────────────
當前情緒：開心 😊
───────────────────────────────────────────────
注視區域：社交刺激區（左側 60% 時間）
```

### 操作按鍵

| 按鍵 | 功能 |
|------|------|
| **Q** | 結束程式 |
| **其他** | 無其他快捷鍵 |

---

## 📊 核心模組詳解

### 1️⃣ **eye_tracker.py** - 眼動追蹤核心

**類別：`EyeTracker`**

#### 主要功能
- 使用 MediaPipe 檢測面部地標（468 個點）
- 提取左右眼虹膜中心位置
- 計算眼瞼開度 (EAR - Eye Aspect Ratio)
- I-VT（Velocity-Threshold）算法識別注視與掃視
- 提取 11 維眼動特徵
- 運行 scikit-learn 分類器進行 ASD 三分類
- 集成 AOI 與情緒分析

#### 提取的眼動特徵（11 維）

| # | 特徵名稱 | 描述 |
|---|---------|------|
| 1 | `fixation_count` | 5秒內的注視次數 |
| 2 | `duration_ms_total` | 所有注視總時長（毫秒） |
| 3 | `duration_ms_mean` | 單次注視平均時長 |
| 4 | `duration_ms_var` | 注視時長方差 |
| 5 | `saccade_amplitude_px_mean` | 掃視振幅平均值（像素） |
| 6 | `saccade_amplitude_px_var` | 掃視振幅方差 |
| 7 | `distance_to_centre_px_mean` | 注視點到螢幕中心的平均距離 |
| 8 | `distance_to_centre_px_var` | 距離方差 |
| 9 | `distance_to_sp_mean_px_mean` | 相鄰注視點間距平均值 |
| 10 | `distance_to_sp_mean_px_var` | 相鄰注視點距離方差 |
| 11 | `blink_rate_per_min` | 每分鐘眨眼次數 |

#### ASD 分類結果

模型輸出三個類別及其機率：

```python
result = {
    "eye_state": "fixation",           # 眼動狀態：fixation / saccade / blink / avoidant / hyperscanning
    "severity": "輕度ASD",              # 主要診斷
    "td_prob": 0.15,                   # TD（正常）機率
    "mild_prob": 0.65,                 # 輕度ASD 機率
    "severe_prob": 0.20,               # 重度ASD 機率
    "face_found": True,                # 是否偵測到臉部
    "features": {...}                  # 提取的 11 維特徵
}
```

#### 關鍵參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `window_sec` | 5.0 | 特徵計算的時間窗口（秒） |
| `fps` | 30 | 視頻幀速率 |
| `screen_w` / `screen_h` | 1280 / 720 | 螢幕解析度 |
| `ivt_threshold` | 30.0 | I-VT 掃視檢測閾值（度/秒） |
| `min_fix_ms` | 80.0 | 最小注視時長（毫秒） |
| `emotion_every_n` | 15 | 每 N 幀執行一次情緒分析 |

#### 使用範例

```python
from modules.eye_tracker import EyeTracker

tracker = EyeTracker(
    model_path="models/eye/severity_classifier.pkl",
    task_model="models/eye/face_landmarker.task",
    window_sec=5.0,
    screen_w=1280,
    screen_h=720,
)

# 處理每一幀
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    tracker.process_frame(frame)
    result = tracker.latest_result
    
    # 繪製疊加層
    output = tracker.draw_overlay(frame.copy())
    cv2.imshow("Output", output)
```

---

### 2️⃣ **emotion_analyzer.py** - 情緒辨識

**類別：`EmotionAnalyzer`**

#### 情緒分類

系統偵測 7 種基本情緒：

| 英文 | 中文 | 臨床意義 |
|------|------|--------|
| `happy` | 開心 😊 | 情緒正向，可趁機推進任務 |
| `neutral` | 平靜 😐 | 情緒平穩，適合結構化引導 |
| `sad` | 悲傷 😢 | 需要情感支持與安撫 |
| `angry` | 憤怒 😠 | 需降低刺激並穩定情緒 |
| `surprise` | 驚訝 😮 | 對刺激有反應，可引導注意力 |
| `fear` | 恐懼 😨 | 需立即減少刺激並給予安全感 |
| `disgust` | 厭惡 🤢 | 需調整互動方式或內容 |

#### 治療策略對應

每種情緒都對應臨床治療建議：

```python
EMOTION_STRATEGY = {
    'happy':    '患者情緒正向，可趁機推進互動任務或正向強化',
    'angry':    '患者出現憤怒情緒，需先降低刺激並穩定情緒',
    'fear':     '患者出現恐懼情緒，需立即減少刺激並給予安全感',
    # ... 其他情緒
}
```

#### 分析參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `detector_backend` | "yunet" | DeepFace 檢測器（輕量級） |
| `analyze_every_n_frames` | 15 | 每 15 幀分析一次（降低計算負荷） |
| `window_sec` | 5.0 | 情緒歷史窗口 |

#### 使用範例

```python
from modules.emotion_analyzer import EmotionAnalyzer

emotion = EmotionAnalyzer(
    detector_backend="yunet",
    analyze_every_n_frames=15
)

result = emotion.analyze_frame(frame)
print(f"情緒：{result['dominant_emotion_zh']}")
print(f"治療建議：{result['strategy']}")
```

---

### 3️⃣ **gaze_aoi.py** - 注視區域分析

**類別：`GazeAOIAnalyzer`**

#### 注視區域定義

系統支援兩種 AOI 模式：

**模式 1：地理偏好模式 (`geopref`)**
```
┌─────────────────────┐
│  社交刺激區         │  幾何刺激區   │
│  (左側 50%)         │  (右側 50%)   │
└─────────────────────┘
```
用於測試患者對社交 vs 物體的偏好。

**模式 2：自由觀看模式 (`freeview`)**
```
┌────────┬────────┬────────┐
│ 左上   │ 上中   │ 右上   │
├────────┼────────┼────────┤
│ 左側   │ 中央   │ 右側   │
├────────┼────────┼────────┤
│ 左下   │ 下中   │ 右下   │
└────────┴────────┴────────┘
```
9 個區域進行全螢幕覆蓋分析。

#### 分析輸出

```python
aoi_summary = {
    "dominant_label": "社交刺激區",          # 主要注視區域
    "dominant_ratio": 0.65,                 # 該區域佔比
    "social_ratio": 0.65,                   # 社交刺激區佔比
    "geometric_ratio": 0.30,                # 幾何刺激區佔比
    "vacant_ratio": 0.05,                   # 空白區域佔比
    "aoi_switches": 12,                     # 注視區域切換次數
    "dispersion_px": 85.3,                  # 注視點散佈範圍（像素）
    "llm_context": "患者目前主要注視【社交刺激區】..."
}
```

#### 臨床解釋規則

- **社交偏好 > 60%** → "患者對社交刺激有明顯偏好"
- **幾何偏好 > 60%** → "患者強烈偏好幾何刺激，社交注視極少"
- **空白區域 > 35%** → "空白注視比例高，需引導"
- **切換 > 15 次** → "注意力切換頻繁"
- **切換 < 3 次** → "注意力幾乎不切換，可能固著"

---

## 🧬 AI/LLM 整合

系統自動生成結構化的 LLM 提示詞，包含三個維度的臨床資訊：

### LLM 提示詞範例

```
【患者當前眼動分析】
嚴重程度：輕度ASD
眼動行為：焦點固著，注視時間長
患者當前情緒：開心（happy）。患者情緒正向，可趁機推進互動任務或正向強化。

【患者當前狀態】
嚴重程度：輕度ASD
眼動行為：焦點固著
注視分析：患者目前主要注視【社交刺激區】（佔 65% 時間）。
          患者對社交刺激有明顯偏好。注意力幾乎不切換（2次），可能固著。

請根據以上資訊，用繁體中文給予適當的語言治療引導（60字以內，語氣溫和）：
```

### 臨床治療師使用流程

1. 運行程式，觀察患者眼動行為
2. 程式生成實時 LLM 提示詞
3. 將提示詞發送給 GPT/Claude 等 LLM
4. 獲得實時治療建議
5. 根據建議調整治療策略

---

## ⚙️ 進階配置

### 自訂 MediaPipe 參數

編輯 [run.py](run.py) 中的 `EyeTracker` 初始化部分：

```python
tracker = EyeTracker(
    model_path=str(BASE_DIR / "models" / "eye" / "severity_classifier.pkl"),
    task_model=ensure_ascii_task_model(),
    window_sec=5.0,        # 改為 3.0 以獲得更快的反應
    screen_w=1920,         # 改為你的螢幕寬度
    screen_h=1080,         # 改為你的螢幕高度
    ivt_threshold=30.0,    # 調整掃視檢測靈敏度
    min_fix_ms=80.0,       # 調整最小注視時長
    emotion_every_n=15,    # 降低此值以提高情緒更新頻率
)
```

### 環境變數設置

若要抑制 TensorFlow 的冗長日誌：

```bash
# Windows PowerShell
$env:TF_CPP_MIN_LOG_LEVEL = '2'
python run.py
```

---

## 🐛 常見問題 & 排除

### ❌ "ModuleNotFoundError: No module named 'cv2'"

**原因**：Python 環境中未安裝 OpenCV

**解決**：
```bash
pip install --user opencv-python
```

### ❌ "RuntimeError: Unable to open file at ... face_landmarker.task"

**原因**：MediaPipe 無法讀取包含中文字符的路徑

**解決**：程式會自動複製模型到 `C:\asd_v4\models\eye\`（ASCII 路徑）

### ❌ 相機無法開啟

**原因**：
- 攝影機驅動程式未安裝
- Windows 權限設置阻止存取
- 攝影機被其他應用程式佔用

**解決**：
1. 檢查 Windows 設置 → 隱私與安全 → 相機，確保應用程式被允許
2. 重啟程式或關閉其他使用攝影機的應用
3. 使用 `python run.py --check` 診斷

### ❌ 程式運行緩慢或 FPS 低

**原因**：
- CPU 使用率過高
- 情緒分析執行過於頻繁
- 螢幕解析度過高

**解決**：
- 降低 `emotion_every_n` 參數（如改為 30）
- 降低 `screen_w` 和 `screen_h`
- 關閉其他後台應用

### ❌ 情緒辨識不準確

**原因**：
- 照明不足
- 臉部被遮擋
- DeepFace 檢測器無法找到臉部

**解決**：
- 改善照明環境
- 確保臉部清晰可見
- 檢查 `detector_backend` 是否設為 "yunet"（輕量級）

---

## 📈 模型性能指標

| 指標 | 值 | 備註 |
|------|-----|------|
| FPS | 28-30 | 在 i5 + webcam 上測試 |
| 臉部檢測延遲 | ~30ms | MediaPipe 推理時間 |
| ASD 分類延遲 | ~5ms | scikit-learn 推理 |
| 情緒分析延遲 | ~200ms | DeepFace + TensorFlow 推理 |
| 臉部檢測準確率 | >95% | MediaPipe 官方指標 |
| ASD 分類精準度 | 72% | 在測試集上（3 類） |

---

## 📝 輸出日誌範例

### 標準輸出

```
[啟動] 初始化新版眼動 + 情緒辨識系統...
[EyeTracker] 三分類器載入（11 特徵）
[EyeTracker] 類別：{0: 'TD', 1: '輕度ASD', 2: '重度ASD'}
[EyeTracker] FaceLandmarker 初始化完成
[EyeTracker] AOI 分析器初始化完成
[EyeTracker] 情緒分析器初始化完成
[啟動] 開啟相機...
[啟動] 請對準鏡頭，按 q 結束。

[frame 30] ASD 分類：輕度ASD（TD: 15%, 輕度: 65%, 重度: 20%）
[frame 45] 情緒分析：開心（開心: 85%, 平靜: 10%, 驚訝: 5%）
[frame 60] AOI 分析：社交刺激區（65% 時間），注視固著
```

---

## 📞 支援與反饋

### 報告問題

請提供：
1. 完整錯誤訊息與堆棧追蹤
2. 系統資訊（OS、Python 版本、GPU）
3. 重現問題的步驟
4. 相關的日誌檔案

### 常見改進建議

- 支援多人同時追蹤
- 新增頭部姿態估計
- 提高情緒分類精準度
- 支援客製化 AOI 定義

---

## 📄 授權與引用

本系統使用以下開源資源：

- **MediaPipe** - Google 人臉檢測模型
- **DeepFace** - 情緒辨識模型
- **scikit-learn** - ASD 分類器
- **OpenCV** - 影像處理

### 引用格式

```bibtex
@software{asd_eye_tracker_v4,
  title={ASD Eye Tracker with Emotion Recognition},
  version={4.0},
  year={2025}
}
```

---

## 🎯 更新歷史

### v4.0（當前版本）
- ✨ 新增情緒辨識模組
- ✨ 整合 LLM 友善提示詞生成
- 📈 改進 I-VT 演算法
- 🐛 修復 MediaPipe 中文路徑問題
- 🎨 新增實時視覺疊加層

### v3.0
- 新增 AOI 分析功能
- 改進 ASD 分類模型

### v1.0
- 初始版本

---

## 📬 聯絡方式

有問題或建議？請聯絡開發團隊。

---

**最後更新**：2026 年 8 月
**版本**：4.0
