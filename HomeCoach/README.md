# 暖伴 AI 居家早療互動教練

這是一個以 Flask MVC 架構製作的本機原型，依據 EMT（加強式情境教學法）
分析親子互動中的等待時間、語句擴展、輪流互動與共同注意。系統整合：

- **faster-whisper**：把家長或孩子的錄音轉成文字。
- **WebGazer**：在瀏覽器端估算使用者是否看向教材區。
- **ASD v4 實驗模組**：從校正後視線計算 TD／輕度／重度模型訊號，以
  Face Mesh 計算眨眼率，並以 DeepFace 分析低頻相機畫面的表情。
- **EMT 規則引擎**：依固定規則計算指標並決定本輪教練目標。
- **Gemini**：預設透過 REST Interactions API 讀取目前教材圖片與最小化文字情境，
  在背景把規則建議改寫成親子共讀提示。
- **Ollama**：Gemini 未設定、停用或請求失敗時的本機備援；兩者都無法使用時仍保留規則提示。
- **SQLite**：保存互動紀錄、逐句分析與統計。

> 本專案是互動練習與開發原型，不是醫療診斷工具，也不能取代治療師或醫療專業人員的判斷。

## 專案配置

以下三個資料夾應位於同一層：

```text
ASD_project-main/
├── HomeCoach/
├── asd_v4/
├── faster-whisper-master/
└── WebGazer-master/
```

`HomeCoach/requirements.txt` 使用 `-e ../faster-whisper-master` 安裝本機
faster-whisper 原始碼；這代表 Python 套件程式來自此資料夾，但語音模型權重不包含在原始碼內。
專案也將 NumPy 固定在 `1.26.x` 相容範圍，避開部分 Apple Silicon 機型使用
NumPy 2／Accelerate 執行 Mel spectrogram 矩陣運算時的警告。

## 安裝與啟動

### Windows（Conda）

先安裝 Miniconda 或 Anaconda，再於專案根目錄雙擊 `setup-windows.bat`，或在
Anaconda Prompt 執行：

```bat
setup-windows.bat
conda activate asd-homecoach
cd HomeCoach
python run.py
```

批次檔會依 `environment-windows.yml` 建立或更新 Python 3.10、CUDA 12.4 與 cuDNN 9
環境，並在完成後檢查 Flask、OpenCV、TensorFlow、faster-whisper 及 NVIDIA GPU
是否可用。執行前須先安裝支援 CUDA 12 的 NVIDIA 顯示卡驅動程式；不需要另外安裝
完整 CUDA Toolkit 或系統版 FFmpeg。Conda 環境啟用時會自動設定
`WHISPER_DEVICE=cuda` 與 `WHISPER_COMPUTE_TYPE=float16`。

原生 Windows 上，這個環境的 faster-whisper／CTranslate2 會使用 NVIDIA GPU；
TensorFlow 2.16 則使用 CPU。TensorFlow 官方在 2.10 之後已停止原生 Windows CUDA
支援；若表情分析也必須使用 GPU，應另採 WSL2 環境。

### macOS / Linux（venv）

```bash
cd HomeCoach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

請用瀏覽器開啟：

```text
http://127.0.0.1:5000
```

`localhost` 與 `127.0.0.1` 都是本機 loopback；本專案已處理 WebGazer 3.5.3
只辨認 `localhost` 所產生的多餘警告。如果 `localhost` 被系統代理或瀏覽器攔截，
請直接使用 `127.0.0.1`。相機與麥克風權限仍受瀏覽器安全環境限制。

如需 Flask 除錯模式：

```bash
FLASK_DEBUG=1 python run.py
```

啟動後可檢查整合狀態：

```bash
curl http://127.0.0.1:5000/api/health
```

若 Whisper 模型已在快取中，Flask 啟動後會在背景預熱，不讓第一句錄音承擔模型
載入時間。若本機尚無模型，第一次實際送出錄音時 faster-whisper 才會下載模型到
`HomeCoach/instance/whisper-models/`；首次使用因此需要網路、足夠磁碟空間與較長
等待時間，之後會重用本機快取。

## 操作方式

1. 進入「即時教練」，先從「切換教材」選擇要看的圖片，再按下開始互動，
   並允許瀏覽器使用相機與麥克風。互動開始後教材會鎖定，避免 LLM 與畫面不同步。
2. 視線模型準備完成後進行九點校正。依序看著畫面上的圓點，每個位置點五次，
   共 45 次。右上角的校正按鈕可重新校正。
3. 家長要說話時點一下「家長說話」，說完再點一次「停止並轉寫」。
4. 孩子要說話時同樣使用「孩子說話」按鈕。系統不會自動辨識說話者，
   按下哪個按鈕就會把這段錄音標記為哪位說話者。
5. Whisper 完成轉寫後，文字、停頓秒數與可用的視線結果會立刻進入 EMT 分析。
   Gemini／Ollama 生成期間只顯示等待狀態，不會先把規則備援誤當成模型提示；完成後
   才顯示文案，並清楚標示「模型生成」或「規則備援・非模型」。這段等待不會阻塞
   下一句錄音；圖片下方的「練習提示」也會在來源確認後同步更新。

這是 **click-to-start / click-to-stop** 的逐句錄音設計。每段錄音最長 30 秒，
後端單檔上限預設為 8 MB。若沒有偵測到清楚語音，該段不會建立對話事件；
也可用頁面上的手動輸入作為備援。上一句在背景轉寫時，家長仍可立刻錄下一句；
瀏覽器會依錄製順序排隊送出，因此模型處理時間不會被算進親子等待秒數。每次完成
後的提示會顯示 Whisper 實際處理秒數，方便區分語音辨識與背景 LLM 延遲。

## Whisper 設定

目前預設值是多語言 `base` 模型、`CPU`、`int8`、中文 `zh`：

```bash
WHISPER_MODEL=base \
WHISPER_DEVICE=cpu \
WHISPER_COMPUTE_TYPE=int8 \
WHISPER_LANGUAGE=zh \
python run.py
```

可用的環境變數如下：

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `WHISPER_ENABLED` | `1` | 設為 `0`、`false`、`no` 或 `off` 可停用轉寫 |
| `WHISPER_SOURCE_DIR` | `../faster-whisper-master` | 本機 faster-whisper 原始碼根目錄 |
| `WHISPER_MODEL` | `base` | 模型名稱，或已下載的 CTranslate2 模型路徑 |
| `WHISPER_DEVICE` | `cpu` | 推論裝置，例如 `cpu` 或支援環境中的 `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | CTranslate2 計算精度；需與硬體相容 |
| `WHISPER_LANGUAGE` | `zh` | 指定辨識語言；改成其他語言代碼，或以空值啟用語言偵測 |
| `WHISPER_DOWNLOAD_ROOT` | `HomeCoach/instance/whisper-models` | 模型下載與快取位置 |
| `WHISPER_MAX_AUDIO_BYTES` | `8388608` | 每段上傳音訊的最大位元組數 |

例如改用已經放在磁碟上的模型：

```bash
WHISPER_MODEL=/absolute/path/to/faster-whisper-model \
WHISPER_DOWNLOAD_ROOT=/absolute/path/to/model-cache \
python run.py
```

主要修改位置：

- `app/services/transcription_service.py`：模型延遲載入、VAD、轉寫參數、執行鎖與輸出格式。
- `app/controllers/api.py`：`POST /api/sessions/<id>/transcriptions`、檔案格式／大小驗證及暫存檔清除。
- `app/static/js/coach.js`：`MediaRecorder`、說話者按鈕、錄音格式、30 秒上限及 multipart 上傳。
- `app/__init__.py`：Whisper 預設環境變數與服務註冊。

若要調整辨識品質或速度，可先修改
`TranscriptionService.transcribe()` 內的 `beam_size`、`vad_filter` 與
`condition_on_previous_text`。目前每次錄的是完整單句，並使用 VAD 過濾靜音；
模型與推論操作會加鎖，避免 Flask 同時載入多份模型或讓多個請求競爭同一模型。
目前使用 `beam_size=1`、固定中文及關閉跨句 context，偏向即時互動速度；回傳 metadata
包含 `processing_seconds`、`model_load_seconds` 與 `inference_seconds`。

## WebGazer 設定

Flask 直接從下列本機檔案提供 WebGazer 與 MediaPipe Face Mesh：

```text
WebGazer-master/www/webgazer.js
WebGazer-master/www/mediapipe/face_mesh/
```

頁面使用 `TFFacemesh` tracker 與 `ridge` regression。九個校正點各點五次後，
WebGazer 才會把瀏覽器視窗座標與教材區比較；最近樣本數足夠時，才會將
`gaze_available=true` 送進 EMT 分析。相機沒有資料或視線樣本不足時會標記為
「視線資料不足」，不會誤判成孩子沒有看教材。

目前 WebGazer **一次只能追蹤一張臉**。使用時應讓要觀察的孩子單獨、清楚地出現在
鏡頭中；多人同框、光線不足、遮擋、眼鏡反光、頭部大幅移動或校正後改變坐姿，
都可能降低準確度。WebGazer 是一般網路攝影機的估算工具，不等同專業眼動儀。

主要修改位置：

- `app/templates/coach.html`：載入 WebGazer、校正對話框與視線狀態 UI。
- `app/static/js/coach.js`：啟動／停止攝影機、九點校正、樣本門檻及教材區判斷。
- `app/controllers/pages.py`：安全地提供 `webgazer.js` 與允許的 Face Mesh 資產。
- `app/__init__.py`：`WEBGAZER_DIR` 的本機路徑。

若要把 WebGazer 資料夾移到別處，可直接設定：

```bash
WEBGAZER_DIR=/absolute/path/to/WebGazer-master/www python run.py
```

如要改教材判定範圍，調整 `coach.js` 的 `handleGazePrediction()`；如要改校正點，
調整 `calibrationPositions` 與 `calibrationClicksPerPoint`。

## ASD v4 實驗訊號

系統重用同一條瀏覽器相機串流，不會再由 Python 開啟第二台相機：

- 校正後的 WebGazer 座標會反算回目前教材的原始圖片座標，只保留圖片範圍內、
  最近約五秒的樣本。後端使用實際時間戳計算 fixation，再建立 ASD v4 模型所需的
  11 個 scanpath 特徵。
- Face Mesh 的 468 點臉部地標在瀏覽器端計算 EAR；眨眼率只使用實際辨識到臉的
  時間作分母。追蹤不到臉時顯示「—」，不會當成 0 次眨眼。
- 每五秒最多擷取一張 480 px 寬的 JPEG 給本機 Flask／DeepFace。影像只在記憶體
  解碼，完成後丟棄；SQLite 只會在對話事件中保存最新的衍生觀察值。
- UI 顯示表情、眨眼率、TD／輕度／重度機率與眼動狀態。沒有足夠 scanpath 時，
  機率顯示「—」，不會以三個 0% 冒充結果。

可用的環境變數：

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `ASD_ANALYSIS_ENABLED` | `1` | 是否啟用 ASD v4 實驗分析 |
| `ASD_EMOTION_ENABLED` | `1` | 是否啟用 DeepFace 表情分析 |
| `ASD_V4_DIR` | `../asd_v4` | ASD v4 模型與 modules 根目錄 |
| `ASD_FRAME_MAX_BYTES` | `1048576` | 單張低頻分析圖片的壓縮大小上限 |

主要修改位置：

- `app/services/asd_analysis_service.py`：模型延遲載入、fixation／特徵、情緒與 session 狀態。
- `asd_v4/modules/emotion_analyzer.py`：DeepFace 分析與失敗可用性回報。
- `app/controllers/api.py`：`POST /api/sessions/<id>/asd-analysis` 與 multipart 驗證。
- `app/static/js/coach.js`：圖片座標反算、EAR、低頻 frame 與即時面板更新。

> TD／輕度／重度是既有研究模型的輸出標籤，不是 ASD 診斷、嚴重度評估或醫療建議。
> 此模型原始訓練資料與一般網路攝影機／居家教材情境並不等同，因此不可據此替孩子
> 貼標籤或決定治療。Gemini／Ollama 只會取得「放慢、降低刺激、增加等待」等
> 去診斷化互動調整，不會收到分類名稱或機率；輸出若含診斷／嚴重度宣告也會直接
> 退回規則提示。

## Gemini 教練與 Ollama 備援

預設 `COACH_PROVIDER=gemini`。後端以 REST 呼叫 Gemini Interactions API，預設模型為
`gemini-3.6-flash`；API Key 只可從伺服器程序的環境變數讀取：

```bash
export COACH_PROVIDER=gemini
export GEMINI_API_KEY="你的 API Key"
export GEMINI_MODEL=gemini-3.6-flash
python run.py
```

請勿把 `GEMINI_API_KEY` 寫進 Python／JavaScript、網址、HTML、SQLite、版本控制或
瀏覽器儲存空間。正式部署應由作業系統、容器或祕密管理服務注入環境變數。Gemini
請求由 Flask 後端送往 REST Interactions API，瀏覽器不會取得 API Key。

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `COACH_PROVIDER` | `gemini` | 主要自然語氣 provider；可改為 `ollama` 強制只用本機模型 |
| `GEMINI_ENABLED` | `1` | 是否啟用 Gemini；停用時直接使用本機備援 |
| `GEMINI_API_KEY` | 無 | Gemini API Key；未設定時 health 顯示 `key_missing` 並改用本機備援 |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini Interactions API 使用的模型 |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | 官方 REST API base；程式呼叫 `/interactions` |
| `GEMINI_TIMEOUT_SECONDS` | `15` | Gemini REST 請求逾時秒數 |
| `GEMINI_THINKING_LEVEL` | `low` | Gemini thinking level |
| `GEMINI_VISION_ENABLED` | `1` | 是否把目前 catalog 對應的教材 PNG 附加到 Gemini 請求 |
| `GEMINI_MAX_IMAGE_BYTES` | `3145728` | 教材 PNG 大小上限（3 MiB） |
| `OLLAMA_ENABLED` | `1` | 是否允許 Ollama 作為 Gemini 的本機備援 |
| `OLLAMA_BASE_URL` | `http://localhost:11434/api` | Ollama API base；程式會再加上 `/chat` 或 `/tags` |
| `OLLAMA_MODEL` | `gemma4:e4b` | Ollama 本機備援模型名稱 |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Ollama 請求逾時秒數 |

本機備援仍需先啟動 Ollama 並備妥模型。例如：

```bash
export OLLAMA_BASE_URL=http://localhost:11434/api
export OLLAMA_MODEL=gemma4:e4b
python run.py
```

`GET /api/health` 的 `modules.coach_provider` 會回報主要 provider、狀態、模型與 Ollama
fallback health。Gemini 狀態會區分 `configured`（已設定、尚未呼叫）、`ready`、
`degraded`、`key_missing` 或 `disabled`；`key_missing` 時 UI 會明確顯示
「Gemini 尚未設定，使用本機備援」。每次背景潤飾的 API 回應則以
`coach_source=gemini|ollama|rule_engine` 表示真正產生該提示的來源。

EMT 規則引擎永遠是 coaching target 的唯一判斷依據。Gemini 或 Ollama 只能改寫
`message`、`example` 與圖片上的 `practice_prompt`，程式會固定保留規則引擎決定的
`tone`、`eyebrow` 與 `title`。模型格式錯誤、逾時或輸出診斷文字時，會依序退回
Ollama 或情境化規則提示，不會讓逐字稿中的提示注入更換 EMT 目標。

## MVC 架構與修改位置

```text
HomeCoach/
├── app/
│   ├── __init__.py                    # App factory、環境設定、依賴組裝
│   ├── materials.py                   # 教材 allowlist、可信圖片描述與共讀提示
│   ├── controllers/
│   │   ├── pages.py                   # HTML 頁面與本機 WebGazer 資產
│   │   └── api.py                     # Session、event、transcription、ASD、health API
│   ├── models/
│   │   └── repository.py              # SQLite schema 與 CRUD
│   ├── services/
│   │   ├── coaching_service.py        # 使用案例流程與各元件協調
│   │   ├── asd_analysis_service.py     # ASD v4 實驗訊號 adapter
│   │   ├── emt_rule_engine.py         # EMT 判斷與統計
│   │   ├── context_builder.py         # LLM 最小化情境資料
│   │   ├── gemini_coach_provider.py   # Gemini Interactions API 與 Ollama 降級
│   │   ├── ollama_coach_provider.py   # 本機 Ollama 與規則備援
│   │   └── transcription_service.py   # faster-whisper adapter
│   ├── templates/                     # View：Jinja HTML
│   └── static/                        # View：CSS 與瀏覽器 JavaScript
├── instance/
│   ├── homecoach.sqlite3              # 執行後建立的本機資料庫
│   └── whisper-models/                # 第一次轉寫後建立的模型快取
├── tests/
├── requirements.txt
└── run.py
```

常見修改方式：

- 改 UI 文字或版面：`templates/coach.html`、`static/css/app.css`。
- 改教材圖片描述或圖上練習提示：`materials.py`。
- 改瀏覽器互動、錄音或視線門檻：`static/js/coach.js`。
- 改 EMT 等待時間與建議優先順序：`services/emt_rule_engine.py`。
- 改統計欄位或資料表：`models/repository.py`，並同步處理既有 SQLite migration。
- 改 API 格式：`controllers/api.py`，並同步更新 `coach.js`。
- 改 LLM 收到的資料量：`services/context_builder.py` 的 `history_limit` 與輸出欄位。
- 換 Whisper 或 LLM 實作：維持 service/provider 的公開介面後，在 `app/__init__.py` 注入新實作。

## 資料流

```text
選擇教材圖片
  → material_id 保存到 session
  → ContextBuilder 只讀該圖片的可信描述與可見元素
  → Gemini provider 只從受信任 catalog 取得目前教材 PNG

按下家長／孩子錄音
  → 瀏覽器 MediaRecorder 產生一段音訊
  → Flask 暫存音訊
  → faster-whisper 轉成文字並立即刪除暫存音訊
  → EMT 規則引擎分析文字、停頓與可用的視線結果
  → 立即回傳情境化備援提示，UI 解鎖下一句錄音
  → ContextBuilder 擷取最近五筆情境
  → Gemini 在背景改寫提示
  → Gemini 未設定／停用／失敗時改用本機 Ollama
  → 兩者皆不可用時保留情境化規則提示
  → SQLite 更新事件與統計
  → UI 只在最新一輪完成時更新，並依 gemini／ollama／rule_engine 顯示來源
```

視覺流程與錄音並行：WebGazer 的完整即時影像仍在瀏覽器端處理；瀏覽器只把校正後的
教材座標、眨眼率，以及每五秒最多一張縮小 JPEG 傳給本機 Flask。JPEG 僅供 DeepFace
當次推論並在記憶體中丟棄，不會寫入 SQLite，也不會送給 Gemini 或 Ollama。Gemini
收到的圖片只會是目前畫面中、由受信任教材 catalog 對應的靜態 PNG。

## 隱私與限制

- 啟用 Gemini 時，只會送出目前教材 PNG、最近五筆逐字文字，以及「放慢、降低刺激、
  增加等待」等去診斷化互動調整。這些資料會經 Gemini API 離開本機。請求中的
  `store=false` 只表示不建立可由 `previous_interaction_id` 延續的伺服器互動狀態，
  不代表 Google 完全不保存或不處理資料；實際處理方式仍依 Gemini 帳戶方案與 Google
  資料條款而定，免費層與付費層可能不同，使用前應確認當時適用的條款。
- 完整相機串流不離開瀏覽器；縮小的低頻相機／DeepFace JPEG 只送到本機 Flask 做
  當次推論，不落盤、不保存，也不會送到 Gemini 或 Ollama。
- WebGazer 設為不跨工作階段保存校正資料，重新校正時也會先清除舊資料。
- 原始麥克風錄音只送往 `localhost` 的 Flask；後端為轉寫建立的暫存檔會在成功或失敗後
  刪除，Gemini 與 Ollama 只取得轉寫文字，不會取得原始錄音。
- TD／輕度／重度分類名稱與機率不會送到 Gemini 或 Ollama；LLM 只取得去診斷化的互動
  調整，不能把研究模型訊號當成診斷。
- SQLite 會保存轉寫文字、說話者、停頓、視線布林值、Whisper metadata、EMT 分析、
  最新衍生互動觀察值與提示，但不保存原始錄音或圖片。
  互動紀錄仍保存在 `instance/homecoach.sqlite3`；若文字本身包含個人資料，也會留在本機紀錄。
- `GEMINI_API_KEY` 只由 Flask 後端的環境變數讀取，不會回傳給瀏覽器或寫入互動紀錄。
- 第一次取得 Whisper 模型可能連線至模型代管服務；模型下載完成後可從本機快取執行。
- faster-whisper 不是說話者分離工具，因此家長／孩子必須手動選擇；多人同時說話會影響結果。
- 中文方言、兒童語音、背景噪音、距離與麥克風品質都會影響轉寫；固定 `zh` 也可能不適合中英混合語句。
- CPU `int8` 節省記憶體，但長錄音仍可能延遲。模型越大通常越耗磁碟、記憶體與時間。
- 視線與語音結果都應視為輔助訊號，不應據此做診斷、風險分級或治療效果保證。

## 驗證

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

單元測試不需要真的下載 Whisper 模型，也不會連線 Gemini 或 Ollama；目前涵蓋 ASD
multipart API、模型類別機率映射、實際時間戳 fixation、教材 context 與診斷語句安全回退。
實機驗收仍應另外測試：

1. `http://127.0.0.1:5000/api/health` 的 `coach_provider`、Ollama fallback、Whisper 與
   `asd_analysis` 狀態；未設定 Key 時應顯示 `key_missing`。
2. 瀏覽器的相機、麥克風權限與九點校正。
3. 家長／孩子各錄一段中文，確認轉寫、提示與紀錄頁內容。
4. 分別確認 `coach_source=gemini`、Gemini 失敗後的 `ollama`，以及兩者皆停用時的
   `rule_engine` 提示與 UI 狀態。
5. 單人入鏡至少 10 秒，確認表情、眨眼率與三分類機率不再顯示「—」；遮住鏡頭時
   應回到無資料，而不是顯示 0 或平靜。
