(() => {
    const state = {
        sessionId: null,
        startedAt: null,
        timerId: null,
        demoTimers: [],
        demoRunId: 0,
        demoRunning: false,
        isSubmittingEvent: false,
        healthPromise: null,
        whisperUsable: null,
        audioStream: null,
        videoStream: null,
        mediaRecorder: null,
        audioChunks: [],
        activeRecording: null,
        transcriptionQueue: [],
        isTranscribing: false,
        latestEventId: null,
        coachRefinementBusy: false,
        pendingCoachRefinement: null,
        coachProviderHealth: null,
        lastUtteranceEndedAt: null,
        gazeCalibrated: false,
        gazeAvailable: false,
        gazeOnTarget: true,
        gazeSamples: [],
        asdGazeSamples: [],
        asdTimer: null,
        asdBusy: false,
        asdAbortController: null,
        asdUsable: false,
        asdEmotionUsable: false,
        asdStartedAt: null,
        blinkEvents: [],
        blinkObservationSamples: [],
        blinkClosed: false,
        blinkClosedFrames: 0,
        blinkTrackingAvailable: false,
        lastFaceLandmarkAt: null,
        faceFound: false,
        selectedMaterial: null,
        materialCatalog: [],
        gazeWatchdog: null,
        lastGazeAt: null,
        calibrationIndex: 0,
        calibrationClicks: 0,
        calibrationResolver: null,
        metrics: {
            average_wait: 0,
            expansion_rate: 0,
            turn_taking_rate: 0,
        },
        lastEventAt: null,
        completed: false,
    };

    const elements = {
        start: document.querySelector("#start-session"),
        demo: document.querySelector("#demo-session"),
        finish: document.querySelector("#finish-session"),
        recordParent: document.querySelector("#record-parent"),
        recordChild: document.querySelector("#record-child"),
        timer: document.querySelector("#session-timer"),
        sessionState: document.querySelector("#session-state-label"),
        speechStatus: document.querySelector("#speech-status"),
        speechDot: document.querySelector("#speech-dot"),
        gazeStatus: document.querySelector("#gaze-status"),
        gazeDot: document.querySelector("#gaze-dot"),
        signalSummary: document.querySelector("#signal-summary"),
        ruleStatus: document.querySelector("#rule-status"),
        cameraVideo: document.querySelector("#camera-video"),
        cameraPlaceholder: document.querySelector("#camera-placeholder"),
        gazeMarker: document.querySelector("#gaze-marker"),
        canvas: document.querySelector("#material-canvas"),
        materialImage: document.querySelector("#material-image"),
        materialSelect: document.querySelector("#material-select"),
        materialNumber: document.querySelector("#material-number"),
        materialTitle: document.querySelector("#material-title"),
        materialSubtitle: document.querySelector("#material-subtitle"),
        materialLabel: document.querySelector("#session-material-label"),
        materialCaption: document.querySelector("#material-caption"),
        materialPromptLabel: document.querySelector("#material-prompt-label"),
        materialPrompt: document.querySelector("#material-practice-prompt"),
        hud: document.querySelector("#coach-hud"),
        hudEyebrow: document.querySelector("#hud-eyebrow"),
        hudSource: document.querySelector("#hud-source"),
        hudTitle: document.querySelector("#hud-title"),
        hudMessage: document.querySelector("#hud-message"),
        hudExample: document.querySelector("#hud-example"),
        dialogue: document.querySelector("#dialogue-stream"),
        dialogueEmpty: document.querySelector("#dialogue-empty"),
        toggleEntry: document.querySelector("#toggle-entry"),
        eventEntry: document.querySelector("#event-entry"),
        eventSpeaker: document.querySelector("#event-speaker"),
        eventText: document.querySelector("#event-text"),
        eventPause: document.querySelector("#event-pause"),
        eventSubmit: document.querySelector(
            "#event-entry button[type='submit']",
        ),
        waitRing: document.querySelector("#wait-ring"),
        waitValue: document.querySelector("#wait-value"),
        expandRing: document.querySelector("#expand-ring"),
        expandValue: document.querySelector("#expand-value"),
        turnRing: document.querySelector("#turn-ring"),
        turnValue: document.querySelector("#turn-value"),
        focus: document.querySelector("#toggle-focus"),
        recalibrate: document.querySelector("#recalibrate-gaze"),
        calibrationDialog: document.querySelector("#calibration-dialog"),
        calibrationTarget: document.querySelector("#calibration-target"),
        calibrationProgress: document.querySelector("#calibration-progress"),
        skipCalibration: document.querySelector("#skip-calibration"),
        finishDialog: document.querySelector("#finish-dialog"),
        closeFinish: document.querySelector("#close-finish-dialog"),
        finishWait: document.querySelector("#finish-wait"),
        finishExpand: document.querySelector("#finish-expand"),
        finishTurn: document.querySelector("#finish-turn"),
        asdStatus: document.querySelector("#asd-status"),
        asdEmotion: document.querySelector("#asd-emotion"),
        asdBlinkRate: document.querySelector("#asd-blink-rate"),
        asdSeverity: document.querySelector("#asd-severity"),
        asdEyeState: document.querySelector("#asd-eye-state"),
        asdTdProb: document.querySelector("#asd-td-prob"),
        asdMildProb: document.querySelector("#asd-mild-prob"),
        asdSevereProb: document.querySelector("#asd-severe-prob"),
        asdTdBar: document.querySelector("#asd-td-bar"),
        asdMildBar: document.querySelector("#asd-mild-bar"),
        asdSevereBar: document.querySelector("#asd-severe-bar"),
    };

    try {
        state.materialCatalog = JSON.parse(
            document.querySelector("#material-catalog")?.textContent || "[]",
        );
    } catch (error) {
        state.materialCatalog = [];
    }
    state.selectedMaterial = state.materialCatalog.find(
        (item) => item.id === elements.materialImage?.dataset.materialId,
    ) || state.materialCatalog[0] || null;

    function buildDemoScript() {
        const material = state.selectedMaterial || {};
        const visible = material.visible_elements || ["畫面"];
        const firstWord = visible[0] || "畫面";
        const secondWord = visible[1] || firstWord;
        const example = String(material.parent_example || `「我看到${firstWord}。」`)
            .replace(/[「」]/g, "");
        return [
            {
                delay: 900,
                speaker: "parent",
                text: material.practice_prompt || "你在圖片裡看到什麼？",
                pause: 0.0,
                gaze: true,
            },
            { delay: 3100, speaker: "child", text: firstWord, pause: 3.4, gaze: true },
            { delay: 2800, speaker: "parent", text: example, pause: 3.6, gaze: true },
            { delay: 3400, speaker: "child", text: secondWord, pause: 3.1, gaze: false },
            {
                delay: 2500,
                speaker: "parent",
                text: `對，我也看到${secondWord}`,
                pause: 3.5,
                gaze: true,
            },
        ];
    }

    const calibrationPositions = [
        [10, 12],
        [50, 12],
        [90, 12],
        [10, 50],
        [90, 50],
        [10, 88],
        [50, 88],
        [90, 88],
        [50, 50],
    ];
    const calibrationClicksPerPoint = 5;

    async function fetchJson(url, options = {}) {
        const headers = new Headers(options.headers || {});
        if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
            headers.set("Content-Type", "application/json");
        }
        const response = await fetch(url, {
            ...options,
            headers,
        });
        const contentType = response.headers.get("content-type") || "";
        const data = contentType.includes("application/json")
            ? await response.json()
            : { error: await response.text() };
        if (!response.ok) {
            throw new Error(data.error || "系統暫時無法完成操作");
        }
        return data;
    }

    function completedCoachStatus(source, { background = false } = {}) {
        const action = background ? "已確認並更新提示" : "已產生提示";
        if (source === "gemini") return `Gemini ${action}`;
        if (source === "ollama") return `Ollama ${action}`;
        return "規則備援提示（非模型生成）";
    }

    function pendingCoachStatus() {
        const health = state.coachProviderHealth || {};
        if (health.provider === "gemini") {
            if (["ready", "configured"].includes(health.status)) {
                return "Gemini 正在生成；完成前不顯示提示";
            }
            if (health.fallback?.status === "ready") {
                return "Ollama 正在生成；完成前不顯示提示";
            }
            return "模型目前無法使用；正在準備規則備援";
        }
        if (health.provider === "ollama" || health.status === "ready") {
            return "Ollama 正在生成；完成前不顯示提示";
        }
        return "模型正在生成；完成前不顯示提示";
    }

    function coachSourceMeta(suggestion = {}, source = suggestion.source) {
        const generatedFields = new Set(
            String(suggestion.model_generated_fields || "")
                .split(",")
                .filter(Boolean),
        );
        const allCopyFromModel = ["message", "example", "practice_prompt"]
            .every((field) => generatedFields.has(field));
        if (source === "gemini") {
            return {
                source,
                label: allCopyFromModel
                    ? "文案 Gemini 生成 · 策略規則"
                    : "Gemini 部分生成 · 含規則備援",
            };
        }
        if (source === "ollama") {
            return {
                source,
                label: allCopyFromModel
                    ? "文案 Ollama 生成 · 策略規則"
                    : "Ollama 部分生成 · 含規則備援",
            };
        }
        return { source: "rule_engine", label: "規則備援 · 非模型" };
    }

    function showPendingCoach() {
        elements.hud.dataset.tone = "ready";
        elements.hud.dataset.mode = "waiting";
        elements.hud.classList.add("is-waiting");
        elements.hudEyebrow.textContent = "模型生成中";
        elements.hudTitle.textContent = "正在讀懂這一輪對話";
        elements.hudMessage.textContent = "完成前不顯示提示，避免把備援內容誤認為模型回覆。";
        elements.hudExample.textContent = "";
        if (elements.hudSource) {
            elements.hudSource.dataset.source = "pending";
            elements.hudSource.textContent = "等待模型回覆";
        }
        elements.materialPrompt.textContent = "";
        if (elements.materialPromptLabel) {
            elements.materialPromptLabel.textContent = "模型正在生成提示…";
        }
        elements.materialCaption?.classList.remove("is-relationship", "is-safety");
        elements.materialCaption?.classList.add("is-waiting");
    }

    function updateCoachProviderHealth(coachProvider, legacyOllama) {
        const health = coachProvider
            ? {
                ...coachProvider,
                provider: coachProvider.provider || "ollama",
            }
            : legacyOllama
                ? { ...legacyOllama, provider: "ollama" }
                : null;
        state.coachProviderHealth = health;

        if (!health) {
            elements.ruleStatus.textContent = "教練模型狀態無法確認";
            return;
        }

        if (health.provider === "gemini") {
            if (health.status === "ready") {
                elements.ruleStatus.textContent = health.model
                    ? `Gemini ${health.model} 已就緒`
                    : "Gemini 已就緒";
                return;
            }
            if (health.status === "configured") {
                elements.ruleStatus.textContent = "Gemini 已設定，等待首次使用";
                return;
            }
            if (health.status === "key_missing") {
                elements.ruleStatus.textContent = "Gemini 尚未設定，使用本機備援";
                return;
            }
            if (health.status === "degraded") {
                if (health.last_error === "dns_unavailable") {
                    elements.ruleStatus.textContent = "Gemini DNS 無法連線，使用本機備援";
                } else if (health.last_error === "timeout") {
                    elements.ruleStatus.textContent = "Gemini 回應逾時，使用本機備援";
                } else if (health.last_error?.startsWith("http_")) {
                    const statusCode = health.last_error.slice(5);
                    elements.ruleStatus.textContent =
                        `Gemini API 錯誤 ${statusCode}，使用本機備援`;
                } else {
                    elements.ruleStatus.textContent = "Gemini 網路連線異常，使用本機備援";
                }
                return;
            }
            if (health.status === "disabled") {
                elements.ruleStatus.textContent = "Gemini 已停用，使用本機備援";
                return;
            }
            elements.ruleStatus.textContent = "Gemini 暫時無法使用，使用本機備援";
            return;
        }

        if (health.provider === "ollama") {
            if (health.status === "ready") {
                elements.ruleStatus.textContent = health.model
                    ? `本機備援 ${health.model} 已就緒`
                    : "本機 Ollama 已就緒";
                return;
            }
            if (health.status === "model_missing") {
                elements.ruleStatus.textContent = health.model
                    ? `本機找不到 ${health.model}`
                    : "找不到本機 Ollama 模型";
                return;
            }
        }

        elements.ruleStatus.textContent = "使用情境化規則備援";
    }

    function selectMaterial(materialId, { updateUrl = true } = {}) {
        if (state.sessionId) return;
        const material = state.materialCatalog.find(
            (item) => item.id === String(materialId),
        );
        if (!material) return;

        state.selectedMaterial = material;
        const index = state.materialCatalog.indexOf(material);
        elements.materialImage.src = `/stimuli/${encodeURIComponent(material.filename)}`;
        elements.materialImage.alt = material.alt_text;
        elements.materialImage.dataset.materialId = material.id;
        elements.materialNumber.textContent = String(index + 1).padStart(2, "0");
        elements.materialTitle.textContent = material.title;
        elements.materialSubtitle.textContent = material.subtitle;
        elements.materialLabel.textContent = material.session_label;
        elements.materialPrompt.textContent = material.practice_prompt;
        if (elements.materialPromptLabel) {
            elements.materialPromptLabel.textContent = "練習提示 · 可以直接這樣問";
        }
        elements.materialCaption?.classList.remove("is-relationship");
        elements.hudExample.textContent = "";
        if (elements.materialSelect.value !== material.id) {
            elements.materialSelect.value = material.id;
        }

        if (updateUrl) {
            const url = new URL(window.location.href);
            url.searchParams.set("material", material.id);
            window.history.replaceState({}, "", url);
        }
    }

    async function startSession() {
        if (state.sessionId) return;
        elements.start.disabled = true;
        elements.start.innerHTML = '<span class="button-spinner"></span> 正在準備';

        try {
            await state.healthPromise;
            await startGazeTracking();
            await startAudioCapture();

            const data = await fetchJson("/api/sessions", {
                method: "POST",
                body: JSON.stringify({
                    child_name: "小宇",
                    material_id: state.selectedMaterial?.id,
                }),
            });

            state.sessionId = data.session.id;
            elements.materialSelect.disabled = true;
            state.startedAt = Date.now();
            state.lastEventAt = Date.now();
            if (state.audioStream?.active) {
                state.lastUtteranceEndedAt = performance.now();
            }
            startTimer();
            updateRecordingControls();

            elements.start.innerHTML = '<span class="live-control-dot"></span> 互動進行中';
            elements.demo.disabled = false;
            elements.finish.disabled = false;
            elements.sessionState.textContent = "互動中";
            elements.sessionState.parentElement.classList.add("is-live");
            elements.signalSummary.textContent = "分析中";
            elements.signalSummary.classList.add("is-live");
            updateHud({
                tone: "ready",
                eyebrow: "親子共讀",
                title: "先跟著孩子看到的地方",
                message: "先描述一個畫面細節，再停下來等待孩子回應。",
                example: "",
                practice_prompt: state.selectedMaterial?.practice_prompt,
            });
            startASDAnalysis();
            window.showToast("互動已開始，暖伴正在分析節奏", "success");
        } catch (error) {
            stopAudioCapture();
            stopGazeTracking();
            stopASDAnalysis();
            elements.start.disabled = false;
            elements.start.innerHTML = '<span class="record-dot"></span> 重新開始';
            window.showToast(error.message, "error");
        }
    }

    async function startGazeTracking() {
        let compatible = false;
        try {
            compatible = Boolean(
                window.webgazer && window.webgazer.detectCompatibility(),
            );
        } catch (error) {
            compatible = false;
        }
        if (!compatible) {
            setGazeFallback("瀏覽器不支援視線追蹤");
            await startCameraPreviewOnly();
            return;
        }

        const runtime = document.querySelector("#webgazer-runtime");
        try {
            window.webgazer.params.faceMeshSolutionPath =
                runtime?.dataset.faceMeshBase || "/mediapipe/face_mesh";

            window.webgazer
                .setRegression("ridge")
                .setTracker("TFFacemesh")
                .saveDataAcrossSessions(false)
                .applyKalmanFilter(true)
                .showVideoPreview(false)
                .showPredictionPoints(false)
                .setGazeListener(handleGazePrediction);

            const isLoopback =
                window.location.hostname === "localhost"
                || window.location.hostname === "127.0.0.1"
                || window.location.hostname === "::1";
            let beginPromise;
            if (window.location.protocol === "http:" && isLoopback) {
                // WebGazer 3.5.3 only recognizes the literal hostname
                // "localhost" before showing its HTTPS warning. Browsers also
                // treat loopback IPs as trustworthy, so suppress that one
                // synchronous vendor alert when 127.0.0.1 is used.
                const originalAlert = window.alert;
                window.alert = () => {};
                try {
                    beginPromise = window.webgazer.begin();
                } finally {
                    window.alert = originalAlert;
                }
            } else {
                beginPromise = window.webgazer.begin();
            }
            await beginPromise;
            window.webgazer.removeMouseEventListeners();
            window.clearInterval(state.gazeWatchdog);
            state.lastGazeAt = null;
            state.gazeWatchdog = window.setInterval(() => {
                if (
                    state.gazeCalibrated
                    && (
                        state.lastGazeAt === null
                        || performance.now() - state.lastGazeAt > 2000
                    )
                ) {
                    if (state.activeRecording) {
                        state.activeRecording.gazeUnavailable = true;
                    }
                    updateGaze(true, false);
                }
            }, 750);

            const sourceVideo = document.getElementById(
                window.webgazer.params.videoElementId,
            );
            state.videoStream = sourceVideo?.srcObject || null;
            if (state.videoStream) {
                elements.cameraVideo.srcObject = state.videoStream;
                elements.cameraPlaceholder.hidden = true;
                elements.cameraVideo.classList.add("is-visible");
            }

            elements.gazeStatus.textContent = "正在載入視線模型";
            await waitForGazeModel();
            await beginGazeCalibration();
        } catch (error) {
            stopGazeTracking();
            setGazeFallback("無法啟動視線追蹤");
            await startCameraPreviewOnly();
            window.showToast("WebGazer 未能啟動，已改用備援模式", "default");
        }
    }

    async function startCameraPreviewOnly() {
        if (state.videoStream?.active || !navigator.mediaDevices?.getUserMedia) {
            return;
        }
        try {
            state.videoStream = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: {
                    facingMode: "user",
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                },
            });
            elements.cameraVideo.srcObject = state.videoStream;
            elements.cameraPlaceholder.hidden = true;
            elements.cameraVideo.classList.add("is-visible");
        } catch (error) {
            state.videoStream = null;
        }
    }

    function setGazeFallback(message = "示範視線") {
        elements.cameraPlaceholder.hidden = false;
        elements.cameraPlaceholder.querySelector("small").textContent = "視線備援模式";
        state.gazeCalibrated = false;
        updateGaze(true, false, message);
        startGazeMotion();
    }

    function startGazeMotion() {
        window.clearInterval(state.gazeAnimation);
        elements.gazeMarker.classList.add("is-visible");
        let phase = 0;
        state.gazeAnimation = window.setInterval(() => {
            phase += 0.22;
            const x = 58 + Math.sin(phase) * 9;
            const y = 42 + Math.cos(phase * 0.72) * 7;
            elements.gazeMarker.style.left = `${x}%`;
            elements.gazeMarker.style.top = `${y}%`;
        }, 420);
    }

    function waitForGazeModel(timeoutMs = 20000) {
        const startedAt = Date.now();
        return new Promise((resolve, reject) => {
            const timer = window.setInterval(() => {
                if (window.webgazer?.getTracker()?.predictionReady === true) {
                    window.clearInterval(timer);
                    resolve();
                    return;
                }
                if (Date.now() - startedAt >= timeoutMs) {
                    window.clearInterval(timer);
                    reject(new Error("WebGazer model timeout"));
                }
            }, 250);
        });
    }

    async function beginGazeCalibration() {
        if (!window.webgazer || !elements.calibrationDialog) return false;

        await window.webgazer.clearData();
        state.gazeCalibrated = false;
        state.gazeAvailable = false;
        state.gazeSamples = [];
        state.calibrationIndex = 0;
        state.calibrationClicks = 0;
        positionCalibrationTarget();
        if (!elements.calibrationDialog.open) {
            elements.calibrationDialog.showModal();
        }
        updateGaze(true, false, "等待視線校正");

        return new Promise((resolve) => {
            state.calibrationResolver = resolve;
        });
    }

    function positionCalibrationTarget() {
        const [x, y] = calibrationPositions[state.calibrationIndex];
        elements.calibrationTarget.style.left = `${x}%`;
        elements.calibrationTarget.style.top = `${y}%`;
        const totalClicks =
            state.calibrationIndex * calibrationClicksPerPoint
            + state.calibrationClicks;
        elements.calibrationProgress.textContent = `${Math.min(totalClicks, 45)} / 45`;
        elements.calibrationDialog.classList.toggle(
            "is-center-target",
            state.calibrationIndex === calibrationPositions.length - 1,
        );
    }

    function handleCalibrationClick() {
        if (!window.webgazer || !state.calibrationResolver) return;

        const rect = elements.calibrationTarget.getBoundingClientRect();
        window.webgazer.recordScreenPosition(
            rect.left + rect.width / 2,
            rect.top + rect.height / 2,
            "click",
        );
        state.calibrationClicks += 1;

        if (state.calibrationClicks >= calibrationClicksPerPoint) {
            state.calibrationIndex += 1;
            state.calibrationClicks = 0;
        }

        if (state.calibrationIndex >= calibrationPositions.length) {
            finishCalibration(true);
            return;
        }
        positionCalibrationTarget();
    }

    function finishCalibration(success) {
        if (elements.calibrationDialog?.open) {
            elements.calibrationDialog.close();
        }
        elements.calibrationDialog?.classList.remove("is-center-target");
        state.gazeCalibrated = success;
        state.gazeAvailable = false;
        state.gazeSamples = [];

        if (success) {
            window.clearInterval(state.gazeAnimation);
            state.lastGazeAt = performance.now();
            elements.gazeMarker.classList.remove("is-away");
            updateGaze(true, false, "校正完成，正在追蹤");
        } else {
            window.webgazer?.pause();
            setGazeFallback("示範視線（不納入分析）");
        }

        const resolve = state.calibrationResolver;
        state.calibrationResolver = null;
        resolve?.(success);
    }

    function handleGazePrediction(data) {
        const signalTime = performance.now();
        updateBlinkSignal(signalTime, Boolean(data));
        if (
            !state.gazeCalibrated
            || !data
            || !Number.isFinite(data.x)
            || !Number.isFinite(data.y)
        ) {
            return;
        }

        const rect = elements.canvas.getBoundingClientRect();
        const onTarget =
            data.x >= rect.left
            && data.x <= rect.right
            && data.y >= rect.top
            && data.y <= rect.bottom;
        const now = signalTime;
        state.lastGazeAt = now;
        const materialPoint = mapGazeToMaterial(data.x, data.y);
        if (materialPoint && onTarget) {
            state.asdGazeSamples.push({
                x: materialPoint.x,
                y: materialPoint.y,
                at: now,
            });
            state.asdGazeSamples = state.asdGazeSamples.filter(
                (sample) => now - sample.at <= 6000,
            );
        }
        state.gazeSamples.push({ at: now, onTarget });
        if (state.activeRecording) {
            state.activeRecording.gazeSamples.push({ at: now, onTarget });
            state.activeRecording.lastValidGazeAt = now;
        }
        state.gazeSamples = state.gazeSamples.filter(
            (sample) => now - sample.at <= 2500,
        );

        const xPercent = Math.max(
            0,
            Math.min(100, ((data.x - rect.left) / rect.width) * 100),
        );
        const yPercent = Math.max(
            0,
            Math.min(100, ((data.y - rect.top) / rect.height) * 100),
        );
        elements.gazeMarker.style.left = `${xPercent}%`;
        elements.gazeMarker.style.top = `${yPercent}%`;
        elements.gazeMarker.classList.add("is-visible");

        if (state.gazeSamples.length < 8) {
            updateGaze(true, false);
            return;
        }

        const targetRatio =
            state.gazeSamples.filter((sample) => sample.onTarget).length
            / state.gazeSamples.length;
        updateGaze(targetRatio >= 0.6, true);
    }

    function mapGazeToMaterial(x, y) {
        const image = elements.materialImage;
        if (!image?.naturalWidth || !image?.naturalHeight) return null;
        const rect = image.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
            return null;
        }

        const scale = Math.max(
            rect.width / image.naturalWidth,
            rect.height / image.naturalHeight,
        );
        const renderedWidth = image.naturalWidth * scale;
        const renderedHeight = image.naturalHeight * scale;
        const renderedLeft = rect.left + (rect.width - renderedWidth) * 0.5;
        const renderedTop = rect.top + (rect.height - renderedHeight) * 0.46;
        return {
            x: Math.max(0, Math.min(
                image.naturalWidth,
                (x - renderedLeft) / scale,
            )),
            y: Math.max(0, Math.min(
                image.naturalHeight,
                (y - renderedTop) / scale,
            )),
        };
    }

    function pointDistance(a, b) {
        if (!a || !b) return 0;
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    function eyeAspectRatio(positions, indices) {
        const points = indices.map((index) => positions[index]);
        if (points.some((point) => !Array.isArray(point))) return null;
        const verticalOne = pointDistance(points[1], points[5]);
        const verticalTwo = pointDistance(points[2], points[4]);
        const horizontal = pointDistance(points[0], points[3]);
        if (!horizontal) return null;
        return (verticalOne + verticalTwo) / (2 * horizontal);
    }

    function updateBlinkSignal(now, hasPrediction) {
        if (!hasPrediction) {
            if (
                state.lastFaceLandmarkAt === null
                || now - state.lastFaceLandmarkAt > 2000
            ) {
                state.faceFound = false;
                state.blinkTrackingAvailable = false;
                state.blinkClosed = false;
                state.blinkClosedFrames = 0;
                state.lastFaceLandmarkAt = null;
            }
            return;
        }
        const positions = window.webgazer?.getTracker?.()?.getPositions?.();
        if (!Array.isArray(positions) || positions.length < 468) {
            if (
                state.lastFaceLandmarkAt === null
                || now - state.lastFaceLandmarkAt > 2000
            ) {
                state.faceFound = false;
                state.blinkTrackingAvailable = false;
                state.blinkClosed = false;
                state.blinkClosedFrames = 0;
                state.lastFaceLandmarkAt = null;
            }
            return;
        }
        if (
            state.lastFaceLandmarkAt !== null
            && now > state.lastFaceLandmarkAt
            && now - state.lastFaceLandmarkAt <= 250
        ) {
            state.blinkObservationSamples.push({
                at: now,
                duration: now - state.lastFaceLandmarkAt,
            });
        }
        state.faceFound = true;
        state.blinkTrackingAvailable = true;
        state.lastFaceLandmarkAt = now;
        const left = eyeAspectRatio(positions, [33, 160, 158, 133, 153, 144]);
        const right = eyeAspectRatio(positions, [362, 385, 387, 263, 373, 380]);
        if (!Number.isFinite(left) || !Number.isFinite(right)) return;

        const ear = (left + right) / 2;
        if (ear < 0.20) {
            state.blinkClosedFrames += 1;
            if (!state.blinkClosed && state.blinkClosedFrames >= 2) {
                state.blinkClosed = true;
                state.blinkEvents.push(now);
            }
        } else if (ear > 0.23) {
            state.blinkClosed = false;
            state.blinkClosedFrames = 0;
        }

        state.blinkEvents = state.blinkEvents.filter(
            (timestamp) => now - timestamp <= 60000,
        );
        state.blinkObservationSamples = state.blinkObservationSamples.filter(
            (sample) => now - sample.at <= 60000,
        );
        const rate = currentBlinkRate(now);
        elements.asdBlinkRate.textContent = Number.isFinite(rate)
            ? rate.toFixed(0)
            : "—";
    }

    function currentBlinkRate(now = performance.now()) {
        if (state.asdStartedAt === null || !state.blinkTrackingAvailable) {
            return null;
        }
        const windowStart = now - 60000;
        const observedSeconds = state.blinkObservationSamples
            .filter((sample) => sample.at >= windowStart)
            .reduce((total, sample) => total + sample.duration, 0) / 1000;
        if (observedSeconds < 5) return null;
        const count = state.blinkEvents.filter(
            (timestamp) => timestamp >= windowStart,
        ).length;
        return (count * 60) / observedSeconds;
    }

    function startASDAnalysis() {
        stopASDAnalysis({ resetDisplay: false });
        state.asdStartedAt = performance.now();
        state.asdGazeSamples = [];
        state.blinkEvents = [];
        state.blinkObservationSamples = [];
        state.blinkClosed = false;
        state.blinkClosedFrames = 0;
        state.blinkTrackingAvailable = false;
        state.lastFaceLandmarkAt = null;
        if (!state.asdUsable) {
            elements.asdStatus.textContent = "未啟用";
            return;
        }
        elements.asdStatus.textContent = "累積訊號";
        elements.asdStatus.classList.add("is-live");
        state.asdTimer = window.setInterval(runASDAnalysis, 5000);
        void runASDAnalysis();
    }

    function stopASDAnalysis({ resetDisplay = false } = {}) {
        window.clearInterval(state.asdTimer);
        state.asdTimer = null;
        state.asdAbortController?.abort();
        state.asdAbortController = null;
        state.asdBusy = false;
        state.asdStartedAt = null;
        state.asdGazeSamples = [];
        state.blinkEvents = [];
        state.blinkObservationSamples = [];
        state.blinkTrackingAvailable = false;
        state.lastFaceLandmarkAt = null;
        state.faceFound = false;
        elements.asdStatus?.classList.remove("is-live");
        if (resetDisplay) updateASDPanel(null);
    }

    async function captureAnalysisFrame() {
        if (
            !state.asdEmotionUsable
            || !elements.cameraVideo?.videoWidth
            || !elements.cameraVideo?.videoHeight
        ) {
            return null;
        }
        const canvas = document.createElement("canvas");
        const width = Math.min(480, elements.cameraVideo.videoWidth);
        const height = Math.round(
            width * elements.cameraVideo.videoHeight / elements.cameraVideo.videoWidth,
        );
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d", { alpha: false }).drawImage(
            elements.cameraVideo,
            0,
            0,
            width,
            height,
        );
        return new Promise((resolve) => {
            canvas.toBlob(resolve, "image/jpeg", 0.72);
        });
    }

    async function runASDAnalysis() {
        if (!state.sessionId || state.completed || state.asdBusy || !state.asdUsable) {
            return;
        }
        state.asdBusy = true;
        const abortController = new AbortController();
        state.asdAbortController = abortController;
        try {
            const now = performance.now();
            const samples = state.asdGazeSamples.filter(
                (sample) => now - sample.at <= 5200,
            );
            const form = new FormData();
            form.set("gaze_samples", JSON.stringify(samples));
            form.set(
                "viewport_width",
                String(elements.materialImage.naturalWidth || 1280),
            );
            form.set(
                "viewport_height",
                String(elements.materialImage.naturalHeight || 720),
            );
            const blinkRate = currentBlinkRate(now);
            form.set("blink_rate_per_min", String(blinkRate || 0));
            form.set("blink_available", String(Number.isFinite(blinkRate)));
            form.set("face_found", String(state.faceFound));
            const frame = await captureAnalysisFrame();
            if (frame) form.set("frame", frame, "emotion-frame.jpg");

            const data = await fetchJson(
                `/api/sessions/${state.sessionId}/asd-analysis`,
                { method: "POST", body: form, signal: abortController.signal },
            );
            updateASDPanel(data.analysis);
        } catch (error) {
            if (error.name !== "AbortError") {
                elements.asdStatus.textContent = "暫時無法分析";
                elements.asdStatus.classList.remove("is-live");
            }
        } finally {
            if (state.asdAbortController === abortController) {
                state.asdAbortController = null;
                state.asdBusy = false;
            }
        }
    }

    function formatProbability(value) {
        const probability = Number(value);
        return Number.isFinite(probability)
            ? Math.max(0, Math.min(100, probability * 100))
            : null;
    }

    function updateASDPanel(analysis) {
        if (!analysis) {
            elements.asdStatus.textContent = "待命";
            elements.asdEmotion.textContent = "—";
            elements.asdBlinkRate.textContent = "—";
            elements.asdSeverity.textContent = "分析中";
            elements.asdEyeState.textContent = "等待校正後的視線資料";
            for (const [label, bar] of [
                [elements.asdTdProb, elements.asdTdBar],
                [elements.asdMildProb, elements.asdMildBar],
                [elements.asdSevereProb, elements.asdSevereBar],
            ]) {
                label.textContent = "—";
                bar.style.width = "0%";
                bar.parentElement.setAttribute("aria-valuenow", "0");
            }
            return;
        }

        const statusLabels = {
            ready: "分析完成",
            partial: "部分訊號",
            collecting: "累積訊號",
            unavailable: "模組未就緒",
            disabled: "未啟用",
        };
        elements.asdStatus.textContent = statusLabels[analysis.status] || "分析中";
        elements.asdStatus.classList.toggle(
            "is-live",
            ["ready", "partial", "collecting"].includes(analysis.status),
        );
        const emotionAvailable = analysis.emotion_available
            ?? Boolean(analysis.emotion && analysis.emotion !== "unknown");
        elements.asdEmotion.textContent = emotionAvailable
            ? (analysis.emotion_zh || "分析中")
            : "—";
        const blinkRate = Number(analysis.blink_rate_per_min);
        const blinkAvailable = analysis.blink_available
            ?? Number.isFinite(blinkRate);
        elements.asdBlinkRate.textContent = blinkAvailable && Number.isFinite(blinkRate)
            ? blinkRate.toFixed(0)
            : "—";
        const classificationAvailable = analysis.classification_available
            ?? Boolean(analysis.severity && analysis.severity !== "unknown");
        elements.asdSeverity.textContent = classificationAvailable
            ? analysis.severity
            : "資料累積中";
        elements.asdEyeState.textContent = classificationAvailable && analysis.eye_state_zh
            ? `眼動：${analysis.eye_state_zh}`
            : "等待更多視線資料";

        const probabilities = analysis.probabilities || {};
        const rows = [
            [formatProbability(probabilities.td), elements.asdTdProb, elements.asdTdBar],
            [formatProbability(probabilities.mild), elements.asdMildProb, elements.asdMildBar],
            [formatProbability(probabilities.severe), elements.asdSevereProb, elements.asdSevereBar],
        ];
        for (const [value, label, bar] of rows) {
            const shownValue = classificationAvailable ? value : null;
            label.textContent = shownValue === null ? "—" : `${shownValue.toFixed(0)}%`;
            bar.style.width = shownValue === null ? "0%" : `${shownValue}%`;
            bar.parentElement.setAttribute(
                "aria-valuenow",
                shownValue === null ? "0" : shownValue.toFixed(0),
            );
        }
    }

    function stopGazeTracking() {
        window.clearInterval(state.gazeAnimation);
        window.clearInterval(state.gazeWatchdog);
        state.gazeWatchdog = null;
        state.lastGazeAt = null;
        try {
            window.webgazer?.clearGazeListener();
            window.webgazer?.removeMouseEventListeners();
            state.videoStream?.getTracks().forEach((track) => track.stop());
            if (
                window.webgazer
                && document.getElementById(window.webgazer.params.videoContainerId)
            ) {
                window.webgazer.end();
            }
        } catch (error) {
            // Resource cleanup is best-effort during navigation.
        }
        state.videoStream = null;
        elements.cameraVideo.srcObject = null;
        elements.cameraVideo.classList.remove("is-visible");
        elements.cameraPlaceholder.hidden = false;
    }

    async function recalibrateGaze() {
        if (!state.sessionId || state.completed) {
            window.showToast("請先開始互動", "default");
            return;
        }
        if (
            state.activeRecording
            || state.isTranscribing
            || state.demoRunning
            || state.isSubmittingEvent
        ) {
            window.showToast("請等目前這一句分析完成再校正", "default");
            return;
        }

        elements.recalibrate.disabled = true;
        window.clearInterval(state.gazeAnimation);
        try {
            if (!state.videoStream) {
                await startGazeTracking();
                return;
            }
            await window.webgazer.resume();
            await waitForGazeModel(10000);
            await beginGazeCalibration();
        } catch (error) {
            setGazeFallback("重新校正失敗（不納入分析）");
            window.showToast("視線重新校正失敗", "error");
        } finally {
            elements.recalibrate.disabled = false;
        }
    }

    async function startAudioCapture() {
        if (state.whisperUsable !== true) {
            setSpeechUnavailable("Whisper 尚未就緒，請使用手動輸入");
            return;
        }
        if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
            setSpeechUnavailable("瀏覽器不支援錄音");
            return;
        }

        try {
            state.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
                video: false,
            });
            state.lastUtteranceEndedAt = performance.now();
            elements.speechStatus.textContent = "點選說話者開始錄音";
            elements.speechDot.classList.add("is-on");
            updateRecordingControls();
        } catch (error) {
            setSpeechUnavailable("無法使用麥克風");
            window.showToast("請允許麥克風權限，才能使用 Whisper 轉寫", "default");
        }
    }

    function setSpeechUnavailable(message) {
        elements.speechStatus.textContent = message;
        elements.speechDot.classList.remove("is-on");
        elements.recordParent.disabled = true;
        elements.recordChild.disabled = true;
    }

    function selectRecordingMimeType() {
        if (typeof MediaRecorder.isTypeSupported !== "function") return "";
        const candidates = [
            "audio/webm;codecs=opus",
            "audio/mp4",
            "audio/webm",
        ];
        return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
    }

    function recordingExtension(mimeType) {
        if (mimeType.includes("mp4")) return "m4a";
        if (mimeType.includes("mpeg")) return "mp3";
        if (mimeType.includes("wav")) return "wav";
        return "webm";
    }

    function toggleRecording(speaker) {
        if (
            !state.sessionId
            || state.completed
            || state.demoRunning
            || state.isSubmittingEvent
        ) {
            return;
        }
        if (state.activeRecording) {
            stopRecording();
            return;
        }
        startRecording(speaker);
    }

    function startRecording(speaker) {
        if (!state.audioStream?.active) {
            window.showToast("麥克風尚未準備好", "error");
            return;
        }
        if (state.transcriptionQueue.length >= 5) {
            window.showToast("轉寫佇列已滿，請稍候一下", "default");
            return;
        }

        const mimeType = selectRecordingMimeType();
        const options = mimeType ? { mimeType } : undefined;
        try {
            state.mediaRecorder = new MediaRecorder(state.audioStream, options);
        } catch (error) {
            setSpeechUnavailable("無法建立錄音");
            window.showToast("這個瀏覽器無法建立可轉寫的錄音", "error");
            return;
        }

        const startedAt = performance.now();
        state.audioChunks = [];
        state.activeRecording = {
            speaker,
            startedAt,
            pauseBefore: state.lastUtteranceEndedAt === null
                ? 0
                : Math.min(
                    30,
                    Math.max(
                        0,
                        (startedAt - state.lastUtteranceEndedAt) / 1000,
                    ),
                ),
            gazeSamples: [],
            gazeUnavailable: !state.gazeAvailable,
            lastValidGazeAt: null,
            timeoutId: null,
        };

        state.mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data.size > 0) state.audioChunks.push(event.data);
        });
        state.mediaRecorder.addEventListener("error", () => {
            window.showToast("錄音中斷，請再試一次", "error");
            stopRecording();
        });
        state.mediaRecorder.addEventListener("stop", handleRecordingStopped, {
            once: true,
        });
        try {
            state.mediaRecorder.start();
        } catch (error) {
            state.mediaRecorder = null;
            state.activeRecording = null;
            state.audioChunks = [];
            setSpeechUnavailable("無法開始錄音");
            window.showToast("錄音啟動失敗，請重新允許麥克風", "error");
            return;
        }
        state.activeRecording.timeoutId = window.setTimeout(stopRecording, 30000);
        elements.speechStatus.textContent =
            speaker === "child" ? "正在錄孩子說話" : "正在錄家長說話";
        updateRecordingControls();
    }

    function stopRecording() {
        if (!state.activeRecording || !state.mediaRecorder) return;
        window.clearTimeout(state.activeRecording.timeoutId);
        state.activeRecording.endedAt = performance.now();
        state.lastUtteranceEndedAt = state.activeRecording.endedAt;
        if (state.mediaRecorder.state !== "inactive") {
            state.mediaRecorder.stop();
        }
    }

    function handleRecordingStopped() {
        const recording = state.activeRecording;
        const recorder = state.mediaRecorder;
        const chunks = state.audioChunks.slice();
        state.activeRecording = null;
        state.mediaRecorder = null;
        state.audioChunks = [];

        if (!recording || !recorder) {
            updateRecordingControls();
            return;
        }

        if ((recording.endedAt || performance.now()) - recording.startedAt < 350) {
            elements.speechStatus.textContent = "錄音太短，請再試一次";
            updateRecordingControls();
            return;
        }

        const mimeType = recorder.mimeType || chunks[0]?.type || "audio/webm";
        const audio = new Blob(chunks, { type: mimeType });
        if (!audio.size) {
            elements.speechStatus.textContent = "沒有收到錄音，請再試一次";
            updateRecordingControls();
            return;
        }

        const gazeIsRecent =
            recording.lastValidGazeAt !== null
            && (recording.endedAt || performance.now())
                - recording.lastValidGazeAt <= 2000;
        const gazeAvailable =
            state.gazeCalibrated
            && !recording.gazeUnavailable
            && recording.gazeSamples.length >= 8
            && gazeIsRecent;
        const gazeOnTarget = gazeAvailable
            ? recording.gazeSamples.filter((sample) => sample.onTarget).length
                / recording.gazeSamples.length >= 0.6
            : true;

        state.transcriptionQueue.push({
            recording,
            audio,
            mimeType,
            gazeAvailable,
            gazeOnTarget,
        });
        if (state.isTranscribing) {
            elements.speechStatus.textContent =
                `Whisper 轉寫中，另有 ${state.transcriptionQueue.length} 句排隊`;
        }
        updateRecordingControls();
        void processTranscriptionQueue();
    }

    async function processTranscriptionQueue() {
        if (state.isTranscribing) return;
        state.isTranscribing = true;
        updateRecordingControls();

        while (state.transcriptionQueue.length && !state.completed) {
            const job = state.transcriptionQueue.shift();
            if (!state.activeRecording) {
                const waiting = state.transcriptionQueue.length;
                elements.speechStatus.textContent = waiting
                    ? `Whisper 轉寫中，另有 ${waiting} 句排隊`
                    : "Whisper 轉寫中，仍可錄下一句";
            }
            updateRecordingControls();

            const form = new FormData();
            form.append(
                "audio",
                job.audio,
                `utterance.${recordingExtension(job.mimeType)}`,
            );
            form.append("speaker", job.recording.speaker);
            form.append(
                "pause_before",
                job.recording.pauseBefore.toFixed(3),
            );
            form.append("gaze_available", String(job.gazeAvailable));
            form.append("gaze_on_target", String(job.gazeOnTarget));

            try {
                const data = await fetchJson(
                    `/api/sessions/${state.sessionId}/transcriptions`,
                    { method: "POST", body: form },
                );
                if (data.status === "no_speech" || !data.event) {
                    window.showToast(
                        "沒有偵測到清楚語音，請靠近麥克風再試一次",
                        "default",
                    );
                } else {
                    applyEventResponse(data);
                    const speakerLabel =
                        job.recording.speaker === "child" ? "孩子" : "家長";
                    const processingSeconds = Number(
                        data.transcription?.processing_seconds,
                    );
                    const timingCopy = Number.isFinite(processingSeconds)
                        ? `（${processingSeconds.toFixed(1)} 秒）`
                        : "";
                    window.showToast(
                        `${speakerLabel}的語音已轉成文字${timingCopy}`,
                        "success",
                    );
                }
            } catch (error) {
                window.showToast(error.message, "error");
            }
        }

        state.isTranscribing = false;
        if (!state.activeRecording && !state.completed) {
            elements.speechStatus.textContent = "點選說話者開始錄音";
        }
        updateRecordingControls();
    }

    function updateRecordingControls() {
        const canRecord =
            Boolean(state.sessionId)
            && !state.completed
            && Boolean(state.audioStream?.active)
            && !state.demoRunning
            && !state.isSubmittingEvent
            && state.transcriptionQueue.length < 5;
        const activeSpeaker = state.activeRecording?.speaker || null;

        elements.recordParent.disabled = !canRecord || activeSpeaker === "child";
        elements.recordChild.disabled = !canRecord || activeSpeaker === "parent";
        elements.recordParent.classList.toggle(
            "is-recording",
            activeSpeaker === "parent",
        );
        elements.recordChild.classList.toggle(
            "is-recording",
            activeSpeaker === "child",
        );
        elements.recordParent.setAttribute(
            "aria-pressed",
            String(activeSpeaker === "parent"),
        );
        elements.recordChild.setAttribute(
            "aria-pressed",
            String(activeSpeaker === "child"),
        );
        elements.recordParent.querySelector("b").textContent =
            activeSpeaker === "parent"
                ? "停止並轉寫"
                : "家長說話";
        elements.recordChild.querySelector("b").textContent =
            activeSpeaker === "child"
                ? "停止並轉寫"
                : "孩子說話";
        if (state.sessionId && !state.completed) {
            elements.finish.disabled =
                Boolean(activeSpeaker)
                || state.isTranscribing
                || state.transcriptionQueue.length > 0
                || state.demoRunning
                || state.isSubmittingEvent;
        }
        if (elements.eventSubmit) {
            elements.eventSubmit.disabled =
                state.isTranscribing
                || state.transcriptionQueue.length > 0
                || state.demoRunning
                || state.isSubmittingEvent;
        }
        if (state.sessionId && !state.completed) {
            elements.recalibrate.disabled =
                Boolean(activeSpeaker)
                || state.isTranscribing
                || state.transcriptionQueue.length > 0
                || state.demoRunning
                || state.isSubmittingEvent;
        }
        if (state.sessionId && !state.completed && !state.demoRunning) {
            elements.demo.disabled =
                Boolean(activeSpeaker)
                || state.isTranscribing
                || state.transcriptionQueue.length > 0
                || state.isSubmittingEvent;
        }
    }

    function stopAudioCapture() {
        if (state.activeRecording && state.mediaRecorder?.state !== "inactive") {
            window.clearTimeout(state.activeRecording.timeoutId);
            state.mediaRecorder.stop();
        }
        state.audioStream?.getTracks().forEach((track) => track.stop());
        state.audioStream = null;
        state.activeRecording = null;
        state.mediaRecorder = null;
        state.transcriptionQueue = [];
        state.isTranscribing = false;
        updateRecordingControls();
    }

    function startTimer() {
        updateTimer();
        state.timerId = window.setInterval(updateTimer, 1000);
    }

    function updateTimer() {
        const elapsed = Math.floor((Date.now() - state.startedAt) / 1000);
        const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const seconds = String(elapsed % 60).padStart(2, "0");
        elements.timer.textContent = `${minutes}:${seconds}`;
    }

    async function runDemo() {
        if (!state.sessionId || state.completed) return;
        clearDemoTimers();
        const runId = state.demoRunId;
        state.demoRunning = true;
        elements.demo.disabled = true;
        elements.demo.textContent = "示範播放中…";
        updateRecordingControls();

        try {
            for (const item of buildDemoScript()) {
                const shouldContinue = await waitForDemoDelay(item.delay, runId);
                if (!shouldContinue || state.completed) return;
                await submitEvent(item);
            }
        } finally {
            if (runId === state.demoRunId) {
                state.demoRunning = false;
                if (!state.completed) {
                    elements.demo.disabled = false;
                    elements.demo.textContent = "再播放一次示範";
                }
            }
            updateRecordingControls();
        }

        if (runId !== state.demoRunId || state.completed) return;
        window.showToast("示範完成，可以查看下方分析", "success");
    }

    function waitForDemoDelay(delay, runId) {
        return new Promise((resolve) => {
            const pending = {
                timer: null,
                resolve,
            };
            pending.timer = window.setTimeout(() => {
                state.demoTimers = state.demoTimers.filter(
                    (entry) => entry !== pending,
                );
                resolve(runId === state.demoRunId);
            }, delay);
            state.demoTimers.push(pending);
        });
    }

    async function submitEvent({
        speaker,
        text,
        pause,
        gaze,
        gazeAvailable = true,
    }) {
        if (!state.sessionId || !text || state.isSubmittingEvent) return false;
        state.isSubmittingEvent = true;
        updateRecordingControls();
        try {
            const data = await fetchJson(`/api/sessions/${state.sessionId}/events`, {
                method: "POST",
                body: JSON.stringify({
                    speaker,
                    text,
                    pause_before: Number(pause || 0),
                    gaze_on_target: gaze,
                    gaze_available: gazeAvailable,
                }),
            });
            applyEventResponse(data);
            return true;
        } catch (error) {
            window.showToast(error.message, "error");
            return false;
        } finally {
            state.isSubmittingEvent = false;
            updateRecordingControls();
        }
    }

    function applyEventResponse(data) {
        state.metrics = data.metrics;
        state.lastEventAt = Date.now();
        state.latestEventId = data.event.id;
        appendEvent(data.event);
        updateMetrics(data.metrics);
        if (data.coach_pending) {
            showPendingCoach();
            elements.ruleStatus.textContent = pendingCoachStatus();
            scheduleCoachRefinement(data.event.id);
        } else {
            updateHud(data.suggestion, data.coach_source);
            elements.ruleStatus.textContent = completedCoachStatus(
                data.coach_source,
            );
        }
    }

    function scheduleCoachRefinement(eventId) {
        state.pendingCoachRefinement = {
            eventId,
            sessionId: state.sessionId,
        };
        void processCoachRefinementQueue();
    }

    async function processCoachRefinementQueue() {
        if (state.coachRefinementBusy) return;
        state.coachRefinementBusy = true;

        try {
            while (state.pendingCoachRefinement && !state.completed) {
                const job = state.pendingCoachRefinement;
                state.pendingCoachRefinement = null;
                try {
                    const data = await fetchJson(
                        `/api/sessions/${job.sessionId}/events/`
                            + `${job.eventId}/coach-refinement`,
                        { method: "POST", body: "{}" },
                    );
                    if (
                        !state.completed
                        && job.eventId === state.latestEventId
                    ) {
                        updateHud(data.suggestion, data.coach_source);
                        updateEventBadge(job.eventId, data.suggestion);
                        elements.ruleStatus.textContent = completedCoachStatus(
                            data.coach_source,
                            { background: true },
                        );
                    }
                } catch (error) {
                    if (
                        !state.completed
                        && job.eventId === state.latestEventId
                    ) {
                        elements.ruleStatus.textContent = "模型生成失敗；尚未顯示提示";
                    }
                }
            }
        } finally {
            state.coachRefinementBusy = false;
            if (state.pendingCoachRefinement && !state.completed) {
                void processCoachRefinementQueue();
            }
        }
    }

    function appendEvent(event) {
        elements.dialogueEmpty?.remove();
        const item = document.createElement("div");
        item.className = `dialogue-event ${event.speaker === "child" ? "is-child" : ""}`;
        item.dataset.eventId = String(event.id);

        const speaker = event.speaker === "child" ? "孩子" : "家長";
        const avatar = event.speaker === "child" ? "孩" : "家";
        let badge = "";
        const responseMode = event.analysis?.suggestion?.response_mode || "";
        if (
            responseMode === "safety_check"
            || event.analysis?.emotional_bid?.category === "urgent_safety"
        ) {
            badge = '<span class="event-badge is-safety">先確認安全</span>';
        } else if (
            responseMode === "repair_connection"
            || event.analysis?.emotional_bid?.active
        ) {
            badge = '<span class="event-badge">先接住情緒</span>';
        } else if (event.analysis.wait_met === true) {
            badge = '<span class="event-badge">等待剛好</span>';
        } else if (event.analysis.expansion_met === true) {
            badge = '<span class="event-badge">成功擴展</span>';
        }

        item.innerHTML = `
            <span class="speaker-avatar">${avatar}</span>
            <div class="dialogue-copy">
                <span>${speaker} · 前段等待 ${Number(event.pause_before).toFixed(1)} 秒</span>
                <strong></strong>
            </div>
            ${badge}
        `;
        item.querySelector("strong").textContent = event.text;
        elements.dialogue.appendChild(item);
        item.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function updateEventBadge(eventId, suggestion) {
        const item = elements.dialogue.querySelector(
            `.dialogue-event[data-event-id="${String(eventId)}"]`,
        );
        if (!item) return;
        const mode = suggestion?.response_mode || "";
        if (!['repair_connection', 'safety_check'].includes(mode)) return;
        let badge = item.querySelector(".event-badge");
        if (!badge) {
            badge = document.createElement("span");
            badge.className = "event-badge";
            item.appendChild(badge);
        }
        const isSafety = mode === "safety_check";
        badge.classList.toggle("is-safety", isSafety);
        badge.textContent = isSafety ? "先確認安全" : "先接住情緒";
    }

    function updateMetrics(metrics) {
        const waitProgress = Math.min(100, (metrics.average_wait / 5) * 100);
        elements.waitValue.textContent = metrics.average_wait ? `${metrics.average_wait}s` : "—";
        elements.expandValue.textContent = metrics.expansion_rate ? `${metrics.expansion_rate}%` : "0%";
        elements.turnValue.textContent = metrics.turn_taking_rate ? `${metrics.turn_taking_rate}%` : "0%";
        elements.waitRing.style.setProperty("--progress", waitProgress);
        elements.expandRing.style.setProperty("--progress", metrics.expansion_rate);
        elements.turnRing.style.setProperty("--progress", metrics.turn_taking_rate);
    }

    function updateHud(suggestion, source = suggestion.source) {
        const responseMode = suggestion.response_mode || "";
        const isSafety = responseMode === "safety_check"
            || suggestion.eyebrow === "安全優先";
        const isRelationship = responseMode === "repair_connection"
            || suggestion.eyebrow === "情緒接住"
            || suggestion.eyebrow === "關係回應";
        const isOffMaterial = isSafety || isRelationship;
        elements.hud.dataset.tone = suggestion.tone || "coach";
        elements.hud.dataset.mode = isSafety
            ? "safety"
            : (isRelationship ? "relationship" : "picture");
        elements.hud.classList.remove("is-waiting");
        elements.materialCaption?.classList.remove("is-waiting");
        const sourceMeta = coachSourceMeta(suggestion, source);
        if (elements.hudSource) {
            elements.hudSource.dataset.source = sourceMeta.source;
            elements.hudSource.textContent = sourceMeta.label;
        }
        elements.hudEyebrow.textContent = suggestion.eyebrow;
        elements.hudTitle.textContent = suggestion.title;
        elements.hudMessage.textContent = suggestion.message;
        // A second sentence beside the coaching card is distracting during
        // ordinary picture-book turns and tends to repeat material details.
        // Keep direct parent wording only when relationship or safety support
        // makes it genuinely useful.
        elements.hudExample.textContent = isOffMaterial
            ? (suggestion.example || "")
            : "";
        elements.materialPrompt.textContent = suggestion.practice_prompt || "";
        if (elements.materialPromptLabel) {
            elements.materialPromptLabel.textContent = isSafety
                ? "先確認安全 · 可以直接這樣說"
                : (isRelationship
                    ? "先接住孩子 · 可以直接這樣說"
                    : "練習提示 · 可以直接這樣問");
        }
        elements.materialCaption?.classList.toggle(
            "is-relationship",
            isOffMaterial,
        );
        elements.materialCaption?.classList.toggle("is-safety", isSafety);
        elements.hud.classList.remove("is-pulsing");
        window.requestAnimationFrame(() => elements.hud.classList.add("is-pulsing"));
    }

    function updateGaze(onTarget, available = true, unavailableMessage = "") {
        state.gazeAvailable = available;
        state.gazeOnTarget = onTarget;
        if (!available) {
            elements.gazeStatus.textContent =
                unavailableMessage || "視線資料不足";
            elements.gazeDot.classList.remove("is-on", "is-away");
            elements.gazeMarker.classList.remove("is-away");
            return;
        }
        elements.gazeStatus.textContent = onTarget ? "注視教材" : "視線暫時離開";
        elements.gazeDot.classList.toggle("is-on", onTarget);
        elements.gazeDot.classList.toggle("is-away", !onTarget);
        elements.gazeMarker.classList.toggle("is-away", !onTarget);
    }

    async function finishSession() {
        if (
            !state.sessionId
            || state.completed
            || state.activeRecording
            || state.isTranscribing
            || state.transcriptionQueue.length > 0
            || state.demoRunning
            || state.isSubmittingEvent
        ) {
            return;
        }
        elements.finish.disabled = true;
        try {
            await fetchJson(`/api/sessions/${state.sessionId}/finish`, {
                method: "POST",
                body: "{}",
            });
            state.completed = true;
            state.pendingCoachRefinement = null;
            clearDemoTimers();
            window.clearInterval(state.timerId);
            stopAudioCapture();
            stopASDAnalysis({ resetDisplay: false });
            stopGazeTracking();
            elements.sessionState.textContent = "已完成";
            elements.sessionState.parentElement.classList.remove("is-live");
            elements.start.innerHTML = "互動已完成";
            elements.demo.disabled = true;
            elements.signalSummary.textContent = "已儲存";

            elements.finishWait.textContent = state.metrics.average_wait
                ? `${state.metrics.average_wait} 秒`
                : "—";
            elements.finishExpand.textContent = `${state.metrics.expansion_rate}%`;
            elements.finishTurn.textContent = `${state.metrics.turn_taking_rate}%`;
            elements.finishDialog.showModal();
        } catch (error) {
            elements.finish.disabled = false;
            window.showToast(error.message, "error");
        }
    }

    function clearDemoTimers() {
        state.demoRunId += 1;
        state.demoRunning = false;
        state.demoTimers.forEach((pending) => {
            window.clearTimeout(pending.timer);
            pending.resolve(false);
        });
        state.demoTimers = [];
        updateRecordingControls();
    }

    async function checkSystemHealth() {
        try {
            const health = await fetchJson("/api/health");
            const whisper = health.modules?.whisper;
            const coachProvider = health.modules?.coach_provider;
            const ollama = health.modules?.ollama;
            const asdAnalysis = health.modules?.asd_analysis;

            if (!state.sessionId && whisper) {
                if (whisper.status === "ready") {
                    state.whisperUsable = true;
                    elements.speechStatus.textContent = whisper.warming
                        ? "Whisper 背景預熱中"
                        : "Whisper 已就緒";
                    elements.speechDot.classList.add("is-on");
                } else if (whisper.status === "model_not_cached") {
                    state.whisperUsable = true;
                    elements.speechStatus.textContent = "首次轉寫會準備模型";
                } else if (whisper.status === "dependency_missing") {
                    state.whisperUsable = false;
                    elements.speechStatus.textContent = "Whisper 套件尚未安裝";
                } else if (whisper.status === "disabled") {
                    state.whisperUsable = false;
                    elements.speechStatus.textContent = "Whisper 已停用";
                } else {
                    state.whisperUsable = false;
                    elements.speechStatus.textContent = "Whisper 尚未就緒";
                }
            }

            if (asdAnalysis) {
                state.asdUsable = ["ready", "partial"].includes(
                    asdAnalysis.status,
                );
                state.asdEmotionUsable = Boolean(
                    asdAnalysis.emotion_ready
                    || asdAnalysis.emotion?.status === "ready",
                );
                elements.asdStatus.textContent = state.asdUsable
                    ? "系統就緒"
                    : asdAnalysis.status === "disabled"
                        ? "未啟用"
                        : "模組未就緒";
            } else {
                state.asdUsable = false;
                state.asdEmotionUsable = false;
                elements.asdStatus.textContent = "模組未就緒";
            }

            updateCoachProviderHealth(coachProvider, ollama);
        } catch (error) {
            state.whisperUsable = false;
            if (!state.sessionId) {
                elements.speechStatus.textContent = "無法確認 Whisper 狀態";
            }
            state.coachProviderHealth = null;
            elements.ruleStatus.textContent = "Gemini 狀態無法確認，使用本機備援";
            state.asdUsable = false;
            state.asdEmotionUsable = false;
            elements.asdStatus.textContent = "無法確認狀態";
        }
    }

    function toggleEntry() {
        elements.eventEntry.hidden = !elements.eventEntry.hidden;
        if (!elements.eventEntry.hidden) elements.eventText.focus();
    }

    async function handleManualEntry(event) {
        event.preventDefault();
        if (!state.sessionId) {
            window.showToast("請先開始互動", "default");
            return;
        }
        const text = elements.eventText.value.trim();
        if (!text) return;
        if (
            state.isTranscribing
            || state.demoRunning
            || state.isSubmittingEvent
        ) {
            window.showToast("請等目前的分析完成", "default");
            return;
        }
        await submitEvent({
            speaker: elements.eventSpeaker.value,
            text,
            pause: Number(elements.eventPause.value || 0),
            gaze: state.gazeOnTarget,
            gazeAvailable: state.gazeAvailable,
        });
        elements.eventText.value = "";
    }

    elements.start?.addEventListener("click", startSession);
    elements.demo?.addEventListener("click", runDemo);
    elements.finish?.addEventListener("click", finishSession);
    elements.recordParent?.addEventListener("click", () => {
        toggleRecording("parent");
    });
    elements.recordChild?.addEventListener("click", () => {
        toggleRecording("child");
    });
    elements.calibrationTarget?.addEventListener("click", handleCalibrationClick);
    elements.skipCalibration?.addEventListener("click", () => {
        finishCalibration(false);
    });
    elements.calibrationDialog?.addEventListener("cancel", (event) => {
        event.preventDefault();
        finishCalibration(false);
    });
    elements.recalibrate?.addEventListener("click", recalibrateGaze);
    elements.materialSelect?.addEventListener("change", (event) => {
        selectMaterial(event.target.value);
        state.asdGazeSamples = [];
    });
    elements.toggleEntry?.addEventListener("click", toggleEntry);
    elements.eventEntry?.addEventListener("submit", handleManualEntry);
    elements.focus?.addEventListener("click", () => {
        document.body.classList.toggle("focus-mode");
    });
    elements.closeFinish?.addEventListener("click", () => {
        elements.finishDialog.close();
    });

    selectMaterial(state.selectedMaterial?.id, { updateUrl: false });
    state.healthPromise = checkSystemHealth();

    window.addEventListener("beforeunload", () => {
        state.pendingCoachRefinement = null;
        stopAudioCapture();
        stopASDAnalysis({ resetDisplay: false });
        stopGazeTracking();
        clearDemoTimers();
    });
})();
