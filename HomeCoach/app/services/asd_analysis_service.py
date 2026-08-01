"""Graceful, session-scoped adapter for the experimental ASD v4 models.

The severity output is an experimental model signal.  It is deliberately
marked as non-diagnostic in every response and must not be used as a medical
diagnosis or risk assessment.
"""

from __future__ import annotations

import copy
import io
import importlib.util
import math
import pickle
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


class ASDAnalysisService:
    """Analyze calibrated WebGazer samples with the local ASD v4 artifacts."""

    TARGET_WIDTH = 1280.0
    TARGET_HEIGHT = 720.0
    FPS = 30.0
    FIXATION_THRESHOLD_PX = 30.0
    MIN_FIXATION_MS = 80.0
    MAX_SAMPLE_GAP_MS = 250.0
    MAX_EMOTION_PIXELS = 4_000_000

    FEATURE_COLUMNS = (
        "sp_fix_count",
        "sp_fix_duration_ms_total",
        "sp_fix_duration_ms_mean",
        "sp_fix_duration_ms_var",
        "sp_len_px_total",
        "sp_saccade_amplitude_px_mean",
        "sp_saccade_amplitude_px_var",
        "sp_distance_to_centre_px_mean",
        "sp_distance_to_centre_px_var",
        "sp_distance_to_sp_mean_px_mean",
        "sp_distance_to_sp_mean_px_var",
    )

    DEFAULT_CLASS_LABELS = {
        0: "TD",
        1: "輕度ASD",
        2: "重度ASD",
    }

    EYE_STATE_ZH = {
        "focused": "專注",
        "avoidant": "迴避",
        "avoidant_mild": "輕度迴避",
        "hyperscanning": "過度掃視",
        "transitional": "轉換中",
        "unknown": "分析中",
    }

    PROBABILITY_KEYS = {
        0: "td",
        1: "mild",
        2: "severe",
    }

    SEVERITY_DEPENDENCIES = ("numpy", "sklearn", "xgboost")
    EMOTION_DEPENDENCIES = (
        "numpy",
        "cv2",
        "deepface",
        "tensorflow",
        "tf_keras",
        "PIL",
    )

    def __init__(
        self,
        model_path,
        emotion_module_path,
        enabled=True,
        emotion_enabled=True,
    ):
        self.model_path = (
            Path(model_path).expanduser().resolve() if model_path else None
        )
        self.emotion_module_path = (
            Path(emotion_module_path).expanduser().resolve()
            if emotion_module_path
            else None
        )
        self.enabled = bool(enabled)
        self.emotion_enabled = bool(emotion_enabled)

        self._model = None
        self._feature_cols: tuple[str, ...] = ()
        self._class_labels = dict(self.DEFAULT_CLASS_LABELS)
        self._severity_load_error: str | None = None

        self._emotion_module: ModuleType | None = None
        self._emotion_module_error: str | None = None
        self._emotion_analyzers: dict[str, Any] = {}

        self._latest: dict[str, dict[str, Any]] = {}
        self._session_locks: dict[str, threading.RLock] = {}

        self._model_lock = threading.RLock()
        self._emotion_lock = threading.RLock()
        self._state_lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        """Report artifact/dependency readiness without loading either model."""

        if not self.enabled:
            return {
                "status": "disabled",
                "severity": {
                    "status": "disabled",
                    "model_path": self._path_text(self.model_path),
                    "loaded": self._model is not None,
                    "missing_dependencies": [],
                },
                "emotion": {
                    "status": "disabled",
                    "module_path": self._path_text(self.emotion_module_path),
                    "loaded": self._emotion_module is not None,
                    "missing_dependencies": [],
                },
                "non_diagnostic": True,
            }

        severity_missing = self._missing_dependencies(
            self.SEVERITY_DEPENDENCIES
        )
        severity_path_ready = bool(
            self.model_path and self.model_path.is_file()
        )
        severity_status = (
            "ready"
            if (
                severity_path_ready
                and not severity_missing
                and not self._severity_load_error
            )
            else "unavailable"
        )

        if not self.emotion_enabled:
            emotion_status = "disabled"
            emotion_missing: list[str] = []
        else:
            emotion_missing = self._missing_dependencies(
                self.EMOTION_DEPENDENCIES
            )
            emotion_path_ready = bool(
                self.emotion_module_path
                and self.emotion_module_path.is_file()
            )
            emotion_status = (
                "ready"
                if (
                    emotion_path_ready
                    and not emotion_missing
                    and not self._emotion_module_error
                )
                else "unavailable"
            )

        available_components = sum(
            component_status == "ready"
            for component_status in (severity_status, emotion_status)
        )
        if severity_status == "ready" and emotion_status in {
            "ready",
            "disabled",
        }:
            overall_status = "ready"
        elif available_components:
            overall_status = "partial"
        else:
            overall_status = "unavailable"

        return {
            "status": overall_status,
            "severity": {
                "status": severity_status,
                "model_path": self._path_text(self.model_path),
                "loaded": self._model is not None,
                "missing_dependencies": severity_missing,
                "error": self._severity_load_error,
            },
            "emotion": {
                "status": emotion_status,
                "module_path": self._path_text(self.emotion_module_path),
                "loaded": self._emotion_module is not None,
                "missing_dependencies": emotion_missing,
                "error": self._emotion_module_error,
            },
            "non_diagnostic": True,
        }

    def analyze(
        self,
        session_id,
        gaze_samples,
        viewport_width,
        viewport_height,
        blink_rate_per_min,
        face_found,
        frame_bytes=None,
        blink_available=None,
    ) -> dict[str, Any]:
        """Return a JSON-safe severity/emotion snapshot for one session."""

        session_key = self._session_key(session_id)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            return self._analyze_locked(
                session_key=session_key,
                gaze_samples=gaze_samples,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                blink_rate_per_min=blink_rate_per_min,
                blink_available=(
                    blink_rate_per_min is not None
                    if blink_available is None
                    else blink_available
                ),
                face_found=face_found,
                frame_bytes=frame_bytes,
            )

    def get_latest(self, session_id) -> dict[str, Any] | None:
        """Return a defensive copy of the latest result for ``session_id``."""

        session_key = self._session_key(session_id)
        with self._state_lock:
            latest = self._latest.get(session_key)
            return copy.deepcopy(latest) if latest is not None else None

    def finish_session(self, session_id) -> dict[str, Any] | None:
        """Remove per-session model history and return its last snapshot."""

        session_key = self._session_key(session_id)
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            with self._state_lock:
                latest = self._latest.pop(session_key, None)
                self._emotion_analyzers.pop(session_key, None)
                # Retain this tiny lock object. Removing it while another
                # caller is waiting could create a second lock for the same
                # session and allow concurrent mutation of its history.
        return copy.deepcopy(latest) if latest is not None else None

    def _analyze_locked(
        self,
        session_key: str,
        gaze_samples,
        viewport_width,
        viewport_height,
        blink_rate_per_min,
        blink_available,
        face_found,
        frame_bytes,
    ) -> dict[str, Any]:
        errors: list[str] = []
        previous = self.get_latest(session_key) or {}
        result = self._empty_result(
            blink_rate_per_min=blink_rate_per_min,
            blink_available=blink_available,
            face_found=face_found,
            previous=previous,
        )

        if not self.enabled:
            result["status"] = "unavailable"
            result["errors"].append("ASD analysis is disabled")
            self._store_latest(session_key, result)
            return copy.deepcopy(result)

        try:
            normalized_gazes, ignored_count = self._normalize_gaze_samples(
                gaze_samples,
                viewport_width,
                viewport_height,
            )
            if ignored_count:
                errors.append(
                    f"ignored {ignored_count} invalid gaze sample(s)"
                )
            fixation_centres, fixation_durations = self._extract_fixations(
                normalized_gazes
            )
            features = self._compute_features(
                fixation_centres,
                fixation_durations,
            )
            result["features"] = features
            if features is None:
                result["status"] = "collecting"
            else:
                severity_result = self._classify(
                    features,
                    fixation_centres,
                )
                result.update(severity_result)
                result["classification_available"] = True
                result["status"] = "ready"
        except Exception as exc:  # graceful degradation is part of the API
            errors.append(self._error_text("severity", exc))
            result["status"] = "partial"

        if frame_bytes is not None and self.emotion_enabled:
            try:
                emotion_result = self._analyze_emotion(
                    session_key,
                    frame_bytes,
                )
                result["emotion"] = emotion_result["emotion"]
                result["emotion_zh"] = emotion_result["emotion_zh"]
                result["emotion_available"] = True
                result["emotion_updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat(timespec="seconds")
            except Exception as exc:
                errors.append(self._error_text("emotion", exc))
        elif frame_bytes is not None and not self.emotion_enabled:
            errors.append("emotion analysis is disabled")

        if errors:
            result["errors"] = errors
            if result["status"] in {"ready", "collecting"}:
                result["status"] = "partial"

        self._store_latest(session_key, result)
        return copy.deepcopy(result)

    def _empty_result(
        self,
        blink_rate_per_min,
        blink_available,
        face_found,
        previous,
    ) -> dict[str, Any]:
        blink_rate = self._finite_float(blink_rate_per_min)
        return {
            "status": "collecting",
            "severity": "unknown",
            "classification_available": False,
            "probabilities": {
                "td": 0.0,
                "mild": 0.0,
                "severe": 0.0,
            },
            "eye_state": "unknown",
            "eye_state_zh": self.EYE_STATE_ZH["unknown"],
            "emotion": str(previous.get("emotion") or "unknown"),
            "emotion_zh": str(previous.get("emotion_zh") or "分析中"),
            "emotion_available": False,
            "emotion_updated_at": previous.get("emotion_updated_at"),
            "blink_rate_per_min": max(0.0, blink_rate or 0.0),
            "blink_available": bool(blink_available),
            "face_found": bool(face_found),
            "features": None,
            "non_diagnostic": True,
            "errors": [],
        }

    def _normalize_gaze_samples(
        self,
        gaze_samples,
        viewport_width,
        viewport_height,
    ) -> tuple[list[tuple[float, float, float | None]], int]:
        width = self._finite_float(viewport_width)
        height = self._finite_float(viewport_height)
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("viewport width and height must be positive")

        if gaze_samples is None:
            return [], 0
        if isinstance(gaze_samples, (str, bytes, bytearray, dict)):
            raise TypeError("gaze_samples must be an iterable of x/y samples")

        normalized: list[tuple[float, float, float | None]] = []
        ignored_count = 0
        try:
            samples = iter(gaze_samples)
        except TypeError as exc:
            raise TypeError("gaze_samples must be iterable") from exc

        for sample in samples:
            coordinates = self._sample_coordinates(sample)
            if coordinates is None:
                ignored_count += 1
                continue
            x, y, timestamp = coordinates
            if x < 0 or x > width or y < 0 or y > height:
                ignored_count += 1
                continue
            normalized.append(
                (
                    x * self.TARGET_WIDTH / width,
                    y * self.TARGET_HEIGHT / height,
                    timestamp,
                )
            )
        return normalized, ignored_count

    def _sample_coordinates(
        self,
        sample,
    ) -> tuple[float, float, float | None] | None:
        if isinstance(sample, dict):
            x = sample.get("x", sample.get("gaze_x"))
            y = sample.get("y", sample.get("gaze_y"))
            timestamp = sample.get("at", sample.get("timestamp"))
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            x, y = sample[0], sample[1]
            timestamp = sample[2] if len(sample) >= 3 else None
        else:
            return None

        x_value = self._finite_float(x)
        y_value = self._finite_float(y)
        if x_value is None or y_value is None:
            return None
        timestamp_value = self._finite_float(timestamp)
        return x_value, y_value, timestamp_value

    def _extract_fixations(
        self,
        gazes: list[tuple[float, ...]],
    ) -> tuple[list[tuple[float, float]], list[float]]:
        if not gazes:
            return [], []

        fixation_centres: list[tuple[float, float]] = []
        fixation_durations: list[float] = []
        current_fixation = [gazes[0]]
        previous_gaze = gazes[0]

        for gaze in gazes[1:]:
            timestamp_gap = self._timestamp_gap_ms(previous_gaze, gaze)
            continuous = (
                timestamp_gap is None
                or 0 <= timestamp_gap <= self.MAX_SAMPLE_GAP_MS
            )
            if (
                continuous
                and self._distance(gaze, previous_gaze)
                < self.FIXATION_THRESHOLD_PX
            ):
                current_fixation.append(gaze)
            else:
                self._append_fixation(
                    current_fixation,
                    fixation_centres,
                    fixation_durations,
                )
                current_fixation = [gaze]
            previous_gaze = gaze

        # The ASD v4 implementation includes the active fixation at the end of
        # a window, so the final segment must also be considered.
        self._append_fixation(
            current_fixation,
            fixation_centres,
            fixation_durations,
        )
        return fixation_centres, fixation_durations

    def _append_fixation(
        self,
        samples: list[tuple[float, ...]],
        centres: list[tuple[float, float]],
        durations: list[float],
    ) -> None:
        duration_ms = self._fixation_duration_ms(samples)
        if duration_ms < self.MIN_FIXATION_MS:
            return
        centres.append(
            (
                self._mean([point[0] for point in samples]),
                self._mean([point[1] for point in samples]),
            )
        )
        durations.append(duration_ms)

    def _fixation_duration_ms(
        self,
        samples: list[tuple[float, ...]],
    ) -> float:
        fallback = len(samples) * (1000.0 / self.FPS)
        if len(samples) < 2 or any(len(sample) < 3 for sample in samples):
            return fallback

        timestamps = [self._finite_float(sample[2]) for sample in samples]
        if any(timestamp is None for timestamp in timestamps):
            return fallback
        gaps = [
            timestamps[index] - timestamps[index - 1]
            for index in range(1, len(timestamps))
        ]
        if not gaps or any(
            gap <= 0 or gap > self.MAX_SAMPLE_GAP_MS for gap in gaps
        ):
            return fallback

        # Include one estimated final sample interval, matching the original
        # ASD v4 frame-count definition while honoring WebGazer's real cadence.
        tail_interval = self._mean(gaps)
        return float(timestamps[-1] - timestamps[0] + tail_interval)

    @classmethod
    def _timestamp_gap_ms(cls, previous, current) -> float | None:
        if len(previous) < 3 or len(current) < 3:
            return None
        previous_at = cls._finite_float(previous[2])
        current_at = cls._finite_float(current[2])
        if previous_at is None or current_at is None:
            return None
        return current_at - previous_at

    def _compute_features(
        self,
        gazes: list[tuple[float, float]],
        durations: list[float],
    ) -> dict[str, Any] | None:
        if not gazes:
            return None

        amplitudes = [
            self._distance(gazes[index], gazes[index - 1])
            for index in range(1, len(gazes))
        ]
        if not amplitudes:
            amplitudes = [0.0]

        centre_x = self.TARGET_WIDTH / 2.0
        centre_y = self.TARGET_HEIGHT / 2.0
        distance_to_centre = [
            math.hypot(x - centre_x, y - centre_y) for x, y in gazes
        ]

        mean_x = self._mean([point[0] for point in gazes])
        mean_y = self._mean([point[1] for point in gazes])
        distance_to_gaze_mean = [
            math.hypot(x - mean_x, y - mean_y) for x, y in gazes
        ]

        features = {
            "sp_fix_count": int(len(gazes)),
            "sp_fix_duration_ms_total": float(sum(durations)),
            "sp_fix_duration_ms_mean": self._mean(durations),
            "sp_fix_duration_ms_var": self._variance(durations),
            "sp_len_px_total": float(sum(amplitudes)),
            "sp_saccade_amplitude_px_mean": self._mean(amplitudes),
            "sp_saccade_amplitude_px_var": self._variance(amplitudes),
            "sp_distance_to_centre_px_mean": self._mean(
                distance_to_centre
            ),
            "sp_distance_to_centre_px_var": self._variance(
                distance_to_centre
            ),
            "sp_distance_to_sp_mean_px_mean": self._mean(
                distance_to_gaze_mean
            ),
            "sp_distance_to_sp_mean_px_var": self._variance(
                distance_to_gaze_mean
            ),
        }
        if any(
            self._finite_float(value) is None for value in features.values()
        ):
            raise ValueError("computed gaze features contain a non-finite value")
        return features

    def _classify(
        self,
        features: dict[str, Any],
        fixation_centres: list[tuple[float, float]],
    ) -> dict[str, Any]:
        self._ensure_severity_loaded()

        missing = [
            column for column in self._feature_cols if column not in features
        ]
        if missing:
            raise ValueError(
                "severity model requires unsupported feature(s): "
                + ", ".join(missing)
            )
        sample = [[float(features[column]) for column in self._feature_cols]]

        with self._model_lock:
            prediction = self._model.predict(sample)[0]
            probabilities = self._model.predict_proba(sample)[0]
            model_classes = list(self._model.classes_)

        if len(model_classes) != len(probabilities):
            raise ValueError("severity model returned mismatched probabilities")

        probability_result = {"td": 0.0, "mild": 0.0, "severe": 0.0}
        for class_id, probability in zip(model_classes, probabilities):
            normalized_id = int(class_id)
            probability_key = self.PROBABILITY_KEYS.get(normalized_id)
            if probability_key is None:
                raise ValueError(
                    f"severity model returned unknown class {normalized_id}"
                )
            value = self._finite_float(probability)
            if value is None:
                raise ValueError("severity model returned a non-finite probability")
            probability_result[probability_key] = value

        prediction_id = int(prediction)
        if prediction_id not in self._class_labels:
            raise ValueError(
                f"severity model returned unknown prediction {prediction_id}"
            )

        dispersion = self._gaze_dispersion(fixation_centres)
        eye_state = self._eye_state(prediction_id, dispersion)
        return {
            "severity": self._class_labels[prediction_id],
            "probabilities": probability_result,
            "eye_state": eye_state,
            "eye_state_zh": self.EYE_STATE_ZH[eye_state],
        }

    def _ensure_severity_loaded(self) -> None:
        if self._model is not None:
            return
        if self._severity_load_error:
            raise RuntimeError(self._severity_load_error)

        with self._model_lock:
            if self._model is not None:
                return
            if not self.model_path or not self.model_path.is_file():
                self._severity_load_error = "severity model file is missing"
                raise FileNotFoundError(self._severity_load_error)

            try:
                # Pickle is intentionally restricted to the explicitly supplied
                # local artifact path.  Callers must not pass an uploaded file.
                with self.model_path.open("rb") as model_file:
                    artifact = pickle.load(model_file)
                if not isinstance(artifact, dict):
                    raise TypeError("severity artifact must be a dictionary")

                model = artifact.get("model")
                feature_cols = tuple(artifact.get("feature_cols") or ())
                if model is None or not callable(getattr(model, "predict", None)):
                    raise TypeError("severity artifact has no usable model")
                if not callable(getattr(model, "predict_proba", None)):
                    raise TypeError("severity model has no predict_proba")
                if not hasattr(model, "classes_"):
                    raise TypeError("severity model has no classes_")
                if feature_cols != self.FEATURE_COLUMNS:
                    raise ValueError(
                        "severity artifact feature schema does not match ASD v4"
                    )

                labels = self._normalize_class_labels(
                    artifact.get("classes") or self.DEFAULT_CLASS_LABELS
                )
                model_class_ids = {int(value) for value in model.classes_}
                if model_class_ids != set(self.PROBABILITY_KEYS):
                    raise ValueError(
                        "severity model classes must be exactly 0, 1, and 2"
                    )
                if model_class_ids != set(labels):
                    raise ValueError(
                        "severity label mapping does not match model.classes_"
                    )

                self._model = model
                self._feature_cols = feature_cols
                self._class_labels = labels
            except Exception as exc:
                self._severity_load_error = self._error_text(
                    "severity model load",
                    exc,
                )
                raise RuntimeError(self._severity_load_error) from exc

    def _analyze_emotion(
        self,
        session_key: str,
        frame_bytes,
    ) -> dict[str, str]:
        if not isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("frame_bytes must contain an encoded image")
        if not frame_bytes:
            raise ValueError("frame_bytes is empty")

        # DeepFace/TensorFlow and the dynamically loaded analyzer are shared
        # process resources, so inference is serialized.  Analyzer history is
        # still kept separately for every coaching session.
        with self._emotion_lock:
            analyzer = self._get_emotion_analyzer(session_key)
            try:
                import cv2
                import numpy as np
                from PIL import Image
            except Exception as exc:
                raise RuntimeError("image decoder dependencies are unavailable") from exc

            try:
                with Image.open(io.BytesIO(frame_bytes)) as image:
                    width, height = image.size
            except Exception as exc:
                raise ValueError("frame_bytes is not a valid image") from exc
            if width <= 0 or height <= 0:
                raise ValueError("frame dimensions must be positive")
            if width * height > self.MAX_EMOTION_PIXELS:
                raise ValueError("encoded frame dimensions are too large")

            encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("frame_bytes is not a decodable image")
            if frame.shape[0] * frame.shape[1] > self.MAX_EMOTION_PIXELS:
                raise ValueError("decoded frame dimensions are too large")

            raw_result = analyzer.analyze_frame(frame)
            if not isinstance(raw_result, dict):
                raise TypeError("emotion analyzer returned an invalid result")
            if not raw_result.get("available", True):
                detail = str(raw_result.get("error") or "no face detected")
                raise RuntimeError(detail)
            emotion = str(raw_result.get("dominant_emotion") or "unknown")
            emotion_zh = str(
                raw_result.get("dominant_emotion_zh") or "分析中"
            )
            return {"emotion": emotion, "emotion_zh": emotion_zh}

    def _get_emotion_analyzer(self, session_key: str):
        with self._state_lock:
            existing = self._emotion_analyzers.get(session_key)
        if existing is not None:
            return existing

        module = self._ensure_emotion_module_loaded()
        analyzer_class = getattr(module, "EmotionAnalyzer", None)
        if analyzer_class is None:
            raise TypeError("emotion module has no EmotionAnalyzer")

        # Frames arrive at a deliberately low rate, therefore every submitted
        # frame is analyzed.  fps=1 keeps the five-second history bounded to
        # roughly five low-frequency samples instead of 150 stale samples.
        analyzer = analyzer_class(
            detector_backend="yunet",
            window_sec=5.0,
            fps=1,
            analyze_every_n_frames=1,
        )
        with self._state_lock:
            return self._emotion_analyzers.setdefault(session_key, analyzer)

    def _ensure_emotion_module_loaded(self) -> ModuleType:
        if self._emotion_module is not None:
            return self._emotion_module
        if self._emotion_module_error:
            raise RuntimeError(self._emotion_module_error)

        if (
            not self.emotion_module_path
            or not self.emotion_module_path.is_file()
        ):
            self._emotion_module_error = "emotion module file is missing"
            raise FileNotFoundError(self._emotion_module_error)

        module_name = "_homecoach_asd_v4_emotion_analyzer"
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                self.emotion_module_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError("could not create an emotion module spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._emotion_module = module
            return module
        except Exception as exc:
            sys.modules.pop(module_name, None)
            self._emotion_module_error = self._error_text(
                "emotion module load",
                exc,
            )
            raise RuntimeError(self._emotion_module_error) from exc

    def _normalize_class_labels(self, raw_labels) -> dict[int, str]:
        if not isinstance(raw_labels, dict):
            raise TypeError("severity class labels must be a dictionary")
        labels: dict[int, str] = {}
        for class_id, label in raw_labels.items():
            labels[int(class_id)] = str(label)
        return labels

    def _gaze_dispersion(
        self,
        gazes: list[tuple[float, float]],
    ) -> float:
        if len(gazes) <= 1:
            return 0.0
        x_values = [point[0] for point in gazes]
        y_values = [point[1] for point in gazes]
        return (
            math.sqrt(self._variance(x_values))
            + math.sqrt(self._variance(y_values))
        ) / 2.0

    def _eye_state(self, prediction_id: int, dispersion: float) -> str:
        if prediction_id == 2:
            return "hyperscanning" if dispersion > 150.0 else "avoidant"
        if prediction_id == 1:
            return "transitional" if dispersion > 100.0 else "avoidant_mild"
        return "focused"

    def _get_session_lock(self, session_key: str) -> threading.RLock:
        with self._state_lock:
            return self._session_locks.setdefault(
                session_key,
                threading.RLock(),
            )

    def _store_latest(self, session_key: str, result: dict[str, Any]) -> None:
        with self._state_lock:
            self._latest[session_key] = copy.deepcopy(result)

    @staticmethod
    def _session_key(session_id) -> str:
        if session_id is None:
            raise ValueError("session_id is required")
        return str(session_id)

    @staticmethod
    def _distance(
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return math.hypot(left[0] - right[0], left[1] - right[1])

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @classmethod
    def _variance(cls, values: list[float]) -> float:
        if not values:
            return 0.0
        mean_value = cls._mean(values)
        return float(
            sum((float(value) - mean_value) ** 2 for value in values)
            / len(values)
        )

    @staticmethod
    def _finite_float(value) -> float | None:
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return normalized if math.isfinite(normalized) else None

    @staticmethod
    def _path_text(path: Path | None) -> str | None:
        return str(path) if path is not None else None

    @staticmethod
    def _error_text(prefix: str, error: Exception) -> str:
        detail = str(error).strip() or error.__class__.__name__
        return f"{prefix}: {detail}"

    @staticmethod
    def _missing_dependencies(module_names) -> list[str]:
        missing = []
        for module_name in module_names:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            if not available:
                missing.append(module_name)
        return missing
