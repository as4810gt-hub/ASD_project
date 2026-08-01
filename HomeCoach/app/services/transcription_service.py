"""Thread-safe, lazy integration with a local faster-whisper checkout.

The module deliberately imports only Python's standard library at import time.
This keeps the Flask application usable when the optional speech-recognition
dependencies are not installed yet.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    """A predictable error raised by :class:`TranscriptionService`."""

    def __init__(self, message: str, *, code: str = "transcription_error"):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": str(self),
        }


class TranscriptionService:
    """Transcribe audio with a lazily loaded, shared faster-whisper model.

    Models are shared by all service instances that use the same source and
    model configuration. Loading and inference are serialized per model. This
    avoids duplicate model allocations and protects CTranslate2 inference when
    Flask handles multiple requests concurrently.
    """

    REQUIRED_DEPENDENCIES = (
        "av",
        "ctranslate2",
        "huggingface_hub",
        "numpy",
        "onnxruntime",
        "tokenizers",
        "tqdm",
    )

    # Mirrors the aliases in faster_whisper.utils. Keeping this small mapping
    # locally lets health() inspect the Hugging Face cache without importing
    # faster-whisper (which would load native dependencies).
    MODEL_REPOSITORIES = {
        "tiny.en": "Systran/faster-whisper-tiny.en",
        "tiny": "Systran/faster-whisper-tiny",
        "base.en": "Systran/faster-whisper-base.en",
        "base": "Systran/faster-whisper-base",
        "small.en": "Systran/faster-whisper-small.en",
        "small": "Systran/faster-whisper-small",
        "medium.en": "Systran/faster-whisper-medium.en",
        "medium": "Systran/faster-whisper-medium",
        "large-v1": "Systran/faster-whisper-large-v1",
        "large-v2": "Systran/faster-whisper-large-v2",
        "large-v3": "Systran/faster-whisper-large-v3",
        "large": "Systran/faster-whisper-large-v3",
        "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
        "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
        "distil-small.en": "Systran/faster-distil-whisper-small.en",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
        "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }

    _registry_guard = threading.RLock()
    _import_guard = threading.RLock()
    _models: dict[tuple[str, ...], Any] = {}
    _load_locks: dict[tuple[str, ...], threading.Lock] = {}
    _inference_locks: dict[tuple[str, ...], threading.Lock] = {}
    _prepare_threads: dict[tuple[str, ...], threading.Thread] = {}
    _prepare_errors: dict[tuple[str, ...], str] = {}

    def __init__(
        self,
        source_dir: str | os.PathLike[str] | None = None,
        model: str | os.PathLike[str] = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "zh",
        download_root: str | os.PathLike[str] | None = None,
        enabled: bool = True,
    ):
        default_source = Path(__file__).resolve().parents[3] / "faster-whisper-master"
        source_path = Path(source_dir or default_source).expanduser()
        source_path = source_path.resolve(strict=False)

        # Also accept a path that points directly at the Python package.
        if (
            source_path.name == "faster_whisper"
            and (source_path / "__init__.py").is_file()
        ):
            source_path = source_path.parent

        model_value = str(model).strip()
        device_value = str(device).strip()
        compute_value = str(compute_type).strip()
        if not model_value:
            raise ValueError("Whisper model must not be empty")
        if not device_value:
            raise ValueError("Whisper device must not be empty")
        if not compute_value:
            raise ValueError("Whisper compute_type must not be empty")

        self.source_dir = source_path
        self.model = self._normalize_model_identifier(model_value)
        self.device = device_value
        self.compute_type = compute_value
        self.language = self._normalize_language(language)
        self.download_root = (
            Path(download_root).expanduser().resolve(strict=False)
            if download_root
            else None
        )
        self.enabled = self._coerce_enabled(enabled)

        self._key = (
            str(self.source_dir),
            self.model,
            self.device,
            self.compute_type,
            str(self.download_root or ""),
        )

    def transcribe(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Transcribe one audio file and return a JSON-serializable result."""

        if not self.enabled:
            raise TranscriptionError(
                "Whisper transcription is disabled",
                code="transcription_disabled",
            )

        try:
            audio_path = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TranscriptionError(
                f"Audio file was not found: {path}",
                code="audio_not_found",
            ) from exc

        if not audio_path.is_file():
            raise TranscriptionError(
                f"Audio path is not a file: {audio_path}",
                code="invalid_audio_path",
            )

        total_started_at = time.perf_counter()
        model_load_started_at = time.perf_counter()
        model = self._get_or_load_model()
        model_load_seconds = time.perf_counter() - model_load_started_at
        inference_lock = self._get_inference_lock()

        try:
            # faster-whisper performs most work while its segment generator is
            # consumed, so materialization must remain inside the lock.
            with inference_lock:
                inference_started_at = time.perf_counter()
                segment_stream, info = model.transcribe(
                    str(audio_path),
                    language=self.language,
                    vad_filter=True,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                raw_segments = list(segment_stream)
                inference_seconds = time.perf_counter() - inference_started_at
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper could not transcribe the audio: {exc}",
                code="transcription_failed",
            ) from exc

        result = self._build_result(raw_segments, info)
        result.update(
            {
                "processing_seconds": round(
                    time.perf_counter() - total_started_at,
                    3,
                ),
                "model_load_seconds": round(model_load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "model": self.model,
                "device": self.device,
                "compute_type": self.compute_type,
            }
        )
        return result

    def prepare_async(self) -> bool:
        """Warm a cached model in the background without delaying a request."""

        if not self.enabled:
            return False
        with self._registry_guard:
            if self._key in self._models:
                return False
            existing = self._prepare_threads.get(self._key)
            if existing is not None and existing.is_alive():
                return False
            model_state, _ = self._inspect_model_state(loaded=False)
            if model_state not in {"cached", "local", "loaded"}:
                return False
            thread = threading.Thread(
                target=self._prepare_safely,
                name="homecoach-whisper-warmup",
                daemon=True,
            )
            self._prepare_threads[self._key] = thread
            self._prepare_errors.pop(self._key, None)
            thread.start()
            return True

    def _prepare_safely(self) -> None:
        try:
            self._get_or_load_model()
        except Exception as exc:  # warm-up failure remains visible in health
            detail = str(exc).strip() or exc.__class__.__name__
            with self._registry_guard:
                self._prepare_errors[self._key] = detail

    def health(self) -> dict[str, Any]:
        """Inspect configuration without importing or loading faster-whisper.

        This method only examines package metadata and filesystem paths. It
        never calls ``WhisperModel`` and therefore cannot download a model.
        """

        source_init = self.source_dir / "faster_whisper" / "__init__.py"
        source_state = "ready" if source_init.is_file() else "missing"

        dependencies = {
            name: self._dependency_is_available(name)
            for name in self.REQUIRED_DEPENDENCIES
        }
        missing_dependencies = [
            name for name, available in dependencies.items() if not available
        ]
        dependency_state = "ready" if not missing_dependencies else "missing"

        with self._registry_guard:
            loaded = self._key in self._models
            prepare_thread = self._prepare_threads.get(self._key)
            warming = bool(prepare_thread and prepare_thread.is_alive())
            warmup_error = self._prepare_errors.get(self._key)

        model_state, model_path = self._inspect_model_state(loaded=loaded)

        if not self.enabled:
            status = "disabled"
        elif source_state != "ready":
            status = "source_missing"
        elif dependency_state != "ready":
            status = "dependency_missing"
        elif model_state in {"missing", "incomplete"}:
            status = "model_missing"
        elif model_state == "download_required":
            status = "model_not_cached"
        else:
            status = "ready"

        return {
            "status": status,
            "enabled": self.enabled,
            "source_dir": str(self.source_dir),
            "source_state": source_state,
            "dependency_state": dependency_state,
            "dependencies": dependencies,
            "missing_dependencies": missing_dependencies,
            "model": self.model,
            "model_state": model_state,
            "model_path": str(model_path) if model_path else None,
            "loaded": loaded,
            "warming": warming,
            "warmup_error": warmup_error,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "download_root": (
                str(self.download_root) if self.download_root else None
            ),
        }

    def _get_or_load_model(self):
        with self._registry_guard:
            cached_model = self._models.get(self._key)
            if cached_model is not None:
                return cached_model
            load_lock = self._load_locks.setdefault(self._key, threading.Lock())
            self._inference_locks.setdefault(self._key, threading.Lock())

        with load_lock:
            # Another request may have completed the load while this request
            # waited for the per-model lock.
            with self._registry_guard:
                cached_model = self._models.get(self._key)
                if cached_model is not None:
                    return cached_model

            whisper_model_class = self._import_whisper_model()
            kwargs = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.download_root is not None:
                kwargs["download_root"] = str(self.download_root)

            try:
                loaded_model = whisper_model_class(self.model, **kwargs)
            except Exception as exc:
                raise TranscriptionError(
                    f"Whisper model '{self.model}' could not be loaded: {exc}",
                    code="model_load_failed",
                ) from exc

            with self._registry_guard:
                # The load lock guarantees this assignment happens once for
                # the key, but setdefault also preserves the invariant if the
                # registry is manipulated by an embedding application.
                return self._models.setdefault(self._key, loaded_model)

    def _get_inference_lock(self) -> threading.Lock:
        with self._registry_guard:
            return self._inference_locks.setdefault(
                self._key,
                threading.Lock(),
            )

    def _import_whisper_model(self):
        package_init = self.source_dir / "faster_whisper" / "__init__.py"
        if not package_init.is_file():
            raise TranscriptionError(
                f"Local faster-whisper source was not found at {self.source_dir}",
                code="whisper_source_missing",
            )

        with self._import_guard:
            existing_module = sys.modules.get("faster_whisper")
            if existing_module is not None:
                self._validate_import_source(existing_module, package_init)
                try:
                    return existing_module.WhisperModel
                except AttributeError as exc:
                    raise TranscriptionError(
                        "The loaded faster_whisper package has no WhisperModel",
                        code="whisper_dependency_invalid",
                    ) from exc

            source_value = str(self.source_dir)
            inserted = source_value not in sys.path
            if inserted:
                sys.path.insert(0, source_value)

            try:
                module = importlib.import_module("faster_whisper")
                self._validate_import_source(module, package_init)
                return module.WhisperModel
            except TranscriptionError:
                raise
            except (ImportError, OSError) as exc:
                missing_name = getattr(exc, "name", None)
                detail = f" Missing dependency: {missing_name}." if missing_name else ""
                raise TranscriptionError(
                    "Local faster-whisper could not be imported."
                    f"{detail} Install faster-whisper requirements first.",
                    code="whisper_dependency_missing",
                ) from exc
            except Exception as exc:
                raise TranscriptionError(
                    f"Local faster-whisper could not be imported: {exc}",
                    code="whisper_import_failed",
                ) from exc
            finally:
                # A successfully imported package keeps its own __path__. The
                # temporary entry is not needed afterward and should not alter
                # unrelated application imports.
                if inserted:
                    try:
                        sys.path.remove(source_value)
                    except ValueError:
                        pass

    def _validate_import_source(self, module, expected_init: Path) -> None:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise TranscriptionError(
                "The loaded faster_whisper package has no source location",
                code="whisper_dependency_invalid",
            )

        try:
            actual_init = Path(module_file).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise TranscriptionError(
                "The loaded faster_whisper source location is invalid",
                code="whisper_dependency_invalid",
            ) from exc

        if actual_init != expected_init.resolve(strict=False):
            raise TranscriptionError(
                "A different faster_whisper package is already loaded "
                f"from {actual_init}; expected {expected_init}",
                code="whisper_source_conflict",
            )

    def _build_result(self, raw_segments, info) -> dict[str, Any]:
        segments = []
        for index, segment in enumerate(raw_segments):
            start = self._safe_float(getattr(segment, "start", 0.0))
            end = max(start, self._safe_float(getattr(segment, "end", start)))
            segment_id = getattr(segment, "id", index)
            try:
                segment_id = int(segment_id)
            except (TypeError, ValueError, OverflowError):
                segment_id = index

            segments.append(
                {
                    "id": segment_id,
                    "start": start,
                    "end": end,
                    "text": str(getattr(segment, "text", "") or "").strip(),
                }
            )

        start = min((segment["start"] for segment in segments), default=0.0)
        end = max((segment["end"] for segment in segments), default=0.0)
        info_duration = self._safe_float(getattr(info, "duration", 0.0))
        duration = max(info_duration, end, 0.0)

        text = "".join(
            str(getattr(segment, "text", "") or "") for segment in raw_segments
        ).strip()
        detected_language = (
            str(getattr(info, "language", "") or self.language or "").strip()
            or None
        )
        language_probability = self._safe_float(
            getattr(info, "language_probability", 0.0)
        )

        return {
            "text": text,
            "language": detected_language,
            "language_probability": language_probability,
            "start": start,
            "end": end,
            "duration": duration,
            "segments": segments,
        }

    def _inspect_model_state(
        self,
        *,
        loaded: bool,
    ) -> tuple[str, Path | None]:
        if loaded:
            return "loaded", self._local_model_path()

        local_path = self._local_model_path()
        if local_path is not None:
            if not local_path.exists():
                return "missing", local_path
            if not local_path.is_dir():
                return "incomplete", local_path
            if self._looks_like_model_directory(local_path):
                return "local", local_path
            return "incomplete", local_path

        cached_path = self._find_cached_model()
        if cached_path is not None:
            return "cached", cached_path
        return "download_required", None

    def _find_cached_model(self) -> Path | None:
        repository = self.MODEL_REPOSITORIES.get(self.model)
        if repository is None and "/" in self.model:
            repository = self.model
        if repository is None:
            return None

        cache_root = self.download_root or self._default_huggingface_cache()
        repository_dir = cache_root / (
            "models--" + repository.replace("/", "--")
        )
        snapshots_dir = repository_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None

        try:
            snapshots = sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None

        return next(
            (
                snapshot
                for snapshot in snapshots
                if self._looks_like_model_directory(snapshot)
            ),
            None,
        )

    def _local_model_path(self) -> Path | None:
        value = Path(self.model).expanduser()
        is_explicit_path = (
            value.is_absolute()
            or self.model.startswith(".")
            or self.model.startswith("~")
            or value.exists()
        )
        return value.resolve(strict=False) if is_explicit_path else None

    @staticmethod
    def _looks_like_model_directory(path: Path) -> bool:
        return (path / "config.json").is_file() and (path / "model.bin").is_file()

    @staticmethod
    def _default_huggingface_cache() -> Path:
        explicit_cache = os.getenv("HF_HUB_CACHE") or os.getenv(
            "HUGGINGFACE_HUB_CACHE"
        )
        if explicit_cache:
            return Path(explicit_cache).expanduser()

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            return Path(hf_home).expanduser() / "hub"

        xdg_cache = os.getenv("XDG_CACHE_HOME")
        base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        return base / "huggingface" / "hub"

    @staticmethod
    def _dependency_is_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        if language is None:
            return None
        normalized = str(language).strip()
        return normalized or None

    @staticmethod
    def _coerce_enabled(value: bool) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off"}
        return bool(value)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        # NaN is the only float that is not equal to itself.
        if result != result or result in {float("inf"), float("-inf")}:
            return 0.0
        return result

    @staticmethod
    def _normalize_model_identifier(value: str) -> str:
        candidate = Path(value).expanduser()
        if (
            candidate.is_absolute()
            or value.startswith(".")
            or value.startswith("~")
            or candidate.exists()
        ):
            return str(candidate.resolve(strict=False))
        return value
