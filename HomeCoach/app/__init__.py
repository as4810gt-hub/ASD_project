import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify

from .models.repository import CoachingRepository
from .services.coaching_service import CoachingService
from .services.asd_analysis_service import ASDAnalysisService
from .services.context_builder import ContextBuilder
from .services.emt_rule_engine import EMTRuleEngine
from .services.gemini_coach_provider import GeminiCoachProvider
from .services.ollama_coach_provider import OllamaCoachProvider
from .services.transcription_service import TranscriptionService


def _env_enabled(name, default=True):
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    project_root = Path(__file__).resolve().parents[2]
    whisper_max_audio_bytes = int(
        os.getenv("WHISPER_MAX_AUDIO_BYTES", str(8 * 1024 * 1024))
    )

    app.config.from_mapping(
        SECRET_KEY="homecoach-local-development",
        DATABASE=str(Path(app.instance_path) / "homecoach.sqlite3"),
        STIMULI_DIR=str(
            project_root / "EyeTism" / "Dashboard" / "content" / "images"
        ),
        WEBGAZER_DIR=os.getenv(
            "WEBGAZER_DIR",
            str(project_root / "WebGazer-master" / "www"),
        ),
        OLLAMA_BASE_URL=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434/api",
        ),
        OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        OLLAMA_TIMEOUT_SECONDS=float(
            os.getenv("OLLAMA_TIMEOUT_SECONDS", "30")
        ),
        OLLAMA_ENABLED=_env_enabled("OLLAMA_ENABLED", default=True),
        OLLAMA_VISION_ENABLED=_env_enabled(
            "OLLAMA_VISION_ENABLED",
            default=True,
        ),
        OLLAMA_MAX_IMAGE_BYTES=int(
            os.getenv("OLLAMA_MAX_IMAGE_BYTES", str(3 * 1024 * 1024))
        ),
        COACH_PROVIDER=os.getenv("COACH_PROVIDER", "gemini").strip().lower(),
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
        GEMINI_BASE_URL=os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        GEMINI_MODEL=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        GEMINI_TIMEOUT_SECONDS=float(
            os.getenv("GEMINI_TIMEOUT_SECONDS", "15")
        ),
        GEMINI_THINKING_LEVEL=os.getenv(
            "GEMINI_THINKING_LEVEL",
            "low",
        ),
        GEMINI_ENABLED=_env_enabled("GEMINI_ENABLED", default=True),
        GEMINI_VISION_ENABLED=_env_enabled(
            "GEMINI_VISION_ENABLED",
            default=True,
        ),
        GEMINI_MAX_IMAGE_BYTES=int(
            os.getenv("GEMINI_MAX_IMAGE_BYTES", str(3 * 1024 * 1024))
        ),
        ASD_V4_DIR=os.getenv(
            "ASD_V4_DIR",
            str(project_root / "asd_v4"),
        ),
        ASD_ANALYSIS_ENABLED=_env_enabled(
            "ASD_ANALYSIS_ENABLED",
            default=True,
        ),
        ASD_EMOTION_ENABLED=_env_enabled(
            "ASD_EMOTION_ENABLED",
            default=True,
        ),
        ASD_FRAME_MAX_BYTES=int(
            os.getenv("ASD_FRAME_MAX_BYTES", str(1024 * 1024))
        ),
        WHISPER_SOURCE_DIR=os.getenv(
            "WHISPER_SOURCE_DIR",
            str(project_root / "faster-whisper-master"),
        ),
        WHISPER_MODEL=os.getenv("WHISPER_MODEL", "base"),
        WHISPER_DEVICE=os.getenv("WHISPER_DEVICE", "cpu"),
        WHISPER_COMPUTE_TYPE=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        WHISPER_LANGUAGE=os.getenv("WHISPER_LANGUAGE", "zh"),
        WHISPER_DOWNLOAD_ROOT=os.getenv(
            "WHISPER_DOWNLOAD_ROOT",
            str(Path(app.instance_path) / "whisper-models"),
        ),
        WHISPER_ENABLED=_env_enabled("WHISPER_ENABLED", default=True),
        WHISPER_MAX_AUDIO_BYTES=whisper_max_audio_bytes,
        MAX_CONTENT_LENGTH=whisper_max_audio_bytes + 256 * 1024,
    )

    if test_config:
        app.config.update(test_config)
        if (
            "WHISPER_MAX_AUDIO_BYTES" in test_config
            and "MAX_CONTENT_LENGTH" not in test_config
        ):
            app.config["MAX_CONTENT_LENGTH"] = (
                int(app.config["WHISPER_MAX_AUDIO_BYTES"])
                + 256 * 1024
            )

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    repository = CoachingRepository(app.config["DATABASE"])
    repository.init_schema()
    repository.seed_demo_data()

    coach_provider = OllamaCoachProvider(
        base_url=app.config["OLLAMA_BASE_URL"],
        model=app.config["OLLAMA_MODEL"],
        timeout_seconds=app.config["OLLAMA_TIMEOUT_SECONDS"],
        enabled=app.config["OLLAMA_ENABLED"] and not app.config.get("TESTING"),
        stimuli_dir=(
            app.config["STIMULI_DIR"]
            if app.config["OLLAMA_VISION_ENABLED"]
            else None
        ),
        max_image_bytes=app.config["OLLAMA_MAX_IMAGE_BYTES"],
    )
    gemini_provider = GeminiCoachProvider(
        api_key=app.config["GEMINI_API_KEY"],
        model=app.config["GEMINI_MODEL"],
        base_url=app.config["GEMINI_BASE_URL"],
        stimuli_dir=(
            app.config["STIMULI_DIR"]
            if app.config["GEMINI_VISION_ENABLED"]
            else None
        ),
        timeout_seconds=app.config["GEMINI_TIMEOUT_SECONDS"],
        enabled=(
            app.config["GEMINI_ENABLED"]
            and not app.config.get("TESTING")
        ),
        thinking_level=app.config["GEMINI_THINKING_LEVEL"],
        max_image_bytes=app.config["GEMINI_MAX_IMAGE_BYTES"],
        fallback_provider=coach_provider,
    )
    selected_coach_provider = (
        gemini_provider
        if app.config["COACH_PROVIDER"] == "gemini"
        else coach_provider
    )
    transcription_service = TranscriptionService(
        source_dir=app.config["WHISPER_SOURCE_DIR"],
        model=app.config["WHISPER_MODEL"],
        device=app.config["WHISPER_DEVICE"],
        compute_type=app.config["WHISPER_COMPUTE_TYPE"],
        language=app.config["WHISPER_LANGUAGE"],
        download_root=app.config["WHISPER_DOWNLOAD_ROOT"],
        enabled=app.config["WHISPER_ENABLED"],
    )
    if not app.config.get("TESTING"):
        transcription_service.prepare_async()
    asd_v4_dir = Path(app.config["ASD_V4_DIR"])
    asd_analysis_service = ASDAnalysisService(
        model_path=asd_v4_dir / "models" / "eye" / "severity_classifier.pkl",
        emotion_module_path=asd_v4_dir / "modules" / "emotion_analyzer.py",
        enabled=(
            app.config["ASD_ANALYSIS_ENABLED"]
            and not app.config.get("TESTING")
        ),
        emotion_enabled=app.config["ASD_EMOTION_ENABLED"],
    )

    app.extensions["coaching_repository"] = repository
    app.extensions["ollama_coach_provider"] = coach_provider
    app.extensions["gemini_coach_provider"] = gemini_provider
    app.extensions["coach_provider"] = selected_coach_provider
    app.extensions["transcription_service"] = transcription_service
    app.extensions["asd_analysis_service"] = asd_analysis_service
    app.extensions["coaching_service"] = CoachingService(
        repository=repository,
        rule_engine=EMTRuleEngine(),
        context_builder=ContextBuilder(history_limit=5),
        coach_provider=selected_coach_provider,
        asd_analysis_service=asd_analysis_service,
    )

    from .controllers.api import api_bp
    from .controllers.pages import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    @app.errorhandler(413)
    def request_too_large(_error):
        max_megabytes = max(
            1,
            app.config["WHISPER_MAX_AUDIO_BYTES"] // (1024 * 1024),
        )
        return jsonify(
            {"error": f"上傳內容過大，audio 不可超過 {max_megabytes} MB"}
        ), 413

    @app.template_filter("datetime_zh")
    def datetime_zh(value):
        if not value:
            return "—"
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.strftime("%m/%d %H:%M")
        except (TypeError, ValueError):
            return value

    @app.template_filter("duration_zh")
    def duration_zh(value):
        try:
            seconds = max(0, int(value or 0))
        except (TypeError, ValueError):
            return "—"
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes} 分 {seconds:02d} 秒"

    return app
