import json
import math
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from ..services.transcription_service import TranscriptionError
from ..materials import DEFAULT_MATERIAL_ID, get_material, get_material_by_title


api_bp = Blueprint("api", __name__, url_prefix="/api")

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
AUDIO_SUFFIXES = {".m4a", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}
MIMETYPE_SUFFIXES = {
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def _service():
    return current_app.extensions["coaching_service"]


def _parse_boolean(value, field_name, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
    raise ValueError(f"{field_name} 必須是 true 或 false")


def _parse_pause(value):
    try:
        return max(0.0, min(float(value or 0), 30.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("pause_before 格式錯誤") from exc


def _audio_suffix(upload):
    filename_suffix = Path(upload.filename or "").suffix.lower()
    if filename_suffix in AUDIO_SUFFIXES:
        return filename_suffix

    mimetype = (upload.mimetype or "").split(";", 1)[0].strip().lower()
    suffix = MIMETYPE_SUFFIXES.get(mimetype)
    if suffix:
        return suffix
    raise ValueError("audio 必須是 WebM、MP4、M4A、MP3、OGG 或 WAV 音訊")


def _save_audio_upload(upload, max_bytes):
    suffix = _audio_suffix(upload)
    upload_dir = Path(current_app.instance_path) / "transcription-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix="utterance-",
            suffix=suffix,
            dir=upload_dir,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while True:
                chunk = upload.stream.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(
                        f"audio 不可超過 {max_bytes // (1024 * 1024)} MB"
                    )
                temporary_file.write(chunk)

        if total_bytes == 0:
            raise ValueError("audio 內容不可為空")
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _transcription_error_status(error):
    service_unavailable_codes = {
        "model_load_failed",
        "transcription_disabled",
        "whisper_dependency_invalid",
        "whisper_dependency_missing",
        "whisper_import_failed",
        "whisper_source_conflict",
        "whisper_source_missing",
    }
    return 503 if error.code in service_unavailable_codes else 422


@api_bp.get("/health")
def health():
    gemini = current_app.extensions["gemini_coach_provider"].health()
    ollama = gemini.get("fallback") or current_app.extensions[
        "ollama_coach_provider"
    ].health()
    selected_provider = current_app.extensions["coach_provider"]
    coach_provider = (
        gemini
        if selected_provider
        is current_app.extensions["gemini_coach_provider"]
        else ollama
    )
    whisper = current_app.extensions["transcription_service"].health()
    asd_analysis = current_app.extensions["asd_analysis_service"].health()
    return jsonify(
        {
            "status": "ok",
            "modules": {
                "speech": "browser-ready",
                "gaze": "camera-ready",
                "rule_engine": "ready",
                "coach": "ready",
                "coach_provider": coach_provider,
                "gemini": gemini,
                "ollama": ollama,
                "whisper": whisper,
                "asd_analysis": asd_analysis,
            },
        }
    )


@api_bp.post("/sessions")
def create_session():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON 內容必須是物件"}), 400
    material_id = str(payload.get("material_id") or "").strip()
    material = get_material(material_id) if material_id else None
    if material_id and material is None:
        return jsonify({"error": "找不到指定的教材圖片"}), 400
    if material is None:
        legacy_title = str(payload.get("material") or "").strip()
        material = get_material_by_title(legacy_title) if legacy_title else None
    if material is None:
        material = get_material(DEFAULT_MATERIAL_ID)

    session = _service().start_session(
        child_name=str(payload.get("child_name") or "小宇").strip()[:40],
        material=material["session_label"],
        material_id=material["id"],
    )
    return jsonify({"session": session}), 201


def _parse_asd_gaze_samples(raw_value):
    if not raw_value:
        return []
    try:
        payload = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except json.JSONDecodeError as exc:
        raise ValueError("gaze_samples 必須是有效 JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("gaze_samples 必須是陣列")

    samples = []
    for item in payload[-600:]:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
            timestamp = float(item.get("at"))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, timestamp)):
            continue
        samples.append({"x": x, "y": y, "at": timestamp})
    return samples


@api_bp.post("/sessions/<int:session_id>/asd-analysis")
def analyze_asd_signals(session_id):
    session = _service().get_session(session_id)
    if not session:
        return jsonify({"error": "找不到這次互動紀錄"}), 404
    if session["status"] != "active":
        return jsonify({"error": "這次互動已經結束"}), 409

    try:
        gaze_samples = _parse_asd_gaze_samples(
            request.form.get("gaze_samples", "[]")
        )
        viewport_width = max(
            1.0,
            min(float(request.form.get("viewport_width", 1280)), 10000.0),
        )
        viewport_height = max(
            1.0,
            min(float(request.form.get("viewport_height", 720)), 10000.0),
        )
        blink_rate = max(
            0.0,
            min(float(request.form.get("blink_rate_per_min", 0)), 180.0),
        )
        blink_available = _parse_boolean(
            request.form.get("blink_available"),
            "blink_available",
            False,
        )
        face_found = _parse_boolean(
            request.form.get("face_found"),
            "face_found",
            False,
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    frame_bytes = None
    frame = request.files.get("frame")
    if frame is not None:
        allowed_types = {"image/jpeg", "image/png", "image/webp"}
        mimetype = (frame.mimetype or "").split(";", 1)[0].lower()
        if mimetype not in allowed_types:
            return jsonify({"error": "frame 必須是 JPEG、PNG 或 WebP 圖片"}), 400
        max_frame_bytes = int(current_app.config["ASD_FRAME_MAX_BYTES"])
        frame_bytes = frame.stream.read(max_frame_bytes + 1)
        if len(frame_bytes) > max_frame_bytes:
            return jsonify({"error": "情緒分析影像過大"}), 413
        if not frame_bytes:
            frame_bytes = None

    try:
        analysis = _service().analyze_asd_signals(
            session_id=session_id,
            gaze_samples=gaze_samples,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            blink_rate_per_min=blink_rate,
            blink_available=blink_available,
            face_found=face_found,
            frame_bytes=frame_bytes,
        )
    except LookupError as exc:
        status = 409 if "已經結束" in str(exc) else 404
        return jsonify({"error": str(exc)}), status
    return jsonify({"analysis": analysis})


@api_bp.get("/sessions/<int:session_id>")
def get_session(session_id):
    session = _service().get_session(session_id)
    if not session:
        return jsonify({"error": "找不到這次互動紀錄"}), 404
    return jsonify(session)


@api_bp.post("/sessions/<int:session_id>/events")
def create_event(session_id):
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON 內容必須是物件"}), 400
    speaker = str(payload.get("speaker") or "").strip().lower()
    text = str(payload.get("text") or "").strip()

    if speaker not in {"parent", "child"}:
        return jsonify({"error": "speaker 必須是 parent 或 child"}), 400
    if not text:
        return jsonify({"error": "請提供對話內容"}), 400

    session = _service().get_session(session_id)
    if not session:
        return jsonify({"error": "找不到這次互動紀錄"}), 404
    if session["status"] != "active":
        return jsonify({"error": "這次互動已經結束"}), 409

    try:
        pause_before = _parse_pause(payload.get("pause_before", 0))
        gaze_available = _parse_boolean(
            payload.get("gaze_available"),
            "gaze_available",
            False,
        )
        gaze_on_target = _parse_boolean(
            payload.get("gaze_on_target"),
            "gaze_on_target",
            True,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        event = _service().record_event(
            session_id=session_id,
            speaker=speaker,
            text=text[:300],
            pause_before=pause_before,
            gaze_on_target=gaze_on_target,
            gaze_available=gaze_available,
            defer_coach=True,
        )
    except LookupError as exc:
        status = 409 if "已經結束" in str(exc) else 404
        return jsonify({"error": str(exc)}), status

    return jsonify(event), 201


@api_bp.post(
    "/sessions/<int:session_id>/events/<int:event_id>/coach-refinement"
)
def refine_event_coach(session_id, event_id):
    try:
        result = _service().refine_event_coach(session_id, event_id)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@api_bp.post("/sessions/<int:session_id>/transcriptions")
def create_transcription(session_id):
    session = _service().get_session(session_id)
    if not session:
        return jsonify({"error": "找不到這次互動紀錄"}), 404
    if session["status"] != "active":
        return jsonify({"error": "這次互動已經結束"}), 409

    max_bytes = max(1, int(current_app.config["WHISPER_MAX_AUDIO_BYTES"]))
    request_limit = max_bytes + 256 * 1024
    if request.content_length and request.content_length > request_limit:
        return (
            jsonify(
                {
                    "error": f"audio 不可超過 {max_bytes // (1024 * 1024)} MB"
                }
            ),
            413,
        )

    speaker = str(request.form.get("speaker") or "").strip().lower()
    if speaker not in {"parent", "child"}:
        return jsonify({"error": "speaker 必須是 parent 或 child"}), 400

    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"error": "請提供 audio 音訊檔"}), 400

    try:
        pause_before = _parse_pause(request.form.get("pause_before", 0))
        gaze_available = _parse_boolean(
            request.form.get("gaze_available"),
            "gaze_available",
            False,
        )
        gaze_on_target = _parse_boolean(
            request.form.get("gaze_on_target"),
            "gaze_on_target",
            True,
        )
        temporary_path = _save_audio_upload(upload, max_bytes)
    except ValueError as exc:
        status = 413 if "不可超過" in str(exc) else 400
        return jsonify({"error": str(exc)}), status

    try:
        transcription = current_app.extensions[
            "transcription_service"
        ].transcribe(temporary_path)
    except TranscriptionError as exc:
        return (
            jsonify(
                {
                    "error": str(exc),
                    "code": exc.code,
                }
            ),
            _transcription_error_status(exc),
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    if not isinstance(transcription, dict):
        return jsonify({"error": "Whisper 回傳格式錯誤"}), 502

    text = str(transcription.get("text") or "").strip()
    if not text:
        return jsonify(
            {
                "status": "no_speech",
                "transcription": transcription,
            }
        )

    try:
        result = _service().record_event(
            session_id=session_id,
            speaker=speaker,
            text=text[:300],
            pause_before=pause_before,
            gaze_on_target=gaze_on_target,
            gaze_available=gaze_available,
            metadata=transcription,
            defer_coach=True,
        )
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(
        {
            **result,
            "transcription": transcription,
        }
    ), 201


@api_bp.post("/sessions/<int:session_id>/finish")
def finish_session(session_id):
    try:
        session = _service().finish_session(session_id)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"session": session})
