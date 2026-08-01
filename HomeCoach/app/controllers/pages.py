from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    render_template,
    request,
    send_from_directory,
)

from ..materials import allowed_filenames, get_material, list_materials


pages_bp = Blueprint("pages", __name__)


def _repository():
    return current_app.extensions["coaching_repository"]


@pages_bp.get("/")
def dashboard():
    repository = _repository()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        summary=repository.get_summary(),
        recent_sessions=repository.list_sessions(limit=4),
    )


@pages_bp.get("/coach")
def coach():
    material = get_material(request.args.get("material"))
    if material is None:
        abort(404)
    materials = list_materials()
    return render_template(
        "coach.html",
        active_page="coach",
        material=material,
        material_number=next(
            index
            for index, item in enumerate(materials, start=1)
            if item["id"] == material["id"]
        ),
        materials=materials,
    )


@pages_bp.get("/records")
def records():
    repository = _repository()
    return render_template(
        "records.html",
        active_page="records",
        summary=repository.get_summary(),
        sessions=repository.list_sessions(limit=50),
    )


@pages_bp.get("/records/<int:session_id>")
def record_detail(session_id):
    repository = _repository()
    session = repository.get_session(session_id)
    if not session:
        abort(404)
    return render_template(
        "record_detail.html",
        active_page="records",
        session=session,
        events=repository.list_events(session_id),
    )


@pages_bp.get("/stimuli/<path:filename>")
def stimulus(filename):
    if filename not in allowed_filenames():
        abort(404)
    directory = Path(current_app.config["STIMULI_DIR"])
    return send_from_directory(directory, filename)


@pages_bp.get("/vendor/webgazer.js")
def webgazer_script():
    directory = Path(current_app.config["WEBGAZER_DIR"])
    return send_from_directory(directory, "webgazer.js")


@pages_bp.get("/mediapipe/face_mesh/<path:filename>")
def webgazer_mediapipe_asset(filename):
    allowed = {
        "face_mesh.binarypb",
        "face_mesh.js",
        "face_mesh_solution_packed_assets.data",
        "face_mesh_solution_packed_assets_loader.js",
        "face_mesh_solution_simd_wasm_bin.data",
        "face_mesh_solution_simd_wasm_bin.js",
        "face_mesh_solution_simd_wasm_bin.wasm",
        "face_mesh_solution_wasm_bin.js",
        "face_mesh_solution_wasm_bin.wasm",
    }
    if filename not in allowed:
        abort(404)
    directory = (
        Path(current_app.config["WEBGAZER_DIR"])
        / "mediapipe"
        / "face_mesh"
    )
    return send_from_directory(directory, filename)
