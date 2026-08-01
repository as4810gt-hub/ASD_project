import argparse
import importlib
import shutil
import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from modules.eye_tracker import EyeTracker

SCREEN_W = 1280
SCREEN_H = 720
WINDOW = "ASD Eye Tracker v4"
REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("mediapipe", "mediapipe"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("deepface", "deepface"),
    ("tensorflow", "tensorflow"),
    ("PIL", "Pillow"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="啟動新版 ASD 眼動 + 情緒辨識系統")
    parser.add_argument("--check", action="store_true", help="只做環境與相機檢查，不啟動主畫面")
    return parser.parse_args()


def ensure_ascii_task_model() -> str:
    src = BASE_DIR / "models" / "eye" / "face_landmarker.task"
    dst = Path(r"C:\asd_v4\models\eye\face_landmarker.task")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and src.exists():
        shutil.copy2(src, dst)
    return str(dst if dst.exists() else src)


def check_environment() -> list[str]:
    missing = []
    for module_name, package_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"- {module_name} ({package_name})：{exc}")

    model_files = [
        BASE_DIR / "models" / "eye" / "severity_classifier.pkl",
        BASE_DIR / "models" / "eye" / "face_landmarker.task",
    ]
    for path in model_files:
        if not path.exists():
            missing.append(f"- 缺少模型檔：{path.name}")

    return missing


def run_check_mode() -> int:
    print("[檢查] 開始檢查執行環境...")
    missing = check_environment()
    if missing:
        print("[檢查] 發現以下問題：")
        for item in missing:
            print(item)
        print("[檢查] 建議執行：python -m pip install -r requirements.txt")
        return 1

    print("[檢查] 依賴與模型檔皆已就緒。")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[檢查] 相機無法開啟，請確認攝影機權限或裝置連接。")
        cap.release()
        return 2

    print("[檢查] 相機可正常開啟，已準備好啟動主程式。")
    cap.release()
    return 0


def main():
    args = parse_args()
    if args.check:
        sys.exit(run_check_mode())

    missing = check_environment()
    if missing:
        print("[啟動] 執行環境檢查未通過：")
        for item in missing:
            print(item)
        print("[啟動] 請先執行：python -m pip install -r requirements.txt")
        return

    print("[啟動] 初始化新版眼動 + 情緒辨識系統...")
    tracker = EyeTracker(
        model_path=str(BASE_DIR / "models" / "eye" / "severity_classifier.pkl"),
        task_model=ensure_ascii_task_model(),
        window_sec=5.0,
        screen_w=SCREEN_W,
        screen_h=SCREEN_H,
        ivt_threshold=30.0,
        min_fix_ms=80.0,
        emotion_every_n=15,
    )

    print("[啟動] 開啟相機...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

    if not cap.isOpened():
        print("ERROR: 無法開啟相機，請確認攝影機權限或裝置連接。")
        tracker.release()
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, SCREEN_W, SCREEN_H)
    print("[啟動] 請對準鏡頭，按 q 結束。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracker.process_frame(frame)
            out = tracker.draw_overlay(frame.copy())
            cv2.imshow(WINDOW, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()


if __name__ == "__main__":
    main()
