from __future__ import annotations

import time
import threading
import config

# Lazy imports — only pulled in when vision is actually started
_cv2 = None
_FER = None

# Mapping from FER emotion labels to the agent's internal emotion vocabulary
_LABEL_MAP: dict[str, str] = {
    "happy": "calm",
    "sad": "sad",
    "angry": "angry",
    "fear": "anxious",
    "disgust": "frustrated",
    "surprise": "overwhelmed",
    "neutral": "neutral",
}

# Shared state — protected by _lock
_lock = threading.Lock()
_latest_label: str = "neutral"
_latest_strength: float = 0.0
_running = False
_thread: threading.Thread | None = None


def _load_deps() -> bool:
    """Import heavy deps lazily so they don't slow down non-vision runs."""
    global _cv2, _FER
    try:
        import cv2 as _cv2_mod
        from fer import FER as _FER_cls
        _cv2 = _cv2_mod
        _FER = _FER_cls
        return True
    except ImportError as e:
        print(f"[vision] Missing dependency: {e}. Run: pip install fer opencv-python-headless")
        return False


def _detection_loop():
    global _running, _latest_label, _latest_strength

    if not _load_deps():
        _running = False
        return

    cap = _cv2.VideoCapture(config.VISION_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[vision] Could not open camera index {config.VISION_CAMERA_INDEX}. Vision disabled.")
        _running = False
        return

    detector = _FER(mtcnn=False)  # mtcnn=False uses Haar cascades — much faster on CPU

    print("[vision] Camera active. Facial emotion detection running.")

    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(config.VISION_FRAME_INTERVAL)
            continue

        try:
            results = detector.detect_emotions(frame)
            if results:
                # Take the first (largest) detected face
                emotions: dict[str, float] = results[0]["emotions"]
                top_label = max(emotions, key=emotions.get)
                top_strength = float(emotions[top_label])
                mapped = _LABEL_MAP.get(top_label, "neutral")

                with _lock:
                    _latest_label = mapped
                    _latest_strength = top_strength
        except Exception as e:
            print(f"[vision] Detection error: {e}")

        time.sleep(config.VISION_FRAME_INTERVAL)

    cap.release()
    print("[vision] Camera released.")


def start():
    """Launch the background detection thread. Safe to call multiple times."""
    global _running, _thread

    if _running:
        return

    _running = True
    _thread = threading.Thread(target=_detection_loop, daemon=True, name="vision-fer")
    _thread.start()


def stop():
    """Signal the detection thread to stop and wait for it to finish."""
    global _running, _thread

    _running = False
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=3.0)
    _thread = None
    print("[vision] Stopped.")


def get_emotion() -> tuple[str, float] | None:
    """
    Return the latest detected facial emotion as (label, strength).

    Returns None if:
    - Vision was never started
    - Camera could not be opened
    - No face has been detected yet (strength == 0 and label == neutral default)
    """
    if not _running and _thread is None:
        return None

    with _lock:
        label = _latest_label
        strength = _latest_strength

    # Don't return a result until at least one real detection has occurred
    if strength == 0.0:
        return None

    return (label, strength)
