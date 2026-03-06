from __future__ import annotations

import time
import threading
import config

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

# Signalled once the camera is confirmed open (or failed)
_started_event = threading.Event()
_camera_ok = False


def _detection_loop():
    global _running, _latest_label, _latest_strength, _camera_ok

    # --- Import deps inside the thread so startup is non-blocking ---
    try:
        import cv2
        from fer import FER
    except ImportError as e:
        print(f"[vision] Missing dependency: {e}")
        print("[vision] Install with: pip install fer opencv-python")
        _camera_ok = False
        _started_event.set()
        _running = False
        return

    cap = cv2.VideoCapture(config.VISION_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[vision] Could not open camera (index {config.VISION_CAMERA_INDEX}).")
        print("[vision] Check that no other app is using the camera, then retry.")
        _camera_ok = False
        _started_event.set()
        _running = False
        return

    # Camera confirmed open — signal main thread before blocking in the loop
    _camera_ok = True
    _started_event.set()
    print(f"[vision] Camera {config.VISION_CAMERA_INDEX} open. Facial emotion detection running.")

    # mtcnn=False → Haar cascade (fast on CPU, no extra model download needed)
    detector = FER(mtcnn=False)

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


def start() -> bool:
    """
    Launch the background detection thread and wait until the camera is
    confirmed open (or failed).

    Returns True if the camera started successfully, False otherwise.
    Safe to call multiple times.
    """
    global _running, _thread

    if _running:
        return _camera_ok

    _started_event.clear()
    _running = True
    _thread = threading.Thread(target=_detection_loop, daemon=True, name="vision-fer")
    _thread.start()

    # Block until the camera is either open or has failed — max 10 s
    # (FER + TensorFlow can take a few seconds to initialise)
    print("[vision] Initialising camera …")
    _started_event.wait(timeout=10.0)

    if not _camera_ok:
        _running = False
        print("[vision] Disabled — running without facial emotion detection.")
        return False

    return True


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
    - Vision was never started or camera failed to open
    - No face has been detected in the current frame
    """
    if not _camera_ok:
        return None

    with _lock:
        label = _latest_label
        strength = _latest_strength

    # strength == 0.0 means no face detected yet
    if strength == 0.0:
        return None

    return (label, strength)
