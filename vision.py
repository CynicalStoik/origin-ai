from __future__ import annotations

import time
import threading
import config

# DeepFace emotion labels → agent's internal emotion vocabulary.
# NOTE: "happy" must stay "happy" — remapping it to "calm" breaks conflict
# detection because agent.py scores valence using the raw label.
_LABEL_MAP: dict[str, str] = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "disgust": "disgusted",
    "surprise": "surprised",
    "neutral": "neutral",
}

# Shared state — protected by _lock
_lock = threading.Lock()
_latest_label: str = "neutral"
_latest_strength: float = 0.0
_has_detection: bool = False   # True once DeepFace has returned at least one result
_running = False
_thread: threading.Thread | None = None

# Signalled once the camera is confirmed open (or failed)
_started_event = threading.Event()
_camera_ok = False


def _detection_loop():
    global _running, _latest_label, _latest_strength, _has_detection, _camera_ok

    try:
        import cv2
        from deepface import DeepFace
    except (ImportError, ValueError, Exception) as e:
        print(f"[vision] Failed to load dependencies: {e}")
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

    _camera_ok = True
    _started_event.set()
    print(
        f"[vision] Camera {config.VISION_CAMERA_INDEX} open. Facial emotion detection running."
    )

    consecutive_errors = 0

    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(config.VISION_FRAME_INTERVAL)
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            face = result[0] if isinstance(result, list) else result
            emotions: dict[str, float] = face["emotion"]
            dominant: str = face["dominant_emotion"]
            strength = float(emotions.get(dominant, 0.0)) / 100.0
            mapped = _LABEL_MAP.get(dominant, "neutral")

            with _lock:
                _latest_label = mapped
                _latest_strength = strength
                _has_detection = True

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors <= 2:
                print(f"[vision] Detection error: {str(e).encode('ascii', errors='replace').decode()}")
            if consecutive_errors == 3:
                print("[vision] Repeated failures - silencing further errors. Vision may be degraded.")
            if consecutive_errors >= 10:
                print("[vision] Too many errors — disabling vision.")
                _running = False
                break

        time.sleep(config.VISION_FRAME_INTERVAL)

    cap.release()
    print("[vision] Camera released.")


def _request_camera_permission() -> bool:
    try:
        import cv2
    except (ImportError, ValueError):
        return True

    print("[vision] Requesting camera access …")
    cap = cv2.VideoCapture(config.VISION_CAMERA_INDEX)
    ok = cap.isOpened()
    cap.release()

    if not ok:
        print(
            f"[vision] Camera (index {config.VISION_CAMERA_INDEX}) could not be opened."
        )
        print(
            "[vision] Grant camera access in System Settings → Privacy → Camera, then retry."
        )
        return False

    return True


def start() -> bool:
    global _running, _thread

    if _running:
        return _camera_ok

    if not _request_camera_permission():
        return False

    _started_event.clear()
    _running = True
    _thread = threading.Thread(
        target=_detection_loop, daemon=True, name="vision-deepface"
    )
    _thread.start()

    print("[vision] Initialising camera …")
    _started_event.wait(timeout=20.0)

    if not _camera_ok:
        _running = False
        print("[vision] Disabled — running without facial emotion detection.")
        return False

    return True


def stop():
    global _running, _thread

    _running = False
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=3.0)
    _thread = None
    print("[vision] Stopped.")


def get_emotion() -> tuple[str, float] | None:
    """
    Return the latest detected facial emotion as (label, strength).
    Returns None if camera failed or no detection has occurred yet.
    """
    if not _camera_ok:
        return None

    with _lock:
        has = _has_detection
        label = _latest_label
        strength = _latest_strength

    # Use _has_detection flag instead of strength == 0.0 check.
    # The old guard blocked valid results when the dominant emotion
    # happened to score very low confidence.
    if not has:
        return None

    return (label, strength)