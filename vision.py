from __future__ import annotations

import time
import threading
import config

# DeepFace emotion labels → agent's internal emotion vocabulary
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

    # Import inside thread — these are heavy and slow to load
    try:
        import cv2
        from deepface import DeepFace
    except ImportError as e:
        print(f"[vision] Missing dependency: {e}")
        print("[vision] Install with: pip install deepface opencv-python")
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

    while _running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(config.VISION_FRAME_INTERVAL)
            continue

        try:
            # enforce_detection=False → no exception if no face in frame
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            # result is a list; take the first (dominant) face
            face = result[0] if isinstance(result, list) else result
            emotions: dict[str, float] = face["emotion"]
            dominant: str = face["dominant_emotion"]
            strength = (
                float(emotions.get(dominant, 0.0)) / 100.0
            )  # DeepFace gives 0–100
            mapped = _LABEL_MAP.get(dominant, "neutral")

            with _lock:
                _latest_label = mapped
                _latest_strength = strength

        except Exception as e:
            print(f"[vision] Detection error: {e}")

        time.sleep(config.VISION_FRAME_INTERVAL)

    cap.release()
    print("[vision] Camera released.")


def _request_camera_permission() -> bool:
    """
    Open and immediately release the camera on the main thread.

    On macOS, AVFoundation requires the permission dialog to be triggered
    from the main thread. If we skip this and open the camera inside a
    background thread, macOS blocks the request entirely with status 0.
    """
    try:
        import cv2
    except (ImportError, ValueError):
        return True  # will be caught again in the thread with a proper message

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
    """
    Launch the background detection thread and block until the camera is
    confirmed open (or has failed). Returns True on success.
    Safe to call multiple times.
    """
    global _running, _thread

    if _running:
        return _camera_ok

    # macOS: trigger the camera permission dialog on the main thread first
    if not _request_camera_permission():
        return False

    _started_event.clear()
    _running = True
    _thread = threading.Thread(
        target=_detection_loop, daemon=True, name="vision-deepface"
    )
    _thread.start()

    print("[vision] Initialising camera …")
    # Wait up to 20 s — DeepFace downloads its model weights on first run
    _started_event.wait(timeout=20.0)

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
    Returns None if camera failed to open or no face detected yet.
    """
    if not _camera_ok:
        return None

    with _lock:
        label = _latest_label
        strength = _latest_strength

    # strength == 0.0 means no real detection has occurred yet
    if strength == 0.0:
        return None

    return (label, strength)
