"""Optional speech input (microphone) for the Sage agent.

Design goals:
- Keep the existing agent flow unchanged: speech -> transcript -> same memory/prompt pipeline.
- Work as an optional dependency: if packages are missing, raise a clear error.

Usage (from agent.py):
  wav_path = record_from_mic()
  text, meta = transcribe_wav(wav_path)
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
import soundfile as sf
import faster_whisper as fw


@dataclass
class ASRMeta:
    confidence: float | None = None
    language: str | None = None
    duration_s: float | None = None


def record_from_mic(
    samplerate: int = 16000,
    channels: int = 1,
) -> tuple[str, float]:
    """Record audio from the default microphone until the user presses Enter.

    Returns a path to a temporary WAV file.
    """

    frames: list[bytes] = []
    start = time.time()
    stop_flag = {"stop": False}

    def callback(indata, _frames, _time, status):
        if status:
            # Status messages are non-fatal (buffer over/under runs). We still keep recording.
            pass
        frames.append(indata.copy())
        if stop_flag["stop"]:
            raise sd.CallbackStop()

    # Start streaming
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        input("Recording… press Enter to stop.")
        stop_flag["stop"] = True

    duration = time.time() - start

    if not frames:
        raise RuntimeError("No audio captured from microphone.")

    audio_np = np.concatenate(frames, axis=0)

    fd, wav_path = tempfile.mkstemp(prefix="sage_mic_", suffix=".wav")
    os.close(fd)
    sf.write(wav_path, audio_np, samplerate)

    return wav_path, duration


def transcribe_wav(
    wav_path: str,
    duration_s: float,
    model_size: str = "small",
    device: str | None = None,
) -> tuple[str, dict]:
    """Transcribe a WAV file using faster-whisper.

    Returns:
      (transcript, meta)

    meta includes a heuristic confidence score in [0, 1] when available.
    """
    
    try:
        if duration_s != duration_s:  # nan check
            duration_s = None
    except Exception:
        duration_s = None

    WhisperModel = fw.WhisperModel
    model = WhisperModel(model_size, device=device or "cpu", compute_type="int8")

    segments, info = model.transcribe(wav_path, beam_size=5)
    texts = []
    confidences = []
    for seg in segments:
        if getattr(seg, "text", None):
            texts.append(seg.text.strip())
        # faster-whisper exposes avg_logprob; map it to 0..1
        lp = getattr(seg, "avg_logprob", None)
        if lp is not None:
            # Typical avg_logprob ranges roughly [-2.5, 0]
            c = max(0.0, min(1.0, (lp + 2.5) / 2.5))
            confidences.append(c)

    transcript = " ".join([t for t in texts if t])
    conf = (sum(confidences) / len(confidences)) if confidences else None

    meta = {
        "confidence": conf,
        "language": getattr(info, "language", None),
        "duration_s": duration_s,
    }
    return transcript, meta
