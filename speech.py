from __future__ import annotations

import io
import queue
import config
import pygame
import asyncio
import edge_tts
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

_whisper_model: WhisperModel | None = None
_pygame_inited = False


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print("[speech] Loading Whisper model …")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def _init_pygame():
    global _pygame_inited
    if not _pygame_inited:
        pygame.mixer.init()
        _pygame_inited = True


def record_audio() -> np.ndarray:
    sr = config.SAMPLE_RATE
    block_dur = 0.1
    block_size = int(sr * block_dur)
    silence_blocks_needed = int(config.SILENCE_DURATION / block_dur)

    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    recording_done = threading.Event()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    frames: list[np.ndarray] = []
    silent_count = 0
    speech_started = False

    print("[speech] Listening … (speak, then pause to finish)")
    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        while not recording_done.is_set():
            try:
                block = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(block**2)))

            if rms > config.SILENCE_THRESHOLD:
                speech_started = True
                silent_count = 0
            else:
                silent_count += 1

            if speech_started:
                frames.append(block)

            if speech_started and silent_count >= silence_blocks_needed:
                recording_done.set()

    if not frames:
        return np.zeros(0, dtype="float32")

    audio = np.concatenate(frames, axis=0).flatten()
    return audio


def transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    model = _get_whisper_model()
    segments, _ = model.transcribe(audio, beam_size=3, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _synthesise(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, config.TTS_VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def speak(text: str):
    if not text:
        return
    _init_pygame()
    mp3_bytes = asyncio.run(_synthesise(text))
    buf = io.BytesIO(mp3_bytes)
    pygame.mixer.music.load(buf, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
