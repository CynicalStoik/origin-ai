from __future__ import annotations
import io
import queue
import config
import pygame
import ollama
import asyncio
import edge_tts
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

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


# Turn-completion projection (Ekstedt & Skantze, 2021)


def _project_turn_completion(
    partial_transcript: str,
    context: list[dict],
) -> float:
    """
    Generate N continuations in parallel and return the ratio
    that signal turn-completion. No transcript punctuation check —
    only LLM continuations decide.
    """
    if not partial_transcript.strip():
        return 0.0

    n = getattr(config, "TURN_PROJECTION_N", 3)
    m = getattr(config, "TURN_PROJECTION_M", 3)

    system_prompt = (
        "You are predicting how a speaker will finish their sentence in a conversation. "
        "Continue the partial utterance naturally in 1-5 words. "
        "If the utterance already sounds complete — including questions, greetings, "
        "or statements that need no more words — respond with exactly: <END>"
    )

    context_str = ""
    for turn in context[-6:]:
        role = "User" if turn["role"] == "student" else "Coach"
        context_str += f"{role}: {turn['content']}\n"
    context_str += f"User (partial): {partial_transcript}"

    def single_completion(_):
        try:
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_str},
                ],
                options={
                    "temperature": 1.0,
                    "top_k": 5,
                    "num_predict": m,
                },
            )
            continuation = response["message"]["content"].strip()
            return (
                "<END>" in continuation
                or continuation == ""
                or continuation.endswith((".", "?", "!"))
                or len(continuation.split()) <= 1
            )
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(single_completion, range(n)))

    return sum(results) / n


def record_audio(context: list[dict] | None = None) -> np.ndarray:
    """
    Record audio using projection-based turn completion (Ekstedt & Skantze, 2021).

    Strategy:
    - Listen in 100ms blocks
    - After each IPU_DURATION silence, transcribe what we have so far
    - Only project if user has spoken for at least MIN_SPEECH_DURATION seconds
    - Reuse last transcript if no new audio frames since last projection
    - Project N continuations in parallel — if ratio >= TURN_RATIO_THRESHOLD, turn complete
    - Fallback: if total silence exceeds TURN_FALLBACK_THRESHOLD, always end turn
    """
    sr = config.SAMPLE_RATE
    ipu_dur = getattr(config, "IPU_DURATION", 0.4)
    ratio_threshold = getattr(config, "TURN_RATIO_THRESHOLD", 0.6)
    fallback_threshold = getattr(config, "TURN_FALLBACK_THRESHOLD", 1.25)
    min_speech_duration = getattr(config, "MIN_SPEECH_DURATION", 1.5)
    block_dur = 0.1
    block_size = int(sr * block_dur)

    if context is None:
        context = []

    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    recording_done = threading.Event()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    frames: list[np.ndarray] = []
    speech_started = False
    ipu_silent_time = 0.0
    total_silent_time = 0.0
    speech_duration = 0.0
    last_transcript = ""
    last_frames_count = 0

    print("[speech] Listening … (speak naturally)")

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
            is_silent = rms <= config.SILENCE_THRESHOLD

            if not is_silent:
                speech_started = True
                ipu_silent_time = 0.0
                total_silent_time = 0.0

            if speech_started:
                frames.append(block)

                if not is_silent:
                    speech_duration += block_dur

                if is_silent:
                    ipu_silent_time += block_dur
                    total_silent_time += block_dur

                    # Fallback: max silence exceeded → always end turn
                    if total_silent_time >= fallback_threshold:
                        print("[speech] Fallback threshold reached — ending turn")
                        recording_done.set()
                        break

                    # IPU triggered → maybe run projection
                    if ipu_silent_time >= ipu_dur:
                        ipu_silent_time = 0.0

                        # Skip if user hasn't spoken long enough
                        if speech_duration < min_speech_duration:
                            print(f"[speech] Skipping projection — speech too short ({speech_duration:.1f}s)")
                            continue

                        # Reuse last transcript if no new frames since last check
                        if len(frames) > last_frames_count:
                            partial_audio = np.concatenate(frames, axis=0).flatten()
                            partial_text = transcribe(partial_audio)
                            last_transcript = partial_text
                            last_frames_count = len(frames)
                        else:
                            partial_text = last_transcript

                        if partial_text:
                            print(f"[speech] Projecting on: '{partial_text}'")
                            ratio = _project_turn_completion(partial_text, context)
                            print(f"[speech] Ratio: {ratio:.2f} (threshold: {ratio_threshold})")

                            if ratio >= ratio_threshold:
                                print("[speech] Turn complete (projection)")
                                recording_done.set()
                                break

    if not frames:
        return np.zeros(0, dtype="float32")

    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    model = _get_whisper_model()
    segments, _ = model.transcribe(audio, beam_size=3, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _synthesise(text: str) -> bytes:
    communicate = edge_tts.Communicate(
        text,
        config.TTS_VOICE,
        rate=config.TTS_RATE,
        pitch=config.TTS_PITCH,
    )
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
    if not mp3_bytes:
        return
    buf = io.BytesIO(mp3_bytes)
    pygame.mixer.music.load(buf, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)