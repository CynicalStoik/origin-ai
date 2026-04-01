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
_tts_loop: asyncio.AbstractEventLoop | None = None
_tts_thread: threading.Thread | None = None


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


def preload():
    """Eagerly load the Whisper model so the first transcription has no load latency."""
    _get_whisper_model()
    print("[speech] Whisper model ready.")


def _init_pygame():
    global _pygame_inited
    if not _pygame_inited:
        pygame.mixer.init()
        _pygame_inited = True


def _get_tts_loop() -> asyncio.AbstractEventLoop:
    """Persistent event loop for TTS — avoids asyncio.run() overhead per call."""
    global _tts_loop, _tts_thread
    if _tts_loop is None:
        _tts_loop = asyncio.new_event_loop()
        _tts_thread = threading.Thread(
            target=_tts_loop.run_forever, daemon=True, name="tts-loop"
        )
        _tts_thread.start()
    return _tts_loop


def _is_turn_complete(transcript: str) -> bool:
    """Heuristic turn-completion using Whisper's punctuation and common patterns."""
    t = transcript.strip()
    if not t:
        return False

    if t[-1] in ".?!":
        return True

    words = t.lower().split()
    if not words:
        return False

    last = words[-1].rstrip(".,!? ")
    one_word = {
        "bye", "goodbye", "thanks", "okay", "alright", "yeah", "yep",
        "nope", "sure", "right", "later", "hmm", "hm", "mhm",
    }
    if last in one_word:
        return True

    if len(words) >= 2:
        tail = " ".join(words[-2:]).rstrip(".,!? ")
        two_word = {
            "thank you", "i guess", "take care", "see you",
            "not really", "that's it", "think so",
        }
        if tail in two_word:
            return True

    return False


# LLM-based turn projection (Ekstedt & Skantze, 2021) — opt-in via config

def _project_turn_completion(
    partial_transcript: str,
    context: list[dict],
) -> float:
    if not partial_transcript.strip():
        return 0.0

    import ollama

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
                options={"temperature": 1.0, "top_k": 5, "num_predict": m},
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

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(single_completion, range(n)))

    return sum(results) / n


def record_audio(context: list[dict] | None = None) -> np.ndarray:
    """Record with silence detection + Whisper heuristic turn completion."""
    sr = config.SAMPLE_RATE
    silence_dur = config.SILENCE_DURATION
    ipu_dur = getattr(config, "IPU_DURATION", 0.4)
    min_speech = getattr(config, "MIN_SPEECH_DURATION", 1.0)
    use_llm = getattr(config, "USE_LLM_PROJECTION", False)
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
    silent_time = 0.0
    speech_duration = 0.0
    checked_this_silence = False

    print("[speech] Listening …")

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
                silent_time = 0.0
                checked_this_silence = False

            if speech_started:
                frames.append(block)

                if not is_silent:
                    speech_duration += block_dur

                if is_silent:
                    silent_time += block_dur

                    # hard cutoff
                    if silent_time >= silence_dur:
                        recording_done.set()
                        break

                    # heuristic check at IPU boundary
                    if (
                        not checked_this_silence
                        and silent_time >= ipu_dur
                        and speech_duration >= 0.3
                    ):
                        checked_this_silence = True
                        partial_audio = np.concatenate(frames, axis=0).flatten()
                        partial_text = transcribe(partial_audio)

                        if partial_text and _is_turn_complete(partial_text):
                            print(f"[speech] Turn complete: '{partial_text}'")
                            recording_done.set()
                            break

                        # optional LLM projection for longer speech
                        if (
                            use_llm
                            and partial_text
                            and speech_duration >= min_speech
                        ):
                            ratio_threshold = getattr(
                                config, "TURN_RATIO_THRESHOLD", 0.6
                            )
                            ratio = _project_turn_completion(partial_text, context)
                            if ratio >= ratio_threshold:
                                print("[speech] Turn complete (LLM projection)")
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
    loop = _get_tts_loop()
    future = asyncio.run_coroutine_threadsafe(_synthesise(text), loop)
    mp3_bytes = future.result(timeout=15)
    if not mp3_bytes:
        return
    buf = io.BytesIO(mp3_bytes)
    pygame.mixer.music.load(buf, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)
