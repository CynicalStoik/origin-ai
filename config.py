import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # no HF network calls
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # silence LOAD REPORT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Study condition ────────────────────────────────────────────────────────────
# True  → Condition A: PAM agent (ENGRAM + Perspective Attributed Memory)
# False → Condition B: baseline agent (ENGRAM only, write-time resolution)
PAM_ENABLED = True
# ──────────────────────────────────────────────────────────────────────────────

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("OLLAMA_HOST", OLLAMA_HOST)

WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
TTS_VOICE = "en-US-EmmaMultilingualNeural"
TTS_RATE = "+5%"
TTS_PITCH = "-3Hz"
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.2

CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
STM_WINDOW_SIZE = 10  # number of recent turns to keep in short-term memory
LPM_PATH = os.path.join(BASE_DIR, "data", "persona.json")

DIVERGENCE_EPSILON = 0.15
EMOTION_THRESHOLD = 0.6
CONFIDENCE_DECAY_RATE = 0.05  # per-hour decay multiplier

# Vision / facial emotion detection
VISION_CAMERA_INDEX = int(os.environ.get("VISION_CAMERA", "0"))  # webcam device index
VISION_FRAME_INTERVAL = 0.5  # seconds between FER inference calls (avoid CPU spike)

# Turn Completion Projection 
IPU_DURATION = 0.7
TURN_RATIO_THRESHOLD = 0.6
TURN_FALLBACK_THRESHOLD = 1.25
MIN_SPEECH_DURATION = 1.0
TURN_PROJECTION_N = 3
TURN_PROJECTION_M = 3

PERCEPTION_FAST_WORD_LIMIT = 2
GROUNDING_MIN_TURNS = 2
GROUNDING_PROBABILITY = 0.6
USE_LLM_PROJECTION = True
