#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${LLM_MODEL:-gemma3:4b}"
MODE="${1:-text}"   # text | speech
VISION="${2:-}"     # vision (optional second arg)

if [ "$MODE" != "text" ] && [ "$MODE" != "speech" ]; then
    echo "Usage: ./run.sh [text|speech] [vision]"
    echo ""
    echo "  Examples:"
    echo "    ./run.sh text                  # text only"
    echo "    ./run.sh speech                # speech (mic + TTS)"
    echo "    ./run.sh speech vision         # speech + webcam emotion detection"
    echo "    ./run.sh text vision           # text + webcam emotion detection"
    echo ""
    echo "  Swap model:  LLM_MODEL=llama3:8b ./run.sh text"
    exit 1
fi

if [ -n "$VISION" ] && [ "$VISION" != "vision" ]; then
    echo "Unknown second argument '$VISION'. Did you mean 'vision'?"
    exit 1
fi

# Python
PYTHON=""
for cmd in python3.13 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done
[ -z "$PYTHON" ] && echo "Python 3.10+ required." && exit 1

# Venv
if [[ ! -d .venv && ! -d venv ]]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi
if [ -d .venv ]; then
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
elif [ -d venv ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
fi
# Deps
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Portaudio (speech mode needs it)
if [ "$MODE" = "speech" ]; then
    if [[ "$(uname)" == "Darwin" ]] && command -v brew &>/dev/null; then
        brew list portaudio &>/dev/null 2>&1 || brew install portaudio
    elif [[ "$(uname)" == "Linux" ]] && command -v apt-get &>/dev/null; then
        dpkg -s libportaudio2 &>/dev/null 2>&1 || sudo apt-get install -y -qq libportaudio2 portaudio19-dev
    fi
fi

# OpenCV system deps (vision mode, Linux only)
if [ "$VISION" = "vision" ]; then
    if [[ "$(uname)" == "Linux" ]] && command -v apt-get &>/dev/null; then
        dpkg -s libgl1 &>/dev/null 2>&1 || sudo apt-get install -y -qq libgl1
    fi
fi

# Ollama
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    if [[ "$(uname)" == "Darwin" ]] && command -v brew &>/dev/null; then
        brew install ollama
    elif [[ "$(uname)" == "Linux" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Install Ollama manually: https://ollama.com/download" && exit 1
    fi
fi

if ! curl -s http://localhost:11434/api/version &>/dev/null; then
    echo "Starting Ollama server..."
    ollama serve &>/dev/null &
    sleep 3
    curl -s http://localhost:11434/api/version &>/dev/null || { echo "Ollama failed to start. Run 'ollama serve' manually."; exit 1; }
fi

echo "Pulling model $MODEL..."
ollama pull "$MODEL"

# Build Python flags
MODE_FLAG=""
[ "$MODE" = "text" ] && MODE_FLAG="--text"

VISION_FLAG=""
[ "$VISION" = "vision" ] && VISION_FLAG="--vision"

# Run
exec env LLM_MODEL="$MODEL" python main.py $MODE_FLAG $VISION_FLAG
