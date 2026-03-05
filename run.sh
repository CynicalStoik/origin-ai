#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${LLM_MODEL:-gemma3:4b}"
MODE="${1:-text}"

if [ "$MODE" != "text" ] && [ "$MODE" != "speech" ]; then
    echo "Usage: ./run.sh [text|speech]"
    echo ""
    echo "  Swap model:  LLM_MODEL=llama3:8b ./run.sh text"
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
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

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

# Run
if [ "$MODE" = "text" ]; then
    exec env LLM_MODEL="$MODEL" python main.py --text
else
    exec env LLM_MODEL="$MODEL" python main.py
fi
