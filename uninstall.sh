#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

echo "This will remove:"
echo "  - .venv/              (Python virtual environment)"
echo "  - data/chroma_db/     (vector database)"
echo "  - data/persona.json   (saved student profile)"
echo "  - Ollama models       (downloaded LLM weights)"
echo "  - Ollama              (the binary itself)"
echo "  - portaudio           (system audio library, if installed by run.sh)"
echo ""
read -p "Continue? [y/N] " confirm
[[ "$confirm" != [yY] ]] && echo "Aborted." && exit 0

# Project data
echo "Removing .venv..."
rm -rf .venv

echo "Removing chroma_db..."
rm -rf data/chroma_db

echo "Removing persona.json..."
rm -f data/persona.json

# Ollama models + binary
if command -v ollama &>/dev/null; then
    echo "Removing all Ollama models..."
    for model in $(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}'); do
        ollama rm "$model" 2>/dev/null || true
    done

    echo "Uninstalling Ollama..."
    if [[ "$(uname)" == "Darwin" ]] && command -v brew &>/dev/null; then
        brew uninstall ollama 2>/dev/null || true
    elif [[ "$(uname)" == "Linux" ]]; then
        sudo rm -f /usr/local/bin/ollama
        sudo rm -rf /usr/share/ollama
        sudo userdel ollama 2>/dev/null || true
        sudo groupdel ollama 2>/dev/null || true
        sudo rm -f /etc/systemd/system/ollama.service
        sudo systemctl daemon-reload 2>/dev/null || true
    fi
    rm -rf ~/.ollama
else
    echo "Ollama not found, skipping."
fi

# Portaudio
if [[ "$(uname)" == "Darwin" ]] && command -v brew &>/dev/null; then
    if brew list portaudio &>/dev/null 2>&1; then
        echo "Removing portaudio..."
        brew uninstall portaudio
    fi
elif [[ "$(uname)" == "Linux" ]] && command -v apt-get &>/dev/null; then
    if dpkg -s libportaudio2 &>/dev/null 2>&1; then
        echo "Removing portaudio..."
        sudo apt-get remove -y -qq libportaudio2 portaudio19-dev
    fi
fi

echo ""
echo "Done. Everything cleaned up."
