#!/bin/bash

MODEL_NAME="llama3.2:1b"
PROJECT_DIR="$HOME/project"
VENV_ACTIVATE="$HOME/env/bin/activate"
MODEL_PATH="$HOME/project/models/garbage_classificatio.tflite"
LABELS_PATH="$HOME/project/models/labels.txt"

echo "[INFO] Starting Ollama model in background: $MODEL_NAME"
ollama run "$MODEL_NAME" > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to become ready
echo -n "[INFO] Waiting for Ollama server to be ready"
until curl -s http://localhost:11434 > /dev/null; do
  echo -n "."
  sleep 0.5
done
echo " Ready!"

# Activate Python virtual environment
echo "[INFO] Activating virtualenv"
source "$VENV_ACTIVATE"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Run your Python program
echo "[INFO] Running main.py with model and labels"
python3 main.py --model-path="$MODEL_PATH" --labels="$LABELS_PATH"

# Cleanup
echo "[INFO] Cleaning up Ollama process ($OLLAMA_PID)"
kill "$OLLAMA_PID"
