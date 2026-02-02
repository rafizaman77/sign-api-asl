#!/bin/sh
# Run Sign Language Recognition API (Linux / macOS)
cd "$(dirname "$0")"
if [ -f "env/bin/activate" ]; then
    . env/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
fi
echo "Starting API..."
python3 api.py
