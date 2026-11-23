#!/usr/bin/env bash
# Color Alignment System Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "======================================================================="
echo "   Professional Color Align & Evaluation Web Service"
echo "======================================================================="
echo "Starting web server on http://localhost:8088 ..."

if command -v uv &> /dev/null; then
    uv run --with "fastapi,uvicorn,python-multipart,numpy,opencv-python,pillow" python3 app.py
else
    python3 app.py
fi
