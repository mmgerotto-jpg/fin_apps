#!/usr/bin/env bash
# One-step setup + launch for macOS / Linux. Usage: ./start.sh   (add --demo for offline data)
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "Python 3 is not installed. Get it from https://www.python.org/downloads/ and run this again."
  exit 1
fi
if [ ! -d .venv ]; then
  echo "First run: creating virtual environment and installing dependencies (1-2 min)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt -q
fi
( sleep 2; (open http://127.0.0.1:5000 || xdg-open http://127.0.0.1:5000) >/dev/null 2>&1 ) &
exec .venv/bin/python -m optionsapp.app "$@"
