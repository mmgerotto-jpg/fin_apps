#!/usr/bin/env bash
# Instala as dependências (só na primeira vez) e abre o app no navegador.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Criando ambiente Python (primeira execução)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt -q
fi
exec .venv/bin/streamlit run app.py
