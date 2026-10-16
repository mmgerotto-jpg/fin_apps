#!/usr/bin/env bash
# Instala/atualiza as dependências quando necessário e abre o app no navegador.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Criando ambiente Python (primeira execução)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
fi
if [ ! -f .venv/.deps-ok ] || [ requirements.txt -nt .venv/.deps-ok ]; then
  echo "Instalando dependências..."
  .venv/bin/pip install -r requirements.txt -q
  touch .venv/.deps-ok
fi
exec .venv/bin/streamlit run app.py
