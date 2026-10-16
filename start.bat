@echo off
rem Instala as dependencias (so na primeira vez) e abre o app no navegador.
cd /d "%~dp0"
if not exist .venv (
  echo Criando ambiente Python (primeira execucao)...
  python -m venv .venv
  .venv\Scripts\pip install --upgrade pip -q
  .venv\Scripts\pip install -r requirements.txt -q
)
.venv\Scripts\streamlit run app.py
