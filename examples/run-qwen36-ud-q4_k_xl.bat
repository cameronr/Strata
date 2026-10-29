@echo off
title Lamina Qwen3.6 UD-Q4_K_XL
rem Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL (the MTP GGUF), on http://127.0.0.1:8083/v1 with qwen36-ud-q4_k_xl.json
rem (README.md).  The engine's Qwen3.6 path runs on one GPU, the config's --main-gpu; the experts that do not fit in
rem its VRAM run on the CPU.
rem A PYTHONPATH set for another Python would mix that Python's packages into the venv's.
set "PYTHONPATH="
cd /d "%~dp0.."
".venv\Scripts\python.exe" serve\server.py --engine strata --config examples\qwen36-ud-q4_k_xl.json
pause
