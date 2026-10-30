@echo off
title Lamina Qwen3.6 UD-Q4_K_XL
rem Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL (the MTP GGUF), on http://127.0.0.1:8083/v1 with qwen36-ud-q4_k_xl.json
rem (README.md).  The engine's qwen35 path splits its layers over the config's --main-gpu and the fewest other
rem GPUs that hold every expert; where none do, the main GPU runs alone, the CPU computing the experts it lacks.
cd /d "%~dp0.."
build\lamina-server.exe --config examples\qwen36-ud-q4_k_xl.json
pause
