@echo off
title Lamina Qwen3.8-27B UD-Q6_K_M
rem Qwen3.8-27B, unsloth's UD-Q6_K_M, on http://127.0.0.1:8084/v1 with qwen38-27b-ud-q6_k_m.json (README.md).
rem The engine's qwen35 path splits its layers over the config's --main-gpu and the fewest of the other GPUs
rem that hold it (~36 GB of VRAM at the full context, K and V in q8_0).
cd /d "%~dp0.."
build\lamina-server.exe --config examples\qwen38-27b-ud-q6_k_m.json
pause
