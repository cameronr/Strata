@echo off
rem Benchmark this Lamina install the way it is used and write a report you can share.
rem It starts the model with your settings (or uses the one already running), takes about 10 minutes and changes nothing.
rem Options: --clients 4 (several requests at once), --long (a 32K prompt), --compare <older results folder>.
setlocal
title Lamina - benchmark
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  Lamina is not set up here yet: run setup.bat --model ... first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\strata_bench.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
