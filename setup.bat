@echo off
rem Lamina's setup on Windows: makes the Python environment .venv on the first run, then runs setup.py in it with these
rem arguments, e.g. setup.bat --model iq3_s (setup.bat --help: all of them). It needs 64-bit Python 3.10 or newer.
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PIP_TARGET="
if exist ".venv\Scripts\python.exe" goto run
call :findpy
if defined PY goto venv
echo 64-bit Python 3.10 or newer is needed: https://www.python.org/downloads/ (check "Add python.exe to PATH")
exit /b 1
:venv
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto run
echo Could not make the Python environment in .venv
exit /b 1
:run
".venv\Scripts\python.exe" setup.py %*
exit /b %ERRORLEVEL%

:findpy
rem the py launcher, then python on the PATH (the Microsoft Store's stub fails the test), then the per-user folders
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10) or sys.maxsize < 2**32)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys; sys.exit(sys.version_info < (3, 10) or sys.maxsize < 2**32)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (314 313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof
