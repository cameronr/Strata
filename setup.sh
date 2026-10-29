#!/bin/sh
# Lamina's setup on Linux: makes the Python environment .venv on the first run, then runs setup.py in it with these
# arguments, e.g. ./setup.sh --model iq3_s (./setup.sh --help: all of them). It needs Python 3.10 or newer with venv
# (Debian and Ubuntu: sudo apt install python3-venv).
cd "$(dirname "$0")" || exit 1
unset PYTHONPATH PIP_TARGET
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PY=$c
      break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10 or newer with venv is needed (Debian and Ubuntu: sudo apt install python3-venv)"
    exit 1
  fi
  "$PY" -m venv .venv || { rm -rf .venv; echo "could not make the Python environment in .venv"; exit 1; }
fi
exec .venv/bin/python setup.py "$@"
