#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Started from the file manager (right-click -> Run as a Program) there is no
# terminal: nothing this script says can be seen, and a failure just looks
# like nothing happening. On a Linux desktop, open a terminal and run there.
# A systemd service (it sets INVOCATION_ID), or anything run with
# MINIMAX_STUDIO_NO_TERMINAL=1, stays where it was started.
if [ "$(uname)" = "Linux" ] && [ -z "${MINIMAX_STUDIO_IN_TERMINAL:-}" ] \
   && [ ! -t 0 ] && [ ! -t 1 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] \
   && [ -z "${INVOCATION_ID:-}${MINIMAX_STUDIO_NO_TERMINAL:-}" ]; then
  export MINIMAX_STUDIO_IN_TERMINAL=1
  SELF="$(pwd)/$(basename "$0")"
  for term in gnome-terminal ptyxis kgx konsole xfce4-terminal mate-terminal \
              tilix x-terminal-emulator xterm; do
    command -v "$term" >/dev/null 2>&1 || continue
    case "$term" in
      gnome-terminal|ptyxis|kgx) "$term" -- bash "$SELF" && exit 0 ;;
      xfce4-terminal|mate-terminal) "$term" -x bash "$SELF" && exit 0 ;;
      *) "$term" -e bash "$SELF" && exit 0 ;;
    esac
  done
  # no terminal found: carry on without one, the page still opens
fi
# in a window we opened, keep it open on failure so the reason can be read
pause_on_error() {
  local code=$?
  if [ -n "${MINIMAX_STUDIO_IN_TERMINAL:-}" ] && [ "$code" -ne 0 ] \
     && [ "$code" -ne 130 ] && [ -t 0 ]; then
    printf "\n  MiniMax Studio stopped (exit code %s). Press Enter to close." "$code"
    read -r _ || true
  fi
}
trap pause_on_error EXIT
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it, then run this again." >&2
  exit 1
fi
# a private venv: current Debian/Ubuntu refuse pip into the system Python
# (PEP 668, "externally-managed-environment")
# A venv whose pip is missing is as good as none: a failed first attempt
# (no python3-venv yet) leaves .venv/bin/python behind without pip, and a
# distro upgrade can remove its interpreter. Rebuild it rather than fail
# the same way on every run.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  "$PY" -m venv .venv || {
    rm -rf .venv
    echo "Could not create a virtual environment. On Debian/Ubuntu install" >&2
    echo "python3-venv (sudo apt install python3-venv), then run this again." >&2
    exit 1
  }
fi
PY=.venv/bin/python
echo "  Using: $($PY -c 'import sys;print(sys.executable)')"
"$PY" -m pip install --disable-pip-version-check --quiet -r requirements.txt
# The server takes this script's place (exec), so whatever stops the
# script stops the server. Only in a window this script opened does
# the shell stay, so the trap above can keep a failure on screen.
if [ -n "${MINIMAX_STUDIO_IN_TERMINAL:-}" ]; then
  "$PY" server.py
else
  exec "$PY" server.py
fi
