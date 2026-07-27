#!/usr/bin/env bash
# SignLink Control Center — one-command launch.
# Activates the venv and starts the web control panel; the browser opens
# automatically and the live camera + mannequin is a separate desktop window.
set -e
cd "$(dirname "$0")"
source venv/bin/activate
exec python -m webapp "$@"
