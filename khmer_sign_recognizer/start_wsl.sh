#!/bin/bash

echo "========================================"
echo " KHMER SIGN RECOGNIZER - WSL LAYER"
echo "========================================"
echo ""
echo "Starting processing layer..."
echo "Press Ctrl+C to quit"
echo ""

# Access D: drive from WSL (updated path after moving from E: drive)
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
# Option B: If using a separate WSL copy, uncomment below instead:
# cd ~/khmer_project
source venv_wsl/bin/activate
python src/main_wsl.py
