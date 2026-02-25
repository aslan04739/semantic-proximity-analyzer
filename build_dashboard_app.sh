#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building embedded dashboard macOS app..."

echo "Activating venv..."
source .venv_dashboard/bin/activate

rm -rf build dist

pyinstaller \
  --onedir \
  --windowed \
  --name "Semantic Analyzer Dashboard" \
  --icon "assets/icon_minimal.icns" \
  --osx-bundle-identifier=com.semanticanalyzer.dashboard \
  --add-data "templates:templates" \
  dashboard_embedded_app.py

echo "Installing to ~/Applications..."
rm -rf ~/Applications/Semantic\ Analyzer\ Dashboard.app
cp -r "dist/Semantic Analyzer Dashboard.app" ~/Applications/

echo "Removing quarantine flag..."
xattr -dr com.apple.quarantine ~/Applications/Semantic\ Analyzer\ Dashboard.app 2>/dev/null || true

echo "Done. Launch from Spotlight: Semantic Analyzer Dashboard"
