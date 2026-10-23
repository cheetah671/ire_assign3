#!/usr/bin/env bash
# Builds report/report.pdf from report/report.html with headless Chrome (or Edge).
set -euo pipefail
cd "$(dirname "$0")"
for b in "${CHROME:-}" "/c/Program Files/Google/Chrome/Application/chrome.exe" \
         "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" google-chrome chromium; do
  if [[ -n "$b" ]] && command -v "$b" >/dev/null 2>&1 || [[ -x "$b" ]]; then BROWSER="$b"; break; fi
done
"$BROWSER" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$(pwd -W 2>/dev/null || pwd)/report.pdf" "file:///$(pwd -W 2>/dev/null || pwd)/report.html"
echo "wrote report/report.pdf"
