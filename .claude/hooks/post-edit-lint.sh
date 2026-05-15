#!/usr/bin/env bash
# Post-edit lint: runs ruff on Python files. Non-blocking.
set -uo pipefail
PAYLOAD=$(cat)
FILE_PATH=$(printf '%s' "$PAYLOAD" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null || echo "")
[ -z "$FILE_PATH" ] && exit 0
case "$FILE_PATH" in *.py) ;; *) exit 0 ;; esac
command -v ruff >/dev/null 2>&1 && ruff check "$FILE_PATH" 2>&1 >&2 || true
exit 0
