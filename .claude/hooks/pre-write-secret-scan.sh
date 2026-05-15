#!/usr/bin/env bash
# Pre-write secret scan: blocks Write/Edit if content contains secret-shaped strings.
set -uo pipefail
PAYLOAD=$(cat)
CONTENT=$(printf '%s' "$PAYLOAD" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ti = d.get("tool_input", {})
print(ti.get("content", "") + "\n" + ti.get("new_string", ""))
' 2>/dev/null || echo "")
[ -z "$CONTENT" ] && exit 0
PATTERNS=(
  'hf_[A-Za-z0-9]{30,}'
  'sk-ant-[A-Za-z0-9_-]{30,}'
  'sk-[A-Za-z0-9]{40,}'
  'AKIA[0-9A-Z]{16}'
  'ghp_[A-Za-z0-9]{30,}'
  'github_pat_[A-Za-z0-9_]{60,}'
  'xox[bpars]-[0-9A-Za-z-]{30,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'ya29\.[A-Za-z0-9_-]{40,}'
  'glpat-[A-Za-z0-9_-]{20,}'
)
for pat in "${PATTERNS[@]}"; do
  if printf '%s' "$CONTENT" | grep -qE -- "$pat"; then
    echo "BLOCKED by pre-write-secret-scan: secret-shaped pattern matched ($pat)" >&2
    exit 2
  fi
done
exit 0
