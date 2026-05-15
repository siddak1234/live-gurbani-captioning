#!/usr/bin/env bash
# Pre-bash guard: blocks destructive commands.
# Exit 0 = allow; Exit 2 = block.
set -uo pipefail
PAYLOAD=$(cat)
CMD=$(printf '%s' "$PAYLOAD" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))' 2>/dev/null || echo "")
[ -z "$CMD" ] && exit 0
BLOCKED=(
  'rm[[:space:]]+-rf?[[:space:]]+/'
  'rm[[:space:]]+-rf?[[:space:]]+~'
  'rm[[:space:]]+-rf?[[:space:]]+\$HOME'
  'rm[[:space:]]+-rf?[[:space:]]+\*'
  'git[[:space:]]+push[[:space:]]+--force'
  'git[[:space:]]+push[[:space:]]+-f[[:space:]]'
  'git[[:space:]]+reset[[:space:]]+--hard[[:space:]]+origin'
  'dd[[:space:]]+if='
  'mkfs'
  '> /dev/sd'
  ':\(\)\{'
  'curl.*\|[[:space:]]*sh'
  'curl.*\|[[:space:]]*bash'
  'wget.*\|[[:space:]]*sh'
  'wget.*\|[[:space:]]*bash'
  'chmod[[:space:]]+-R[[:space:]]+777[[:space:]]+/'
  'sudo[[:space:]]+rm'
)
for pat in "${BLOCKED[@]}"; do
  if printf '%s' "$CMD" | grep -qE -- "$pat"; then
    echo "BLOCKED by pre-bash-guard: pattern '$pat' matched: $CMD" >&2
    exit 2
  fi
done
exit 0
