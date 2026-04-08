#!/usr/bin/env bash

# Local project runtime bootstrap.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-not_activated}"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
