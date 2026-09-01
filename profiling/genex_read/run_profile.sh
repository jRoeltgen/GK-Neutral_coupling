#!/usr/bin/env bash
set -euo pipefail

: "${GENEX_PATH:?Set GENEX_PATH to a completed GENE-X run directory}"

MODE=${1:-averaged}
SCOPE=${2:-read}
ITERATIONS=${ITERATIONS:-1}
TIME_INDEX=${TIME_INDEX:--1}
PROFILE_ROOT=${PROFILE_ROOT:-"$(pwd)/profile-results"}
TORX_ENV=${TORX_ENV:-"${HOME}/genex_all/torx/env"}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="${PROFILE_ROOT%/}/${MODE}-${STAMP}"
mkdir -p "$OUT_DIR"

source "$TORX_ENV/bin/activate"
export MPLCONFIGDIR="$OUT_DIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

profile_args=()
if [[ "${CPROFILE:-1}" == "1" ]]; then
    profile_args=(--cprofile "$OUT_DIR/profile.pstats")
fi

eirene_args=()
if [[ "$SCOPE" == "interpolate" ]]; then
    : "${EIRENE_PATH:?Set EIRENE_PATH for interpolation profiling}"
    eirene_args=(--eirene-path "$EIRENE_PATH")
fi

/usr/bin/time -v -o "$OUT_DIR/time.txt" \
    python "$(dirname "$0")/profile_genex_read.py" \
        --genex-path "$GENEX_PATH" \
        --mode "$MODE" \
        --scope "$SCOPE" \
        --time-index "$TIME_INDEX" \
        --iterations "$ITERATIONS" \
        --output "$OUT_DIR/results.json" \
        "${profile_args[@]}" \
        "${eirene_args[@]}" \
        >"$OUT_DIR/stdout.log" 2>"$OUT_DIR/stderr.log"

echo "$OUT_DIR"
