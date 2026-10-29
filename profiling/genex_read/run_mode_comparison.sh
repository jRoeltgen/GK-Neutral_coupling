#!/usr/bin/env bash
set -euo pipefail

: "${GENEX_PATH:?Set GENEX_PATH to a completed GENE-X run directory}"
PROFILE_ROOT=${PROFILE_ROOT:-"$(pwd)/profile-results/comparison-$(date -u +%Y%m%dT%H%M%SZ)"}
export PROFILE_ROOT

reports=()
for mode in legacy averaged full; do
    result_dir=$("$(dirname "$0")/run_profile.sh" "$mode")
    reports+=("$result_dir/results.json")
done

python "$(dirname "$0")/compare_profiles.py" --require-equal "${reports[@]}"
