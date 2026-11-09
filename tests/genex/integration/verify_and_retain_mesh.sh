#!/bin/bash
# Verify conservative interpolation and retain the large mesh outside the repo.
set -uo pipefail

RUN_DIR=$1
SCRIPT_DIR=$2
SCRATCH_ROOT=$3
MESH_PATH="${RUN_DIR%/}/mesh.nc"

if [[ ! -f "$MESH_PATH" ]]; then
    echo "GENE-X did not create $MESH_PATH" >&2
    exit 1
fi

GENEX_INTERPOLATION_MESH="$MESH_PATH" \
    python -m pytest -q "$SCRIPT_DIR/../test_real_mesh_interpolation.py"

if [[ -n "${GENEX_MESH_ARTIFACT:-}" ]]; then
    ARTIFACT_PATH=$GENEX_MESH_ARTIFACT
else
    ARTIFACT_PATH="${SCRATCH_ROOT%/}/genex-interpolation-mesh-${SLURM_JOB_ID:-manual}-$(basename "$RUN_DIR").nc"
fi
if [[ -e "$ARTIFACT_PATH" ]]; then
    echo "Refusing to overwrite existing mesh artifact: $ARTIFACT_PATH" >&2
    exit 1
fi
mv -- "$MESH_PATH" "$ARTIFACT_PATH"
echo "Retained interpolation mesh artifact: $ARTIFACT_PATH"
