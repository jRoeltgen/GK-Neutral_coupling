#!/bin/bash -l
# Job Name
#SBATCH -J gx_neutral_integration
# Standard output and error
#SBATCH -o job.out
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --qos=regular
# Number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of openmp threads
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
##SBATCH --mail-type=ALL          # Uncomment, enter your email and remove
##SBATCH --mail-user=XXXXXXX # these comments to receive job notifications

set -uo pipefail

: "${GENEX_BUILD_DIR:?Set GENEX_BUILD_DIR to the GENE-X build directory}"
TORX_ENV="${TORX_ENV:-$HOME/genex_all/torx/env}"

module purge
source "$GENEX_BUILD_DIR/toolchain.sh"

if [[ -n "${INTEGRATION_DIR:-}" ]]; then
    SCRIPT_DIR=$(cd -- "$INTEGRATION_DIR" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/input_files" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
elif [[ -d "$(pwd)/input_files" ]]; then
    SCRIPT_DIR=$(pwd)
else
    echo "Could not find integration input_files directory." >&2
    echo "Submit from tests/genex/integration or set INTEGRATION_DIR." >&2
    exit 1
fi
cp "$SCRIPT_DIR/input_files/fort.31.template" "$SCRIPT_DIR/input_files/fort.31"
cd "$SCRIPT_DIR/input_files" || exit 1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KMP_AFFINITY=noverbose

if [[ -n "${NEUTRAL_COUPLING_SCRATCH:-}" ]]; then
    SCRATCH_ROOT="$NEUTRAL_COUPLING_SCRATCH"
elif [[ -n "${PSCRATCH:-}" ]]; then
    SCRATCH_ROOT="$PSCRATCH"
elif [[ -n "${SCRATCH:-}" ]]; then
    SCRATCH_ROOT="$SCRATCH"
else
    echo "No scratch directory configured." >&2
    exit 1
fi

RUN_DIR=$(mktemp -d \
    "${SCRATCH_ROOT%/}/neutral-coupling-${SLURM_JOB_ID:-manual}-XXXXXX")
GENEX_OUT_DIR="${RUN_DIR%/}/"
PARAM_FILE="params_multiple_temps.txt"
cp $SCRIPT_DIR/input_files/params* $RUN_DIR
mv $RUN_DIR/$PARAM_FILE "${RUN_DIR}/params_in.txt"
cp $SCRIPT_DIR/input_files/D3D_184833_t4.800.nc $RUN_DIR
rm -rf eirene_sources* input_sources*

echo "Outputting temporary GENE-X files to ${RUN_DIR}"
srun -Q --chdir="$SCRIPT_DIR/input_files" \
    "$GENEX_BUILD_DIR/bin/genex" -o "$GENEX_OUT_DIR" "$PARAM_FILE" \
    >"$SCRIPT_DIR/genex_output.log" 2>&1 &
GENEX_PID=$!

echo "GENEX PID: $GENEX_PID"
echo "$GENEX_PID" > genex.pid

if [[ ! -f "$TORX_ENV/bin/activate" ]]; then
    echo "Tor-X environment not found at $TORX_ENV" >&2
    exit 1
fi
source "$TORX_ENV/bin/activate"
echo "Python: $(command -v python) ($(python --version 2>&1))"
python -c "import neutral_coupling"

export INTEGRATION_DIR="$SCRIPT_DIR"
export GENEX_PATH="$RUN_DIR"
export EIRENE_PATH="$(pwd)"

if python -c "
import os
import sys
sys.path.insert(0, os.environ['INTEGRATION_DIR'])
from imitation_eirene_driver import test_full_coupling
test_full_coupling(
    './',
    False,
    temperature_mode='multiple_temperatures',
    collision_types=(
        'atom-plasma',
        'molecule-plasma',
        'testion-plasma',
    ),
    genex_path=os.environ['GENEX_PATH'],
    eirene_path=os.environ['EIRENE_PATH'],
)
"; then
    rm -rf -- "$RUN_DIR/mesh.nc" "$RUN_DIR"/part*
    cp -a "$RUN_DIR"/. .
else
    status=$?
    echo "Integration test failed; preserving output in $RUN_DIR" >&2
    exit "$status"
fi
