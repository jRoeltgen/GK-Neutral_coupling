# GENE-X integration tests

These optional Slurm tests run GENE-X together with the checked EIRENE
substitute. They are intended for occasional validation on an HPC system, not
for the normal unit-test suite. Running both tests
simultaneously may cause errors.

## Environment

Install this repository in the existing Tor-X environment:

```bash
source /path/to/torx/env/bin/activate
python -m pip install -e ".[genex]"
```

Set the GENE-X build directory before submitting:

```bash
export GENEX_BUILD_DIR=/path/to/genex/build
```

The directory must contain `bin/genex` and `toolchain.sh`.

The scripts use `$HOME/genex_all/torx/env` by default. Override it when the
Tor-X environment is elsewhere:

```bash
export TORX_ENV=/path/to/torx/env
```

Temporary output uses `NEUTRAL_COUPLING_SCRATCH`, then `PSCRATCH`, then
`SCRATCH`. Set the override only when those system defaults are unsuitable:

```bash
export NEUTRAL_COUPLING_SCRATCH=/path/to/shared/scratch
```

Also update the `#SBATCH` account and email directives in each submission
script for the target system.

## Running

Submit from the integration-test directory:

```bash
cd tests/genex/integration
sbatch submit_summed_temp.sh
sbatch submit_multiple_temps.sh
```

If submitting from somewhere else, set `INTEGRATION_DIR` to the absolute path
of `tests/genex/integration` before calling `sbatch`.

Each job prints its temporary run directory. Large GENE-X mesh and partition
files (~10GB) are deleted after a successful test; failed-run output is preserved in
scratch for diagnosis. The summed_temp=True test may be machine dependent and increasing the number of timesteps in `params_summed_temp.txt` may allow a previously failed run to pass.
