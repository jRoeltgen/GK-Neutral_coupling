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

Each launcher removes generated coupling state from `input_files` before
starting, including the evolving `fort.31`, EIRENE source outputs, and cached
boundary mappings. Static geometry inputs such as `fort.30/33/34/35` are
preserved. This makes reruns independent of whether the preceding job passed
or failed.

The launchers use `coupling_summed_temp.yaml` and
`coupling_multiple_temps.yaml` in `input_files`. Both files explicitly list and
describe every public coupling input, including default-valued settings, so
they can also be copied as templates. `${GENEX_PATH}` is expanded at load time;
other relative paths are interpreted from the YAML file's directory.

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

Each job prints its temporary run directory. Partition files are deleted after
a successful test. The generated `mesh.nc` (~10GB) is first used by the
standalone conservative-interpolation test and then moved to scratch as a
reusable artifact; its path is printed by the job. Set `GENEX_MESH_ARTIFACT`
before submission to choose an explicit artifact path. Re-run the lightweight
check without a job using:

```bash
GENEX_INTERPOLATION_MESH=/path/to/mesh.nc \
    python -m pytest -q ../test_real_mesh_interpolation.py
```

Failed-run output is preserved in scratch for diagnosis. The summed_temp=True
test may be machine dependent and increasing the number of timesteps in
`params_summed_temp.txt` may allow a previously failed run to pass.
