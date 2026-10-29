# GENE-X read profiling

These opt-in tools profile completed GENE-X diagnostics. They do not launch
GENE-X or EIRENE and never modify the diagnostic directory.

The default `read` scope covers loading, derived fields, toroidal averaging,
and materialization. The `interpolate` scope additionally loads static EIRENE
geometry and performs moment interpolation, but still does not execute EIRENE:

```bash
export EIRENE_PATH=/path/to/eirene_setup_files
./profiling/genex_read/run_profile.sh averaged interpolate
```

`averaged` is the production default. `full` eagerly loads the selected raw
timestep before deriving and averaging fields. `legacy` is available only in
this profiler and reproduces the former compute-and-discard pass.

Run on a Perlmutter CPU node:

```bash
export GENEX_PATH=/pscratch/sd/j/jonroelt/TCV-X21/genex_run_wo_neutrals
export ITERATIONS=3
./profiling/genex_read/run_profile.sh legacy
./profiling/genex_read/run_profile.sh averaged
./profiling/genex_read/run_profile.sh full
```

Each result directory contains JSON measurements, sampled RSS, `/usr/bin/time
-v` output, and, by default, cProfile output. For authoritative timing, set
`CPROFILE=0`; cProfile is intended for attribution:

```bash
CPROFILE=0 ./profiling/genex_read/run_profile.sh averaged
```

Compare reports with:

```bash
python profiling/genex_read/compare_profiles.py profile-results/*/results.json
```

First iterations can benefit from different filesystem cache states. Compare
multiple process-level runs and alternate mode order before drawing timing
conclusions.
