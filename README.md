# Neutral Coupling

This repository contains the `neutral_coupling` Python package for coupling
gyrokinetic plasma codes to the Monte Carlo neutral code EIRENE.

GENE-X users should first activate the environment created by Tor-X, then run:

```bash
python -m pip install -e ".[genex]"
```

Tor-X is an external prerequisite and is not installed or modified by this
package.

Gkeyll users can install independently in a Python 3.10 or newer environment (further details in README in `neutral_coupling\gkeyll_coupling`):

```bash
python -m pip install -e ".[gkeyll]"
```

These editable installs make `neutral_coupling` importable without modifying
`PYTHONPATH`.

Backend-independent EIRENE modules are in `neutral_coupling/common`.

## Boundary particle-flux conservation

The GENE-X/EIRENE driver conserves outgoing ion particle flux by default on
geometrically matched boundary-surface groups. Boundary components are derived
from active-cell connectivity rather than hard-coded single-null names, so
split SOLPS boundaries and differing component counts can be grouped. Use
`--boundary-flux-normalization off` for legacy behavior; accepted correction
factors default to 0.5--2 and can be changed with
`--boundary-flux-factor-min` and `--boundary-flux-factor-max`. Each iteration
archives `boundary_flux_normalization.json` with the pre/post totals and factors.
The geometry match itself is constructed once and saved as
`boundary_flux_mapping.json`. On restart, a changed `b2fgmtry` or GENE-X
`mesh.nc` is fatal. Changes to EIRENE `fort.33/34/35` produce a warning and the
mapping is reused: with fixed `b2fgmtry`, the triangular grid inside the plasma
domain is expected to be fixed and only external neutral-region triangles may
change.
The mapping file includes matched and excluded face counts plus face-level
geometry. If an EIRENE boundary cannot be covered, initialization fails after
writing `boundary_flux_mapping_failure.json` with both meshes' candidate
components for diagnosis.

The generalized component logic has synthetic tests representing multiple
topologies. A paired production-quality double-null GENE-X/EIRENE dataset is
still required for future end-to-end validation of four-target identification,
cut conventions, outward signs, and velocity-component selection. Until that
fixture is available, double-null support should be regarded as generalized by
construction rather than validated on production data.

Tests and their fixtures are in `tests`.

Backend implementations are in `neutral_coupling/genex_coupling` and
`neutral_coupling/gkeyll_coupling`.

Documentation relating filenames used by this repository for transferring Eirene sources to the source type is in `doc\fort_filenames.tex`

# Gkeyll coupling

The Gkeyll package contains `geometry`, `io`, and `examples` subpackages.

    1. gkeyllGeometry: Contains files used to generate mappings to interpolate data between Gkeyll and EIRENE
    2. gkeyllIO: Contains files used to pass plasma data from Gkeyll to EIRENE and neutral/plasma source data from EIRENE to Gkeyll
    3. gkeyllExamples: Contains some example uses of the files in gkeyllGeometry and gkeyllIO


The LLMs GPT and related Codex were used to help in creating the code used for the GENE-X coupling, primarily in the creation of unit tests.
