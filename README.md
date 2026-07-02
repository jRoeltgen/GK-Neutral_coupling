# Neutral Coupling

This repository contains the `neutral_coupling` Python package for coupling
gyrokinetic plasma codes to the Monte Carlo neutral code EIRENE.

GENE-X users should first activate the environment created by Tor-X, then run:

```bash
python -m pip install -e ".[genex]"
```

Tor-X is an external prerequisite and is not installed or modified by this
package.

Gkeyll users can install independently in a Python 3.10 or newer environment:

```bash
python -m pip install -e ".[gkeyll]"
```

These editable installs make `neutral_coupling` importable without modifying
`PYTHONPATH`.

Backend-independent EIRENE modules are in `neutral_coupling/common`.

Tests and their fixtures are in `tests`.

Backend implementations are in `neutral_coupling/genex_coupling` and
`neutral_coupling/gkeyll_coupling`.

Documentation with helpful details is in the doc folder

# Gkeyll coupling

The Gkeyll package contains `geometry`, `io`, and `examples` subpackages.

    1. gkeyllGeometry: Contains files used to generate mappings to interpolate data between Gkeyll and EIRENE
    2. gkeyllIO: Contains files used to pass plasma data from Gkeyll to EIRENE and neutral/plasma source data from EIRENE to Gkeyll
    3. gkeyllExamples: Contains some example uses of the files in gkeyllGeometry and gkeyllIO 
