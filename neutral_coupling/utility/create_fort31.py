"""Create a geometry-correct EIRENE ``fort.31`` without a B2 plasma state.

The layout and geometry conversion follow B2.5's ``copy_background`` and
``write_f31`` routines.  Plasma-state quantities are deliberately initialized
to benign placeholders; coupling code replaces the quantities it evolves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge

from neutral_coupling.common.b2_io import B2
from neutral_coupling.common.eirene_input_parser import EireneInputParser
from neutral_coupling.common.eirene_io import eirene


def _flux_masks(gmtry: dict, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Return valid left/poloidal and bottom/radial B2 face masks.

    This is the topology-only part of B2.5 ``copy_background``.  True entries
    receive a nonzero placeholder; unavailable faces and the unused final face
    planes remain zero.
    """
    poloidal = np.zeros((nx + 2, ny + 2), dtype=bool)
    radial = np.zeros_like(poloidal)

    # B2 indices are (-1:nx, -1:ny), while arrays read from b2fgmtry use
    # Python indices (0:nx+2, 0:ny+2).
    for iy in range(-1, ny + 1):
        jy = iy + 1
        for ix in range(0, nx + 1):
            poloidal[ix, jy] = (
                gmtry["rightix"][ix, jy] != nx + 1
                and gmtry["leftix"][ix + 1, jy] != -2
            )

    for iy in range(0, ny + 1):
        for ix in range(-1, nx + 1):
            radial[ix + 1, iy] = gmtry["bottomiy"][ix + 1, iy + 1] != -2

    return poloidal, radial


def build_fort31(
    input_dat: str | Path,
    b2fgmtry: str | Path,
    fort30: str | Path,
) -> eirene:
    """Build an in-memory ``fort.31`` from EIRENE and B2 geometry inputs."""
    input_dat = Path(input_dat)
    b2fgmtry = Path(b2fgmtry)
    fort30 = Path(fort30)

    parser = EireneInputParser(input_dat)
    parser.parse_species()
    species = parser.species["bulk_ions"]
    ns = len(species)
    if ns == 0:
        raise ValueError(f"No physical bulk-ion species found in {input_dat}")

    b2 = B2()
    b2.read_b2fgmtry(b2fgmtry)
    gmtry = b2.gmtry

    result = eirene()
    result.read_ft30(fort30)
    nx = result.plasma_gmtry["nx"]
    ny = result.plasma_gmtry["ny"]
    shape = (nx + 2, ny + 2)
    if gmtry["vol"].shape != shape:
        raise ValueError(
            "Geometry dimensions disagree: "
            f"fort.30 implies {shape}, b2fgmtry contains {gmtry['vol'].shape}"
        )

    species_shape = shape if ns == 1 else (*shape, ns)
    scalar_zero = np.zeros(shape)
    species_zero = np.zeros(species_shape)
    density = np.full(species_shape, 1.0e19)
    temperature = np.full(shape, elementary_charge)  # 1 eV in joules
    poloidal_mask, radial_mask = _flux_masks(gmtry, nx, ny)
    if ns == 1:
        fnax = poloidal_mask.astype(float)
        fnay = radial_mask.astype(float)
    else:
        fnax = np.repeat(poloidal_mask[:, :, None], ns, axis=2).astype(float)
        fnay = np.repeat(radial_mask[:, :, None], ns, axis=2).astype(float)

    bb = np.asarray(gmtry["bb"], dtype=float).copy()
    total_field = bb[:, :, 3]
    pitch = np.divide(
        np.abs(bb[:, :, 0]),
        np.abs(total_field),
        out=np.zeros(shape),
        where=total_field != 0.0,
    )

    result.fort31 = {
        "na": density,
        "up": species_zero.copy(),
        "vv": species_zero.copy(),
        "ww": species_zero.copy(),
        "te": temperature,
        "ti": temperature.copy(),
        "pr": temperature * (1.0e19 * (ns + 1)),
        "ua": species_zero.copy(),
        "pitch_angle": pitch,
        "fnax": fnax,
        "fnay": fnay,
        "fhix": scalar_zero.copy(),
        "fhiy": scalar_zero.copy(),
        "fhex": scalar_zero.copy(),
        "fhey": scalar_zero.copy(),
        "uadia": species_zero.copy(),
        "vadia": species_zero.copy(),
        "po": scalar_zero.copy(),
        "vol": np.asarray(gmtry["vol"], dtype=float).copy(),
        "bb": bb,
        "dummy3D": species_zero.copy(),
        "dummy2D": scalar_zero.copy(),
        "delta_sheathxb": scalar_zero.copy(),
        "delta_sheathyb": scalar_zero.copy(),
        # The evolving mean charge is not available from geometry-only inputs;
        # one is a safe bootstrap value for the usual singly charged fluids.
        "ion_charge": np.ones(species_shape),
    }
    return result


def create_fort31(
    input_dat: str | Path,
    b2fgmtry: str | Path,
    fort30: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Create and write a ``fort.31`` file, returning its path."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
    data = build_fort31(input_dat, b2fgmtry, fort30)
    data.write_ft31(output)
    return output


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="input.dat", help="EIRENE input.dat")
    parser.add_argument("--geometry", default="b2fgmtry", help="B2 b2fgmtry")
    parser.add_argument("--fort30", default="fort.30", help="EIRENE fort.30")
    parser.add_argument("-o", "--output", default="fort.31", help="output path")
    parser.add_argument("--force", action="store_true", help="overwrite output")
    args = parser.parse_args(argv)
    output = create_fort31(
        args.input, args.geometry, args.fort30, args.output,
        overwrite=args.force,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
