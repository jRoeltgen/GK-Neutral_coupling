#!/usr/bin/env python3
"""Plot toroidally averaged GENE-X and EIRENE output on their native meshes.

Examples
--------
    python plot_genex_eirene.py /path/to/genex /path/to/eirene
    python plot_genex_eirene.py GENEX EIRENE --time 0.02 --species D

The default is the last complete GENE-X snapshot.  A requested GENE-X time is
matched to the closest available ``tau`` coordinate.  EIRENE's
``fort.4??`` files do not contain a time axis; point ``eirene_path`` at the
directory belonging to the desired coupling iteration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.tri import Triangulation
import numpy as np

from torx.specializations.genex import load_snaps_genex

from neutral_coupling.common.eirene_io import eirene
from neutral_coupling.common.eirene_input_parser import EireneInputParser
from neutral_coupling.common.triangle_mesh import triangle_mesh
from neutral_coupling.genex_coupling.genex_interface import (
    calculate_temperatures,
    get_genex_species,
    wait_for_genex_init,
)


# Convenient single-variable alternative to the CLI option.  None selects the
# latest snapshot; otherwise this is a GENE-X tau value, not an array index.
DEFAULT_GENEX_TIME = None
ANALYSIS_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _nearest_time_index(array, requested_time: float | None):
    """Return the index and value of the closest available GENE-X tau."""
    if "tau" not in array.dims or "tau" not in array.coords:
        raise ValueError("GENE-X diagnostic has no tau coordinate")
    tau = np.asarray(array.coords["tau"].values, dtype=float).reshape(-1)
    if tau.size == 0:
        raise ValueError("GENE-X diagnostic has an empty tau coordinate")
    if requested_time is None:
        index = tau.size - 1
    else:
        if not np.isfinite(requested_time):
            raise ValueError("Requested GENE-X time must be finite")
        finite = np.isfinite(tau)
        if not np.any(finite):
            raise ValueError("GENE-X diagnostic has no finite tau values")
        candidates = np.flatnonzero(finite)
        index = int(candidates[np.argmin(np.abs(tau[finite] - requested_time))])
    return index, float(tau[index])


def _toroidal_average(array):
    """Average a diagnostic over its toroidal dimension, when present."""
    phi_dims = [dim for dim in array.dims if dim == "phi" or dim.endswith("_phi")]
    return array.mean(dim=phi_dims[0]) if phi_dims else array


def _select_snapshot(array, time_index: int, average_phi: bool = True):
    """Select tau and, by default, take the toroidal average."""
    if "tau" in array.dims:
        array = array.isel(tau=time_index)
    if average_phi:
        array = _toroidal_average(array)
    return array.load()


def _magnitude(quantity, unit: str) -> float:
    """Return a pint-like normalization in the requested unit."""
    return float(quantity.to(unit).magnitude)


def _genex_matrix(grid, array, scale: float):
    """Use GENE-X's native vector-to-poloidal-matrix topology."""
    matrix = grid.vector_to_matrix(array) if array.ndim == 1 else array
    return np.asarray(matrix) * scale


def _genex_in_target_matrix(genex_path: Path, grid):
    """Return GENE-X's static target mask in native matrix topology."""
    target = load_snaps_genex(genex_path, None, "in_target")
    if "tau" in target.dims:
        target = target.isel(tau=-1)
    phi_dims = [dim for dim in target.dims
                if dim == "phi" or dim.endswith("_phi")]
    if phi_dims:
        # in_target is static and toroidally invariant. Avoid averaging a mask.
        target = target.isel({phi_dims[0]: 0})
    target = target.load()
    matrix = grid.vector_to_matrix(target) if target.ndim == 1 else target
    values = np.asarray(matrix)
    # vector_to_matrix represents cells absent from the point diagnostic as
    # NaN. They are not target cells and are excluded separately by finite-data
    # checks, so avoid bool(NaN) incorrectly marking them as in_target.
    return np.isfinite(values) & (values != 0)


def load_genex_moments(genex_path: Path, requested_time: float | None,
                       ):
    """Load physical n, T and u_parallel for every GENE-X species."""
    grid, _equi, params, norm, r, z, compute, _in_target = (
        wait_for_genex_init(genex_path)
    )
    species = get_genex_species(params)
    raw_species_names = params["params_species"]["names"]
    parameter_names = {str(raw).strip(): raw for raw in raw_species_names
                       if str(raw).strip()}
    result = {}
    in_target = _genex_in_target_matrix(genex_path, grid)
    time_index = selected_time = None

    for item in species:
        name = item.name
        # Diagnostic paths use normalized names (e.g. "ions"), whereas Torx's
        # temperature routine calls list.index() on the fixed-width names read
        # from params_out.txt (e.g. "ions   ").  It therefore needs the exact
        # original value rather than the stripped display/diagnostic name.
        parameter_name = parameter_names[name]
        n_all = load_snaps_genex(genex_path, name, "n")
        if time_index is None:
            time_index, selected_time = _nearest_time_index(
                n_all, requested_time
            )
        n = _select_snapshot(n_all, time_index, average_phi=False)
        upar = _select_snapshot(
            load_snaps_genex(genex_path, name, "u_par"), time_index,
            average_phi=False,
        )
        epar = _select_snapshot(
            load_snaps_genex(genex_path, name, "E_par"), time_index,
            average_phi=False,
        )
        eperp = _select_snapshot(
            load_snaps_genex(genex_path, name, "E_perp"), time_index,
            average_phi=False,
        )
        n.attrs["norm"] = norm.n0
        upar.attrs["norm"] = norm.c_s0
        epar.attrs["norm"] = norm.Te0 * norm.n0
        eperp.attrs["norm"] = norm.Te0 * norm.n0
        temperature = calculate_temperatures(
            params, norm, parameter_name, n, upar, epar, eperp
        )[0]
        # Match the coupling: derive temperature at each toroidal plane, then
        # average the completed moments sent to EIRENE.
        n = _toroidal_average(n).load()
        upar = _toroidal_average(upar).load()
        temperature = _toroidal_average(temperature).load()

        result[name] = {
            "density": _genex_matrix(grid, n, _magnitude(norm.n0, "1/m^3")),
            "temperature": _genex_matrix(
                grid, temperature, _magnitude(temperature.attrs["norm"], "eV")
            ),
            "upar": _genex_matrix(grid, upar, _magnitude(norm.c_s0, "m/s")),
        }
        for field in result[name].values():
            field[in_target] = np.nan

    r0_metres = _magnitude(norm.R0, "m")
    r_matrix = np.asarray(grid.r_s) * r0_metres
    z_matrix = np.asarray(grid.z_s) * r0_metres
    return r_matrix, z_matrix, None, result, time_index, selected_time


def _finite_vectors(r, z, values, mask=None):
    r, z, values = np.ravel(r), np.ravel(z), np.ravel(values)
    if r.size != z.size:
        raise ValueError(
            f"GENE-X R and Z coordinate sizes differ: {r.size} versus {z.size}"
        )

    if mask is not None:
        mask = np.ravel(mask).astype(bool)
        if mask.size != r.size:
            raise ValueError(
                f"GENE-X compute-mask size {mask.size} does not match the "
                f"coordinate size {r.size}"
            )

        if values.size == np.count_nonzero(mask):
            # Recent moment diagnostics omit ghost/filler cells, while mesh.nc
            # retains them. Compress the coordinates to the diagnostic layout.
            r = r[mask]
            z = z[mask]
            mask = None
        elif values.size != r.size:
            raise ValueError(
                f"GENE-X field has {values.size} values, but the mesh has "
                f"{r.size} total and {np.count_nonzero(mask)} compute cells"
            )
    elif values.size != r.size:
        raise ValueError(
            f"GENE-X field size {values.size} does not match coordinate size {r.size}"
        )

    valid = np.isfinite(r) & np.isfinite(z) & np.isfinite(values)
    if mask is not None:
        valid &= mask
    return r[valid], z[valid], values[valid]


def _norm(values, signed=False):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None
    if signed:
        limit = np.nanmax(np.abs(finite))
        if limit == 0:
            return None
        return SymLogNorm(linthresh=max(limit * 1e-4, np.finfo(float).tiny),
                          vmin=-limit, vmax=limit)
    positive = finite[finite > 0]
    if positive.size and np.nanmax(positive) / np.nanmin(positive) > 100:
        return LogNorm(np.nanmin(positive), np.nanmax(positive))
    return None


def _point_plot(ax, r, z, values, title, label, *, mask=None, signed=False):
    r = np.asarray(r)
    z = np.asarray(z)
    values = np.asarray(values)
    coordinates_match = (
        (r.ndim == 1 and z.ndim == 1
         and values.shape == (z.size, r.size))
        or (r.shape == values.shape and z.shape == values.shape)
    )
    if not coordinates_match:
        raise ValueError(
            f"Unsupported GENE-X coordinate layout: R={r.shape}, "
            f"Z={z.shape}, field={values.shape}; expected 1-D axes with "
            "field shape (len(Z), len(R)) or matching 2-D coordinate grids"
        )
    artist = ax.pcolormesh(r, z, values, shading="auto",
                           cmap="coolwarm" if signed else "viridis",
                           norm=_norm(values, signed))
    ax.set(title=title, xlabel="R [m]", ylabel="Z [m]")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(artist, ax=ax, label=label)


def plot_genex(r, z, compute, moments, output: Path | None, only_species: list[str] | None):
    names = only_species or list(moments)
    missing = set(names) - set(moments)
    if missing:
        raise ValueError(f"GENE-X species not found: {sorted(missing)}; available: {list(moments)}")
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 4.5 * len(names)), squeeze=False,
                             constrained_layout=True)
    fields = (("density", "n [m$^{-3}$]", False),
              ("temperature", "T [eV]", False),
              ("upar", "$u_\\parallel$ [m/s]", True))
    for row, name in enumerate(names):
        for col, (field, label, signed) in enumerate(fields):
            _point_plot(axes[row, col], r, z, moments[name][field],
                        f"GENE-X {name}: toroidal-average {field}", label,
                        mask=compute, signed=signed)
    if output is not None:
        fig.savefig(output, dpi=180)
        plt.close(fig)


def _triangle_plot(ax, mesh, values, title, label, *, signed=False):
    values = np.asarray(values).reshape(-1)
    triangulation = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cells - 1)
    artist = ax.tripcolor(triangulation, facecolors=values, shading="flat",
                          cmap="coolwarm" if signed else "viridis",
                          norm=_norm(values, signed))
    ax.set(title=title, xlabel="R [m]", ylabel="Z [m]")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(artist, ax=ax, label=label)


def _leaf_arrays(tree: Mapping, prefix=()):
    for key, value in tree.items():
        if isinstance(value, Mapping):
            yield from _leaf_arrays(value, prefix + (str(key),))
        elif isinstance(value, np.ndarray):
            yield prefix + (str(key),), value


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_")
    return cleaned or "unnamed"


def _source_groups(source_tree: Mapping, ntriangles: int,
                   sum_collisions: bool):
    """Return plot entries grouped into one figure per collision category."""
    groups = {}
    for path, values in _leaf_arrays(source_tree):
        if values.size != ntriangles:
            continue
        if sum_collisions:
            # Collision-summed layout is moment/species/stratum. Only the
            # aggregate SUM stratum is useful in the summary figures.
            if len(path) < 3 or path[-1] != "SUM":
                continue
            group = path[0]
            panel_path = path[1:-1]
        else:
            # Full source layout is moment/collision/species/stratum.
            group = path[1] if len(path) > 1 else "unclassified"
            panel_path = path[:1] + path[2:]
        groups.setdefault(group, []).append((" / ".join(panel_path), values))
    return groups


SOURCE_UNITS = {
    "particle": r"particle source [m$^{-3}$ s$^{-1}$]",
    "momentum": r"momentum source [N m$^{-3}$]",
    "energy": r"energy source [W m$^{-3}$]",
}


def _plot_source_group(mesh, group: str, entries, output_dir: Path | None,
                       pdf=None):
    ncols = min(3, len(entries))
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows),
        constrained_layout=True, squeeze=False,
    )
    fig.suptitle(f"EIRENE sources: {group}")
    for ax, (title, values) in zip(axes.flat, entries):
        finite = values[np.isfinite(values)]
        signed = bool(finite.size and np.nanmin(finite) < 0 < np.nanmax(finite))
        # In collision-summed mode the group is the moment. In detailed mode
        # the first title component is the moment and the group is collision.
        moment = group if group in SOURCE_UNITS else title.split(" / ", 1)[0]
        _triangle_plot(
            ax, mesh, values, title, SOURCE_UNITS.get(moment, "source [SI]"),
            signed=signed,
        )
    for ax in axes.flat[len(entries):]:
        ax.set_visible(False)
    if output_dir is not None:
        if pdf is not None:
            pdf.savefig(fig)
        fig.savefig(output_dir / f"eirene_sources_{_safe_filename(group)}.png", dpi=180)
        plt.close(fig)


def _eirene_output_path(eirene_path: Path, file_pattern: str,
                        iteration: int | None) -> tuple[Path, int]:
    """Resolve FILE_PATTERN_NNNNNN, choosing the latest iteration by default."""
    prefix = file_pattern.rstrip("_")
    if iteration is not None:
        output_path = eirene_path / f"{prefix}_{iteration:06d}"
        if not output_path.is_dir():
            raise FileNotFoundError(f"EIRENE output directory not found: {output_path}")
        return output_path, iteration

    expression = re.compile(rf"^{re.escape(prefix)}_(\d{{6}})$")
    candidates = []
    for path in eirene_path.glob(f"{prefix}_??????"):
        match = expression.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No EIRENE output directories matching {prefix}_NNNNNN under {eirene_path}"
        )
    return max(candidates, key=lambda item: item[0])[1], max(candidates)[0]


def plot_eirene(eirene_path: Path, output_dir: Path | None,
                source_pattern: str, file_pattern: str,
                iteration: int | None, *, include_molecules: bool = False,
                include_test_ions: bool = False,
                sum_collisions: bool = True):
    archive_path, selected_iteration = _eirene_output_path(
        eirene_path, file_pattern, iteration
    )
    data = eirene()
    data.triangle_mesh = triangle_mesh(eirene_path)
    data.read_ft44(archive_path / "fort.44")
    data.read_ft46(archive_path / "fort.46")
    mesh = data.triangle_mesh
    input_data = EireneInputParser(eirene_path / "input.dat")
    input_schema = input_data.parse_all()
    data.load_extra_forts(
        archive_path,
        extension=source_pattern,
        requested_strata=input_schema["requested_strata"],
        expected_species=input_schema["species"],
        vol_rec_mapping=input_schema["volume_recombination"],
        convert_units=True,
    )

    # fort.46 gives neutral moments directly on the triangle mesh.  Temperature
    # is mean kinetic energy per particle in eV; speed is plotted because EIRENE
    # does not define a GENE-X-like magnetic-field-parallel neutral velocity.
    blocks = [("a", "atoms", "atoms")]
    if include_molecules:
        blocks.append(("m", "molecules", "molecules"))
    if include_test_ions:
        blocks.append(("i", "test ions", "test_ions"))
    pages = []
    for suffix, class_label, class_key in blocks:
        density = data.fort46[f"pden{suffix}"]
        labels = [x.strip() for x in data.fort46[{"a": "atom labels", "m": "molecule labels", "i": "ion labels"}[suffix]]]
        for index, name in enumerate(labels):
            n = density[:, index]
            energy = data.fort46[f"eden{suffix}"][:, index]
            temp = np.divide((2.0 / 3.0) * energy, n,
                             out=np.full_like(energy, np.nan), where=n > 0) / 1.602176634e-19
            momentum = np.sqrt(sum(data.fort46[f"v{axis}den{suffix}"][:, index] ** 2
                                   for axis in "xyz"))
            mass = input_data.masses[class_key][index]
            speed = np.divide(momentum, mass * n, out=np.full_like(momentum, np.nan), where=n > 0)
            pages.append((f"{class_label} {name}", n, temp, speed))

    if pages and output_dir is not None:
        with PdfPages(output_dir / "eirene_neutral_moments.pdf") as pdf:
            for title, density, temp, speed in pages:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
                _triangle_plot(axes[0], mesh, density, f"EIRENE {title}: density", "n [m$^{-3}$]")
                _triangle_plot(axes[1], mesh, temp, f"EIRENE {title}: temperature", "T [eV]")
                _triangle_plot(axes[2], mesh, speed, f"EIRENE {title}: speed", "|u| [m/s]")
                pdf.savefig(fig)
                fig.savefig(output_dir / f"eirene_neutral_moments_{_safe_filename(title)}.png",
                            dpi=180)
                plt.close(fig)
    elif pages:
        for title, density, temp, speed in pages:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            _triangle_plot(axes[0], mesh, density, f"EIRENE {title}: density", "n [m$^{-3}$]")
            _triangle_plot(axes[1], mesh, temp, f"EIRENE {title}: temperature", "T [eV]")
            _triangle_plot(axes[2], mesh, speed, f"EIRENE {title}: speed", "|u| [m/s]")

    source_tree = data.sources if sum_collisions else data.full_source_in_SI
    source_groups = _source_groups(
        source_tree, mesh.cells.shape[0], sum_collisions
    )
    source_count = sum(len(entries) for entries in source_groups.values())
    if output_dir is not None:
        with PdfPages(output_dir / "eirene_sources.pdf") as pdf:
            for group, entries in source_groups.items():
                _plot_source_group(mesh, group, entries, output_dir, pdf)
    else:
        for group, entries in source_groups.items():
            _plot_source_group(mesh, group, entries, None)
    return (len(pages), source_count, len(source_groups), archive_path,
            selected_iteration)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("genex_path", type=Path, help="GENE-X run directory containing mesh.nc/mom_2d.nc")
    parser.add_argument("eirene_path", type=Path, help="EIRENE directory containing mesh and fort.4* files")
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=ANALYSIS_OUTPUT_DIR / "plots",
    )
    parser.add_argument("-t", "--time", type=float, default=DEFAULT_GENEX_TIME,
                        help="GENE-X tau; closest available time is selected "
                             "(default: latest snapshot)")
    parser.add_argument("--species", action="append",
                        help="GENE-X species to plot; repeat option for several (default: all)")
    parser.add_argument("--source-pattern", default="???",
                        help="three-character source extension glob (default: ???, e.g. fort.100-fort.345)")
    parser.add_argument("--eirene-file-pattern", default="eirene_sources",
                        help="archived output directory prefix (default: eirene_sources)")
    parser.add_argument("--eirene-iteration", type=int,
                        help="six-digit EIRENE iteration index (default: latest)")
    parser.add_argument("--eirene-molecules", action="store_true",
                        help="include EIRENE molecule moments (default: off)")
    parser.add_argument("--eirene-test-ions", action="store_true",
                        help="include EIRENE test-ion moments (default: off)")
    parser.add_argument(
        "--sum-collisions", action=argparse.BooleanOptionalAction, default=True,
        help="sum sources over collision types (default: on; use "
             "--no-sum-collisions for one figure per collision type)",
    )
    parser.add_argument("--show", action="store_true",
                        help="display plots interactively instead of writing files")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    output_dir = None if args.show else args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    r, z, compute, moments, time_index, selected_time = load_genex_moments(
        args.genex_path, args.time
    )
    requested = "latest" if args.time is None else f"{args.time:.16g}"
    print(f"GENE-X requested tau={requested}; selected time_index={time_index}, "
          f"tau={selected_time:.16g}")
    genex_output = None if output_dir is None else output_dir / "genex_moments.png"
    plot_genex(r, z, compute, moments, genex_output, args.species)
    neutral_count, source_count, source_figure_count, archive_path, selected_iteration = plot_eirene(
        args.eirene_path, output_dir, args.source_pattern,
        args.eirene_file_pattern, args.eirene_iteration,
        include_molecules=args.eirene_molecules,
        include_test_ions=args.eirene_test_ions,
        sum_collisions=args.sum_collisions,
    )
    print(f"Using EIRENE iteration {selected_iteration:06d}: {archive_path}")
    if args.show:
        print(f"Displaying GENE-X, {neutral_count} EIRENE neutral, and "
              f"{source_count} EIRENE source fields in "
              f"{source_figure_count} grouped figure(s)")
        print("Close all plot windows to exit.")
        try:
            plt.show(block=True)
        finally:
            # Explicit cleanup avoids GUI backends retaining managers/event
            # loops after the last visible window has been closed.
            plt.close("all")
    else:
        print(f"Wrote plots to {output_dir}")
        print(f"  GENE-X: {genex_output.name}")
        print(f"  EIRENE neutrals: PDF plus {neutral_count} PNG(s)")
        print(f"  EIRENE sources: {source_count} fields in "
              f"{source_figure_count} grouped PNG(s), plus PDF")


if __name__ == "__main__":
    main()
