#!/usr/bin/env python3
"""Compare EIRENE and GENE-X particle flow through plasma-grid boundaries.

The default snapshot and EIRENE archive are the latest available.  Reported
rates are toroidally integrated using axisymmetric face areas ``2*pi*R*dl``.
GENE-X additionally produces edge line-outs of density, parallel velocity and
poloidal velocity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree
from torx.specializations.genex import load_snaps_genex

from neutral_coupling.common.b2_io import B2
from neutral_coupling.common.eirene_io import eirene
from neutral_coupling.genex_coupling.genex_interface import (
    get_genex_species,
    load_latest_genex_fields,
    wait_for_genex_init,
)
from neutral_coupling.genex_coupling.genex_eirene_coupling import (
    normalize_genex_params,
)
from plot_genex_eirene import (_eirene_output_path, _magnitude,
                               _genex_in_target_matrix, _nearest_time_index,
                               _select_snapshot)
from plot_genex_eirene import _norm

ANALYSIS_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "boundary"


def _species_index(values, index):
    values = np.asarray(values)
    return values[..., index] if values.ndim == 3 else values


def _line_area(r, z):
    """Axisymmetric face areas represented by points along a boundary."""
    r, z = np.asarray(r), np.asarray(z)
    ds = np.zeros(r.size)
    if r.size > 1:
        segments = np.hypot(np.diff(r), np.diff(z))
        ds[0] = segments[0] / 2
        ds[-1] = segments[-1] / 2
        if r.size > 2:
            ds[1:-1] = (segments[:-1] + segments[1:]) / 2
    return 2 * np.pi * r * ds


def _integral(values, area):
    return float(np.nansum(np.asarray(values) * np.asarray(area)))


def _eirene_surfaces(geom, active_x_rows, active_x_cols,
                     active_y_rows, active_y_cols):
    """Define the five physical boundary surfaces from SOLPS cut topology."""
    first_x, last_x = int(active_x_rows[0]), int(active_x_rows[-1])
    first_y, last_y = int(active_y_cols[0]), int(active_y_cols[-1])
    first_row, last_row = int(active_y_rows[0]), int(active_y_rows[-1])
    left_cut = int(np.ravel(geom["leftcut"])[0])
    right_cut = int(np.ravel(geom["rightcut"])[0])
    return {
        "inner target": {
            "direction": "x", "sign": -1.0,
            "segments": [(first_x, active_x_cols)],
        },
        "outer target": {
            "direction": "x", "sign": 1.0,
            "segments": [(last_x, active_x_cols)],
        },
        "private region": {
            "direction": "y", "sign": -1.0,
            "segments": [
                (np.arange(first_row, left_cut + 1), first_y),
                (np.arange(right_cut + 1, last_row + 1), first_y),
            ],
        },
        "core": {
            "direction": "y", "sign": -1.0,
            "segments": [(np.arange(left_cut + 1, right_cut + 1), first_y)],
        },
        "outer SOL": {
            "direction": "y", "sign": 1.0,
            "segments": [(active_y_rows, last_y)],
        },
    }


def load_eirene_boundary(base: Path, file_pattern: str, iteration: int | None,
                         species_index: int, template_path: Path | None = None):
    archive, selected = _eirene_output_path(base, file_pattern, iteration)
    geom = B2(base).gmtry
    nx, ny = geom["vol"].shape
    data = eirene()
    data.read_ft31(archive / "fort.31", nx, ny, max(species_index + 1, 1))

    if template_path is None:
        colocated_template = base / "fort.31.template"
        template_path = (
            colocated_template if colocated_template.is_file()
            else base / "make_fort31" / "fort.31"
        )
    template = eirene()
    template.read_ft31(template_path, nx, ny, max(species_index + 1, 1))
    fnax_mask = _species_index(template.fort31["fnax"], species_index) != 0
    fnay_mask = _species_index(template.fort31["fnay"], species_index) != 0

    active_x_rows = np.flatnonzero(fnax_mask.any(axis=1))
    active_x_cols = np.flatnonzero(fnax_mask.any(axis=0))
    active_y_rows = np.flatnonzero(fnay_mask.any(axis=1))
    active_y_cols = np.flatnonzero(fnay_mask.any(axis=0))
    if not all(values.size for values in
               (active_x_rows, active_x_cols, active_y_rows, active_y_cols)):
        raise ValueError(f"Template has no active fnax/fnay region: {template_path}")

    surfaces = _eirene_surfaces(
        geom, active_x_rows, active_x_cols, active_y_rows, active_y_cols
    )
    output = {}
    for label, surface in surfaces.items():
        direction, outward_sign = surface["direction"], surface["sign"]
        stored_total = reconstructed_total = 0.0
        physical_segments = []
        for ix, iy in surface["segments"]:
            r = np.mean(geom["crx"][ix, iy], axis=-1)
            z = np.mean(geom["cry"][ix, iy], axis=-1)
            area = _line_area(r, z)
            stored = _species_index(
                data.fort31["fnax" if direction == "x" else "fnay"],
                species_index,
            )[ix, iy]
            density = _species_index(data.fort31["na"], species_index)[ix, iy]
            velocity = _species_index(
                data.fort31["up" if direction == "x" else "vv"], species_index
            )[ix, iy]
            stored_total += outward_sign * _integral(stored, area)
            reconstructed_total += outward_sign * _integral(
                density * velocity, area
            )
            physical_segments.append((np.asarray(r), np.asarray(z)))
        surface["physical_segments"] = physical_segments
        surface["pitch_segments"] = [
            np.asarray(data.fort31["bb"][ix, iy, 0])
            for ix, iy in surface["segments"]
        ]
        output[label] = {
            "stored": stored_total,
            "reconstructed": reconstructed_total,
        }
    archived_fnax = _species_index(data.fort31["fnax"], species_index)
    archived_fnay = _species_index(data.fort31["fnay"], species_index)
    mask_report = {
        "template": template_path,
        "fnax_active_bounds": (active_x_rows[0], active_x_rows[-1],
                               active_x_cols[0], active_x_cols[-1]),
        "fnay_active_bounds": (active_y_rows[0], active_y_rows[-1],
                               active_y_cols[0], active_y_cols[-1]),
        "archive_fnax_nonzero": int(np.count_nonzero(archived_fnax)),
        "archive_fnay_nonzero": int(np.count_nonzero(archived_fnay)),
        "template_fnax_nonzero": int(np.count_nonzero(fnax_mask)),
        "template_fnay_nonzero": int(np.count_nonzero(fnay_mask)),
    }
    return output, surfaces, archive, selected, mask_report


def _matrix(grid, array):
    return np.asarray(grid.vector_to_matrix(array) if array.ndim == 1 else array)


def load_genex_boundary(path: Path, requested_time: float | None,
                        requested_species: str | None, surfaces):
    grid, equi, params, norm, *_ = wait_for_genex_init(path)
    params = normalize_genex_params(params)
    genex_species = get_genex_species(params)
    available = [item.name for item in genex_species if not item.is_electron]
    species = requested_species or available[0]
    if species not in available:
        raise ValueError(f"Ion species {species!r} not found; available: {available}")

    n_all = load_snaps_genex(path, species, "n")
    time_index, selected_time = _nearest_time_index(n_all, requested_time)
    # Use the coupling's own field construction so u_rad contains the same
    # toroidally averaged ExB + diamagnetic radial drift sent to EIRENE.
    fields, loaded_time = load_latest_genex_fields(
        path, genex_species, grid, equi, params, norm, time_index,
        timeout=30, read_mode="averaged",
    )
    n = fields["n"][species]
    upar = fields["u_par"][species]
    urad = fields["u_rad"][species]
    selected_time = float(loaded_time)
    n_values = _matrix(grid, n) * _magnitude(norm.n0, "1/m^3")
    upar_values = _matrix(grid, upar) * _magnitude(norm.c_s0, "m/s")
    urad_values = _matrix(grid, urad) * _magnitude(norm.c_s0, "m/s")

    r_axis = np.asarray(grid.r_s) * _magnitude(norm.R0, "m")
    z_axis = np.asarray(grid.z_s) * _magnitude(norm.R0, "m")
    rr, zz = np.meshgrid(r_axis, z_axis)
    in_target = _genex_in_target_matrix(path, grid)
    valid = (np.isfinite(n_values) & np.isfinite(upar_values)
             & np.isfinite(urad_values) & ~in_target)
    valid_row, valid_col = np.where(valid)
    tree = cKDTree(np.column_stack((rr[valid], zz[valid])))
    output, lineouts = {}, {}
    for label, surface in surfaces.items():
        rate = 0.0
        pieces = []
        for segment_index, (eirene_r, eirene_z) in enumerate(
            surface["physical_segments"]
        ):
            distance_to_cell, nearest = tree.query(
                np.column_stack((eirene_r, eirene_z))
            )
            # Preserve EIRENE ordering, but remove repeated GENE-X cells.
            keep = np.r_[True, np.diff(nearest) != 0]
            nearest = nearest[keep]
            row, col = valid_row[nearest], valid_col[nearest]
            r, z = rr[row, col], zz[row, col]
            distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
            area = _line_area(r, z)
            mapped_upol = (upar_values[row, col]
                           * surface["pitch_segments"][segment_index][keep])
            flux_velocity = (mapped_upol if surface["direction"] == "x"
                             else urad_values[row, col])
            rate += surface["sign"] * _integral(
                n_values[row, col] * flux_velocity, area
            )
            pieces.append((distance, n_values[row, col], upar_values[row, col],
                           mapped_upol, r, z,
                           float(np.max(distance_to_cell))))
        # NaN separators prevent disconnected private-region legs being joined.
        joined = tuple(np.concatenate([piece[i] if j == len(pieces) - 1
                                       else np.r_[piece[i], np.nan]
                                       for j, piece in enumerate(pieces)])
                       for i in range(6))
        output[label] = rate
        lineouts[label] = joined + (max(piece[6] for piece in pieces),)
    context = {
        "grid": grid,
        "rr": rr,
        "zz": zz,
        "valid_row": valid_row,
        "valid_col": valid_col,
        "tree": tree,
        "density": n_values,
        "upar": upar_values,
        "urad": urad_values,
    }
    return output, lineouts, species, time_index, selected_time, context


def plot_surface_map(lineouts, surfaces, output: Path | None):
    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    for label, values in lineouts.items():
        ax.plot(values[4], values[5], "o-", ms=2.5, lw=1, label=f"GENE-X {label}")
        for r, z in surfaces[label]["physical_segments"]:
            ax.plot(r, z, color=ax.lines[-1].get_color(), lw=3, alpha=0.3)
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="Matched EIRENE/GENE-X surfaces")
    ax.set_aspect("equal")
    ax.legend(fontsize="small", ncols=2)
    if output:
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return fig


def target_neighborhoods(genex_path: Path, eirene_path: Path, archive: Path,
                         time_index: int, species: str,
                         species_index: int, layers: int = 3):
    """Extract matching EIRENE/GENE-X rows at and behind both targets."""
    geom = B2(eirene_path).gmtry
    nx, ny = geom["vol"].shape
    edata = eirene()
    edata.read_ft31(archive / "fort.31", nx, ny, max(species_index + 1, 1))

    grid, equi, _params, norm, *_ = wait_for_genex_init(genex_path)
    n = _select_snapshot(
        load_snaps_genex(genex_path, species, "n"), time_index
    )
    upar = _select_snapshot(
        load_snaps_genex(genex_path, species, "u_par"), time_index
    )
    n_matrix = _matrix(grid, n) * _magnitude(norm.n0, "1/m^3")
    upar_matrix = _matrix(grid, upar) * _magnitude(norm.c_s0, "m/s")
    r_axis = np.asarray(grid.r_s) * _magnitude(norm.R0, "m")
    z_axis = np.asarray(grid.z_s) * _magnitude(norm.R0, "m")
    rr, zz = np.meshgrid(r_axis, z_axis)
    in_target = _genex_in_target_matrix(genex_path, grid)
    valid = np.isfinite(n_matrix) & np.isfinite(upar_matrix) & ~in_target
    vr, vc = np.where(valid)
    tree = cKDTree(np.column_stack((rr[valid], zz[valid])))

    # ix increases from inner target into the plasma and decreases from the
    # outer target into the plasma. iy=1:ny-1 spans each target face.
    target_rows = {
        "inner target": np.arange(0, layers),
        "outer target": np.arange(nx - 2, nx - 2 - layers, -1),
    }
    result = {}
    for target, rows in target_rows.items():
        e_fields = {"density": [], "upar": [], "upol": []}
        g_fields = {"density": [], "upar": [], "upol": []}
        eirene_r, eirene_z = [], []
        mapped_r, mapped_z, match_distance = [], [], []
        distance = None
        for ix in rows:
            iy = np.arange(1, ny - 1)
            er = np.mean(geom["crx"][ix, iy], axis=-1)
            ez = np.mean(geom["cry"][ix, iy], axis=-1)
            if distance is None:
                distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(er), np.diff(ez)))]
            distances, nearest = tree.query(np.column_stack((er, ez)))
            row, col = vr[nearest], vc[nearest]
            e_fields["density"].append(
                _species_index(edata.fort31["na"], species_index)[ix, iy]
            )
            e_fields["upar"].append(
                _species_index(edata.fort31["ua"], species_index)[ix, iy]
            )
            e_fields["upol"].append(
                _species_index(edata.fort31["up"], species_index)[ix, iy]
            )
            g_fields["density"].append(n_matrix[row, col])
            g_fields["upar"].append(upar_matrix[row, col])
            # Mirror prepare_fort31 exactly: mapped GENE-X upar multiplied by
            # EIRENE's local B_pol/B pitch factor.
            g_fields["upol"].append(
                upar_matrix[row, col] * edata.fort31["bb"][ix, iy, 0]
            )
            eirene_r.append(er)
            eirene_z.append(ez)
            mapped_r.append(rr[row, col])
            mapped_z.append(zz[row, col])
            match_distance.append(distances)
        result[target] = {
            "distance": distance,
            "eirene_rows": rows,
            "eirene": {key: np.asarray(value) for key, value in e_fields.items()},
            "genex": {key: np.asarray(value) for key, value in g_fields.items()},
            "eirene_r": np.asarray(eirene_r),
            "eirene_z": np.asarray(eirene_z),
            "genex_r": np.asarray(mapped_r),
            "genex_z": np.asarray(mapped_z),
            "match_distance": np.asarray(match_distance),
        }
    return result


def plot_target_neighborhoods(neighborhoods, output: Path | None):
    fields = (("density", "n [m$^{-3}$]", False),
              ("upar", "$u_\\parallel$ [m/s]", True),
              ("upol", "$u_{pol}$ [m/s]", True))
    rows = [(target, code) for target in neighborhoods for code in ("eirene", "genex")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 3.2 * len(rows)),
                             constrained_layout=True, squeeze=False)
    for plot_row, (target, code) in enumerate(rows):
        data = neighborhoods[target]
        x = data["distance"]
        y = np.arange(data[code]["density"].shape[0])
        for col, (field, unit, signed) in enumerate(fields):
            values = data[code][field]
            combined = np.concatenate([
                np.ravel(data["eirene"][field]),
                np.ravel(data["genex"][field]),
            ])
            norm = _norm(combined, signed)
            if norm is None:
                finite = combined[np.isfinite(combined)]
                if finite.size:
                    low, high = np.min(finite), np.max(finite)
                    if low == high:
                        high = low + max(abs(low), 1.0) * 1e-12
                    norm = Normalize(vmin=low, vmax=high)
            artist = axes[plot_row, col].pcolormesh(
                x, y, values, shading="nearest",
                cmap="coolwarm" if signed else "viridis",
                norm=norm,
            )
            axes[plot_row, col].axhspan(-0.45, 0.45, facecolor="none",
                                        edgecolor="red", lw=2)
            axes[plot_row, col].set(
                title=f"{target}: {code} {field}", ylabel="row behind target",
                xlabel="distance along target [m]",
            )
            axes[plot_row, col].set_yticks(y)
            fig.colorbar(artist, ax=axes[plot_row, col], label=unit)
    fig.suptitle("Target neighborhoods; red box = row used for flux")
    if output:
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return fig


def save_target_neighborhoods(neighborhoods, filename: Path):
    arrays = {}
    for target, data in neighborhoods.items():
        prefix = target.replace(" ", "_")
        arrays[f"{prefix}_distance"] = data["distance"]
        arrays[f"{prefix}_eirene_rows"] = data["eirene_rows"]
        for code in ("eirene", "genex"):
            for field, values in data[code].items():
                arrays[f"{prefix}_{code}_{field}"] = values
        for key in ("eirene_r", "eirene_z", "genex_r", "genex_z",
                    "match_distance"):
            arrays[f"{prefix}_{key}"] = data[key]
    np.savez(filename, **arrays)


def radial_boundary_neighborhoods(eirene_path: Path, archive: Path, surfaces,
                                  genex, species_index: int, layers: int = 3):
    """Extract density, radial velocity, and flux beside radial boundaries."""
    geom = B2(eirene_path).gmtry
    nx, ny = geom["vol"].shape
    edata = eirene()
    edata.read_ft31(archive / "fort.31", nx, ny, max(species_index + 1, 1))
    result = {}

    for boundary in ("private region", "core", "outer SOL"):
        surface = surfaces[boundary]
        for segment_number, (boundary_ix, boundary_iy) in enumerate(
            surface["segments"], start=1
        ):
            ix = np.atleast_1d(boundary_ix).astype(int)
            iy0 = int(boundary_iy)
            step = 1 if surface["sign"] < 0 else -1
            iy_layers = iy0 + step * np.arange(layers)
            if np.any((iy_layers < 0) | (iy_layers >= ny)):
                raise ValueError(
                    f"Requested radial layers leave EIRENE mesh at {boundary}"
                )
            label = (boundary if len(surface["segments"]) == 1
                     else f"{boundary} leg {segment_number}")
            e_fields = {key: [] for key in ("density", "velocity", "flux")}
            g_fields = {key: [] for key in ("density", "velocity", "flux")}
            eirene_r, eirene_z = [], []
            mapped_r, mapped_z, match_distance = [], [], []
            distance = None

            for iy in iy_layers:
                er = np.mean(geom["crx"][ix, iy], axis=-1)
                ez = np.mean(geom["cry"][ix, iy], axis=-1)
                if distance is None:
                    distance = np.r_[
                        0.0, np.cumsum(np.hypot(np.diff(er), np.diff(ez)))
                    ]
                distances, nearest = genex["tree"].query(
                    np.column_stack((er, ez))
                )
                row = genex["valid_row"][nearest]
                col = genex["valid_col"][nearest]
                en = _species_index(edata.fort31["na"], species_index)[ix, iy]
                ev = _species_index(edata.fort31["vv"], species_index)[ix, iy]
                ef = _species_index(edata.fort31["fnay"], species_index)[ix, iy]
                gn = genex["density"][row, col]
                gv = genex["urad"][row, col]
                e_fields["density"].append(en)
                e_fields["velocity"].append(ev)
                e_fields["flux"].append(ef)
                g_fields["density"].append(gn)
                g_fields["velocity"].append(gv)
                g_fields["flux"].append(gn * gv)
                eirene_r.append(er)
                eirene_z.append(ez)
                mapped_r.append(genex["rr"][row, col])
                mapped_z.append(genex["zz"][row, col])
                match_distance.append(distances)

            result[label] = {
                "distance": distance,
                "eirene_columns": iy_layers,
                "eirene": {key: np.asarray(value)
                           for key, value in e_fields.items()},
                "genex": {key: np.asarray(value)
                          for key, value in g_fields.items()},
                "eirene_r": np.asarray(eirene_r),
                "eirene_z": np.asarray(eirene_z),
                "genex_r": np.asarray(mapped_r),
                "genex_z": np.asarray(mapped_z),
                "match_distance": np.asarray(match_distance),
            }
    return result


def plot_radial_neighborhoods(neighborhoods, output: Path | None):
    fields = (("density", "n [m$^{-3}$]", False),
              ("velocity", "$u_{rad}$ [m/s]", True),
              ("flux", "$n u_{rad}$ [m$^{-2}$ s$^{-1}$]", True))
    rows = [(surface, code) for surface in neighborhoods
            for code in ("eirene", "genex")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 3.1 * len(rows)),
                             constrained_layout=True, squeeze=False)
    for plot_row, (surface, code) in enumerate(rows):
        data = neighborhoods[surface]
        y = np.arange(data[code]["density"].shape[0])
        for col, (field, unit, signed) in enumerate(fields):
            combined = np.concatenate([
                np.ravel(data["eirene"][field]),
                np.ravel(data["genex"][field]),
            ])
            norm = _norm(combined, signed)
            if norm is None:
                finite = combined[np.isfinite(combined)]
                if finite.size:
                    low, high = np.min(finite), np.max(finite)
                    if low == high:
                        high = low + max(abs(low), 1.0) * 1e-12
                    norm = Normalize(vmin=low, vmax=high)
            artist = axes[plot_row, col].pcolormesh(
                data["distance"], y, data[code][field], shading="nearest",
                cmap="coolwarm" if signed else "viridis", norm=norm,
            )
            axes[plot_row, col].axhspan(
                -0.45, 0.45, facecolor="none", edgecolor="red", lw=2
            )
            axes[plot_row, col].set(
                title=f"{surface}: {code} {field}",
                xlabel="distance along boundary [m]",
                ylabel="cell layer from boundary",
            )
            axes[plot_row, col].set_yticks(y)
            fig.colorbar(artist, ax=axes[plot_row, col], label=unit)
    fig.suptitle("Radial-boundary neighborhoods; red box = flux boundary row")
    if output:
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return fig


def save_radial_neighborhoods(neighborhoods, filename: Path):
    arrays = {}
    for surface, data in neighborhoods.items():
        prefix = surface.replace(" ", "_")
        arrays[f"{prefix}_distance"] = data["distance"]
        arrays[f"{prefix}_eirene_columns"] = data["eirene_columns"]
        for code in ("eirene", "genex"):
            for field, values in data[code].items():
                arrays[f"{prefix}_{code}_{field}"] = values
        for key in ("eirene_r", "eirene_z", "genex_r", "genex_z",
                    "match_distance"):
            arrays[f"{prefix}_{key}"] = data[key]
    np.savez(filename, **arrays)


def plot_radial_density_rz(neighborhoods, output: Path | None):
    """Compare all matched radial-neighborhood densities in the R-Z plane."""
    eirene_r = np.concatenate([
        np.ravel(data["eirene_r"]) for data in neighborhoods.values()
    ])
    eirene_z = np.concatenate([
        np.ravel(data["eirene_z"]) for data in neighborhoods.values()
    ])
    eirene_n = np.concatenate([
        np.ravel(data["eirene"]["density"])
        for data in neighborhoods.values()
    ])
    genex_r = np.concatenate([
        np.ravel(data["genex_r"]) for data in neighborhoods.values()
    ])
    genex_z = np.concatenate([
        np.ravel(data["genex_z"]) for data in neighborhoods.values()
    ])
    genex_n = np.concatenate([
        np.ravel(data["genex"]["density"])
        for data in neighborhoods.values()
    ])
    combined = np.concatenate((eirene_n, genex_n))
    finite = combined[np.isfinite(combined)]
    norm = _norm(combined, signed=False)
    if norm is None and finite.size:
        low, high = np.min(finite), np.max(finite)
        if low == high:
            high = low + max(abs(low), 1.0) * 1e-12
        norm = Normalize(vmin=low, vmax=high)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    eirene = ax.scatter(
        eirene_r, eirene_z, c=eirene_n, norm=norm, cmap="viridis",
        marker="o", s=18, alpha=0.7, label="EIRENE",
    )
    ax.scatter(
        genex_r, genex_z, c=genex_n, norm=norm, cmap="viridis",
        marker="^", s=24, alpha=0.7, label="GENE-X",
    )
    ax.set(
        title="Full-domain matched radial-boundary density samples (R-Z)",
        xlabel="R [m]",
        ylabel="Z [m]",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.colorbar(eirene, ax=ax, shrink=0.7, pad=0.1,
                 label="density [m$^{-3}$]")
    if output:
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return fig


def plot_lineouts(lineouts, output: Path | None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             constrained_layout=True)
    fields = ((1, "density [m$^{-3}$]"),
              (2, "$u_\\parallel$ [m/s]"),
              (3, "$u_{pol}$ [m/s]"))
    for label, values in lineouts.items():
        if label == "core":
            continue
        for ax, (index, ylabel) in zip(axes, fields):
            ax.plot(values[0], values[index], label=label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("distance along each boundary [m]")
    if output:
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return fig


def _print_rates(title, rates):
    print(f"\n{title}")
    total = 0.0
    wall_total = 0.0
    for label, value in rates.items():
        if isinstance(value, dict):
            print(f"  {label:20s} stored={value['stored']: .6e} s^-1  "
                  f"n*u reconstructed={value['reconstructed']: .6e} s^-1")
            contribution = value["reconstructed"]
        else:
            print(f"  {label:20s} {value: .6e} s^-1")
            contribution = value
        total += contribution
        if label != "core":
            wall_total += contribution
    print(f"  {'physical-wall total':20s} {wall_total: .6e} s^-1")
    print(f"  {'all-boundary total':20s} {total: .6e} s^-1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genex_path", type=Path)
    parser.add_argument("eirene_path", type=Path)
    parser.add_argument("-t", "--time", type=float,
                        help="GENE-X tau; closest available time is selected "
                             "(default: latest snapshot)")
    parser.add_argument("--species", help="GENE-X ion species (default: first ion)")
    parser.add_argument("--eirene-species-index", type=int, default=0)
    parser.add_argument("--eirene-file-pattern", default="eirene_sources")
    parser.add_argument("--eirene-iteration", type=int)
    parser.add_argument("--fort31-template", type=Path,
                        help="mask template (default: EIRENE_PATH/fort.31.template, "
                             "falling back to make_fort31/fort.31)")
    parser.add_argument("-o", "--output", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "genex_boundary_lineouts.png")
    parser.add_argument("--surface-output", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "matched_boundary_surfaces.png")
    parser.add_argument("--target-output", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "target_neighborhoods.png")
    parser.add_argument("--target-data", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "target_neighborhoods.npz")
    parser.add_argument("--target-layers", type=int, default=3,
                        help="total rows: target plus plasma-side rows (default: 3)")
    parser.add_argument("--radial-output", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "radial_boundary_neighborhoods.png")
    parser.add_argument("--radial-data", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "radial_boundary_neighborhoods.npz")
    parser.add_argument("--radial-density-rz-output", type=Path,
                        default=ANALYSIS_OUTPUT_DIR / "radial_density_rz.png")
    parser.add_argument("--radial-layers", type=int, default=3,
                        help="boundary plus plasma-side radial cells (default: 3)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    for output_path in (
        args.output,
        args.surface_output,
        args.target_output,
        args.target_data,
        args.radial_output,
        args.radial_data,
        args.radial_density_rz_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    eirene_rates, surfaces, archive, iteration, mask_report = load_eirene_boundary(
        args.eirene_path, args.eirene_file_pattern, args.eirene_iteration,
        args.eirene_species_index, args.fort31_template,
    )
    (genex_rates, lineouts, species, time_index, selected_time,
     genex_context) = load_genex_boundary(
         args.genex_path, args.time, args.species, surfaces
     )
    requested = "latest" if args.time is None else f"{args.time:.16g}"
    print(f"GENE-X requested tau={requested}; selected time_index={time_index}, "
          f"tau={selected_time:.16g}")
    print(f"EIRENE archive: {archive} (iteration {iteration:06d})")
    print(f"fort.31 mask template: {mask_report['template']}")
    print("  fnax active [ix first:last, iy first:last] = "
          f"{mask_report['fnax_active_bounds']}")
    print("  fnay active [ix first:last, iy first:last] = "
          f"{mask_report['fnay_active_bounds']}")
    print("  nonzero mask counts, template -> archive: "
          f"fnax {mask_report['template_fnax_nonzero']} -> "
          f"{mask_report['archive_fnax_nonzero']}, fnay "
          f"{mask_report['template_fnay_nonzero']} -> "
          f"{mask_report['archive_fnay_nonzero']}")
    _print_rates(f"EIRENE species index {args.eirene_species_index}", eirene_rates)
    _print_rates(
        f"GENE-X {species}, toroidal average "
        "(targets: upar projected to upol; radial boundaries: u_rad)",
        genex_rates,
    )
    print("\nMaximum EIRENE-face to GENE-X-cell matching distance:")
    for label, values in lineouts.items():
        print(f"  {label:20s} {values[6]:.6e} m")
    plot_lineouts(lineouts, None if args.show else args.output)
    plot_surface_map(
        lineouts, surfaces, None if args.show else args.surface_output
    )
    neighborhoods = target_neighborhoods(
        args.genex_path, args.eirene_path, archive, time_index,
        species, args.eirene_species_index,
        layers=args.target_layers,
    )
    plot_target_neighborhoods(
        neighborhoods, None if args.show else args.target_output
    )
    save_target_neighborhoods(neighborhoods, args.target_data)
    for target, data in neighborhoods.items():
        print(
            f"{target}: EIRENE ix rows {data['eirene_rows'].tolist()}, "
            f"maximum mapped-cell distance "
            f"{np.max(data['match_distance']):.6e} m"
        )
    radial = radial_boundary_neighborhoods(
        args.eirene_path, archive, surfaces, genex_context,
        args.eirene_species_index, layers=args.radial_layers,
    )
    plot_radial_neighborhoods(
        radial, None if args.show else args.radial_output
    )
    save_radial_neighborhoods(radial, args.radial_data)
    plot_radial_density_rz(
        radial, None if args.show else args.radial_density_rz_output
    )
    for surface, data in radial.items():
        print(
            f"{surface}: EIRENE iy columns "
            f"{data['eirene_columns'].tolist()}, maximum mapped-cell distance "
            f"{np.max(data['match_distance']):.6e} m"
        )
    if args.show:
        try:
            plt.show(block=True)
        finally:
            plt.close("all")
    else:
        print(f"\nWrote {args.output}")
        print(f"Wrote {args.surface_output}")
        print(f"Wrote {args.target_output}")
        print(f"Wrote {args.target_data}")
        print(f"Wrote {args.radial_output}")
        print(f"Wrote {args.radial_data}")
        print(f"Wrote {args.radial_density_rz_output}")


if __name__ == "__main__":
    main()
