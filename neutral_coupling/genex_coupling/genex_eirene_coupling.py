
import psutil
from pathlib import Path
from collections import defaultdict
from .interpolate import (ConservativeSourceMapper, interpolate_all_sources,
                         interpolate_source, interp_moments, build_triangulation)
from .eirene_interface import (eirene_interface, run_eirene, prepare_fort31,
                              get_temperatures, status)
from neutral_coupling.common.temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
)
from . import genex_interface
from neutral_coupling.common import source_post_processor as SPP
from .write_netcdf import write_sources_nc
from os import replace
from time import (sleep, perf_counter)
from types import SimpleNamespace
import numpy as np
import dask
import shutil
import argparse
import os
import re
import faulthandler
import scipy.constants as pyconst
import hashlib
import json
from dataclasses import asdict
from .boundary_flux import (
    BoundaryComponent,
    eirene_boundary_components,
    map_boundary_cell_sequences,
    normalize_eirene_particle_fluxes,
    structured_boundary_components,
    write_normalization_diagnostics,
)

default_eirene_path = Path("/pscratch/sd/j/jonroelt/tcv_genex_coupling/coupled_run/")
default_genex_path = Path("/pscratch/sd/j/jonroelt/source_testing/3D_source/branch_all_source_file_params/")
default_stop = Path("/global/cfs/cdirs/m2116/solps-iter")

CONFIG_DEFAULTS = {
    "pid": None,
    "pid_file": None,
    "sum_temperatures": True,
    "filepattern": "./input_sources",
    "eirene_command": "eirobjx",
    "genex_path": default_genex_path,
    "eirene_path": default_eirene_path,
    "max_timeouts": 1,
    "eirene_time": 200,
    "solpstop": default_stop,
    "genex_time_index_override": False,
    "genex_read_mode": "averaged",
    "genex_read_attempts": 6,
    "genex_retry_delay": 0.5,
    "genex_retry_max_delay": 10.0,
    "boundary_flux": {
        "normalization": "surface",
        "allow_misalignment": False,
        "factor_min": 0.5,
        "factor_max": 2.0,
        "negligible_fraction": 0.01,
        "radial_scaling": 1.0,
    },
}

_PATH_KEYS = {"pid_file", "genex_path", "eirene_path", "solpstop", "filepattern"}


def _expand_config_value(value, config_dir):
    """Expand environment variables and resolve configured filesystem paths."""
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    if re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded):
        raise ValueError(f"configuration references an unset environment variable: {value}")
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else config_dir / path


def load_coupling_config(path):
    """Load and validate the public coupling YAML schema."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML configuration requires PyYAML; install neutral-coupling[genex]"
        ) from error

    path = Path(path).resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("coupling configuration must be a YAML mapping")

    unknown = set(data) - set(CONFIG_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown coupling configuration keys: {sorted(unknown)}")
    boundary = data.get("boundary_flux", {})
    if not isinstance(boundary, dict):
        raise ValueError("boundary_flux must be a YAML mapping")
    unknown_boundary = set(boundary) - set(CONFIG_DEFAULTS["boundary_flux"])
    if unknown_boundary:
        raise ValueError(
            f"unknown boundary_flux configuration keys: {sorted(unknown_boundary)}"
        )

    values = {**CONFIG_DEFAULTS, **data}
    values["boundary_flux"] = {**CONFIG_DEFAULTS["boundary_flux"], **boundary}
    expected = {
        "pid": (int, type(None)), "pid_file": (str, Path, type(None)),
        "sum_temperatures": (bool,), "filepattern": (str, Path),
        "eirene_command": (str,), "genex_path": (str, Path),
        "eirene_path": (str, Path), "max_timeouts": (int,),
        "eirene_time": (int, float), "solpstop": (str, Path),
        "genex_time_index_override": (bool,), "genex_read_mode": (str,),
        "genex_read_attempts": (int,),
        "genex_retry_delay": (int, float),
        "genex_retry_max_delay": (int, float),
    }
    for key, types in expected.items():
        if not isinstance(values[key], types) or (
                bool not in types and isinstance(values[key], bool)):
            raise ValueError(f"{key} has the wrong type")
    boundary_expected = {
        "normalization": (str,), "allow_misalignment": (bool,),
        "factor_min": (int, float), "factor_max": (int, float),
        "negligible_fraction": (int, float),
        "radial_scaling": (int, float),
    }
    for key, types in boundary_expected.items():
        value = values["boundary_flux"][key]
        if not isinstance(value, types) or (
                bool not in types and isinstance(value, bool)):
            raise ValueError(f"boundary_flux.{key} has the wrong type")
    for key in _PATH_KEYS:
        if values[key] is not None:
            values[key] = _expand_config_value(values[key], path.parent)
    command = values["eirene_command"]
    if "/" in command:
        values["eirene_command"] = str(
            _expand_config_value(command, path.parent)
        )
    return values


def _config_namespace(values):
    """Translate public YAML names to the legacy namespace consumed by main."""
    boundary = values["boundary_flux"]
    return SimpleNamespace(
        pid=values["pid"], pid_file=values["pid_file"],
        SumTemp=values["sum_temperatures"], filepattern=values["filepattern"],
        eirene_command=values["eirene_command"],
        genex_path=values["genex_path"], eirene_path=values["eirene_path"],
        MAX_TIMEOUTS=values["max_timeouts"], eirene_time=values["eirene_time"],
        solpstop=values["solpstop"],
        genex_time_index_override=values["genex_time_index_override"],
        genex_read_mode=values["genex_read_mode"],
        genex_read_attempts=values["genex_read_attempts"],
        genex_retry_delay=values["genex_retry_delay"],
        genex_retry_max_delay=values["genex_retry_max_delay"],
        boundary_flux_normalization=boundary["normalization"],
        allow_boundary_misalignment=boundary["allow_misalignment"],
        boundary_flux_factor_min=boundary["factor_min"],
        boundary_flux_factor_max=boundary["factor_max"],
        boundary_flux_negligible_fraction=boundary["negligible_fraction"],
        radial_flux_scaling=boundary["radial_scaling"],
    )


def coupling_args_from_file(path):
    """Return a validated coupling namespace from a YAML file."""
    args = _config_namespace(load_coupling_config(path))
    _validate_args(args)
    return args


def _validate_args(args, error=ValueError):
    """Apply common validation to YAML and command-line configurations."""
    if args.genex_read_mode not in ("averaged", "full"):
        raise error("genex_read_mode must be 'averaged' or 'full'")
    if args.boundary_flux_normalization not in ("surface", "off"):
        raise error("boundary flux normalization must be 'surface' or 'off'")
    if args.MAX_TIMEOUTS < 1 or args.eirene_time <= 0:
        raise error("max_timeouts and eirene_time must be positive")
    if args.genex_read_attempts < 1:
        raise error("genex_read_attempts must be positive")
    if args.genex_retry_delay < 0 or args.genex_retry_max_delay < 0:
        raise error("GENE-X retry delays must be non-negative")
    if args.boundary_flux_factor_min <= 0:
        raise error("boundary flux factor minimum must be positive")
    if args.boundary_flux_factor_max < args.boundary_flux_factor_min:
        raise error("boundary flux factor maximum must be >= minimum")
    if (not np.isfinite(args.boundary_flux_negligible_fraction)
            or not 0 <= args.boundary_flux_negligible_fraction <= 1):
        raise error("boundary flux negligible fraction must be finite and in [0, 1]")
    if not np.isfinite(args.radial_flux_scaling) or args.radial_flux_scaling < 0:
        raise error("radial flux scaling must be finite and non-negative")


def resolve_pid(args):
    """Resolve and validate a configured PID or PID file."""
    if args.pid is None:
        pid_file = getattr(args, "pid_file", None)
        if pid_file is None:
            pid_file = Path(args.genex_path) / "genex.pid"
        try:
            args.pid = int(Path(pid_file).read_text().strip())
        except (OSError, ValueError) as error:
            raise ValueError(f"could not read a valid GENE-X PID from {pid_file}") from error
    if args.pid <= 0:
        raise ValueError("GENE-X PID must be a positive integer")
    if not psutil.pid_exists(args.pid):
        raise ValueError(f"GENE-X process {args.pid} is not running")
    return args

def main(args, deps=None):
    faulthandler.enable(all_threads=True)
    if deps is None:
        deps = SimpleNamespace(
            run_eirene=run_eirene,
            write_nc=write_sources_nc,
            sleep=sleep,
            replace=replace,
            pid_exists=psutil.pid_exists,
            collision_mappers=None,
            sparse_temperature_handler=default_sparse_temperature_handler,
            pseudo_temperature_handler=(
                default_single_species_pseudo_temperature_handler
            ),
        )
    """
        Main driver for GENE-X/Eirene coupling.
        It is meant to run simultaneously with a GENE-X run to include
        Eirene neutrals.

        args - command line arguments. Run with --help to print or see at the
            bottom of this file

        deps - Functional dependencies, primarily to allow replacements during
            testing. However, several of special note allowing change in
            functionality, are:
            collision_mappers - How to map Eirene sources to the different
                species' temperatures. If none is selected, a default for a pure
                D+ simulation is selected.
            sparse_temperature_handler - How to handle cells where the
                temperature is 0 for groupling sources by temperature. Default
                is for a case with pure D+.
            pseudo_temperature_handler - How to handle the temperature
                of pseudo species. Default is for a case with pure D+.
    """
    dask.config.set(scheduler="synchronous")
    eirene_path = Path(args.eirene_path)
    genex_path = Path(args.genex_path)
    # Load static in time data
    edat, b2dat, pol_mask, rad_mask = eirene_interface(eirene_path, eirene_path)
    grid, equi, params, norm, r_all, z_all, compute, in_target, buf_zone = (
        genex_interface.wait_for_genex_init(genex_path)
    )
    params = normalize_genex_params(params)
    genex_species = genex_interface.get_genex_species(params)
    genex_electrons = get_genex_electron_name(genex_species)
    check_species_consistency(edat.species_names["bulk_ions"], genex_species)
    grid_r = np.asarray(r_all*norm["R0"])
    # TODO: Need to change this negative to function of grid._flipped_z and equi._flipped_Z
    grid_z = np.asarray(z_all*norm["R0"])
    in_plasma, plasma_field_mask = make_in_plasma_mask(
        compute, in_target, buf_zone
    )
    buf_zone_on_compute = np.asarray(buf_zone).reshape(-1)[
        np.asarray(compute, dtype=bool)
    ]
    if params["params_time_loop"]["start_from_checkpoint"]:
        index = next_eirene_index(eirene_path, args.filepattern)
    else:
        index = 0
    MAX_TIMEOUTS = args.MAX_TIMEOUTS
    if args.genex_time_index_override:
        time_index = 0
        ntau = 40
    else:
        time_index = -1
    last_tau = -1
    # Precompute triangulation
    tri = build_triangulation(grid_r[in_plasma], grid_z[in_plasma])
    genex_weights = genex_interface.get_genex_source_weights(
        genex_path, norm["R0"]
    )
    conservative_mapper = None
    boundary_mode = getattr(args, "boundary_flux_normalization", "surface")
    eirene_boundary = None
    boundary_groups = None
    genex_boundary = None
    boundary_mapping_path = eirene_path / "boundary_flux_mapping.json"
    boundary_report_path = eirene_path / "boundary_flux_report.txt"
    boundary_fingerprints = None
    saved_boundary_mapping = None
    allow_boundary_misalignment = getattr(
        args, "allow_boundary_misalignment", False
    )
    if boundary_mode == "surface":
        eirene_boundary = eirene_boundary_components(
            b2dat.gmtry, ~np.asarray(pol_mask), ~np.asarray(rad_mask)
        )
        boundary_fingerprints = geometry_fingerprints(
            eirene_path, genex_path
        )
        saved_boundary_mapping = load_boundary_mapping(
            boundary_mapping_path, boundary_fingerprints,
            report_path=boundary_report_path,
            allow_boundary_misalignment=allow_boundary_misalignment,
        )
    timeout = 600
    last_tau_advance = perf_counter()
    # Main loop - runs for duration of GENE-X
    while deps.pid_exists(args.pid):
        genex_fields, tau = genex_interface.load_latest_genex_fields(
            genex_path, genex_species, grid, equi, params, norm,
            time_index, timeout,
            read_mode=getattr(args, "genex_read_mode", "averaged"),
            read_attempts=getattr(args, "genex_read_attempts", 6),
            retry_delay=getattr(args, "genex_retry_delay", 0.5),
            retry_max_delay=getattr(args, "genex_retry_max_delay", 10.0),
        )

        timeout = 300
        if (tau <= last_tau):
            if (perf_counter()-last_tau_advance > timeout*3):
                raise RuntimeError(f"GENE-X does not appear to be advancing."
                                   f"Diagnostics have been at t={tau} for more"
                                   f"than {timeout*3} seconds.")
            deps.sleep(5)
            continue
        unnormalize_all(genex_fields)
        genex_fields_2D = genex_fields
        ion_temperature_values = {}
        if not args.SumTemp:
            # Prepare GENE-X ion temperatures for use as source temperature
            ion_temperature_values = prepare_genex_ion_temperatures(
                genex_fields_2D,
                genex_species,
                grid_r,
                grid_z,
                in_plasma,
                deps.sparse_temperature_handler,
                field_mask=plasma_field_mask,
            )
        interpolated = interpolate_all_moments(b2dat.gmtry, tri,
                                               genex_fields_2D, rad_mask,
                                               pol_mask,
                                               field_mask=plasma_field_mask)
        prepare_fort31(edat, interpolated, genex_electrons,
                     edat.species_names["bulk_ions"])
        normalization_records = None
        if boundary_mode == "surface":
            native_fluxes, current_boundary = prepare_native_boundary_fluxes(
                grid, compute, in_target, buf_zone_on_compute, genex_fields_2D,
                edat.species_names["bulk_ions"],
                length_scale=norm["R0"],
            )
            if genex_boundary is None:
                if saved_boundary_mapping is None:
                    try:
                        (boundary_groups, genex_boundary,
                         mapping_report) = map_boundary_cell_sequences(
                            eirene_boundary, current_boundary,
                            allow_boundary_misalignment=
                                allow_boundary_misalignment,
                        )
                    except ValueError as error:
                        write_boundary_mapping_failure(
                            eirene_path / "boundary_flux_mapping_failure.json",
                            boundary_fingerprints,
                            error,
                            eirene_boundary,
                            current_boundary,
                            report_path=boundary_report_path,
                        )
                        raise
                    save_boundary_mapping(
                        boundary_mapping_path,
                        boundary_fingerprints,
                        boundary_groups,
                        mapping_report,
                        eirene_boundary,
                        genex_boundary,
                        report_path=boundary_report_path,
                    )
                else:
                    genex_boundary = [
                        component_from_summary(item)
                        for item in saved_boundary_mapping["genex_components"]
                    ]
                    boundary_groups = validate_saved_groups(
                        saved_boundary_mapping["groups"],
                        len(eirene_boundary),
                        len(genex_boundary),
                    )
                    mapping_report = saved_boundary_mapping["mapping_report"]
                write_boundary_mapping_report(
                    mapping_report, boundary_mapping_path,
                    boundary_report_path,
                )
            normalization_records = normalize_eirene_particle_fluxes(
                edat.fort31,
                eirene_boundary,
                native_fluxes,
                genex_boundary,
                boundary_groups,
                edat.species_names["bulk_ions"],
                factor_min=getattr(args, "boundary_flux_factor_min", .5),
                factor_max=getattr(args, "boundary_flux_factor_max", 2.0),
                negligible_fraction=getattr(
                    args, "boundary_flux_negligible_fraction", .01
                ),
                radial_flux_scaling=getattr(args, "radial_flux_scaling", 1.0),
                allow_boundary_misalignment=allow_boundary_misalignment,
            )
            write_boundary_normalization_report(
                normalization_records, index, tau, boundary_report_path
            )
            write_normalization_diagnostics(
                eirene_path / "boundary_flux_normalization.json",
                normalization_records,
            )
        del genex_fields_2D, genex_fields
        edat.write_ft31(eirene_path / Path("fort.31"))

        print(f"[{index}] Running EIRENE with GENE-X tau={tau}", flush=True)
        num_timeouts = 0
        eirene_time = args.eirene_time
        while num_timeouts<MAX_TIMEOUTS:
            attempt = num_timeouts + 1
            output_file = (
                eirene_path
                / f"run_{index:06d}_attempt_{attempt:02d}.log"
            )
            eirene_status = deps.run_eirene(
                eirene_time,
                eirene_path=eirene_path,
                command=args.eirene_command,
                solpstop=getattr(args, "solpstop", default_stop),
                output_file=output_file,
            )
            if eirene_status == status.SUCCESS:
                break
            elif eirene_status == status.TIMEOUT:
                num_timeouts += 1
                print(f"Eirene timeout after {eirene_time} s.")
                if num_timeouts < MAX_TIMEOUTS:
                    eirene_time *= 2
                    print(f"Re-running Eirene with {eirene_time} s.")
                else:
                    raise RuntimeError(
                        f"EIRENE timed out during coupling iteration {index} "
                        f"after {num_timeouts} attempts; incomplete files "
                        f"remain in {eirene_path}"
                    )
            elif eirene_status == status.ERROR:
                raise RuntimeError(
                    f"EIRENE failed during coupling iteration {index}; "
                    f"complete output is in {output_file}; incomplete files "
                    f"remain in {eirene_path}"
                )
        # Source files and fort.46 are independent EIRENE outputs.
        # load_extra_forts reads only the former, while conservative mapping
        # requires the physical triangle volumes stored in fort.46 in both
        # summed- and split-temperature modes.
        edat.read_ft46(eirene_path / "fort.46")
        edat.load_extra_forts(eirene_path=eirene_path, coll_to_adjust=None,
                              convert_units=True)
        if conservative_mapper is None:
            try:
                eirene_weights = np.asarray(
                    edat.fort46["volumes"], dtype=float
                ).reshape(-1)
            except KeyError as error:
                raise ValueError(
                    "EIRENE fort.46 did not contain triangle volumes after "
                    "a successful run; conservative source interpolation "
                    "requires the 'volumes' field"
                ) from error
            conservative_mapper = ConservativeSourceMapper(
                edat.triangle_mesh.incenter,
                np.column_stack((grid_r[in_plasma], grid_z[in_plasma])),
                eirene_weights,
                genex_weights[in_plasma],
            )

        interp_temperatures = None
        write_temperatures = False
        if args.SumTemp:
            sources = {
                mom: {
                    species: {"SUM": strata["SUM"]}
                    for species, strata in species_dict.items()
                    if "SUM" in strata
                }
                for mom, species_dict in edat.sources.items()
            }
        else: # Split sources by temperature
            temps = get_temperatures(
                edat,
                eirene_path,
                ion_temperature_values,
                tri,
                sparse_temperature_handler=deps.sparse_temperature_handler,
                pseudo_temperature_handler=deps.pseudo_temperature_handler,
            )
            spp = SPP.SourcePostProcessor(
                edat.full_source_in_SI,
                temps,
                collision_mappers=deps.collision_mappers,
            )
            sources, temps = spp.regroup_by_temperature()
            direct_ion_temperatures = {
                f"Ti_{species}": np.asarray(value) * pyconst.elementary_charge
                for species, value in ion_temperature_values.items()
            }
            interp_temperatures = interpolate_temperature_values(
                edat.triangle_mesh,
                temps,
                grid_r,
                grid_z,
                in_plasma,
                direct_values=direct_ion_temperatures,
            )
            write_temperatures = True

        source_diagnostics = {}
        interp_sources = interpolate_all_sources_wrapper(
            edat.triangle_mesh, sources,
            grid_r, grid_z, in_plasma,
            mapper=conservative_mapper,
            diagnostics=source_diagnostics,
            genex_electrons=genex_electrons,
        )
        diagnostics_path = eirene_path / f"source_mapping_{index:06d}.json"
        diagnostics_path.write_text(json.dumps(
            {label: asdict(report) for label, report in source_diagnostics.items()},
            indent=2,
        ) + "\n")
        significant = [label for label, report in source_diagnostics.items()
                       if report.significant]
        worst = max(source_diagnostics.items(),
                    key=lambda item: item[1].worst_relative_error,
                    default=None)
        if worst is not None:
            print(
                f"[{index}] Conservative source mapping: max global residual="
                f"{max(r.global_residual for r in source_diagnostics.values()):.3e}, "
                f"worst regional error={worst[1].worst_relative_error:.3e} "
                f"({worst[0]}), flagged fields={len(significant)}",
                flush=True,
            )

        filename = source_filename(args.filepattern, index)
        filename_tmp = filename + ".tmp"
        print(f"[{index}] Writing {filename_tmp}", flush=True)
        deps.write_nc(
            filename_tmp,
            interp_sources,
            grid_r.size,
            temperature_values=interp_temperatures,
            write_temperature=write_temperatures,
            genex_tau=tau,
        )
        deps.replace(filename_tmp, filename)
        backup_eirene_files(eirene_path, index)
        index += 1
        last_tau = tau
        last_tau_advance = perf_counter()
        if args.genex_time_index_override:
            time_index += 1
            if ntau<time_index:
                break

def get_genex_electron_name(genex_species):
    for sp in genex_species:
        if(sp.is_electron):
            return sp.name

def check_species_consistency(eirene_species, genex_species):
    """ Ensure GENE-X and Eirene species are consistent """
    for sp in genex_species:
        if (sp.name not in eirene_species and not sp.is_electron):
            raise ValueError(f"Genex Species {sp.name} not known to Eirene. "
                             f"Eirene ion species are "
                             f"{' ,'.join(eirene_species)}.")

def unnormalize_all(genex_out):
    """ Convert GENE-X data to SI"""
    for field, field_block in genex_out.items():
        for species, value in field_block.items():
            genex_out[field][species] = genex_interface.unnormalize(value)


def make_in_plasma_mask(compute, in_target, buf_zone):
    """Return full-grid and active-relative masks for unmodified plasma."""
    compute = np.asarray(compute, dtype=bool)
    target_on_compute = np.asarray(in_target, dtype=bool).reshape(-1)
    buf_zone = np.asarray(buf_zone).reshape(-1)
    if buf_zone.size != compute.size:
        raise ValueError("buf_zone size does not match the full GENE-X grid")
    if target_on_compute.size != np.count_nonzero(compute):
        raise ValueError(
            "in_target size does not match the number of GENE-X compute cells"
        )
    buffer_on_compute = buf_zone[compute] != 0
    plasma_field_mask = ~target_on_compute & ~buffer_on_compute
    in_plasma = np.zeros_like(compute)
    in_plasma[compute] = plasma_field_mask
    return in_plasma, plasma_field_mask


def _grid_matrix(grid, values):
    """Use GENE-X's native logical matrix representation."""
    array = np.asarray(values)
    matrix = grid.vector_to_matrix(values) if array.ndim == 1 else values
    return np.asarray(matrix)


def _grid_coordinate_matrices(grid, shape):
    r = np.asarray(grid.r_s, dtype=float)
    z = np.asarray(grid.z_s, dtype=float)
    if r.ndim == z.ndim == 1:
        rr, zz = np.meshgrid(r, z)
    else:
        rr, zz = np.broadcast_arrays(r, z)
    if rr.shape != shape:
        if rr.T.shape == shape:
            rr, zz = rr.T, zz.T
        else:
            raise ValueError(
                f"GENE-X coordinate matrix {rr.shape} does not match fields {shape}"
            )
    return rr, zz


def prepare_native_boundary_fluxes(grid, compute, in_target,
                                   buf_zone_on_compute, fields,
                                   species_names, *, length_scale=1.0):
    """Build native GENE-X boundary components and n*u flux matrices."""
    n_compute = np.count_nonzero(np.asarray(compute))
    if np.asarray(in_target).size != n_compute:
        raise ValueError("GENE-X target mask does not match the compute mask")
    if np.asarray(buf_zone_on_compute).size != n_compute:
        raise ValueError(
            "GENE-X buffer-zone mask does not match the compute mask"
        )
    if not species_names:
        raise ValueError("boundary normalization requires at least one ion")
    first = species_names[0]
    for key in ("n", "u_pol", "u_rad"):
        if first not in fields.get(key, {}):
            raise ValueError(f"GENE-X field {key}/{first} is unavailable")
    density0 = _grid_matrix(grid, fields["n"][first])
    if hasattr(length_scale, "to"):
        length_scale = length_scale.to("m").magnitude
    rr, zz = _grid_coordinate_matrices(grid, density0.shape)
    rr = rr * float(length_scale)
    zz = zz * float(length_scale)

    target_matrix = _grid_matrix(grid, np.asarray(in_target, dtype=float))
    if target_matrix.shape != density0.shape:
        raise ValueError("GENE-X target mask does not match native field matrix")
    target_matrix = np.isfinite(target_matrix) & (target_matrix != 0)
    buffer_matrix = _grid_matrix(
        grid, np.asarray(buf_zone_on_compute, dtype=float)
    )
    if buffer_matrix.shape != density0.shape:
        raise ValueError("GENE-X buffer zones do not match native field matrix")
    # Build topology from static compute, target, and buffer-zone masks.
    # Density can legitimately be zero at startup and must not change the
    # saved geometry mapping.
    compute_matrix = _grid_matrix(
        grid, np.ones(n_compute, dtype=float)
    )
    if compute_matrix.shape != density0.shape:
        raise ValueError("GENE-X compute mask does not match native field matrix")
    active = (
        np.isfinite(compute_matrix)
        & ~target_matrix
        & (buffer_matrix == 0)
    )
    components = structured_boundary_components(
        active, rr, zz, target=(buffer_matrix == 4), prefix="genex"
    )

    fluxes = {}
    for kind, velocity_key in (("poloidal", "u_pol"), ("radial", "u_rad")):
        blocks = []
        for species in species_names:
            density = _grid_matrix(grid, fields["n"][species])
            velocity = _grid_matrix(grid, fields[velocity_key][species])
            if density.shape != density0.shape or velocity.shape != density0.shape:
                raise ValueError(f"native GENE-X {species} field shapes disagree")
            blocks.append(density * velocity)
        fluxes[kind] = blocks[0] if len(blocks) == 1 else np.stack(blocks, axis=-1)
    return fluxes, components


def _append_boundary_report(path, lines):
    with Path(path).open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def write_boundary_normalization_report(records, iteration, tau, path):
    lines = [f"[{iteration}] Boundary particle-flux normalization tau={tau}"]
    species_seen = set()
    for item in records:
        suffix = f"  WARNING: {item['warning']}" if "warning" in item else ""
        surface = "+".join(item["eirene_components"])
        lines.append(
            f"  group={item['surface_group']:02d} surface={surface} "
            f"species={item['species']} "
            f"out(GX,E)={item['genex_outward']:.6e},"
            f"{item['eirene_outward_before']:.6e}->"
            f"{item['eirene_outward_after']:.6e} "
            f"in(GX,E)={item['genex_inward']:.6e},"
            f"{item['eirene_inward']:.6e} "
            f"net(GX,E)={item['genex_net']:.6e},"
            f"{item['eirene_net_before']:.6e} "
            f"net_residual={item['net_residual']:.6g} "
            f"global_contribution="
            f"{item['global_net_residual_contribution']:.6g} "
            f"factor={item['factor']:.6g}{suffix}"
        )
        species_seen.add(item["species"])
    for species in species_seen:
        value = next(item["global_net_residual"] for item in records
                     if item["species"] == species)
        lines.append(
            f"  species={species} global_net_residual={value:.6g}"
        )
    _append_boundary_report(path, lines)


BOUNDARY_MAPPING_VERSION = 9


def _file_fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_fingerprints(eirene_path, genex_path):
    """Fingerprint restart-critical grids and warning-only triangle files."""
    mesh_path = Path(genex_interface.filepath_resolver(genex_path, "mesh.nc"))
    fingerprints = {
        "b2fgmtry": _file_fingerprint(Path(eirene_path) / "b2fgmtry"),
        "genex_mesh": _file_fingerprint(mesh_path),
        "triangular_mesh": {
            name: _file_fingerprint(Path(eirene_path) / name)
            for name in ("fort.33", "fort.34", "fort.35")
        },
    }
    if fingerprints["b2fgmtry"] is None:
        raise FileNotFoundError(Path(eirene_path) / "b2fgmtry")
    if fingerprints["genex_mesh"] is None:
        raise FileNotFoundError(mesh_path)
    return fingerprints


def load_boundary_mapping(path, current_fingerprints, report_path=None,
                          allow_boundary_misalignment=False):
    """Load a restart mapping, enforcing only geometry changes that invalidate it."""
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("version") != BOUNDARY_MAPPING_VERSION:
        if report_path is not None:
            _append_boundary_report(report_path, [
                f"Ignoring obsolete boundary mapping format in {path}"
            ])
        return None
    saved_override = payload.get("mapping_report", {}).get(
        "allow_boundary_misalignment", False
    )
    if saved_override and not allow_boundary_misalignment:
        if report_path is not None:
            _append_boundary_report(report_path, [
                "Ignoring boundary mapping created with the boundary-"
                "misalignment override"
            ])
        return None
    saved = payload.get("fingerprints", {})
    for key in ("b2fgmtry", "genex_mesh"):
        if saved.get(key) != current_fingerprints.get(key):
            raise RuntimeError(
                f"Cannot reuse {path}: restart-critical {key} geometry changed"
            )
    if saved.get("triangular_mesh") != current_fingerprints.get("triangular_mesh"):
        if report_path is not None:
            _append_boundary_report(report_path, [
                "WARNING: EIRENE triangular mesh fort.33/34/35 changed since "
                "the boundary mapping was saved. The plasma-domain "
                "triangulation is expected to remain fixed by b2fgmtry; "
                "continuing because only triangles outside that domain "
                "should differ."
            ])
    if report_path is not None:
        _append_boundary_report(report_path, [
            f"Reusing boundary flux mapping from {path}"
        ])
    return payload


def validate_saved_groups(groups, eirene_count, genex_count):
    output = []
    for eids, gids in groups:
        eids = [int(value) for value in eids]
        gids = [int(value) for value in gids]
        if (not eids or not gids or max(eids) >= eirene_count
                or max(gids) >= genex_count or min(eids) < 0 or min(gids) < 0):
            raise RuntimeError("saved boundary mapping contains invalid indices")
        output.append((eids, gids))
    return output


def _component_summary(component):
    return {
        "name": component.name,
        "kind": component.kind,
        "sign": component.sign,
        "indices": component.indices.tolist(),
        "centres": component.centres.tolist(),
        "areas": component.areas.tolist(),
    }


def component_from_summary(values):
    return BoundaryComponent(
        values["name"], values["kind"],
        np.asarray(values["indices"], dtype=int), float(values["sign"]),
        np.asarray(values["centres"], dtype=float),
        np.asarray(values["areas"], dtype=float),
    )


def save_boundary_mapping(path, fingerprints, groups, mapping_report,
                          eirene_components, genex_components,
                          report_path=None):
    payload = {
        "version": BOUNDARY_MAPPING_VERSION,
        "fingerprints": fingerprints,
        "groups": [[list(eids), list(gids)] for eids, gids in groups],
        "mapping_report": mapping_report,
        "eirene_components": [
            _component_summary(component) for component in eirene_components
        ],
        "genex_components": [
            _component_summary(component) for component in genex_components
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if report_path is not None:
        _append_boundary_report(report_path, [
            f"Saved boundary flux mapping to {path}"
        ])


def write_boundary_mapping_failure(path, fingerprints, error,
                                   eirene_components, genex_components,
                                   report_path=None):
    payload = {
        "version": BOUNDARY_MAPPING_VERSION,
        "error": str(error),
        "fingerprints": fingerprints,
        "eirene_components": [
            _component_summary(component) for component in eirene_components
        ],
        "genex_candidate_components": [
            _component_summary(component) for component in genex_components
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if report_path is not None:
        _append_boundary_report(report_path, [
            f"Wrote failed boundary mapping report to {path}: {error}"
        ])


def write_boundary_mapping_report(report, path, report_path):
    lines = [f"Boundary mapping: {path}",
        "  EIRENE surfaces={eirene_component_count}  "
        "GENE-X candidates={genex_candidate_component_count}  "
        "matched_faces={genex_matched_face_count}  "
        "excluded_faces={genex_excluded_face_count}  "
        "rejected={rejected_component_count}  tolerance={distance_tolerance:.6e} m"
        .format(**report)]
    if report.get("fallback_match_count", 0):
        lines.append(
            f"  fallback nearest-face matches={report['fallback_match_count']} "
            f"(limit={report['fallback_distance_limit']:.6e} m)"
        )
    if report.get("allow_boundary_misalignment", False):
        lines.append(
            "  WARNING: boundary-misalignment distance errors are disabled"
        )
    _append_boundary_report(report_path, lines)

def prepare_genex_ion_temperatures(
    genex_fields_2D,
    genex_species,
    grid_r,
    grid_z,
    compute,
    sparse_temperature_handler,
    field_mask=None,
):
    """Prepare GENE-X ion temperature to be used as source temperature.
    Nearest-fill GENE-X ion temperatures from any usable samples."""
    points = np.column_stack([grid_r[compute], grid_z[compute]])
    temperatures = {}

    for species in genex_species:
        if species.is_electron:
            continue
        temperature = np.asarray(
            genex_fields_2D["Ttot"][species.name]
        ).reshape(-1)
        if field_mask is not None:
            field_mask = np.asarray(field_mask, dtype=bool).reshape(-1)
            if temperature.size != field_mask.size:
                raise ValueError(
                    f"GENE-X temperature size for '{species.name}' "
                    "does not match the compute-cell mask"
                )
            temperature = temperature[field_mask]
        if temperature.size != points.shape[0]:
            raise ValueError(
                f"GENE-X temperature size for '{species.name}' "
                f"does not match the active grid"
            )
        # Density is deliberately not consulted here. GENE-X owns the
        # validity of zero-density compute cells; this step only fills missing
        # temperature samples when at least one finite, nonzero value exists.
        available = np.isfinite(temperature) & (temperature != 0)
        availability = available.astype(float)
        handled = sparse_temperature_handler(
            temperature,
            availability,
            points,
            threshold=1.0,
        )
        if handled is not None:
            temperatures[species.name] = handled

    return temperatures

def interpolate_all_moments(gmtry, tri, genex_out, radial_mask, poloidal_mask,
                            field_mask=None):
    """ Interpolate GENE-X data onto plasma grid used by Eirene"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for field, field_block in genex_out.items():
        for species, value in field_block.items():
            arr = value.data
            if dask.is_dask_collection(arr):
                raise TypeError(
                    "Interpolation requires materialized GENE-X fields"
                )
            arr = np.asarray(arr).reshape(-1)
            if field_mask is not None:
                active = np.asarray(field_mask, dtype=bool).reshape(-1)
                if arr.size != active.size:
                    raise ValueError(
                        f"GENE-X field '{field}/{species}' size does not "
                        "match the compute-cell mask"
                    )
                arr = arr[active]
            if field.endswith("ax"):
                ind = [0,2]
                mask = poloidal_mask
            elif field.endswith("ay"):
                ind = [2,3]
                mask = radial_mask
            else:
                ind = [0,1,2,3]
                mask = np.zeros_like(poloidal_mask, dtype=bool)
            out[field][species] = interp_moments(gmtry, tri, arr, ind)
            out[field][species][mask] = 0
    return out

def normalize_genex_params(params):
    """ Strip whitespace from parameter keys"""
    ps = params.get("params_species", {})
    if "names" in ps:
        ps["names"] = [n.strip() for n in ps["names"]]
    return params

def interpolate_all_sources_wrapper(
    tria,
    source_dict,
    grid_r,
    grid_z,
    compute,
    method="linear",
    fill_mode="constant",
    fill_value=0.0,
    genex_electrons="ELECTRONS",
    mapper=None,
    diagnostics=None,
):
    """
    Wrapper around interpolate_all_sources that:
      1) Computes only on a subset of grid points
      2) Expands results back to full grid with zeros elsewhere
      3) Optionally renames the ELECTRONS species key
    """

    # Subselect grid
    r_sub = grid_r[compute]
    z_sub = grid_z[compute]

    # Call original function (unchanged)
    interpolation_options = dict(
        method=method, fill_mode=fill_mode, fill_value=fill_value
    )
    if mapper is not None:
        interpolation_options["mapper"] = mapper
    if diagnostics is not None:
        interpolation_options["diagnostics"] = diagnostics
    sub_out = interpolate_all_sources(
        tria, source_dict, r_sub, z_sub, **interpolation_options
    )

    # Prepare full output structure
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    n_full = len(grid_r)

    for mom, mom_block in sub_out.items():
        for species, species_block in mom_block.items():
            # Handle species renaming
            target_species = (
                genex_electrons if species == "ELECTRONS" else species
            )

            for source, values in species_block.items():
                # values shape assumed (1, n_sub)
                vals_sub = values.reshape(-1)

                vals_full = np.zeros(n_full, dtype=vals_sub.dtype)
                vals_full[compute] = vals_sub

                out[mom][target_species][source] = vals_full.reshape(1, n_full)

    return out

def interpolate_temperature_values(
    tria,
    temperature_values,
    grid_r,
    grid_z,
    compute,
    *,
    direct_values=None,
    method="linear",
    fill_mode="constant",
    fill_value=0.0,
):
    """Interpolate temperatures, bypassing interpolation for direct values."""
    direct_values = direct_values or {}
    r_sub = grid_r[compute]
    z_sub = grid_z[compute]
    n_full = len(grid_r)
    out = {}

    for label, values in temperature_values.items():
        if label in direct_values:
            values_sub = np.asarray(direct_values[label]).reshape(-1)
            if values_sub.size != np.count_nonzero(compute):
                raise ValueError(
                    f"Direct temperature '{label}' has {values_sub.size} "
                    f"values, expected {np.count_nonzero(compute)}"
                )
        else:
            values_sub = interpolate_source(
                tria,
                values,
                r_sub,
                z_sub,
                method=method,
                fill_mode=fill_mode,
                fill_value=fill_value,
            ).reshape(-1)

        values_full = np.zeros(n_full, dtype=values_sub.dtype)
        values_full[compute] = values_sub
        out[label] = values_full.reshape(1, n_full)

    return out

def backup_eirene_files(eirene_path, index):
    """ Backup a single iteration's eirene files """
    # Create directory name like eirene_sources_000000
    dest_dir = eirene_path / Path(f"eirene_sources_{index:06d}")
    dest_dir.mkdir(exist_ok=True)

    # Match:
    #   fort.???  -> exactly 3 chars after "fort."
    #   fort.4?   -> 2 chars starting with 4
    #   fort.1?   -> 2 chars starting with 1
    patterns = [
        "fort.???",
        "fort.4?",
        "fort.1?",
        f"run_{index:06d}_attempt_*.log",
        "boundary_flux_normalization.json",
    ]

    shutil.copy(Path(eirene_path) / Path("fort.31"), dest_dir)
    for pattern in patterns:
        for file_path in Path(eirene_path).glob(pattern):
            if file_path.is_file():
                shutil.move(str(file_path), dest_dir / file_path.name)

def normalize_source_filepattern(filepattern):
    """Return source file pattern without trailing index separators."""
    return str(filepattern).rstrip("_")


def source_filename(filepattern, index):
    """Return the GENE-X source filename for a base pattern and index."""
    return f"{normalize_source_filepattern(filepattern)}_{index:06d}.nc"


def next_eirene_index(eirene_path, filepattern):
    """
    Scan files of the form:
        eirene_path / f"{filepattern}_{index:06d}.nc"

    Returns:
        max_index + 1 (or 0 if no matching files exist)
    """

    eirene_path = Path(eirene_path)

    filepattern = normalize_source_filepattern(filepattern)

    # match: filepattern_000123.nc
    regex = re.compile(rf"^{re.escape(filepattern)}_(\d{{6}})\.nc$")

    max_index = -1

    for f in eirene_path.glob(f"{filepattern}_*.nc"):
        m = regex.match(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > max_index:
            max_index = idx

    return max_index + 1

def _as_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("true", "yes", "1"):
        return True
    if normalized in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_argument_parser():
    parser = argparse.ArgumentParser(prog="genex_eirene_coupling", description="Couple Gene-X and Eirene throught I/O and interpolate onto the other's grid")
    parser.add_argument("--config", type=Path,
                        help="YAML coupling configuration. If omitted, "
                             "coupling.yaml in the current directory is used "
                             "when present.")
    parser.add_argument("--pid", type=int, help="Gene-X Process ID")
    parser.add_argument("--pid-file", type=Path,
                        help="File containing the GENE-X process ID. Defaults "
                             "to genex.pid in the GENE-X run directory.")
    parser.add_argument("--SumTemp", type=_as_bool, default=True,
                        help="Sums sources over collisions if true")
    parser.add_argument("--filepattern", type=str, default="./input_sources",
                        help="File path and name (without extension) of "
                        "source file expected by GENE-X")
    parser.add_argument("--eirene_command", type=str, default="eirobjx",
                        help="Command to run in as neutral code. Primarly for "
                        "use in testing.")
    parser.add_argument("--genex_path", type=str, default=default_genex_path,
                        help="Path to GENE-X run directory")
    parser.add_argument("--eirene_path", type=str, default=default_eirene_path,
                        help="Path to Eirene run directory")
    parser.add_argument("--MAX_TIMEOUTS", type=int, default=1,
                        help="Maximum number of Eirene timeouts before GENE-X "
                        "simulation is killed")
    parser.add_argument("--eirene_time", type=int, default=200,
                        help="Number of seconds to allow Eirene to run.")
    parser.add_argument("--solpstop", type=str, default=default_stop,
                        help="Path to top of solps directory tree as expected "
                        "by Eirene. For reaction paths.")
    parser.add_argument("--genex_time_index_override", type=_as_bool, default=False,
                        help="Internal/testing only. Overrides GENE-X time selection. "
                            "Default (False) selects latest time slice. "
                            "Changing this alters coupling semantics and should "
                            "NOT be used in production runs.")
    parser.add_argument(
        "--genex-read-mode", choices=("averaged", "full"), default="averaged",
        help="Materialize toroidally averaged fields (default) or eagerly "
             "load the selected full 3-D timestep before averaging.",
    )
    parser.add_argument("--genex-read-attempts", type=int, default=6,
                        help="Maximum fresh-file attempts after HDF errors.")
    parser.add_argument("--genex-retry-delay", type=float, default=0.5,
                        help="Initial HDF retry delay in seconds.")
    parser.add_argument("--genex-retry-max-delay", type=float, default=10.0,
                        help="Maximum HDF retry delay in seconds.")
    parser.add_argument(
        "--boundary-flux-normalization", choices=("surface", "off"),
        default="surface",
        help="Conserve outgoing particle flux per matched boundary surface "
             "(default: surface).",
    )
    parser.add_argument(
        "--allow-boundary-misalignment", action="store_true",
        help="Allow nearest GENE-X/EIRENE boundary matches beyond the hard "
             "fallback distance. Intended for coarse-grid testing; all such "
             "matches are still reported. Also extends the accepted boundary "
             "normalization-factor range to at least 0.01 through 100.",
    )
    parser.add_argument(
        "--boundary-flux-factor-min", type=float, default=0.5,
        help="Smallest accepted boundary normalization factor.",
    )
    parser.add_argument(
        "--boundary-flux-factor-max", type=float, default=2.0,
        help="Largest accepted boundary normalization factor.",
    )
    parser.add_argument(
        "--boundary-flux-negligible-fraction", type=float, default=0.01,
        help="Accept a zero GENE-X outward flux when the corresponding "
             "EIRENE outward flux is at most this fraction of the largest "
             "GENE-X or EIRENE gross surface flux (default: 0.01).",
    )
    parser.add_argument(
        "--radial-flux-scaling", type=float, default=1.0,
        help="Additional multiplier for normalized radial boundary fluxes "
             "(default: 1, no additional scaling).",
    )
    return parser


def parse_coupling_args(argv=None):
    """Parse CLI arguments, loading authoritative YAML when available."""
    parser = build_argument_parser()
    cli_args = parser.parse_args(argv)
    config_path = cli_args.config
    if config_path is None:
        candidate = Path.cwd() / "coupling.yaml"
        if candidate.is_file():
            config_path = candidate
    if config_path is not None:
        try:
            args = coupling_args_from_file(config_path)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        args.config = Path(config_path).resolve()
    else:
        args = cli_args
    try:
        _validate_args(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def cli():
    """Run the GENE-X–EIRENE coupling command."""
    parser = build_argument_parser()
    args = parse_coupling_args()
    try:
        resolve_pid(args)
    except ValueError as error:
        parser.error(str(error))
    main(args)


if __name__ == "__main__":
    cli()
