
import psutil
from pathlib import Path
from collections import defaultdict
from interpolate import (interpolate_all_sources, interpolate_source, interp_moments,
                         build_triangulation)
from eirene_interface import (eirene_interface, run_eirene, prepare_fort31,
                              get_temperatures, status)
from temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
)
import genex_interface
import SourcePostProcessor as SPP
from write_netcdf import write_sources_nc
from os import (killpg, replace)
from signal import SIGTERM
from time import sleep
from types import SimpleNamespace
import numpy as np
import dask
import shutil
import argparse
import re
import faulthandler
import scipy.constants as pyconst

default_eirene_path = Path("/pscratch/sd/j/jonroelt/tcv_genex_coupling/coupled_run/")
default_genex_path = Path("/pscratch/sd/j/jonroelt/source_testing/3D_source/branch_all_source_file_params/")
default_stop = Path("/global/cfs/cdirs/m2116/solps-iter")

def main(args, deps=None):
    faulthandler.enable(all_threads=True)
    if deps is None:
        deps = SimpleNamespace(
            run_eirene=run_eirene,
            write_nc=write_sources_nc,
            killpg=killpg,
            sleep=sleep,
            replace=replace,
            pid_exists=psutil.pid_exists,
            collision_mappers=None,
            sparse_temperature_handler=default_sparse_temperature_handler,
            pseudo_temperature_handler=(
                default_single_species_pseudo_temperature_handler
            ),
        )
    dask.config.set(scheduler="synchronous")
    eirene_path = Path(args.eirene_path)
    genex_path = Path(args.genex_path)
    edat, b2dat, pol_mask, rad_mask = eirene_interface(eirene_path, eirene_path)
    grid, equi, params, norm, r_all, z_all, compute = genex_interface.wait_for_genex_init(genex_path)
    params = normalize_genex_params(params)
    genex_species = genex_interface.get_genex_species(params)
    genex_electrons = get_genex_electron_name(genex_species)
    check_species_consistency(edat.species_names["bulk_ions"], genex_species)
    grid_r = np.asarray(r_all*norm["R0"])
    # Need to change this negative to function of grid._flipped_z and equi._flipped_Z
    grid_z = np.asarray(z_all*norm["R0"])
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
    tri = build_triangulation(grid_r[compute], grid_z[compute])
    timeout = 600
    while deps.pid_exists(args.pid):
        #gc.collect()

        for attempt in range(3):
            try:
                genex_fields, tau = genex_interface.load_latest_genex_fields(
                    genex_path, genex_species, grid, equi, params, norm,
                    time_index, timeout)
                break
            except RuntimeError as e:
                if "HDF error" in str(e):
                    print(f"Transient HDF error, retry {attempt+1}")
                    continue
                raise
        else:
            raise RuntimeError("Repeated NetCDF HDF errors.")

        timeout = 300
        if (tau <= last_tau):
            deps.sleep(5)
            continue
        unnormalize_all(genex_fields)
        genex_fields_2D = genex_interface.toroidal_avg(genex_fields)
        ion_temperature_values = {}
        if not args.SumTemp:
            ion_temperature_values = prepare_genex_ion_temperatures(
                genex_fields_2D,
                genex_species,
                grid_r,
                grid_z,
                compute,
                deps.sparse_temperature_handler,
            )
        interpolated = interpolate_all_moments(b2dat.gmtry, tri,
                                               genex_fields_2D, pol_mask,
                                               rad_mask)
        del genex_fields_2D, genex_fields
        prepare_fort31(edat, interpolated, genex_electrons,
                     edat.species_names["bulk_ions"])
        edat.write_ft31(eirene_path / Path("fort.31"))

        print(f"[{index}] Running EIRENE", flush=True)
        num_timeouts = 0
        eirene_time = args.eirene_time
        while num_timeouts<MAX_TIMEOUTS:
            eirene_status = deps.run_eirene(eirene_time, eirene_path=eirene_path,
                                            command=args.eirene_command)
            if eirene_status == status.SUCCESS:
                break
            elif eirene_status == status.TIMEOUT:
                num_timeouts += 1
                print(f"Eirene timeout after {eirene_time} s.")
                if num_timeouts < MAX_TIMEOUTS:
                    eirene_time *= 2
                    print(f"Re-running Eirene with {eirene_time} s.")
                else:
                    print(f"Eirene timed out {num_timeouts} time. Terminating GENE-X")
                    # Will a GENE-X checkpoint be written if this is done?
                    deps.killpg(args.pid, SIGTERM)
                    raise RuntimeError("EIRENE timed out more than max timeouts")
            elif eirene_status == status.ERROR:
                deps.killpg(args.pid, SIGTERM)
                raise RuntimeError("EIRENE failed. Exiting")
        edat.load_extra_forts(eirene_path=eirene_path, coll_to_adjust=None,
                              convert_units=True)

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
        else:
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
                compute,
                direct_values=direct_ion_temperatures,
            )
            write_temperatures = True

        interp_sources = interpolate_all_sources_wrapper(
            edat.triangle_mesh, sources,
            grid_r, grid_z, compute,
            genex_electrons=genex_electrons,
        )

        filename = args.filepattern + f"_{index:06d}" + ".nc"
        filename_tmp = filename + ".tmp"
        print(f"[{index}] Writing {filename_tmp}", flush=True)
        deps.write_nc(
            filename_tmp,
            interp_sources,
            grid_r.size,
            temperature_values=interp_temperatures,
            write_temperature=write_temperatures,
        )
        deps.replace(filename_tmp, filename)
        backup_eirene_files(eirene_path, index)
        index += 1
        last_tau = tau
        if args.genex_time_index_override:
            time_index += 1
            if ntau<time_index:
                break

def get_genex_electron_name(genex_species):
    for sp in genex_species:
        if(sp.is_electron):
            return sp.name

def check_species_consistency(eirene_species, genex_species):
    for sp in genex_species:
        if (sp.name not in eirene_species and not sp.is_electron):
            raise ValueError(f"Genex Species {sp.name} not known to Eirene. "
                             f"Eirene ion species are "
                             f"{' ,'.join(eirene_species)}.")

def unnormalize_all(genex_out):
    for field, field_block in genex_out.items():
        for species, value in field_block.items():
            genex_out[field][species] = genex_interface.unnormalize(value)


def prepare_genex_ion_temperatures(
    genex_fields_2D,
    genex_species,
    grid_r,
    grid_z,
    compute,
    sparse_temperature_handler,
):
    """Nearest-fill GENE-X ion temperatures from any usable samples."""
    points = np.column_stack([grid_r[compute], grid_z[compute]])
    temperatures = {}

    for species in genex_species:
        if species.is_electron:
            continue
        temperature = np.asarray(
            genex_fields_2D["Ttot"][species.name]
        ).reshape(-1)
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

def interpolate_all_moments(gmtry, tri, genex_out, radial_mask, poloidal_mask):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    flattened = []
    keys = []

    for field, field_block in genex_out.items():
        for species, value in field_block.items():
            flattened.append(value.data)   # keep as dask array
            keys.append((field, species))

    for k, arr in zip(keys, flattened):
        if dask.is_dask_collection(arr):
            _ = arr.compute()
    computed = dask.compute(*flattened)

    for (field, species), arr in zip(keys, computed):
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
    sub_out = interpolate_all_sources(
        tria,
        source_dict,
        r_sub,
        z_sub,
        method=method,
        fill_mode=fill_mode,
        fill_value=fill_value,
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
    ]

    shutil.copy(Path(eirene_path) / Path("fort.31"), dest_dir)
    for pattern in patterns:
        for file_path in Path(eirene_path).glob(pattern):
            if file_path.is_file():
                shutil.move(str(file_path), dest_dir / file_path.name)

def next_eirene_index(eirene_path, filepattern):
    """
    Scan files of the form:
        eirene_path / f"{filepattern}_{index:06d}.nc"

    Returns:
        max_index + 1 (or 0 if no matching files exist)
    """

    eirene_path = Path(eirene_path)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="genex_eirene_coupling", description="Couple Gene-X and Eirene throught I/O and interpolate onto the other's grid")
    parser.add_argument("--pid", type=int, help="Gene-X Process ID")
    parser.add_argument("--SumTemp", type=bool, default=True,
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
                        help="To be written. For reaction paths.")
    parser.add_argument("--genex_time_index_override", type=bool, default=False,
                        help="Internal/testing only. Overrides GENE-X time selection. "
                            "Default (False) selects latest time slice. "
                            "Changing this alters coupling semantics and should "
                            "NOT be used in production runs.")
    args = parser.parse_args()
    main(args)
