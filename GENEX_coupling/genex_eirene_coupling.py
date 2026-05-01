
import psutil
from pathlib import Path
from collections import defaultdict
from interpolate import (interpolate_all_sources, interp_moments,
                         build_triangulation)
from eirene_interface import (eirene_interface, run_eirene, prepare_fort31,
                              status)
import genex_interface
from write_netcdf import write_sources_nc
from os import (killpg, replace)
from signal import SIGTERM
from time import sleep
from types import SimpleNamespace
import numpy as np
import dask
import numbers
import pdb

default_eirene_path = Path("/pscratch/sd/j/jonroelt/tcv_genex_coupling/coupled_run/")
default_genex_path = Path("/pscratch/sd/j/jonroelt/source_testing/3D_source/branch_all_source_file_params/")

def main(args, deps=None):
    if deps is None:
        deps = SimpleNamespace(
            run_eirene=run_eirene,
            write_nc=write_sources_nc,
            killpg=killpg,
            sleep=sleep,
            replace=replace,
            pid_exists=psutil.pid_exists,
        )
    print("Entered main ...", flush=True)
    eirene_path = Path(args.eirene_path)
    genex_path = Path(args.genex_path)
    edat, b2dat = eirene_interface(eirene_path, eirene_path)
    print("Next genex init...", flush=True)
    grid, equi, params, norm, r_all, z_all, compute = genex_interface.wait_for_genex_init(genex_path)
    print("Normalize params", flush=True)
    params = normalize_genex_params(params)
    print("Getting genex species...", flush=True)
    genex_species = genex_interface.get_genex_species(params)
    print("Getting genex electron name", flush=True)
    genex_electrons = get_genex_electron_name(genex_species)
    check_species_consistency(edat.species_names["bulk_ions"], genex_species)
    grid_r = np.asarray(r_all*norm["R0"])
    grid_z = -np.asarray(z_all*norm["R0"])
    index = 0
    MAX_TIMEOUTS = args.MAX_TIMEOUTS
    print("args.genex_time_index_override is ",args.genex_time_index_override)
    if args.genex_time_index_override:
        time_index = 0
        ntau = 40
    else:
        time_index = -1
    last_tau = -1
    # Precompute triangulation
    print("Build triangulation...", flush=True)
    tri = build_triangulation(grid_r[compute], grid_z[compute])
    print("Start main loop...", flush=True)
    timeout = 600
    while deps.pid_exists(args.pid):
        genex_fields, tau = genex_interface.load_latest_genex_fields(genex_path,
                genex_species, grid, equi, params, norm, time_index, timeout)
        timeout = 300
        print("tau=",tau,flush=True)
        if (tau <= last_tau):
            deps.sleep(5)
            continue
        unnormalize_all(genex_fields)
        print("Toroidal avg")
        genex_fields_2D = genex_interface.toroidal_avg(genex_fields)
        print("Interpolate all moments:")
        interpolated = interpolate_all_moments(b2dat.gmtry, tri,
                                               genex_fields_2D)
        del genex_fields_2D, genex_fields
        print("prepare fort 31")
        prepare_fort31(edat, interpolated, genex_electrons,
                     edat.species_names["bulk_ions"])
        print("write fort 31")
        edat.write_ft31(eirene_path / Path("fort.31"))

        print(f"[{index}] Running EIRENE", flush=True)
        num_timeouts = 0
        eirene_time = args.eirene_time
        while num_timeouts<MAX_TIMEOUTS:
            eirene_status = deps.run_eirene(eirene_time, eirene_path=eirene_path)
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

        if args.SumTemp:
            sources = edat.sources
        else:
            raise NotImplementedError("SumTemp=False not implemented")

        interp_sources = interpolate_all_sources_wrapper(
            edat.triangle_mesh, sources,
            grid_r, grid_z, compute,
            genex_electrons=genex_electrons,
        )

        filename = args.filepattern + f"{index:06d}" + ".nc"
        filename_tmp = filename + ".tmp"
        print(f"[{index}] Writing {filename_tmp}", flush=True)
        deps.write_nc(filename_tmp, interp_sources, grid_r.size)
        deps.replace(filename_tmp, filename)
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

def interpolate_all_moments(gmtry, tri, genex_out):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    flattened = []
    keys = []

    for field, field_block in genex_out.items():
        for species, value in field_block.items():
            flattened.append(value.data)   # keep as dask array
            keys.append((field, species))

    computed = dask.compute(*flattened)
    for (field, species), arr in zip(keys, computed):
        if field == "poloidal_fluxes": # Not currently used
            ind = [0,2]
        elif field == "radial_fluxes": # Not currently used
            ind = [2,3]
        else:
            ind = [0,1,2,3]
        out[field][species] = interp_moments(gmtry, tri, arr, ind)
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
    parser.add_argument("--genex_time_index_override", type=bool, default=False,
                        help="Internal/testing only. Overrides GENE-X time selection. "
                            "Default (False) selects latest time slice. "
                            "Changing this alters coupling semantics and should "
                            "NOT be used in production runs.")
    args = parser.parse_args()
    main(args)
