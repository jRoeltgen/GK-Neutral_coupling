
import psutil
from pathlib import Path
from collections import defaultdict
from interpolate import (interpolate_all_sources, interp_moments)
from eirene_interface import (eirene_interface, run_eirene, prepare_fort31,
                              status)
import genex_interface
from write_netcdf import write_sources_nc
from os import (killpg, replace)
from signal import SIGTERM
from time import sleep
from types import SimpleNamespace

eirene_path = Path("/pscratch/sd/j/jonroelt/tcv_genex_coupling/coupled_run/")
genex_path = Path("/pscratch/sd/j/jonroelt/source_testing/3D_source/branch_all_source_file_params/")

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
    edat, b2dat = eirene_interface(eirene_path, eirene_path)
    grid, equi, params, norm = genex_interface.initialise_genex(genex_path)
    genex_species = genex_interface.get_genex_species(genex_path)
    genex_electrons = get_genex_electron_name(genex_species)
    check_species_consistency(edat.species_names["bulk_ions"], genex_species)
    grid_r = grid.r_u
    grid_z = -grid.z_u
    index = 0
    MAX_TIMEOUTS = args.MAX_TIMEOUTS
    last_tau = -1
    while deps.pid_exists(args.pid):
        genex_fields = genex_interface.load_latest_genex_fields(genex_path,
                                        genex_species, grid, equi, params, norm)
        tau = genex_fields["es_pot"]["N/A"].coords["tau"].values[-1]
        if (tau <= last_tau):
            deps.sleep(5)
            continue
        genex_fields_2D = genex_interface.toroidal_avg(genex_fields)
        interpolated = interpolate_all_moments(b2dat.gmtry, grid_r, grid_z,
                                               genex_fields_2D)

        prepare_fort31(edat, interpolated, genex_electrons,
                     edat.species_names["bulk_ions"])

        edat.write_ft31(eirene_path / Path("fort.31"))

        print(f"[{index}] Running EIRENE")
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

        interp_sources = interpolate_all_sources(edat.tria, sources, grid_r,
                                                grid_z)

        filename = args.filepattern + f"{index:06d}" + ".nc"
        filename_tmp = filename + ".tmp"
        print(f"[{index}] Writing {filename_tmp}")
        deps.write_nc(filename_tmp, interp_sources)
        deps.replace(filename_tmp, filename)
        index += 1
        last_tau = tau

def get_genex_electron_name(genex_species):
    for sp in genex_species:
        if(sp.is_electron):
            return sp.name

def check_species_consistency(eirene_species, genex_species):
    for sp in genex_species:
        if (sp.name not in eirene_species and not sp.is_electron):
            raise ValueError(f"Genex Species {sp.name} not known to Eirene"
                             f"Eirene ion species are "
                             f"{' ,'.join(eirene_species)}")

def interpolate_all_moments(gmtry, grid_r, grid_z, genex_out):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for field, field_block in genex_out.items():
        if field == "poloidal_fluxes": # Not currently used
            ind = [0,2]
        elif field == "radial_fluxes": # Not currently used
            ind = [2,3]
        else:
            ind = [0,1,2,3]
        for species, value in field_block.items():
            out[field][species] = interp_moments(gmtry, grid_r, grid_z, value, ind)
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
    parser.add_argument("--genex_path", type=str, default=genex_path,
                        help="Path to GENE-X run directory")
    parser.add_argument("--eirene_path", type=str, default=eirene_path,
                        help="Path to Eirene run directory")
    parser.add_argument("--MAX_TIMEOUTS", type=int, default=1,
                        help="Maximum number of Eirene timeouts before GENE-X "
                        "simulation is killed")
    args = parser.parse_args()
    main(args)
