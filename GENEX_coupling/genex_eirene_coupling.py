
import psutil
from pathlib import Path
from interpolate import (interpolate_all_sources, interp_moments)
from eirene_interface import (eirene_interface, run_eirene, write_fort31)
import genex_interface
from write_netcdf import write_sources_nc

def main(eirene_path, genex_path, args):
    edat, b2dat = eirene_interface(eirene_path, eirene_path)
    spec = ["ions", "electrons"] # Get from input.dat
    grid, equi, params, norm = genex_interface.initialise_genex(genex_path)
    grid_r = grid.r_u
    grid_z = -grid.z_u
    while psutil.pid_exists(args.pid):

        genex_fields = genex_interface.load_latest_genex_fields(genex_path,
                                        spec, grid, equi, params, norm)

        interpolated = interpolate_all_moments(b2dat.gmtry, grid_r, grid_z,
                                               genex_fields)

        write_fort31(eirene_path / Path("fort.31"))

        run_eirene()

        edat.load_extra_forts(eirene_path=eirene_path)

        if args.SumTemp:
            sources = edat.sources
        else:
            pass

        interp_sources = interpolate_all_sources(edat.tria, sources, grid_r,
                                                grid_z)

        write_sources_nc(...)

def interpolate_all_moments(gmtry, grid_r, grid_z, genex_out):

    if key == "poloidal_fluxes":
        ind = [0,2]
    elif key == "radial_fluxes":
        ind = [2,3]
    else:
        ind = [0,1,2,3]
    interp_moments(gmtry, grid_r, grid_z, genex_out, ind)

if __name__ == "__main__":
    eirene_path = Path("/pscratch/sd/j/jonroelt/tcv_genex_coupling/coupled_run/")
    b2_path = Path("/pscratch/sd/j/jonroelt/tcv_genex_coupling/baserun/")
    genex_path = Path(torx.genex_xpoint_resources_dir)
    print(genex_path)
    parser = argparse.ArgumentParser(prog="genex_eirene_coupling", description="Couple Gene-X and Eirene throught I/O and interpolate onto the other's grid")
    parser.add_argument("--pid", type=int, help="Gene-X Process ID")
    parser.add_argument("--SumTemp", type=bool, default=True,
                        help="Sums sources over collisions if true")
    args = parser.parse_args()
    main(eirene_path, b2_path, genex_path, args)
