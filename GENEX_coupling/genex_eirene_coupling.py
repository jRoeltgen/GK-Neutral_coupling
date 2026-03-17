import subprocess
import psutil
import numpy as np
import time
from collections import defaultdict
from pathlib import Path
from netCDF4 import Dataset
import argparse



# Should remove when finalized
import pdb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(eirene_path, b2_path, genex_path, args):
    eirene_time = 600 # 10 minutes for 1 step
    # Read meshes
    edat = eireneIO.eirene()
    tria = triangle_mesh.triangle_mesh(eirene_path)
    b2dat = B2IO.B2(b2_path)
    edat.triangle_mesh = tria
    nx,ny = b2dat.gmtry["vol"].shape
    if (eirene_path / Path("fort.44")).is_file():
        edat.read_ft44(eirene_path / Path("fort.44"))
        ns = edat.fort44["meta"]["npls"]
    else:
        ns = 2
    edat.read_ft31(eirene_path / Path("fort.31"),nx,ny,ns)
    tria.calc_incenter()
    grid, equi, params, norm = initialize_genex_from_filepath(genex_path)
    grid_r = grid.r_u
    grid_z = -grid.z_u
    ntria = tria.cells.shape[0]

    genex_out = {"density":{}, "u_par":{}, "Tpar":{},
                 "Tperp":{}, "Ttot":{}, "es_pot":{},
                 "q_es":{}, "Q_par":{}, "Q_perp":{},
                 "u_phi":{}, "u_rad":{}}
    interp_values = genex_out
    electron_spec = ["electrons"]
    ion_spec = ["ions"]
    spec = ion_spec + electron_spec
    # spec = get_genex_species(genex_path)
    last_tau = -1
    while psutil.pid_exists(args.pid):
        # Read in data
        genex_out["es_pot"]["N/A"] = load_snaps_genex(genex_path, None,
                                                      "es_pot").isel({"tau":-1})
        tau = genex_out["es_pot"]["N/A"].tau.values
        if tau <= last_tau:
            time.sleep(5)
            continue
        genex_out["es_pot"]["N/A"].attrs["norm"] = (norm.Te0 / norm.elementary_charge).to("V")
        efield = electric_field(grid, genex_out["es_pot"]["N/A"])
        radial_vExB = velocities_m.ExB_velocity(efield, grid=grid,
                                                equi=equi, norm=norm, component="radial")
        for s in spec:
            genex_out["density"][s] = load_snaps_genex(genex_path, s, "n").isel({"tau":-1})
            genex_out["density"][s].attrs["norm"] = norm.n0
            genex_out["u_par"][s] = load_snaps_genex(genex_path, s, "u_par").isel({"tau":-1})
            genex_out["u_par"][s].attrs["norm"] = norm.c_s0
            upar = grid.vector_to_matrix(genex_out["u_par"][s])
            uvec = velocities_m.parallel_ion_velocity_vector(grid, equi, upar)
            genex_out["u_phi"][s]=grid.matrix_to_vector(uvec.sel(vector='ePhi'))
            E_par = load_snaps_genex(genex_path, s, "E_par").isel({"tau":-1})
            E_perp = load_snaps_genex(genex_path, s, "E_perp").isel({"tau":-1})
            E_par.attrs["norm"] = norm.Te0 * norm.n0
            E_perp.attrs["norm"] = norm.Te0 * norm.n0
            genex_out["Ttot"][s] = calculate_temperatures(genex_path, params, norm, s,
                                genex_out["density"][s], genex_out["u_par"][s], E_par, E_perp)[0]
            radial_vDia = velocities_m.diamagnetic_velocity(genex_out["Ttot"][s], grid=grid,
                                                equi=equi, norm=norm, spec=s, component="radial")
            genex_out["u_rad"][s] = radial_vExB + radial_vDia
            genex_out["q_es"][s] = electrostatic_ExB_heat_flux(grid, equi, norm,
                                                genex_out["es_pot"]["N/A"], E_par, E_perp)
            genex_out["Q_par"][s] = load_snaps_genex(genex_path, s, "Q_par").isel({"tau":-1})
            genex_out["Q_par"][s].attrs["norm"] = norm.Ti0 * norm.n0 * norm.c_s0
            genex_out["Q_perp"][s] = load_snaps_genex(genex_path, s, "Q_perp").isel({"tau":-1})
            genex_out["Q_perp"][s].attrs["norm"] = norm.Ti0 * norm.n0 * norm.c_s0

        # Toroidal average and interpolate
        for key in genex_out.keys():
            for s in genex_out[key].keys():
                genex_out[key][s] = genex_out[key][s].mean(dim="phi", keep_attrs=True)
                interp_values[key][s] = interp_moments(b2dat.gmtry, grid_r, grid_z,
                                                       genex_out[key][s], key)

        write_fort31(edat, interp_values, norm, electron_spec, ion_spec)
        run_eirene(eirene_time)

        # interpolate eirene moments
        edat.load_extra_forts(eirene_path)
        norm_sources = {}
        interp_source = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        if args.SumTemp:
            for moment, sp_dict in edat.sources.items():
                norm_sources[moment] = {}
                for species, data in sp_dict.items():
                    norm_sources[moment][species] = {None: data}
        else:
            pass
        for moment, sp_dict in norm_sources.items():
            for species, temp_dict in sp_dict.items():
                for temp, data in temp_dict.items():
                    interpolated = interpolate_source(tria, data, grid_r, grid_z)
                    interp_source[moment][species][temp] = interpolated

        start = 0
        inc = interp_source["particle"]["ELECTRONS"].shape[0]
        last = inc
        with Dataset("eirene_sources.nc", "w", format="NETCDF4") as nc:
            dim_RZ = nc.createDimension("dim_RZ", grid_r.shape[0])
            dim_phi = nc.createDimension("dim_phi", 1)
            grp = {}
            for s in interp_source["particle"].keys():
                grp[s] = nc.createGroup(s)
                for source in interp_source.keys():
                    var = grp[s].createVariable(source, "f8", ("dim_RZ"))
                    if not s in edat.units[source].keys():
                        edat.units[source][s] = "N/A"
                    var.units = edat.units[source][s]
                    #var[:] = interp_source[source][s]
                    source = np.arange(start, last, dtype="f8")
                    #pdb.set_trace()
                    var[:] = source
                    start = start+inc
                    last = last+inc
        break

def calculate_temperatures(genex_path, params, norm, spec, n, u_par, E_par, E_perp):
    n.attrs["norm"] = norm.n0
    u_par.attrs["norm"] = norm.c_s0
    E_par.attrs["norm"] = norm.Te0 * norm.n0
    Tpar = parallel_temperature(params, norm, n, E_par, u_par, spec)
    Tperp = perpendicular_temperature(params, norm, n, E_perp)
    Ttot = Tpar + Tperp
    Ttot.attrs["norm"] = Tpar.norm
    return Ttot, Tpar, Tperp

def toroidal_avg(dataArray):
    return np.mean(dataArray.data,axis=0)

def run_eirene(Eirene_time):
    output_file = "run.log"
    try:
        with open(output_file, "w") as outfile:
            result = subprocess.run("ls", stdout=outfile, stderr=subprocess.STDOUT,
                                    text=True, timeout=Eirene_time)
        if result.returncode != 0:
            print(f"❌eirobjx failed with return code {result.returncode}")
            raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        print(f"Error: eirobjx timed out after {Eirene_time} seconds.")
        raise SystemExit(1)
    except FileNotFoundError:
        print("⚠️ Error: eirobjx command not found. Make sure it’s in your PATH.")
        raise SystemExit(1)
    except Exception as e:
        print(f"⚠️  Unexpected error running eirobjx: {e}")
        raise SystemExit(1)

def write_fort31(edat, genex_data, norm, eorder, iorder):
    # vExB = ExB_velocity() see analyze_moments.py
    nions = len(iorder)
    if nions>1:
        feinsum = 'ijk,ij->ijk'
    else:
        feinsum = 'ij,ij->ij'
    upar = dict_to_array(genex_data["u_par"], iorder, edat.fort31["ua"].shape)
    upol = np.einsum(feinsum,upar,edat.fort31["bb"][:,:,0])
    urad = dict_to_array(genex_data["u_rad"], iorder, edat.fort31["ua"].shape)


    edat.fort31["na"] = dict_to_array(genex_data["density"], iorder, edat.fort31["na"].shape)
    edat.fort31["up"] = upol
    edat.fort31["vv"] = urad
    edat.fort31["ww"] = dict_to_array(genex_data["u_phi"], iorder, edat.fort31["ww"].shape)
    edat.fort31["te"] = dict_to_array(genex_data["Ttot"], eorder, edat.fort31["te"].shape)
    edat.fort31["ti"] = dict_to_array(genex_data["Ttot"], iorder, edat.fort31["ti"].shape)
    edat.fort31["ua"] = upar
    # pitch angle - constant in time
    edat.fort31["fnax"] = np.einsum(feinsum,upol,edat.fort31["na"])
    edat.fort31["fnay"] = np.einsum(feinsum,urad,edat.fort31["na"])
    edat.fort31["uadia"] = np.zeros((edat.fort31["uadia"].shape))
    edat.fort31["vadia"] = np.zeros((edat.fort31["vadia"].shape))
    edat.fort31["po"] = dict_to_array(genex_data["es_pot"], None)

    # pressure and heat fluxes only used for eirene output/graphics
    edat.fort31["pr"] = total_pressure(genex_data["density"][eorder[0]],
                            genex_data["Ttot"][eorder[0]], genex_data["Ttot"][iorder[0]], norm).values
    qepar = dict_to_array(genex_data["Q_par"], eorder, edat.fort31["fhex"].shape)
    qipar = dict_to_array(genex_data["Q_par"], iorder, edat.fort31["fhix"].shape)
    if nions>1:
        qipar = np.mean(qipar,axis=2)
    edat.fort31["fhix"] = qipar * edat.fort31["bb"][:,:,0] # Poloidal ion heat flux
    # Radial ion heat flux -
    edat.fort31["fhex"] = qepar * edat.fort31["bb"][:,:,0] # Poloidal electron heat flux
    # Radial electron heat flux

    edat.write_ft31("fort.31")

def dict_to_array(dict_in, order, dim=None):
    if not order:
        return next(iter(dict_in.values()))
    num_keys = len(order)
    nx,ny = dict_in[order[0]].shape
    arr = np.zeros((nx,ny,num_keys))
    for i,key in enumerate(order):
        arr[:,:,i] = dict_in[key]
    arr = np.squeeze(arr)
    if dim and dim != arr.shape:
        raise ValueError(
            f"Dimensions of new fort.31 field {arr.shape} don't match original: {dim}")
    return arr

def get_genex_species(genex_path):
    nml = f90nml.read('params_in.txt')
    species = nml['params_species']['names']
    return species

def write_sources_nc(filename, sources, temperature_values=None, write_temperature=False):
    """
    Write sources to NetCDF with structure:

    temperature_<id>/
        attr: temp_id
        temperature (optional)
        species_<name>/
            mom_<moment>

    Parameters
    ----------
    filename : str
        Output NetCDF filename

    sources : dict
        sources[moment][species][temperature] -> ndarray

    temperature_values : dict, optional
        temperature_values[temp_key] -> temperature value/array

    write_temperature : bool
        Whether to write the temperature variable
    """

    moments = list(sources.keys())
    species = list(next(iter(sources.values())).keys())

    # collect temperature keys
    temps = sorted({
        t
        for m in sources
        for sp in sources[m]
        for t in sources[m][sp]
    })

    with Dataset(filename, "w") as nc:

        # global dimensions
        nc.createDimension("RZ", None)
        nc.createDimension("phi", None)

        for temp in temps:

            # group name
            grp_name = f"temperature_{temp if temp is not None else 'implicit'}"
            temp_grp = nc.createGroup(grp_name)

            # ALWAYS write temp_id attribute
            temp_grp.temp_id = "implicit" if temp is None else str(temp)

            # optional temperature variable
            if write_temperature and temp is not None and temperature_values is not None:

                temp_var = temp_grp.createVariable("temperature", "f8")
                temp_var[:] = temperature_values[temp]

            # species groups
            for sp in species:

                sp_grp = temp_grp.createGroup(f"species_{sp}")

                for mom in moments:

                    temp_dict = sources[mom].get(sp, {})

                    if temp not in temp_dict:
                        continue

                    data = temp_dict[temp]

                    var = sp_grp.createVariable(
                        f"mom_{mom}",
                        "f8",
                        ("RZ", "phi"),
                    )

                    var[:] = data

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
