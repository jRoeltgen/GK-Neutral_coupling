from collections import defaultdict
from pathlib import Path
import numpy as np
import time
import xarray as xr
import torx
from torx.specializations.genex import (
    initialize_genex_from_filepath,
    load_snaps_genex,
)
from torx.fileio import filepath_resolver
from torx.measure import (
    parallel_temperature,
    perpendicular_temperature,
    pressure_m,
    velocities_m,
    total_pressure,
    electrostatic_ExB_heat_flux,
    electric_field
)

def wait_for_genex_init(genex_path, timeout=300, poll=3):
    import time
    t0 = time.time()
    last_err = None

    while True:
        try:
            grid, equi, params, norm = initialize_genex_from_filepath(genex_path)
            # minimal sanity checks
            assert hasattr(grid, "r_u") and hasattr(grid, "z_u")
            genex_path = Path(genex_path)
            r_all, z_all, not_compute = get_grid_with_ghost_filler(genex_path)
            return grid, equi, params, norm, r_all, z_all, not_compute
        except Exception as e:
            last_err = e

        if time.time() - t0 > timeout:
            raise TimeoutError(f"GENE-X init not ready: {last_err}")

        time.sleep(poll)

def get_grid_with_ghost_filler(directory_path):
    grid_group = xr.open_dataset(
        filepath_resolver(directory_path, "mesh.nc"),
        group = "RZ_grid"
    )

    if("x" in grid_group):
        R_str = "x"
        Z_str = "y"
    else:
        R_str = "R"
        Z_str = "Z"

    if("dim_RZ_grid_phi" in grid_group[R_str].dims):
        phi_string = "dim_RZ_grid_phi"
    elif("dim_phi" in grid_group[R_str].dims):
        phi_string = "dim_phi"
    else:
        raise NotImplementedError(
            "Expected phi dimension 'dim_phi' or 'dim_RZ_grid_phi'."
        )

    not_ghost = grid_group["not_ghost"].isel({phi_string: 0})
    not_filler = grid_group["not_filler"].isel({phi_string: 0})

    compute = (not_ghost.astype(bool)) & (not_filler.astype(bool))

    x_unstructured = grid_group[R_str].isel({phi_string: 0})
    y_unstructured = grid_group[Z_str].isel({phi_string: 0})

    return x_unstructured, y_unstructured, compute

def get_genex_species(params):
    names = params['params_species']['names']
    charges = params['params_species']['charges']

    all_species = []
    for idx, sp in enumerate(names):
        if len(sp.strip())>0:
            all_species.append(species(sp.strip(), charges[idx]))

    return all_species

class species:
    def __init__(self, name, charge):
        self.name = name
        self.charge = charge
        self.is_electron = charge<0

def load_latest_genex_fields(gpath, all_spec, grid, equi, params, norm,
                             time_index, timeout):
    """
    Load latest GENE-X data and compute derived quantities.
    """
    with xr.set_options(file_cache_maxsize=1):
        spec = []
        electrons = []
        ions = []
        for s in all_spec:
            spec.append(s.name)
            if s.is_electron:
                electrons.append(s.name)
            else:
                ions.append(s.name)
        if (len(ions)>1):
            raise ValueError("Torx library functions not generalized to 2+ ion species")
        if (len(electrons)>1):
            raise ValueError("Multiple assumed electrons (charge<0) given")

        NO_SPECIES = "N/A"
        EXPECTED_FIELDS = {"es_pot", "n", "u_par", "E_par", "E_perp",
            "Q_par", "Q_perp", "Ttot", "u_phi", "u_rad", "q_es", "pr",
        }
        VALID_SPECIES = set(spec) | {NO_SPECIES}
        out = defaultdict(dict)

        def set_field(field, species, value):
            if field not in EXPECTED_FIELDS:
                raise KeyError(f"Unknown field '{field}'")
            if species not in VALID_SPECIES:
                raise KeyError(f"Invalid Species {species}. Allowed: {VALID_SPECIES}")
            out[field][species] = value

        def get_field(field, species):
            if field not in out:
                raise KeyError(f"Field '{field}' not initialized")
            if species not in out[field]:
                raise KeyError(f"Species '{species}' missing for field '{field}'")
            return out[field][species]

        def load_field(field_name, species, norm_value):
            da = load_snaps_genex(gpath, species, field_name).isel({"tau": time_index})
            da.attrs["norm"] = norm_value
            if species is None:
                set_field(field_name, NO_SPECIES, da)
            else:
                set_field(field_name, species, da)
            return da

        tau_arr = wait_until_genex_stable(
            gpath,
            spec,
            check_interval=0.5,
            stable_time=2.0,
            timeout=timeout,
        )
        stable_idx = tau_arr.size - 1
        if time_index < 0:
            time_index = stable_idx
        elif stable_idx < time_index:
            raise ValueError(f"time_index {time_index} not less than or equal "
                        f"to last stable index found ({stable_idx})")

        load_field("es_pot", None, (norm.Te0 / norm.elementary_charge).to("V"))
        efield = electric_field(grid, get_field("es_pot", NO_SPECIES))

        radial_vExB = velocities_m.ExB_velocity(efield, grid=grid, equi=equi,
                                                norm=norm, component="radial")

        for s in spec:
            load_field("n", s, norm.n0)
            load_field("u_par", s, norm.c_s0)
            load_field("E_par", s, norm.Te0 * norm.n0)
            load_field("E_perp", s, norm.Te0 * norm.n0)
            load_field("Q_par", s, norm.Ti0 * norm.n0 * norm.c_s0)
            load_field("Q_perp", s, norm.Ti0 * norm.n0 * norm.c_s0)
            set_field("Ttot",s, calculate_temperatures(params, norm, s,
                                    get_field("n",s), get_field("u_par",s),
                                    get_field("E_par",s), get_field("E_perp",s))[0])

            upar = grid.vector_to_matrix(get_field("u_par",s))
            uvec = velocities_m.parallel_ion_velocity_vector(grid, equi, upar)
            radial_vDia = velocities_m.diamagnetic_velocity(get_field("Ttot", s),
                                                            grid=grid, equi=equi,
                                                            norm=norm, spec=s,
                                                            component="radial")
            set_field("u_phi", s, grid.matrix_to_vector(uvec.sel(vector='ePhi')))

            set_field("u_rad", s, radial_vExB + radial_vDia)
            set_field("q_es", s, electrostatic_ExB_heat_flux(grid, equi, norm,
                                                get_field("es_pot", NO_SPECIES),
                                                get_field("E_par",s),
                                                get_field("E_perp",s)))
        set_field("pr", NO_SPECIES, total_pressure(get_field("n",electrons[0]),
                                        get_field("Ttot", electrons[0]),
                                        get_field("Ttot", ions[0]), norm))

        return out, tau_arr[time_index]


def calculate_temperatures(params, norm, spec, n, u_par, E_par, E_perp):
    n.attrs["norm"] = norm.n0
    u_par.attrs["norm"] = norm.c_s0
    E_par.attrs["norm"] = norm.Te0 * norm.n0
    Tpar = parallel_temperature(params, norm, n, E_par, u_par, spec)
    Tperp = perpendicular_temperature(params, norm, n, E_perp)
    Ttot = Tpar + Tperp
    Ttot.attrs["norm"] = Tpar.attrs["norm"]
    return Ttot, Tpar, Tperp

# Torx might have one implemented
def toroidal_avg(genex_out):
    out = defaultdict(dict)
    # Toroidal average
    for key in genex_out.keys():
        for s in genex_out[key].keys():
            out[key][s] = genex_out[key][s].mean(dim="phi", keep_attrs=True)
    return out

def unnormalize(var):
    return var*var.norm

def get_mom_paths(genex_path):
    paths = []
    for d in genex_path.iterdir():
        f = d / "mom_2d.nc"
        if f.is_file():
            paths.append(f)
    paths.sort()

    if paths:
        return paths
    single = genex_path / "mom_2d.nc"
    return [single] if single.exists() else []

def get_tau_for_vars(genex_path, species):
    vars_to_check = ["n", "Q_perp"]

    taus = []
    for s in species:
        for var in vars_to_check:
            try:
                da = load_snaps_genex(genex_path, s, var)
                tau = da.coords["tau"].values
                del da
                taus.append(tau)
            except Exception:
                return None  # mid-write or unavailable

    # consistency check
    ref = taus[0]
    for t in taus[1:]:
        if len(t) != len(ref) or (t != ref).any():
            return None

    return ref

def wait_until_genex_stable(
    genex_path,
    species_list,
    check_interval=0.5,
    stable_time=2.0,
    timeout=300.0,
):
    start = time.time()

    last_tau = None
    stable_start = None

    while True:
        if time.time() - start > timeout:
            raise RuntimeError("GENE-X did not stabilize")
        paths = get_mom_paths(genex_path)
        if not paths:
            time.sleep(check_interval)
            continue
        tau = get_tau_for_vars(genex_path, species_list)
        if tau is None:
            stable_start = None
            time.sleep(check_interval)
            continue
        if tau.size < 1:
            continue
        if last_tau is not None:
            if tau.size != last_tau.size:
                stable_start = None
            elif last_tau is not None and np.allclose(tau, last_tau, atol=1e-10):
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_time:
                    return tau  # stable and consistent
            else:
                stable_start = None

        last_tau = tau
        time.sleep(check_interval)