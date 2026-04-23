from collections import defaultdict
import f90nml
from pathlib import Path
import numpy as np
import torx
from torx.specializations.genex import (
    initialize_genex_from_filepath,
    load_snaps_genex,
    load_trace_genex,
)
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
            return grid, equi, params, norm
        except Exception as e:
            last_err = e

        if time.time() - t0 > timeout:
            raise TimeoutError(f"GENE-X init not ready: {last_err}")

        time.sleep(poll)

def get_genex_species(params):
    names = params['params_species']['names']
    charges = params['params_species']['charges']

    all_species = []
    for idx, sp in enumerate(names):
        if len(sp.strip())>0:
            all_species.append(species(sp, charges[idx]))

    return all_species

class species:
    def __init__(self, name, charge):
        self.name = name
        self.charge = charge
        self.is_electron = charge<0

def load_latest_genex_fields(gpath, all_spec, grid, equi, params, norm,
                             time_index):
    """
    Load latest GENE-X data and compute derived quantities.
    """
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
    set_field("pr", s, total_pressure(get_field("n",electrons[0]),
                                      get_field("Ttot", electrons[0]),
                                      get_field("Ttot", ions[0]), norm))

    tau = load_trace_genex(gpath, spec[0], "n").tau
    tau.attrs["norm"] = (norm.R0 / norm.c_s0).to("s")
    time = (tau * tau.norm)[time_index].data.magnitude

    return out, time


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