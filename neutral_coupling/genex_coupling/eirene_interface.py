from neutral_coupling.common import eirene_io as eireneIO
from neutral_coupling.common import b2_io as B2IO
from neutral_coupling.common import triangle_mesh
from neutral_coupling.common.eirene_input_parser import EireneInputParser
from pathlib import Path
import scipy.constants as pyconst
from scipy.interpolate import LinearNDInterpolator
from neutral_coupling.common.temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
)
import subprocess
import numpy as np
import os
import signal
import enum
class status(enum.IntEnum):
    SUCCESS = 0
    TIMEOUT = -1
    ERROR = -2

def eirene_interface(eirene_path, b2_path):
    edat = eireneIO.eirene()
    edat.triangle_mesh = triangle_mesh.triangle_mesh(eirene_path)
    # b2dat is only needed for the crx/cry variables
    # It can be replaced by reading the fort.30 file into python
    # To fully remove this portion from using SOLPS infrastructure only needs
    #   read/write fort.30 and some mesh generation which might also be possible
    #   in python. It can then write the fort.33/34/35 files using the
    #   triangle_mesh class. It appears the writing of input.dat is independent
    #   of the mesh (uinp is called before b2ag in triang).
    # Assumes fluxes in fort.31 already have correct elements zeroed out.
    b2dat = B2IO.B2(b2_path)

    eip = EireneInputParser(eirene_path / Path("input.dat"))
    eip.parse_species()
    edat.species_names = eip.species
    edat.masses = eip.masses
    ns = len(edat.species_names["bulk_ions"])

    edat.read_ft30(eirene_path / Path("fort.30"))

    nx = edat.plasma_gmtry["nx"]
    ny = edat.plasma_gmtry["ny"]

    edat.read_ft31(eirene_path / Path("fort.31"), nx+2, ny+2, ns)
    if edat.fort31["fnax"].ndim == 3:
        pol_mask = edat.fort31["fnax"][:,:,0] == 0
        rad_mask = edat.fort31["fnay"][:,:,0] == 0
    else:
        pol_mask = edat.fort31["fnax"] == 0
        rad_mask = edat.fort31["fnay"] == 0

    edat.triangle_mesh.calc_incenter()

    return edat, b2dat, pol_mask, rad_mask

"""Mapping between different naming schemes"""
TEMPERATURE_CLASS_INFO = {
    "a": ("atoms", "atom labels", "Tn", "ATOMS"),
    "m": ("molecules", "molecule labels", "Tm", "MOLECULES"),
    "i": ("test_ions", "ion labels", "Tti", "TEST IONS"),
}


def _temperature_from_moments(energy_density, density, momentum_squared, mass):
    """Apply the existing provisional EIRENE temperature expression."""
    internal_energy = (
        energy_density
        - np.sqrt(momentum_squared)
        + 0.5 * mass * momentum_squared * density
    )
    return np.divide(
        internal_energy,
        density,
        out=np.full_like(energy_density, np.nan, dtype=float),
        where=density > 0,
    )


def map_ion_temperatures_to_triangles(
    edat,
    ion_temperature_values,
    genex_triangulation,
    *,
    scale=pyconst.elementary_charge,
):
    """Map GENE-X ion temperatures directly onto EIRENE triangles."""
    mapped = {}
    for species, temperature in ion_temperature_values.items():
        interpolator = LinearNDInterpolator(
            genex_triangulation,
            np.asarray(temperature).reshape(-1),
            fill_value=0.0,
        )
        mapped[f"Ti_{species}"] = (
            np.asarray(interpolator(edat.triangle_mesh.incenter)) * scale
        )
    return mapped


def get_temperatures(
    edat,
    eirene_path,
    ion_temperature_values=None,
    genex_triangulation=None,
    *,
    sparse_temperature_handler=default_sparse_temperature_handler,
    pseudo_temperature_handler=(
        default_single_species_pseudo_temperature_handler
    ),
    sparse_threshold=0.5,
):
    """Return explicit temperature arrays on the EIRENE triangle mesh."""
    eirene_path = Path(eirene_path)
    edat.read_ft46(eirene_path / "fort.46")
    triangle_points = edat.triangle_mesh.incenter
    temperatures = {}

    for key, (particle_class, labels_key, prefix, pseudo) in (
        TEMPERATURE_CLASS_INFO.items()
    ):
        density_block = edat.fort46["pden" + key]
        raw_temperatures = []
        labels = [label.strip() for label in edat.fort46[labels_key]]

        for index, species in enumerate(labels):
            density = density_block[:, index]
            momentum_squared = sum(
                edat.fort46[component + "den" + key][:, index] ** 2
                for component in ("vx", "vy", "vz")
            )
            raw = _temperature_from_moments(
                edat.fort46["eden" + key][:, index],
                density,
                momentum_squared,
                edat.masses[particle_class][index],
            )
            raw_temperatures.append(raw)
            handled = sparse_temperature_handler(
                raw,
                density,
                triangle_points,
                threshold=sparse_threshold,
            )
            if handled is not None:
                temperatures[f"{prefix}_{species}"] = handled

        # The default aggregation is intentionally limited to one physical
        # species per particle class. Multi-species runs must inject a policy.
        if raw_temperatures:
            pseudo_raw, pseudo_density = pseudo_temperature_handler(
                raw_temperatures,
                density_block,
                particle_class=particle_class,
                species_labels=labels,
            )
            handled = sparse_temperature_handler(
                pseudo_raw,
                pseudo_density,
                triangle_points,
                threshold=sparse_threshold,
            )
            if handled is not None:
                temperatures[f"{prefix}_{pseudo}"] = handled

    if ion_temperature_values is not None:
        if genex_triangulation is None:
            raise ValueError(
                "genex_triangulation is required with ion temperatures"
            )
        temperatures.update(
            map_ion_temperatures_to_triangles(
                edat, ion_temperature_values, genex_triangulation
            )
        )

    return temperatures

def prepare_fort31(edat, genex_data, eorder, iorder):
    """Fill fort31 dictionary with gene-x data"""
    # vExB = ExB_velocity() see analyze_moments.py
    upar = dict_to_array(genex_data["u_par"], iorder, edat.fort31["ua"].shape)
    nions = len(iorder)
    eorder = [eorder] if not isinstance(eorder, list) else eorder
    if nions>1:
        upol = upar * edat.fort31["bb"][:,:,0][:, :, None]
    else:
        upol = upar * edat.fort31["bb"][:,:,0]

    urad = dict_to_array(genex_data["u_rad"], iorder, edat.fort31["ua"].shape)
    edat.fort31["na"] = dict_to_array(genex_data["n"], iorder, edat.fort31["na"].shape)
    edat.fort31["up"] = upol
    edat.fort31["vv"] = urad
    edat.fort31["ww"] = dict_to_array(genex_data["u_phi"], iorder, edat.fort31["ww"].shape)
    edat.fort31["te"] = dict_to_array(genex_data["Ttot"], eorder,
                                      edat.fort31["te"].shape) * pyconst.elementary_charge
    edat.fort31["ti"] = dict_to_array(genex_data["Ttot"], iorder,
                                      edat.fort31["ti"].shape) * pyconst.elementary_charge
    edat.fort31["ua"] = upar
    # pitch angle - constant in time
    edat.fort31["fnax"] = upol * dict_to_array(genex_data["fnax"], iorder, edat.fort31["fnax"].shape)
    edat.fort31["fnay"] = urad * dict_to_array(genex_data["fnay"], iorder, edat.fort31["fnay"].shape)
    edat.fort31["uadia"] = np.zeros((edat.fort31["uadia"].shape))
    edat.fort31["vadia"] = np.zeros((edat.fort31["vadia"].shape))
    edat.fort31["po"] = dict_to_array(genex_data["es_pot"], None)

    # pressure and heat fluxes only used for eirene output/graphics
    edat.fort31["pr"] = dict_to_array(genex_data["pr"], None)
    qepar = dict_to_array(genex_data["Q_par"], eorder, edat.fort31["fhex"].shape)
    qipar = dict_to_array(genex_data["Q_par"], iorder)
    if nions>1:
        qipar = np.mean(qipar,axis=2)
    edat.fort31["fhix"] = qipar * edat.fort31["bb"][:,:,0] # Poloidal ion heat flux
    # Radial ion heat flux -
    edat.fort31["fhex"] = qepar * edat.fort31["bb"][:,:,0] # Poloidal electron heat flux
    # Radial electron heat flux

"""Make sure array is correct shape for fort.31"""
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

"""Wrapper for running Eirene"""
def run_eirene(Eirene_time, eirene_path, command="eirobjx", solpstop=""):
    output_file = eirene_path / Path("run.log")
    try:
        with open(output_file, "w") as outfile:
            env = os.environ.copy()
            env["SOLPSTOP"] = solpstop
            proc = subprocess.Popen([command], stdout=outfile,
                                    stderr=subprocess.STDOUT,
                                    text=True, cwd=eirene_path,
                                    preexec_fn=os.setsid)
            try:
                proc.wait(timeout=Eirene_time)

            except subprocess.TimeoutExpired:
                print(f"⏱️ {command} exceeded {Eirene_time}s — killing process group")

                # Kill entire process group
                if proc.pid:
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    print("⚠️ Invalid PID, skipping killpg")

                # Optional: escalate if needed
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if proc.pid:
                        print("⚠️ Force killing EIRENE")
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        print("⚠️ Invalid PID, skipping killpg")

                with open(output_file, "a") as outfile:
                    outfile.write(f"\n--- TIMEOUT after {Eirene_time}s ---\n")

                return status.TIMEOUT
        if proc.returncode != 0:
            print(f"❌{command} failed with return code {proc.returncode}")
            return status.ERROR
    except FileNotFoundError:
        print(f"⚠️ Error: {command} command not found. Make sure it’s in your PATH.")
        return status.ERROR
    except Exception as e:
        print(f"⚠️  Unexpected error running {command}: {e}")
        return status.ERROR
    return status.SUCCESS
