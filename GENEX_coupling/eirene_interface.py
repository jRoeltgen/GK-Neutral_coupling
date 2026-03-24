import eireneIO
import B2IO
import triangle_mesh
from EireneInputParser import EireneInputParser
from pathlib import Path
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
    b2dat = B2IO.B2(b2_path)

    eip = EireneInputParser(eirene_path / Path("input.dat"))
    eip.parse_species()
    edat.species_names = eip.species
    ns = len(edat.species_names["bulk_ions"])

    edat.read_ft30(eirene_path / Path("fort.30"))

    nx = edat.plasma_gmtry["nx"]
    ny = edat.plasma_gmtry["ny"]

    edat.read_ft31(eirene_path / "fort.31", nx, ny, ns)

    edat.triangle_mesh.calc_incenter()

    return edat, b2dat

def prepare_fort31(edat, genex_data, eorder, iorder):
    # vExB = ExB_velocity() see analyze_moments.py
    upar = dict_to_array(genex_data["u_par"], iorder, edat.fort31["ua"].shape)
    nions = len(iorder)
    if nions>1:
        upol = upar * edat.fort31["bb"][:,:,0][:, :, None]
    else:
        upol = upar * edat.fort31["bb"][:,:,0]

    urad = dict_to_array(genex_data["u_rad"], iorder, edat.fort31["ua"].shape)
    edat.fort31["na"] = dict_to_array(genex_data["n"], iorder, edat.fort31["na"].shape)
    edat.fort31["up"] = upol
    edat.fort31["vv"] = urad
    edat.fort31["ww"] = dict_to_array(genex_data["u_phi"], iorder, edat.fort31["ww"].shape)
    edat.fort31["te"] = dict_to_array(genex_data["Ttot"], eorder, edat.fort31["te"].shape)
    edat.fort31["ti"] = dict_to_array(genex_data["Ttot"], iorder, edat.fort31["ti"].shape)
    edat.fort31["ua"] = upar
    # pitch angle - constant in time
    edat.fort31["fnax"] = upol * edat.fort31["na"]
    edat.fort31["fnay"] = urad * edat.fort31["na"]
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

def run_eirene(Eirene_time, eirene_path, command="eirobjx"):
    output_file = eirene_path / Path("run.log")
    try:
        with open(output_file, "w") as outfile:
            proc = subprocess.Popen([command], stdout=outfile,
                                    stderr=subprocess.STDOUT,
                                    text=True, timeout=Eirene_time,
                                    cwd=eirene_path, preexec_fn=os.setsid)
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
