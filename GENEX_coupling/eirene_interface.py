import eireneIO
import B2IO
import triangle_mesh
from EireneInputParser import EireneInputParser
from pathlib import Path
import subprocess
import numpy as np

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
    ns = len(eip.species["bulk_ions"])

    edat.read_ft30(eirene_path / Path("fort.30"))

    nx = edat.plasma_gmtry["nx"]
    ny = edat.plasma_gmtry["ny"]

    edat.read_ft31(eirene_path / "fort.31", nx, ny, ns)

    edat.triangle_mesh.calc_incenter()

    return edat, b2dat

def write_fort31(edat, genex_data, eorder, iorder):
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

def run_eirene(Eirene_time, command="eirobjx"):
    output_file = "run.log"
    try:
        with open(output_file, "w") as outfile:
            result = subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT,
                                    text=True, timeout=Eirene_time)
        if result.returncode != 0:
            print(f"❌{command} failed with return code {result.returncode}")
            raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        print(f"Error: {command} timed out after {Eirene_time} seconds.")
        raise SystemExit(1)
    except FileNotFoundError:
        print("⚠️ Error: {command} command not found. Make sure it’s in your PATH.")
        raise SystemExit(1)
    except Exception as e:
        print(f"⚠️  Unexpected error running {command}: {e}")
        raise SystemExit(1)
