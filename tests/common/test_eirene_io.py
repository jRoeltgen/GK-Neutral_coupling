import filecmp
import sys
from pathlib import Path
if sys.version_info < (3, 6):
    raise ImportError("Python >=3.6 is needed for eireneIO")
from neutral_coupling.common import eirene_io as eireneIO
import numpy as np
import scipy.constants as pyconst

filepath = str(Path(__file__).parents[1] / "test_data" / "eirene_data") + "/"
edat = eireneIO.eirene(filepath)
edat.load_extra_forts(filepath)
edat.write_ft31(filepath+"new_fort.31")
edat.write_ft44(filepath+"new_fort.44")
edat.write_ft46(filepath+"new_fort.46")
edat.triangle_mesh.write_ft33(filepath+"new_fort.33")
edat.triangle_mesh.write_ft34(filepath+"new_fort.34")
edat.triangle_mesh.write_ft35(filepath+"new_fort.35")
epsilon = 5e-5

all_true = True
if (not filecmp.cmp(filepath+"fort.31",filepath+"new_fort.31", shallow=False)):
    all_true = False
    print("Error. Fort.31 files differ")

if (not filecmp.cmp(filepath+"fort.33",filepath+"new_fort.33", shallow=False)):
    all_true = False
    print("Error. Fort.33 files differ")

if (not filecmp.cmp(filepath+"fort.34",filepath+"new_fort.34", shallow=False)):
    all_true = False
    print("Error. Fort.34 files differ")

if (not filecmp.cmp(filepath+"fort.35",filepath+"new_fort.35", shallow=False)):
    all_true = False
    print("Error. Fort.35 files differ")

if (not filecmp.cmp(filepath+"fort.44",filepath+"new_fort.44", shallow=False)):
    all_true = False
    print("Error. Fort.44 files differ")

if (not filecmp.cmp(filepath+"fort.46",filepath+"new_fort.46", shallow=False)):
    all_true = False
    print("Error. Fort.46 files differ")

source = np.array(edat.loaded_sources.extra_sources["N/A"]["atom-plasma"]["D"]["array"])
flag = (np.abs(1-(1+source*1e6)/(1+edat.fort46["pdena"][:,0]))<epsilon).all()
if (not flag):
    all_true = False
    print("Error. Fort.401 and pdena from fort.46 differ")

if all_true:
    print("No errors in eireneIO file I/O.")

# ---- Input data ----
d = {
    "particle": {
        "a": np.array([1.0, 2.0]),
        "nested": {
            "b": np.array([3.0])
        }
    },
    "momentum": {
        "c": np.array([4.0])
    },
    "energy": {
        "d": np.array([5.0])
    }
}

# Keep reference for in-place check
original_id = id(d["particle"])

# ---- Run conversion ----
edat.convert_source_dict_to_SI(d)
eV = pyconst.elementary_charge

# ---- Expected values ----
assert np.allclose(d["particle"]["a"], np.array([1.0, 2.0]) * 1e6 / eV)
assert np.allclose(d["particle"]["nested"]["b"], np.array([3.0]) * 1e6 / eV)

assert np.allclose(d["momentum"]["c"], np.array([4.0]) * 10 / eV)

assert np.allclose(d["energy"]["d"], np.array([5.0]) * 1e6)

# ---- In-place structure check ----
assert id(d["particle"]) == original_id

print("Testing convert_source_dict_to_SI passed.")
