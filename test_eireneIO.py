import filecmp
import eireneIO
import numpy as np
from pathlib import Path

filepath = "./test_data/eirene_data/" 
edat = eireneIO.eirene(Path(filepath))
edat.load_extra_forts(Path(filepath))
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
    
flag = (np.abs(1-(1+edat.extra_source["D"]*1e6)/(1+edat.fort46["pdena"][:,0]))<epsilon).all()
if (not flag):
    all_true = False
    print("Error. Fort.401 and pdena from fort.46 differ")
    
if all_true:
    print("No errors in eireneIO.")
    
