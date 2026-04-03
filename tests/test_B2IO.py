import filecmp
import B2IO
import numpy as np

filepath = "./test_data/b2_data/" 
bdat = B2IO.B2(filepath)
bdat.read_b2fstate(filepath+"b2fstate")
bdat.write_b2fstate("./new_b2fstate")
bdat.write_b2fgmtry("./new_b2fgmtry")
epsilon = 5e-5

all_true = True
if (not filecmp.cmp(filepath+"b2fstate","./new_b2fstate", shallow=False)):
    all_true = False
    print("Error. b2fstate files differ")

if (not filecmp.cmp(filepath+"b2fgmtry","./new_b2fgmtry", shallow=False)):
    all_true = False
    print("Error. b2fgmtry files differ")
    
if all_true:
    print("No errors in B2IO.")
    
