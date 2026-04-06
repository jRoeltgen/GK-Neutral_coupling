import sys
import os
import yaml
import scipy.constants as pyconst
import numpy as np
import postgkyl as pg

from scipy.ndimage import median_filter

# Load configuration from YAML file
config_file = sys.argv[1] if len(sys.argv) > 1 else 'gkeyllCoupling/gkeyllExamples/config.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Extract paths from config
paths = config['paths']

target_dir = paths['lib_path'] + "common/"
sys.path.insert(0, target_dir)
import B2IO as b2
import eireneIO as eirene
import triangle_mesh as tri_mesh

# Set paths from config
gkeyll_data_path = paths['gkeyll_data_path']
gkeyll_simulation_name = paths['gkeyll_simulation_name']
eirene_data_path = paths['eirene_data_path']
gkeyll_text_input_path = paths['gkeyll_text_input_path']

# Block range for half-domain (fixed for now)
bmin = 0
bmax = 8
 
# Helper Function for de-noising
def despike_source(data, kernel_size=3):
    """
    Applies a median filter to remove single-pixel outliers (Monte Carlo noise).
    A 3x3 kernel is usually sufficient to kill spikes without blurring features.
    """
    # Check for NaNs just in case and replace with 0
    data = np.nan_to_num(data, nan=0.0)
    return median_filter(data, size=kernel_size)


# Eirene Data Loading
ion = "D+"
molecule = "D2+"
# read fort.44 and fort.46 from given director ("./")
# read fort.33, fort.34, and fort.35 from given director ("./")
edat = eirene.eirene(eirene_data_path)
edat.load_extra_forts(eirene_data_path)
edat.triangle_mesh.calc_incenter()
eR = edat.triangle_mesh.incenter[:,0]
eZ = edat.triangle_mesh.incenter[:,1]

# Calculations
mass_ion = 3.34e-27
mass_elc = 9.11e-31
mass_molecule = mass_ion*2.0
eV = 1.602e-19

pisource = edat.sources["particle"][ion]["SUM"]
misource = edat.sources["momentum"][ion]["SUM"]
eisource = edat.sources["energy"][ion]["SUM"]

pesource = edat.sources["particle"]["ELECTRONS"]["SUM"]
mesource = 0.0
eesource = edat.sources["energy"]["ELECTRONS"]["SUM"]

pmsource = edat.sources["particle"][molecule]["SUM"]
mmsource = 0.0
emsource = edat.sources["energy"]["TEST IONS"]["SUM"]

# Step 1: load the Gkeyll grid information
#     same as process_eirene_output.py's Step 1
simNames = ['%s_b%d'%(gkeyll_data_path+gkeyll_simulation_name,i) for i in range(bmin,bmax)]
Rlist = []
Zlist = []
nodal_grid_list = []
jlist = []
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodesint.gkyl")
    vals = data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    phi = vals[:,:,2]
    
    Rlist.append(R)
    Zlist.append(Z)


# Step 2: Fill Nodal Gkeyll data by finding closest point from Eirene
# Changed to use data calculated/loaded from Eirene
M0i_list = []
M2i_list = []
M1i_list = []

M0e_list = []
M2e_list = []
M1e_list = []

M0m_list = []
M2m_list = []
M1m_list = []
for i, simName in enumerate(simNames):
    nx, nz = Rlist[i].shape
    M0i = np.zeros((nx,nz))
    M1i = np.zeros((nx,nz))
    M2i = np.zeros((nx,nz))

    M0e = np.zeros((nx,nz))
    M1e = np.zeros((nx,nz))
    M2e = np.zeros((nx,nz))

    M0m = np.zeros((nx,nz))
    M1m = np.zeros((nx,nz))
    M2m = np.zeros((nx,nz))
    for ix in range(nx):
        for iz in range(nz):
            lindist = np.sqrt((Rlist[i][ix,iz] - eR)**2 + (Zlist[i][ix,iz] - eZ)**2)
            linidx = np.argmin(lindist)
            
            # M0 source calculation
            # ni = PAEL * 1e6/eV
            M0i[ix,iz] = pisource[linidx]/eV*1e6
            M0e[ix,iz] = pesource[linidx]/eV*1e6
            M0m[ix,iz] = pmsource[linidx]/eV*1e6

            # M1 source calculation
            M1i[ix,iz] = misource[linidx]*10/mass_ion/eV
            M1e[ix,iz] = 0.0 
            M1m[ix,iz] = 0.0 

            # M2 source Calculation
            M2i[ix,iz] = eisource[linidx]*1e6/mass_ion*2.0
            M2e[ix,iz] = eesource[linidx]*1e6/mass_elc*2.0
            M2m[ix,iz] = emsource[linidx]*1e6/mass_molecule*2.0


    # Smooth data
    M1i_smoothed = despike_source(M1i, kernel_size=3)

    # Append the clipped/smoothed data
    M0i_list.append(M0i)
    M1i_list.append(M1i_smoothed)
    M2i_list.append(M2i)
    M0e_list.append(M0e)
    M1e_list.append(M1e)
    M2e_list.append(M2e)
    M0m_list.append(M0m)
    M1m_list.append(M1m)
    M2m_list.append(M2m)

## Step 3: Write nodal data to text file 

fNames = ['%s_b%d'%(gkeyll_simulation_name,i) for i in range(bmin,bmax)]
for i, fname in enumerate(fNames):
    np.savetxt(gkeyll_text_input_path+fname+"-ion_M0source.txt", M0i_list[i].flatten())
    np.savetxt(gkeyll_text_input_path+fname+"-ion_M1source.txt", M1i_list[i].flatten())
    np.savetxt(gkeyll_text_input_path+fname+"-ion_M2source.txt", M2i_list[i].flatten())


    np.savetxt(gkeyll_text_input_path+fname+"-elc_M0source.txt", M0e_list[i].flatten())
    np.savetxt(gkeyll_text_input_path+fname+"-elc_M1source.txt", M1e_list[i].flatten())
    np.savetxt(gkeyll_text_input_path+fname+"-elc_M2source.txt", M2e_list[i].flatten())

    np.savetxt(gkeyll_text_input_path+fname+"-molecule_M0source.txt", M0m_list[i].flatten())
    np.savetxt(gkeyll_text_input_path+fname+"-molecule_M1source.txt", M1m_list[i].flatten())
    np.savetxt(gkeyll_text_input_path+fname+"-molecule_M2source.txt", M2m_list[i].flatten())

            
print("Finished converting text to Gkeyll input")   


#Just  for plotting
Rall = np.array([])
Zall = np.array([])
M0iall = np.array([])
M1iall = np.array([])
M2iall = np.array([])
M0eall = np.array([])
M1eall = np.array([])
M2eall = np.array([])
M0mall = np.array([])
M1mall = np.array([])
M2mall = np.array([])
for i in range(8):
    Rall=np.append(Rall,Rlist[i].flatten())
    Zall=np.append(Zall,Zlist[i].flatten())
    M0iall=np.append(M0iall,M0i_list[i].flatten())
    M1iall=np.append(M1iall,M1i_list[i].flatten())
    M2iall=np.append(M2iall,M2i_list[i].flatten())

    M0eall=np.append(M0eall,M0e_list[i].flatten())
    M1eall=np.append(M1eall,M1e_list[i].flatten())
    M2eall=np.append(M2eall,M2e_list[i].flatten())

    M0mall=np.append(M0mall,M0m_list[i].flatten())
    M1mall=np.append(M1mall,M1m_list[i].flatten())
    M2mall=np.append(M2mall,M2m_list[i].flatten())

