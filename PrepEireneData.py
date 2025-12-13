import B2IO as b2
import eireneIO as eirene
import triangle_mesh as tri_mesh
import scipy.constants as pyconst
import numpy as np
import postgkyl as pg

# --- START: NEW HELPER FUNCTION FOR SMOOTHING AND CLIPPING ---
def smooth_and_clip(data_array, min_val, max_val):
    """
    Clips data_array values outside [min_val, max_val] and replaces them 
    with the average of their nearest valid neighbors.
    """
    clipped = np.copy(data_array)
    nx, nz = clipped.shape
    
    # 1. First Pass: Clip all values and identify invalid points
    invalid_mask = (clipped < min_val) | (clipped > max_val) | np.isnan(clipped)
    clipped[clipped < min_val] = min_val  # Simple floor/cap
    clipped[clipped > max_val] = max_val  # Simple ceiling/cap
    clipped[np.isnan(clipped)] = min_val  # Treat NaNs as below min_val for initial clipping

    # 2. Second Pass: Inpaint the points that were originally out-of-bounds
    # We iterate over the array to find points that were marked as invalid
    # but which now have neighbors whose clipped values we can average.
    
    # The while loop continues until no more points are inpainted in a pass,
    # which is robust for handling clusters of invalid points.
    inpaint_mask = invalid_mask.copy()
    num_inpainted_in_pass = 1
    
    # max_iterations prevents infinite loops, though should converge quickly
    max_iterations = nx * nz 
    iteration = 0

    while np.any(inpaint_mask) and num_inpainted_in_pass > 0 and iteration < max_iterations:
        num_inpainted_in_pass = 0
        new_inpaint_mask = inpaint_mask.copy()
        
        for ix in range(nx):
            for iz in range(nz):
                if inpaint_mask[ix, iz]:
                    neighbor_values = []
                    
                    # Iterate over the 8 neighbors (including corners)
                    for dix in [-1, 0, 1]:
                        for diz in [-1, 0, 1]:
                            if dix == 0 and diz == 0:
                                continue
                            
                            nix, niz = ix + dix, iz + diz
                            
                            # Check boundaries
                            if 0 <= nix < nx and 0 <= niz < nz:
                                # Only use neighbors that were NOT originally out-of-bounds
                                # The 'inpaint_mask' still holds the status of being originally invalid
                                if not invalid_mask[nix, niz]:
                                    neighbor_values.append(data_array[nix, niz])
                                elif not new_inpaint_mask[nix, niz]:
                                    # Use the value that was already corrected in a previous pass
                                    neighbor_values.append(clipped[nix, niz])

                    if neighbor_values:
                        # Replace the point with the average of valid neighbors
                        clipped[ix, iz] = np.mean(neighbor_values)
                        new_inpaint_mask[ix, iz] = False  # Mark as resolved
                        num_inpainted_in_pass += 1
        
        inpaint_mask = new_inpaint_mask
        iteration += 1

    # Final Pass: Use the simple cap for any remaining unresolved points
    # (This happens if the entire array or a whole region was invalid)
    clipped[inpaint_mask] = np.clip(data_array[inpaint_mask], min_val, max_val)
    
    return clipped
# --- END: NEW HELPER FUNCTION ---


# --- START: USER-DEFINED BOUNDARIES ---
# Conversions: 1 eV ~ 1.602e-19 J; 1 keV ~ 1.602e-16 J
T_MAX_KEV = 10.0
T_MIN_KEV = -10.0
T_max_J = T_MAX_KEV * 1.602e-16
T_min_J = T_MIN_KEV * 1.602e-16

U_MAX_M_S = 5e5  # Cap velocity at 1000 km/s (10^6 m/s)
U_MIN_M_S = -5e5      # Set minimum velocity at 1 m/s (to avoid division by zero artifacts)
U_max_m_s = U_MAX_M_S 
U_min_m_s = U_MIN_M_S
# --- END: USER-DEFINED BOUNDARIES ---


ion = "D+"
# read fort.44 and fort.46 from given director ("./")
# read fort.33, fort.34, and fort.35 from given director ("./")
eirene_data_path = "./eirene_data_step_extra/"
edat = eirene.eirene(eirene_data_path)
edat.load_extra_forts(eirene_data_path)
edat.triangle_mesh.calc_incenter()
eR = edat.triangle_mesh.incenter[:,0]
eZ = edat.triangle_mesh.incenter[:,1]

# Calculations
mass_ion = 3.34e-27
mass_elc = 9.11e-31
eV = 1.602e-19

pisource = edat.particle_source[ion]
misource = edat.momentum_source[ion]
eisource = edat.energy_source[ion]
pesource = edat.particle_source["ELECTRONS"]
mesource = edat.momentum_source["ELECTRONS"]
eesource = edat.energy_source["ELECTRONS"]

# Step 1: load the Gkeyll grid information
#     same as process_eirene_output.py's Step 1
gkeyll_data_path = './test_data/gkeyll_data/'
gkeyll_simulation_name = 'step23'
bmin = 0
bmax = 8
simNames = ['%s_b%d'%(gkeyll_data_path+gkeyll_simulation_name,i) for i in range(bmin,bmax)]
Rlist = []
Zlist = []
nodal_grid_list = []
jlist = []
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    phi = vals[:,:,2]
    temp_nodal_grid = data.get_grid()
    # This nodal grid is the true (psi,theta) coords
    nodal_grid = []
    for d in range(0,len(temp_nodal_grid)):
        nodal_grid.append( np.linspace(temp_nodal_grid[d][0], temp_nodal_grid[d][-1], len(temp_nodal_grid[d])-1) )
    
    Rlist.append(R)
    Zlist.append(Z)
    nodal_grid_list.append(nodal_grid)


# Step 2: Fill Nodal Gkeyll data by finding closest point from Eirene
# Changed to use data calculated/loaded from Eirene
M0i_list = []
M2i_list = []
M1i_list = []

M0e_list = []
M2e_list = []
M1e_list = []
for i, simName in enumerate(simNames):
    nx, nz = Rlist[i].shape
    M0i = np.zeros((nx,nz))
    M1i = np.zeros((nx,nz))
    M2i = np.zeros((nx,nz))

    M0e = np.zeros((nx,nz))
    M1e = np.zeros((nx,nz))
    M2e = np.zeros((nx,nz))
    for ix in range(nx):
        for iz in range(nz):
            lindist = np.sqrt((Rlist[i][ix,iz] - eR)**2 + (Zlist[i][ix,iz] - eZ)**2)
            linidx = np.argmin(lindist)
            
            # Density calculation (M0 source)
            # ni = PAEL * 1e6/eV
            M0i[ix,iz] = pisource[linidx]/eV*1e6
            M0e[ix,iz] = pesource[linidx]/eV*1e6

            # Velocity calculation (u_parallel source)
            M1i[ix,iz] = misource[linidx]*10/mass_ion/eV
            M1e[ix,iz] = 0.0 # Electron parallel momentum source often set to 0.0 or a simplified value for stability

            # Temperature calculation (T_source)
            M2i[ix,iz] = eisource[linidx]*1e6 
            M2e[ix,iz] = eesource[linidx]*1e6 


    # --- START: NEW SMOOTHING AND CLIPPING APPLICATION ---
    # Ion Temperature
    M2i_clipped = smooth_and_clip(M2i, T_min_J, T_max_J)
    
    # Ion Parallel Velocity
    #M1i_clipped = smooth_and_clip(M1i, U_min_m_s, U_max_m_s)

    ## Electron Temperature
    #M2e_clipped = smooth_and_clip(M2e, T_min_J, T_max_J)
    #
    ## Electron Parallel Velocity (if not already set to 0.0)
    #M1e_clipped = smooth_and_clip(M1e, U_min_m_s, U_max_m_s)
    
    # --- END: NEW SMOOTHING AND CLIPPING APPLICATION ---
    
    # Append the clipped/smoothed data
    M0i_list.append(M0i)
    M1i_list.append(M1i)
    M2i_list.append(M2i)
    M0e_list.append(M0e)
    M1e_list.append(M1e)
    M2e_list.append(M2e)

## Step 3: Write nodal data to text file 

fNames = ['%s_b%d'%(gkeyll_simulation_name,i) for i in range(bmin,bmax)]
for i, fname in enumerate(fNames):
    np.savetxt('gkeyll_text_input/'+fname+"-ion_M0source.txt", M0i_list[i].flatten())
    np.savetxt('gkeyll_text_input/'+fname+"-ion_M1source.txt", M1i_list[i].flatten())
    np.savetxt('gkeyll_text_input/'+fname+"-ion_M2source.txt", M2i_list[i].flatten())


    np.savetxt('gkeyll_text_input/'+fname+"-elc_M0source.txt", M0e_list[i].flatten())
    np.savetxt('gkeyll_text_input/'+fname+"-elc_M1source.txt", M1e_list[i].flatten())
    np.savetxt('gkeyll_text_input/'+fname+"-elc_M2source.txt", M2e_list[i].flatten())

            
print("Finished converting text to Gkeyll input")   


#Just  for plotting
Rall = np.array([])
Zall = np.array([])
M0iall = np.array([])
M1iall = np.array([])
M2iall = np.array([])
for i in range(8):
    Rall=np.append(Rall,Rlist[i].flatten())
    Zall=np.append(Zall,Zlist[i].flatten())
    M0iall=np.append(M0iall,M0i_list[i].flatten())
    M1iall=np.append(M1iall,M1i_list[i].flatten())
    M2iall=np.append(M2iall,M2i_list[i].flatten())


