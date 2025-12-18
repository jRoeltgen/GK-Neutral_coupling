import B2IO as b2
import eireneIO as eirene
import triangle_mesh as tri_mesh
import scipy.constants as pyconst
import numpy as np
import postgkyl as pg

ion = "D+"
molecules = "MOLECULES"
# read fort.44 and fort.46 from given director ("./")
# read fort.33, fort.34, and fort.35 from given director ("./")
eirene_data_path = "./step_full_device_D+D2/gkeyll_w_2D2/"
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

pisource = edat.particle_source[ion]
misource = edat.momentum_source[ion]
eisource = edat.energy_source[ion]
pesource = edat.particle_source["ELECTRONS"]
mesource = edat.momentum_source["ELECTRONS"]
eesource = edat.energy_source["ELECTRONS"]
pmsource = edat.particle_source['MOLECULES']
mmsource = edat.momentum_source['MOLECULES']
emsource = edat.energy_source['MOLECULES']

# Step 1: load the Gkeyll grid information
#     same as process_eirene_output.py's Step 1
gkeyll_data_path = './'
gkeyll_simulation_name = 'hstep26'
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


    # Append the clipped/smoothed data
    M0i_list.append(M0i)
    M1i_list.append(M1i)
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
    np.savetxt('./gkeyll_text_input/'+fname+"-ion_M0source.txt", M0i_list[i].flatten())
    np.savetxt('./gkeyll_text_input/'+fname+"-ion_M1source.txt", M1i_list[i].flatten())
    np.savetxt('./gkeyll_text_input/'+fname+"-ion_M2source.txt", M2i_list[i].flatten())


    np.savetxt('./gkeyll_text_input/'+fname+"-elc_M0source.txt", M0e_list[i].flatten())
    np.savetxt('./gkeyll_text_input/'+fname+"-elc_M1source.txt", M1e_list[i].flatten())
    np.savetxt('./gkeyll_text_input/'+fname+"-elc_M2source.txt", M2e_list[i].flatten())

    np.savetxt('./gkeyll_text_input/'+fname+"-molecule_M0source.txt", M0m_list[i].flatten())
    np.savetxt('./gkeyll_text_input/'+fname+"-molecule_M1source.txt", M1m_list[i].flatten())
    np.savetxt('./gkeyll_text_input/'+fname+"-molecule_M2source.txt", M2m_list[i].flatten())

            
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

