import B2IO as b2
import eireneIO as eirene
import triangle_mesh as tri_mesh
import scipy.constants as pyconst
import numpy as np
import postgkyl as pg

filepath = "./"
ion = "D+"
# Read data
b2dat = b2.B2()
b2dat.read_b2fgmtry(filepath) 
b2dat.read_b2fstate(filepath) 

# read fort.44 and fort.46 from given director ("./")
# read fort.33, fort.34, and fort.35 from given director ("./")
edat = eirene.eirene(filepath) 
edat.load_extra_forts(filepath)

# Calculations
etria.calc_incenter()
mass = pyconst.proton_mass*b2dat.state["am"][0]
b2dat.gmtry["R"] = b2dat.gmtry["crx"].mean(axis=2)
b2dat.gmtry["Z"] = b2dat.gmtry['cry'].mean(axis=2)
rS = b2dat.gmtry["R"]
zS = b2dat.gmtry["Z"]
pisource = edat.particle_source[ion]
misource = edat.momentum_source[ion]
eisource = edat.energy_source[ion]
pesource = edat.particle_source["ELECTRONS"]
mesource = edat.momentum_source["ELECTRONS"]
eesource = edat.energy_source["ELECTRONS"]

# Step 1: load the Gkeyll grid information
#     same as process_eirene_output.py's Step 1
baseName = 'h15'
bmin = 0
bmax = 12
simNames = ['%s_b%d'%(baseName,i) for i in range(bmin,bmax)]
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
nlist = []
Tlist = []
uxlist = []
uylist = []
uzlist = []
ulist = []
for i, simName in enumerate(simNames):
    nx, nz = Rlist[i].shape
    density = np.zeros((nx,nz))
    temp = np.zeros((nx,nz))
    ux = np.zeros((nx,nz))
    uy = np.zeros((nx,nz))
    uz = np.zeros((nx,nz))
    u = np.zeros((nx,nz,3))
    for ix in range(nx):
        for iz in range(nz):
            lindist = np.sqrt((Rlist[i][ix,iz] - etria.incenter[:,0])**2 + (Zlist[i][ix,iz] - etria.incenter[:,1])**2)
            linidx = np.argmin(lindist)
            ux[ix,iz] = vx[linidx]
            uy[ix,iz] = vy[linidx]
            uz[ix,iz] = vz[linidx]
            sDn[ix,iy] = pisource[linidx]
            sDm[ix,iy] = pmsource[linidx]
            sDe[ix,iy] = pesource[linidx]
            sen[ix,iy] = eesource[linidx]
            sem[ix,iy] = emsource[linidx]
            see[ix,iy] = essource[linidx]
            u[ix,iz,:] = np.r_[ux[ix,iz], uy[ix,iz], uz[ix,iz]]

    # Apply a floor
    density[density < 1e8] = 1e8
    temp[temp<0.0] = 50.0*eV;
    nlist.append(density)
    Tlist.append(temp)
    uxlist.append(ux)
    uylist.append(uy)
    uzlist.append(uz)
    ulist.append(u)

# Step 2.1: Read previous gkeyll data and average
#    from here on is the same as process_eirene_output.py
for i, simName in enumerate(simNames):
    nlist[i] = (nlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_M0.txt").reshape((nlist[i].shape[0], nlist[i].shape[1])))/2.0
    Tlist[i] = (Tlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_Temp.txt").reshape((Tlist[i].shape[0], Tlist[i].shape[1])))/2.0
    uxlist[i] = (uxlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_ux.txt").reshape((uxlist[i].shape[0], uxlist[i].shape[1])))/2.0
    uylist[i] = (uylist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_uy.txt").reshape((uylist[i].shape[0], uylist[i].shape[1])))/2.0
    uzlist[i] = (uzlist[i] + np.genfromtxt('gkeyll_text_input/'+simName+"-H0_uz.txt").reshape((uzlist[i].shape[0], uzlist[i].shape[1])))/2.0

# Step 3: Write nodal data to text file 
for i, simName in enumerate(simNames):
    np.savetxt('gkeyll_text_input/'+simName+"-H0_M0.txt", nlist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_Temp.txt", Tlist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_ux.txt", uxlist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_uy.txt", uylist[i].flatten())
    np.savetxt('gkeyll_text_input/'+simName+"-H0_uz.txt", uzlist[i].flatten())

            
print("Finished converting text to Gkeyll input")    


    
