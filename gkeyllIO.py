
# coding: utf-8
import numpy as np
import postgkyl as pg
import os
import math
import sys

import scipy.interpolate 
from scipy.interpolate import RegularGridInterpolator, interp1d
import scipy.integrate as sci


import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


class gkeyll:
    def __init__(self, filepath=None, name=None, half_domain=False):
        # Universal params
        self.mp = 1.67262192e-27
        self.me = 9.1093837e-31
        self.eV = 1.602e-19
        self.mu_0 = 12.56637061435917295385057353311801153679e-7
        self.eps0=8.854187817620389850536563031710750260608e-12
        self.masses = {}
        self.masses["elc"] = self.me
        self.masses["ion"] = 2.014*self.mp
        self.charges = {}
        self.charges["elc"] = -self.eV 
        self.charges["ion"] = self.eV

        #Set half domain options
        self.half_domain = half_domain
        self.bmin = 0
        if self.half_domain:
            self.bmax = 8
        else:
            self.bmax = 12

        # Set up names for data loading
        self.sim_dir = filepath
        self.base_name = self.sim_dir+name
        self.sim_names = ['%s_b%d'%(self.base_name,i) for i in range(self.bmin,self.bmax)]

        #Set up empty block data
        self.mom_data_list = [None]*12
        self.grid_list = [None]*12
        self.coeff_data_list = [None]*12




    def __fix_gridvals(self, grid):
        """Output file grids have cell-edge coordinates by default, but the values are at cell centers.
        This function will return the cell-center coordinate grid.
        Usage: cell_center_grid = __fix_gridvals(cell_edge_grid)
        """
        grid = np.array(grid).squeeze()
        grid = (grid[0:-1] + grid[1:])/2
        return grid

    def __cellavginterpolate(self, raw_x,raw_z,x,z,data):
        """
           Interpolates gkeyll data onto the interpolation points using bilinear interpolation
           on the psi,theta grid rather than interpolating by evaluating the DG expansions.

           Note: Bilinear interpolation on the psi, theta grid is better posed than on the R,Z grid 
           because the psi, theta grid is rectangular
        """
        myinterpolator = RegularGridInterpolator((raw_x,raw_z), data, bounds_error=False, fill_value=None)
        x0,z0 = np.meshgrid(x,z)
        return myinterpolator((x0,z0)).T

    def read_data(self, frame):
        if(self.half_domain):
            self.__read_data_half_domain(frame)
        else:
            self.__read_data_full_domain(frame)

    def __read_data_full_domain(self, frame):
        print("Full Domain Not Implemented Yet")

    def read_coeffs(self, frame):
        if(self.half_domain):
            self.__read_coeffs_half_domain(frame)
        else:
            self.__read_coeffs_full_domain(frame)

    def __read_coeffs_full_domain(self, frame):
        print("Full Domain Not Implemented Yet")

    def read_geometry(self):
        if(self.half_domain):
            self.__read_geometry_half_domain()
        else:
            self.__read_geometry_full_domain()

    def __read_geometry_full_domain(self, frame):
        print("Full Domain Not Implemented Yet")

    def __read_geometry_half_domain(self):
        half_grid_list = []
        for isim, sim_name in enumerate(self.sim_names):
            grid = {}
            mdata = pg.GData('%s-bmag.gkyl'%sim_name)
            x, z = mdata.get_grid()
            grid["x"] = x
            grid["z"] = z
            grid["xc"] = self.__fix_gridvals(x)
            grid["zc"] = self.__fix_gridvals(z)
            # Append data
            half_grid_list.append(grid)

        grid_list = self.grid_list

        #Unmodified but renumbered blocks
        grid_list[0] = half_grid_list[0]
        grid_list[1] = half_grid_list[1]
        grid_list[8] = half_grid_list[4]
        grid_list[9] = half_grid_list[5]
        
        
        # Reflected blocks
        grid_list[3] = {}
        grid_list[6] = {}
        grid_list[4] = {}
        grid_list[5] = {}

        # Doubled blocks
        grid_list[2] = {}
        grid_list[7] = {}
        grid_list[10] = {}
        grid_list[11] = {}

        #Do x myself manually
        grid_list[3]["x"] = grid_list[1]["x"]
        grid_list[6]["x"] = grid_list[8]["x"]
        grid_list[4]["x"] = grid_list[0]["x"]
        grid_list[5]["x"] = grid_list[9]["x"]
        
        grid_list[2]["x"] = grid_list[1]["x"]
        grid_list[7]["x"] = grid_list[8]["x"]
        grid_list[10]["x"] = half_grid_list[6]["x"]
        grid_list[11]["x"] = half_grid_list[7]["x"]

        #Do z myself manually
        grid_list[3]["z"] = -np.flip(grid_list[1]["z"])
        grid_list[6]["z"] = -np.flip(grid_list[8]["z"])
        grid_list[4]["z"] = -np.flip(grid_list[0]["z"])
        grid_list[5]["z"] = -np.flip(grid_list[9]["z"])
        
        grid_list[2]["z"] = np.r_[half_grid_list[2]["z"], -np.flip(half_grid_list[2]["z"][:-1])]
        grid_list[7]["z"] = np.r_[-np.flip(half_grid_list[3]["z"][1:]), half_grid_list[3]["z"]]
        grid_list[10]["z"] = np.array([half_grid_list[6]["z"][0] + np.diff(half_grid_list[6]["z"])[0]*i for i in range(2*len(half_grid_list[6]["z"])-1)])
        grid_list[11]["z"] = np.flip(np.array([half_grid_list[7]["z"][-1] - np.diff(half_grid_list[7]["z"])[0]*i for i in range(2*len(half_grid_list[7]["z"])-1)]))

        #Do xc myself manually
        grid_list[3]["xc"] = grid_list[1]["xc"]
        grid_list[6]["xc"] = grid_list[8]["xc"]
        grid_list[4]["xc"] = grid_list[0]["xc"]
        grid_list[5]["xc"] = grid_list[9]["xc"]
        
        grid_list[2]["xc"] = grid_list[1]["xc"]
        grid_list[7]["xc"] = grid_list[8]["xc"]
        grid_list[10]["xc"] = half_grid_list[6]["xc"]
        grid_list[11]["xc"] = half_grid_list[7]["xc"]

        #Do zc myself manually
        grid_list[3]["zc"] = -np.flip(grid_list[1]["zc"])
        grid_list[6]["zc"] = -np.flip(grid_list[8]["zc"])
        grid_list[4]["zc"] = -np.flip(grid_list[0]["zc"])
        grid_list[5]["zc"] = -np.flip(grid_list[9]["zc"])
        
        grid_list[2]["zc"] = np.r_[half_grid_list[2]["zc"], -np.flip(half_grid_list[2]["zc"])]
        grid_list[7]["zc"] = np.r_[-np.flip(half_grid_list[3]["zc"]), half_grid_list[3]["zc"]]
        grid_list[10]["zc"] = np.array([half_grid_list[6]["zc"][0] + np.diff(half_grid_list[6]["zc"])[0]*i for i in range(2*len(half_grid_list[6]["zc"]))])
        grid_list[11]["zc"] = np.flip(np.array([half_grid_list[7]["zc"][-1] - np.diff(half_grid_list[7]["zc"])[0]*i for i in range(2*len(half_grid_list[7]["zc"]))]))

    def __read_coeffs_half_domain(self, frame):
        #frame = int(np.genfromtxt("gkeyll_text_output/new_data_flag"))
        half_mom_data_list = []
        for isim, sim_name in enumerate(self.sim_names):
            mom_data = {}
            #Load moment data
            for species in ["elc", "ion"]:
                for mom in ["M0", "M1", "M2"]:
                    mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
                    mom_data[species+mom] = mdata.get_values()
                #Set derived moment data
                mom_data[species+"Temp"] =  (self.masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / self.eV
                mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
            # Load the potential
            mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
            mom_data["phi"] = mdata.get_values()

            ncoeffs = mom_data["elcM0"].shape[2]
        
            # Append data
            half_mom_data_list.append(mom_data)
        
        # Construct the 12 block data
        mom_data_list = self.coeff_data_list
        even_keys = ["elcM0", "elcTemp", "ionM0", "ionTemp", "phi"]
        odd_keys = ["elcM1", "elcUpar", "ionM1", "ionUpar"]
        
        #Unmodified but renumbered blocks
        mom_data_list[0] = half_mom_data_list[0]
        mom_data_list[1] = half_mom_data_list[1]
        mom_data_list[8] = half_mom_data_list[4]
        mom_data_list[9] = half_mom_data_list[5]
        
        # Reflected blocks
        mom_data_list[3] = {}
        mom_data_list[6] = {}
        mom_data_list[4] = {}
        mom_data_list[5] = {}
        
        # Doubled blocks
        mom_data_list[2] = {}
        mom_data_list[7] = {}
        mom_data_list[10] = {}
        mom_data_list[11] = {}
        for key in even_keys:
            mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1],ncoeffs))
            mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1],ncoeffs))
            mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1],ncoeffs))
            mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1],ncoeffs))
        
        for key in odd_keys:
            mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1],ncoeffs))
            mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1],ncoeffs))
            mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1],ncoeffs))
            mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1],ncoeffs))
        
        # Flip some and double/reflect some
        for key in even_keys:
            #Upper SOL
            mom_data_list[3][key] = np.flip(mom_data_list[1][key], axis=1).copy()
            mom_data_list[6][key] = np.flip(mom_data_list[8][key], axis=1).copy()
            mom_data_list[3][key][:,:,ncoeffs//2:]*=-1
            mom_data_list[6][key][:,:,ncoeffs//2:]*=-1

            #Upper PF
            mom_data_list[4][key] = np.flip(mom_data_list[0][key], axis=1).copy()
            mom_data_list[5][key] = np.flip(mom_data_list[9][key], axis=1).copy()
            mom_data_list[4][key][:,:,ncoeffs//2:]*=-1
            mom_data_list[5][key][:,:,ncoeffs//2:]*=-1        

            #Outer Middle
            mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
            mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = np.flip(half_mom_data_list[2][key], axis=1)
            mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:][:,:,ncoeffs//2:]*=-1
        
            mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
            mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = np.flip(half_mom_data_list[6][key], axis=1)
            mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:][:,:,ncoeffs//2:]*=-1
        
            #Inner Middle
            mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = np.flip(half_mom_data_list[3][key], axis=1) 
            mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
            mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2][:,:,ncoeffs//2:]*=-1
        
            mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = np.flip(half_mom_data_list[7][key], axis=1) 
            mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
            mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2][:,:,ncoeffs//2:]*=-1
        
        for key in odd_keys:
            #Upper SOL
            mom_data_list[3][key] = -np.flip(mom_data_list[1][key], axis=1)
            mom_data_list[6][key] = -np.flip(mom_data_list[8][key], axis=1)
            mom_data_list[3][key][:,:,ncoeffs//2:]*=-1
            mom_data_list[6][key][:,:,ncoeffs//2:]*=-1

            #Upper PF
            mom_data_list[4][key] = -np.flip(mom_data_list[0][key], axis=1)
            mom_data_list[5][key] = -np.flip(mom_data_list[9][key], axis=1)
            mom_data_list[4][key][:,:,ncoeffs//2:]*=-1
            mom_data_list[5][key][:,:,ncoeffs//2:]*=-1
        
            #Outer Middle
            mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
            mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = -np.flip(half_mom_data_list[2][key], axis=1)
            mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:][:,:,ncoeffs//2:]*=-1 
        
            mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
            mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = -np.flip(half_mom_data_list[6][key], axis=1)
            mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:][:,:,ncoeffs//2:]*=-1
        
            #Inner Middle
            mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = -np.flip(half_mom_data_list[3][key], axis=1) 
            mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
            mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2][:,:,ncoeffs//2:]*=-1
        
            mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = -np.flip(half_mom_data_list[7][key], axis=1) 
            mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
            mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2][:,:,ncoeffs//2:]*=-1
        
        print("Processed Gkeyll output for frame %d"%frame)
 

   

    def __read_data_half_domain(self, frame):
        #frame = int(np.genfromtxt("gkeyll_text_output/new_data_flag"))
        half_mom_data_list = []
        for isim, sim_name in enumerate(self.sim_names):
            mom_data = {}
            raw_mom_data = {}
            #Load geometry
            bdata = pg.GData('%s-bmag.gkyl'%sim_name)
            grid,val = pg.data.GInterpModal(bdata,poly_order=1,basis_type='ms').interpolate(0)
            mom_data["B"] = val.squeeze()
            raw_mom_data["B"] = bdata.get_values()[:,:,0]/2
            
            ci = 0
            for c1 in ["x","y","z"]:
                for c2 in ["x","y","z"]:
                    if( c1=="y" and c2 =="x"):
                        continue
                    if( c1=="z" and c2 !="z"):
                        continue
                    gdata = pg.GData('%s-g_ij.gkyl'%sim_name)
                    grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
                    mom_data["g_%s%s"%(c1,c2)] = val.squeeze()
                    raw_mom_data["g_%s%s"%(c1,c2)] = gdata.get_values()[:,:,0]/2
                    ci+=1
            ci = 0
            for c1 in ["x","y","z"]:
                for c2 in ["x","y","z"]:
                    if( c1=="y" and c2 =="x"):
                        continue
                    if( c1=="z" and c2 !="z"):
                        continue
                    gdata = pg.GData('%s-gij.gkyl'%sim_name)
                    grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
                    mom_data["g%s%s"%(c1,c2)] = val.squeeze()
                    raw_mom_data["g%s%s"%(c1,c2)] = gdata.get_values()[:,:,0]/2
                    ci+=1
            
            jdata = pg.GData('%s-jacobgeo.gkyl'%sim_name)
            grid,val = pg.data.GInterpModal(jdata,poly_order=1,basis_type='ms').interpolate(0)
            mom_data["J"] = val.squeeze()
            raw_mom_data["J"] = jdata.get_values()[:,:,0]/2
            raw_grid = jdata.get_grid()
            geo_fac = 1/mom_data["J"]/mom_data["B"]
        
            #Load grid data
            node_data = pg.GData(sim_name+"-nodes.gkyl")
            vals = node_data.get_values()
            R = vals[:,:,0]
            Z = vals[:,:,1]
            PHI = vals[:,:,2]
            mom_data["R"] = R
            mom_data["Z"] = Z
        
            #Get interpolated physical coords
            temp_nodal_grid = node_data.get_grid()
            nodal_grid = []
            for d in range(0,len(temp_nodal_grid)):
                nodal_grid.append( np.linspace(temp_nodal_grid[d][0], temp_nodal_grid[d][-1], len(temp_nodal_grid[d])-1) )
        
            Rinterpolator = RegularGridInterpolator((nodal_grid[0], nodal_grid[1]), R)
            Zinterpolator = RegularGridInterpolator((nodal_grid[0], nodal_grid[1]), Z)
            g0, g1 = np.meshgrid(grid[0], grid[1])
            R = Rinterpolator((g0,g1))
            Z = Zinterpolator((g0, g1))
        
            g0i, g1i = np.meshgrid(self.__fix_gridvals(grid[0]), self.__fix_gridvals(grid[1]))
            Ri = Rinterpolator((g0i,g1i))
            Zi = Zinterpolator((g0i, g1i))
        
            mom_data["Ri"] = Ri.T
            mom_data["Zi"] = Zi.T
        
            #Get Plate angle information
            if isim == 1 : 
                plate_data = np.genfromtxt(self.sim_dir+"stepplate_data/highres/osol.txt", delimiter = ",")
            if isim == 0 : 
                plate_data = np.genfromtxt(self.sim_dir+"stepplate_data/highres/opf.txt", delimiter = ",")
            if isim == 4 : 
                plate_data = np.genfromtxt(self.sim_dir+"stepplate_data/highres/isol.txt", delimiter = ",")
            if isim == 5 : 
                plate_data = np.genfromtxt(self.sim_dir+"stepplate_data/highres/ipf.txt", delimiter = ",")
        
            #Load moment data
            for species in ["elc", "ion"]:
                for mom in ["M0", "M1", "M2"]:
                    mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
                    raw_mom_data[species+mom] = mdata.get_values()[:,:,0]/2
                    raw_grid = mdata.get_grid()
                    raw_x = self.__fix_gridvals(raw_grid[0])
                    raw_z = self.__fix_gridvals(raw_grid[1])
                    grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
                    x = self.__fix_gridvals(grid[0])
                    z = self.__fix_gridvals(grid[1])
                    val = val.squeeze()
                    mom_data[species+mom] = val
        
                #Set cell avg moment data
                raw_mom_data[species+"Temp"] =  (self.masses[species]/3) * (raw_mom_data[species+"M2"] - raw_mom_data[species+"M1"]**2 / raw_mom_data[species+"M0"])/raw_mom_data[species+"M0"] / self.eV
                raw_mom_data[species+"Upar"] =  raw_mom_data[species+"M1"]/raw_mom_data[species+"M0"]
        
                # Set interpolated moment data
                mom_data[species+"Temp"] =  (self.masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / self.eV
                mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
                #Set interpolated data using interpolating function
                #mom_data[species+"M1"] = self.__cellavginterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"M1"])
                #mom_data[species+"M0"] = self.__cellavginterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"M0"])
                #mom_data[species+"Temp"] = self.__cellavginterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Temp"])
                #mom_data[species+"Upar"] = self.__cellavginterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Upar"])
            
            
            # Calculate sound speed for ion species
            for species in ["ion" ]:
                mom_data[species+"cs"] = np.sqrt( (mom_data["elcTemp"]*self.eV + mom_data[species+"Temp"]*self.eV) / self.masses[species])
            
            for species in ["ion", "elc" ]:
                mom_data[species+"normUpar"] = mom_data[species+"Upar"]/mom_data["ioncs"]
        
        
            # Load the potential
            mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
            grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
            raw_mom_data["phi"] = mdata.get_values()[:,:,0]/2
            x = self.__fix_gridvals(grid[0])
            z = self.__fix_gridvals(grid[1])
            val = val.squeeze()
            mom_data["phi"] = val
        
            # Interpolate B ratio at plates
            if isim in [1,0,4,5]:
                Binterpolator = interp1d(plate_data[:,0], plate_data[:,1])
                Bratio = Binterpolator(x)
                mom_data["Bratio" ] = Bratio
        
            #Save grids
            mom_data["x"] = x
            mom_data["z"] = z
        
            #Apply a floor to n and T
            for species in ["elc", "ion"]:
                mom_data[species+"M0"][mom_data[species+"M0"] < 0] = 1e12
            for species in ["elc", "ion"]:
                mom_data[species+"Temp"][mom_data[species+"Temp"] < 0] = 10
        
            # Append data
            half_mom_data_list.append(mom_data)
        
        # Construct the 12 block data
        mom_data_list = self.mom_data_list
        even_keys = ["elcM0", "elcTemp", "ionM0", "ionTemp", "phi", "Ri", "gxx", "gzz"]
        odd_keys = ["elcM1", "elcUpar", "ionM1", "ionUpar", "Zi"]
        
        #Unmodified but renumbered blocks
        mom_data_list[0] = half_mom_data_list[0]
        mom_data_list[1] = half_mom_data_list[1]
        mom_data_list[8] = half_mom_data_list[4]
        mom_data_list[9] = half_mom_data_list[5]
        
        # Reflected blocks
        mom_data_list[3] = {}
        mom_data_list[6] = {}
        mom_data_list[4] = {}
        mom_data_list[5] = {}
        #Do Bratio myself manually
        mom_data_list[3]["Bratio"] = mom_data_list[1]["Bratio"]
        mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
        mom_data_list[4]["Bratio"] = mom_data_list[0]["Bratio"]
        mom_data_list[5]["Bratio"] = mom_data_list[9]["Bratio"]
        
        
        
        # Doubled blocks
        mom_data_list[2] = {}
        mom_data_list[7] = {}
        mom_data_list[10] = {}
        mom_data_list[11] = {}
        for key in even_keys:
            mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1]))
            mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1]))
            mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1]))
            mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1]))
        
        for key in odd_keys:
            mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1]))
            mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1]))
            mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1]))
            mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1]))
        
        #Do x myself manually
        mom_data_list[3]["x"] = mom_data_list[1]["x"]
        mom_data_list[6]["x"] = mom_data_list[8]["x"]
        mom_data_list[4]["x"] = mom_data_list[0]["x"]
        mom_data_list[5]["x"] = mom_data_list[9]["x"]
        
        mom_data_list[2]["x"] = mom_data_list[1]["x"]
        mom_data_list[7]["x"] = mom_data_list[8]["x"]
        mom_data_list[10]["x"] = half_mom_data_list[6]["x"]
        mom_data_list[11]["x"] = half_mom_data_list[7]["x"]
        
        
        
        # Flip some and double/reflect some
        for key in even_keys:
            #Upper SOL
            mom_data_list[3][key] = np.flip(mom_data_list[1][key], axis=-1)
            mom_data_list[6][key] = np.flip(mom_data_list[8][key], axis=-1)
            #Upper PF
            mom_data_list[4][key] = np.flip(mom_data_list[0][key], axis=-1)
            mom_data_list[5][key] = np.flip(mom_data_list[9][key], axis=-1)
        
            #Outer Middle
            mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
            mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = np.flip(half_mom_data_list[2][key], axis=-1)
        
            mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
            mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = np.flip(half_mom_data_list[6][key], axis=-1)
        
            #Inner Middle
            mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = np.flip(half_mom_data_list[3][key], axis=-1) 
            mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
        
            mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = np.flip(half_mom_data_list[7][key], axis=-1) 
            mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
        
        for key in odd_keys:
            #Upper SOL
            mom_data_list[3][key] = -np.flip(mom_data_list[1][key], axis=-1)
            mom_data_list[6][key] = -np.flip(mom_data_list[8][key], axis=-1)
            #Upper PF
            mom_data_list[4][key] = -np.flip(mom_data_list[0][key], axis=-1)
            mom_data_list[5][key] = -np.flip(mom_data_list[9][key], axis=-1)
        
            #Outer Middle
            mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
            mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = -np.flip(half_mom_data_list[2][key], axis=-1)
        
            mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
            mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = -np.flip(half_mom_data_list[6][key], axis=-1)
        
            #Inner Middle
            mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = -np.flip(half_mom_data_list[3][key], axis=-1) 
            mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
        
            mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = -np.flip(half_mom_data_list[7][key], axis=-1) 
            mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
        
        
        
        for bi in range(12):
            mom_data_list[bi]["wallflux"] = np.zeros(mom_data_list[bi]["elcM0"].shape)
            mom_data_list[bi]["plateflux"] = np.zeros(mom_data_list[bi]["elcM0"].shape)
        
        # Now look at Particle Flux to Side Wall. Use surface method
        D = 0.22
        edge_inds = [-1, 0,0,0, -1,-1, 0,0,0, -1]
        for bi in [0,1,2,3,4,5,6,7,8,9]:
            diff_density_surf = {}
            for species in ["ion"]:
                diff_density_surf[species] = 0.0
                edge_ind = edge_inds[bi]
                dM0dx = np.gradient(mom_data_list[bi]["ionM0"], mom_data_list[bi]["x"], axis=0, edge_order=2)
                sign = 1
                mom_data_list[bi]["wallflux"][edge_ind] = sign*D*dM0dx[edge_ind]*np.sqrt(mom_data_list[bi]["gxx"][edge_ind])
        
        #Calculate Particle Flux
        total_pflux = 0
        zedge = [0, 0, -1, -1, 0, 0, -1, -1]
        for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
            sign = -1
            mom_data_list[bidx]["plateflux"][:, zedge[bi]] = sign*mom_data_list[bidx]["ionM1"][:, zedge[bi]]*mom_data_list[bidx]["Bratio"]/np.sqrt(mom_data_list[bidx]["gzz"][:, zedge[bi]])
        
        
        self.Rall = np.array([])
        self.Zall = np.array([])
        self.niall = np.array([])
        self.neall = np.array([])
        self.Tiall = np.array([])
        self.Teall = np.array([])
        self.upariall = np.array([])
        self.phiall = np.array([])
        self.GammaRadall = np.array([])
        self.GammaParall = np.array([])
        bmax=12
        for i in range(self.bmin,bmax):
            self.Rall = np.append(self.Rall, mom_data_list[i]["Ri"].flatten())
            self.Zall = np.append(self.Zall, mom_data_list[i]["Zi"].flatten())
            self.niall = np.append(self.niall, mom_data_list[i]["ionM0"].flatten())
            self.neall = np.append(self.neall, mom_data_list[i]["elcM0"].flatten())
            self.Tiall = np.append(self.Tiall, mom_data_list[i]["ionTemp"].flatten())
            self.Teall = np.append(self.Teall, mom_data_list[i]["elcTemp"].flatten())
            self.upariall = np.append(self.upariall, mom_data_list[i]["ionUpar"].flatten())
            self.phiall = np.append(self.phiall, mom_data_list[i]["phi"].flatten())
            self.GammaRadall = np.append(self.GammaRadall, mom_data_list[i]["wallflux"].flatten())
            self.GammaParall = np.append(self.GammaParall, mom_data_list[i]["plateflux"].flatten())
        
        #alldata = np.column_stack((Rall, Zall, niall, neall, Tiall, Teall, upariall, phiall, GammaRadall, GammaParall))
        #np.savetxt("./gkeyll_text_output/ehl2data.txt", alldata,  header='R Z ni ne Ti Te upari phi Gamma_R Gamma_Z', comments='')
        
        #celldata = np.zeros((12,3), dtype="int")
        #for i in range(bmin,bmax):
        #    nR = mom_data_list[i]["ionM0"].shape[0]
        #    nZ = mom_data_list[i]["ionM0"].shape[1]
        #    celldata[i] = np.r_[i, nR, nZ]
        #np.savetxt("./gkeyll_text_output/cells_ehl2data.txt", celldata,  header='blockid nR nZ', comments='', fmt="%d")
        
        print("Processed Gkeyll output for frame %d"%frame)
    
    
   
    def plot_data(self):
        """
        Currently just plots density on the R,Z grid,
        but can be improved later to pass field names
        """
        gR = self.Rall
        gZ = self.Zall
        gvals = self.niall
        points = np.vstack((gR,gZ)).T
        
        
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (5,9))
        markersize = 1.0
        
        gnorm=mpl.colors.LogNorm(vmin=gvals.min(), vmax=gvals.max())
        
        gim = ax.scatter(gR,gZ,c=gvals,cmap='inferno',s=markersize, norm=gnorm)
        
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.0)
        cbar = fig.colorbar(gim, cax=cax, orientation='vertical')
        cbar.set_label(r'$n$', rotation=270, labelpad=25, fontsize=16)
        ax.set_title('Gkeyll')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.axis("tight")
        fig.tight_layout()

    def __find_cell(self, xc, x0):
        return np.argmin(np.abs(x0-xc))

    def eval_basis(self, coeffs, x, y):
        basis = np.r_[1/2, np.sqrt(3)*x/2, np.sqrt(3)*y/2, 3*x*y/2]
        return np.sum(coeffs*basis)

    def interpolate_data(self, ptb):
        nR = ptb.shape[0]
        nZ = ptb.shape[1]
        out = np.zeros((nR, nZ,6))
        #out = np.zeros((nR, nZ))
        for i in range(nR):
            for j in range(nZ):
                psi, theta, block = ptb[i,j]
                block = block.astype(int)
                ip = self.__find_cell(self.grid_list[block]["xc"], psi)
                it = self.__find_cell(self.grid_list[block]["zc"], theta)
                pc = self.grid_list[block]["xc"][ip]
                tc = self.grid_list[block]["zc"][it]
                xlogical = 2*(psi - pc)/np.diff(self.grid_list[block]["x"])[0]
                zlogical = 2*(theta - tc)/np.diff(self.grid_list[block]["z"])[0]
                #out[i,j] = block,ip,it,xlogical, zlogical, self.eval_basis(self.coeff_data_list[block]["ionM1"][ip,it], xlogical, zlogical)
                out[i,j] = self.eval_basis(self.coeff_data_list[block]["ionM0"][ip,it], xlogical, zlogical)

        out[out < 0] = 1e12

        return out


