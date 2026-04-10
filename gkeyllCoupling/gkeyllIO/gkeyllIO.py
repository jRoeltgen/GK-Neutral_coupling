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
    def __init__(self, filepath=None, name=None, half_domain=False, diffusivity=0.5, extra_species = [], fast_reflection=True, final_cu_rec_coeff = 0.95, final_li_recyc_coeff = 0.99):
        # Universal params
        self.mp = 1.67262192e-27
        self.me = 9.1093837e-31
        self.eV = 1.602e-19
        self.mu_0 = 12.56637061435917295385057353311801153679e-7
        self.eps0=8.854187817620389850536563031710750260608e-12
        self.masses = {}
        self.masses["elc"] = self.me
        self.masses["ion"] = 2.014*self.mp
        self.masses["molecule"] = 4.028*self.mp
        self.charges = {}
        self.charges["elc"] = -self.eV 
        self.charges["ion"] = self.eV
        self.charges["molecule"] = self.eV
        self.D = diffusivity
        self.fast_reflection = fast_reflection
        self.final_cu_rec_coeff = final_cu_rec_coeff
        self.final_li_recyc_coeff = final_li_recyc_coeff

        self.species_list = ["elc", "ion"] + extra_species

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

        self.interpolated_data = {}
        self.interpolated_surfr_data = {}
        self.interpolated_surfz_data = {}

        if self.fast_reflection:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            copper_data = np.genfromtxt(dir_path + '/reflection_data/65_DonCu.txt', skip_header=1,delimiter=',')
            copper_data[:,0] = copper_data[:,0]*1000*self.eV # Convert from keV to J
            self.Cuinterpolator = interp1d(copper_data[:,0], copper_data[:,1], bounds_error=False, fill_value='extrapolate')

            lithium_data = np.genfromtxt(dir_path + '/reflection_data/65_DonLi.txt', skip_header=1,delimiter=',')
            lithium_data[:,0] = lithium_data[:,0]*1000*self.eV # Convert from keV to J
            self.Liinterpolator = interp1d(lithium_data[:,0], lithium_data[:,1], bounds_error=False, fill_value='extrapolate')





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

            #Load geometry
            bdata = pg.GData('%s-bmag.gkyl'%sim_name)
            mom_data["B"] = bdata.get_values()
            ci = 0
            for c1 in ["x","y","z"]:
                for c2 in ["x","y","z"]:
                    if( c1=="y" and c2 =="x"):
                        continue
                    if( c1=="z" and c2 !="z"):
                        continue
                    gdata = pg.GData('%s-g_ij.gkyl'%sim_name)
                    grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
                    mom_data["g_%s%s"%(c1,c2)] = gdata.get_values()[:,:,4*ci:4*(ci+1)]
                    ci+=1
            ci = 0
            for c1 in ["x","y","z"]:
                for c2 in ["x","y","z"]:
                    if( c1=="y" and c2 =="x"):
                        continue
                    if( c1=="z" and c2 !="z"):
                        continue
                    gdata = pg.GData('%s-gij.gkyl'%sim_name)
                    mom_data["g%s%s"%(c1,c2)] = gdata.get_values()[:,:,4*ci:4*(ci+1)]
                    ci+=1
            
            jdata = pg.GData('%s-jacobgeo.gkyl'%sim_name)
            mom_data["J"] = jdata.get_values()

            #Load moment data
            for species in self.species_list:
                for mom in ["M0", "M1", "M2"]:
                    mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
                    mom_data[species+mom] = mdata.get_values()
                #Set derived moment data
                #mom_data[species+"Temp"] =  (self.masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / self.eV
                #mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
            # Load the potential
            mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
            mom_data["phi"] = mdata.get_values()

            ncoeffs = mom_data["elcM0"].shape[2]
        
            # Append data
            half_mom_data_list.append(mom_data)
        
        # Construct the 12 block data
        mom_data_list = self.coeff_data_list
        even_keys = [self.species_list[j] + ["M0", "M2"][i] for i in range(2) for j in range(len(self.species_list))] + ["phi" , "gxx", "gzz"]
        odd_keys = [self.species_list[j] + ["M1"][i] for i in range(1) for j in range(len(self.species_list))]
        
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
        
            #Load moment data
            for species in self.species_list:
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
        
            #Save grids
            mom_data["x"] = x
            mom_data["z"] = z
        
            #Apply a floor to n and T
            for species in self.species_list:
                mom_data[species+"M0"][mom_data[species+"M0"] < 0] = 1e12
            for species in self.species_list:
                mom_data[species+"Temp"][mom_data[species+"Temp"] < 0] = 10
        
            # Append data
            half_mom_data_list.append(mom_data)
        
        # Construct the 12 block data
        mom_data_list = self.mom_data_list
        even_keys = [self.species_list[j] + ["M0", "M2", "Temp"][i] for i in range(3) for j in range(len(self.species_list))] + ["phi" , "gxx", "gzz", "Ri"]
        odd_keys = [self.species_list[j] + ["M1", "Upar"][i] for i in range(2) for j in range(len(self.species_list))] + ["Zi"]
        
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
        
        
        
        self.Rall = np.array([])
        self.Zall = np.array([])
        self.niall = np.array([])
        self.neall = np.array([])
        self.Tiall = np.array([])
        self.Teall = np.array([])
        self.upariall = np.array([])
        self.phiall = np.array([])
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
        
        print("Processed Gkeyll output for frame %d"%frame)
    
    
   
    def plot_data(self, field_name, norm=None):
        """
        Currently just plots density on the R,Z grid,
        but can be improved later to pass field names
        """
        gR = self.Rall
        gZ = self.Zall
        points = np.vstack((gR,gZ)).T


        gvals = np.array([])
        bmax=12
        for i in range(self.bmin,bmax):
            gvals = np.append(gvals, self.mom_data_list[i][field_name].flatten())
        
        
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (5,9))
        markersize = 1.0
        
        gim = ax.scatter(gR,gZ,c=gvals,cmap='inferno',s=markersize, norm=norm)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.0)
        cbar = fig.colorbar(gim, cax=cax, orientation='vertical')
        cbar.set_label(r'$n$', rotation=270, labelpad=25, fontsize=16)
        ax.set_title('Gkeyll')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.axis("tight")
        fig.tight_layout()
        return gvals

    def __find_cell(self, xc, x0):
        return np.argmin(np.abs(x0-xc))

    def eval_basis(self, coeffs, x, y):
        basis = np.r_[1/2, np.sqrt(3)*x/2, np.sqrt(3)*y/2, 3*x*y/2]
        return np.sum(coeffs*basis)

    def eval_basis_grad(self, coeffs, x, y, dir):
        if dir == 0:
            basis = np.r_[0, np.sqrt(3)/2, 0, 3*y/2]
        if dir == 1:
            basis = np.r_[0, 0, np.sqrt(3)/2, 3*x/2]
        return np.sum(coeffs*basis)


    def interpolate_data(self, ptb):
        nR = ptb.shape[0]
        nZ = ptb.shape[1]
        keys = [self.species_list[j] + ["M0", "M1", "M2"][i] for i in range(3) for j in range(len(self.species_list))] + ["phi" , "gxx", "gzz"]
        for k, key in enumerate(keys):
            self.interpolated_data[key] = np.zeros((nR, nZ))
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
                    self.interpolated_data[key][i,j] = self.eval_basis(self.coeff_data_list[block][key][ip,it], xlogical, zlogical)

        mingxx = self.interpolated_data["gxx"][self.interpolated_data["gxx"]>0].min()
        self.interpolated_data["gxx"][self.interpolated_data["gxx"]<0] = mingxx

        grad_keys = [self.species_list[j] + ["M0"][i] for i in range(1) for j in range(len(self.species_list))]
        for k, key in enumerate(grad_keys):
            self.interpolated_data[key+"dx"] = np.zeros((nR, nZ))
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
                    self.interpolated_data[key + "dx"][i,j] = self.eval_basis_grad(self.coeff_data_list[block][key][ip,it], xlogical, zlogical, 0) * 2.0/np.diff(self.mom_data_list[block]["x"])[0]



    def interpolate_surfr_data(self, ptb):
        nR = ptb.shape[0]
        nZ = ptb.shape[1]
        keys = [self.species_list[j] + ["M0", "M1", "M2"][i] for i in range(3) for j in range(len(self.species_list))] + ["phi" , "gxx"]
        for k, key in enumerate(keys):
            self.interpolated_surfr_data[key] = np.zeros((nR, nZ))
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
                    self.interpolated_surfr_data[key][i,j] = self.eval_basis(self.coeff_data_list[block][key][ip,it], xlogical, zlogical)


        mingxx = self.interpolated_surfr_data["gxx"][self.interpolated_surfr_data["gxx"]>0].min()
        self.interpolated_surfr_data["gxx"][self.interpolated_surfr_data["gxx"]<0] = mingxx

        grad_keys = [self.species_list[j] + ["M0"][i] for i in range(1) for j in range(len(self.species_list))]
        for k, key in enumerate(grad_keys):
            self.interpolated_surfr_data[key+"dx"] = np.zeros((nR, nZ))
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
                    self.interpolated_surfr_data[key + "dx"][i,j] = self.eval_basis_grad(self.coeff_data_list[block][key][ip,it], xlogical, zlogical, 0) * 2.0/np.diff(self.mom_data_list[block]["x"])[0]

    def interpolate_surfz_data(self, ptb):
        nR = ptb.shape[0]
        nZ = ptb.shape[1]
        keys = [self.species_list[j] + ["M0", "M1", "M2"][i] for i in range(3) for j in range(len(self.species_list))] + ["phi" , "gxx", "gzz"]
        for k, key in enumerate(keys):
            self.interpolated_surfz_data[key] = np.zeros((nR, nZ))
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
                    self.interpolated_surfz_data[key][i,j] = self.eval_basis(self.coeff_data_list[block][key][ip,it], xlogical, zlogical)

        mingzz = self.interpolated_surfz_data["gzz"][self.interpolated_surfz_data["gzz"]>0].min()
        self.interpolated_surfz_data["gzz"][self.interpolated_surfz_data["gzz"]<0] = mingzz




    def calc_derived_data(self, b2dat, edat):
        multi_species_keys = ["na", "ua", "up", "ww", "vv"]
        for mk in multi_species_keys:
            self.interpolated_data[mk] = np.zeros((self.interpolated_data["elcM0"].shape[0], self.interpolated_data["elcM0"].shape[1], len(self.species_list)-1))
        for i in range(1, len(self.species_list)):
            self.interpolated_data["na"][:,:,i-1] = self.interpolated_data[self.species_list[i]+"M0"]
            self.interpolated_data["ua"][:,:,i-1] = -self.interpolated_data[self.species_list[i]+"M1"]/self.interpolated_data[self.species_list[i]+"M0"]
            self.interpolated_data["up"][:,:,i-1] = self.interpolated_data["ua"][:,:,i-1]*-np.sin(edat.fort31["pitch_angle"])
            self.interpolated_data["ww"][:,:,i-1] = self.interpolated_data["ua"][:,:,i-1]*np.cos(edat.fort31["pitch_angle"])
            self.interpolated_data['vv'][:,:,i-1] = self.D*self.interpolated_data[self.species_list[i]+"M0dx"]*np.sqrt(self.interpolated_data["gxx"])/self.interpolated_data[self.species_list[i]+"M0"]

        self.interpolated_data["po"] = self.interpolated_data["phi"]

        self.interpolated_data["te"] =  (self.masses["elc"]/3) * (self.interpolated_data["elcM2"] - self.interpolated_data["elcM1"]**2 / self.interpolated_data["elcM0"])/self.interpolated_data["elcM0"]
        self.interpolated_data["ti"] =  (self.masses["ion"]/3) * (self.interpolated_data["ionM2"] - self.interpolated_data["ionM1"]**2 / self.interpolated_data["ionM0"])/self.interpolated_data["ionM0"]

        # Apply some floors
        self.interpolated_data["na"][self.interpolated_data["na"] < 0] = 1e12
        self.interpolated_data["ti"][self.interpolated_data["ti"] < 0] = 1.0e3*self.eV
        self.interpolated_data["te"][self.interpolated_data["te"] < 0] = 1.0e3*self.eV

        self.interpolated_data["pr"] = self.interpolated_data["na"][:,:,0] * self.interpolated_data["ti"]  + self.interpolated_data["elcM0"] * self.interpolated_data["te"]
        if(len(self.species_list)>2):
            for i in range(2, len(self.species_list)):
                self.interpolated_data["pr"] += self.interpolated_data["na"][:, :, i-1]*self.interpolated_data["ti"]


    def calc_derived_surfr_data(self, b2dat, edat):
        multi_species_keys = ["fnay"]
        for mk in multi_species_keys:
            self.interpolated_surfr_data[mk] = np.zeros((self.interpolated_surfr_data["elcM0"].shape[0], self.interpolated_surfr_data["elcM0"].shape[1], len(self.species_list)-1))
        for i in range(1, len(self.species_list)):
            self.interpolated_surfr_data['fnay'][:,:,i-1] = self.D*self.interpolated_surfr_data[self.species_list[i]+"M0dx"]*np.sqrt(self.interpolated_surfr_data["gxx"])*b2dat.gmtry["vol"]/b2dat.gmtry["hy"]

        if self.fast_reflection:
            self.interpolated_surfr_data["tm"] =  (self.masses["molecule"]/3) * (self.interpolated_surfr_data["moleculeM2"] - self.interpolated_surfr_data["moleculeM1"]**2 / self.interpolated_surfr_data["moleculeM0"])/self.interpolated_surfr_data["moleculeM0"]
            self.interpolated_surfr_data["tm"][self.interpolated_surfr_data["tm"] < 0] = 100*self.eV
            self.interpolated_surfr_data["um"] = -self.interpolated_data["moleculeM1"]/self.interpolated_data["moleculeM0"]
            # Energy used for fast reflection which is half of energy
            self.interpolated_surfr_data["em"] = 0.5 * (self.interpolated_surfr_data["phi"]*self.eV + self.interpolated_surfr_data["tm"] + 0.5*self.masses["molecule"]*self.interpolated_surfr_data["um"]*self.interpolated_surfr_data["um"])


    def calc_derived_surfz_data(self, b2dat, edat):
        multi_species_keys = ["fnax"]
        for mk in multi_species_keys:
            self.interpolated_surfz_data[mk] = np.zeros((self.interpolated_surfz_data["elcM0"].shape[0], self.interpolated_surfz_data["elcM0"].shape[1], len(self.species_list)-1))
        for i in range(1, len(self.species_list)):
            self.interpolated_surfz_data['fnax'][:,:,i-1] = self.interpolated_surfz_data[self.species_list[i]+"M1"]*-np.sin(edat.fort31["pitch_angle"])*b2dat.gmtry["vol"]/b2dat.gmtry["hx"]


        if self.fast_reflection:
            self.interpolated_surfz_data["tm"] =  (self.masses["molecule"]/3) * (self.interpolated_surfz_data["moleculeM2"] - self.interpolated_surfz_data["moleculeM1"]**2 / self.interpolated_surfz_data["moleculeM0"])/self.interpolated_surfz_data["moleculeM0"]
            self.interpolated_surfz_data["tm"][self.interpolated_surfz_data["tm"] < 0] = 100*self.eV
            self.interpolated_surfz_data["um"] = -self.interpolated_data["moleculeM1"]/self.interpolated_data["moleculeM0"]
            # Energy used for fast reflection which is half of energy
            self.interpolated_surfz_data["em"] = 0.5 * (self.interpolated_surfz_data["phi"]*self.eV + self.interpolated_surfz_data["tm"] + 0.5*self.masses["molecule"]*self.interpolated_surfz_data["um"]*self.interpolated_surfz_data["um"])


    def populate_ft31(self, b2dat, edat):
        ft31 = edat.fort31

        if self.fast_reflection:
            zero_volume_keys = ["na"]
            for key in zero_volume_keys:
                last_col = np.zeros_like(self.interpolated_data[key][:,:,-1])
                self.interpolated_data[key] = np.dstack((self.interpolated_data[key], last_col, last_col, last_col))
            duplicated_volume_keys = ["ua", "up", "ww", "vv"]
            for key in duplicated_volume_keys:
                last_col = self.interpolated_data[key][:,:,-1].copy()
                self.interpolated_data[key] = np.dstack((self.interpolated_data[key], last_col, last_col, last_col))


        #Volume data
        ft31["na"] = self.interpolated_data["na"]
        #Set up an ionizing core boundary
        core_indices = np.sort(np.concatenate((np.arange(b2dat.gmtry["leftcut"][0],b2dat.gmtry["leftcut"][1]), np.arange(b2dat.gmtry["rightcut"][1],b2dat.gmtry["rightcut"][0]))))
        ft31["na"][core_indices, 0, 0] = 1.0e30;

        ft31["up"] = self.interpolated_data["up"]
        ft31["vv"] = self.interpolated_data["vv"]
        ft31["ww"] = self.interpolated_data["ww"]
        ft31["ua"] = self.interpolated_data["ua"]

        if self.fast_reflection:
            dummy_keys = ["dummy3D", "uadia", "vadia"]
            for key in dummy_keys:
                ft31[key] = np.zeros_like(ft31["na"])
            ft31["ion_charge"] = np.ones_like(ft31["na"])

        if self.fast_reflection:
            CuCoeff = self.Cuinterpolator(self.interpolated_surfz_data['em'])
            LiCoeff = self.Liinterpolator(self.interpolated_surfr_data['em'])
            species2_x = 2.0*CuCoeff
            species2_y = 0.0
            species3_x = self.final_cu_rec_coeff - CuCoeff
            species3_x[species3_x <0] = 0
            species3_y = 1.0
            species4_x = 2.0*LiCoeff
            species4_y = 0.0
            species5_x = self.final_li_recyc_coeff - LiCoeff
            species5_x[species5_x <0] = 0
            species5_y = 0.0
            xcoeffs = [species2_x, species3_x, species4_x, species5_x]
            ycoeffs = [species2_y, species3_y, species4_y, species5_y]

        if self.fast_reflection:
            last_col = self.interpolated_surfz_data['fnax'][:, :, -1].copy()
            self.interpolated_surfz_data['fnax'] = np.dstack((self.interpolated_surfz_data['fnax'], last_col, last_col, last_col))
            last_col = self.interpolated_surfr_data['fnay'][:, :, -1].copy()
            self.interpolated_surfr_data['fnay'] = np.dstack((self.interpolated_surfr_data['fnay'], last_col, last_col, last_col))
            for col in range(1, 5):
                self.interpolated_surfz_data["fnax"][:, :, col] = self.interpolated_surfz_data["fnax"][:, :, col]*xcoeffs[col-1]
                self.interpolated_surfr_data["fnay"][:, :, col] = self.interpolated_surfr_data["fnay"][:, :, col]*ycoeffs[col-1]


        # Surface Data
        ft31["fnax"] = self.interpolated_surfz_data["fnax"]
        ft31["fnay"] = self.interpolated_surfr_data["fnay"]


        # Data with no species index
        ft31["te"] = self.interpolated_data["te"]
        ft31["ti"] = self.interpolated_data["ti"]
        ft31["pr"] = self.interpolated_data["pr"]
        ft31["po"] = self.interpolated_data["po"]






