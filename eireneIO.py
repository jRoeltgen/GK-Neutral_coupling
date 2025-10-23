import numpy as np
import warnings
import triangle_mesh as triangles
import glob
import scipy.constants as pyconst
import math
import platform 
from pathlib import Path
# for debugging
import pdb

class eirene:
    def __init__(self, filepath=None):
        version = platform.python_version()
        if not(int(version[0]) >= 3 and int(version[2:4])>=10):
            print("Python version 3.10 or greater needed for eirene.load_extra_forts()")
            return
        self.fort44 = {"meta":{}, "neut":{}, "wld":{}, "res":{}}
        self.fort44_expanded = {}
        self.fort46 = {}
        self.triangle_mesh = triangles.triangle_mesh(filepath)
        if(filepath):
            if isinstance(filepath, str):
                filepath = Path(filepath)
            self.read_ft44(filepath / Path("fort.44"))
            self.read_ft46(filepath / Path("fort.46"))
            nx = self.fort44["meta"]["nx"]
            ny = self.fort44["meta"]["ny"]
            ns = self.fort44["meta"]["npls"]
            self.read_ft31(filepath / Path("fort.31"), nx+2, ny+2, ns)
        
    def read_ft44(self, filename):
        print("Ft44Reader: assuming nlwrmsh = 1, nfla = 1.")
        with open(filename, "r") as fid:
            # --- Dimensions
            line = fid.readline()
            nx, ny, ver = map(int, line.split()[0:3])
            label = line.split()[-1]
            self.fort44["meta"]["nx"] = nx
            self.fort44["meta"]["ny"] = ny
            self.fort44["meta"]["ver"] = ver
            self.fort44["meta"]["label"] = label            
            
            if ver not in (20081111, 20160829, 20170328, 20201006):
                raise ValueError("Ft44Reader: unknown fort.44 format version")
            #if ver == 20201006:
            #    warnings.warn("Ft44Reader: format 20201006 not fully tested")

            # species counts
            natm, nmol, nion = map(int, fid.readline().split()[0:3])
            self.fort44["meta"]["natm"] = natm
            self.fort44["meta"]["nmol"] = nmol
            self.fort44["meta"]["nion"] = nion

            self.fort44["meta"]["atom labels"] = []
            self.fort44["meta"]["molecule labels"] = []
            self.fort44["meta"]["ion labels"] = []
            self.fort44["meta"]["plasma labels"] = []
            
            # species labels
            for i in range(natm):
                self.fort44["meta"]["atom labels"].append(fid.readline())
            for i in range(nmol):
                self.fort44["meta"]["molecule labels"].append(fid.readline())
            for i in range(nion):
                self.fort44["meta"]["ion labels"].append(fid.readline())

            neut = {}
            # --- Basic data
            neut["dab2"]     = self.__read_ft_field(fid, ver, "dab2",    (nx, ny, natm))
            neut["tab2"]     = self.__read_ft_field(fid, ver, "tab2",    (nx, ny, natm))
            neut["dmb2"]     = self.__read_ft_field(fid, ver, "dmb2",    (nx, ny, nmol))
            neut["tmb2"]     = self.__read_ft_field(fid, ver, "tmb2",    (nx, ny, nmol))
            neut["dib2"]     = self.__read_ft_field(fid, ver, "dib2",    (nx, ny, nion))
            neut["tib2"]     = self.__read_ft_field(fid, ver, "tib2",    (nx, ny, nion))
            neut["rfluxa"]   = self.__read_ft_field(fid, ver, "rfluxa",  (nx, ny, natm))
            neut["rfluxm"]   = self.__read_ft_field(fid, ver, "rfluxm",  (nx, ny, nmol))
            neut["pfluxa"]   = self.__read_ft_field(fid, ver, "pfluxa",  (nx, ny, natm))
            neut["pfluxm"]   = self.__read_ft_field(fid, ver, "pfluxm",  (nx, ny, nmol))
            neut["refluxa"]  = self.__read_ft_field(fid, ver, "refluxa", (nx, ny, natm))
            neut["refluxm"]  = self.__read_ft_field(fid, ver, "refluxm", (nx, ny, nmol))
            neut["pefluxa"]  = self.__read_ft_field(fid, ver, "pefluxa", (nx, ny, natm))
            neut["pefluxm"]  = self.__read_ft_field(fid, ver, "pefluxm", (nx, ny, nmol))
            neut["emiss"]    = self.__read_ft_field(fid, ver, "emiss",   (nx, ny, 1))
            neut["emissmol"] = self.__read_ft_field(fid, ver, "emissmol",(nx, ny, 1))
            neut["srcml"]    = self.__read_ft_field(fid, ver, "srcml",   (nx, ny, nmol))
            neut["edissml"]  = self.__read_ft_field(fid, ver, "edissml", (nx, ny, nmol))

            # --- Wall loading dimensions
            wld = {}
            nlim, nsts, nstra = map(int, fid.readline().split())
            nstrata = nstra + 1
            self.fort44["meta"]["nlim"] = nlim
            self.fort44["meta"]["nsts"] = nsts
            self.fort44["meta"]["nstra"] = nstra

            wld["wldnek"] = np.zeros((nlim + nsts, nstra + 1))
            wld["wldnep"] = np.zeros((nlim + nsts, nstra + 1))
            wld["wldna"]  = np.zeros((nlim + nsts, natm, nstra + 1))
            wld["ewlda"]  = np.zeros((nlim + nsts, natm, nstra + 1))
            wld["wldnm"]  = np.zeros((nlim + nsts, nmol, nstra + 1))
            wld["ewldm"]  = np.zeros((nlim + nsts, nmol, nstra + 1))
            wld["wldra"]  = np.zeros((nlim + nsts, natm, nstra + 1))            
            wld["wldrm"]  = np.zeros((nlim + nsts, nmol, nstra + 1))

            wld["wldnek"][:,0] = self.__read_ft_field(fid, ver, "wldnek", nlim+nsts)
            wld["wldnep"][:,0] = self.__read_ft_field(fid, ver, "wldnep", nlim+nsts)
            wld["wldna"][:,:,0] = self.__read_ft_field(fid, ver, "wldna", (nlim+nsts,natm))
            wld["ewlda"][:,:,0] = self.__read_ft_field(fid, ver, "ewlda", (nlim+nsts,natm))
            wld["wldnm"][:,:,0] = self.__read_ft_field(fid, ver, "wldnm", (nlim+nsts,nmol))
            wld["ewldm"][:,:,0] = self.__read_ft_field(fid, ver, "ewldm", (nlim+nsts,nmol))

            wld["wall_geometry"] = self.__read_ft_field(fid, ver, "wall_geometry", (4,nlim))
            wld["wldra"][:,:,0] = self.__read_ft_field(fid, ver, "wldra", (nlim+nsts,natm))
            wld["wldrm"][:,:,0] = self.__read_ft_field(fid, ver, "wldrm", (nlim+nsts,nmol))

            if nstra > 1:
                loop_max = nstrata
            else:
                loop_max = 1
            for i in range(1, loop_max):
                wld["wldnek"][:,i] = self.__read_ft_field(fid, ver, "wldnek", nlim+nsts)
                wld["wldnep"][:,i] = self.__read_ft_field(fid, ver, "wldnep", nlim+nsts)
                wld["wldna"][:,:,i] = self.__read_ft_field(fid, ver, "wldna", (nlim+nsts,natm))
                wld["ewlda"][:,:,i] = self.__read_ft_field(fid, ver, "ewlda", (nlim+nsts,natm))
                wld["wldnm"][:,:,i] = self.__read_ft_field(fid, ver, "wldnm", (nlim+nsts,nmol))
                wld["ewldm"][:,:,i] = self.__read_ft_field(fid, ver, "ewldm", (nlim+nsts,nmol))
                wld["wldra"][:,:,i] = self.__read_ft_field(fid, ver, "wldra", (nlim+nsts,natm))
                wld["wldrm"][:,:,i] = self.__read_ft_field(fid, ver, "wldrm", (nlim+nsts,nmol))

            wldpp0 = self.__read_ft_field(fid, ver, "wldpp", (nlim+nsts, -1))
            _, npls = wldpp0.shape
            self.fort44["meta"]["npls"] = npls
            
            wld["wldpp"] = np.zeros((nlim+nsts, npls, nstra+1))
            wld["wldpa"] = np.zeros((nlim+nsts, natm, nstra+1))
            wld["wldpm"] = np.zeros((nlim+nsts, nmol, nstra+1))
            wld["wldpeb"] = np.zeros((nlim+nsts, nstra+1))
            wld["wldspt"] = np.zeros((nlim+nsts, nstra+1))
            wld["wldspta"] = np.zeros((nlim+nsts, natm, nstra+1))
            wld["wldsptm"] = np.zeros((nlim+nsts, nmol, nstra+1))

            wld["wldpp"][:,:,0] = wldpp0
            wld["wldpa"][:,:,0] = self.__read_ft_field(fid, ver, "wldpa", (nlim+nsts, natm))
            wld["wldpm"][:,:,0] = self.__read_ft_field(fid, ver, "wldpm", (nlim+nsts, nmol))
            wld["wldpeb"][:,0] = self.__read_ft_field(fid, ver, "wldpeb", (nlim+nsts))
            wld["wldspt"][:,0] = self.__read_ft_field(fid, ver, "wldspt", (nlim+nsts))
            wld["wldspta"][:,:,0] = self.__read_ft_field(fid, ver, "wldspta", (nlim+nsts, natm))
            wld["wldsptm"][:,:,0] = self.__read_ft_field(fid, ver, "wldsptm", (nlim+nsts, nmol))

            for i in range(1, loop_max):
                wld["wldpp"][:,:,i] = self.__read_ft_field(fid, ver, "wldpp", (nlim+nsts, npls))
                wld["wldpa"][:,:,i] = self.__read_ft_field(fid, ver, "wldpa", (nlim+nsts, natm))
                wld["wldpm"][:,:,i] = self.__read_ft_field(fid, ver, "wldpm", (nlim+nsts, nmol))
                wld["wldpeb"][:,i] = self.__read_ft_field(fid, ver, "wldpeb", (nlim+nsts))
                wld["wldspt"][:,i] = self.__read_ft_field(fid, ver, "wldspt", (nlim+nsts))
                wld["wldspta"][:,:,i] = self.__read_ft_field(fid, ver, "wldspta", (nlim+nsts, natm))
                wld["wldsptm"][:,:,i] = self.__read_ft_field(fid, ver, "wldsptm", (nlim+nsts, nmol))
            wld["isrftype"] = self.__read_ft_field(fid, ver, "isrftype", (nlim+nsts)).astype(int)
            wld["wlarea"] = self.__read_ft_field(fid, ver, "wlarea", (nlim+nsts))
            wld["wlabsrp(A)"] = self.__read_ft_field(fid, ver, "wlabsrp(A)", (nlim+nsts, natm), natm)
            wld["wlabsrp(M)"] = self.__read_ft_field(fid, ver, "wlabsrp(M)", (nlim+nsts, nmol), nmol)
            wld["wlabsrp(I)"] = self.__read_ft_field(fid, ver, "wlabsrp(I)", (nlim+nsts, nion), nion)
            wld["wlabsrp(P)"] = self.__read_ft_field(fid, ver, "wlabsrp(P)", (nlim+nsts, npls), npls, "plasma labels")
            wld["wlpump(A)"] = self.__read_ft_field(fid, ver, "wlpump(A)", (nlim+nsts, natm), natm)
            wld["wlpump(M)"] = self.__read_ft_field(fid, ver, "wlpump(M)", (nlim+nsts, nmol), nmol)
            wld["wlpump(I)"] = self.__read_ft_field(fid, ver, "wlpump(I)", (nlim+nsts, nion), nion)
            wld["wlpump(P)"] = self.__read_ft_field(fid, ver, "wlpump(P)", (nlim+nsts, npls), npls)

            # neutrad
            neut["eneutrad"] = self.__read_ft_field(fid, ver, "eneutrad", (nx, ny, natm))
            neut["emolrad"] = self.__read_ft_field(fid, ver, "emolrad", (nx, ny, nmol))
            neut["eionrad"] = self.__read_ft_field(fid, ver, "eionrad", (nx, ny, nion))
            
            self.__read_eirdiag(fid)
            ncl = self.fort44["meta"]["eirdiag_nds_ind"][-1]
            # Only non-standard surfaces
            res = {}
            res["sarea_res"] = self.__read_ft_field(fid, ver, "sarea_res", (1, ncl))
            res["wldna_res"] = self.__read_ft_field(fid, ver, "wldna_res", (natm, ncl))
            res["wldnm_res"] = self.__read_ft_field(fid, ver, "wldnm_res", (nmol, ncl))
            res["ewlda_res"] = self.__read_ft_field(fid, ver, "ewlda_res", (natm, ncl))
            res["ewldm_res"] = self.__read_ft_field(fid, ver, "ewldm_res", (nmol, ncl))
            res["ewldea_res"] = self.__read_ft_field(fid, ver, "ewldea_res", (natm, ncl))
            res["ewldem_res"] = self.__read_ft_field(fid, ver, "ewldem_res", (nmol, ncl))
            res["ewldrp_res"] = self.__read_ft_field(fid, ver, "ewldrp_res", (1, ncl))
            res["ewldmr_res"] = self.__read_ft_field(fid, ver, "ewldmr_res", (nmol, ncl))
            res["wldspt_res"] = self.__read_ft_field(fid, ver, "wldspt_res", (1, ncl))
            res["wldspta_res"] = self.__read_ft_field(fid, ver, "wldspta_res", (natm, ncl))
            res["wldsptm_res"] = self.__read_ft_field(fid, ver, "wldsptm_res", (nmol, ncl))
            res["wlpump_res(A)"] = self.__read_ft_field(fid, ver, "wlpump_res(A)", (ncl, natm), natm)
            res["wlpump_res(M)"] = self.__read_ft_field(fid, ver, "wlpump_res(M)", (ncl, nmol), nmol)
            res["wlpump_res(I)"] = self.__read_ft_field(fid, ver, "wlpump_res(I)", (ncl, nion), nion)
            res["wlpump_res(P)"] = self.__read_ft_field(fid, ver, "wlpump_res(P)", (ncl, npls), npls)
            res["ewldt_res"] = self.__read_ft_field(fid, ver, "ewldt_res", (ncl, 1))
            # Integrated quantities
            neut_int = {}
            neut_int["pdena_int"] = self.__read_ft_field(fid, ver, "pdena_int", (natm, nstrata))
            neut_int["pdenm_int"] = self.__read_ft_field(fid, ver, "pdenm_int", (nmol, nstrata))
            neut_int["pdeni_int"] = self.__read_ft_field(fid, ver, "pdeni_int", (nion, nstrata))
            neut_int["pdena_int_b2"] = self.__read_ft_field(fid, ver, "pdena_int_b2", (natm, nstrata))
            neut_int["pdenm_int_b2"] = self.__read_ft_field(fid, ver, "pdenm_int_b2", (nmol, nstrata))
            neut_int["pdeni_int_b2"] = self.__read_ft_field(fid, ver, "pdeni_int_b2", (nion, nstrata))
            neut_int["edena_int"] = self.__read_ft_field(fid, ver, "edena_int", (natm, nstrata))
            neut_int["edenm_int"] = self.__read_ft_field(fid, ver, "edenm_int", (nmol, nstrata))
            neut_int["edeni_int"] = self.__read_ft_field(fid, ver, "edeni_int", (nion, nstrata))
            neut_int["edena_int_b2"] = self.__read_ft_field(fid, ver, "edena_int_b2", (natm, nstrata))
            neut_int["edenm_int_b2"] = self.__read_ft_field(fid, ver, "edenm_int_b2", (nmol, nstrata))
            neut_int["edeni_int_b2"] = self.__read_ft_field(fid, ver, "edeni_int_b2", (nion, nstrata))
            
        self.fort44["wld"] = wld
        self.fort44["neut"] = neut
        self.fort44["neut_int"] = neut_int
        self.fort44["res"] = res
        self._expand_field()
        self._combine_strings("atom labels")
        self._combine_strings("molecule labels")
        self._combine_strings("ion labels")
        self._combine_strings("plasma labels")

    def _expand_field(self):
        for key, arr in self.fort44["neut"].items():
            nx, ny, ns = arr.shape
            self.fort44_expanded[key] = np.zeros((nx+2, ny+2, ns))
            for k in range(ns):
                self.fort44_expanded[key][1:-1, 1:-1, k] = arr[:, :, k]
                self.fort44_expanded[key][0, 1:-1, k]    = self.fort44_expanded[key][1, 1:-1, k]
                self.fort44_expanded[key][-1, 1:-1, k]   = self.fort44_expanded[key][-2, 1:-1, k]
                self.fort44_expanded[key][:, 0, k]       = self.fort44_expanded[key][:, 1, k]
                self.fort44_expanded[key][:, -1, k]      = self.fort44_expanded[key][:, -2, k]

    def _combine_strings(self, fieldname):
        new_string=""
        for s in self.fort44["meta"][fieldname]:
            new_string = new_string+s.replace("\n","")
        new_list = new_string.split(" ")
        self.fort44["meta"][fieldname] = [item for item in new_list if item != ""]
        
    def read_ft46(self, filename):
        with open(filename, "r") as fid:
            # --- Dimensions/version
            line = fid.readline()
            ntri, ver = map(int, line.split()[0:2])
            label = line.split()[-1]
            self.fort46["ntri"] = ntri
            self.fort46["ver"] = ver
            self.fort46["label"] = label
            if ver not in (20160513, 20160829, 20170930):
                raise ValueError("Ft46Reader: unknown fort.46 format version")

            # species counts
            natm, nmol, nion = map(int, fid.readline().split()[0:3])
            self.fort46["natm"] = natm
            self.fort46["nmol"] = nmol
            self.fort46["nion"] = nion

            self.fort46["atom labels"] = []
            self.fort46["molecule labels"] = []
            self.fort46["ion labels"] = []
            
            # species labels
            for i in range(natm):
                self.fort46["atom labels"].append(fid.readline())
            for i in range(nmol):
                self.fort46["molecule labels"].append(fid.readline())
            for i in range(nion):
                self.fort46["ion labels"].append(fid.readline())
            
            # --- Basic data
            self.fort46["pdena"]   = self.__read_ft_field(fid, ver, "pdena",  (ntri, natm))*1e6 #m^-3
            self.fort46["pdenm"]   = self.__read_ft_field(fid, ver, "pdenm",  (ntri, nmol))*1e6 #m^-3
            self.fort46["pdeni"]   = self.__read_ft_field(fid, ver, "pdeni",  (ntri, nion))*1e6 #m^-3
            self.fort46["edena"]   = self.__read_ft_field(fid, ver, "edena",  (ntri, natm))*1e6*pyconst.elementary_charge #Jm^-3
            self.fort46["edenm"]   = self.__read_ft_field(fid, ver, "edenm",  (ntri, nmol))*1e6*pyconst.elementary_charge
            self.fort46["edeni"]   = self.__read_ft_field(fid, ver, "edeni",  (ntri, nion))*1e6*pyconst.elementary_charge
            self.fort46["vxdena"]  = self.__read_ft_field(fid, ver, "vxdena", (ntri, natm))*10 #kg s^-1 m^-2
            self.fort46["vxdenm"]  = self.__read_ft_field(fid, ver, "vxdenm", (ntri, nmol))*10
            self.fort46["vxdeni"]  = self.__read_ft_field(fid, ver, "vxdeni", (ntri, nion))*10
            self.fort46["vydena"]  = self.__read_ft_field(fid, ver, "vydena", (ntri, natm))*10 #kg s^-1 m^-2
            self.fort46["vydenm"]  = self.__read_ft_field(fid, ver, "vydenm", (ntri, nmol))*10
            self.fort46["vydeni"]  = self.__read_ft_field(fid, ver, "vydeni", (ntri, nion))*10
            self.fort46["vzdena"]  = self.__read_ft_field(fid, ver, "vzdena", (ntri, natm))*10 #kg s^-1 m^-2
            self.fort46["vzdenm"]  = self.__read_ft_field(fid, ver, "vzdenm", (ntri, nmol))*10
            self.fort46["vzdeni"]  = self.__read_ft_field(fid, ver, "vzdeni", (ntri, nion))*10
            self.fort46["volumes"] = self.__read_ft_field(fid, ver, "volumes",(ntri, 1))*1e-6 #m^-3

            self.fort46["pux"] = self.__read_ft_field(fid, ver, "pux",(ntri, 1))
            self.fort46["puy"] = self.__read_ft_field(fid, ver, "puy",(ntri, 1))
            self.fort46["pvx"] = self.__read_ft_field(fid, ver, "pvx",(ntri, 1))
            self.fort46["pvy"] = self.__read_ft_field(fid, ver, "pvy",(ntri, 1))


    def read_ft31(self, filename, nx, ny, ns):
        self.fort31 = {}
        with open(filename, 'r') as f:
            # ion density
            self.fort31["na"] = self.__read_ft31_field(f, nx, ny, ns)
            # poloidal velocity
            self.fort31["up"] = self.__read_ft31_field(f, nx, ny, ns)
            # radial velocity
            self.fort31["vv"] = self.__read_ft31_field(f, nx, ny, ns)
            # toroidal velocity
            self.fort31["ww"] = self.__read_ft31_field(f, nx, ny, ns)
            # electron temperature
            self.fort31["te"] = self.__read_ft31_field(f, nx, ny)
            # ion temperature
            self.fort31["ti"] = self.__read_ft31_field(f, nx, ny)
            # pressure
            self.fort31["pr"] = self.__read_ft31_field(f, nx, ny)
            # parallel velocity
            self.fort31["ua"] = self.__read_ft31_field(f, nx, ny, ns)
            # pitch angle
            self.fort31["pitch_angle"] = self.__read_ft31_field(f, nx, ny)
            # Poloidal ion flux (left face)
            self.fort31["fnax"] = self.__read_ft31_field(f, nx, ny, ns)
            # Radial ion flux (bottom face)
            self.fort31["fnay"] = self.__read_ft31_field(f, nx, ny, ns)
            # Poloidal ion heat flux (left face)
            self.fort31["fhix"] = self.__read_ft31_field(f, nx, ny)
            # radial ion heat flux (bottom face)
            self.fort31["fhiy"] = self.__read_ft31_field(f, nx, ny)
            # poloidal electron heat flux (left face)
            self.fort31["fhex"] = self.__read_ft31_field(f, nx, ny)
            # radial electron heat flux (bottom face)
            self.fort31["fhey"] = self.__read_ft31_field(f, nx, ny)
            # Total ion drift velocity (diamagnetic)
            self.fort31["uadia"] = self.__read_ft31_field(f, nx, ny, ns)
            # Total ion drift velocity (radial)
            self.fort31["vadia"] = self.__read_ft31_field(f, nx, ny, ns)
            # Potential
            self.fort31["po"] = self.__read_ft31_field(f, nx, ny)
            # Cell volumes
            self.fort31["vol"] = self.__read_ft31_field(f, nx, ny)
            
            # Magnetic field
            bb4 = self.__read_ft31_field(f, nx, ny)
            bb1 = self.__read_ft31_field(f, nx, ny)
            bb2 = self.__read_ft31_field(f, nx, ny)
            bb3 = self.__read_ft31_field(f, nx, ny)
            self.fort31["bb"] = np.stack([bb1, bb2, bb3, bb4], axis=-1)
            
            # dummies
            for _ in range(4):
                self.fort31["dummy3D"] = self.__read_ft31_field(f, nx, ny, ns)
            for _ in range(8):
                self.fort31["dummy2D"] = self.__read_ft31_field(f, nx, ny)
            # not sure what these two are
            self.fort31["delta_sheathxb"] = self.__read_ft31_field(f, nx, ny)
            self.fort31["delta_sheathyb"] = self.__read_ft31_field(f, nx, ny)
            # Ion charge
            self.fort31["ion_charge"] = self.__read_ft31_field(f, nx, ny, ns)

    # Load extra eirene files
    #    tria is the eirene triangular mesh
    #    eirene_path is the path to the output files
    #    First index is type of source (1-particle, 2-momentum, 3-energy)
    #    Second index is the type of collision
    #       0 - atom-plasma; 1 - molecule-plasma; 2 - test ion-plasma; 3 - photon-plasma
    #    Third index is the source "species"
    #       0 - Electrons; 1 - Atoms; 2 - Molecules; 3 - Bulk Ions
    def load_extra_forts(self, eirene_path, extension="???"):
        if isinstance(eirene_path, str):
            eirene_path = Path(eirene_path)
       # path = eirene_path / Path("fort."+extension)
        filelist = eirene_path.glob("fort."+extension)
        ntria = len(self.triangle_mesh.cells[:,0])
        self.particle_source = {}
        self.momentum_source = {}
        self.energy_source = {}
        self.extra_source = {}
        file_read = False
        for current_file in filelist:
            file_read = True
            current_source = np.zeros(ntria)
            with open(current_file, 'r') as fid:
                lines_list = fid.read().split('\n')
            x = 1
            header_lines = 8
            Ncells = int(lines_list[header_lines-1].split()[0])-1
            if (Ncells != ntria):
                raise ValueError(f"Number of triangles from mesh unequal to number from {current_file}")
            add_cells = int(lines_list[header_lines-1].split()[4])-1 - Ncells
            start_line = header_lines
            while x*Ncells < len(lines_list):                
                current_source = np.array([float(s.split()[2]) for s in lines_list[start_line:start_line+Ncells]])
                self.__increment_sources(current_source, current_file.suffix, lines_list[start_line-6:start_line-4])
                x += 1
                start_line += add_cells + header_lines + Ncells + 5
                
        if not file_read:
            raise Exception("No sources read.")
            
    def write_ft44(self, filename):
        meta = self.fort44["meta"]
        neut = self.fort44["neut"]
        wld = self.fort44["wld"]
        with open(filename, "w") as fid:
            # Write dimensions and label
            fid.write(f'{meta["nx"]:4d}  {meta["ny"]:4d}  {meta["ver"]:8d}  {meta["label"]:32s}\n')
            # Write number of atoms, molecules, ions
            fid.write(f'{meta["natm"]:4d}  {meta["nmol"]:4d}  {meta["nion"]:4d}\n')
            # Write species labels
            for i in range(meta["natm"]):
                fid.write(f' {meta["atom labels"][i]:8s}\n')
            for i in range(meta["nmol"]):
                fid.write(f' {meta["molecule labels"][i].ljust(8)}\n')
            for i in range(meta["nion"]):
                fid.write(f' {meta["ion labels"][i].ljust(8)}\n')

            self.__write_ft_field(fid, "dab2")
            self.__write_ft_field(fid, "tab2")
            self.__write_ft_field(fid, "dmb2")
            self.__write_ft_field(fid, "tmb2")
            self.__write_ft_field(fid, "dib2")
            self.__write_ft_field(fid, "tib2")
            self.__write_ft_field(fid, "rfluxa")
            self.__write_ft_field(fid, "rfluxm")
            self.__write_ft_field(fid, "pfluxa")
            self.__write_ft_field(fid, "pfluxm")
            self.__write_ft_field(fid, "refluxa")
            self.__write_ft_field(fid, "refluxm")
            self.__write_ft_field(fid, "pefluxa")
            self.__write_ft_field(fid, "pefluxm")
            self.__write_ft_field(fid, "emiss")
            self.__write_ft_field(fid, "emissmol")
            self.__write_ft_field(fid, "srcml")
            self.__write_ft_field(fid, "edissml")
            
            # Write number of wall loading dimensions
            fid.write(f'  {meta["nlim"]:4d}  {meta["nsts"]:4d}  {meta["nstra"]:4d}\n')            
            self.__write_ft_field(fid, "wldnek", 0)
            self.__write_ft_field(fid, "wldnep", 0)
            self.__write_ft_field(fid, "wldna", 0)
            self.__write_ft_field(fid, "ewlda", 0)
            self.__write_ft_field(fid, "wldnm", 0)
            self.__write_ft_field(fid, "ewldm", 0)
            self.__write_ft_field(fid, "wall_geometry")
            self.__write_ft_field(fid, "wldra", 0)
            self.__write_ft_field(fid, "wldrm", 0)
            if meta["nstra"] > 1:
                loop_max = meta["nstra"] + 1
            else:
                loop_max = 1
            for i in range(1, loop_max):
                self.__write_ft_field(fid, "wldnek", i)
                self.__write_ft_field(fid, "wldnep", i)
                self.__write_ft_field(fid, "wldna", i)
                self.__write_ft_field(fid, "ewlda", i)
                self.__write_ft_field(fid, "wldnm", i)
                self.__write_ft_field(fid, "ewldm", i)
                self.__write_ft_field(fid, "wldra", i)
                self.__write_ft_field(fid, "wldrm", i)

            for i in range(0, loop_max):
                self.__write_ft_field(fid, "wldpp", i)
                self.__write_ft_field(fid, "wldpa", i)
                self.__write_ft_field(fid, "wldpm", i)
                self.__write_ft_field(fid, "wldpeb", i)
                self.__write_ft_field(fid, "wldspt", i)
                self.__write_ft_field(fid, "wldspta", i)
                self.__write_ft_field(fid, "wldsptm", i)
            self.__write_ft_field(fid, "isrftype")
            self.__write_ft_field(fid, "wlarea")
            self.__write_ft_field(fid, "wlabsrp(A)", header="atom labels")
            self.__write_ft_field(fid, "wlabsrp(M)", header="molecule labels")
            self.__write_ft_field(fid, "wlabsrp(I)", header="ion labels")
            self.__write_ft_field(fid, "wlabsrp(P)", header="plasma labels")
            self.__write_ft_field(fid, "wlpump(A)", header="atom labels")
            self.__write_ft_field(fid, "wlpump(M)", header="molecule labels")
            self.__write_ft_field(fid, "wlpump(I)", header="ion labels")
            self.__write_ft_field(fid, "wlpump(P)", header="plasma labels")
            self.__write_ft_field(fid, "eneutrad")
            self.__write_ft_field(fid, "emolrad")
            self.__write_ft_field(fid, "eionrad")
            
            self.__write_ft_field(fid, "eirdiag_nds_ind")
            self.__write_ft_field(fid, "eirdiag_nds_typ")
            self.__write_ft_field(fid, "eirdiag_nds_srf")
            self.__write_ft_field(fid, "eirdiag_nds_start")
            self.__write_ft_field(fid, "eirdiag_nds_end")
            self.__write_ft_field(fid, "sarea_res")
            self.__write_ft_field(fid, "wldna_res")
            self.__write_ft_field(fid, "wldnm_res")
            self.__write_ft_field(fid, "ewlda_res")
            self.__write_ft_field(fid, "ewldm_res")
            self.__write_ft_field(fid, "ewldea_res")
            self.__write_ft_field(fid, "ewldem_res")
            self.__write_ft_field(fid, "ewldrp_res")
            self.__write_ft_field(fid, "ewldmr_res")
            self.__write_ft_field(fid, "wldspt_res")
            self.__write_ft_field(fid, "wldspta_res")                                                
            self.__write_ft_field(fid, "wldsptm_res")
            self.__write_ft_field(fid, "wlpump_res(A)", header="atom labels")
            self.__write_ft_field(fid, "wlpump_res(M)", header="molecule labels")
            self.__write_ft_field(fid, "wlpump_res(I)", header="ion labels")
            self.__write_ft_field(fid, "wlpump_res(P)", header="plasma labels")
            self.__write_ft_field(fid, "ewldt_res")
            # Integrated quantities
            self.__write_ft_field(fid, "pdena_int")
            self.__write_ft_field(fid, "pdenm_int")
            self.__write_ft_field(fid, "pdeni_int")
            self.__write_ft_field(fid, "pdena_int_b2")
            self.__write_ft_field(fid, "pdenm_int_b2")
            self.__write_ft_field(fid, "pdeni_int_b2")
            self.__write_ft_field(fid, "edena_int")
            self.__write_ft_field(fid, "edenm_int")
            self.__write_ft_field(fid, "edeni_int")
            self.__write_ft_field(fid, "edena_int_b2")
            self.__write_ft_field(fid, "edenm_int_b2")
            self.__write_ft_field(fid, "edeni_int_b2")            
            
        fid.close()

    def write_ft46(self, filename):
        fort46 = self.fort46
        with open(filename, "w") as fid:
            # Write dimensions and label
            fid.write(f'{fort46["ntri"]:6d}  {fort46["ver"]:8d}  {fort46["label"]:32s}\n')
            # Write number of atoms, molecules, ions
            fid.write(f'{fort46["natm"]:4d}  {fort46["nmol"]:4d}  {fort46["nion"]:4d}\n')
            # Write species labels
            for i in range(fort46["natm"]):
                fid.write(f'{fort46["atom labels"][i]:4s}')
            for i in range(fort46["nmol"]):
                fid.write(f'{fort46["molecule labels"][i]:4s}')
            for i in range(fort46["nion"]):
                fid.write(f'{fort46["ion labels"][i]:4s}')

            self.__write_ft_field(fid, "pdena")
            self.__write_ft_field(fid, "pdenm")
            self.__write_ft_field(fid, "pdeni")
            self.__write_ft_field(fid, "edena")
            self.__write_ft_field(fid, "edenm")
            self.__write_ft_field(fid, "edeni")
            self.__write_ft_field(fid, "vxdena")
            self.__write_ft_field(fid, "vxdenm")
            self.__write_ft_field(fid, "vxdeni")
            self.__write_ft_field(fid, "vydena")
            self.__write_ft_field(fid, "vydenm")
            self.__write_ft_field(fid, "vydeni")
            self.__write_ft_field(fid, "vzdena")
            self.__write_ft_field(fid, "vzdenm")
            self.__write_ft_field(fid, "vzdeni")

            self.__write_ft_field(fid, "volumes")
            self.__write_ft_field(fid, "pux")
            self.__write_ft_field(fid, "puy")
            self.__write_ft_field(fid, "pvx")
            self.__write_ft_field(fid, "pvy")

        fid.close()

    def write_ft31(self, filename):
        with open(filename, 'w') as f:
            # ion density
            self.__write_ft31_field(f, "na")
            # poloidal velocity
            self.__write_ft31_field(f, "up")
            # radial velocity
            self.__write_ft31_field(f, "vv")
            # toroidal velocity
            self.__write_ft31_field(f, "ww")
            # electron temperature
            self.__write_ft31_field(f, "te")
            # ion temperature
            self.__write_ft31_field(f, "ti")
            # pressure
            self.__write_ft31_field(f, "pr")
            # parallel velocity
            self.__write_ft31_field(f, "ua")
            # pitch angle
            self.__write_ft31_field(f, "pitch_angle")
            # Poloidal ion flux (left face)
            self.__write_ft31_field(f, "fnax")
            # Radial ion flux (bottom face)
            self.__write_ft31_field(f, "fnay")
            # Poloidal ion heat flux (left face)
            self.__write_ft31_field(f, "fhix")
            # radial ion heat flux (bottom face)
            self.__write_ft31_field(f, "fhiy")
            # poloidal electron heat flux (left face)
            self.__write_ft31_field(f, "fhex")
            # radial electron heat flux (bottom face)
            self.__write_ft31_field(f, "fhey")
            # Total ion drift velocity (diamagnetic)
            self.__write_ft31_field(f, "uadia")
            # Total ion drift velocity (radial)
            self.__write_ft31_field(f, "vadia")
            # Potential
            self.__write_ft31_field(f, "po")
            # Cell volumes
            self.__write_ft31_field(f, "vol")
            
            # Magnetic field
            self.__write_ft31_field(f, "bb")
            
            # dummies            
            for _ in range(4):
                self.__write_ft31_field(f, "dummy3D")
            for _ in range(8):
                self.__write_ft31_field(f, "dummy2D")
            # not sure what these two are
            self.__write_ft31_field(f, "delta_sheathxb")
            self.__write_ft31_field(f, "delta_sheathyb")
            # Ion charge
            self.__write_ft31_field(f, "ion_charge")


    def __read_ft_field(self, fid, ver, fieldname, dims, num_hentries=0, species_type=None):
        """
        Read a real field from fort.44 file, consistent with MATLAB read_ft44_rfield.
        
        Parameters
        ----------
        fid : file object
           Open text file handle for fort.44
        ver : int
           File format version (>=20160829 has headers)
        fieldname : str
           Name of the field to search for in the file
        dims : tuple[int]
           Shape of the expected array (Fortran-order)
           if a dimension is negative, it is calculated from value read in
    
        Returns
        -------
        np.ndarray
           Array of shape `dims`, with data read in Fortran ordering.
        """
        if isinstance(dims, int):
            has_negative = dims<0
        else:
            has_negative = any(x<0 for x in dims)            
        # --- Version >= 20160829: search for header line
        if ver >= 20160829:
            line = fid.readline()
            while fieldname not in line:
                line = fid.readline()
                if not line:  # EOF reached
                    raise EOFError(f"EOF reached without finding {fieldname}.")

            try:
                numin = int(line.strip().split()[-1])
            except Exception:
                raise ValueError(f"Could not parse size from header for {fieldname}.")
            # When one dimension needs to be calculated
            if has_negative:
                remain_dim = int(-numin/np.prod(dims))
                if( remain_dim<0):
                    raise ValueError(f"Tyring to calculate the size of more than one dimension.")
                ind = list(dims).index(-1)
                dims = tuple(remain_dim if x<0 else x for x in dims)
            # When all values are dimensions are known
            # Consistency check: last token in header line should equal prod(dims)
            else:
                if numin != np.prod(dims):
                    raise ValueError(
                        f"read_ft44_rfield: inconsistent number of input elements "
                        f"for {fieldname} (expected {np.prod(dims)}, found {numin})."
                    )
        # Read extra labels of some wall blocks
        found = []
        while num_hentries>0:
            found.append(fid.readline())
            num_hentries -= 6
        if species_type:
            self.fort44["meta"][species_type] = found

        # --- Read the data block
        count = np.prod(dims)
        data = []
        while len(data) < count:
            line = fid.readline()
            if not line:
                raise EOFError(f"Unexpected EOF while reading data for {fieldname}.")
            data.extend(map(float, line.split()))

        arr = np.array(data[:count])
        if not isinstance(dims,int):
            arr = arr.reshape(dims, order="F")  # Fortran ordering, like MATLAB
        return arr

    def __write_ft_field(self, fid, key, ind=-1, header=False):
        nline = 5
        fort44 = True
        isfloat = True
        format_spec = ' 14.7E'
        nlim = float('inf')
        eirdiag = False
        if key in self.fort44["neut"]:
            arr = self.fort44["neut"][key]
        elif key in self.fort44["meta"]:
            arr = self.fort44["meta"][key]
            eirdiag = True
            first_eirdiag = False
            if key == "eirdiag_nds_ind":
                first_eirdiag = True
        elif key in self.fort44["neut_int"]:
            arr = self.fort44["neut_int"][key]
            nline = 6
            nstrata = self.fort44["meta"]["nstra"] + 1
            arr = arr * 10
            fort44 = False # different formatting
            arr = np.reshape(arr, [nstrata*arr.shape[0],1,1], order='F')
        elif key in self.fort44["wld"] or key in self.fort44["res"]:
            if key in self.fort44["wld"]:
                arr = self.fort44["wld"][key]
                nlim = self.fort44["meta"]["nlim"]
                nsts = self.fort44["meta"]["nsts"]
            else:
                ncl = self.fort44["meta"]["eirdiag_nds_ind"][-1]
                arr = self.fort44["res"][key]
                nline = 6
            if header or key=="wlarea" or key=="isrftype":
                format_spec = '11.5E'
                isfloat = False
                nline = 6
            else:
                arr = arr * 10
                fort44 = False # different formatting
                if key in self.fort44["wld"]:
                    format_spec = ' 14.8E'
                elif key != "ewldt_res":
                    arr = np.reshape(arr, [ncl*arr.shape[0],1,1], order='F')
                
        elif key in self.fort46:
            arr = self.fort46[key] * 10  # x10 for weird formatting
            nx,ny = arr.shape
            ns = 1            
            arr = np.reshape(arr,[nx*ny, 1, ns],order='F')
            nline = 6
            fort44 = False
            match key:
               case "pdena" | "pdenm" | "pdeni":
                  arr = arr/1e6
               case "edena" | "edenm" | "edeni":
                  arr = arr/1e6/pyconst.elementary_charge
               case "vxdena" | "vxdenm" | "vxdeni" | "vydena" | "vydenm"| "vydeni" | "vzdena" | "vzdenm" | "vzdeni":
                  arr = arr/10
               case "volumes":
                  arr = arr*1e6
        if arr.dtype.kind in ('u','i'):
            nline = 18
            format_spec = '2d'
            isfloat = False
        if(ind>-1):
            if(ind>0):
                key = key+f'({ind: 3})'
            else:
                key = key+f'({ind})'
            if(len(arr.shape)==2):
                arr = np.reshape(arr[:,ind],[arr.shape[0],1,1],order='F')
            elif(len(arr.shape)==3):
                arr = np.reshape(arr[:,:,ind],[arr.shape[0],arr.shape[1],1],order='F')
            else:
                raise ValueError(f"Field {key} should have 2 or 3 dimensions")
        if(len(arr.shape)==1):
            arr = np.reshape(arr,[arr.shape[0],1,1],order='F')        
        elif(len(arr.shape)==2):
            arr = np.reshape(arr,[arr.shape[0],arr.shape[1],1],order='F')
        nx,ny,ns = arr.shape
        if not eirdiag:
            fid.write(f"*eirene data field {key:s} with size {nx*ny*ns:6d}\n")
        elif first_eirdiag:
            num = nx+self.fort44["meta"]["eirdiag_nds_typ"].shape[0]+\
                self.fort44["meta"]["eirdiag_nds_srf"].shape[0]+\
                self.fort44["meta"]["eirdiag_nds_start"].shape[0]+\
                self.fort44["meta"]["eirdiag_nds_end"].shape[0]
            fid.write(f"*eirene data field eirdiag with size {num:6d}\n")
        if eirdiag:
            isfloat = False
            nline = 12
            format_spec = '4d'
        if header:
            arr = np.reshape(arr,[nx*ny,1,1],order='F')
            nlim = nlim*ny
            nx,ny,ns = arr.shape    
            nh = len(self.fort44["meta"][header])
            for i in range(0, nh, nline):
                line = ""
                for v in self.fort44["meta"][header][i:min(i+nline, nh)]:
                    line = line+"     "+ f"{v:8s}".ljust(8)
                fid.write(line+"\n")
        if key.lower() == "wall_geometry".lower():
            nline = 8
            format_spec = "8.4f"
            isfloat = False
            arr = np.reshape(arr/10,[nx*ny, 1, ns],order='F')
            nx,ny,ns = arr.shape
            nlim = 2*nx
        # write species
        for k in range(0, ns):
            for j in range(0, ny):
                values = arr[0:min(nlim,nx),j,k]
                values2 = arr[:,j,k]
                i = 0
                offset = 0
                first_time = True
                while i+offset<nx:
                #for i in range(0, nx, nline):
                    line = ""                    
                    for ind,v in enumerate(values[i+offset:min(i+offset+nline,len(values))]):
                        formatted_number = f"{v:{format_spec}}"
                        if isfloat:
                            mantissa, exponent = formatted_number.split('E')                        
                            if fort44: # Denotes it being a fort.44 file
                                formatted_exponent = f"{int(exponent):+04d}"
                            else:
                                mantissa = mantissa[:-1].replace(".","").replace("-","-0.").replace(" "," 0.")
                                formatted_exponent = f"{int(exponent):+03d}"                            
                            final_num = f"{mantissa}E{formatted_exponent}"
                        else:
                            final_num = " "+formatted_number
                        line = line+" "+final_num

                    if (i+nline>nlim and first_time):                        
                        values = values2
                        offset = ind + 1
                        first_time = False
                    else:
                        i=i+nline
                    fid.write(line + "\n")

    def __read_eirdiag(self, fid):
        fieldname = "eirdiag"
        nsts = self.fort44["meta"]["nsts"]
        expect = nsts*5 + 1
        line = fid.readline()
        while fieldname not in line:
            line = fid.readline()
            if not line:  # EOF reached
                raise EOFError(f"EOF reached without finding {fieldname}.")
        try:
            numin = int(line.strip().split()[-1])
        except Exception:
            raise ValueError(f"Could not parse size from header for {fieldname}.")
        if numin != expect:
            raise ValueError(
                f"read_eirdiag: inconsistent number of input elements "
                f"for {fieldname} (expected {expect}, found {numin})."
            )
        self.fort44["meta"]["eirdiag_nds_ind"] = self.__read_eirdiag_helper(fid, nsts+1)
        self.fort44["meta"]["eirdiag_nds_typ"] = self.__read_eirdiag_helper(fid, nsts)
        self.fort44["meta"]["eirdiag_nds_srf"] = self.__read_eirdiag_helper(fid, nsts)
        self.fort44["meta"]["eirdiag_nds_start"] = self.__read_eirdiag_helper(fid, nsts)
        self.fort44["meta"]["eirdiag_nds_end"] = self.__read_eirdiag_helper(fid, nsts)

    def __read_eirdiag_helper(self, f, total):
        nsts = self.fort44["meta"]["nsts"]
        my_list = []
        while nsts>0:
            line = f.readline()
            my_list.extend(line.split())
            nsts = nsts - 12
        # Consistency check
        if len(my_list) != total:
            raise ValueError(f"read_eirdiag_error: inconsistent number of input elements")
        my_list = list(map(int, my_list))
        return np.array(my_list)
            
    def __read_ft31_field(self, f, nx, ny, ns=1, f_is_list=False):
        field = np.zeros((nx, ny, ns))
        cols = 5
        rcols = nx % cols
        nl = int(np.ceil(nx / cols))
        for i in range(ns):
            for j in range(ny):
                values = []
                for k in range(nl):
                    if(f_is_list):
                        line = f.pop()
                    else:
                        line = f.readline()
                    if not line:
                        raise EOFError("Unexpected end of file while reading field.")
                    numbers = []
                    try:
                        numbers = list(map(float, line.split()))
                    except ValueError:
                        my_list = line.split()
                        for v in my_list:
                            if "E" in v:
                                numbers.append(float(v))
                            else:
                                if "+" in v:
                                    numbers.append(float(v.replace("+","E+")))
                                else:
                                    numbers.append(float("E-".join(v.rsplit("-",1))))
                                
                    values.extend(numbers)
                field[:, j, i] = values[:nx]
                    
        if ns == 1:
            field = field[:, :, 0]  # drop the extra dimension if scalar
        return field

    def __write_ft31_field(self, fid, fieldname):
        arr = self.fort31[fieldname]
        if fieldname.lower() == "bb".lower():
            idx = [3,0,1,2]
            arr = arr[:,:,idx]
        dim = arr.shape
        nx = dim[0]
        ny = dim[1]
        if (len(dim)==2):
            ns = 1
            arr = np.reshape(arr, [nx, ny, ns], order='F')
        else:
            ns = dim[2]
        cols = 5
        for i in range(ns):
            for j in range(ny):
                values = arr[:,j,i]
                for k in range(0, nx, cols):
                    for v in values[k:min(k+cols,len(values))]:
                        formatted_number = f"{v: 16.8E}"
                        if (np.abs(v)<1e-100 or np.abs(v)>1e100) and np.abs(v)>0:
                            mantissa, exponent = formatted_number.split('E')
                            formatted_number = " "+mantissa+exponent
                        fid.write(formatted_number)
                    fid.write("\n")

    def __read_header(self, filename, starting_line, lines_to_read):
        lines_list = []
        with open(filename, 'r') as fid:
            for i,line in enumerate(fid):
                if (i>=starting_line and i<(starting_line+lines_to_read)):
                    lines_list.append(line)
        return lines_list
            
    def __read_until_pattern(self, filepath, pattern):
        """
        Reads a file line by line and collects lines into a list
        until a specified pattern is found.
        
        Args:
          filepath (str): The path to the file to read.
          pattern (str): The string pattern to search for.

        Returns:
          list: A list of lines read from the file before the pattern was found.
                If the pattern is not found, all lines are returned.
        """
        started = False
        with open(filepath, 'r') as file:
            for line in file:
                if started:
                    if pattern in line:
                        break  # Stop reading when the pattern is found
                    else:
                        last_line_read = line.strip()  # Add line (without newline char) to the list
                else:
                    if "ADDITIONAL" in line:
                        started = True
        return last_line_read

    def __increment_sources(self, current_source, current_file, info):
        species = info[-1].strip()
        # Add new species to dictionary
        if(species not in self.particle_source.keys()):
            self.particle_source[species] = 0
            self.momentum_source[species] = 0
            self.energy_source[species] = 0
            self.extra_source[species] = 0
           # self.particle_source[species+"_nescl"] = 0
           # self.momentum_source[species+"_nescl"] = 0
           # self.energy_source[species+"_nescl"] = 0
           # self.extra_source[species+"_nescl"] = 0
        # Add to source 
        if(current_file[-3]=='1'):
            self.particle_source[species] += current_source
           # self.particle_source[species+"_nescl"] += current_source
        elif(current_file[-3]=='2'):
            self.momentum_source[species] += current_source
           # self.momentum_source[species+"_nescl"] += current_source
        elif(current_file[-3]=='3'):
            self.energy_source[species] += current_source
           # self.energy_source[species+"_nescl"] += current_source
        else:
            self.extra_source[species] += current_source
           # self.extra_source[species+"_nescl"] += current_source
            

