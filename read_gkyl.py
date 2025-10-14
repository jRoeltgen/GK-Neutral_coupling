from enum import Enum
import pandas
import numpy as np
import math

class topology(Enum):
    single_null = 1
    double_null = 2

class gkeyll_data:
    def __init__(self, P_ind_num=None, R_ind_num=None):
        self.topology = topology.double_null
        self.data = {}
        self.blocked_data = {}
        if(P_ind_num and R_ind_num and len(P_ind_num)!=len(R_ind_num)):
            print("Error. Inconsistent number of blocks")
            return
        if(P_ind_num):
            self.poloidal_ind_num = P_ind_num
            self.number_of_blocks = len(P_ind_num)
        if(R_ind_num):
            self.radial_ind_num = R_ind_num

    # Assumes column headers are R, Z, ni, ne, Ti, Te, upari, phi, Gamma_R, Gamma_Z
    # order is irrelevant
    def read_data(self, filename):
        self.data = pandas.read_csv(filename, delim_whitespace=True)

    # Not sure what the format of this would be, so leaving it as a stub
    def read_block_ind(self, filename):
        data = pandas.read_csv(filename, delim_whitespace=True)
        if(any(list(map(math.isnan,data["nR"]))) | any(list(map(math.isnan,data["nZ"])))):
           print("Error in reading number of cells/block")
           return
        self.poloidal_ind_num = data["nZ"]
        self.radial_ind_num = data["nR"]
        self.number_of_blocks = len(data["nR"])

    def regrid_data(self):
        if(self.topology == topology.single_null):
            print("Warning! Single null is untested")
        for key in self.data.keys():
            ind = 0
            self.blocked_data[key] = {}
            for i in range(0,self.number_of_blocks):
                mat = np.array(self.data[key][ind:ind+self.poloidal_ind_num[i]*self.radial_ind_num[i]])
                ind = ind + self.poloidal_ind_num[i]*self.radial_ind_num[i]
                self.blocked_data[key]["block"+str(i)] = mat.reshape([self.poloidal_ind_num[i], self.radial_ind_num[i]],order='F')

    def replace_zero(self):
        for key in self.data.keys():
            if (key.lower() == "Gamma_R".lower() or (key.lower() == "Gamma_Z".lower())):
                for block in self.blocked_data[key].keys():
                    self.blocked_data[key][block][self.blocked_data[key][block]==0] = np.nan
