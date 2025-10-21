from enum import Enum
import pandas
import numpy as np
import math
from shapely.geometry import Point, Polygon, MultiPolygon

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
        self.data = pandas.read_csv(filename, sep=r'\s+')

    # Not sure what the format of this would be, so leaving it as a stub
    def read_block_ind(self, filename):
        data = pandas.read_csv(filename, sep=r'\s+')
        if(any(list(map(math.isnan,data["nR"]))) | any(list(map(math.isnan,data["nZ"])))):
           print("Error in reading number of cells/block")
           return
        self.poloidal_ind_num = data["nZ"]
        self.radial_ind_num = data["nR"]
        self.number_of_blocks = len(data["nR"])
        self.connections = {"1stR":data["1stR"], "lastR":data["lastR"],
                            "1stZ":data["1stZ"], "lastZ":data["lastZ"]}

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

    # The x,y indecies are the bottom left corner of the specified polygon,
    #   so the corresponding polygon has points: (x,y), (x+1,y), (x+1, y+1), and (x, y+1)
    def create_polygons(self):
        if len(self.blocked_data)==0:
            self.regrid_data()
        self.blocked_data["polygons"] = {}
        self.blocked_data["xind"] = {}
        self.blocked_data["yind"] = {}
        for block_key in self.blocked_data["R"]:
            blockR = self.blocked_data["R"][block_key]
            blockZ = self.blocked_data["Z"][block_key]
            polygons = []
            xind = []
            yind = []
            nZ, nR = blockR.shape
            for x in range(nZ-1):
                for y in range(nR-1):
                    polygons.append(Polygon([(blockR[x,y],blockZ[x,y]),(blockR[x+1,y],blockZ[x+1,y]),
                                    (blockR[x+1,y+1],blockZ[x+1,y+1]),(blockR[x,y+1],blockZ[x,y+1])]))
                    xind.append(x)
                    yind.append(y)
            self.blocked_data["polygons"][block_key] = polygons
            self.blocked_data["xind"][block_key] = xind
            self.blocked_data["yind"][block_key] = yind
        # Create polygons between blocks
        polygons = []
        xind = []
        yind = []
        for i,v in enumerate(self.connections["1stR"]):
            if v<0:
                continue
            block = "block"+str(i)
            cblock = "block"+str(v)
            if self.connections["1stR"][v] == i:
                idx = 0
            if self.connections["lastR"][v] == i:
                idx = self.radial_ind_num[v]-1
            r = np.vstack([self.blocked_data["R"][block][:,0],self.blocked_data["R"][cblock][:,idx]])
            z = np.vstack([self.blocked_data["Z"][block][:,0],self.blocked_data["Z"][cblock][:,idx]])
            for j in range(self.poloidal_ind_num[i]-1):
                polygons.append(Polygon([(r[0,j],z[0,j]),(r[0,j+1],z[0,j+1]),(r[1,j+1],z[1,j+1]),(r[1,j],z[1,j])]))
                xind.append(-1)
                yind.append(-1)
        lastZ_connections = self.connections["lastZ"]
        for i,v in enumerate(self.connections["1stZ"]):
            if v<0:
                continue
            block = "block"+str(i)
            cblock = "block"+str(v)
            if self.connections["lastZ"][v] == i:
                idx = self.poloidal_ind_num[v]-1
                lastZ_connections[v] = -1
            elif self.connections["1stZ"][v] == i:
                idx = 0            
            r = np.vstack([self.blocked_data["R"][block][0,:],self.blocked_data["R"][cblock][idx,:]])
            z = np.vstack([self.blocked_data["Z"][block][0,:],self.blocked_data["Z"][cblock][idx,:]])
            for j in range(self.radial_ind_num[i]-1):
                polygons.append(Polygon([(r[0,j],z[0,j]),(r[0,j+1],z[0,j+1]),(r[1,j+1],z[1,j+1]),(r[1,j],z[1,j])]))
                xind.append(-1)
                yind.append(-1)                
        self.blocked_data["polygons"]["block99999"] = polygons
        self.blocked_data["xind"]["block99999"] = xind
        self.blocked_data["yind"]["block99999"] = yind


