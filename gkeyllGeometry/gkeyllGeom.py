import numpy as np
import gkeyllGeometry.gkeyllTracer as gkeyllTracer
import gkeyllGeometry.gkeyllEFIT as gkeyllEFIT
import os
import pickle

class gkeyllGeom:
    def __init__(self, load_from=None, efit=None, gridspec:list=None):
        if load_from is not None and os.path.exists(load_from):
             print(f"Loading from pickle: {load_from}")
             with open(load_from, "rb") as f:
                 obj = pickle.load(f)
             # Copy loaded attributes into self
             self.__dict__.update(obj.__dict__)
             return

        self.num_blocks=len(gridspec)
        self.efit = efit
        self.blocks = []
        for gspec in gridspec:
            gt = gkeyllTracer.gkeyllTracer(efit, gspec)
            gt.set_extent()
            self.blocks.append(gt)

    def write(self, path='./gkeyllGeometry/stored_data/gkeyllGeometry.pkl'):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def psitheta(self,R,Z):
        psi = self.efit.eval_psi(R,Z)

        #First determine which block the point is in

        if psi > self.efit.psisep: #Inside
            if Z > self.efit.Zxpt[1]: # Upper PF
                self.blocks[4].find_endpoints(psi)
                if R > self.blocks[4].arc_ctx["rminturn"]:
                    block = 4
                    gt = self.blocks[4]
                    gt.arc_ctx["right"] = True 
                    gt.arc_ctx["rclose"] = gt.arc_ctx["rright"]
                    gt.arc_ctx["zmax"] = gt.arc_ctx["zmax_right"]
                else:
                    self.blocks[5].find_endpoints(psi)
                    block = 5
                    gt = self.blocks[5]
                    gt.arc_ctx["right"] =  False
                    gt.arc_ctx["rclose"] = gt.arc_ctx["rleft"]
                    gt.arc_ctx["zmax"] = gt.arc_ctx["zmax_left"]
            elif Z < self.efit.Zxpt[0]: # Lower PF
                self.blocks[0].find_endpoints(psi)
                if R > self.blocks[0].arc_ctx["rmaxturn"]:
                    block = 0
                    gt = self.blocks[0]
                    gt.arc_ctx["right"] = True 
                    gt.arc_ctx["rclose"] = gt.arc_ctx["rright"]
                    gt.arc_ctx["zmin"] = gt.arc_ctx["zmin_right"]
                else:
                    self.blocks[9].find_endpoints(psi)
                    block = 9
                    gt = self.blocks[9]
                    gt.arc_ctx["right"] =  False
                    gt.arc_ctx["rclose"] = gt.arc_ctx["rleft"]
                    gt.arc_ctx["zmin"] = gt.arc_ctx["zmin_left"]
            else: #Core
                self.blocks[10].find_endpoints(psi)
                if R > self.blocks[10].arc_ctx["rstart"]: #Outer Core
                    block = 10
                    gt = self.blocks[10]
                    gt.arc_ctx["right"] = True if R >= gt.arc_ctx["rminturn"] else False
                    gt.arc_ctx["pre"] = True if R < gt.arc_ctx["rminturn"] and Z < gt.efit.zmaxis else False
                    gt.arc_ctx["rclose"] = gt.arc_ctx["rright"] if R >= gt.arc_ctx["rminturn"] else gt.arc_ctx["rleft"]
                else: #Inner Core
                    self.blocks[11].find_endpoints(psi)
                    block = 11
                    gt = self.blocks[11]
                    gt.arc_ctx["right"] =  False
                    gt.arc_ctx["pre"] = False
                    gt.arc_ctx["rclose"] = gt.arc_ctx["rleft"]



        elif psi  < self.efit.psisep: #Outside
            if R < self.efit.Rxpt[0] : #Inboard SOL
                self.blocks[7].find_endpoints(psi)
                #if Z > self.blocks[7].arc_ctx["zmax"]:
                if Z > self.blocks[7].efit.Zxpt[1]:
                    self.blocks[6].find_endpoints(psi)
                    block = 6
                    gt = self.blocks[6]
                #elif Z > self.blocks[7].arc_ctx["zmin"]:
                elif Z > self.blocks[7].efit.Zxpt[0]:
                    block = 7
                    gt = self.blocks[7]
                else:
                    block = 8
                    self.blocks[8].find_endpoints(psi)
                    gt = self.blocks[8]

            else: #Outboard SOL
                self.blocks[2].find_endpoints(psi)
                #if Z < self.blocks[2].arc_ctx["zmin"]:
                if Z < self.blocks[2].efit.Zxpt[0]:
                    self.blocks[1].find_endpoints(psi)
                    block = 1
                    gt = self.blocks[1]
                #elif Z < self.blocks[2].arc_ctx["zmax"]:
                elif Z < self.blocks[2].efit.Zxpt[1]:
                    block = 2
                    gt = self.blocks[2]
                else:
                    self.blocks[3].find_endpoints(psi)
                    block = 3
                    gt = self.blocks[3]

        arcL = gt.arc_length_func(Z)
        theta = arcL*(2*np.pi/gt.arc_ctx["arcL_tot"]) - np.pi
        return np.r_[psi,theta, block]

