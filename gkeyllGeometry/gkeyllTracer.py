# coding: utf-8
import numpy as np
import postgkyl as pg
import os
import math
import sys
import re
import gkeyllGeometry.gkeyllEFIT as gkeyllEFIT

import scipy.interpolate 
from scipy.interpolate import RegularGridInterpolator, interp1d
import scipy.integrate as sci

import scipy.optimize as sco
import statistics


import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

class gkeyllTracer:
    def __init__(self, efit, gridspec:dict={}):
        self.efit = efit

        #self.rleft = gridspec["rleft"]
        #self.rright = gridspec["rright"]
        #self.rclose = gridspec["rclose"]
        #self.rmin = gridspec["rmin"]
        #self.rmax = gridspec["rmax"]
        #self.ftype = gridspec["ftype"]
        #self.plate_spec = gridspec["plate_spec"]
        #self.plate_func_lower = gridspec["plate_func_lower"]
        #self.plate_func_upper = gridspec["plate_func_upper"]

        self.rleft = gridspec.get("rleft", None)
        self.rright = gridspec.get("rright", None)
        self.rclose = gridspec.get("rclose", None)
        self.zmin = gridspec.get("zmin", None)
        self.zmax = gridspec.get("zmax", None)
        self.zmin_left = gridspec.get("zmin_left", None)
        self.zmin_right = gridspec.get("zmin_right", None)
        self.zmax_left = gridspec.get("zmax_left", None)
        self.zmax_right = gridspec.get("zmax_right", None)
        self.rmin = gridspec.get("rmin", None)
        self.rmax = gridspec.get("rmax", None)
        self.ftype = gridspec.get("ftype", None)
        self.plate_spec = gridspec.get("plate_spec", None)
        self.plate_func_lower = gridspec.get("plate_func_lower", None)
        self.plate_func_upper = gridspec.get("plate_func_upper", None)

        self.arc_ctx = { }
        self.plate_ctx = { }
        self.contour_ctx = { }

    def __calc_roots(self, psi, psi0, Z, xc, dx):
        sol = {}
        sol["nsol"] = 0;
        sol["R"] = np.zeros(4, dtype='float')
        sol["dRdZ"] = np.zeros(4, dtype='float')
        y = (Z-xc[1])/(dx[1]*0.5);
        
        aq = 0.125*(45.0*psi[8]*y**2+23.2379000772445*psi[6]*y-15.0*psi[8]+13.41640786499874*psi[4]);
        bq = 0.125*(23.2379000772445*psi[7]*y**2+12.0*psi[3]*y-7.745966692414834*psi[7]+6.928203230275509*psi[1]) ;
        cq = 0.125*((13.41640786499874*psi[5]-15.0*psi[8])*y**2+(6.928203230275509*psi[2]-7.745966692414834*psi[6])*y+5.0*psi[8]- 4.47213595499958*psi[5]-4.47213595499958*psi[4]+4.0*psi[0] ) - psi0;
        delta2 = bq*bq - 4*aq*cq;
        
        if delta2 > 0 :
            delta = math.sqrt(delta2);
            qq = -0.5*(bq + (bq/math.fabs(bq)) * delta);
            r1 = qq/aq;
            r2 = cq/qq;
            
            sidx = 0;
            if ((-1<=r1) and (r1 < 1)):
                sol["nsol"] += 1;
                sol["R"][sidx] = r1*dx[0]*0.5 + xc[0];
                
                x = r1;
                C = 0.125*(x**2*(90.0*psi[8]*y+23.2379000772445*psi[6])+x*(46.47580015448901*psi[7]*y+12.0*psi[3])+2* (13.41640786499874*psi[5]-15.0*psi[8])*y-7.745966692414834*psi[6]+6.928203230275509*psi[2]) ;
                A = 0.125*(2*x*(45.0*psi[8]*y**2+23.2379000772445*psi[6]*y-15.0*psi[8]+13.41640786499874*psi[4])+23.2379000772445*psi[7]*y**2+12.0*psi[3]*y-7.745966692414834*psi[7]+6.928203230275509*psi[1]); 
                sol["dRdZ"][sidx] = -C/A*dx[0]/dx[1];
                sidx += 1;

            if (-1<=r2) and (r2 < 1) :
                sol["nsol"] += 1;
                sol["R"][sidx] = r2*dx[0]*0.5 + xc[0];
                
                x = r2;
                C = 0.125*(x**2*(90.0*psi[8]*y+23.2379000772445*psi[6])+x*(46.47580015448901*psi[7]*y+12.0*psi[3])+2* (13.41640786499874*psi[5]-15.0*psi[8])*y-7.745966692414834*psi[6]+6.928203230275509*psi[2]) ;
                A = 0.125*(2*x*(45.0*psi[8]*y**2+23.2379000772445*psi[6]*y-15.0*psi[8]+13.41640786499874*psi[4])+23.2379000772445*psi[7]*y**2+12.0*psi[3]*y-7.745966692414834*psi[7]+6.928203230275509*psi[1]); 
                sol["dRdZ"][sidx] = -C/A*dx[0]/dx[1];
                sidx += 1;

        return sol

    def R_psiZ(self, psi:float, Z:float) :
        R = np.zeros(4, dtype='float')
        dRdZ = np.zeros(4, dtype='float')
        idxtemp = math.floor((Z - self.efit.Zgrid[0])/self.efit.dZ)
        idxtemp = min(idxtemp, self.efit.nZ-1)
        idxtemp = max(idxtemp, 0)
        zidx = idxtemp
        
        sidx = 0
        idx = [ 0, zidx]
        dx = [self.efit.dR, self.efit.dZ]
        
        
        for ridx in range(0, self.efit.nR):
            psih = self.efit.psicoeffs[ridx, zidx]
            xc = [ (self.efit.Rgrid[ridx] + self.efit.Rgrid[ridx+1])/2.0, (self.efit.Zgrid[zidx] + self.efit.Zgrid[zidx+1])/2.0]
            sol = self.__calc_roots(psih, psi, Z, xc, dx)
            if sol["nsol"] > 0 : 
                for s in range(0, sol["nsol"]):
                    if sol["R"][s] > self.rmin and sol["R"][s] < self.rmax :
                        R[sidx] = sol["R"][s]
                        dRdZ[sidx] = sol["dRdZ"][s]
                        sidx+=1


        return sidx, R, dRdZ


    def __tok_plate_psi_func(self, s):
        if self.plate_ctx["lower"]:
            RZ = self.plate_func_lower(s)
        else:
            RZ = self.plate_func_upper(s)
        R = RZ[0];
        Z = RZ[1];

        psi = self.efit.eval_psi(R, Z)
        return psi - self.plate_ctx["psi_curr"];

    def __set_upper_plate(self, psi_curr):
        self.plate_ctx["psi_curr"] = psi_curr;
        self.plate_ctx["lower"] = False;
        a = 0;
        b = 1;
        fa = self.__tok_plate_psi_func(a)
        fb = self.__tok_plate_psi_func(b)
        smax = sco.ridder(self.__tok_plate_psi_func, a, b)
        rzplate = self.plate_func_upper(smax);
        self.arc_ctx["zmax"] = rzplate[1];

    def __set_lower_plate(self, psi_curr):
        self.plate_ctx["psi_curr"] = psi_curr;
        self.plate_ctx["lower"] = True;
        a = 0;
        b = 1;
        fa = self.__tok_plate_psi_func(a)
        fb = self.__tok_plate_psi_func(b)
        smin = sco.ridder(self.__tok_plate_psi_func, a, b)
        rzplate = self.plate_func_lower(smin);
        self.arc_ctx["zmin"] = rzplate[1];

    def __contour_func(self, Z) :
        nR, aR, adRdZ = self.R_psiZ(self.contour_ctx["psi"], Z)
        minidx = np.argmin(np.abs(aR - self.contour_ctx["rclose"]))
        dRdZ = adRdZ[minidx]
      
        return math.sqrt(1+dRdZ*dRdZ) if nR > 0 else 0.0;


    def integrate_psi_contour(self, psi:float, zmin, zmax, rclose:float) :
        self.contour_ctx["psi"] = psi
        self.contour_ctx["rclose"] = rclose
        res, err = sci.quad(self.__contour_func, zmin, zmax)
        return res;

    def __ridders_integrate_psi_contour(self, Z, rclose) :
        self.contour_ctx["psi"] = self.arc_ctx['psi']
        self.contour_ctx["rclose"] = rclose
        res, err = sci.quad(self.__contour_func, self.arc_ctx["zmin"], Z)
        return res - self.arc_ctx["arcL_start"]

    def find_upper_turning_point(self, psi_curr, zlo, tolerance=1e-12):
        zup=self.arc_ctx["zmax"].copy()
        zlo_last = zlo
        while(True):
            nlo, R, dR = self.R_psiZ(psi_curr, zlo)
            nup, Rup, dRup = self.R_psiZ(psi_curr, zup)
            if (nup > 0) :
                self.arc_ctx["zmax"] = zup
                self.arc_ctx["rmaxturn"] = Rup[0]
                break;
            if (nlo>=1):
                if(abs(zlo-zup)<tolerance):
                    self.arc_ctx["zmax"] = zlo
                    self.arc_ctx["rmaxturn"] = R[0]
                    break;
                zlo_last = zlo
                zlo = (zlo+zup)/2
            if(nlo==0):
                zup = zlo;
                zlo = zlo_last;
    
    def find_lower_turning_point(self, psi_curr, zup, tolerance=1e-12):
        nup = 0
        zlo=self.arc_ctx["zmin"].copy()
        zup_last = zup
        while(True):
            nup, R, dR = self.R_psiZ(psi_curr, zup)
            nlo, Rlo, dRlo = self.R_psiZ(psi_curr, zlo)
            if (nlo > 0) :
                self.arc_ctx["zmin"] = zlo
                self.arc_ctx["rminturn"] = Rlo[0]
                break;
            if(nup>=1):
                if(abs(zlo-zup)<tolerance):
                    self.arc_ctx["zmin"] = zup
                    self.arc_ctx["rminturn"] = R[0]
                    break;
                zup_last = zup
                zup = (zlo+zup)/2
            if(nup==0):
                zlo = zup
                zup = zup_last
 
    def set_extent(self):
        delta = 1.0e-14
        arc_ctx = self.arc_ctx
        pctx = self.plate_ctx

        if self.ftype in ["GKYL_DN_SOL_OUT", "GKYL_DN_SOL_OUT_LO", "GKYL_DN_SOL_OUT_MID", "GKYL_DN_SOL_OUT_UP"]:
            arc_ctx["rclose"] = self.rright
            if self.plate_spec:
                self.__set_upper_plate(self.efit.psisep)
                self.__set_lower_plate(self.efit.psisep)
            else:
                arc_ctx["zmin"] = self.zmin
                arc_ctx["zmax"] = self.zmax
            zxpt_up =  self.efit.Zxpt[1]
            zxpt_lo =  self.efit.Zxpt[0]
            arcL_tot = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rclose"])
            arcL_lo = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], zxpt_lo, arc_ctx["rclose"])
            arcL_mid = self.integrate_psi_contour(self.efit.psisep, zxpt_lo, zxpt_up, arc_ctx["rclose"])
            arcL_up = self.integrate_psi_contour(self.efit.psisep, zxpt_up, arc_ctx["zmax"], arc_ctx["rclose"])
            if (self.ftype == "GKYL_DN_SOL_OUT"):
                theta_lo = -np.pi+delta
                theta_up = np.pi-delta
            elif (self.ftype == "GKYL_DN_SOL_OUT_LO"):
                theta_lo = -np.pi+delta
                theta_up = -np.pi+delta + arcL_lo/arcL_tot*2.0*np.pi
            elif (self.ftype == "GKYL_DN_SOL_OUT_MID"):
                theta_lo = -np.pi+delta + arcL_lo/arcL_tot*2.0*np.pi
                theta_up =  np.pi-delta - arcL_up/arcL_tot*2.0*np.pi
            elif (self.ftype == "GKYL_DN_SOL_OUT_UP"):
                theta_lo = np.pi-delta - arcL_up/arcL_tot*2.0*np.pi
                theta_up = np.pi-delta
        
        elif self.ftype in ["GKYL_DN_SOL_IN", "GKYL_DN_SOL_IN_LO", "GKYL_DN_SOL_IN_MID", "GKYL_DN_SOL_IN_UP"]:
            arc_ctx["rclose"] = self.rleft
            if self.plate_spec :
                self.__set_upper_plate(self.efit.psisep)
                self.__set_lower_plate(self.efit.psisep)
            else:
                arc_ctx["zmin"] = self.zmin
                arc_ctx["zmax"] = self.zmax
            zxpt_up = self.efit.Zxpt[1]
            zxpt_lo = self.efit.Zxpt[0]
            arcL_tot = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rclose"])
            arcL_lo = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], zxpt_lo, arc_ctx["rclose"])
            arcL_mid = self.integrate_psi_contour(self.efit.psisep, zxpt_lo, zxpt_up, arc_ctx["rclose"])
            arcL_up = self.integrate_psi_contour(self.efit.psisep, zxpt_up, arc_ctx["zmax"], arc_ctx["rclose"])
            if (self.ftype == "GKYL_DN_SOL_IN"):
                theta_lo = -np.pi+delta
                theta_up = np.pi-delta
            elif (self.ftype == "GKYL_DN_SOL_IN_UP"):
                theta_lo = -np.pi+delta
                theta_up = -np.pi+delta + arcL_lo/arcL_tot*2.0*np.pi
            elif (self.ftype == "GKYL_DN_SOL_IN_MID"):
                theta_lo =  -np.pi+delta + arcL_lo/arcL_tot*2.0*np.pi
                theta_up = np.pi-delta - arcL_up/arcL_tot*2.0*np.pi
            elif (self.ftype == "GKYL_DN_SOL_IN_LO"):
                theta_lo = np.pi-delta - arcL_up/arcL_tot*2.0*np.pi
                theta_up = np.pi-delta
        elif self.ftype in ["GKYL_CORE", "GKYL_CORE_R", "GKYL_CORE_L"]:
            arc_ctx["rright"] = self.rright
            arc_ctx["rleft"] = self.rleft
        
            zxpt_up = self.efit.Zxpt[1]
            zxpt_lo = self.efit.Zxpt[0]
            arc_ctx["zmax"] = self.zmax if self.zmax !=None else zxpt_up
            zlo = self.efit.zmaxis;
            self.find_upper_turning_point(self.efit.psisep, zlo)
            arc_ctx["zmin"] = zxpt_lo;
            zup =self.efit.zmaxis;
            self.find_lower_turning_point(self.efit.psisep, zup, arc_ctx["zmin"])
            arc_ctx["right"] = True;
            arcL_r = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rright"])
            arc_ctx["right"] = False
            arcL_l = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rleft"])
            arcL_tot = arcL_l + arcL_r
        
            if (self.ftype == "GKYL_CORE"):
                theta_lo = -np.pi+delta;
                theta_up = np.pi-delta;
            elif (self.ftype == "GKYL_CORE_R"):
                theta_lo = -np.pi+delta
                theta_up =  -np.pi+delta + arcL_r/arcL_tot*2.0*np.pi
            elif (self.ftype == "GKYL_CORE_L") :
                theta_lo =  np.pi-delta - arcL_l/arcL_tot*2.0*np.pi
                theta_up = np.pi-delta
        
        elif self.ftype in ["GKYL_PF_LO_R", "GKYL_PF_LO_L"]:
            arc_ctx["rright"] = self.rright
            arc_ctx["rleft"] = self.rleft
        
            zxpt_lo = self.efit.Zxpt[0];
            arc_ctx["zmax"] = zxpt_lo
            zlo = min(self.zmin_left, self.zmin_right)
            self.find_upper_turning_point(self.efit.psisep, zlo, 1e-15);
        
            if self.plate_spec :
                self.plate_ctx["psi_curr"]=self.efit.psisep
                self.plate_ctx["lower"]=False
                a = 0
                b = 1
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smax = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_upper(smax)
                arc_ctx["zmin_left"] = rzplate[1];
        
        
                self.plate_ctx["lower"]=True;
                a = 0;
                b = 1;
                fa = self.__tok_plate_psi_func(a);
                fb = self.__tok_plate_psi_func(b);
                smin = sco.ridder(self.__tok_plate_psi_func, a, b)
                self.plate_func_lower(smin);
                arc_ctx["zmin_right"] = rzplate[1];
            else:
                arc_ctx["zmin_left"] = self.zmin_left;
                arc_ctx["zmin_right"] = self.zmin_right;
        
            arc_ctx["rclose"] = self.rright;
            arc_ctx["right"] = True;
            arcL_r = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin_right"], arc_ctx["zmax"], arc_ctx["rright"] )
        
            arc_ctx["rclose"] = self.rleft;
            arc_ctx["right"] = False;
            arcL_l = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin_left"], arc_ctx["zmax"], arc_ctx["rleft"])
            arcL_tot = arcL_l + arcL_r;
        
            if (self.ftype == "GKYL_PF_LO_R"):
                theta_lo = -np.pi+delta;
                theta_up = -np.pi+delta + arcL_r/arcL_tot*2.0*np.pi;
            elif (self.ftype == "GKYL_PF_LO_L"):
                theta_lo = np.pi-delta - arcL_l/arcL_tot*2.0*np.pi;
                theta_up = np.pi-delta;
        
        elif self.ftype in ["GKYL_PF_UP_L", "GKYL_PF_UP_R"]:
            arc_ctx["rright"] = self.rright;
            arc_ctx["rleft"] = self.rleft;
            zxpt_up = self.efit.Zxpt[1];
            arc_ctx["zmin"] = zxpt_up;
            zup = max(self.zmax_left,  self.zmax_right);
            self.find_lower_turning_point(self.efit.psisep, zup, 1e-15);
        
            if self.plate_spec :
                self.plate_ctx["psi_curr"]=self.efit.psisep;
                self.plate_ctx["lower"]=False;
                a = 0;
                b = 1;
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smax = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_upper(smax);
                arc_ctx["zmax_right"]= rzplate[1];
        
                self.plate_ctx["lower"]=True;
                a = 0
                b = 1
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smin = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_lower(smin)
                arc_ctx["zmax_left"]= rzplate[1]
            else:
                arc_ctx["zmax_left"] = self.zmax_left
                arc_ctx["zmax_right"] = self.zmax_right
        
            arc_ctx["rclose"] = self.rleft
            arc_ctx["right"] = False
            arcL_l = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], arc_ctx["zmax_left"], arc_ctx["rleft"])
        
            arc_ctx["rclose"] = self.rright
            arc_ctx["right"] = True
            arcL_r = self.integrate_psi_contour(self.efit.psisep, arc_ctx["zmin"], arc_ctx["zmax_right"], arc_ctx["rright"])
            arcL_tot = arcL_r + arcL_l
        
            if (self.ftype == "GKYL_PF_UP_L"):
                theta_lo = -np.pi+delta;
                theta_up = -np.pi+delta + arcL_l/arcL_tot*2.0*np.pi;
            elif (self.ftype == "GKYL_PF_UP_R"):
                theta_lo = np.pi-delta - arcL_r/arcL_tot*2.0*np.pi;
                theta_up = np.pi-delta;

        self.theta_lo = theta_lo
        self.theta_up = theta_up
        


    def find_endpoints(self, psi_curr):
        """
        Python port of tok_find_endpoints. Operates on self.arc_ctx, self.plate_ctx, self.contour_ctx.
        - inp["ftype"] should be a string, e.g. "GKYL_CORE", ...
        - self.efit is used as 'geo' (it must expose fields used below).
        - self.integrate_psi_contour(psi, zmin, zmax, rclose) replaces self.integrate_psi_contour.
        - self.find_upper_turning_point / self.find_lower_turning_point are expected to be methods that
          return the located z (float).
        """
        arc_ctx = self.arc_ctx
        pctx = self.plate_ctx
        contour_ctx = self.contour_ctx
        geo = self.efit  # used in place of the C 'geo' (or geo->efit)

        # Always set psi
        arc_ctx["psi"] = psi_curr

        # --- CORE cases ---
        if self.ftype in ("GKYL_CORE", "GKYL_CORE_R", "GKYL_CORE_L"):
            arc_ctx["rright"] = self.rright
            arc_ctx["rleft"] = self.rleft

            zxpt_up = geo.Zxpt[1]
            zxpt_lo = geo.Zxpt[0]
            arc_ctx["zmax"] = self.zmax if self.zmax != None else zxpt_up  # initial guess

            zlo = geo.zmaxis
            # find_upper_turning_point returns the found zmax (float)
            self.find_upper_turning_point(psi_curr, zlo)

            arc_ctx["zmin"] = zxpt_lo  # initial guess
            zup = geo.zmaxis
            self.find_lower_turning_point(psi_curr, zup)


            arc_ctx["arcL_right"] = self.integrate_psi_contour( psi_curr, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rright"])
            arc_ctx["right"] = False
            arc_ctx["arcL_left"] = self.integrate_psi_contour( psi_curr, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rleft"])
            arc_ctx["arcL_tot"] = arc_ctx["arcL_left"] + arc_ctx["arcL_right"]

            # Adjust theta-start for left/right core subdomains
            if self.ftype == "GKYL_CORE_R":
                theta_extent = self.theta_up - self.theta_lo
                arcL_extent = theta_extent / (2.0 * math.pi) * arc_ctx["arcL_tot"]
                extra_arcL = arcL_extent - arc_ctx["arcL_right"]
                arc_ctx["arcL_start"] = extra_arcL / 2.0

            elif self.ftype == "GKYL_CORE_L":
                theta_extent = 2.0 * math.pi - (self.theta_up - self.theta_lo)
                arcL_extent = theta_extent / (2.0 * math.pi) * arc_ctx["arcL_tot"]
                extra_arcL = arcL_extent - arc_ctx["arcL_right"]
                arc_ctx["arcL_start"] = extra_arcL / 2.0

            # Need to set the location of rstart,zstart associated with arcL_start
            a = arc_ctx["zmin"]
            b = a+0.1 #geo.zmaxis
            fa = - arc_ctx["arcL_start"]
            fb = self.__ridders_integrate_psi_contour(b, arc_ctx["rleft"]) - arc_ctx["arcL_start"]
            arc_ctx["zstart"] = sco.ridder(self.__ridders_integrate_psi_contour, a, b, args = (self.arc_ctx["rleft"]))
            nR, aR, adRdZ = self.R_psiZ(self.arc_ctx["psi"], self.arc_ctx["zstart"])
            minidx = np.argmin(np.abs(aR - self.arc_ctx["rleft"]))
            self.arc_ctx["rstart"] = aR[minidx]

            arc_ctx["right"] = True
            arc_ctx["rclose"] = arc_ctx["rright"]

            if self.ftype == "GKYL_CORE_L":
                arc_ctx["right"] = False
                arc_ctx["rclose"] = self.rleft

        # --- LOWER PF (PF_LO_R / PF_LO_L) ---
        elif self.ftype in ("GKYL_PF_LO_R", "GKYL_PF_LO_L"):
            arc_ctx["rright"] = self.rright
            arc_ctx["rleft"] = self.rleft

            zxpt_lo = geo.Zxpt[0]
            arc_ctx["zmax"] = zxpt_lo  # initial guess
            zlo = min(self.zmin_left, self.zmin_right)
            self.find_upper_turning_point(psi_curr, zlo)

            # zmin left/right come from plates or fixed inputs
            if self.plate_spec :
                pctx["psi_curr"] = psi_curr
                pctx["psi_curr"] = psi_curr
                pctx["lower"] = False
                a = 0
                b = 1
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smax = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_upper(smax)
                arc_ctx["zmin_left"] = rzplate[1]

                pctx["lower"] = True;
                a = 0;
                b = 1;
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smin = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_lower(smin);
                arc_ctx["zmin_right"] = rzplate[1];
            else:
                arc_ctx["zmin_left"] = self.zmin_left
                arc_ctx["zmin_right"] = self.zmin_right

            # compute arc lengths (right and left)
            arc_ctx["rclose"] = self.rright
            arc_ctx["right"] = True
            arc_ctx["arcL_right"] = self.integrate_psi_contour( psi_curr, arc_ctx["zmin_right"], arc_ctx["zmax"], arc_ctx["rright"])

            arc_ctx["rclose"] = self.rleft
            arc_ctx["right"] = False
            arcL_l = self.integrate_psi_contour( psi_curr, arc_ctx["zmin_left"], arc_ctx["zmax"], arc_ctx["rleft"])
            arc_ctx["arcL_tot"] = arcL_l + arc_ctx["arcL_right"]

            if self.ftype == "GKYL_PF_LO_R":
                arc_ctx["right"] = True
                arc_ctx["rclose"] = self.rright
            else:  # GKYL_PF_LO_L
                arc_ctx["right"] = False
                arc_ctx["rclose"] = self.rleft

        # --- UPPER PF (PF_UP_L / PF_UP_R) ---
        elif self.ftype in ("GKYL_PF_UP_L", "GKYL_PF_UP_R"):
            arc_ctx["rright"] = self.rright
            arc_ctx["rleft"] = self.rleft

            zxpt_up = geo.Zxpt[1]
            arc_ctx["zmin"] = zxpt_up  # initial guess
            zup = max(self.zmax_left, self.zmax_right)
            self.find_lower_turning_point(psi_curr, zup)

            # zmax left/right come from plates or fixed inputs
            if self.plate_spec :
                pctx["psi_curr"] = psi_curr
                pctx["lower"] = False
                a = 0;
                b = 1;
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smax = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_upper(smax);
                arc_ctx["zmax_right"] = rzplate[1];

                pctx["lower"] = True
                a = 0;
                b = 1;
                fa = self.__tok_plate_psi_func(a)
                fb = self.__tok_plate_psi_func(b)
                smin = sco.ridder(self.__tok_plate_psi_func, a, b)
                rzplate = self.plate_func_lower(smin);
                arc_ctx["zmax_left"] = rzplate[1];
            else:
                arc_ctx["zmax_left"] = self.zmax_left
                arc_ctx["zmax_right"] = self.zmax_right

            # compute lengths
            arc_ctx["rclose"] = self.rleft
            arc_ctx["right"] = False
            arc_ctx["arcL_left"] = self.integrate_psi_contour( psi_curr, arc_ctx["zmin"], arc_ctx["zmax_left"], arc_ctx["rleft"])

            arc_ctx["rclose"] = self.rright
            arc_ctx["right"] = True
            arcL_r = self.integrate_psi_contour( psi_curr, arc_ctx["zmin"], arc_ctx["zmax_right"], arc_ctx["rright"])
            arc_ctx["arcL_tot"] = arcL_r + arc_ctx["arcL_left"]

            if self.ftype == "GKYL_PF_UP_R":
                arc_ctx["right"] = True
                arc_ctx["rclose"] = self.rright
            else:
                arc_ctx["right"] = False
                arc_ctx["rclose"] = self.rleft

        # --- DN_SOL_OUT variants ---
        elif self.ftype in ("GKYL_DN_SOL_OUT", "GKYL_DN_SOL_OUT_LO", "GKYL_DN_SOL_OUT_MID", "GKYL_DN_SOL_OUT_UP"):
            arc_ctx["rclose"] = self.rright
            if self.plate_spec :
                self.__set_upper_plate(arc_ctx["psi"])
                self.__set_lower_plate(arc_ctx["psi"])
            else:
                arc_ctx["zmin"] = self.zmin
                arc_ctx["zmax"] = self.zmax

            arc_ctx["arcL_tot"] = self.integrate_psi_contour(
                psi_curr, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rclose"]
            )

        # --- DN_SOL_IN variants ---
        elif self.ftype in ("GKYL_DN_SOL_IN", "GKYL_DN_SOL_IN_LO", "GKYL_DN_SOL_IN_MID", "GKYL_DN_SOL_IN_UP"):
            arc_ctx["rclose"] = self.rleft
            if self.plate_spec :
                self.__set_upper_plate(arc_ctx["psi"])
                self.__set_lower_plate(arc_ctx["psi"])
            else:
                arc_ctx["zmin"] = self.zmin
                arc_ctx["zmax"] = self.zmax

            arc_ctx["arcL_tot"] = self.integrate_psi_contour(
                psi_curr, arc_ctx["zmin"], arc_ctx["zmax"], arc_ctx["rclose"]
            )

  
        return

    def arc_length_func(self, Z):
        actx = self.arc_ctx
        psi = actx["psi"]
        rclose = actx["rclose"]
        zmin = actx["zmin"]
        zmax = actx["zmax"]
        ival = 0.0;

    
        if self.ftype in ("GKYL_CORE"):
            if(actx["right"]==True):
                ival = self.integrate_psi_contour(psi, zmin, Z, rclose)
            else:
                ival = self.integrate_psi_contour(psi, Z, zmax, rclose) + actx["arcL_right"]
    
        elif(self.ftype in ("GKYL_CORE_L", "GKYL_CORE_R")):
            if actx["pre"]==True:
                ival = actx["arcL_start"] - integrate_psi_contour(psi, zmin, Z, rclose)
            elif actx["right"]==True:
                ival = self.integrate_psi_contour(psi, zmin, Z, rclose) + actx["arcL_start"]
            else:
                ival = self.integrate_psi_contour(psi, Z, zmax, rclose) + actx["arcL_right"] + actx["arcL_start"]
    
        elif self.ftype in ("GKYL_PF_LO_L", "GKYL_PF_LO_R"):
            if actx["right"]==True:
                ival = self.integrate_psi_contour(psi, zmin, Z, rclose)
            else :
                ival = self.integrate_psi_contour(psi, Z, zmax, rclose)  + actx["arcL_right"]
    
        elif self.ftype in ("GKYL_PF_UP_L", "GKYL_PF_UP_R"):
            if actx["right"]==False:
                ival = self.integrate_psi_contour(psi, Z, zmax, rclose)
            else:
                ival = self.integrate_psi_contour(psi, zmin, Z, rclose) + actx["arcL_left"]
    
        elif self.ftype in ("GKYL_DN_SOL_OUT", "GKYL_DN_SOL_OUT_LO", "GKYL_DN_SOL_OUT_MID", "GKYL_DN_SOL_OUT_UP"):
            ival = self.integrate_psi_contour(psi, zmin, Z, rclose)

        elif self.ftype in ("GKYL_DN_SOL_IN", "GKYL_DN_SOL_IN_LO", "GKYL_DN_SOL_IN_MID", "GKYL_DN_SOL_IN_UP"):
            ival = self.integrate_psi_contour(psi, Z, zmax, rclose)
    
       
        return ival

        





