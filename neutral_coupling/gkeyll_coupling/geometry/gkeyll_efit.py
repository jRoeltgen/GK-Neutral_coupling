# coding: utf-8
import numpy as np
import postgkyl as pg
import math
import re

class gkeyllEFIT:
    def __init__(self, filepath:str, name:str, reflect:bool=True):
        self.filepath = filepath
        self.name = name

        # EQDSK metadata
        self.rz_poly_order = 2 
        self.flux_poly_order = 1
        self.reflect = reflect
        self.nr = 0
        self.nz = 0
        self.rdim = 0.0
        self.zdim = 0.0
        self.rcentr = 0.0
        self.rleft = 0.0
        self.zmid = 0.0
        self.rmaxis = 0.0
        self.zmaxis = 0.0
        self.simag = 0.0
        self.sibry = 0.0
        self.bcentr = 0.0
        self.current = 0.0
        self.rmin = None
        self.rmax = None
        self.zmin = None
        self.zmax = None
        self.fpol = None
        self.fpolprime = None
        self.pprime = None
        self.psizr = None
        self.q_profile = None
        self.r_grid = None
        self.z_grid = None
        self.psisep = None
        self.xpoints = []

        # DG data
        psidata = pg.GData(self.filepath + self.name + '_psi.gkyl')
        self.Rgrid, self.Zgrid = psidata.get_grid()
        self.nR = len(self.Rgrid)-1
        self.nZ = len(self.Zgrid)-1
        self.dR = np.diff(self.Rgrid)[0]
        self.dZ = np.diff(self.Zgrid)[0]
        self.psicoeffs = psidata.get_values()
        self.ncoeff = self.psicoeffs.shape[2]

    def __basis(self, x, y):
        return np.r_[1/2, (np.sqrt(3)*x)/2, (np.sqrt(3)*y)/2, (3*x*y)/2, (3*np.sqrt(5)*(x**2-1/3))/4, (3*np.sqrt(5)*(y**2-1/3))/4, (3*np.sqrt(15)*(x**2*y-y/3))/4, (3*np.sqrt(15)*(x*y**2-x/3))/4, (45*(x**2*y**2-(y**2-1/3)/3-(x**2-1/3)/3-1/9))/8]
    
    def __psifunc(self, idx, xy):
        return np.sum(self.psicoeffs[idx[0], idx[1]]*self.__basis(xy[0],xy[1]), axis = 0)
    
    def __find_cell(self, R, Z):
        idxtemp = math.floor((R - self.Rgrid[0])/self.dR)
        idxtemp = min(idxtemp, self.nR-1)
        idxtemp = max(idxtemp, 0)
        ridx = idxtemp
        idxtemp = math.floor((Z - self.Zgrid[0])/self.dZ)
        idxtemp = min(idxtemp, self.nZ-1)
        idxtemp = max(idxtemp, 0)
        zidx = idxtemp
        rcenter = (self.Rgrid[ridx] + self.Rgrid[ridx+1])/2
        zcenter = (self.Zgrid[zidx] + self.Zgrid[zidx+1])/2
        x = (R - rcenter)*2/self.dR
        y = (Z - zcenter)*2/self.dZ
        idx = [ridx,zidx]
        xy = [x,y]
        return idx, xy
    
    def eval_psi(self,R,Z):
        idx, xy = self.__find_cell(R,Z)
        return self.__psifunc(idx, xy)


    def __read_all_numbers(self, text:str):
        #nums = re.findall(r'[+-]?(?:\\d+\\.\\d*|\\.\\d+|\\d+)(?:[eE][+-]?\\d+)?', text)
        nums = re.findall(r'[+-]?\d+\.\d+e[+-]?\d+', text)
        return [float(x) for x in nums]

    def load_eqdsk(self):
        txt = None
        with open(self.filepath+self.name+'.geqdsk', 'r') as f:
            txt = f.read()
        lines = txt.splitlines()
        if len(lines) == 0:
            raise ValueError("Empty file: " + self.filepath)
        header_line = lines[0][48:]
        header_nums = re.findall(r'\d+', header_line)
        if len(header_nums) >= 3:
            try:
                self.nr = int(header_nums[-2])
                self.nz = int(header_nums[-1])
            except Exception:
                pass
        if not (self.nr and self.nz):
            all_ints = re.findall(r'[-+]?\\d+', txt)
            if len(all_ints) >= 2:
                self.nr = int(all_ints[0])
                self.nz = int(all_ints[1])
        if self.nr <= 0 or self.nz <= 0:
            raise ValueError("Failed to determine nr/nz from file")

        geo_lines = lines[1:5]
        geo = self.__read_all_numbers(''.join(geo_lines))
        if len(geo) < 20:
            raise ValueError("File does not contain expected geometry floats (need 20)")
        self.rdim     = geo[0]
        self.zdim     = geo[1]
        self.rcentr   = geo[2]
        self.rleft    = geo[3]
        self.zmid     = geo[4]
        self.rmaxis   = geo[5]
        self.zmaxis   = 0.0 if self.reflect else geo[6]
        self.simag    = geo[7]
        self.sibry    = geo[17]
        self.bcentr   = geo[9]
        self.current  = geo[10]
        self.zmin = self.zmid - self.zdim/2.0
        self.zmax = self.zmid + self.zdim/2.0
        self.rmin = self.rleft
        self.rmax = self.rleft + self.rdim

    # --- Low-level polynomial evaluation helpers converted from C --- #


    def __eval_laplacian(self, direction: int, z: np.ndarray, f: np.ndarray) -> float:
        """
        direction: 0 or 1
        z: length-2 array-like (z0,z1)
        f: coefficient array (must be indexable like f[8], f[6], etc.)
        """
        z0 = float(z[0])
        z1 = float(z[1])
        if direction == 0:
            return 11.25 * f[8] * z1 * z1 + 5.809475019311125 * f[6] * z1 - 3.75 * f[8] + 3.354101966249685 * f[4]
        elif direction == 1:
            return 11.25 * f[8] * z0 * z0 + 5.809475019311125 * f[7] * z0 - 3.75 * f[8] + 3.354101966249685 * f[5]
        else:
            raise ValueError("dir must be 0 or 1")
    
    
    def __eval_mixedpartial(self, z: np.ndarray, f: np.ndarray) -> float:
        """
        z: length-2 vector, f: coefficients
        """
        z0 = float(z[0])
        z1 = float(z[1])
        return (22.5 * f[8] * z0 * z1
                + 5.809475019311125 * f[7] * z1
                + 5.809475019311125 * f[6] * z0
                + 1.5 * f[3])

    def __eval_grad(self, dir, z, f):
        """
        Parameters
        ----------
        dir : int
            0 for ∂/∂x, 1 for ∂/∂y
        z : array-like, shape (2,)
            Point coordinates [x, y]
        f : array-like, shape (9,)
            Expansion coefficients corresponding to the P2 tensor basis.
        
        Returns
        -------
        grad : float
            Evaluated derivative in the specified direction at z.
        """
        x, y = z

        # Precompute powers
        x2, y2 = x*x, y*y

        if dir == 0:
            phi_x = np.array([ 0.0, np.sqrt(3)/2, 0.0, (3*y)/2, (3*np.sqrt(5)/2)*x/2, 0.0, (3*np.sqrt(15)/4)*(2*x*y - y/3), (3*np.sqrt(15)/4)*(y**2 - 1/3), (45/8)*(2*x*y**2 - 2*x/3)      
            ])
            return np.dot(f, phi_x)

        elif dir == 1:
            phi_y = np.array([ 0.0, 0.0, np.sqrt(3)/2, (3*x)/2, 0.0, (3*np.sqrt(5)/2)*y/2, (3*np.sqrt(15)/4)*(x**2 - 1/3), (3*np.sqrt(15)/4)*(2*x*y - x/3), (45/8)*(2*y*x**2 - 2*y/3) ])
            return np.dot(f, phi_y)

        else:
            raise ValueError("dir must be 0 or 1")
    
    
    # --- Newton-Raphson for locating critical/saddle points --- #
    def __newton_raphson(self, coeffs: np.ndarray, cubics: bool = False,
                       ntrial: int = 100, tol_f: float = 1e-18, tol_x: float = 1e-18):
        """
        Port of the C newton_raphson function.
    
        Parameters
        ----------
        up : object
            The "up" object (GkylEfit-like). Must provide:
              - up.rzbasis with method eval_grad_expand(i, x, coeffs) and eval_expand(x, coeffs)
        coeffs : array-like
            Local coefficient array for the cell (length depends on polynomial order).
        cubics : bool
            If True, use cubic evaluation paths (call evf). If False, use P2 tensors (the helper functions above).
        Returns
        -------
        (success: bool, xsol: np.ndarray(length 2))
        """
        n = 2
        x = np.array([0.0, 0.0], dtype=float)
        dx = np.array([0.0, 0.0], dtype=float)
        fvec = np.zeros(2, dtype=float)
        p = np.zeros(2, dtype=float)
        fjac = np.zeros((2, 2), dtype=float)
        fjac_inv = np.zeros((2, 2), dtype=float)
    
        xsol = np.zeros(2, dtype=float)
    
        for niter in range(ntrial):
            # Fill fvec (gradient)
            for i in range(n):
                fvec[i] = self.__eval_grad(i, x, coeffs)
            fjac[0, 0] = self.__eval_laplacian(0, x, coeffs)
            fjac[0, 1] = self.__eval_mixedpartial(x, coeffs)
            fjac[1, 0] = self.__eval_mixedpartial(x, coeffs)
            fjac[1, 1] = self.__eval_laplacian(1, x, coeffs)
    
            # compute norm of fvec
            errf = float(math.sqrt(float(np.dot(fvec, fvec))))
            if errf <= tol_f:
                xsol[:] = x
                return True, xsol
    
            p[:] = -fvec
    
            # invert 2x2 jacobian manually (same as C)
            det = fjac[0, 0] * fjac[1, 1] - fjac[0, 1] * fjac[1, 0]
            if abs(det) < 1e-300:
                # singular jacobian: fail
                return False, xsol
    
            fjac_inv[0, 0] = fjac[1, 1] / det
            fjac_inv[1, 1] = fjac[0, 0] / det
            fjac_inv[0, 1] = -fjac[0, 1] / det
            fjac_inv[1, 0] = -fjac[1, 0] / det
    
            dx[0] = fjac_inv[0, 0] * p[0] + fjac_inv[0, 1] * p[1]
            dx[1] = fjac_inv[1, 0] * p[0] + fjac_inv[1, 1] * p[1]
    
            errx = float(np.sum(np.abs(dx)))
            x[:] = x + dx
    
            if errx <= tol_x:
                xsol[:] = x
                return True, xsol
    
        # no convergence
        return False, xsol
    
    
    # --- Find X-points across the rz grid --- #
    
    def find_xpts(self):
        """
        Port of the C find_xpts function.
        Returns:
          (num_xpts, Rxpt_array, Zxpt_array)
        Side-effects:
          sets psisep, numxpts, xpts if a separatrix psi was found.
        """
        found_xpt = False
        Rsep = 0.0
        Zsep = 0.0
        psisep = float('inf')
    
        # Determine z half-limit when reflect is true (mimic the C condition)
        z_half_limit = self.nZ // 2 + 1 if getattr(self, "reflect", True) else self.nZ
    
        # iterate over cells (mimicking range iterator in C)
        for iz in range(self.nZ):
            # mimic the C check: skip top half cells if reflect and iz >= half
            if iz >= z_half_limit and getattr(self, "reflect", True):
                # skip mirrored part
                continue
            for ir in range(self.nR):
                # fetch coefficients for this cell
                coeffs = self.psicoeffs[ir, iz, :]

    
                # run Newton-Raphson with the coeffs
                status, xsol = self.__newton_raphson(coeffs, cubics=False)
                x0 = float(xsol[0])
                y0 = float(xsol[1])
    
                # evaluate psi at the local expanded point
                psi0 = self.__psifunc([ir,iz], [x0,y0])
    
                if -1.0 <= x0 <= 1.0 and -1.0 <= y0 <= 1.0 and status:
                    found_xpt = True
                    xc0 = (self.Rgrid[ir] + self.Rgrid[ir+1])/2.0
                    xc1 = (self.Zgrid[iz] + self.Zgrid[iz+1])/2.0
                    R0 = self.dR * x0 / 2.0 + xc0
                    Z0 = self.dZ * y0 / 2.0 + xc1
                    #print("Found X-point at R=", R0, " Z=", Z0, " psi=", psi0, "ir,iz = ", ir, iz)
    
                    if abs(psi0 - self.sibry) <= abs(psisep - self.sibry):
                        Rsep = R0
                        Zsep = Z0
                        psisep = psi0
    
        # assemble return arrays and set up.psisep as in C
        if not found_xpt:
            return 0, np.array([]), np.array([])
    
        if getattr(self, "reflect", True):
            num_xpts = 2
            self.Rxpt = np.array([Rsep, Rsep], dtype=float)
            self.Zxpt = np.array([Zsep, -Zsep], dtype=float)
            self.psisep = psisep
        else:
            num_xpts = 1
            self.Rxpt = np.array([Rsep], dtype=float)
            self.Zxpt = np.array([Zsep], dtype=float)
            self.psisep = psisep
    
    

        
