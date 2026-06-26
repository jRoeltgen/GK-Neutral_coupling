import numpy as np
from pathlib import Path
from eireneIO import eirene
from eirene_interface import status
import triangle_mesh as triangles
import B2IO
from scipy.constants import elementary_charge

class CheckedRunEirene:
    def __init__(self):
        self.edat = eirene()
        self.prev_norm = None
        self.history = []
        self.counter = 1  # for oscillation

    def __call__(self, timeout, eirene_path, command=None):
        # --- Read current GENE-X state (what EIRENE would see) ---
        self.edat.read_ft31(eirene_path / Path("fort.31"), 192, 38, 1)
        self.edat.triangle_mesh = triangles.triangle_mesh(eirene_path)
        self.edat.triangle_mesh.calc_incenter()
        b2dat = B2IO.B2(eirene_path)
        n = self.edat.fort31["na"]

        # --- Density checks (this is your GENE-X validation) ---
        assert np.isfinite(n).all(), "NaNs in GENE-X density"

        norm = np.linalg.norm(n)
        self.history.append(norm)

        if self.prev_norm is not None:
            # ensure density is evolving
            if np.isclose(norm, self.prev_norm, rtol=1e-8):
                raise AssertionError("GENE-X density is not evolving between iterations")

        self.prev_norm = norm

        # --- Build synthetic EIRENE response ---
        R = self.edat.triangle_mesh.incenter[:,0]
        Z = self.edat.triangle_mesh.incenter[:,1]

        self.counter *= -1

        #R0, Z0 = 2.272, 0.0
        R0, Z0 = 2.26, 0.0
        sigmasq_z = 0.01
        sigmasq_r = 0.00003
        dt = 1e-7

        r_plasma = np.mean(b2dat.gmtry["crx"],axis=2)
        z_plasma = np.mean(b2dat.gmtry["cry"],axis=2)
        dist = ((r_plasma - R0)**2 + (z_plasma - Z0)**2).ravel()
        closest_idx = np.argmin(dist)
        amplitude = n.ravel()[closest_idx]*10
        print("Source amplitude=",amplitude)

        gaussian = np.exp(
            -((R - R0)**2) / sigmasq_r
            -((Z - Z0)**2) / sigmasq_z
        )
        gaussian /= np.max(gaussian)

        unit_conversion = elementary_charge/1e6
        source = amplitude * gaussian * self.counter / dt
        print("density=",np.max(n))
        print("Gaussian max",np.max(np.abs(gaussian)))
        print("counter=",self.counter)
        print("dt=",dt)
        print("source max amp (#/m^3s)",np.max(np.abs(source)))
        source *= unit_conversion
        print("source max amp (amp/cm^3s)",np.max(np.abs(source)))
        # --- Source sanity checks ---
        assert np.isfinite(source).all(), "NaNs in synthetic EIRENE source"
        assert not np.all(source == 0), "Zero source generated"

        # --- Spatial structure checks ---
        idx_max = np.argmax(np.abs(source))

        # Check peak location
        dist = np.sqrt((R - R0)**2 + (Z - Z0)**2)
        closest_idx = np.argmin(dist)

        assert idx_max == closest_idx or dist[idx_max] < 2 * np.min(dist[dist >= 0])

        # Check decay: most of domain near zero
        threshold = 1e-5 * np.max(np.abs(source))
        fraction_small = np.mean(np.abs(source) < threshold)

        assert fraction_small > 0.9, "Source not sufficiently localized"

        # --- Write fort files ---
        self.write_fort_source(eirene_path / "fort.100", source, "ELECTRONS")
        self.write_fort_source(eirene_path / "fort.105", source, "ions")

        return status.SUCCESS

    def write_fort_source(self, filepath, values, species_name):
        """
        values: array of length Ntri (triangle-ordered)
        species_name: "ELECTRONS" or ion name
        """
        Ntri = len(values)

        with open(filepath, "w") as f:
            # Header
            f.write(" ========================================================================\n")
            f.write(" ========================================================================\n")
            f.write(" PARTICLE SOURCE (ELECTRONS) FROM ATOM-PLASMA INTERACTION\n")
            f.write(f" {species_name}\n")
            f.write(" AMP*CM**-3\n")
            f.write(" ========================================================================\n")
            f.write(" ========================================================================\n")
            f.write(f"{Ntri+1:10d}           1           1           1{Ntri+1:10d}\n")

            # Body
            for i, val in enumerate(values, start=1):
                f.write(f"{i:10d}    0    {val: .6E}\n")

            # Footer
            f.write("========================================================================\n")
            f.write(" AVERAGE VALUE   0.0000E+00\n")
            f.write("========================================================================\n")
            f.write(" ADDITIONAL CELLS\n")
            f.write("========================================================================\n")