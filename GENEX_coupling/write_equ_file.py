import numpy as np
from pathlib import Path


def _write_fortran_5e15_8(fh, values):
    """
    Equivalent to Fortran:
        8000 format(5(3x,e15.8))
    """
    values = list(values)
    for i in range(0, len(values), 5):
        line = "".join(f"{float(v):18.8e}" for v in values[i:i+5])
        fh.write(line + "\n")


def _write_equ_dg(
    filename,
    r,
    z,
    psi,
    psib,
    btf,
    rtf,
):
    """
    Write SOLPS/DivGeo-compatible .equ file.

    Parameters
    ----------
    filename : str or Path
        Output .equ filename.
    r : array-like, shape (nr,)
        Radial grid coordinates [m].
    z : array-like, shape (nz,)
        Vertical grid coordinates [m].
    psi : array-like, shape (nr, nz)
        Poloidal flux on grid [Wb/rad], indexed psi[i_R, i_Z].
    psib : float
        Poloidal flux at separatrix [Wb/rad].
    btf : float
        Toroidal magnetic field [T].
    rtf : float
        Major radius at which btf is specified [m].

    Notes
    -----
    Writes psi as:
        ((psi(j,k)-psib,j=1,jm),k=1,km)

    i.e. R index varies fastest, then Z index.
    """

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    psi = np.asarray(psi, dtype=float)

    nr = r.size
    nz = z.size

    if psi.shape != (nr, nz):
        raise ValueError(
            f"psi shape {psi.shape} does not match (len(r), len(z)) = {(nr, nz)}"
        )

    filename = Path(filename)

    with filename.open("w") as fh:
        fh.write("    jm   :=  no. of grid points in radial direction;\n")
        fh.write("    km   :=  no. of grid points in vertical direction;\n")
        fh.write("    r    :=  radial   coordinates of grid points  [m];\n")
        fh.write("    z    :=  vertical coordinates of grid points  [m];\n")
        fh.write("    psi  :=  flux per radian at grid points      [Wb/rad];\n")
        fh.write("    psib :=  psi at plasma boundary              [Wb/rad];\n")
        fh.write("    btf  :=  toroidal magnetic field                  [T];\n")
        fh.write("    rtf  :=  major radius at which btf is specified   [m];\n")
        fh.write("\n")
        fh.write("\n")

        fh.write(f"    jm    = {nr:12d};\n")
        fh.write(f"    km    = {nz:12d};\n")
        fh.write(f"    psib  = {0.0:18.8e} Wb/rad;\n")
        fh.write(f"    btf   = {float(btf):18.8e} t;\n")
        fh.write(f"    rtf   = {float(rtf):18.8e} m;\n")
        fh.write("\n")

        fh.write("    r(1:jm);\n")
        _write_fortran_5e15_8(fh, r)
        fh.write("\n")

        fh.write("    z(1:km);\n")
        _write_fortran_5e15_8(fh, z)
        fh.write("\n")

        fh.write("      ((psi(j,k)-psib,j=1,jm),k=1,km)\n")

        # Fortran loop:
        #   write(lun,8000) ((pfm(i,j)-psib,i=1,nr),j=1,nz)
        #
        # This means R index i varies fastest, then Z index j.
        psi_shifted = psi - float(psib)
        _write_fortran_5e15_8(fh, psi_shifted.ravel(order="F"))

def write_equ_dg(filename, equi):
    r = equi.magnetic_geometry["R"]*1.7344390285254232
    z = equi.magnetic_geometry["Z"]*1.7344390285254232
    psi = np.asarray(equi.magnetic_geometry["psi"]).T/(-2*np.pi)
    psib = np.asarray(equi.poloidal_flux_on_separatrix)/(-2*np.pi)
    btf = np.asarray(equi.magnetic_geometry["axis_Btor"])
    rtf = np.asarray(equi.magnetic_geometry["magnetic_axis_R"])
    _write_equ_dg(filename, r, z, psi, psib, btf, rtf)