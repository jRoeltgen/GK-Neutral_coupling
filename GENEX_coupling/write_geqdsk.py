import freeqdsk
import numpy as np

def write_eqdsk(equi, filename, comment="", shot=-1):
    offset = 0.0482458132
    sibdry = -np.asarray(equi.poloidal_flux_on_separatrix)/(2*np.pi)
    simagx = -np.asarray(equi.poloidal_flux_on_axis)/(2*np.pi)
    print("equi.poloidal_flux_on_separatrix=",np.asarray(equi.poloidal_flux_on_separatrix))
    print("equi.poloidal_flux_on_axis=",np.asarray(equi.poloidal_flux_on_axis))
    data_dict = equi.data_dictionary
    rdim = data_dict["width"]
    zdim = data_dict["height"]
    rleft = data_dict["r_min"]
    zmid = data_dict["z_midplane"]
    nlim = data_dict["n_limiter_pts"]
    rlim = data_dict["r_limiter"]
    zlim = data_dict["z_limiter"]
    print("rdim, zdim, rleft, zmid=",rdim, zdim, rleft, zmid)
    #print("nlim,rlim,zlim=",nlim,rlim,zlim)

    mag_geo = equi.magnetic_geometry
    psi = -mag_geo["psi"].T/(2*np.pi)
    nr = mag_geo["R"].shape[0]
    nz = mag_geo["Z"].shape[0]
    print("psi shape",psi.shape)
    rcenter = mag_geo["magnetic_axis_R"]
    rmagx = mag_geo["magnetic_axis_R"]
    zmagx = mag_geo["magnetic_axis_Z"]
    #bcentr = mag_geo["axis_Btor"]
    bcentr = 1.6955
    print("nr,nz=",nr,nz)
    print("rcenter, rmagx,zmagx,bcentr=",rcenter, rmagx,zmagx,bcentr)


    nbrdy = data_dict["n_boundary_points"]
    rbdry = data_dict["r_boundary_points"]
    zbdry = data_dict["z_boundary_points"]

    fake_data = np.full(nr, 100.0)
    # Not in equi object
    cpasma = -1 #np.nan
    fpol = fake_data
    pres = fake_data
    qpsi = fake_data
    ffprime = fake_data # np.full(nr, np.nan)
    pprime = fake_data # np.full(nr, np.nan)


    gdat =freeqdsk.geqdsk.GEQDSKFile(comment, shot, nr, nz, rdim, zdim, rcenter,
                                    rleft, zmid, rmagx, zmagx, simagx, sibdry,
                                    bcentr, cpasma, fpol, pres, ffprime, pprime,
                                    psi, qpsi, nbrdy, nlim, rbdry, zbdry, rlim,
                                    zlim)

    with open(filename, "w") as fid:
        freeqdsk.geqdsk.write(gdat, fid, shot=shot)