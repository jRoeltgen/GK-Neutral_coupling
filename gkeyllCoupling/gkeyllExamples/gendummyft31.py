import numpy as np
import B2IO as b2
import eireneIO

def genft31(b2dat, ns, n0=1e19, T0=200):
    nx = b2dat.gmtry["crx"].mean(axis=2).shape[0]
    ny = b2dat.gmtry["crx"].mean(axis=2).shape[1]
    eV = 1.602e-19

    fort31 = {}
    # ion density
    fort31["na"] = np.ones((nx, ny, ns))*n0
    # poloidal velocity
    fort31["up"] = np.zeros((nx, ny, ns))
    # radial velocity
    fort31["vv"] = np.zeros((nx, ny, ns))
    # toroidal velocity
    fort31["ww"] = np.zeros((nx, ny, ns))
    # electron temperature
    fort31["te"] = np.ones((nx, ny))*eV*T0
    # ion temperature
    fort31["ti"] = np.ones((nx, ny))*eV*T0
    # total static pressure
    fort31["pr"] = fort31["na"]*(fort31["te"] + fort31["ti"])
    # parallel velocity
    fort31["ua"] = np.zeros((nx, ny, ns))
    # pitch angle
    # Poloidal ion flux (left face)
    #    energy fluxes and pressure are not directly used by Eirene
    #    only used in Eirene output
    fort31["fnax"] = np.zeros((nx, ny, ns))
    # Radial ion flux (bottom face)
    fort31["fnay"] = np.zeros((nx, ny, ns))
    # Poloidal ion heat flux (left face)
    fort31["fhix"] = np.zeros((nx, ny))
    # radial ion heat flux (bottom face)
    fort31["fhiy"] = np.zeros((nx, ny))
    # poloidal electron heat flux (left face)
    fort31["fhex"] = np.zeros((nx, ny))
    # radial electron heat flux (bottom face)
    fort31["fhey"] = np.zeros((nx, ny))
    # Poloidal ion drift velocity (ExB+Diamagnetic)
    fort31["uadia"] = np.zeros((nx, ny, ns))
    # Radial ion drift velocity (ExB+Diamagnetic)
    fort31["vadia"] = np.zeros((nx, ny, ns))
    # Potential
    fort31["po"] = np.ones((nx, ny))*T0
    # Cell volumes
    fort31["vol"] = b2dat.gmtry['vol']

    # Magnetic field
    fort31["pitch_angle"] = -np.arctan(b2dat.gmtry['bb'][:,:,0]/b2dat.gmtry['bb'][:,:, 2])
    fort31["bb"] = b2dat.gmtry['bb'][3,0,1,2] #Note: original indexing bb[:,:,0]**2 + bb[:,:,2]**2 = bb[:,:,3]**2

    # dummies
    for _ in range(4):
        fort31["dummy3D"] = np.zeros((nx, ny, ns))
    for _ in range(8):
        fort31["dummy2D"] = np.zeros((nx, ny))
    # not sure what these two are
    fort31["delta_sheathxb"] = np.zeros((nx, ny))
    fort31["delta_sheathyb"] = np.zeros((nx, ny))
    # Ion charge
    fort31["ion_charge"] = np.ones((nx, ny, ns))

    return fort31

    def write_ft31_field(fort31, fid, fieldname):
        arr = fort31[fieldname]
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

def write_ft31(fort31, filename):
    with open(filename, 'w') as f:
        # ion density
        write_ft31_field(f, "na")
        # poloidal velocity
        write_ft31_field(f, "up")
        # radial velocity
        write_ft31_field(f, "vv")
        # toroidal velocity
        write_ft31_field(f, "ww")
        # electron temperature
        write_ft31_field(f, "te")
        # ion temperature
        write_ft31_field(f, "ti")
        # pressure
        write_ft31_field(f, "pr")
        # parallel velocity
        write_ft31_field(f, "ua")
        # pitch angle
        write_ft31_field(f, "pitch_angle")
        # Poloidal ion flux (left face)
        write_ft31_field(f, "fnax")
        # Radial ion flux (bottom face)
        write_ft31_field(f, "fnay")
        # Poloidal ion heat flux (left face)
        write_ft31_field(f, "fhix")
        # radial ion heat flux (bottom face)
        write_ft31_field(f, "fhiy")
        # poloidal electron heat flux (left face)
        write_ft31_field(f, "fhex")
        # radial electron heat flux (bottom face)
        write_ft31_field(f, "fhey")
        # Total ion drift velocity (diamagnetic)
        write_ft31_field(f, "uadia")
        # Total ion drift velocity (radial)
        write_ft31_field(f, "vadia")
        # Potential
        write_ft31_field(f, "po")
        # Cell volumes
        write_ft31_field(f, "vol")

        # Magnetic field
        write_ft31_field(f, "bb")

        # dummies
        for _ in range(4):
            write_ft31_field(f, "dummy3D")
        for _ in range(8):
            write_ft31_field(f, "dummy2D")
        # not sure what these two are
        write_ft31_field(f, "delta_sheathxb")
        write_ft31_field(f, "delta_sheathyb")
        # Ion charge
        write_ft31_field(f, "ion_charge")



b2_data_path = '../baserun/'
b2dat = b2.B2(b2_data_path)
fort31 = genft31(b2dat, 1)

write_ft31(fort31, 'fort31'):
