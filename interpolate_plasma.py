import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.ndimage import uniform_filter
import scipy.constants as pyconst
import B2IO as b2
import eireneIO as eirene
import read_gkyl as gkyl
import pandas
#For debugging
import pdb

# --- helper: MATLAB smoothdata (simple 2D uniform smoothing) ---
def smooth2d(arr, size=5, axes=0):
    # mirror 'smoothdata' behavior loosely with a small uniform filter
    # (adjust 'size' if you want stronger/weaker smoothing)
    if arr.ndim == 1:
        # 1D path if ever needed
        k = max(1, int(size))
        return uniform_filter(arr.astype(float), size=k, mode='nearest', axes=axes)
    elif arr.ndim == 2:
        if axes>0:
            k = (max(1, int(size)), max(1, int(size)))
        else:
            k = max(1, int(size))
        return uniform_filter(arr.astype(float), size=k, mode='nearest', axes=axes)
    else:
        raise ValueError("smooth2d expects 1D or 2D array.")

    # Isn't griddat from scipy.interpolate an almost exact equivalent?
# --- helper: scatteredInterpolant(linear,nearest) equivalent ---
def scattered_interp_linear_with_nearest_fallback(x, y, v):
    """
    Returns a function F(X, Y) that performs a linear scattered interpolation
    with nearest fallback for points outside the convex hull (or NaNs).
    """
    if len(x)==0: # Array is empty
        def F(X,Y):
            return []
        return F
    pts = np.column_stack([x, y])
    lin = LinearNDInterpolator(pts, v, fill_value=np.nan)
    nei = NearestNDInterpolator(pts, v)

    def F(X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        vals = lin(X, Y)
        mask = ~np.isfinite(vals)
        if np.any(mask):
            vals[mask] = nei(X[mask], Y[mask])
        return vals
    return F

# --- replace_negative: fills (i,j) where all ni/Ti/Te invalid using nearest neighbors / scattered interp ---
def replace_negative(Rind, Zind, gkylR, gkylZ, field, maskAll):
    R1 = Rind
    R2 = Rind
    Z1 = Zind
    Z2 = Zind
    scale = 1e-5
    x = []
    y = []

    # search in R (row) direction
    while R1 >= 0 and maskAll[R1, Zind]:
        R1 -= 1
    while R2 < maskAll.shape[0] and maskAll[R2, Zind]:
        R2 += 1
    # search in Z (col) direction
    while Z1 >= 0 and maskAll[Rind, Z1]:
        Z1 -= 1
    while Z2 < maskAll.shape[1] and maskAll[Rind, Z2]:
        Z2 += 1

    # collect neighboring coordinates
    if Z1 >= 0:
        x.extend([gkylR[Rind, Z1], gkylR[Rind, Z1]*(1 + scale*(np.random.rand()-0.5))])
        y.extend([gkylZ[Rind, Z1], gkylZ[Rind, Z1]*(1 + scale*(np.random.rand()-0.5))])
    if Z2 < maskAll.shape[1]:
        x.extend([gkylR[Rind, Z2], gkylR[Rind, Z2]*(1 + scale*(np.random.rand()-0.5))])
        y.extend([gkylZ[Rind, Z2], gkylZ[Rind, Z2]*(1 + scale*(np.random.rand()-0.5))])
    if R1 >= 0 and len(x) < 4:
        x.append(gkylR[R1, Zind])
        y.append(gkylZ[R1, Zind])
    if R2 < maskAll.shape[0] and len(x) < 5:
        x.append(gkylR[R2, Zind])
        y.append(gkylZ[R2, Zind])

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # copy the dict of arrays
    fields = {k: v.copy() for k, v in field.items()}
    for key in fields.keys():
        fvals = []
        if Z1 >= 0:
            fvals.extend([field[key][Rind, Z1], field[key][Rind, Z1]])
        if Z2 < maskAll.shape[1]:
            fvals.extend([field[key][Rind, Z2], field[key][Rind, Z2]])
        if R1 >= 0 and len(fvals) < 4:
            fvals.append(field[key][R1, Zind])
        if R2 < maskAll.shape[0] and len(fvals) < 5:
            fvals.append(field[key][R2, Zind])
        fvals = np.asarray(fvals, dtype=float)
        if len(fvals) >= 3:
            F = scattered_interp_linear_with_nearest_fallback(x, y, fvals)
            fields[key][Rind, Zind] = F(np.array([gkylR[Rind, Zind]]), np.array([gkylZ[Rind, Zind]]))[0]
        else:
            # if we cannot interpolate, just set to 0 (or keep as-is)
            fields[key][Rind, Zind] = fields[key][Rind, Zind]
    return fields

# --- helper: unique-mean on (x,y) pairs (groupsummary) ---
def unique_mean_by_xy(v, x, y):
    """
    For arrays v,x,y (1D, same length). Groups by (x,y) and returns:
      v_mean, (x_unique, y_unique)
    """
    pts = np.column_stack([x, y])
    uniq, idx, inv = np.unique(pts, axis=0, return_index=True, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    means = sums / counts
    return means, uniq

# --- interpolate_block ---
def interpolate_block(gkylR, gkylZ, gField, keys, r, z, r_face, z_face, r_face_zc, z_face_rc):
    NP = 20
    NR = 5
    gkylR = np.asarray(gkylR, dtype=float)
    gkylZ = np.asarray(gkylZ, dtype=float)
    N1, N2 = gkylR.shape

    # mask where (ni<=0 or Ti<=0 or Te<=0)
    maskAll = ((gField['ni'] > 0) + (gField['Ti'] > 0) + (gField['Te'] > 0) - 3) < 0

    interpR = np.zeros((N1, NP*(N2-1)), dtype=float)
    interpZ = np.zeros_like(interpR)
    interpField = {k: np.zeros_like(interpR) for k in keys}

    for i in range(N1):
        for j in range(N2-1):
            if maskAll[i, j]:
                gField = replace_negative(i, j, gkylR, gkylZ, gField, maskAll)

            segR = np.linspace(gkylR[i, j], gkylR[i, j+1], NP)
            F = interp1d(gkylR[i,[j,j+1]],gkylZ[i,[j,j+1]])
            #segZ = np.interp(segR, [gkylR[i, j], gkylR[i, j+1]],
            #                        [gkylZ[i, j], gkylZ[i, j+1]])
            segZ = F(segR)
            interpR[i, j*NP:(j+1)*NP] = segR
            interpZ[i, j*NP:(j+1)*NP] = segZ
            
            mag = np.sqrt((gkylR[i, j+1]-gkylR[i, j])**2 + (gkylZ[i, j+1]-gkylZ[i, j])**2)
            mag = max(mag, 1e-300)
            w1 = np.sqrt((segR - gkylR[i, j])**2 + (segZ - gkylZ[i, j])**2) / mag
            w2 = np.sqrt((segR - gkylR[i, j+1])**2 + (segZ - gkylZ[i, j+1])**2) / mag

            for k in keys:
                a = gField[k][i, j]
                b = gField[k][i, j+1]
                if np.isnan(a):
                    interpField[k][i, j*NP:(j+1)*NP] = b
                elif np.isnan(b):
                    interpField[k][i, j*NP:(j+1)*NP] = a
                else:
                    interpField[k][i, j*NP:(j+1)*NP] = w2*a + w1*b

    # interpolate across radial (between rows)
    interpR2 = np.zeros(((N1-1)*NR, interpR.shape[1]), dtype=float)
    interpZ2 = np.zeros_like(interpR2)
    interpField2 = {k: np.zeros_like(interpR2) for k in keys}

    for i in range(N1-1):
        for j in range(interpR.shape[1]):
            colZ = np.linspace(interpZ[i, j], interpZ[i+1, j], NR)
            F = interp1d(interpZ[[i,i+1], j], interpR[[i,i+1], j])
            colR = F(colZ)
            #colR = np.interp(colZ, [interpZ[i, j], interpZ[i+1, j]],
            #                       [interpR[i, j], interpR[i+1, j]])
            interpZ2[i*NR:(i+1)*NR, j] = colZ
            interpR2[i*NR:(i+1)*NR, j] = colR

            mag = np.sqrt((interpR[i+1, j]-interpR[i, j])**2 + (interpZ[i+1, j]-interpZ[i, j])**2)
            mag = max(mag, 1e-300)
            w1 = np.sqrt((colR - interpR[i, j])**2 + (colZ - interpZ[i, j])**2) / mag
            w2 = np.sqrt((colR - interpR[i+1, j])**2 + (colZ - interpZ[i+1, j])**2) / mag

            for k in keys:
                a = interpField[k][i, j]
                b = interpField[k][i+1, j]
                if np.isnan(a):
                    #interpField2[k][i*NR:(i+1)*NR, j] = b
                    interpField2[k][i*NR:(i+1)*NR, j] = float('nan')
                    interpField2[k][(i+1)*NR-1, j] = b
                elif np.isnan(b):
                    interpField2[k][i*NR:(i+1)*NR, j] = float('nan')
                    interpField2[k][i*NR, j] = a
                else:
                    interpField2[k][i*NR:(i+1)*NR, j] = w2*a + w1*b
            #if(i==35):
            #    pdb.set_trace()
    # final scattered interpolation to requested positions (r,z) or faces
    out = {}
    for k in keys:
        arr = interpField2[k]
        mask = np.isfinite(arr)
        vx = interpR2[mask]
        vy = interpZ2[mask]
        vv = arr[mask]
#        pdb.set_trace()
        # deduplicate (vx,vy) by averaging vv
        v_mean, xy = unique_mean_by_xy(vv, vx, vy)
        XU = xy[:, 0]
        YU = xy[:, 1]
        F = scattered_interp_linear_with_nearest_fallback(XU, YU, v_mean)

        if k == 'Gamma_R':
            # F(r_face, r_face_zc)
            rf = np.asarray(r_face, dtype=float)
            rzc = np.asarray(r_face_zc, dtype=float)
            out[k] = F(rf, rzc)
        elif k == 'Gamma_Z':
            # F(z_face_rc, z_face)
            zrc = np.asarray(z_face_rc, dtype=float)
            zf  = np.asarray(z_face, dtype=float)
            out[k] = F(zrc, zf)
        else:
            out[k] = F(r, z)
    print(len(out["Gamma_R"]))
 #   pdb.set_trace()
    return out

# --- region-specific wrappers ---
def interpolate_OSOL(gmtry, gDat):
    innerDiv = gmtry["innerDiv"]
    gkylR = np.vstack([gDat.blocked_data["R"]["block1"], gDat.blocked_data["R"]["block2"], gDat.blocked_data["R"]["block3"]])
    gkylZ = np.vstack([gDat.blocked_data["Z"]["block1"], gDat.blocked_data["Z"]["block2"], gDat.blocked_data["Z"]["block3"]])
    keys = list(gDat.blocked_data.keys())
    gField = {k: np.vstack([gDat.blocked_data[k]["block1"], gDat.blocked_data[k]["block2"], gDat.blocked_data[k]["block3"]]) for k in keys}

    r = gmtry['crx'][innerDiv+1:, gmtry['topcut'][0]+1:, :].mean(axis=2)
    z = gmtry['cry'][innerDiv+1:, gmtry['topcut'][0]+1:, :].mean(axis=2)

    r_face     = gmtry['crx'][innerDiv+2:-1, -1, [2,3]].mean(axis=1)
    r_face_zc  = gmtry['cry'][innerDiv+2:-1, -1, [2,3]].mean(axis=1)
    z_face     = np.hstack([
        gmtry['cry'][innerDiv+2, gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
        gmtry['cry'][-1,gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0)])
    z_face_rc  = np.hstack([
        gmtry['crx'][innerDiv+2, gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
        gmtry['crx'][-1,         gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
    ])
    return interpolate_block(gkylR, gkylZ, gField, keys, r, z, r_face, z_face, r_face_zc, z_face_rc)

def interpolate_ISOL(gmtry, gDat):
    innerDiv = gmtry["innerDiv"]
    gkylR = np.vstack([gDat.blocked_data["R"]["block6"], gDat.blocked_data["R"]["block7"], gDat.blocked_data["R"]["block8"]])
    gkylZ = np.vstack([gDat.blocked_data["Z"]["block6"], gDat.blocked_data["Z"]["block7"], gDat.blocked_data["Z"]["block8"]])
    keys = list(gDat.blocked_data.keys())
    gField = {k: np.vstack([gDat.blocked_data[k]["block6"], gDat.blocked_data[k]["block7"], gDat.blocked_data[k]["block8"]]) for k in keys}

    r = gmtry['crx'][:innerDiv+1, gmtry['topcut'][0]+1:, :].mean(axis=2)
    z = gmtry['cry'][:innerDiv+1, gmtry['topcut'][0]+1:, :].mean(axis=2)

    r_face     = gmtry['crx'][1:innerDiv, -1, [2,3]].mean(axis=1)
    r_face_zc  = gmtry['cry'][1:innerDiv, -1, [2,3]].mean(axis=1)
    z_face     = np.hstack([
        gmtry['cry'][1,        gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
        gmtry['cry'][innerDiv, gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
    ])
    z_face_rc  = np.hstack([
        gmtry['crx'][1,        gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
        gmtry['crx'][innerDiv, gmtry['topcut'][0]+1:-1, [0,2]].mean(axis=0),
    ])
    return interpolate_block(gkylR, gkylZ, gField, keys, r, z, r_face, z_face, r_face_zc, z_face_rc)

def interpolate_ICORE(gmtry, gDat):
    gkylR = gdat.blocked_data["R"]["block11"]
    gkylZ = gdat.blocked_data["Z"]["block11"]
    keys = list(gDat.blocked_data.keys())
    gField = {k: gDat.blocked_data[k]["block11"] for k in keys}

    r = gmtry['crx'][gmtry['leftcut'][0]+1:gmtry['leftcut'][1]+1, :gmtry['topcut'][0]+1, :].mean(axis=2)
    z = gmtry['cry'][gmtry['leftcut'][0]+1:gmtry['leftcut'][1]+1, :gmtry['topcut'][0]+1, :].mean(axis=2)
    out = interpolate_block(gkylR, gkylZ, gField, keys, r, z, [], [], [], [])
    out['rflux'] = np.array([])
    out['pflux'] = np.array([])
    return out

def interpolate_OCORE(gmtry, gDat):
    gkylR = gDat.blocked_data["R"]["block10"]
    gkylZ = gDat.blocked_data["Z"]["block10"]
    keys = list(gDat.blocked_data.keys())
    gField = {k: gDat.blocked_data[k]["block10"] for k in keys}

    r = gmtry['crx'][gmtry['rightcut'][1]+1:gmtry['rightcut'][0]+1, :gmtry['topcut'][0]+1, :].mean(axis=2)
    z = gmtry['cry'][gmtry['rightcut'][1]+1:gmtry['rightcut'][0]+1, :gmtry['topcut'][0]+1, :].mean(axis=2)

    out = interpolate_block(gkylR, gkylZ, gField, keys, r, z, [], [], [], [])
    out['rflux'] = np.array([])
    out['pflux'] = np.array([])
    return out

def interpolate_UPFR(gmtry, gDat):
    innerDiv = gmtry["innerDiv"]
    gkylR = np.vstack([gDat.blocked_data["R"]["block5"], gDat.blocked_data["R"]["block4"]])
    gkylZ = np.vstack([gDat.blocked_data["Z"]["block5"], gDat.blocked_data["Z"]["block4"]])
    keys = list(gDat.blocked_data.keys())
    gField = {k: np.vstack([gDat.blocked_data[k]["block5"], gDat.blocked_data[k]["block4"]]) for k in keys}

    r = np.vstack([
        gmtry['crx'][innerDiv:gmtry['leftcut'][1]:-1, :gmtry['topcut'][0]+1, :].mean(axis=2),
        gmtry['crx'][gmtry['rightcut'][1]:innerDiv:-1,     :gmtry['topcut'][0]+1, :].mean(axis=2),
    ])
    z = np.vstack([
        gmtry['cry'][innerDiv:gmtry['leftcut'][1]:-1, :gmtry['topcut'][0]+1, :].mean(axis=2),
        gmtry['cry'][gmtry['rightcut'][1]:innerDiv:-1,     :gmtry['topcut'][0]+1, :].mean(axis=2),
    ])

    r_face    = np.hstack([
        gmtry['crx'][innerDiv-1:gmtry['leftcut'][1]-1:-1, 1, [2,3]].mean(axis=1),
        gmtry['crx'][gmtry['rightcut'][1]+1:innerDiv+1:-1, 1, [2,3]].mean(axis=1),
    ])
    r_face_zc = np.hstack([
        gmtry['cry'][innerDiv-1:gmtry['leftcut'][1]-1:-1, 1, [2,3]].mean(axis=1),
        gmtry['cry'][gmtry['rightcut'][1]+1:innerDiv+1:-1, 1, [2,3]].mean(axis=1),
    ])
    z_face    = gmtry['cry'][np.ix_([innerDiv, innerDiv+2], np.arange(1,gmtry['topcut'][0]+1), [0,2])].mean(axis=2)
    z_face_rc = gmtry['crx'][np.ix_([innerDiv, innerDiv+2], np.arange(1,gmtry['topcut'][0]+1), [0,2])].mean(axis=2)

    return interpolate_block(gkylR, gkylZ, gField, keys, r, z, r_face, z_face, r_face_zc, z_face_rc)

def interpolate_LPFR(gmtry, gDat):
    gkylR = np.vstack([gDat.blocked_data["R"]["block0"], gDat.blocked_data["R"]["block9"]])
    gkylZ = np.vstack([gDat.blocked_data["Z"]["block0"], gDat.blocked_data["Z"]["block9"]])
    keys = list(gDat.blocked_data.keys())
    gField = {k: np.vstack([gDat.blocked_data[k]["block0"], gDat.blocked_data[k]["block9"]]) for k in keys}

    r = np.vstack([
        gmtry['crx'][:gmtry['leftcut'][0]+1, :gmtry['topcut'][0]+1, :].mean(axis=2),
        gmtry['crx'][gmtry['rightcut'][0]+1:, :gmtry['topcut'][0]+1, :].mean(axis=2),
    ])
    z = np.vstack([
        gmtry['cry'][:gmtry['leftcut'][0]+1, :gmtry['topcut'][0]+1, :].mean(axis=2),
        gmtry['cry'][gmtry['rightcut'][0]+1:, :gmtry['topcut'][0]+1, :].mean(axis=2),
    ])

    r_face    = np.hstack([
        gmtry['crx'][1:gmtry['leftcut'][0]+2, 1, [2,3]].mean(axis=1),
        gmtry['crx'][gmtry['rightcut'][0]:-1, 1, [2,3]].mean(axis=1),
    ])
    r_face_zc = np.hstack([
        gmtry['cry'][1:gmtry['leftcut'][0]+2, 1, [2,3]].mean(axis=1),
        gmtry['cry'][gmtry['rightcut'][0]:-1, 1, [2,3]].mean(axis=1),
    ])
    z_face    = gmtry['cry'][np.ix_([1, gmtry['crx'].shape[0]-1], np.arange(1,gmtry['topcut'][0]+1), [0,2])].mean(axis=2)
    z_face_rc = gmtry['crx'][np.ix_([1, gmtry['crx'].shape[0]-1], np.arange(1,gmtry['topcut'][0]+1), [0,2])].mean(axis=2)

    return interpolate_block(gkylR, gkylZ, gField, keys, r, z, r_face, z_face, r_face_zc, z_face_rc)

def interpolate_all(bdat, gdat):
    gmtry = bdat.gmtry
    bdat.interp_data = {}
    print("Interpolate outer SOL")
    bdat.interp_data["osol"]  = interpolate_OSOL(gmtry, gdat)
    print("Interpolate inner SOL")
    bdat.interp_data["isol"]  = interpolate_ISOL(gmtry, gdat)
    print("Interpolate inner core")
    bdat.interp_data["icore"] = interpolate_ICORE(gmtry, gdat)
    print("Interpolate outer core")
    bdat.interp_data["ocore"] = interpolate_OCORE(gmtry, gdat)
    print("Interpolate upper PFR")
    bdat.interp_data["upfr"]  = interpolate_UPFR(gmtry, gdat)
    print("Interpolate lower PFR")
    bdat.interp_data["lpfr"]  = interpolate_LPFR(gmtry, gdat)

    keys = list(gdat.blocked_data.keys())
    Nx, Ny = gmtry['hx'].shape
    field = {}
    innerDiv = gmtry["innerDiv"]

    for k in keys:
        field[k] = np.zeros((Nx, Ny), dtype=float)

        if k.lower() == 'Gamma_R'.lower():
            field[k][innerDiv+2:-1, -1] = bdat.interp_data["osol"][k]
            field[k][1:innerDiv, -1]  = bdat.interp_data["isol"][k]
            rng1 = np.arange(innerDiv-1, gmtry['leftcut'][1]-1, -1)
            field[k][rng1, 1] = bdat.interp_data["upfr"][k][:len(rng1)]
            rng2 = np.arange(gmtry['rightcut'][1]+1, innerDiv+1, -1)
            field[k][rng2, 1] = bdat.interp_data["upfr"][k][len(rng1):]
            field[k][1:gmtry['leftcut'][0]+2, 1] = bdat.interp_data["lpfr"][k][:len(np.arange(1, gmtry['leftcut'][0]+2))]
            field[k][gmtry['rightcut'][0]:-1, 1] = bdat.interp_data["lpfr"][k][len(np.arange(1, gmtry['leftcut'][0]+2)):]
        elif k.lower() == 'Gamma_Z'.lower():
            rng_end = np.arange(gmtry['topcut'][0]+1, Ny-1)
            field[k][innerDiv+2, gmtry['topcut'][0]+1:-1] = bdat.interp_data["osol"][k][:len(rng_end)]
            field[k][-1,       gmtry['topcut'][0]+1:-1]   = bdat.interp_data["osol"][k][len(rng_end):]
            field[k][1,        gmtry['topcut'][0]+1:-1]   = bdat.interp_data["isol"][k][:len(rng_end)]
            field[k][innerDiv, gmtry['topcut'][0]+1:-1] = bdat.interp_data["isol"][k][len(rng_end):]
            field[k][[innerDiv, innerDiv+2], 1:gmtry['topcut'][0]+1] = bdat.interp_data["upfr"][k]
            field[k][[1, gmtry['crx'].shape[0]-1], 1:gmtry['topcut'][0]+1] = bdat.interp_data["lpfr"][k]
        else:
            field[k][:innerDiv+1, gmtry['topcut'][0]+1:]    = bdat.interp_data["isol"][k]
            field[k][innerDiv+1:, gmtry['topcut'][0]+1:]    = bdat.interp_data["osol"][k]
            field[k][gmtry['leftcut'][0]+1:gmtry['leftcut'][1]+1, :gmtry['topcut'][0]+1] = bdat.interp_data["icore"][k]
            field[k][gmtry['rightcut'][1]+1:gmtry['rightcut'][0]+1, :gmtry['topcut'][0]+1] = bdat.interp_data["ocore"][k]
            blk1 = bdat.interp_data["upfr"][k][len(np.arange(innerDiv, gmtry['leftcut'][1]+1, -1))::-1, :]
            blk2 = bdat.interp_data["upfr"][k][blk1.shape[0]:, :]
            field[k][gmtry['leftcut'][1]+1:innerDiv+1, :gmtry['topcut'][0]+1] = blk1
            field[k][innerDiv+1:gmtry['rightcut'][1]+1, :gmtry['topcut'][0]+1] = blk2[::-1]

            field[k][:gmtry['leftcut'][0]+1, :gmtry['topcut'][0]+1] = bdat.interp_data["lpfr"][k][:gmtry['leftcut'][0]+1, :]
            field[k][gmtry['rightcut'][0]+1:, :gmtry['topcut'][0]+1] = bdat.interp_data["lpfr"][k][gmtry['leftcut'][0]+1:, :]

    return field

def parallel_smooth(gmtry, field):
    innerDiv = gmtry["innerDiv"]
    SOL = np.vstack([
        smooth2d(field[:innerDiv, gmtry['topcut'][0]+1:]),
        smooth2d(field[innerDiv:, gmtry['topcut'][0]+1:])
    ])
    temp = np.concatenate([field[:, :gmtry['topcut'][0]+1], SOL], axis=1)

    BPFR = smooth2d(np.vstack([
        field[:gmtry['leftcut'][0]+1, :gmtry['topcut'][0]+1],
        field[gmtry['rightcut'][0]+1:, :gmtry['topcut'][0]+1]
    ]))
    TPFR = smooth2d(np.vstack([
        field[innerDiv-1:gmtry['leftcut'][1]+1:-1, :gmtry['topcut'][0]+1],
        field[gmtry['rightcut'][1]::-1,            :gmtry['topcut'][0]+1]
    ]))
    RCore = smooth2d(field[gmtry['rightcut'][1]+1:gmtry['rightcut'][0]+1, :gmtry['topcut'][0]+1])
    LCore = smooth2d(field[gmtry['leftcut'][0]+1:gmtry['leftcut'][1]+1,   :gmtry['topcut'][0]+1])

    temp[:gmtry['leftcut'][0]+1, :gmtry['topcut'][0]+1] = BPFR[:gmtry['leftcut'][0]+1, :]
    temp[gmtry['rightcut'][0]+1:, :gmtry['topcut'][0]+1] = BPFR[gmtry['leftcut'][0]+1:, :]

    n1 = len(np.arange(innerDiv, gmtry['leftcut'][1]+1, -1))
    temp[gmtry['leftcut'][1]+1:innerDiv, :gmtry['topcut'][0]+1] = TPFR[n1-1::-1, :]
    temp[innerDiv:gmtry['rightcut'][1]+1, :gmtry['topcut'][0]+1] = TPFR[:-(n1 or None)-1:-1, :]

    temp[gmtry['rightcut'][1]+1:gmtry['rightcut'][0]+1, :gmtry['topcut'][0]+1] = RCore
    temp[gmtry['leftcut'][0]+1:gmtry['leftcut'][1]+1,   :gmtry['topcut'][0]+1] = LCore
    return temp

# =========================
# Main translation body
# =========================

# Load inputs (you should provide these variables)
# ehl2data: numpy array Nx x M
b2dat = b2.B2()
b2dat_new = b2.B2()
b2dat.read_b2fstate() # Read default file "./b2fstate"
b2dat.read_b2fgmtry() # Read default file "./b2fgmtry"
gdat = gkyl.gkeyll_data()
gdat.read_data("./ehl2data.txt")
gdat.read_block_ind("./cells_ehl2data.txt")

# Definitions
b2dat.gmtry["R"] = b2dat.gmtry['crx'].mean(axis=2)
b2dat.gmtry["Z"] = b2dat.gmtry['cry'].mean(axis=2)
b2dat.gmtry["innerDiv"] = np.argmax(np.diff(b2dat.gmtry["R"][:, 0], n=1))


electron_charge = pyconst.elementary_charge

gdat.regrid_data()
gdat.replace_zero() # replace zeros of flux with NaNs

fieldS = interpolate_all(b2dat, gdat)

maskr = fieldS['Gamma_R'] != 0
maskp = fieldS['Gamma_Z'] != 0

fnax = b2dat.state['fna'][:, :, 0, 1].copy()
fnay = b2dat.state['fna'][:, :, 1, 1].copy()
fnax[maskp] = -fieldS['Gamma_Z'][maskp]
fnay[maskr] =  fieldS['Gamma_R'][maskr]

# Regularize
#fieldS['upari'] = parallel_smooth(b2dat.gmtry, -fieldS['upari'])  # sign flip to match Gkeyll dir

# Write data
state_new = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b2dat.state.items()}
state_new['te'] = np.maximum(fieldS['Te'], 1.1e-6) * electron_charge
state_new['ti'] = np.maximum(fieldS['Ti'], 1.1e-6) * electron_charge
state_new['ua'] = state_new['ua'].copy()
state_new['ua'][:, :, 1] = fieldS['upari']
state_new['na'] = state_new['na'].copy()
state_new['na'][:, :, 1] = np.maximum(fieldS['ni'], 5)
state_new['ne'] = state_new['na'][:, :, 1]
state_new['po'] = fieldS['phi']
state_new['fna'] = state_new['fna'].copy()
state_new['fna'][:, :, 0, 1] = fnax
state_new['fna'][:, :, 1, 1] = fnay

b2dat_new.state = state_new
b2dat_new.write_b2fstate("./b2fstati_test", "State with Gkeyll plasma")
