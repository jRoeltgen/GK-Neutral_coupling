from scipy.interpolate import griddata
import numpy as np

# TODO add better out of bounds value/check
def interpolate_source(tria, source, grid_r, grid_z):
    interp_source = griddata(tria.incenter, source, (grid_r, grid_z),
                                        method='linear')
    # Create shapely.MultiPolygon
    # Check if points of grid_r, grid_z are contained within polygon
    # 0 out points outside of domain or something else special
    # This is important for something like the hl2a baffling
    interp_source[np.isnan(interp_source)] = 0
    return interp_source

# There are 3 interpolation routines in torx
# TODO add better out of bounds value/check
def interp_moments(gmtry, grid_r, grid_z, field, key):
    if key == "poloidal_fluxes":
        ind = [0,2]
    elif key == "radial_fluxes":
        ind = [2,3]
    else:
        ind = [0,1,2,3]
    r = np.mean(gmtry["crx"][:,:,ind],2)
    z = np.mean(gmtry["cry"][:,:,ind],2)
    # Maybe RBF interpolator?
    interp_data = griddata((grid_r, grid_z), field, (r, z), method = 'linear')
    interp_data[np.isnan(interp_data)] = 0
    return interp_data