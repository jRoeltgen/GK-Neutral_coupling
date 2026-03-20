from scipy.interpolate import griddata
from collections import defaultdict
import numpy as np

def interpolate_all_sources(tria, source_dict, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for mom, mom_block in source_dict.items():
        for species, species_block in mom_block.items():
            for source, value in species_block.items():
                out[mom][species][source] = interpolate_source(tria, value,
                            grid_r, grid_z, method=method, fill_mode=fill_mode,
                            fill_value=fill_value)
    return out

# TODO add better out of bounds value/check
def interpolate_source(tria, source, grid_r, grid_z, method, fill_mode,
                       fill_value):
    interp_source = griddata(tria.incenter, source, (grid_r, grid_z),
                                        method=method)
    # Create shapely.MultiPolygon
    # Check if points of grid_r, grid_z are contained within polygon
    # 0 out points outside of domain or something else special
    # This is important for something like the hl2a baffling
    if (fill_mode == "constant"):
        interp_source[np.isnan(interp_source)] = fill_value
    elif (fill_mode == "nearest"):
        pass

    return interp_source

# There are 3 interpolation routines in torx
# TODO add better out of bounds value/check
def interp_moments(gmtry, grid_r, grid_z, field, ind):
    r = np.mean(gmtry["crx"][:,:,ind],2)
    z = np.mean(gmtry["cry"][:,:,ind],2)
    # Maybe RBF interpolator?
    interp_data = griddata((grid_r, grid_z), field, (r, z), method = 'linear')
    interp_data[np.isnan(interp_data)] = 0
    return interp_data