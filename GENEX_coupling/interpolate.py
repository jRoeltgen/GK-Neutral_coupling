from scipy.interpolate import (griddata, LinearNDInterpolator)
from scipy.spatial import Delaunay
from collections import defaultdict
import numpy as np
import warnings

def interpolate_all_sources(tria, source_dict, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for mom, mom_block in source_dict.items():
        for species, species_block in mom_block.items():
            for source, value in species_block.items():
                out[mom][species][source] = interpolate_source(tria, value,
                            grid_r, grid_z, method=method, fill_mode=fill_mode,
                            fill_value=fill_value).reshape(1,len(grid_r))
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

def build_triangulation(r, z):
    points = np.column_stack([r.ravel(), z.ravel()])
    return Delaunay(points)

# There are 3 interpolation routines in torx
# TODO add better out of bounds value/check
def interp_moments(gmtry, tri, field, ind):
    VALID_INDS = {
        (0, 2),
        (2, 3),
        (0, 1, 2, 3),
    }
    if isinstance(ind, slice):
        ind_tuple = tuple(range(*ind.indices(4)))
    else:
        ind_tuple = tuple(ind)
    if ind_tuple not in VALID_INDS:
        warnings.warn(
            f"interp_moments received non-standard ind={ind_tuple}. "
            "This is geometrically valid but not used in standard physics.",
            UserWarning,
            stacklevel=2,
        )
    r = np.mean(gmtry["crx"][:,:,ind],2)
    z = np.mean(gmtry["cry"][:,:,ind],2)
    # Maybe RBF interpolator?
    interp = LinearNDInterpolator(tri, field.ravel())
    xi = np.column_stack([r.ravel(), z.ravel()])
    interp_data = interp(xi).reshape(r.shape)
    interp_data[np.isnan(interp_data)] = 0
    return interp_data