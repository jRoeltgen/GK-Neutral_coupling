import B2IO as b2
import read_gkyl as gkyl
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
import pdb
import time

def create_polygons(bdat):
    crx = b2dat.gmtry["crx"]
    cry = b2dat.gmtry["cry"]
    nx,ny,nz = crx.shape
    polygons = []
    xind = []
    yind = []
    for x in range(nx):
        for y in range(ny):
            my_list = []
            for z in range(nz):
                my_list.append((crx[x,y,z],cry[x,y,z]))
            polygons.append(Polygon(my_list))
            xind.append(x)
            yind.append(y)
    return xind, yind, polygons, MultiPolygon(polygons)

def find_closest_point(p, gdat):
    diff_x = p.x - gdat.data["R"]
    diff_y = p.y - gdat.data["Z"]
    min_loc = np.argmin(diff_x**2+diff_y**2)
    return min_loc

def check_single_polygon(p, gdat):
    for block_key in gdat.blocked_data["polygons"].keys():                
        for k, poly in enumerate(gdat.blocked_data["polygons"][block_key]):
            if poly.contains(p):
                bnum = int(block_key[5:])
                xind = gdat.blocked_data["xind"][block_key][k]
                yind = gdat.blocked_data["yind"][block_key][k]
                return bnum, xind, yind, True, poly
    return -1, -1, -1, False, None

def corresponding_polygon(gdat, bdat):
    r = np.mean(bdat.gmtry["crx"],axis=2)
    z = np.mean(bdat.gmtry["cry"],axis=2)
    nx, ny = r.shape
    gkyl_poly_for_b2 = {"block": np.zeros((nx,ny)), "xyind": np.zeros((nx,ny,2),dtype=int),
                        "contained":np.full((nx,ny), False, dtype=bool), "polygon": []}
    gkyl_poly_for_b2["polygon"] = [[0]*ny for _ in range(nx)]
    for i in range(nx):
        print(f"i={i}")
        for j in range(ny):
            p = Point(r[i,j],z[i,j])
            bnum, xind, yind, found, poly = check_single_polygon(p,gdat)
            gkyl_poly_for_b2["block"][i,j] = bnum
            gkyl_poly_for_b2["xyind"][i,j,:] = np.array([xind, yind])
            gkyl_poly_for_b2["contained"][i,j] = found
            gkyl_poly_for_b2["polygon"][i][j] = poly
            if not found:
                ind = find_closest_point(p, gdat)
                gkyl_poly_for_b2["xyind"][i,j,:] = ind

    return gkyl_poly_for_b2

b2dat = b2.B2("./test_data/b2_data/")
gdat = gkyl.gkeyll_data()
gdat.read_data("./test_data/gkeyll_data/ehl2data.txt")
gdat.read_block_ind("./test_data/gkeyll_data/cells_ehl2data_copy.txt")
gdat.regrid_data()
gdat.create_polygons()
start_time = time.perf_counter()
gkyl_poly_for_b2 = corresponding_polygon(gdat, b2dat)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Corresponding_polygon took {elapsed_time:.4f} seconds")
