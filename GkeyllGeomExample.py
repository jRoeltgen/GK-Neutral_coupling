import gkeyllGeometry.gkeyllGeom as gkeyllGeom
import gkeyllGeometry.gkeyllEFIT as gkeyllEFIT
import numpy as np

def shaped_pfunc_lower_outer(s):
    p0 = np.array([5.488 - 0.6, -8.600])
    p1 = np.array([5.855 - 0.6, -8.52318])

    # Extrapolate p1 outward
    p1 = (p1 - p0) * 2 + p1

    R = (1 - s) * p0[0] + s * p1[0]
    Z = (1 - s) * p0[1] + s * p1[1]

    return np.array([R, Z])


def shaped_pfunc_upper_outer(s):
    p0 = np.array([5.488 - 0.6, 8.600])
    p1 = np.array([5.855 - 0.6, 8.52318])

    # Extrapolate p1 outward
    p1 = (p1 - p0) * 2 + p1

    R = (1 - s) * p0[0] + s * p1[0]
    Z = (1 - s) * p0[1] + s * p1[1]

    return np.array([R, Z])

def shaped_pfunc_upper_inner(s):
    R = 1.651 + (1.8 - 1.651)*s
    Z = 6.331 + (6.777 - 6.331)*s
    return np.array([R, Z])

def shaped_pfunc_lower_inner(s):
    R = 1.651 + (1.8 - 1.651)*s
    Z = -(6.331 + (6.777 - 6.331)*s)
    return np.array([R, Z])

#Setup all inputs
zinner = 6.34
zouter = 8.29
rright_out = 5.2

inp0 = {}
inp0["ftype"] = "GKYL_PF_LO_R"
inp0["rright"] = rright_out
inp0["rleft"] = 0.0
inp0["rmin"] = 1.7
inp0["rmax"] = 6.2
inp0["zmin_right"] = -zouter
inp0["zmin_left"] = -zinner
inp0["plate_spec"] = True
inp0["plate_func_lower"] = shaped_pfunc_lower_outer
inp0["plate_func_upper"] = shaped_pfunc_lower_inner

inp1 = {}
inp1["ftype"] = "GKYL_DN_SOL_OUT_LO"
inp1["rclose"] = 6.2
inp1["rright"] = rright_out
inp1["rleft"] = 0.0
inp1["rmin"] = 0.7
inp1["rmax"] = 6.2
inp1["zmin"] = -zouter
inp1["zmax"] = zouter
inp1["plate_spec"] = True
inp1["plate_func_lower"] = shaped_pfunc_lower_outer
inp1["plate_func_upper"] = shaped_pfunc_upper_outer

inp2 = {}
inp2["ftype"] = "GKYL_DN_SOL_OUT_MID"
inp2["rclose"] = 6.2
inp2["rleft"] = 0.0
inp2["rright"] = rright_out
inp2["rmin"] = 0.7
inp2["rmax"] = 6.2
inp2["plate_spec"] = True
inp2["plate_func_lower"] = shaped_pfunc_lower_outer
inp2["plate_func_upper"] = shaped_pfunc_upper_outer

inp3 = {}
inp3["ftype"] = "GKYL_DN_SOL_OUT_UP"
inp3["rclose"] = 6.2
inp3["rright"] = rright_out
inp3["rleft"] = 0.0
inp3["rmin"] = 0.7
inp3["rmax"] = 6.2
inp3["zmin"] = -zouter
inp3["zmax"] = zouter
inp3["plate_spec"] = True
inp3["plate_func_lower"] = shaped_pfunc_lower_outer
inp3["plate_func_upper"] = shaped_pfunc_upper_outer

inp4 = {}
inp4["ftype"] = "GKYL_PF_UP_R"
inp4["rright"] = rright_out
inp4["rleft"] = 0.0
inp4["rmin"] = 1.7
inp4["rmax"] = 6.2
inp4["zmax_right"] = zouter
inp4["zmax_left"] = zinner
inp4["plate_spec"] = True
inp4["plate_func_lower"] = shaped_pfunc_upper_inner
inp4["plate_func_upper"] = shaped_pfunc_upper_outer

inp5 = {}
inp5["ftype"] = "GKYL_PF_UP_L"
inp5["rright"] = rright_out
inp5["rleft"] = 0.0
inp5["rmin"] = 1.7
inp5["rmax"] = 6.2
inp5["zmax_right"] = zouter
inp5["zmax_left"] = zinner
inp5["plate_spec"] = True
inp5["plate_func_lower"] = shaped_pfunc_upper_inner
inp5["plate_func_upper"] = shaped_pfunc_upper_outer

inp6 = {}
inp6["ftype"] = "GKYL_DN_SOL_IN_UP"
inp6["rleft"] = 2.0
inp6["rright"] = rright_out
inp6["rmin"] = 0.0
inp6["rmax"] = 6.2
inp6["zmin"] = -zinner
inp6["zmax"] = zinner
inp6["plate_spec"] = True
inp6["plate_func_upper"] = shaped_pfunc_upper_inner
inp6["plate_func_lower"] = shaped_pfunc_lower_inner

inp7 = {}
inp7["ftype"] = "GKYL_DN_SOL_IN_MID"
inp7["rleft"] = 2.0
inp7["rright"] = rright_out
inp7["rmin"] = 0.0
inp7["rmax"] = 6.2
inp7["zmin"] = -zinner
inp7["zmax"] = zinner
inp7["plate_spec"] = True
inp7["plate_func_upper"] = shaped_pfunc_upper_inner
inp7["plate_func_lower"] = shaped_pfunc_lower_inner

inp8 = {}
inp8["ftype"] = "GKYL_DN_SOL_IN_LO"
inp8["rleft"] = 2.0
inp8["rright"] = rright_out
inp8["rmin"] = 0.0
inp8["rmax"] = 6.2
inp8["zmin"] = -zinner
inp8["zmax"] = zinner
inp8["plate_spec"] = True
inp8["plate_func_upper"] = shaped_pfunc_upper_inner
inp8["plate_func_lower"] = shaped_pfunc_lower_inner

inp9 = {}
inp9["ftype"] = "GKYL_PF_LO_L"
inp9["rright"] = rright_out
inp9["rleft"] = 0.0
inp9["rmin"] = 1.7
inp9["rmax"] = 6.2
inp9["zmin_right"] = -zouter
inp9["zmin_left"] = -zinner
inp9["plate_spec"] = True
inp9["plate_func_lower"] = shaped_pfunc_lower_outer
inp9["plate_func_upper"] = shaped_pfunc_lower_inner

inp10 = {}
inp10["ftype"] = "GKYL_CORE_R"
inp10["rclose"] = 6.2
inp10["rright"] = rright_out
inp10["rleft"] = 2.0
inp10["rmin"] = 1.58
inp10["rmax"] = 6.2

inp11 = {}
inp11["ftype"] = "GKYL_CORE_L"
inp11["rclose"] = 0.0
inp11["rright"] = rright_out
inp11["rleft"] = 2.0
inp11["rmin"] = 1.58
inp11["rmax"] = 6.2

inp = [inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9, inp10, inp11]


#If not loading from file do this:

# gefit = gkeyllEFIT.gkeyllEFIT('./test_data/gkeyll_data/', 'step')
# gefit.load_eqdsk()
# gefit.find_xpts()
# # Can replace find_xpts with the line below to save time
# #gefit.Rxpt = np.array([2.489562969077551, 2.489562969077551])
# #gefit.Zxpt = np.array([-6.18070007806902,  6.18070007806902])
# #gefit.psisep=1.5092
# gg = gkeyllGeom.gkeyllGeom(gefit, inp)

#If loading from file we can do this:
gg = gkeyllGeom.gkeyllGeom('./gkeyllGeometry/stored_data/gkeyllGeometry.pkl')



# Tests 
# Call  it on a point in the inner SOL
#gg.psitheta(1.59,0.0)


#Here is how we would loop over the whole B2 grid to get all the data
# but right now it is way too slow. Need to figure out a way to speed it up
#import B2IO as b2
#b2dat = b2.B2("./test_data/b2_data/")
#R = b2dat.gmtry["crx"].mean(axis=2)
#Z = b2dat.gmtry["cry"].mean(axis=2)
#
#ptb = np.zeros((R.shape[0], R.shape[1], 3))
#
#for i in range(R.shape[0]):
#    for j in range(R.shape[1]):
#        print(i,j)
#        ptb[i,j] = gg.psitheta(R[i,j], Z[i,j])




