# This script uses the library to do the coupling step for a Gkeyll EIRENE Simulation
# This script assumes a coodinate mapping between the two grids has already been generated
# B2 data is used only for the geometry

import numpy as np
import gkeyllIO
import B2IO as b2
import eireneIO

# First set data input paths
gkeyll_data_path = './'
gkeyll_simulation_name = 'hstep23'
gkeyll_half_domain=True
gkeyll_diffusivity=0.5

eirene_data_path = "./step_full_device_D_only/gkeyll_coupling/"
b2_data_path = "./step_full_device_D_only/baserun/"

coordinate_mapping_path = './GK-Neutral_coupling/gkeyllGeometry/stored_data/'

#Set data output paths and get frame number
gkeyll_text_output_path = './gkeyll_text_output/'
eirene_output_path = './eirene_text_output/'
frame = int(np.genfromtxt(gkeyll_text_output_path+"new_data_flag"))

#Second load data
g = gkeyllIO.gkeyll(gkeyll_data_path, gkeyll_simulation_name, gkeyll_half_domain, gkeyll_diffusivity)
g.read_geometry()
g.read_data(frame)
g.read_coeffs(frame)
ptb=np.load(coordinate_mapping_path+'gkeyll_b2_coordmapping.npy')
ptb_surfr=np.load(coordinate_mapping_path+'gkeyll_b2_coordmapping_radialfaces.npy')
ptb_surfz=np.load(coordinate_mapping_path+'gkeyll_b2_coordmapping_parallelfaces.npy')

edat = eireneIO.eirene(eirene_data_path)
b2dat = b2.B2(b2_data_path)

#Third do the interpolation
g.interpolate_data(ptb)
g.interpolate_surfr_data(ptb_surfr)
g.interpolate_surfz_data(ptb_surfz,)

g.calc_derived_data(b2dat, edat)
g.calc_derived_surfr_data(b2dat, edat)
g.calc_derived_surfz_data(b2dat, edat)

#4th populate and write eirene data
g.populate_ft31(edat)
edat.write_ft31(eirene_data_path+'fort.31_new')
