# This script uses the library to do the coupling step for a Gkeyll EIRENE Simulation
# This script assumes a coodinate mapping between the two grids has already been generated
# B2 data is used only for the geometry

import numpy as np
import gkeyllIO
import yaml
import sys
import os

# Load configuration from YAML file
config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Extract paths and options
paths = config['paths']
gkeyll_options = config['gkeyll_options']

# Validate that required paths exist
required_paths = ['eirene_data_path', 'b2_data_path', 'coordinate_mapping_path', 'gkeyll_text_output_path', 'lib_path']
for path_key in required_paths:
    if not os.path.exists(paths[path_key]):
        print(f"Warning: {path_key} does not exist: {paths[path_key]}")

target_dir = paths['lib_path'] + "common/"
sys.path.insert(0, target_dir)
import B2IO as b2
import eireneIO

# Set data input paths from config
gkeyll_data_path = paths['gkeyll_data_path']
gkeyll_simulation_name = paths['gkeyll_simulation_name']
gkeyll_half_domain = gkeyll_options['half_domain']
gkeyll_diffusivity = gkeyll_options['diffusivity']
gkeyll_extra_species = gkeyll_options['extra_species']
gkeyll_plate_material = gkeyll_options['plate_material']
gkeyll_wall_material = gkeyll_options['wall_material']
gkeyll_plate_rec_coeff = gkeyll_options['final_plate_rec_coeff']
gkeyll_wall_rec_coeff = gkeyll_options['final_wall_rec_coeff']

# Ensure extra_species is a list
if not isinstance(gkeyll_extra_species, list):
    gkeyll_extra_species = [gkeyll_extra_species] if gkeyll_extra_species else []

eirene_data_path = paths['eirene_data_path']
b2_data_path = paths['b2_data_path']
coordinate_mapping_path = paths['coordinate_mapping_path']

# Set data output paths and get frame number
gkeyll_text_output_path = paths['gkeyll_text_output_path']
frame = int(np.genfromtxt(gkeyll_text_output_path+"new_data_flag"))

#Second load data
g = gkeyllIO.gkeyll(gkeyll_data_path, gkeyll_simulation_name, gkeyll_half_domain, gkeyll_diffusivity, gkeyll_extra_species, final_plate_rec_coeff = gkeyll_plate_rec_coeff, final_wall_rec_coeff=gkeyll_wall_rec_coeff, plate_material = gkeyll_plate_material, wall_material = gkeyll_wall_material)
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
g.interpolate_surfz_data(ptb_surfz)

g.calc_derived_data(b2dat, edat)
g.calc_derived_surfr_data(b2dat, edat)
g.calc_derived_surfz_data(b2dat, edat)

#4th populate and write eirene data
g.populate_ft31(b2dat, edat)
edat.write_ft31(eirene_data_path+'fort.31')
