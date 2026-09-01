# Breakdown of Gkeyll Coupling module

Within the neutral_coupling/gkeyll_coupling folder are three subfolders:

    1. geometry: Contains files used to generate mappings to interpolate data between Gkeyll and EIRENE

    2. io: Contains files used to pass plasma data from Gkeyll to EIRENE and neutral/plasma source data from EIRENE to Gkeyll

    3. examples: Contains some example uses of the files in geometry and io 

# Relevant Files for users:

    1. Geometry generation: examples/gkeyll_geom_example.py is an example script that looks similar to the geometry part of a Gkeyll (https://github.com/ammarhakim/gkeyll) input file, which defines the Gkeyll grid. This file uses the geometry module to generate an interpolation mapping between Gkeyll and Eirene grids.

    2. Plotting: examples/interpolation_example.py is an example of how to use the io module and the common module to make plots of some Gkeyll and Eirene data. To use this, you must first generate an interpolation mapping.

    3. Coupling the codes: 

        - examples/coupling_script.sh is an example slurm submission script that runs both codes and data processing steps in between.

        - examples/eirene_submission_script is called by coupling_script.sh and runs eirene

        - io/prep_gkeyll_data.py is called by coupling_script.sh. It processes the Gkeyll output and prepares the eirene input data

        - io/prep_eirene_data.py is called by coupling_script.sh. It processes the Eirene output and prepares the Gkeyll input data
