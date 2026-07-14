# Breakdown of Gkeyll Coupling module

Within the gkeyllCoupling folder are three subfolders:

    1. gkeyllGeometry: Contains files used to generate mappings to interpolate data between Gkeyll and EIRENE

    2. gkeyllIO: Contains files used to pass plasma data from Gkeyll to EIRENE and neutral/plasma source data from EIRENE to Gkeyll

    3. gkeyllExamples: Contains some example uses of the files in gkeyllGeometry and gkeyllIO 

# Relevant Files for users:

    1. Geometry generation: gkeyllExamples/GkeyllGeomExample.py is an example script that looks similar to the geometry part of a Gkeyll (https://github.com/ammarhakim/gkeyll) input file, which defines the Gkeyll grid. This file uses the gkeyllGeometry module to generate an interpolation mapping between Gkeyll and Eirene grids.

    2. Plotting: gkeyllExamples/InterpolationExample.py is an example of how to use the gkeyllIO module and the common module to make plots of some Gkeyll and Eirene data. To use this, you must first generate an interpolation mapping.

    3. Coupling the codes: 

        - gkeyllExamples/coupling_script.sh is an example slurm submission script that runs both codes and data processing steps in between.

        - gkeyllExamples/eirene_submission_script.sh is called by coupling_script.sh and runs eirene

        - gkeyllExamples/PrepGkeyllData.py is called by coupling_script.sh. It processes the Gkeyll output and prepares the eirene input data

        - gkeyllExamples/PrepEireneData.py is called by coupling_script.sh. It processes the Eirene output and prepares the Gkeyll input data

        - gkeyllExamples/gendummyft31.py: Depending on how you have eirene installed you may be able to generate a fort.31 starting file with SOLPS. Otherwise use this file


