# Overview of Repo
This repo contains a python library that can be used to couple gyrokinetic codes to the monte-carlo neutral code EIRENE.

Modules used to extract/process eirene data are stored in the common folder

Tests for the modules in common along with the test data are in the tests folder 

Modules used to couple eirene to gkeyll are in the gkeyllCoupling folder.

Documentation with helpful details is in the doc folder

# Breakdown of Gkeyll Coupling module
Within the gkeyllCoupling folder are three subfolders:

    1. gkeyllGeometry: Contains files used to generate mappings to interpolate data between Gkeyll and EIRENE
    2. gkeyllIO: Contains files used to pass plasma data from Gkeyll to EIRENE and neutral/plasma source data from EIRENE to Gkeyll
    3. gkeyllExamples: Contains some example uses of the files in gkeyllGeometry and gkeyllIO 



# Dependencies
For All Coupling : numpy, scipy, filecmp, colorama, hypothesis

Additionally For Gkeyll Coupling : postgkyl
