#!/bin/bash


#Activate python environment first
#Test user interface script
RUNDIR=/home/oelsmann/Julius/Scripts/discotimes/discotimes
OUTDIR=/home/oelsmann/Julius/Scripts/discotimes/discotimes/tests/test_output/

cd $RUNDIR

echo 'testing user interface commands'
python discofit.py --help
# run and save single file | as nc | plot results
python discofit.py examples/GEO1.txt -o $OUTDIR -s tests/custom_settings.py -r M -p
# run and save multiple files | as nc
python discofit.py examples/*.txt -o $OUTDIR -s tests/custom_settings.py -r M -c
# run and save multiple files | as dt
python discofit.py examples/*.txt -t dt -o $OUTDIR -s tests/custom_settings.py -r M

# check if running stops when existing file is detected
python discofit.py examples/GEO1.txt -o $OUTDIR -s tests/custom_settings.py -r M

