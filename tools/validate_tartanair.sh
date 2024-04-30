#!/bin/bash


# TARTANAIR_PATH=datasets/TartanAir
TARTANAIR_PATH=/ocean/projects/cis220039p/shared/tartanair_v2

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis  $@

