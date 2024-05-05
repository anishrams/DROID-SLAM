#!/bin/bash

KITTI360_PATH=/ocean/projects/cis220039p/shared/datasets/KITTI_360_data/KITTI-360/data_2d_raw/

python evaluation_scripts/validate_kitti360.py --datapath=$KITTI360_PATH --weights=droid.pth --disable_vis  $@

