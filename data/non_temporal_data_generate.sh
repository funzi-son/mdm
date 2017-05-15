#!/bin/bash
# Add dependency onto projects
PROJ_DIR=$HOME/WORK/projects/multiresidential/Codes
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH
python3 $PROJ_DIR/data/non_temporal_data_generate.py
