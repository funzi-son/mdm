#!/bin/bash
# Add dependency onto projects
PROJ_DIR=$HOME/WORK/projects/multiresidential/Codes
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH
python3 $PROJ_DIR/examples/mixhmm_run.py
