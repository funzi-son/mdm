#!/bin/bash
# Add dependency onto projects
PROJ_DIR=/home/tra161/WORK/projects/multiresidential/Codes
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH
python3 $PROJ_DIR/examples/result_analysis.py
