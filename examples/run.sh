#!/bin/bash
# Add dependency onto projects
PROJ_DIR="$(dirname $PWD)"
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH

python3 $PROJ_DIR/examples/run.py $1 $2 $3
