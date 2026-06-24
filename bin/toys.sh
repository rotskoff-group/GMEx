#!/bin/bash
# bin/toys.sh

# Simulate toy Markov jump processes on four-state cycles.
# The hard toy lacks separation of timescales:
# all its counterclockwise and clockwise rates are respectively 1.0 and 0.5.
# The easy toy has separation of timescales:
# only its unobserved counterclockwise and clockwise rate are respectively 1.0 and 0.5.
# Its observed counterclockwise and clockwise rates are respectively 0.25 and 0.125.
# States 2 and 3 are coarse-grained together in both toys, leaving states 0, 1, and 2U3.
# TCL-GME-DT propgators and NZ-GME-DT memory kernels are analytically accessible.
# This script takes about 8 hrs to run.

python3 scripts/preprocess/toys.py
python3 scripts/train/toys.py
python3 scripts/postprocess/toys.py
