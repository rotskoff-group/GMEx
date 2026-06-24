#!/bin/bash
# bin/BMF1.sh

# Script for TCL-GME-DT analysis of bovine mitochondrial F1 ATPase rotation traces from
# R. Kobayashi, H. Ueno, C.B. Li, and H. Noji,
# Rotary catalysis of bovine mitochondrial F1-ATPase studied by single-molecule experiments,
# Proc. Natl. Acad. Sci. U.S.A. 117, 1447 (2021).
# The trajectory is coarse-grained by crude assignment to histogram peaks.
# This script takes about 16 hrs to run.

if [ -n "$GMEX_DATA_DIR" ]; then
    datadir="$GMEX_DATA_DIR"/BMF1
else
    datadir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/data/BMF1
fi
mkdir -p "$datadir"/ATP_3mM/ "$datadir"/ATPgS_1uM/
curl -L "https://zenodo.org/records/5567245/files/Fig2D_ATP3mM_45kfps.txt?download=1" -o "$datadir"/ATP_3mM/raw.txt
curl -L "https://zenodo.org/records/5567245/files/Fig3D_ATPgS1uM_1kfps.txt?download=1" -o "$datadir"/ATPgS_1uM/raw.txt

python3 scripts/preprocess/BMF1.py
python3 scripts/train/BMF1_ATPgS_1uM.py
python3 scripts/train/BMF1_ATP_3mM.py
python3 scripts/postprocess/BMF1.py
