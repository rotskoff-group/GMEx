#!/bin/bash
# bin/CFTR.sh

# Script for DTGME analysis of CFTR FRET traces from
# J. Levring, D.S. Terry, Z. Kilic, G. Fitzgerald, S.C. Blanchard, and J. Chen,
# CFTR function, pathology, and pharmacology at single-molecule resolution,
# Nature 616, 606 (2023).
# Traces are coarse-grained to the paper's idealized efficiencies.
# This script takes about 6 hrs to run.

if [ -n "$GMEX_DATA_DIR" ]; then
    datadir="$GMEX_DATA_DIR"
else
    datadir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/data
fi
mkdir -p "$datadir"
curl -L "https://zenodo.org/records/20823191/files/Levring-2023-CFTR-data.zip?download=1" -o "$datadir"/Levring-2023-CFTR-data.zip
unzip -o "$datadir"/Levring-2023-CFTR-data.zip -d "$datadir"
rm "$datadir"/Levring-2023-CFTR-data.zip

python3 scripts/preprocess/CFTR.py
python3 scripts/train/CFTR.py
python3 scripts/postprocess/CFTR.py
