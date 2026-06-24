#!/bin/bash
# bin/HP35.sh

# Script for TCL-GME-DT analysis of Lys24-Nle/Lys29-Nle HP35 trajectory from
# S. Piana, K. Lindorff-Larsen, D.E. Shaw,
# Protein folding kinetics and thermodynamics from atomistic simulation,
# Proc. Natl. Acad. Sci. U.S.A. 109, 17845 (2012).
# The trajectory is coarse-grained following the contact-based procedure of
# D. Nagel, S. Sartore, and G. Stock,
# Selecting features for Markov modeling: a case study of HP35,
# J. Chem. Theory Comput. 19, 3391 (2023).
# The native state remains a macrostate,
# the two near-native states are together a macrostate,
# and the remaining nine unfolded states are together a macrostate.
# WARNING: This script takes about 72 hrs to run.

if [ -n "$GMEX_DATA_DIR" ]; then
    datadir="$GMEX_DATA_DIR"/HP35
else
    datadir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/data/HP35
fi
mkdir -p "$datadir"
curl -L https://github.com/moldyn/HP35/raw/refs/heads/main/MPP/hp35.mindists2.gaussian10f_microstates_pcs5_p153.mpp50_transitions.dat.renamed_by_q.pop0.005_qmin0.50.macrotraj_lumped13 -o "$datadir"/raw.txt

python3 scripts/preprocess/HP35.py
python3 scripts/train/HP35.py
python3 scripts/postprocess/HP35.py
