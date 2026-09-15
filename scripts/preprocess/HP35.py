"""scripts/preprocess/HP35.py

Preprocess Lys24-Nle/Lys29-Nle HP35 trajectory from
S. Piana, K. Lindorff-Larsen, D.E. Shaw,
Protein folding kinetics and thermodynamics from atomistic simulation,
Proc. Natl. Acad. Sci. U.S.A. 109, 17845 (2012).
The trajectory is coarse-grained following the contact-based procedure of
D. Nagel, S. Sartore, and G. Stock,
Selecting features for Markov modeling: a case study of HP35,
J. Chem. Theory Comput. 19, 3391 (2023).
The native state remains a macrostate,
the two near-native states are together a macrostate,
and the remaining nine unfolded states are together a macrostate.
This script takes about 30 s to run.
"""

import json
import os
from pathlib import Path

import numpy as np

from gmex.utils import (
    GROUPS_FN,
    METADATA_FN,
    STATE_LIST_FN,
    get_data_dir,
    save_metadata,
)

OUTDIR = get_data_dir() / Path("HP35/")
DT = 0.2  # data are sampled at an interval of 0.2 ns
# coarse-grained trajectories are processed into this file by Nagel's code
RAWPATH = OUTDIR / Path("raw.txt")
# native, backbone-folded, hydrophobically collapsed, and fully unfolded
GROUPS = [[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11]]


def _load_traj(p: Path) -> np.ndarray:
    """Load raw data as an np.ndarray."""
    if not p.exists():
        raise FileNotFoundError(str(p))
    x = np.loadtxt(p, dtype=int, comments="#")
    x = np.atleast_1d(x).astype(int) - 1  # they use 1-indexing
    return x


if __name__ == "__main__":
    print("Saving HP35 data...")
    data = _load_traj(RAWPATH).tolist()
    with open(OUTDIR / STATE_LIST_FN, "w") as f:
        json.dump(data, f)
    with open(OUTDIR / GROUPS_FN, "w") as f:
        json.dump(GROUPS, f)
    METADATA = {
        "outdir": os.path.basename(OUTDIR),
        "seed": None,
        "nmicro": None,
        "nmacro": len(GROUPS),
        "ntraj": 1,
        "nsteps": None,
        "dt": DT,
        "rates": None,
        "probs": None,
        "type": "Markov chain",
        "comment": "CWJL: I filled this in manually. Timestep dt is in units of ns.",
    }
    save_metadata(METADATA, OUTDIR / METADATA_FN)
    print(f"HP35 data saved to {Path(os.getcwd()) / OUTDIR!s}.")
