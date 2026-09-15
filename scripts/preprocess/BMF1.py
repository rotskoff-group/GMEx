"""scripts/preprocess/BMF1.py

Preprocess bovine mitochondrial F1 ATPase trajectories from
R. Kobayashi, H. Ueno, C.B. Li, and H. Noji,
Rotary catalysis of bovine mitochondrial F1-ATPase studied by single-molecule experiments,
Proc. Natl. Acad. Sci. U.S.A. 117, 1447 (2021).
The trajectory is coarse-grained by crude assignment to histogram peaks.
This script takes about 30 s to run.
"""

import json
import os
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from gmex.utils import METADATA_FN, STATE_LIST_FN, get_data_dir, save_metadata


class _ExptData(TypedDict):
    """Timestep and rotor-angle cuptoints."""

    dt: float
    cp: list[int]


OUTDIR = get_data_dir() / Path("BMF1/")
EXPTS: dict[str, _ExptData] = {  # 3 mM ATP sampled at 45 kfps
    "ATP_3mM": {"dt": 1.0 / 45.0, "cp": [80, 120, 200, 240, 300, 360]},
    "ATPgS_1uM": {"dt": 1.0, "cp": [40, 120, 160, 240, 300, 360]},
}  # 1 uM ATPgS sampled at 1 kfps
RAW_FN = "raw.txt"
METADATA: dict[str, str | int | float | None] = {
    "outdir": None,
    "seed": None,
    "nmicro": None,
    "nmacro": None,
    "ntraj": 1,
    "nsteps": None,
    "dt": None,
    "rates": None,
    "probs": None,
    "type": "Markov chain",
    "comment": "CWJL: I filled this in manually. Timestep dt is in units of ms.",
}


def _load_traj(p: Path, cutpoints: list[int]) -> list[int]:
    """Load raw trajectory, discretize, and return as a list."""
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p, sep="\t")
    # we want intervals open on the right
    # they also put in some kind of 290 degree offset
    return np.searchsorted(
        cutpoints, (df["Angle"] + 290) % 360, side="right"
    ).tolist()  # return as a list


if __name__ == "__main__":
    for expt in EXPTS:
        print(f"Saving {expt} BMF1 data...")
        outdir = OUTDIR / expt
        traj = _load_traj(outdir / RAW_FN, EXPTS[expt]["cp"])
        with open(outdir / STATE_LIST_FN, "w") as f:
            json.dump(traj, f)

        metadata = dict(METADATA)
        metadata["outdir"] = str(Path("BMF1/") / expt)
        metadata["dt"] = float(EXPTS[expt]["dt"])
        metadata["nmacro"] = len(EXPTS[expt]["cp"])
        save_metadata(metadata, outdir / METADATA_FN)
        print(
            f"{expt.replace('_', ' ')} BMF1 data saved to {Path(os.getcwd()) / outdir!s}."
        )
