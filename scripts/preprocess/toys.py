"""scripts/preprocess/toys.py

Simulate toy Markov jump processes on four-state cycles.
The hard toy lacks separation of timescales:
all its counterclockwise and clockwise rates are respectively 1.0 and 0.5.
The easy toy has separation of timescales:
only its unobserved counterclockwise and clockwise rate are respectively 1.0 and 0.5.
Its observed counterclockwise and clockwise rates are respectively 0.25 and 0.125.
States 2 and 3 are coarse-grained together in both toys, leaving states 0, 1, and 2U3.
TCL-GME-DT propgators and NZ-GME-DT memory kernels are analytically accessible.
This script takes about 5 min to run.
"""

### IMPORTS ###
import json

import lightning as L
import torch

from gmex.markov_process import MarkovChain, MarkovJumpProcess
from gmex.nakajima_zwanzig import DiscreteTimeGroundTruthNZGME
from gmex.utils import (
    GROUPS_FN,
    METADATA_FN,
    RATES_FN,
    STATE_LIST_FN,
    get_data_dir,
    save_metadata,
)

### CONFIGURATIONS ###
SEED = 42  # randomization seed for sampling trajectories
N_STEPS = int(1e6)  # number of steps to simulate
N_TRAJS = 10  # number of replicates
DT = 1.0  # timestep for discretization
N_LAGS_TOTAL = 10  # total nonzero lags for which to compute ground-truth propagators
RATES: dict[str, torch.Tensor] = {
    "hard": torch.tensor(
        [
            [0.0, 0.5, 0.0, 1.0],
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    ),
    "easy": torch.tensor(
        [
            [0.000, 0.125, 0.000, 0.250],
            [0.250, 0.000, 0.125, 0.000],
            [0.000, 0.250, 0.000, 0.500],
            [0.125, 0.000, 1.000, 0.000],
        ]
    ),
}
GROUPS = [[0], [1], [2, 3]]  # states 2 and 3 are coarse-grained together

DATA_DIR = get_data_dir() / "toys"
TRUE_US_FN = "toy-Us-true.pt"
TRUE_KS_FN = "toy-Ks-true.pt"  # computed analytically
TRUE_GS_FN = "toy-Gs-true.pt"
METADATA = {
    "outdir": None,
    "seed": SEED,
    "nmicro": 4,
    "nmacro": len(GROUPS),
    "ntraj": N_TRAJS,
    "nsteps": N_STEPS,
    "dt": DT,
    "rates": None,
    "probs": None,
    "type": "Markov jump process",
    "comment": "CWJL: I filled this in manually. Timestep dt is in arbitrary units.",
}


if __name__ == "__main__":
    ### SETUP ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for toyname, rates in RATES.items():
        print(f"Saving {toyname!s} toy data...")
        L.seed_everything(SEED)  # easier to replicate if we reseed for each toy
        outdir = DATA_DIR / str(toyname)
        outdir.mkdir(parents=True, exist_ok=True)

        ### SAVE METADATA ###
        metadata = dict(METADATA)
        metadata["outdir"] = str(outdir)
        metadata["rates"] = str(outdir / RATES_FN)
        save_metadata(metadata, outdir / METADATA_FN)
        torch.save(rates, outdir / RATES_FN)
        with open(outdir / GROUPS_FN, "w") as f:
            json.dump(GROUPS, f)

        ### GROUND-TRUTH PROPAGATORS ###
        mjp = MarkovJumpProcess(device=device)
        mjp.parameterize_from_rates(rates)  # parameterize a Markov jump process...
        mc = MarkovChain(  # to parameterize a Markov chain...
            torch.linalg.matrix_exp(mjp.transition_rate_matrix()), 1.0, device=device
        )  # to parameterize a ground-truth NZ-GME-DT...
        assert isinstance(mc.L, torch.Tensor), "Expected homogeneous Markov chain."
        nzgme = DiscreteTimeGroundTruthNZGME(mc, groups=GROUPS)

        Us_true = torch.empty(
            (N_LAGS_TOTAL + 1, nzgme.n_macrostates, nzgme.n_macrostates),
            dtype=torch.float64,
            device=device,
        )
        for lag in range(N_LAGS_TOTAL + 1):
            Us_true[lag] = nzgme.get_U_dtgme(mc.L, lag)
        torch.save(Us_true, outdir / TRUE_US_FN)

        Ks_true = torch.empty(
            (N_LAGS_TOTAL, nzgme.n_macrostates, nzgme.n_macrostates),
            dtype=torch.float64,
            device=device,
        )
        for lag in range(N_LAGS_TOTAL):
            Ks_true[lag] = nzgme.get_K_dtgme(mc.L, lag)
        torch.save(Ks_true, outdir / TRUE_KS_FN)

        Gs_true = torch.empty(
            (N_LAGS_TOTAL + 1, nzgme.n_macrostates, nzgme.n_macrostates),
            dtype=torch.float64,
            device=device,
        )
        Gs_true[0] = torch.eye(nzgme.n_macrostates, dtype=torch.float64, device=device)
        Gs_true[1:] = torch.linalg.solve(
            Us_true[:-1].transpose(-1, -2),
            Us_true[1:].transpose(-1, -2),
        ).transpose(-1, -2)  # to parameterize a ground-truth TCL-GME-DT
        torch.save(Gs_true, outdir / TRUE_GS_FN)

        ### SAMPLE TRAJECTORIES ###
        _, trajs = mc.sample(N_STEPS, n_trajs=N_TRAJS, verbose=True)
        with open(outdir / STATE_LIST_FN, "w") as f:
            f.write("[")
            for i in range(N_TRAJS):  # less memory intensive than writing all at once
                if i > 0:
                    f.write(",")
                json.dump(trajs[i].tolist(), f)
            f.write("]")
        del trajs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Saved {toyname!s} toy data to {outdir!s}.")
