"""GMEx/scripts/train/toys.py

Train DTGME models of a toy Markov jump process on a four-state cycle.
The hard toy lacks separation of timescales:
all its counterclockwise and clockwise rates are respectively 1.0 and 0.5.
The easy toy has separation of timescales:
only its unobserved counterclockwise and clockwise rate are respectively 1.0 and 0.5.
Its observed counterclockwise and clockwise rates are respectively 0.25 and 0.125.
States 2 and 3 are coarse-grained together in both toys, leaving states 0, 1, and 2U3.
TCL-GME-DT propgators and NZ-GME-DT memory kernels are analytically accessible.
This script takes about 8 hrs to run.
"""

### IMPORTS ###
import json
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import torch
from tqdm import tqdm

from gmex.max_likelihood import get_Gs_mle, get_Us_mle
from gmex.utils import (
    StateSeqDataset,
    get_data_dir,
    get_results_dir,
    get_split,
    load_times_macrostates,
)

### CONFIGURATIONS ###
N_STEPS = [int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)]  # trajectory lengths
N_LAGS_TRUNC = 3  # TCL-GME-DT propagators plateau at a lag of 3
REVERSIBLE = False  # the toy MJP is a cycle

TOL = 1e-12  # numerical tolerance
MIN_ENTRY = 1e-24  # smallest permitted entry of a stochastic matrix
ETA = 0.1  # initial learning rate
GRAD_CLIP = 1e3  # gradient clipping
LINE_SEARCH = False  # whether to line search mirror-descent iterations
MAX_ITERS = 10000000  # maximum mirror-descent iterations; vast majority will take much less than this
MAX_ITERS_PROJ = (
    100000  # maximum outer projection iterations per mirror-descent iteration
)
MAX_ITERS_LS = 50  # maximum line-search iterations per mirror-descent iteration (ignored as LINE_SEARCH == False)
MAX_ITERS_PROJ_SYMM = 2500  # maximum KRU iterations per outer projection iteration (ignored as REVERSIBLE == False)
MAX_ITERS_PROJ_COMM = 250  # maximum Netwon iterations per outer projection iteration (ignored as REVERSIBLE == False)
MAX_ITERS_PROJ_LS = 25  # maximum line-search iterations per Newton iteration (ignored as REVERSIBLE == False)
ARMIJO = 1e-9  # Armijo line-search improvement coefficient for mirror descent (ignored as LINE_SEARCH == False)
ARMIJO_PROJ = 1e-4  # Armijo line-search improvement coefficient for projection (ignored as REVERSIBLE == False)
LS_UPDATE = 0.5  # factor by which to change learning rate per line-search iteration (ignored as LINE_SEARCH == False)

DATA_DIR = get_data_dir() / "toys"
RES_DIR = get_results_dir() / "toys"
TOYNAMES = ["easy", "hard"]
REP_US_FN = "toy-Us-rep*.pt"
REP_US_MD_FN = "toy-Us-rep*-metadata.json"
REP_GS_FN = "toy-Gs-rep*.pt"
REP_GS_MD_FN = "toy-Gs-rep*-metadata.json"


### HELPER ###
def _n_power10_tag(n: int) -> str:
    """Convert powers of ten like int(1e2) -> 'N1e2'."""
    p = int(np.log10(n))
    if 10**p != n:
        raise ValueError(f"Expected an integer power of ten, got {n}.")
    return f"N1e{p}"


if __name__ == "__main__":
    ### SETUP ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for toyname in TOYNAMES:
        datadir = DATA_DIR / str(toyname)
        dt, _, _, trajs, nmacro = load_times_macrostates(datadir, device=device)
        resdir = RES_DIR / str(toyname)
        for n in N_STEPS:  # iterate over sample sizes
            ndir = resdir / _n_power10_tag(n)
            ndir.mkdir(parents=True, exist_ok=True)

            ### GME PARAMETERIZATION ###
            warn(
                "Sinkhorn scaling sometimes converges to loss around 2e-12 during intermediate mirror steps in the adversarial example. "
                "Sinkhorn always converges to loss less than 1e-12 during final iterations, "
                "so this does not affect the estimated solution.",
                UserWarning,
            )
            with catch_warnings():
                simplefilter("ignore", category=UserWarning)
                sample_iter = tqdm(
                    range(len(trajs)), desc=f"Samples {n:.0e}", leave=False
                )
                for i in sample_iter:  # iterate over trajectories
                    dataset = get_split(
                        dt,
                        data=trajs[i][1 : n + 1],
                        train_frac=1.0,
                        context_length=1,
                        verbose=False,
                    )
                    assert isinstance(dataset, StateSeqDataset)
                    Cs = dataset.get_count_matrices(
                        max_lag=N_LAGS_TRUNC,
                        sample_interval=1,
                        bootstrap=False,
                        verbose=False,
                    )
                    lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

                    Us, Us_info = get_Us_mle(
                        Cs,
                        lim_dist,
                        tol=TOL,
                        min_entry=MIN_ENTRY,
                        eta=ETA,
                        grad_clip=GRAD_CLIP,
                        line_search=LINE_SEARCH,
                        max_iters=MAX_ITERS,
                        max_iters_proj=MAX_ITERS_PROJ,
                        max_iters_ls=MAX_ITERS_LS,
                        ls_update=LS_UPDATE,
                        armijo=ARMIJO,
                        reversible=REVERSIBLE,
                        verbose=False,
                    )
                    torch.save(Us, ndir / REP_US_FN.replace("*", str(i)))
                    with open(ndir / REP_US_MD_FN.replace("*", str(i)), "w") as f:
                        json.dump(Us_info, f)
                    del Us, Us_info

                    Gs, Gs_info = get_Gs_mle(
                        Cs,
                        lim_dist,
                        tol=TOL,
                        min_entry=MIN_ENTRY,
                        eta=ETA,
                        grad_clip=GRAD_CLIP,
                        line_search=LINE_SEARCH,
                        max_iters=MAX_ITERS,
                        max_iters_proj=MAX_ITERS_PROJ,
                        max_iters_ls=MAX_ITERS_LS,
                        max_iters_proj_symm=MAX_ITERS_PROJ_SYMM,
                        max_iters_proj_comm=MAX_ITERS_PROJ_COMM,
                        max_iters_proj_ls=MAX_ITERS_PROJ_LS,
                        ls_update=LS_UPDATE,
                        armijo=ARMIJO,
                        armijo_proj=ARMIJO_PROJ,
                        reversible=REVERSIBLE,
                        precomputed=None,  # no fit reuse to show convergence
                        verbose=False,
                        postfix_callback=lambda postfix, i=i, bar=sample_iter: (
                            bar.set_postfix({"rep": i, **postfix})
                        ),
                    )
                    torch.save(Gs, ndir / REP_GS_FN.replace("*", str(i)))
                    with open(ndir / REP_GS_MD_FN.replace("*", str(i)), "w") as f:
                        json.dump(Gs_info, f)
                    del Gs, Gs_info

    print(f"Toy-model results saved to {RES_DIR!s}")
