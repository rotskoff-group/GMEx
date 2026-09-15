"""scripts/train/BMF1_ATPgS_1uM.py

Train TCL-GME-DT models on trajectories of bovine mitochondrial F1 ATPase in 1uM ATPgS from
R. Kobayashi, H. Ueno, C.B. Li, and H. Noji,
Rotary catalysis of bovine mitochondrial F1-ATPase studied by single-molecule experiments,
Proc. Natl. Acad. Sci. U.S.A. 117, 1447 (2021).
The trajectory has been coarse-grained by crude assignment to histogram peaks.
This script takes about 4 hrs to run.
"""

### IMPORTS ###
import json

import lightning as L
import torch
from tqdm import tqdm

from gmex.max_likelihood import get_Gs_mle
from gmex.utils import (
    StateSeqDataset,
    get_data_dir,
    get_results_dir,
    get_split,
    load_times_macrostates,
)

### CONFIGURATIONS ###
SEED = 42  # randomization seed for bootstrapping
RESAMPLING = 1  # we want a 1 ms timestep and data are sampled at 1 ms
N_PER_SEG = 120  # we want 50 records of 120 ms each from a 6000-sample trajectory
N_LAGS_TOTAL = 16  # number of lags for which to compute count matrices
N_LAGS_TRUNC = 2  # TCL-GME-DT propagator plateaus at lag of 2 ms
N_BOOTSTRAPS = 1000  # number of bootstrap samples
REVERSIBLE = False  # it is an ATPase

TOL = 1e-12  # numerical tolerance
MIN_ENTRY = 1e-24  # smallest permitted entry of a stochastic matrix
ETA = 1.0  # initial learning rate
GRAD_CLIP = 1e3  # gradient clipping
LINE_SEARCH = False  # whether to line search mirror-descent iterations
MAX_ITERS = 1000000  # maximum mirror-descent iterations (the vast majority will take much less than this)
MAX_ITERS_PROJ = (
    10000  # maximum outer projection iterations per mirror-descent iteration
)
MAX_ITERS_LS = 50  # maximum line-search iterations per mirror-descent iteration (ignored as LINE_SEARCH == False)
MAX_ITERS_PROJ_SYMM = 2500  # maximum KRU iterations per outer projection iteration (ignored as REVERSIBLE == False)
MAX_ITERS_PROJ_COMM = 250  # maximum Netwon iterations per outer projection iteration (ignored as REVERSIBLE == False)
MAX_ITERS_PROJ_LS = 25  # maximum line-search iterations per Newton iteration (ignored as REVERSIBLE == False)
ARMIJO = 1e-9  # Armijo line-search improvement coefficient for mirror descent (ignored as LINE_SEARCH == False)
ARMIJO_PROJ = 1e-4  # Armijo line-search improvement coefficient for projection (ignored as REVERSIBLE == False)
LS_UPDATE = 0.5  # factor by which to change learning rate per line-search iteration (ignored as LINE_SEARCH == False)

DATA_DIR = get_data_dir() / "BMF1/ATPgS_1uM"
RES_DIR = get_results_dir() / "BMF1/ATPgS_1uM"  # directory for saving results
MAIN_RES_GS_FN = (
    "BMF1-ATPgS-1uM-M6-L16-TCL-Gs.pt"  # filename to save mean TCL-GME-DT propagators
)
MAIN_RES_MD_FN = "BMF1-ATPgS-1uM-M6-L16-TCL-Gs-metadata.json"  # filename to save metadata for preceding
BOOTSTRAP_DIR = RES_DIR / "bootstraps"  # directory for saving bootstrap results
BOOTSTRAP_GS_FN = "BMF1-ATPgS-1uM-M6-L2-TCL-Gs-BS*.pt"  # filename to save bootstrap TCL-GME-DT propagators
BOOTSTRAP_MD_FN = "BMF1-ATPgS-1uM-M6-L2-TCL-Gs-BS*-metadata.json"  # filename to save metadata for preceding


if __name__ == "__main__":
    ### SETUP ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.seed_everything(SEED)
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)

    ### DATALOADING ###
    dt, _, _, traj, nmacro = load_times_macrostates(DATA_DIR, device=device)
    data = [
        traj[  # segment trajectory into records
            i * N_PER_SEG : (i + 1) * N_PER_SEG
        ]
        for i in range(len(traj) // N_PER_SEG)
    ]
    dataset = get_split(dt, data=data, train_frac=1.0, context_length=1, verbose=False)
    assert isinstance(dataset, StateSeqDataset)

    ### MEAN RESULTS ###
    Cs = dataset.get_count_matrices(
        max_lag=N_LAGS_TOTAL, sample_interval=RESAMPLING, bootstrap=False, verbose=False
    )
    lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

    Gs, info = get_Gs_mle(
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
        verbose=True,
    )

    torch.save(Gs, RES_DIR / MAIN_RES_GS_FN)
    with open(RES_DIR / MAIN_RES_MD_FN, "w") as f:
        json.dump(info, f)

    ### BOOTSTRAPS ###
    for i in tqdm(range(N_BOOTSTRAPS), desc="Bootstraps"):
        Cs = dataset.get_count_matrices(
            max_lag=N_LAGS_TRUNC,
            sample_interval=RESAMPLING,
            bootstrap=True,
            verbose=False,
        )
        lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

        Gs, info = get_Gs_mle(
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
            verbose=False,
        )

        Gs_fn = BOOTSTRAP_GS_FN.replace("*", str(i))
        metadata_fn = BOOTSTRAP_MD_FN.replace("*", str(i))
        torch.save(Gs, BOOTSTRAP_DIR / Gs_fn)
        with open(BOOTSTRAP_DIR / metadata_fn, "w") as f:
            json.dump(info, f)

    print(f"BMF1 (1uM ATPgS) models saved to {RES_DIR!s}")
