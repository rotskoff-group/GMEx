#!/usr/bin/env python3
# GMEx/scripts/train/HP35.py

"""
Train TCL-GME-DT models on Lys24-Nle/Lys29-Nle HP35 trajectory from
S. Piana, K. Lindorff-Larsen, D.E. Shaw,
Protein folding kinetics and thermodynamics from atomistic simulation,
Proc. Natl. Acad. Sci. U.S.A. 109, 17845 (2012).
The trajectory was first coarse-grained following the procedure of
D. Nagel, S. Sartore, and G. Stock,
Selecting features for Markov modeling: a case study of HP35,
J. Chem. Theory Comput. 19, 3391 (2023).
The native state remains a macrostate,
the two near-native states are together a macrostate,
and the remaining nine unfolded states are together a macrostate.
WARNING: This script takes about 72 hrs to run.
"""


### IMPORTS ###
import json

import lightning as L
import torch
from tqdm import tqdm

from gmex.max_likelihood import get_Gs_mle
from gmex.utils import get_data_dir, get_split, load_times_macrostates, get_results_dir


### CONFIGURATIONS ###
RESAMPLING = 5 # we want a 1 ns timestep and data are sampled every 200 ps
N_PER_SEG = 4860 # we want 314 records of 972 ns each from a 1526040-sample trajectory
N_LAGS_TOTAL = 64 # number of lags for which to compute count matrices
N_LAGS_TRUNC = 32 # TCL-GME-DT propagator plateaus at lag of 32 ns
N_BOOTSTRAPS = 1000 # number of bootstrap samples
REVERSIBLE = True # protein folding is reversible

TOL = 1e-12 # numerical tolerance
MIN_ENTRY = 1e-24 # smallest permitted entry of a stochastic matrix
ETA = 1.0 # initial learning rate
GRAD_CLIP = 1e3 # gradient clipping
LINE_SEARCH = False # whether to line search mirror-descent iterations
MAX_ITERS = 100000 # maximum mirror-descent iterations; vast majority will take much less than this
MAX_ITERS_PROJ = 10000 # maximum outer projection iterations per mirror-descent iteration
MAX_ITERS_LS = 50 # maximum line-search iterations per mirror-descent iteration (ignored as LINE_SEARCH == False)
MAX_ITERS_PROJ_SYMM = 2500 # maximum KRU iterations per outer projection iteration
MAX_ITERS_PROJ_COMM = 250 # maximum Netwon iterations per outer projection iteration
MAX_ITERS_PROJ_LS = 25 # maximum line-search iterations per Newton iteration
ARMIJO = 1e-9 # Armijo line-search improvement coefficient for mirror descent (ignored as LINE_SEARCH == False)
ARMIJO_PROJ = 1e-4 # Armijo line-search improvement coefficient for projection
LS_UPDATE = 0.5 # factor by which to change learning rate per line-search iteration (ignored as LINE_SEARCH == False)

DATA_DIR = get_data_dir() / 'HP35' # directory in which data are saved
RES_DIR = get_results_dir() / 'HP35' # directory for saving results
MAIN_RES_GS_FN = 'HP35-M3-L64-TCL-Gs.pt' # filename to save mean TCL-GME-DT propagators
MAIN_RES_MD_FN = 'HP35-M3-L64-TCL-Gs-metadata.json' # filename to save metadata for preceding
BOOTSTRAP_DIR = RES_DIR / 'bootstraps' # directory for saving bootstrap results
BOOTSTRAP_GS_FN = 'HP35-M3-L32-TCL-Gs-BS*.pt' # filename to save bootstrap TCL-GME-DT propagators
BOOTSTRAP_MD_FN = 'HP35-M3-L32-TCL-Gs-BS*-metadata.json' # filename to save metadata for preceding


if __name__ == "__main__":

    ### SETUP ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)

    ### DATALOADING ###
    dt, _, _, traj, nmacro = load_times_macrostates(DATA_DIR, device=device)
    traj = traj[1:] # drop initial state
    data = [traj[ # segment trajectory into records
        i * N_PER_SEG:(i + 1) * N_PER_SEG
        ] for i in range(len(traj) // N_PER_SEG)]
    dataset = get_split(
        dt, data=data, train_frac=1.0, context_length=1, verbose=False
    )
    
    ### MEAN RESULTS ###
    Cs = dataset.get_count_matrices(
        max_lag=N_LAGS_TOTAL,
        sample_interval=RESAMPLING,
        bootstrap=False,
        verbose=False
    )
    lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

    Gs, info = get_Gs_mle(Cs, lim_dist,
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
                          verbose=True)

    torch.save(Gs, RES_DIR / MAIN_RES_GS_FN)
    with open(RES_DIR / MAIN_RES_MD_FN, 'w') as f:
        json.dump(info, f)

    ### BOOTSTRAPS ###
    bootstrap_iter = tqdm(range(N_BOOTSTRAPS), desc='Bootstraps', leave=False)
    for i in bootstrap_iter:
        # each bootstrap takes 5 min
        # we reseed every bootstrap in case we need to restart
        L.seed_everything(i)

        Cs = dataset.get_count_matrices(
            max_lag=N_LAGS_TRUNC,
            sample_interval=RESAMPLING,
            bootstrap=True,
            verbose=False
        )
        lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

        Gs, info = get_Gs_mle(
            Cs, lim_dist,
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
            postfix_callback=lambda postfix: bootstrap_iter.set_postfix({'bootstrap': i, **postfix})
        )
        
        Gs_fn = BOOTSTRAP_GS_FN.replace('*', str(i))
        metadata_fn = BOOTSTRAP_MD_FN.replace('*', str(i))
        torch.save(Gs, BOOTSTRAP_DIR / Gs_fn)
        with open(BOOTSTRAP_DIR / metadata_fn, 'w') as f:
            json.dump(info, f)

    print(f'HP35 models saved to {str(RES_DIR)}')
