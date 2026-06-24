#!/usr/bin/env python3
# GMEx/scripts/train/CFTR.py

"""
Train GME models on CFTR FRET traces from
J. Levring, D.S. Terry, Z. Kilic, G. Fitzgerald, S.C. Blanchard, and J. Chen,
CFTR function, pathology, and pharmacology at single-molecule resolution,
Nature 616, 606 (2023).
Traces are coarse-grained to the paper's idealized efficiencies.
This script takes about 6 hrs to run.
"""


### IMPORTS ###
import json

import lightning as L
import torch
from tqdm import tqdm

from gmex.max_likelihood import get_Us_mle, get_Gs_mle
from gmex.utils import get_data_dir, get_split, load_times_macrostates, get_results_dir


### CONFIGURATIONS ###
SEED = 42 # randomization seed for bootstrapping
RESAMPLING = 1 # we want a 0.1 s timestep and data are sampled every 0.1 s
N_LAGS_TOTAL = 40 # number of lags for which to compute count matrices
N_LAGS_TRUNC = 16 # kernels are negligible past 1.6 s
N_BOOTSTRAPS = 1000 # number of bootstrap samples
REVERSIBLE = False # CFTR exhibits nonequilibrium gating

TOL = 1e-12 # numerical tolerance
MIN_ENTRY = 1e-24 # smallest permitted entry of a stochastic matrix
ETA = 1.0 # initial learning rate
GRAD_CLIP = 1e3 # gradient clipping
LINE_SEARCH = False # whether to line search mirror-descent iterations
MAX_ITERS = 1000000 # maximum mirror-descent iterations (the vast majority will take much less than this)
MAX_ITERS_PROJ = 10000 # maximum outer projection iterations per mirror-descent iteration
MAX_ITERS_LS = 50 # maximum line-search iterations per mirror-descent iteration (ignored as LINE_SEARCH == False)
MAX_ITERS_PROJ_SYMM = 2500 # maximum KRU iterations per outer projection iteration (ignored as REVERSIBLE == False)
MAX_ITERS_PROJ_COMM = 250 # maximum Netwon iterations per outer projection iteration (ignored as REVERSIBLE == False)
MAX_ITERS_PROJ_LS = 25 # maximum line-search iterations per Newton iteration (ignored as REVERSIBLE == False)
ARMIJO = 1e-9 # Armijo line-search improvement coefficient for mirror descent (ignored as LINE_SEARCH == False)
ARMIJO_PROJ = 1e-4 # Armijo line-search improvement coefficient for projection (ignored as REVERSIBLE == False)
LS_UPDATE = 0.5 # factor by which to change learning rate per line-search iteration (ignored as LINE_SEARCH == False)

DATA_DIR = get_data_dir() / 'CFTR'
RES_DIR = get_results_dir() / 'CFTR' # directory for saving results
BOOTSTRAP_DIR = 'bootstraps' # directory for saving bootstrap results
MAIN_RES_US_FN = 'CFTR-L40-NZ-Us.pt' # filename to save mean transition matrices
MAIN_RES_US_MD_FN = 'CFTR-L40-NZ-Us-metadata.json' # filename to save metadata for preceding
BOOTSTRAP_US_FN = 'CFTR-L16-NZ-Us-BS*.pt' # filename to save bootstrap transition matrices
BOOTSTRAP_US_MD_FN = 'CFTR-L16-NZ-Us-BS*-metadata.json' # filename to save metadata for preceding
MAIN_RES_GS_FN = 'CFTR-L2-TCL-Gs.pt' # filename to save mean TCL-GME-DT propagators
MAIN_RES_GS_MD_FN = 'CFTR-L2-TCL-Gs-metadata.json' # filename to save metadata for preceding
BOOTSTRAP_GS_FN = 'CFTR-L2-TCL-Gs-BS*.pt' # filename to save bootstrap TCL-GME-DT propagators
BOOTSTRAP_GS_MD_FN = 'CFTR-L2-TCL-Gs-BS*-metadata.json' # filename to save metadata for preceding

EXPTS = [
    'WT_3mM_ATP',
    'WT_3mM_ATP_10uM_GLPG1837',
    'G551D_3mM_ATP',
    'G551D_3mM_ATP_10uM_GLPG1837',
    'L927P_3mM_ATP',
    'L927P_3mM_ATP_10uM_GLPG1837'
]

if __name__ == "__main__":
    
    ### SETUP ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L.seed_everything(SEED)
    
    _get_Us = lambda Cs, pi, v: get_Us_mle(
            Cs,
            pi,
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
            verbose=v
        )

    _get_Gs = lambda Cs, pi, pc, v: get_Gs_mle(
        Cs[:3], # we only use up to second lag for irreversibility estimates
        pi,
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
        precomputed=pc, # we want to use first two transition matrices as a warm start
        verbose=v
    )

    for expt in EXPTS:
        print(f'Training {expt.replace('_', ' ')} CFTR models...')
        res_dir = RES_DIR / expt
        bootstrap_dir = res_dir / BOOTSTRAP_DIR
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        ### DATALOADING ###
        dt, _, _, data, nmacro = load_times_macrostates(DATA_DIR / expt, device=device)
        dataset = get_split(dt, data=data, train_frac=1.0, context_length=1)

        ### MEAN RESULTS ###
        Cs = dataset.get_count_matrices(
            max_lag=N_LAGS_TOTAL,
            sample_interval=RESAMPLING,
            bootstrap=False,
            verbose=False
        )
        lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

        Us, info = _get_Us(Cs, lim_dist, True)
        torch.save(Us, res_dir / MAIN_RES_US_FN)
        with open(res_dir / MAIN_RES_US_MD_FN, 'w') as f:
            json.dump(info, f)
        info_abridged = {0: {}, 1: dict(info[round(N_LAGS_TRUNC / 2)])}

        Gs, info = _get_Gs(
            Cs[::round(N_LAGS_TRUNC / 2)][:3],
            lim_dist,
            (Us[::round(N_LAGS_TRUNC / 2)][:2], info_abridged),
            True
        )
        torch.save(Gs, res_dir / MAIN_RES_GS_FN)
        with open(res_dir / MAIN_RES_GS_MD_FN, 'w') as f:
            json.dump(info, f)
        del Us, Gs, info
        

        ### BOOTSTRAPS ###
        for i in tqdm(range(N_BOOTSTRAPS), desc='Bootstraps'):
            Cs = dataset.get_count_matrices(
                max_lag=N_LAGS_TRUNC + 1, # we need ell + 1 positive lags for ell kernels
                sample_interval=RESAMPLING,
                bootstrap=True,
                verbose=False
            )
            lim_dist = torch.diag(Cs[0]) / Cs[0].sum()

            Us, info = _get_Us(Cs, lim_dist, False)
            Us_fn = BOOTSTRAP_US_FN.replace('*', str(i))
            Us_metadata_fn = BOOTSTRAP_US_MD_FN.replace('*', str(i))
            torch.save(Us, bootstrap_dir / Us_fn)
            with open(bootstrap_dir / Us_metadata_fn, 'w') as f:
                json.dump(info, f)
            info_abridged = {0: {}, 1: dict(info[round(N_LAGS_TRUNC / 2)])}

            Gs, info = _get_Gs(
                Cs[::round(N_LAGS_TRUNC / 2)][:3],
                lim_dist,
                (Us[::round(N_LAGS_TRUNC / 2)][:2], info_abridged),
                False
            )
            Gs_fn = BOOTSTRAP_GS_FN.replace('*', str(i))
            Gs_metadata_fn = BOOTSTRAP_GS_MD_FN.replace('*', str(i))
            torch.save(Gs, bootstrap_dir / Gs_fn)
            with open(bootstrap_dir / Gs_metadata_fn, 'w') as f:
                json.dump(info, f)
            del Us, Gs, info

        print(f'{expt.replace('_', ' ')} CFTR models saved to {str(res_dir)}')
