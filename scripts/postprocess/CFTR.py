"""scripts/postprocess/CFTR.py

Analyze DTGME models of CFTR FRET traces from
J. Levring, D.S. Terry, Z. Kilic, G. Fitzgerald, S.C. Blanchard, and J. Chen,
CFTR function, pathology, and pharmacology at single-molecule resolution,
Nature 616, 606 (2023).
Traces are coarse-grained to the paper's idealized efficiencies.
This script takes about 5 min to run.
"""

### IMPORTS ###
import fnmatch
import json
import os
from math import ceil
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
from tqdm import tqdm

from gmex.markov_process import MarkovChain
from gmex.nakajima_zwanzig import TransferMatrixNZGME, get_Ks_from_Us
from gmex.utils import get_bootstrap_curve_CI, get_data_dir, get_results_dir

### CONFIGURATIONS ###
DATA_DIR = get_data_dir() / "CFTR"  # directory in which data are saved
RES_DIR = get_results_dir() / "CFTR"  # directory for saving results
MAIN_RES_GS_FN = "CFTR-L2-TCL-Gs.pt"  # filename of saved mean TCL-GME-DT propagators
MAIN_RES_US_FN = "CFTR-L40-NZ-Us.pt"  # filename of saved mean transition matrices
BOOTSTRAP_DIR = "bootstraps"  # directory for saving bootstrap results
BOOTSTRAP_GS_FN = (
    "CFTR-L2-TCL-Gs-BS*.pt"  # filename of saved bootstrap TCL-GME-DT propagators
)
BOOTSTRAP_US_FN = (
    "CFTR-L16-NZ-Us-BS*.pt"  # filename of saved bootstrap transition matrices
)
# filename for experimental solvent-exchange curves from Fig. 3(a) of Levring et al.
SOLVENTX_EXPT_FN = DATA_DIR / "Levring-Fig-3a-lower-panel.xlsx"
TAU_EXCHANGE = 0.115  # mixing timescale in imaging chamber in s
MIXING_TIME = TAU_EXCHANGE * np.log(20)  # time to exchange 0.95 of solvent
NPATCHES_SOLVENTX = 42  # number of patches used for solvent-exchange curves
NSTEPS_RELAX = 1500  # we integrate full relaxations for 150 s
# filename for ATP turnover rates from Fig. 4(c) of Levring et al.
ATPASE_RATES_FN = DATA_DIR / "Levring-Fig-4c.csv"

# filename for solvent-exchange curves in panel (e)
SOLVENTX_CURVES_FN = "solvent-exchange-model-curves.json"
# filename for full model-based relaxation curves in panel (f)
RELAXATION_FN = "model-low-FRET-relaxations.json"
# filename for irreversibility measurements
IRREVERSIBILITY_FN = "irreversibilities.csv"
# filename for CFTR figure without ribbons in panel (a)
FIG_FN = "CFTR-fig-no-ribbon.pdf"

SEED = 42  # randomization seed for bootstrapping
CI_PCT = 0.95  # confidence-interval width
DT = 0.1  # data are sampled every 0.1 s
RESAMPLING = 1  # we want a 0.1 s timestep
N_LAGS_TOTAL = 40  # total number of kernels
N_LAGS_TRUNC = 16  # kernels vanish by a lag of 16
WT_EXPTS = ["WT_3mM_ATP", "WT_3mM_ATP_10uM_GLPG1837"]
G551D_EXPTS = ["G551D_3mM_ATP", "G551D_3mM_ATP_10uM_GLPG1837"]
L927P_EXPTS = ["L927P_3mM_ATP", "L927P_3mM_ATP_10uM_GLPG1837"]
MUT_EXPTS = G551D_EXPTS + L927P_EXPTS
EXPTS = WT_EXPTS + MUT_EXPTS
WT_APO_DIR = WT_EXPTS[0]


### HELPERS ###
def _get_bootstrap_fns(template: str, bootstrap_dir: Path):
    """Get filenames in BOOTSTRAP_DIR that match template."""
    return sorted(
        bootstrap_dir / name
        for name in os.listdir(bootstrap_dir)
        if fnmatch.fnmatch(name, template)
    )


if __name__ == "__main__":
    ### SETUP ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_CI_ppf_one_sided = -norm.ppf(0.5 - CI_PCT / 2.0).item()
    try:
        plt.style.use("rotskoff")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "CMU",
                "text.latex.preamble": r"\usepackage{amsfonts}",
            }
        )
    except OSError:
        pass
    plt.rcParams.update({"hatch.linewidth": 0.5})

    ### SOLVENT-EXCHANGE CURVES ###
    if SOLVENTX_CURVES_FN in os.listdir(RES_DIR):
        with open(RES_DIR / SOLVENTX_CURVES_FN, "r") as f:
            solventx_curves = json.load(f)
    else:
        L.seed_everything(SEED)
        solventx_curves = {}

        # load experimental solvent-exchange curves
        df = pd.read_excel(SOLVENTX_EXPT_FN)
        dt_inject = df["Time (s)"].iloc[1] - df["Time (s)"].iloc[0]
        start_timestep = ceil(MIXING_TIME / dt_inject)
        expt_solventx_mean = (
            df["Mean high FRET occupancy (%)"].iloc[start_timestep:] / 100
        )
        expt_solventx_err = (
            df["SD (%)"].iloc[start_timestep:]
            / 100
            / np.sqrt(NPATCHES_SOLVENTX)
            * norm_CI_ppf_one_sided
        )

        # save processed experimental solvent-exchange curves
        solventx_curves["expt_means"] = expt_solventx_mean.to_list()
        solventx_curves["expt_lower"] = (
            expt_solventx_mean - expt_solventx_err
        ).to_list()
        solventx_curves["expt_upper"] = (
            expt_solventx_mean + expt_solventx_err
        ).to_list()
        solventx_curves["expt_times"] = (
            df["Time (s)"].iloc[start_timestep:] - df["Time (s)"].iloc[start_timestep]
        ).to_list()

        # create function to sample initial high-FRET fractions
        solventx_start_mean = float(expt_solventx_mean.iloc[0])
        solventx_start_err = float(expt_solventx_err.iloc[0])
        _get_start_frac_high_FRET = lambda: float(
            norm.rvs(loc=solventx_start_mean, scale=solventx_start_err)
        )
        nsteps_solventx_models = ceil(
            (df["Time (s)"].max() - df["Time (s)"].iloc[start_timestep]) / DT
        )
        del df, expt_solventx_mean, expt_solventx_err

        # initialize solvent-exchange model curves
        bootstrap_WT_apo_dir = RES_DIR / WT_APO_DIR / BOOTSTRAP_DIR
        bootstrap_WT_apo_Us_fns = _get_bootstrap_fns(
            BOOTSTRAP_US_FN, bootstrap_WT_apo_dir
        )
        print(f"{len(bootstrap_WT_apo_Us_fns)} bootstraps detected")
        msm_bootstrap_solventx_buffer = np.empty(
            (len(bootstrap_WT_apo_Us_fns), nsteps_solventx_models + 1)
        )
        gme_bootstrap_solventx_buffer = msm_bootstrap_solventx_buffer.copy()

        # iterate over solvent-exchange bootstrap models
        for i in range(len(bootstrap_WT_apo_Us_fns)):
            p_high_0 = _get_start_frac_high_FRET()
            initial_dist = torch.tensor([1.0 - p_high_0, p_high_0], dtype=torch.float64)
            Us = torch.load(
                bootstrap_WT_apo_dir / bootstrap_WT_apo_Us_fns[i], weights_only=True
            )

            msm = TransferMatrixNZGME(Us[:2])
            _, integrals_msm = msm.integrate(
                nsteps_solventx_models, initial_dist=initial_dist
            )
            msm_bootstrap_solventx_buffer[i] = integrals_msm[:, 1].cpu().numpy()

            gme = TransferMatrixNZGME(Us[: N_LAGS_TRUNC + 2])
            _, integrals_gme = gme.integrate(
                nsteps_solventx_models, initial_dist=initial_dist
            )
            gme_bootstrap_solventx_buffer[i] = integrals_gme[:, 1].cpu().numpy()

        solventx_msm_lower, solventx_msm_upper = get_bootstrap_curve_CI(
            msm_bootstrap_solventx_buffer, CI_PCT
        )
        solventx_gme_lower, solventx_gme_upper = get_bootstrap_curve_CI(
            gme_bootstrap_solventx_buffer, CI_PCT
        )

        solventx_curves["msm_lower"] = solventx_msm_lower.tolist()
        solventx_curves["msm_upper"] = solventx_msm_upper.tolist()
        solventx_curves["gme_lower"] = solventx_gme_lower.tolist()
        solventx_curves["gme_upper"] = solventx_gme_upper.tolist()
        solventx_curves["model_times"] = (
            np.arange(nsteps_solventx_models + 1) * DT
        ).tolist()
        del (
            bootstrap_WT_apo_Us_fns,
            msm_bootstrap_solventx_buffer,
            gme_bootstrap_solventx_buffer,
            solventx_msm_lower,
            solventx_msm_upper,
            solventx_gme_lower,
            solventx_gme_upper,
        )

        # get mean solvent-exchange model curves
        Us = torch.load(RES_DIR / WT_APO_DIR / MAIN_RES_US_FN, weights_only=True)
        initial_dist = torch.tensor(
            [1.0 - solventx_start_mean, solventx_start_mean], dtype=torch.float64
        )

        msm = TransferMatrixNZGME(Us[:2])
        _, integrals_msm = msm.integrate(
            nsteps_solventx_models, initial_dist=initial_dist
        )
        solventx_curves["msm_means"] = integrals_msm[:, 1].cpu().numpy().tolist()

        gme = TransferMatrixNZGME(Us[: N_LAGS_TRUNC + 2])
        _, integrals_gme = gme.integrate(
            nsteps_solventx_models, initial_dist=initial_dist
        )
        solventx_curves["gme_means"] = integrals_gme[:, 1].cpu().numpy().tolist()

        del Us, msm, gme, integrals_msm, integrals_gme

        # save solvent-exchange curves
        with open(RES_DIR / SOLVENTX_CURVES_FN, "w") as f:
            json.dump(solventx_curves, f)

    #### INFERRED RELAXATION CURVES ###
    if RELAXATION_FN in os.listdir(RES_DIR):
        with open(RES_DIR / RELAXATION_FN, "r") as f:
            relax_curves = json.load(f)
    else:
        relax_curves = {}

        for expt in EXPTS:  # iterate over experiments
            bootstrap_dir = RES_DIR / expt / BOOTSTRAP_DIR
            bootstrap_Us_fns = _get_bootstrap_fns(BOOTSTRAP_US_FN, bootstrap_dir)
            print(
                f"{expt.replace('_', ' ')} CFTR: {len(bootstrap_Us_fns)} bootstraps detected."
            )

            msm_bootstrap_relax_buffer = np.empty(
                (len(bootstrap_Us_fns), NSTEPS_RELAX + 1)
            )
            gme_bootstrap_relax_buffer = msm_bootstrap_relax_buffer.copy()
            initial_dist = torch.tensor(
                [1.0, 0.0, 0.0] if expt in MUT_EXPTS else [1.0, 0.0],
                dtype=torch.float64,
            )

            for i in tqdm(range(len(bootstrap_Us_fns)), desc="Bootstraps"):
                Us = torch.load(bootstrap_dir / bootstrap_Us_fns[i], weights_only=True)

                msm = TransferMatrixNZGME(Us[:2])
                _, integrals_msm = msm.integrate(
                    NSTEPS_RELAX, initial_dist=initial_dist, verbose=False
                )
                msm_bootstrap_relax_buffer[i] = integrals_msm[:, -1].cpu().numpy()

                gme = TransferMatrixNZGME(Us[: N_LAGS_TRUNC + 2])
                _, integrals_gme = gme.integrate(
                    NSTEPS_RELAX, initial_dist=initial_dist, verbose=False
                )
                gme_bootstrap_relax_buffer[i] = integrals_gme[:, -1].cpu().numpy()

            relax_msm_lower, relax_msm_upper = get_bootstrap_curve_CI(
                msm_bootstrap_relax_buffer, CI_PCT
            )
            relax_gme_lower, relax_gme_upper = get_bootstrap_curve_CI(
                gme_bootstrap_relax_buffer, CI_PCT
            )

            relax_curves[expt + "_msm_lower"] = relax_msm_lower.tolist()
            relax_curves[expt + "_msm_upper"] = relax_msm_upper.tolist()
            relax_curves[expt + "_gme_lower"] = relax_gme_lower.tolist()
            relax_curves[expt + "_gme_upper"] = relax_gme_upper.tolist()
            del (
                bootstrap_Us_fns,
                msm_bootstrap_relax_buffer,
                gme_bootstrap_relax_buffer,
                relax_msm_lower,
                relax_msm_upper,
                relax_gme_lower,
                relax_gme_upper,
            )

            Us = torch.load(RES_DIR / expt / MAIN_RES_US_FN, weights_only=True)

            msm = TransferMatrixNZGME(Us[:2])
            _, integrals_msm = msm.integrate(
                NSTEPS_RELAX, initial_dist=initial_dist, verbose=False
            )
            relax_curves[expt + "_msm_means"] = (
                integrals_msm[:, -1].cpu().numpy().tolist()
            )

            gme = TransferMatrixNZGME(Us[: N_LAGS_TRUNC + 2])
            _, integrals_gme = gme.integrate(
                NSTEPS_RELAX, initial_dist=initial_dist, verbose=False
            )
            relax_curves[expt + "_gme_means"] = (
                integrals_gme[:, -1].cpu().numpy().tolist()
            )

            del Us, msm, gme, integrals_msm, integrals_gme

        with open(RES_DIR / RELAXATION_FN, "w") as f:
            json.dump(relax_curves, f)

    ### IRREVERSIBILITIES ###
    if IRREVERSIBILITY_FN in os.listdir(RES_DIR):
        # Ethiopian People's Revolutionary Democratic Front? D:
        epr_df = pd.read_csv(RES_DIR / IRREVERSIBILITY_FN, index_col=0)
    else:
        epr_df = pd.read_csv(RES_DIR / ATPASE_RATES_FN, index_col=0)
        new_cols = [
            "MSM_EPR_mean (kB / s)",
            "MSM_EPR_lower (kB / s)",
            "MSM_EPR_upper (kB / s)",
            "GME_EPR_mean (kB / s)",
            "GME_EPR_lower (kB / s)",
            "GME_EPR_upper (kB / s)",
        ]
        epr_df[new_cols] = np.nan
        epr_df[new_cols] = epr_df[new_cols].astype(float)

        for expt in EXPTS:
            bootstrap_dir = RES_DIR / expt / BOOTSTRAP_DIR
            bootstrap_Gs_fns = _get_bootstrap_fns(BOOTSTRAP_GS_FN, bootstrap_dir)
            bootstrap_Us_fns = _get_bootstrap_fns(BOOTSTRAP_US_FN, bootstrap_dir)
            assert len(bootstrap_Us_fns) == len(bootstrap_Gs_fns), (
                "unequal NZ and TCL bootstraps"
            )
            print(
                f"{expt.replace('_', ' ')} CFTR: {len(bootstrap_Gs_fns)} bootstraps detected."
            )
            epr_buffer = np.empty((len(bootstrap_Gs_fns), 2), dtype=np.float64)

            for i in tqdm(range(len(bootstrap_Gs_fns)), desc="Bootstraps", leave=False):
                U = torch.load(bootstrap_dir / bootstrap_Us_fns[i], weights_only=True)[
                    N_LAGS_TRUNC
                ]
                msm = MarkovChain(U, DT * N_LAGS_TRUNC, device="cpu")
                epr_buffer[i, 0] = msm.get_2nd_local_epr().item()

                Gs = torch.load(bootstrap_dir / bootstrap_Gs_fns[i], weights_only=True)
                assert Gs.shape[0] >= 3, "Gs should have lags 0, 1, and 2."
                gme = MarkovChain(Gs, DT * round(N_LAGS_TRUNC / 2), device="cpu")
                epr_buffer[i, 1] = gme.get_3rd_local_epr().item()

            epr_lower, epr_upper = get_bootstrap_curve_CI(epr_buffer, CI_PCT)
            epr_df.at[expt, "MSM_EPR_lower (kB / s)"] = float(epr_lower[0])
            epr_df.at[expt, "MSM_EPR_upper (kB / s)"] = float(epr_upper[0])
            epr_df.at[expt, "GME_EPR_lower (kB / s)"] = float(epr_lower[1])
            epr_df.at[expt, "GME_EPR_upper (kB / s)"] = float(epr_upper[1])
            del epr_lower, epr_upper, epr_buffer

            U = torch.load(RES_DIR / expt / MAIN_RES_US_FN, weights_only=True)[
                N_LAGS_TRUNC
            ]
            msm = MarkovChain(U, DT * N_LAGS_TRUNC, device="cpu")
            epr_df.at[expt, "MSM_EPR_mean (kB / s)"] = float(
                msm.get_2nd_local_epr().item()
            )

            Gs = torch.load(RES_DIR / expt / MAIN_RES_GS_FN, weights_only=True)
            assert Gs.shape[0] >= 3, "Gs should have lags 0, 1, and 2."
            gme = MarkovChain(Gs, DT * round(N_LAGS_TRUNC / 2), device="cpu")
            epr_df.at[expt, "GME_EPR_mean (kB / s)"] = float(
                gme.get_3rd_local_epr().item()
            )

        epr_df.to_csv(RES_DIR / IRREVERSIBILITY_FN)

    ### PLOTTING ###
    ### plot setup ###
    fig = plt.figure(figsize=(6, 7))
    outer = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1.5, 1.75, 1.5, 1.5])

    # top two rows: merged left panel, with row-specific right panels
    gs_r12 = outer[:2].subgridspec(
        2, 4, width_ratios=[3, 2, 2, 2], hspace=0.5, wspace=0.2
    )
    ax_r1_0 = fig.add_subplot(gs_r12[:, 0])
    ax_r2_0 = ax_r1_0

    # upper-right panel split vertically into 2 equal axes with shared x
    gs_r1_12_split = gs_r12[0, 1:].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
    ax_r1_1 = fig.add_subplot(gs_r1_12_split[0, 0])
    ax_r1_2 = fig.add_subplot(gs_r1_12_split[1, 0], sharex=ax_r1_1)
    ax_r1_1.tick_params(labelbottom=False)

    # middle row
    gs_r2_123 = gs_r12[1, 1:].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.75)
    ax_r2_1 = fig.add_subplot(gs_r2_123[0, 0])
    ax_r2_2 = fig.add_subplot(gs_r2_123[0, 1])
    ax_r2_3 = fig.add_subplot(gs_r2_123[0, 2])

    # ground row
    gs_r3 = outer[2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    gs_r3_01_split = gs_r3[0, 0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
    ax_r3_0 = fig.add_subplot(gs_r3_01_split[0, 0])
    ax_r3_1 = fig.add_subplot(gs_r3_01_split[1, 0])
    ax_r3_2 = fig.add_subplot(gs_r3[0, 1])
    ax_r3_0.tick_params(labelbottom=False)

    # basement row
    gs_r4 = outer[3].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.125)
    ax_r4_0 = fig.add_subplot(gs_r4[0, 0])
    ax_r4_1 = fig.add_subplot(gs_r4[0, 1], sharey=ax_r4_0)
    ax_r4_2 = fig.add_subplot(gs_r4[0, 2], sharey=ax_r4_0)
    ax_r4_1.tick_params(labelleft=False)
    ax_r4_2.tick_params(labelleft=False)

    # flat list of all 12 axes
    axes = [
        ax_r1_0,
        ax_r1_1,
        ax_r1_2,
        ax_r2_0,
        ax_r2_1,
        ax_r2_2,
        ax_r2_3,
        ax_r3_0,
        ax_r3_1,
        ax_r3_2,
        ax_r4_0,
        ax_r4_1,
        ax_r4_2,
    ]
    for a in axes:
        a.tick_params(axis="both", which="both", labelsize=8)
        for axis in ["right", "top", "left", "bottom"]:
            a.spines[axis].set_linewidth(0.5)
    ax_r1_0.set_axis_off()
    fig.subplots_adjust(hspace=0.5)

    for ax in [ax_r2_1, ax_r2_2, ax_r2_3]:
        pos = ax.get_position()
        ax.set_position((pos.x0, pos.y0 + 0.005, pos.width, pos.height))

    ### panel (a) ###
    ax_r1_0.text(
        -0.25,
        1.007,
        "High-FRET,\nNBD-dimeric",
        transform=ax_r1_0.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax_r1_0.text(
        0.6,
        -0.065,
        "Low-FRET,\nNBD-monomeric",
        transform=ax_r1_0.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ### panel (b) ###
    nsteps_example = 1000
    plot_times = np.arange(nsteps_example + 1) * DT

    df_wt_example = pd.read_csv(DATA_DIR / "WT_3mM_ATP/expt_1.csv", index_col=False)
    wt_FRET_example = np.array(df_wt_example["Fret_223"].iloc[: nsteps_example + 1])
    wt_ref_example = np.array(df_wt_example["Ref_223"].iloc[: nsteps_example + 1])
    del df_wt_example

    df_mut_example = pd.read_csv(DATA_DIR / "G551D_3mM_ATP/expt_4.csv", index_col=False)
    mut_FRET_example = np.array(df_mut_example["Fret_329"].iloc[: nsteps_example + 1])
    mut_ref_example = np.array(df_mut_example["Ref_329"].iloc[: nsteps_example + 1])
    del df_mut_example

    ax_r1_1.plot(plot_times, wt_FRET_example, c="k", alpha=0.25)
    ax_r1_1.plot(plot_times, wt_ref_example, c="k")
    ax_r1_1.set_xlim(plot_times[0], plot_times[-1])
    ax_r1_1.set_ylim(0.0, 0.7)
    ax_r1_1.set_yticks(0.25 * np.arange(3))
    ax_r1_1.text(
        0.99,
        0.03,
        "WT",
        transform=ax_r1_1.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )

    ax_r1_2.plot(plot_times, mut_FRET_example, c="k", alpha=0.25)
    ax_r1_2.plot(plot_times, mut_ref_example, c="k")
    ax_r1_2.set_xlim(plot_times[0], plot_times[-1])
    ax_r1_2.set_ylim(0.0, 0.7)
    ax_r1_2.set_yticks(0.25 * np.arange(3))
    ax_r1_2.text(
        0.99,
        0.03,
        "Mutant (G551D)",
        transform=ax_r1_2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax_r1_2.set_xlabel("Time (s)", fontsize=8)
    r1_trace_positions = [ax.get_position() for ax in [ax_r1_1, ax_r1_2]]
    fig.text(
        min(pos.x0 for pos in r1_trace_positions) - 0.07,
        (
            min(pos.y0 for pos in r1_trace_positions)
            + max(pos.y1 for pos in r1_trace_positions)
        )
        / 2,
        "FRET",
        rotation="vertical",
        ha="center",
        va="center",
        fontsize=8,
    )

    ### panel (c) ###
    bar_width = 0.35
    turnover_color = "k"
    turnover_group_x = np.array([0, 1])
    turnover_bar_x = np.array(
        [
            turnover_group_x[0] - bar_width / 2,
            turnover_group_x[0] + bar_width / 2,
            turnover_group_x[1] - bar_width / 2,
            turnover_group_x[1] + bar_width / 2,
        ]
    )
    turnover_means = [
        epr_df.loc[expt]["ATP turnover mean (1 / min)"] / 60 for expt in MUT_EXPTS
    ]
    turnover_errs = [
        norm_CI_ppf_one_sided * epr_df.loc[expt]["ATP turnover SE (1 / min)"] / 60
        for expt in MUT_EXPTS
    ]
    turnover_bars = ax_r2_1.bar(
        turnover_bar_x,
        turnover_means,
        yerr=turnover_errs,
        width=bar_width,
        facecolor=to_rgba(turnover_color, 0.25),
        edgecolor=turnover_color,
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 0.5, "capthick": 0.5},
    )
    ax_r2_1.bar(
        turnover_bar_x[[1, 3]],
        np.array(turnover_means)[[1, 3]],
        width=bar_width,
        facecolor="none",
        edgecolor=turnover_color,
        linewidth=0.5,
        hatch="////",
        zorder=turnover_bars[0].get_zorder() + 0.1,
    )
    ax_r2_1.set_xticks(turnover_group_x)
    ax_r2_1.set_xticklabels(["G551D", "L927P"])
    ax_r2_1.set_ylabel(r"$k_{\mathrm{ATPase}}$ (Hz)", fontsize=8)
    ax_r2_1.set_ylim(0, 0.2)

    msm_epr_color = "C1"
    msm_epr_means = np.array(
        [1e3 * epr_df.loc[expt]["MSM_EPR_mean (kB / s)"] for expt in MUT_EXPTS]
    )
    msm_epr_lower = np.array(
        [1e3 * epr_df.loc[expt]["MSM_EPR_lower (kB / s)"] for expt in MUT_EXPTS]
    )
    msm_epr_upper = np.array(
        [1e3 * epr_df.loc[expt]["MSM_EPR_upper (kB / s)"] for expt in MUT_EXPTS]
    )
    msm_epr_bars = ax_r2_2.bar(
        turnover_bar_x,
        msm_epr_means,
        yerr=np.vstack([msm_epr_means - msm_epr_lower, msm_epr_upper - msm_epr_means]),
        width=bar_width,
        facecolor=to_rgba(msm_epr_color, 0.25),
        edgecolor=msm_epr_color,
        linewidth=0.5,
        capsize=3,
        error_kw={"ecolor": msm_epr_color, "elinewidth": 0.5, "capthick": 0.5},
    )
    ax_r2_2.bar(
        turnover_bar_x[[1, 3]],
        msm_epr_means[[1, 3]],
        width=bar_width,
        facecolor="none",
        edgecolor=msm_epr_color,
        linewidth=0.5,
        hatch="////",
        zorder=msm_epr_bars[0].get_zorder() + 0.1,
    )
    ax_r2_2.set_xticks(turnover_group_x)
    ax_r2_2.set_xticklabels(["G551D", "L927P"])
    ax_r2_2.set_yticks([0.00, 0.03, 0.06, 0.09])
    ax_r2_2.set_ylim(0.00, 0.09)
    ax_r2_2.set_ylabel(r"$\hat{\sigma}_{\mathrm{MSM}}$ (m$k_B$/s)", fontsize=8)

    gme_epr_color = "C0"
    gme_epr_means = np.array(
        [1e3 * epr_df.loc[expt]["GME_EPR_mean (kB / s)"] for expt in MUT_EXPTS]
    )
    gme_epr_lower = np.array(
        [1e3 * epr_df.loc[expt]["GME_EPR_lower (kB / s)"] for expt in MUT_EXPTS]
    )
    gme_epr_upper = np.array(
        [1e3 * epr_df.loc[expt]["GME_EPR_upper (kB / s)"] for expt in MUT_EXPTS]
    )
    gme_epr_bars = ax_r2_3.bar(
        turnover_bar_x,
        gme_epr_means,
        yerr=np.vstack([gme_epr_means - gme_epr_lower, gme_epr_upper - gme_epr_means]),
        width=bar_width,
        facecolor=to_rgba(gme_epr_color, 0.25),
        edgecolor=gme_epr_color,
        linewidth=0.5,
        capsize=3,
        error_kw={"ecolor": gme_epr_color, "elinewidth": 0.5, "capthick": 0.5},
    )
    ax_r2_3.bar(
        turnover_bar_x[[1, 3]],
        gme_epr_means[[1, 3]],
        width=bar_width,
        facecolor="none",
        edgecolor=gme_epr_color,
        linewidth=0.5,
        hatch="////",
        zorder=gme_epr_bars[0].get_zorder() + 0.1,
    )
    ax_r2_3.set_xticks(turnover_group_x)
    ax_r2_3.set_xticklabels(["G551D", "L927P"])
    ax_r2_3.set_yticks(40 * np.arange(4))
    ax_r2_3.set_ylabel(
        r"$\hat{\sigma}_{\mathrm{GME}}$ (m$k_B$/s)", fontsize=8, labelpad=0
    )

    r2_positions = [ax.get_position() for ax in [ax_r2_1, ax_r2_2, ax_r2_3]]
    r2_xcenter = (
        min(pos.x0 for pos in r2_positions) + max(pos.x1 for pos in r2_positions)
    ) / 2
    r2_ybottom = min(pos.y0 for pos in r2_positions)
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="k", linewidth=0.5, label="Apo"),
            Patch(
                facecolor="white",
                edgecolor="k",
                linewidth=0.5,
                hatch="////",
                label="GLPG1837",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(r2_xcenter, r2_ybottom - 0.025),
        ncols=2,
        fontsize=8,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
    )

    ### panel (d) ###
    plot_Ks_lagtimes = np.arange(1, N_LAGS_TOTAL) * DT

    def _plot_kernels(
        ax: Axes, expt: str, linestyle: str, label: str | None = None
    ) -> None:
        Us_fn = RES_DIR / expt / MAIN_RES_US_FN
        Us = torch.load(Us_fn, weights_only=True)
        Ks = get_Ks_from_Us(Us, verbose=False)
        ax.plot(
            plot_Ks_lagtimes,
            Ks.reshape(Ks.shape[0], -1)[1:].cpu(),
            color="C0",
            linewidth=0.5,
            linestyle=linestyle,
            label=label,
        )

    _plot_kernels(ax_r3_0, "WT_3mM_ATP", "-")
    _plot_kernels(ax_r3_0, "G551D_3mM_ATP", ":")
    _plot_kernels(ax_r3_0, "L927P_3mM_ATP", "--")
    ax_r3_0.axvline(
        x=N_LAGS_TRUNC * DT, linestyle="--", label="Cutoff", color="k", linewidth=0.5
    )
    ax_r3_0.set_xlim(plot_Ks_lagtimes[0], plot_Ks_lagtimes[-1])
    ax_r3_0.set_ylim(-1e-2, 1e-2)
    ax_r3_0.text(
        0.96,
        0.1,
        "Apo",
        transform=ax_r3_0.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax_r3_0.legend(
        handles=[
            Line2D([0], [0], color="k", linestyle="--", linewidth=0.5, label="Cutoff"),
            Line2D([0], [0], color="C0", linestyle="-", linewidth=0.5, label="WT"),
        ],
        loc="upper right",
        ncols=2,
        fontsize=6,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
    )

    _plot_kernels(ax_r3_1, "WT_3mM_ATP_10uM_GLPG1837", "-")
    _plot_kernels(ax_r3_1, "G551D_3mM_ATP_10uM_GLPG1837", ":")
    _plot_kernels(ax_r3_1, "L927P_3mM_ATP_10uM_GLPG1837", "--")
    ax_r3_1.axvline(
        x=N_LAGS_TRUNC * DT, linestyle="--", label="Cutoff", color="k", linewidth=0.5
    )
    ax_r3_1.set_xlim(plot_Ks_lagtimes[0], plot_Ks_lagtimes[-1])
    ax_r3_1.set_ylim(-1e-2, 1e-2)
    ax_r3_1.text(
        0.96,
        0.1,
        r"GLPG1837",
        transform=ax_r3_1.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax_r3_1.set_xlabel(r"Delay n$\tau$ (s)", fontsize=8)
    ax_r3_1.set_ylabel(r"$\hat{K}_{ij}^{(n)}$", fontsize=8, labelpad=0)
    ax_r3_1.yaxis.set_label_coords(-0.2, 0.975)
    ax_r3_1.set_xticks(0.1 + 0.6 * np.arange(7))
    ax_r3_1.legend(
        handles=[
            Line2D([0], [0], color="C0", linestyle=":", linewidth=0.5, label="G551D"),
            Line2D([0], [0], color="C0", linestyle="--", linewidth=0.5, label="L927P"),
        ],
        loc="upper right",
        ncols=2,
        fontsize=6,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
    )

    ### panel (e) ###
    ax_r3_2.plot(solventx_curves["expt_times"], solventx_curves["expt_means"], c="k")
    ax_r3_2.fill_between(
        solventx_curves["expt_times"],
        solventx_curves["expt_lower"],
        solventx_curves["expt_upper"],
        color="k",
        alpha=0.25,
    )
    ax_r3_2.plot(solventx_curves["model_times"], solventx_curves["msm_means"], c="C1")
    ax_r3_2.fill_between(
        solventx_curves["model_times"],
        solventx_curves["msm_lower"],
        solventx_curves["msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r3_2.plot(solventx_curves["model_times"], solventx_curves["gme_means"], c="C0")
    ax_r3_2.fill_between(
        solventx_curves["model_times"],
        solventx_curves["gme_lower"],
        solventx_curves["gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r3_2.set_xlim(0.0, solventx_curves["expt_times"][-1])
    ax_r3_2.set_ylim(0.56, 0.84)
    ax_r3_2.set_xlabel("Time from solvent exchange (s)", fontsize=8)
    ax_r3_2.set_ylabel("Dimer fraction", fontsize=8, labelpad=0)
    ax_r3_2.yaxis.set_label_coords(-0.17, 0.42)
    ax_r3_2.legend(
        handles=[
            (
                Patch(facecolor="k", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="k", linestyle="-", linewidth=0.5),
            ),
            (
                Patch(facecolor="C1", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="C1", linestyle="-", linewidth=0.5),
            ),
            (
                Patch(facecolor="C0", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="C0", linestyle="-", linewidth=0.5),
            ),
        ],
        labels=["Experiment (0.95 CI)", "MSM (0.95 CI)", "GME (0.95 CI)"],
        loc="upper left",
        fontsize=6,
        frameon=False,
        handlelength=1.0,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )

    ### panel (f) ###
    relax_times = np.arange(NSTEPS_RELAX + 1) * DT

    # WT models
    ax_r4_0.plot(relax_times, relax_curves["WT_3mM_ATP_msm_means"], c="C1")
    ax_r4_0.fill_between(
        relax_times,
        relax_curves["WT_3mM_ATP_msm_lower"],
        relax_curves["WT_3mM_ATP_msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r4_0.plot(relax_times, relax_curves["WT_3mM_ATP_gme_means"], c="C0")
    ax_r4_0.fill_between(
        relax_times,
        relax_curves["WT_3mM_ATP_gme_lower"],
        relax_curves["WT_3mM_ATP_gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r4_0.plot(
        relax_times,
        relax_curves["WT_3mM_ATP_10uM_GLPG1837_msm_means"],
        c="C1",
        linestyle="-.",
    )
    ax_r4_0.fill_between(
        relax_times,
        relax_curves["WT_3mM_ATP_10uM_GLPG1837_msm_lower"],
        relax_curves["WT_3mM_ATP_10uM_GLPG1837_msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r4_0.plot(
        relax_times,
        relax_curves["WT_3mM_ATP_10uM_GLPG1837_gme_means"],
        c="C0",
        linestyle="-.",
    )
    ax_r4_0.fill_between(
        relax_times,
        relax_curves["WT_3mM_ATP_10uM_GLPG1837_gme_lower"],
        relax_curves["WT_3mM_ATP_10uM_GLPG1837_gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r4_0.set_xlim(relax_times[0], relax_times[-1])
    ax_r4_0.set_ylim(0.0, 1.0)
    ax_r4_0.set_xlabel("Time (s)", fontsize=8)
    ax_r4_0.yaxis.set_label_coords(-0.275, 0.0)
    ax_r4_0.set_ylabel("Dimer fraction", fontsize=8, labelpad=0)
    ax_r4_0.text(
        0.98,
        0.02,
        "WT",
        transform=ax_r4_0.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )

    # G551D models
    ax_r4_1.plot(relax_times, relax_curves["G551D_3mM_ATP_msm_means"], c="C1")
    ax_r4_1.fill_between(
        relax_times,
        relax_curves["G551D_3mM_ATP_msm_lower"],
        relax_curves["G551D_3mM_ATP_msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r4_1.plot(relax_times, relax_curves["G551D_3mM_ATP_gme_means"], c="C0")
    ax_r4_1.fill_between(
        relax_times,
        relax_curves["G551D_3mM_ATP_gme_lower"],
        relax_curves["G551D_3mM_ATP_gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r4_1.plot(
        relax_times,
        relax_curves["G551D_3mM_ATP_10uM_GLPG1837_msm_means"],
        c="C1",
        linestyle="-.",
    )
    ax_r4_1.fill_between(
        relax_times,
        relax_curves["G551D_3mM_ATP_10uM_GLPG1837_msm_lower"],
        relax_curves["G551D_3mM_ATP_10uM_GLPG1837_msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r4_1.plot(
        relax_times,
        relax_curves["G551D_3mM_ATP_10uM_GLPG1837_gme_means"],
        c="C0",
        linestyle="-.",
    )
    ax_r4_1.fill_between(
        relax_times,
        relax_curves["G551D_3mM_ATP_10uM_GLPG1837_gme_lower"],
        relax_curves["G551D_3mM_ATP_10uM_GLPG1837_gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r4_1.set_xlim(relax_times[0], relax_times[-1])
    ax_r4_1.set_xlabel("Time (s)", fontsize=8)
    ax_r4_1.text(
        0.98,
        0.02,
        "G551D",
        transform=ax_r4_1.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax_r4_1.legend(
        handles=[
            (
                Patch(facecolor="C1", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="C1", linestyle="-", linewidth=0.5),
            ),
            (
                Patch(facecolor="C0", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="C0", linestyle="-", linewidth=0.5),
            ),
            (
                Patch(facecolor="C1", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="C1", linestyle="-.", linewidth=0.5),
            ),
            (
                Patch(facecolor="C0", edgecolor="none", alpha=0.25),
                Line2D([0], [0], color="C0", linestyle="-.", linewidth=0.5),
            ),
        ],
        labels=[
            "Apo MSM (0.95 CI)",
            "Apo GME (0.95 CI)",
            "GLPG1837 MSM (0.95 CI)",
            "GLPG1837 GME (0.95 CI)",
        ],
        loc="upper center",
        fontsize=6,
        frameon=False,
        handlelength=1.0,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )

    # L927P models
    ax_r4_2.plot(relax_times, relax_curves["L927P_3mM_ATP_msm_means"], c="C1")
    ax_r4_2.fill_between(
        relax_times,
        relax_curves["L927P_3mM_ATP_msm_lower"],
        relax_curves["L927P_3mM_ATP_msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r4_2.plot(relax_times, relax_curves["L927P_3mM_ATP_gme_means"], c="C0")
    ax_r4_2.fill_between(
        relax_times,
        relax_curves["L927P_3mM_ATP_gme_lower"],
        relax_curves["L927P_3mM_ATP_gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r4_2.plot(
        relax_times,
        relax_curves["L927P_3mM_ATP_10uM_GLPG1837_msm_means"],
        c="C1",
        linestyle="-.",
    )
    ax_r4_2.fill_between(
        relax_times,
        relax_curves["L927P_3mM_ATP_10uM_GLPG1837_msm_lower"],
        relax_curves["L927P_3mM_ATP_10uM_GLPG1837_msm_upper"],
        color="C1",
        alpha=0.25,
    )
    ax_r4_2.plot(
        relax_times,
        relax_curves["L927P_3mM_ATP_10uM_GLPG1837_gme_means"],
        c="C0",
        linestyle="-.",
    )
    ax_r4_2.fill_between(
        relax_times,
        relax_curves["L927P_3mM_ATP_10uM_GLPG1837_gme_lower"],
        relax_curves["L927P_3mM_ATP_10uM_GLPG1837_gme_upper"],
        color="C0",
        alpha=0.25,
    )
    ax_r4_2.set_xlim(relax_times[0], relax_times[-1])
    ax_r4_2.set_xlabel("Time (s)", fontsize=8)
    ax_r4_2.text(
        0.98,
        0.02,
        "L927P",
        transform=ax_r4_2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )

    ### annotations ###
    ax_r1_0.text(
        -0.42, 1.01, "(a)", transform=ax_r1_0.transAxes, ha="left", va="top", fontsize=8
    )
    ax_r1_1.text(
        -0.17, 1.07, "(b)", transform=ax_r1_1.transAxes, ha="left", va="top", fontsize=8
    )
    ax_r2_1.text(
        -0.77, 1.07, "(c)", transform=ax_r2_1.transAxes, ha="left", va="top", fontsize=8
    )
    ax_r3_0.text(
        -0.28, 1.08, "(d)", transform=ax_r3_0.transAxes, ha="left", va="top", fontsize=8
    )
    ax_r3_2.text(
        -0.24, 1.04, "(e)", transform=ax_r3_2.transAxes, ha="left", va="top", fontsize=8
    )
    ax_r4_0.text(
        -0.39, 1.1, "(f)", transform=ax_r4_0.transAxes, ha="left", va="top", fontsize=8
    )

    # Equalize the gaps from each preceding panel to the irreversibility ylabel.
    fig.draw_without_rendering()
    inv_fig = fig.transFigure.inverted()
    msm_label_bbox = ax_r2_2.yaxis.label.get_window_extent().transformed(inv_fig)
    gme_label_bbox = ax_r2_3.yaxis.label.get_window_extent().transformed(inv_fig)
    msm_pos = ax_r2_2.get_position()
    msm_label_gap = msm_label_bbox.x0 - ax_r2_1.get_position().x1
    gme_label_gap = gme_label_bbox.x0 - msm_pos.x1
    msm_shift = (gme_label_gap - msm_label_gap) / 2
    ax_r2_2.set_position(
        (msm_pos.x0 + msm_shift, msm_pos.y0, msm_pos.width, msm_pos.height)
    )

    # Save directly to avoid pyplot's extra Agg redraw, which requires dvipng.
    fig.savefig(RES_DIR / FIG_FN, bbox_inches="tight")
