"""scripts/postprocess/BMF1.py

Analyze TCL-GME-DT models of bovine mitochondrial F1 ATPase trajectories from
R. Kobayashi, H. Ueno, C.B. Li, and H. Noji,
Rotary catalysis of bovine mitochondrial F1-ATPase studied by single-molecule experiments,
Proc. Natl. Acad. Sci. U.S.A. 117, 1447 (2021).
The trajectory is coarse-grained by crude assignment to histogram peaks.
This script takes about 30 s to run.
"""

### IMPORTS ###
import fnmatch
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

from gmex.markov_process import MarkovChain
from gmex.utils import (
    get_bootstrap_curve_CI,
    get_data_dir,
    get_results_dir,
    load_times_macrostates,
)

### CONFIGURATIONS ###
### shared ###
DATA_DIR = get_data_dir() / "BMF1"
RES_DIR = get_results_dir() / "BMF1"
SEED = 42
CI_PCT = 0.95
DWELL_FN = "BMF1-dwell.json"
FIG_FN = "BMF1-fig-no-ribbon.pdf"
DELTA = 290  # for some reason all the angles are shifted 290 degs relative to the paper

### 1uM ATPgS model ###
DATA_DIR_ATPgS = DATA_DIR / "ATPgS_1uM"
RAW_FN_ATPgS = DATA_DIR_ATPgS / "raw.txt"
RES_DIR_ATPgS = RES_DIR / "ATPgS_1uM"
MAIN_RES_GS_FN_ATPgS = RES_DIR_ATPgS / "BMF1-ATPgS-1uM-M6-L16-TCL-Gs.pt"
BOOTSTRAP_DIR_ATPgS = RES_DIR_ATPgS / "bootstraps"
BOOTSTRAP_GS_FN_ATPgS = "BMF1-ATPgS-1uM-M6-L2-TCL-Gs-BS*.pt"
N_LAGS_TRUNC_ATPgS = 2

### 3mM ATP model ###
DATA_DIR_ATP = DATA_DIR / "ATP_3mM"
RAW_FN_ATP = DATA_DIR_ATP / "raw.txt"
RES_DIR_ATP = RES_DIR / "ATP_3mM"
MAIN_RES_GS_FN_ATP = RES_DIR_ATP / "BMF1-ATP-3mM-M6-L16-TCL-Gs.pt"
BOOTSTRAP_DIR_ATP = RES_DIR_ATP / "bootstraps"
BOOTSTRAP_GS_FN_ATP = "BMF1-ATP-3mM-M6-L5-TCL-Gs-BS*.pt"
N_LAGS_TRUNC_ATP = 5


### HELPERS ###
def _get_even_odd_mean_dwells(
    Gs: torch.Tensor, dt: float, cutoff: int, max_timestep: int = 1000
) -> tuple[float, float, float, float]:
    """Get even- and odd-peak mean dwell times."""
    gme = MarkovChain(Gs[: cutoff + 1], dt, device=device)
    msm = MarkovChain(Gs[1], dt, device=device)
    gme_dwell_probs = gme.dwell_probabilities().to(device)
    msm_dwell_probs = msm.dwell_probabilities().to(device)
    gme_dwell_means = (gme_dwell_probs * torch.arange(max_timestep, device=device)).sum(
        dim=1
    ) * dt
    msm_dwell_means = (msm_dwell_probs * torch.arange(max_timestep, device=device)).sum(
        dim=1
    ) * dt
    return (
        gme_dwell_means[::2].mean().cpu().item(),
        msm_dwell_means[::2].mean().cpu().item(),
        gme_dwell_means[1::2].mean().cpu().item(),
        msm_dwell_means[1::2].mean().cpu().item(),
    )


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
    L.seed_everything(SEED)
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

    ### LOAD 1uM ATPgS EXPERIMENTAL DATA ###
    dt_ATPgS, _, _, states_ATPgS, _ = load_times_macrostates(
        "BMF1/ATPgS_1uM", device="cpu"
    )
    df_ATPgS = pd.read_csv(RAW_FN_ATPgS, sep="\t")
    df_ATPgS["Time"] = df_ATPgS["Frame"] * dt_ATPgS
    df_ATPgS["State"] = states_ATPgS
    df_ATPgS["Angle"] = (df_ATPgS["Angle"] + DELTA) % 360

    ### LOAD 3mM ATP EXPERIMENTAL DATA ###
    dt_ATP, _, _, states_ATP, _ = load_times_macrostates("BMF1/ATP_3mM", device="cpu")
    df_ATP = pd.read_csv(RAW_FN_ATP, sep="\t")
    df_ATP["Time"] = df_ATP["Frame"] * dt_ATP
    df_ATP["State"] = states_ATP
    df_ATP["Angle"] = (df_ATP["Angle"] + DELTA) % 360

    ### DWELL TIMES ###
    norm_CI_ppf_one_sided = -norm.ppf(0.5 - CI_PCT / 2.0).item()
    dwells = {
        "ATPgS_cat_cpd": 12.4,
        "ATPgS_cat_cpd_error": 0.5 * norm_CI_ppf_one_sided,
        "ATPgS_bind_cpd": 28.4,
        "ATPgS_bind_cpd_error": 1.0 * norm_CI_ppf_one_sided,
    }

    ### 1uM ATPgS mean dwell times ###
    Gs_ATPgS = torch.load(MAIN_RES_GS_FN_ATPgS, weights_only=True)
    dwells_ATPgS_mean = _get_even_odd_mean_dwells(
        Gs_ATPgS, dt_ATPgS, N_LAGS_TRUNC_ATPgS
    )
    dwells["ATPgS_cat_gme"] = dwells_ATPgS_mean[0]
    dwells["ATPgS_cat_msm"] = dwells_ATPgS_mean[1]
    dwells["ATPgS_bind_gme"] = dwells_ATPgS_mean[2]
    dwells["ATPgS_bind_msm"] = dwells_ATPgS_mean[3]

    ### 3mM ATP dwell times ###
    Gs_ATP = torch.load(MAIN_RES_GS_FN_ATP, weights_only=True)
    dwells_ATP_mean = _get_even_odd_mean_dwells(Gs_ATP, dt_ATP, N_LAGS_TRUNC_ATP)
    dwells["ATP_cat_gme"] = dwells_ATP_mean[0]
    dwells["ATP_cat_msm"] = dwells_ATP_mean[1]
    dwells["ATP_short_gme"] = dwells_ATP_mean[2]
    dwells["ATP_short_msm"] = dwells_ATP_mean[3]

    ### 1uM ATPgS bootstrap confidence intervals ###
    bootstrap_fns_ATPgS = _get_bootstrap_fns(BOOTSTRAP_GS_FN_ATPgS, BOOTSTRAP_DIR_ATPgS)
    print(f"{len(bootstrap_fns_ATPgS)} 1uM ATPgS bootstraps detected")
    bootstrap_dwell_buffer_ATPgS = np.empty(
        (len(bootstrap_fns_ATPgS), 4), dtype=np.float64
    )
    for i in range(len(bootstrap_fns_ATPgS)):
        Gs_bootstrap_ATPgS = torch.load(bootstrap_fns_ATPgS[i], weights_only=True)
        bootstrap_dwell_buffer_ATPgS[i] = np.array(
            _get_even_odd_mean_dwells(Gs_bootstrap_ATPgS, dt_ATPgS, N_LAGS_TRUNC_ATPgS)
        )
    dwells_ATPgS_lower, dwells_ATPgS_upper = get_bootstrap_curve_CI(
        bootstrap_dwell_buffer_ATPgS, CI_PCT
    )
    dwells["ATPgS_cat_gme_lower"] = dwells["ATPgS_cat_gme"] - dwells_ATPgS_lower[0]
    dwells["ATPgS_cat_msm_lower"] = dwells["ATPgS_cat_msm"] - dwells_ATPgS_lower[1]
    dwells["ATPgS_bind_gme_lower"] = dwells["ATPgS_bind_gme"] - dwells_ATPgS_lower[2]
    dwells["ATPgS_bind_msm_lower"] = dwells["ATPgS_bind_msm"] - dwells_ATPgS_lower[3]
    dwells["ATPgS_cat_gme_upper"] = dwells_ATPgS_upper[0] - dwells["ATPgS_cat_gme"]
    dwells["ATPgS_cat_msm_upper"] = dwells_ATPgS_upper[1] - dwells["ATPgS_cat_msm"]
    dwells["ATPgS_bind_gme_upper"] = dwells_ATPgS_upper[2] - dwells["ATPgS_bind_gme"]
    dwells["ATPgS_bind_msm_upper"] = dwells_ATPgS_upper[3] - dwells["ATPgS_bind_msm"]

    ### 3mM ATP bootstrap confidence intervals ###
    bootstrap_fns_ATP = _get_bootstrap_fns(BOOTSTRAP_GS_FN_ATP, BOOTSTRAP_DIR_ATP)
    print(f"{len(bootstrap_fns_ATP)} 3mM ATP bootstraps detected")
    bootstrap_dwell_buffer_ATP = np.empty((len(bootstrap_fns_ATP), 4), dtype=np.float64)
    for i in range(len(bootstrap_fns_ATP)):
        Gs_bootstrap_ATP = torch.load(bootstrap_fns_ATP[i], weights_only=True)
        bootstrap_dwell_buffer_ATP[i] = np.array(
            _get_even_odd_mean_dwells(Gs_bootstrap_ATP, dt_ATP, N_LAGS_TRUNC_ATP)
        )
    dwells_ATP_lower, dwells_ATP_upper = get_bootstrap_curve_CI(
        bootstrap_dwell_buffer_ATP, CI_PCT
    )
    dwells["ATP_cat_gme_lower"] = dwells["ATP_cat_gme"] - dwells_ATP_lower[0]
    dwells["ATP_cat_msm_lower"] = dwells["ATP_cat_msm"] - dwells_ATP_lower[1]
    dwells["ATP_short_gme_lower"] = dwells["ATP_short_gme"] - dwells_ATP_lower[2]
    dwells["ATP_short_msm_lower"] = dwells["ATP_short_msm"] - dwells_ATP_lower[3]
    dwells["ATP_cat_gme_upper"] = dwells_ATP_upper[0] - dwells["ATP_cat_gme"]
    dwells["ATP_cat_msm_upper"] = dwells_ATP_upper[1] - dwells["ATP_cat_msm"]
    dwells["ATP_short_gme_upper"] = dwells_ATP_upper[2] - dwells["ATP_short_gme"]
    dwells["ATP_short_msm_upper"] = dwells_ATP_upper[3] - dwells["ATP_short_msm"]

    ### PLOTTING ###
    ### plot setup ###
    fig = plt.figure(figsize=(6, 6.5))
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 2.2])

    # row 1: text, ATPgS trace/histogram, ATP trace/histogram
    gs_r1 = outer[0].subgridspec(
        1, 5, width_ratios=[2.0, 0.85, 0.65, 0.85, 0.65], wspace=0.0
    )
    ax_r1_0 = fig.add_subplot(gs_r1[0, 0])
    ax_r1_1 = fig.add_subplot(gs_r1[0, 1])
    ax_r1_2 = fig.add_subplot(gs_r1[0, 2], sharey=ax_r1_1)
    ax_r1_3 = fig.add_subplot(gs_r1[0, 3], sharey=ax_r1_1)
    ax_r1_4 = fig.add_subplot(gs_r1[0, 4], sharey=ax_r1_1)

    # row 2: stacked propagators and dwell-time comparisons
    gs_r2 = outer[1].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.32)
    gs_r2_0 = gs_r2[0, 0].subgridspec(2, 1, hspace=0.22)
    gs_r2_dwell = gs_r2[0, 1].subgridspec(1, 2, wspace=0.48)
    ax_r2_0 = fig.add_subplot(gs_r2_0[0, 0])
    ax_r2_1 = fig.add_subplot(gs_r2_0[1, 0])
    ax_r2_2 = fig.add_subplot(gs_r2_dwell[0, 0])
    ax_r2_3 = fig.add_subplot(gs_r2_dwell[0, 1])

    axes = [
        ax_r1_0,
        ax_r1_1,
        ax_r1_2,
        ax_r1_3,
        ax_r1_4,
        ax_r2_0,
        ax_r2_1,
        ax_r2_2,
        ax_r2_3,
    ]
    for a in axes:
        a.tick_params(axis="both", which="both", labelsize=8)
        for axis in ["right", "top", "left", "bottom"]:
            a.spines[axis].set_linewidth(0.5)
    fig.subplots_adjust(left=0.09, top=0.9, hspace=0.5)

    r1_right_axes = [ax_r1_1, ax_r1_2, ax_r1_3, ax_r1_4]
    r1_right_positions = [ax.get_position() for ax in r1_right_axes]
    r1_legend_y = max(pos.y1 for pos in r1_right_positions) + 0.025
    r1_xlabel_y = -0.2
    r1_xlabel_fig_y = fig.transFigure.inverted().transform(
        ax_r1_1.transAxes.transform((0, r1_xlabel_y))
    )[1]
    rotor_y = ax_r1_0.transAxes.inverted().transform(
        fig.transFigure.transform((0, r1_legend_y))
    )[1]
    ring_y = ax_r1_0.transAxes.inverted().transform(
        fig.transFigure.transform((0, r1_xlabel_fig_y))
    )[1]
    ring_x_fig = fig.transFigure.inverted().transform(
        ax_r1_0.transAxes.transform((-0.3, ring_y))
    )[0]
    prop_left_extension = 0.015
    for ax in [ax_r2_0, ax_r2_1]:
        pos = ax.get_position()
        ax.set_position(
            (
                pos.x0 - prop_left_extension,
                pos.y0,
                pos.width + prop_left_extension,
                pos.height,
            )
        )
    prop_ylabel_x = ax_r2_0.transAxes.inverted().transform(
        fig.transFigure.transform((ring_x_fig, 0))
    )[0]

    ### panel (a) ###
    ax_r1_0.axis("off")
    rotor_text = ax_r1_0.text(
        0.5,
        rotor_y,
        r"Subunit $\gamma$",
        transform=ax_r1_0.transAxes,
        color="#B8860B",
        ha="right",
        va="center",
        fontsize=8,
    )

    ### panel (b) ###
    ATPgS_start_frame = 3354
    ATPgS_end_frame = 3474
    ATPgS_n_frames_plotted = ATPgS_end_frame - ATPgS_start_frame + 1
    ATPgS_plot_offset = (
        df_ATPgS["Angle"][ATPgS_start_frame] - df_ATPgS["Accum"][ATPgS_start_frame]
    )
    ax_r1_1.plot(
        dt_ATPgS * np.arange(ATPgS_n_frames_plotted),
        df_ATPgS["Accum"][ATPgS_start_frame - 1 : ATPgS_end_frame] + ATPgS_plot_offset,
        "k",
    )
    ax_r1_1.set_xlim(0, ATPgS_n_frames_plotted - 1)
    ax_r1_1.set_ylim(0, 360)
    ax_r1_1.set_xticks(np.arange(5) * 30)
    ax_r1_1.set_yticks(np.arange(4) * 120)
    ax_r1_1.set_xlabel("Time (ms)", fontsize=8)
    ax_r1_1.xaxis.set_label_coords(0.5, r1_xlabel_y)
    ax_r1_1.set_ylabel(r"Imaged angle (rad)", fontsize=8)
    ax_r1_1.text(
        0.9,
        0.05,
        r"ATP$\gamma$S",
        transform=ax_r1_1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    bin_edges = [i * 2.5 for i in range(145)]
    ax_r1_2.hist(
        df_ATPgS.loc[df_ATPgS["State"] % 2 == 0, "Angle"],
        bins=bin_edges,
        orientation="horizontal",
        color="C6",
    )
    ax_r1_2.hist(
        df_ATPgS.loc[df_ATPgS["State"] % 2 == 1, "Angle"],
        bins=bin_edges,
        orientation="horizontal",
        color="C5",
    )
    ax_r1_2.set_xlim(0, 175)
    ax_r1_2.set_xticks([50, 100])
    ax_r1_2.set_xlabel(r"Frames $N$", fontsize=8)
    ax_r1_2.xaxis.set_label_coords(0.5, r1_xlabel_y)

    ATP_start_frame = 493
    ATP_end_frame = 551
    ATP_n_frames_plotted = ATP_end_frame - ATP_start_frame + 1
    ATP_plot_offset = (
        df_ATP["Angle"][ATP_start_frame] - df_ATP["Accum"][ATP_start_frame]
    )
    ax_r1_3.plot(
        dt_ATP * np.arange(ATP_n_frames_plotted),
        df_ATP["Accum"][ATP_start_frame - 1 : ATP_end_frame] + ATP_plot_offset,
        "k",
    )
    ax_r1_3.set_xlim(0, 1.2)
    ax_r1_3.set_xticks(np.arange(5) * 0.3)
    ax_r1_3.set_xlabel("Time (ms)", fontsize=8)
    ax_r1_3.xaxis.set_label_coords(0.5, r1_xlabel_y)
    ax_r1_3.text(
        0.9,
        0.05,
        "ATP",
        transform=ax_r1_3.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax_r1_4.hist(
        df_ATP.loc[df_ATP["State"] % 2 == 0, "Angle"],
        bins=bin_edges,
        orientation="horizontal",
        color="C6",
    )
    ax_r1_4.hist(
        df_ATP.loc[df_ATP["State"] % 2 == 1, "Angle"],
        bins=bin_edges,
        orientation="horizontal",
        color="darkolivegreen",
    )
    ax_r1_4.set_xlim(0, 750)
    ax_r1_4.set_xticks([300, 600])
    ax_r1_4.set_xlabel(r"Frames $N$", fontsize=8)
    ax_r1_4.xaxis.set_label_coords(0.5, r1_xlabel_y)
    for ax in [ax_r1_2, ax_r1_3, ax_r1_4]:
        ax.tick_params(labelleft=False)
    for ax in [ax_r1_1, ax_r1_2, ax_r1_3, ax_r1_4]:
        ax.set_ylim(0, 360)
        ax.set_yticks(np.arange(4) * 120)
        ax.set_yticklabels(["0", r"$\frac{2\pi}{3}$", r"$\frac{4\pi}{3}$", r"$2\pi$"])

    fig.legend(
        handles=[
            Patch(facecolor="C6", edgecolor="none", label="Catalytic"),
            Patch(facecolor="C5", edgecolor="none", label="Binding"),
            Patch(facecolor="darkolivegreen", edgecolor="none", label="Short"),
        ],
        loc="center",
        bbox_to_anchor=(
            (
                min(pos.x0 for pos in r1_right_positions)
                + max(pos.x1 for pos in r1_right_positions)
            )
            / 2,
            r1_legend_y,
        ),
        ncol=3,
        fontsize=8,
        frameon=False,
        handlelength=0.75,
        columnspacing=1.0,
    )

    ### panel (c) ###
    lags_ATPgS = np.arange(1, Gs_ATPgS.shape[0]) * dt_ATPgS
    ax_r2_0.plot(
        lags_ATPgS,
        Gs_ATPgS.reshape(Gs_ATPgS.shape[0], -1)[1:].cpu(),
        linewidth=0.5,
        color="C0",
    )
    ax_r2_0.set_xlim(lags_ATPgS[0], lags_ATPgS[-1])
    ax_r2_0.set_ylim(0, 1)
    ax_r2_0.set_xticks(np.arange(6) * 3 + 1)
    ax_r2_0.set_yticks(np.arange(6) * 0.2)
    ax_r2_0.set_ylabel(r"$\hat{G}^{(n)}_{ij}$", fontsize=8)
    ax_r2_0.yaxis.set_label_coords(prop_ylabel_x, 0.5)
    ax_r2_0.yaxis.label.set_verticalalignment("top")
    cutoff_line = ax_r2_0.axvline(
        x=N_LAGS_TRUNC_ATPgS * dt_ATPgS,
        linestyle="--",
        label="Cutoff",
        color="k",
        linewidth=0.5,
    )
    ax_r2_0.text(
        0.95,
        0.5,
        r"ATP$\gamma$S",
        transform=ax_r2_0.transAxes,
        ha="right",
        va="center",
        fontsize=8,
    )
    lags_ATP = np.arange(1, Gs_ATP.shape[0]) * dt_ATP
    ax_r2_1.plot(
        lags_ATP,
        Gs_ATP.reshape(Gs_ATP.shape[0], -1)[1:].cpu(),
        linewidth=0.5,
        color="C0",
    )
    ax_r2_1.set_xlim(lags_ATP[0], lags_ATP[-1])
    ax_r2_1.set_ylim(0, 1)
    ax_r2_1.set_xticks(np.arange(0, 4) * 0.1 + 0.05)
    ax_r2_1.set_yticks(np.arange(6) * 0.2)
    ax_r2_1.set_xlabel(r"Lagtime $n\tau$ (ms)", fontsize=8)
    ax_r2_1.set_ylabel(r"$\hat{G}^{(n)}_{ij}$", fontsize=8)
    ax_r2_1.yaxis.set_label_coords(prop_ylabel_x, 0.5)
    ax_r2_1.yaxis.label.set_verticalalignment("top")
    ax_r2_1.axvline(
        x=N_LAGS_TRUNC_ATP * dt_ATP, linestyle="--", color="k", linewidth=0.5
    )
    ax_r2_1.text(
        0.95,
        0.5,
        "ATP",
        transform=ax_r2_1.transAxes,
        ha="right",
        va="center",
        fontsize=8,
    )

    ### panel (d) ###
    max_feasible_dwell_ATPgS = np.ptp(df_ATPgS["Time"]) / np.ptp(df_ATPgS["Revo"]) / 3.0
    ax_r2_2.errorbar(
        [dwells["ATPgS_cat_cpd"]],
        [dwells["ATPgS_bind_cpd"]],
        xerr=[dwells["ATPgS_cat_cpd_error"]],
        yerr=[dwells["ATPgS_bind_cpd_error"]],
        marker="o",
        markerfacecolor="none",
        capsize=3,
        color="C2",
        elinewidth=1.0,
        label="CPD (0.95 CI)",
    )
    ax_r2_2.errorbar(
        [dwells["ATPgS_cat_msm"]],
        [dwells["ATPgS_bind_msm"]],
        xerr=[[dwells["ATPgS_cat_msm_lower"]], [dwells["ATPgS_cat_msm_upper"]]],
        yerr=[[dwells["ATPgS_bind_msm_lower"]], [dwells["ATPgS_bind_msm_upper"]]],
        marker="o",
        markerfacecolor="none",
        capsize=3,
        color="C1",
        elinewidth=1.0,
        label="MSM (0.95 CI)",
    )
    ax_r2_2.errorbar(
        [dwells["ATPgS_cat_gme"]],
        [dwells["ATPgS_bind_gme"]],
        xerr=[[dwells["ATPgS_cat_gme_lower"]], [dwells["ATPgS_cat_gme_upper"]]],
        yerr=[[dwells["ATPgS_bind_gme_lower"]], [dwells["ATPgS_bind_gme_upper"]]],
        marker="o",
        markerfacecolor="none",
        capsize=3,
        color="C0",
        elinewidth=1.0,
        label="GME (0.95 CI)",
    )
    ax_r2_2.fill_between(
        [0.0, max_feasible_dwell_ATPgS + 1.0],
        [max_feasible_dwell_ATPgS, -1.0],
        y2=100,
        alpha=0.25,
        color="k",
        label="Infeasible region",
    )
    ax_r2_2.set_xlim(0, 20)
    ax_r2_2.set_ylim(0, 50)
    ax_r2_2.set_xlabel("Mean catalytic dwell (ms)", fontsize=8)
    ax_r2_2.set_ylabel("Mean binding dwell (ms)", fontsize=8)
    ax_r2_2.set_box_aspect(5 / 2)
    ax_r2_2.set_anchor("S")
    ax_r2_2.text(
        0.93,
        0.02,
        r"ATP$\gamma$S",
        transform=ax_r2_2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    max_feasible_dwell_ATP = np.ptp(df_ATP["Time"]) / np.ptp(df_ATP["Revo"]) / 3.0
    ax_r2_3.errorbar(
        [dwells["ATP_short_msm"]],
        [dwells["ATP_cat_msm"]],
        xerr=[[dwells["ATP_short_msm_lower"]], [dwells["ATP_short_msm_upper"]]],
        yerr=[[dwells["ATP_cat_msm_lower"]], [dwells["ATP_cat_msm_upper"]]],
        marker="o",
        markerfacecolor="none",
        capsize=3,
        color="C1",
        elinewidth=1.0,
        label="MSM (0.95 CI)",
    )
    ax_r2_3.errorbar(
        [dwells["ATP_short_gme"]],
        [dwells["ATP_cat_gme"]],
        xerr=[[dwells["ATP_short_gme_lower"]], [dwells["ATP_short_gme_upper"]]],
        yerr=[[dwells["ATP_cat_gme_lower"]], [dwells["ATP_cat_gme_upper"]]],
        marker="o",
        markerfacecolor="none",
        capsize=3,
        color="C0",
        elinewidth=1.0,
        label="GME (0.95 CI)",
    )
    ax_r2_3.fill_between(
        [0.0, max_feasible_dwell_ATP + 1.0],
        [max_feasible_dwell_ATP, -1.0],
        y2=100,
        alpha=0.25,
        color="k",
        label="Infeasible region",
    )
    ax_r2_3.set_xlim(0, 0.2)
    ax_r2_3.set_ylim(0, 0.5)
    ax_r2_3.set_xlabel("Mean short dwell (ms)", fontsize=8)
    ax_r2_3.set_ylabel("Mean catalytic dwell (ms)", fontsize=8)
    ax_r2_3.set_box_aspect(5 / 2)
    ax_r2_3.set_anchor("S")
    ax_r2_3.text(
        0.93,
        0.02,
        "ATP",
        transform=ax_r2_3.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    dwell_handles, dwell_labels = ax_r2_2.get_legend_handles_labels()
    dwell_legend_items = dict(zip(dwell_labels, dwell_handles))
    dwell_labels = [
        "Infeasible region",
        "MSM (0.95 CI)",
        "CPD (0.95 CI)",
        "GME (0.95 CI)",
    ]
    dwell_handles = [dwell_legend_items[label] for label in dwell_labels]

    ### final legend positions and panel labels ###
    panel_label_fontsize = 8
    cutoff_legend_center_offset = 0.017
    dwell_positions = [ax.get_position() for ax in [ax_r2_2, ax_r2_3]]
    prop_positions = [ax.get_position() for ax in [ax_r2_0, ax_r2_1]]
    lower_legend_top_y = max(pos.y1 for pos in dwell_positions) + 0.08
    fig.legend(
        handles=[cutoff_line],
        labels=["Cutoff"],
        loc="upper center",
        bbox_to_anchor=(
            (
                min(pos.x0 for pos in prop_positions)
                + max(pos.x1 for pos in prop_positions)
            )
            / 2,
            lower_legend_top_y - cutoff_legend_center_offset,
        ),
        fontsize=8,
        frameon=False,
        handlelength=1.5,
    )
    dwell_legend = fig.legend(
        handles=dwell_handles,
        labels=dwell_labels,
        loc="upper center",
        bbox_to_anchor=(
            (
                min(pos.x0 for pos in dwell_positions)
                + max(pos.x1 for pos in dwell_positions)
            )
            / 2,
            lower_legend_top_y,
        ),
        ncol=2,
        fontsize=8,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
    )
    ax_r1_0.text(
        -0.3,
        rotor_y,
        "(a)",
        transform=ax_r1_0.transAxes,
        ha="left",
        va="center",
        fontsize=panel_label_fontsize,
    )
    fig.draw_without_rendering()
    angle_ylabel_bbox = ax_r1_1.yaxis.label.get_window_extent()
    panel_a_x = fig.transFigure.inverted().transform(
        ax_r1_0.transAxes.transform((-0.3, rotor_y))
    )[0]
    panel_b_x = (
        fig.transFigure.inverted().transform(
            (angle_ylabel_bbox.x0, angle_ylabel_bbox.y1)
        )[0]
        - 0.005
    )
    dwell_legend_text_bboxes = [
        text.get_window_extent() for text in dwell_legend.get_texts()
    ]
    dwell_legend_first_row_y = fig.transFigure.inverted().transform(
        (0, max((bbox.y0 + bbox.y1) / 2 for bbox in dwell_legend_text_bboxes))
    )[1]
    fig.text(
        panel_b_x,
        r1_legend_y,
        "(b)",
        ha="left",
        va="center",
        fontsize=panel_label_fontsize,
    )
    fig.text(
        panel_a_x,
        dwell_legend_first_row_y,
        "(c)",
        ha="left",
        va="center",
        fontsize=panel_label_fontsize,
    )
    fig.text(
        panel_b_x,
        dwell_legend_first_row_y,
        "(d)",
        ha="left",
        va="center",
        fontsize=panel_label_fontsize,
    )

    fig.savefig(RES_DIR / FIG_FN, bbox_inches="tight")
    print(f"BMF1 figure saved to {RES_DIR / FIG_FN!s}")

    with open(RES_DIR / DWELL_FN, "w") as f:
        json.dump(dwells, f)
