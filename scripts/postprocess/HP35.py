"""scripts/postprocess/HP35.py

Analyze TCL-GME-DT models of the Lys24-Nle/Lys29-Nle HP35 trajectory from
S. Piana, K. Lindorff-Larsen, D.E. Shaw,
Protein folding kinetics and thermodynamics from atomistic simulation,
Proc. Natl. Acad. Sci. U.S.A. 109, 17845 (2012).
The trajectory was coarse-grained following the contact-based procedure of
# D. Nagel, S. Sartore, and G. Stock,
# Selecting features for Markov modeling: a case study of HP35,
# J. Chem. Theory Comput. 19, 3391 (2023).
This script takes about 1 hr to run.
"""

### IMPORTS ###
import fnmatch
import json
import os

import lightning as L
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import expon, norm, sem
from scipy.stats import t as student
from tqdm import tqdm

from gmex.markov_process import MarkovChain
from gmex.utils import (
    fpts_from_traj,
    get_bootstrap_curve_CI,
    get_data_dir,
    get_results_dir,
    load_times_macrostates,
)

### CONFIGURATIONS ###
DATA_DIR = get_data_dir() / "HP35"
RES_DIR = get_results_dir() / "HP35"  # directory for saving results
MAIN_RES_GS_FN = "HP35-M3-L64-TCL-Gs.pt"  # filename to save mean TCL-GME-DT propagators
BOOTSTRAP_DIR = RES_DIR / "bootstraps"  # directory for saving bootstrap results
BOOTSTRAP_GS_FN = (
    "HP35-M3-L32-TCL-Gs-BS*.pt"  # filename to save bootstrap TCL-GME-DT propagators
)
FPT_FN = "HP35-fpt.json"  # filename to save first-passage times from mean model
MFPT_FN = "HP35-mfpt.json"  # filename to save mean first passage times
FIG_FN = (
    "HP35-fig-no-ribbon.pdf"  # filename for HP35 figure without ribbon in panel (a)
)

RESAMPLING = 5  # we want a 1 ns timestep and MD trajectory is sampled every 200 ps
TAU_21_LAB_MEAN = (
    730  # mean experimental unfolded -> near-native first-passage time in ns
)
TAU_21_LAB_SE = 50  # experimental standard error of preceding
TAU_10_LAB = 70  # approximate experimental near-native -> native first-passage time
N_LAGS_TRUNC = 32  # TCL-GME-DT propagator plateaus at lag of 32 ns
N_TRAJ_FPT = 10000  # number of trajectories to simulate per first-passage time

SEED = 42  # randomization seed for FPT simulations
CI_PCT = 0.95  # confidence-interval width


### HELPER ###
def _get_bootstrap_fns(template: str) -> list[str]:
    """Get filenames in BOOTSTRAP_DIR that match template."""
    return sorted(
        name for name in os.listdir(BOOTSTRAP_DIR) if fnmatch.fnmatch(name, template)
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

    ### FIRST-PASSAGE TIMES ###
    if FPT_FN in os.listdir(RES_DIR) and MFPT_FN in os.listdir(RES_DIR):
        with open(RES_DIR / FPT_FN, "r") as f:
            fpts = json.load(f)
        Gs = torch.load(RES_DIR / MAIN_RES_GS_FN, weights_only=True)
        pctiles = np.arange(N_TRAJ_FPT) / (N_TRAJ_FPT - 1)
    else:  # getting FPTs is expensive and sometimes I just need to adjust the figure
        fpts = {}
        mfpts = {}

        # unfortunately we have to use multiple loops because the buffer is huge
        bootstrap_fns = _get_bootstrap_fns(BOOTSTRAP_GS_FN)
        print(f"{len(bootstrap_fns)} bootstraps detected")
        bootstrap_fpt_buffer = np.empty(
            (len(bootstrap_fns), N_TRAJ_FPT), dtype=np.uint32
        )

        # get GME 2 -> 1 FPT CIs
        for i in tqdm(
            range(len(bootstrap_fns)), desc="GME 2 -> 1 FPT CIs", leave=False
        ):
            Gs = torch.load(BOOTSTRAP_DIR / bootstrap_fns[i], weights_only=True)
            gme = MarkovChain(Gs[: N_LAGS_TRUNC + 1], 1.0, device=device)
            bootstrap_fpt_buffer[i] = (
                gme.first_passage_times(2, 1, n_trajs=N_TRAJ_FPT)
                .cpu()
                .numpy()
                .astype("uint32")
            )

        mfpts_gme_21 = np.mean(bootstrap_fpt_buffer, axis=1)
        mfpts_gme_21_lower, mfpts_gme_21_upper = get_bootstrap_curve_CI(
            mfpts_gme_21[:, None], CI_PCT
        )
        mfpts["gme_21_lower"] = mfpts_gme_21_lower.item()
        mfpts["gme_21_upper"] = mfpts_gme_21_upper.item()
        del mfpts_gme_21, mfpts_gme_21_lower, mfpts_gme_21_upper

        fpts_gme_21_lower, fpts_gme_21_upper = get_bootstrap_curve_CI(
            bootstrap_fpt_buffer, CI_PCT
        )
        fpts["gme_21_lower"] = fpts_gme_21_lower.tolist()
        fpts["gme_21_upper"] = fpts_gme_21_upper.tolist()

        # get GME 1 -> 0 FPT CIs
        for i in tqdm(
            range(len(bootstrap_fns)), desc="GME 1 -> 0 FPT CIs", leave=False
        ):
            Gs = torch.load(BOOTSTRAP_DIR / bootstrap_fns[i], weights_only=True)
            gme = MarkovChain(Gs[: N_LAGS_TRUNC + 1], 1.0, device=device)
            bootstrap_fpt_buffer[i] = (
                gme.first_passage_times(1, 0, n_trajs=N_TRAJ_FPT)
                .cpu()
                .numpy()
                .astype("uint32")
            )

        mfpts_gme_10 = np.mean(bootstrap_fpt_buffer, axis=1)
        mfpts_gme_10_lower, mfpts_gme_10_upper = get_bootstrap_curve_CI(
            mfpts_gme_10[:, None], CI_PCT
        )
        mfpts["gme_10_lower"] = mfpts_gme_10_lower.item()
        mfpts["gme_10_upper"] = mfpts_gme_10_upper.item()
        del mfpts_gme_10, mfpts_gme_10_lower, mfpts_gme_10_upper

        fpts_gme_10_lower, fpts_gme_10_upper = get_bootstrap_curve_CI(
            bootstrap_fpt_buffer, CI_PCT
        )
        fpts["gme_10_lower"] = fpts_gme_10_lower.tolist()
        fpts["gme_10_upper"] = fpts_gme_10_upper.tolist()

        # get MSM 2 -> 1 FPT CIs
        for i in tqdm(
            range(len(bootstrap_fns)), desc="MSM 2 -> 1 FPT CIs", leave=False
        ):
            Gs = torch.load(BOOTSTRAP_DIR / bootstrap_fns[i], weights_only=True)
            msm = MarkovChain(Gs[1], 1.0, device=device)
            bootstrap_fpt_buffer[i] = (
                msm.first_passage_times(2, 1, n_trajs=N_TRAJ_FPT)
                .cpu()
                .numpy()
                .astype("uint32")
            )

        mfpts_msm_21 = np.mean(bootstrap_fpt_buffer, axis=1)
        mfpts_msm_21_lower, mfpts_msm_21_upper = get_bootstrap_curve_CI(
            mfpts_msm_21[:, None], CI_PCT
        )
        mfpts["msm_21_lower"] = mfpts_msm_21_lower.item()
        mfpts["msm_21_upper"] = mfpts_msm_21_upper.item()
        del mfpts_msm_21, mfpts_msm_21_lower, mfpts_msm_21_upper

        fpts_msm_21_lower, fpts_msm_21_upper = get_bootstrap_curve_CI(
            bootstrap_fpt_buffer, CI_PCT
        )
        fpts["msm_21_lower"] = fpts_msm_21_lower.tolist()
        fpts["msm_21_upper"] = fpts_msm_21_upper.tolist()

        # get MSM 1 -> 0 FPT CIs
        for i in tqdm(
            range(len(bootstrap_fns)), desc="MSM 1 -> 0 FPT CIs", leave=False
        ):
            Gs = torch.load(BOOTSTRAP_DIR / bootstrap_fns[i], weights_only=True)
            msm = MarkovChain(Gs[1], 1.0, device=device)
            bootstrap_fpt_buffer[i] = (
                msm.first_passage_times(1, 0, n_trajs=N_TRAJ_FPT)
                .cpu()
                .numpy()
                .astype("uint32")
            )

        mfpts_msm_10 = np.mean(bootstrap_fpt_buffer, axis=1)
        mfpts_msm_10_lower, mfpts_msm_10_upper = get_bootstrap_curve_CI(
            mfpts_msm_10[:, None], CI_PCT
        )
        mfpts["msm_10_lower"] = mfpts_msm_10_lower.item()
        mfpts["msm_10_upper"] = mfpts_msm_10_upper.item()
        del mfpts_msm_10, mfpts_msm_10_lower, mfpts_msm_10_upper

        fpts_msm_10_lower, fpts_msm_10_upper = get_bootstrap_curve_CI(
            bootstrap_fpt_buffer, CI_PCT
        )
        fpts["msm_10_lower"] = fpts_msm_10_lower.tolist()
        fpts["msm_10_upper"] = fpts_msm_10_upper.tolist()

        del bootstrap_fpt_buffer

        # get FPT means
        Gs = torch.load(RES_DIR / MAIN_RES_GS_FN, weights_only=True)

        gme = MarkovChain(Gs[: N_LAGS_TRUNC + 1], 1.0, device=device)
        fpts["gme_21_mean"] = gme.first_passage_times(2, 1, n_trajs=N_TRAJ_FPT).tolist()
        fpts["gme_10_mean"] = gme.first_passage_times(1, 0, n_trajs=N_TRAJ_FPT).tolist()
        mfpts["gme_21_mean"] = sum(fpts["gme_21_mean"]) / len(fpts["gme_21_mean"])
        mfpts["gme_10_mean"] = sum(fpts["gme_10_mean"]) / len(fpts["gme_10_mean"])

        msm = MarkovChain(Gs[1], 1.0, device=device)
        fpts["msm_21_mean"] = msm.first_passage_times(2, 1, n_trajs=N_TRAJ_FPT).tolist()
        fpts["msm_10_mean"] = msm.first_passage_times(1, 0, n_trajs=N_TRAJ_FPT).tolist()
        mfpts["msm_21_mean"] = sum(fpts["msm_21_mean"]) / len(fpts["msm_21_mean"])
        mfpts["msm_10_mean"] = sum(fpts["msm_10_mean"]) / len(fpts["msm_10_mean"])

        pctiles = np.arange(N_TRAJ_FPT) / (N_TRAJ_FPT - 1)
        norm_CI_ppf_one_sided = -norm.ppf(0.5 - CI_PCT / 2.0).item()
        fpts["lab_21_mean"] = expon.ppf(pctiles, scale=TAU_21_LAB_MEAN).tolist()
        fpts["lab_21_lower"] = expon.ppf(
            pctiles, scale=TAU_21_LAB_MEAN - norm_CI_ppf_one_sided * TAU_21_LAB_SE
        ).tolist()
        fpts["lab_21_upper"] = expon.ppf(
            pctiles, scale=TAU_21_LAB_MEAN + norm_CI_ppf_one_sided * TAU_21_LAB_SE
        ).tolist()
        fpts["lab_10"] = expon.ppf(pctiles, scale=TAU_10_LAB).tolist()

        student_CI_ppf_one_sided = lambda n: (
            -student.ppf(0.5 - CI_PCT / 2.0, n - 1).item()
        )
        _, _, _, traj, _ = load_times_macrostates(DATA_DIR, device=device)
        traj = torch.tensor(traj)
        fpts["sim_21"] = (fpts_from_traj(traj, 2, 1).cpu() / RESAMPLING).tolist()
        fpts["sim_10"] = (fpts_from_traj(traj, 1, 0).cpu() / RESAMPLING).tolist()
        mfpts["sim_21_mean"] = float(np.mean(fpts["sim_21"]))
        mfpts["sim_10_mean"] = float(np.mean(fpts["sim_10"]))
        mfpts["sim_21_err"] = float(
            student_CI_ppf_one_sided(len(fpts["sim_21"])) * sem(fpts["sim_21"])
        )
        mfpts["sim_10_err"] = float(
            student_CI_ppf_one_sided(len(fpts["sim_10"])) * sem(fpts["sim_10"])
        )
        del traj

        with open(RES_DIR / FPT_FN, "w") as f:
            json.dump(fpts, f)

        with open(RES_DIR / MFPT_FN, "w") as f:
            json.dump(mfpts, f)

    ### PLOTTING ###
    pctiles_sim_21 = np.arange(len(fpts["sim_21"])) / (len(fpts["sim_21"]) - 1)
    pctiles_sim_10 = np.arange(len(fpts["sim_10"])) / (len(fpts["sim_10"]) - 1)

    ### plot setup ###
    fig, ax = plt.subplots(
        2, 2, figsize=(6, 3.5), gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.subplots_adjust(wspace=1 / 3, hspace=1 / 2)  # increase spacing between subplots
    for a in ax.flat:
        a.tick_params(axis="both", which="both", labelsize=8)
        for axis in ["right", "top", "left", "bottom"]:
            a.spines[axis].set_linewidth(0.5)

    ### panel (a) ###
    ax[0, 0].axis("off")

    ### panel (b) ###
    def broken_y(values, low=(0.0, 0.05), high=(0.95, 1.0), gap=0.08):
        values = np.asarray(values)
        mapped = np.full(values.shape, np.nan, dtype=float)
        lower = (low[0] <= values) & (values <= low[1])
        upper = (high[0] <= values) & (values <= high[1])
        lower_height = 0.5 - gap / 2
        upper_start = 0.5 + gap / 2
        mapped[lower] = (values[lower] - low[0]) / (low[1] - low[0]) * lower_height
        mapped[upper] = (
            upper_start + (values[upper] - high[0]) / (high[1] - high[0]) * lower_height
        )
        return mapped

    Gs_y = Gs.reshape(Gs.shape[0], -1)[1:].cpu().numpy()
    ax[0, 1].plot(
        torch.arange(1, Gs.shape[0]).cpu(), broken_y(Gs_y), linewidth=0.5, color="C0"
    )
    ax[0, 1].set_ylim(0, 1)
    yticks = np.array([0.0, 0.02, 0.04, 0.96, 0.98, 1.0])
    ax[0, 1].set_yticks(broken_y(yticks))
    ax[0, 1].set_yticklabels(["0.00", "0.02", "0.04", "0.96", "0.98", "1.00"])
    for y in (0.5 - 0.08 / 2, 0.5 + 0.08 / 2):
        ax[0, 1].plot(
            (-0.015, 0.015),
            (y - 0.015, y + 0.015),
            transform=ax[0, 1].transAxes,
            color="k",
            linewidth=0.5,
            clip_on=False,
        )
    ax[0, 1].set_ylabel(r"$\hat{G}^{(n)}_{ij}$", fontsize=8)
    ax[0, 1].set_xlabel(r"Lagtime $n\tau$ (ns)", fontsize=8)
    ax[0, 1].axvline(
        x=N_LAGS_TRUNC, linestyle="--", label="Cutoff", color="k", linewidth=0.5
    )
    ax[0, 1].set_xlim(1, Gs.shape[0] - 1)
    ax[0, 1].set_xticks(10 * np.arange(Gs.shape[0] // 10) + 10)
    ax[0, 1].legend(fontsize=6, loc="center right")

    ### panel (c) ###
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_xlim(0.1, 1e4)
    ax[1, 0].set_xlabel(r"Unfolded $\to$ near-native FPT (ns)", fontsize=8)
    ax[1, 0].set_ylabel("CDF", fontsize=8)

    (sim_21,) = ax[1, 0].plot(fpts["sim_21"], pctiles_sim_21, color="C2", label="MD")
    ax[1, 0].plot(fpts["msm_21_mean"], pctiles, color="C1", label="_nolegend_")
    ax[1, 0].plot(fpts["gme_21_mean"], pctiles, color="C0", label="_nolegend_")
    ax[1, 0].plot(fpts["lab_21_mean"], pctiles, color="k", label="_nolegend_")

    ax[1, 0].fill_betweenx(
        pctiles,
        fpts["msm_21_lower"],
        fpts["msm_21_upper"],
        color="C1",
        alpha=0.25,
        label="_nolegend_",
    )
    ax[1, 0].fill_betweenx(
        pctiles,
        fpts["gme_21_lower"],
        fpts["gme_21_upper"],
        color="C0",
        alpha=0.25,
        label="_nolegend_",
    )
    ax[1, 0].fill_betweenx(
        pctiles,
        fpts["lab_21_lower"],
        fpts["lab_21_upper"],
        color="k",
        alpha=0.25,
        label="_nolegend_",
    )

    msm_handle = (
        Patch(facecolor="C1", edgecolor="none", alpha=0.25),
        Line2D([0], [0], color="C1", lw=1.0),
    )
    gme_handle = (
        Patch(facecolor="C0", edgecolor="none", alpha=0.25),
        Line2D([0], [0], color="C0", lw=1.0),
    )
    lab_handle = (
        Patch(facecolor="k", edgecolor="none", alpha=0.25),
        Line2D([0], [0], color="k", lw=1.0),
    )
    ax[1, 0].legend(
        handles=[lab_handle, sim_21, msm_handle, gme_handle],
        labels=["$T$-jump\n(0.95 CI)", "MD", "MSM\n(0.95 CI)", "GME\n(0.95 CI)"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=6,
        loc="upper left",
    )

    ### panel (d) ###
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_xlim(0.1, 1e4)
    ax[1, 1].set_xlabel(r"Near-native $\to$ native FPT (ns)", fontsize=8)
    ax[1, 1].set_ylabel("CDF", fontsize=8)

    (sim_10,) = ax[1, 1].plot(fpts["sim_10"], pctiles_sim_10, color="C2", label="MD")
    ax[1, 1].plot(fpts["msm_10_mean"], pctiles, color="C1", label="_nolegend_")
    ax[1, 1].plot(fpts["gme_10_mean"], pctiles, color="C0", label="_nolegend_")
    (lab_10,) = ax[1, 1].plot(
        fpts["lab_10"], pctiles, color="k", linestyle=":", label="$T$-jump\n(approx.)"
    )

    ax[1, 1].fill_betweenx(
        pctiles,
        fpts["msm_10_lower"],
        fpts["msm_10_upper"],
        color="C1",
        alpha=0.25,
        label="_nolegend_",
    )
    ax[1, 1].fill_betweenx(
        pctiles,
        fpts["gme_10_lower"],
        fpts["gme_10_upper"],
        color="C0",
        alpha=0.25,
        label="_nolegend_",
    )

    lab_md_legend = ax[1, 1].legend(
        handles=[lab_10, sim_10],
        labels=["$T$-jump\n(approx.)", "MD"],
        fontsize=6,
        loc="upper left",
    )
    ax[1, 1].add_artist(lab_md_legend)
    ax[1, 1].legend(
        handles=[msm_handle, gme_handle],
        labels=["MSM\n(0.95 CI)", "GME\n(0.95 CI)"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=6,
        loc="lower right",
    )

    ### annotation and saving ###
    fig.draw_without_rendering()
    fig_inv = fig.transFigure.inverted()

    def text_center_x(text):
        bbox = text.get_window_extent()
        return fig_inv.transform(((bbox.x0 + bbox.x1) / 2, bbox.y0))[0]

    def data_y_to_fig(ax, y):
        return fig_inv.transform(ax.transData.transform((ax.get_xlim()[0], y)))[1]

    panel_label_x = {
        "left": text_center_x(ax[1, 0].yaxis.label),
        "right": text_center_x(ax[1, 1].yaxis.label),
    }
    panel_label_y = {
        "top": data_y_to_fig(ax[0, 1], 1.0),
        "bottom": data_y_to_fig(ax[1, 1], 1.0),
    }
    panel_label_x_nudge = (
        -0.01
    )  # figure-coordinate units; more negative moves labels left
    for label, x_key, y_key in [
        ("(a)", "left", "top"),
        ("(b)", "right", "top"),
        ("(c)", "left", "bottom"),
        ("(d)", "right", "bottom"),
    ]:
        fig.text(
            panel_label_x[x_key] + panel_label_x_nudge,
            panel_label_y[y_key],
            label,
            ha="center",
            va="center",
            fontsize=8,
        )

    ### saving figure ###
    fig.savefig(RES_DIR / FIG_FN, bbox_inches="tight")
    print(f"HP35 figure saved to {RES_DIR / FIG_FN!s}")
