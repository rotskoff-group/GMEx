#!/usr/bin/env python3
# GMEx/scripts/postprocess/toys.py

"""
Analyze DTGME models of a toy Markov jump processes on four-state cycles.
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
import fnmatch
import os
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
import numpy as np
from scipy.stats import t as student
import torch

from gmex.nakajima_zwanzig import get_Ks_from_Us
from gmex.utils import get_data_dir, get_results_dir, load_times_macrostates


### CONFIGURATIONS ###
N_STEPS = [int(1e2), int(1e3), int(1e4), int(1e5), int(1e6)] # trajectory lengths
N_LAGS_TRUNC = 3 # TCL-GME-DT propagators plateau at a lag of 3
CI_PCT = 0.95 # confidence-interval width
N_REPS = 10 # number of replicates

DATA_DIR = get_data_dir() / 'toys'
RES_DIR = get_results_dir() / 'toys'
TRUE_US_FN = 'toy-Us-true.pt'
TRUE_KS_FN = 'toy-Ks-true.pt'
TRUE_GS_FN = 'toy-Gs-true.pt'
REP_US_FN = 'toy-Us-rep*.pt'
REP_GS_FN = 'toy-Gs-rep*.pt'
SCHEME_FN = 'schematic.pdf'
ADVERSARIAL_EXAMPLE_FN = 'adversarial-example.pdf'


### HELPERS ###
def _get_replicate_fns(template: str, replicate_dir: Path):
    '''Get replicate filenames matching template.'''
    return sorted(
        replicate_dir / name
        for name in os.listdir(replicate_dir)
        if fnmatch.fnmatch(name, template)
    )

def _n_power10_tag(n: int) -> str:
    """Convert powers of ten like int(1e2) -> 'N1e2'."""
    p = int(np.log10(n))
    if 10 ** p != n:
        raise ValueError(f'Expected an integer power of ten, got {n}.')
    return f'N1e{p}'

def _add_filled_arrowhead(
        fig, tip, direction, color='k', zorder=4, head_length=0.06,
        head_width=0.055
    ):
    """Add a filled arrowhead in figure coordinates and return its base."""
    fig_size = np.array(fig.get_size_inches())
    tip_in = np.array(tip) * fig_size
    direction_in = np.array(direction) * fig_size
    direction_in = direction_in / np.linalg.norm(direction_in)
    normal_in = np.array([-direction_in[1], direction_in[0]])
    base_in = tip_in - head_length * direction_in
    vertices = np.vstack([
        tip_in,
        base_in + 0.5 * head_width * normal_in,
        base_in - 0.5 * head_width * normal_in
    ]) / fig_size
    fig.patches.append(Polygon(
        vertices,
        closed=True,
        transform=fig.transFigure,
        facecolor=color,
        edgecolor=color,
        linewidth=0,
        zorder=zorder,
        clip_on=False
    ))
    return base_in / fig_size

class _HandlerArrow(HandlerBase):
    """Draw right-pointing arrows in legend handles."""

    def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height,
            fontsize, trans
        ):
        y = ydescent + 0.5 * height
        if getattr(orig_handle, '_filled_head_only', False):
            color = orig_handle.get_edgecolor()
            tip_x = xdescent + 0.82 * width
            head_base_x = xdescent + 0.50 * width
            shaft = Line2D(
                [xdescent, head_base_x],
                [y, y],
                linewidth=orig_handle.get_linewidth(),
                linestyle=orig_handle.get_linestyle(),
                color=color,
                transform=trans
            )
            head = Polygon(
                [
                    (tip_x, y),
                    (head_base_x, y + 0.34 * height),
                    (head_base_x, y - 0.34 * height)
                ],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                transform=trans
            )
            return [shaft, head]
        arrow = FancyArrowPatch(
            (xdescent, y),
            (xdescent + width, y),
            arrowstyle=orig_handle.get_arrowstyle(),
            mutation_scale=fontsize,
            linewidth=orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
            facecolor=orig_handle.get_edgecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=trans
        )
        return [arrow]

def _load_toy_summary(toyname: str):
    """Load a toy dataset, true operators, and replicate estimation errors."""
    _, _, _, data, nmacro = load_times_macrostates(DATA_DIR / toyname, 'cpu')
    trace = data[0][:101]
    del data # data is huge

    Us_true = torch.load(DATA_DIR / toyname / TRUE_US_FN, weights_only=True).to('cpu')
    Ks_diagrammatic = get_Ks_from_Us(Us_true, verbose=False).to('cpu')
    Ks_analytic = torch.load(DATA_DIR / toyname / TRUE_KS_FN, weights_only=True).to('cpu')
    Gs_true = torch.load(DATA_DIR / toyname / TRUE_GS_FN, weights_only=True).to('cpu')
    plot_lags = np.arange(Us_true.shape[0])

    err_U_means = np.empty((N_LAGS_TRUNC, len(N_STEPS)))
    err_U_stderrs = np.empty((N_LAGS_TRUNC, len(N_STEPS)))
    err_G_means = np.empty((N_LAGS_TRUNC, len(N_STEPS)))
    err_G_stderrs = np.empty((N_LAGS_TRUNC, len(N_STEPS)))
    err_K_means = np.empty((N_LAGS_TRUNC, len(N_STEPS)))
    err_K_stderrs = np.empty((N_LAGS_TRUNC, len(N_STEPS)))

    for i in range(len(N_STEPS)):
        rep_dir = RES_DIR / toyname / _n_power10_tag(N_STEPS[i])
        rep_U_fns = _get_replicate_fns(REP_US_FN, rep_dir)
        rep_G_fns = _get_replicate_fns(REP_GS_FN, rep_dir)
        assert len(rep_U_fns) == len(rep_G_fns) == N_REPS

        rep_U_buffer = torch.empty(
            (len(rep_U_fns), N_LAGS_TRUNC), dtype=torch.float64, device='cpu'
        )
        rep_K_buffer = torch.empty(
            (len(rep_U_fns), N_LAGS_TRUNC), dtype=torch.float64, device='cpu'
        )
        rep_G_buffer = torch.empty(
            (len(rep_G_fns), N_LAGS_TRUNC), dtype=torch.float64, device='cpu'
        )

        for j in range(len(rep_U_fns)):
            Us_est = torch.load(rep_U_fns[j], weights_only=True).to('cpu')
            Us_diff = (
                Us_true[1:N_LAGS_TRUNC + 1] - Us_est[1:N_LAGS_TRUNC + 1]
            )
            rep_U_buffer[j] = torch.linalg.matrix_norm(Us_diff, ord=2)

            Ks_est = get_Ks_from_Us(Us_est, verbose=False)
            Ks_diff = (
                Ks_analytic[:N_LAGS_TRUNC] - Ks_est[:N_LAGS_TRUNC]
            )
            rep_K_buffer[j] = torch.linalg.matrix_norm(Ks_diff, ord=2)

        for j in range(len(rep_G_fns)):
            Gs_est = torch.load(rep_G_fns[j], weights_only=True).to('cpu')
            Gs_diff = (
                Gs_true[1:N_LAGS_TRUNC + 1] - Gs_est[1:N_LAGS_TRUNC + 1]
            )
            rep_G_buffer[j] = torch.linalg.matrix_norm(Gs_diff, ord=2)

        err_U_means[:, i] = rep_U_buffer.mean(dim=0)
        err_U_stderrs[:, i] = rep_U_buffer.std(dim=0) / np.sqrt(rep_U_buffer.shape[0])
        err_G_means[:, i] = rep_G_buffer.mean(dim=0)
        err_G_stderrs[:, i] = rep_G_buffer.std(dim=0) / np.sqrt(rep_G_buffer.shape[0])
        err_K_means[:, i] = rep_K_buffer.mean(dim=0)
        err_K_stderrs[:, i] = rep_K_buffer.std(dim=0) / np.sqrt(rep_K_buffer.shape[0])

    return {
        'trace': trace,
        'nmacro': nmacro,
        'Us_true': Us_true,
        'Ks_diagrammatic': Ks_diagrammatic,
        'Ks_analytic': Ks_analytic,
        'Gs_true': Gs_true,
        'plot_lags': plot_lags,
        'err_U_means': err_U_means,
        'err_U_stderrs': err_U_stderrs,
        'err_G_means': err_G_means,
        'err_G_stderrs': err_G_stderrs,
        'err_K_means': err_K_means,
        'err_K_stderrs': err_K_stderrs,
    }


def _draw_compact_diagrams(ax: Axes):
    """Draw the Feynman diagrams in a compact schematic panel."""
    dot_x = np.array([0.0, 1.25, 2.50])
    kernel_terms = [
        (3, True,  True), # o-o-o
        (2, False, True), # o o-o
        (1, False, False), # o o o
        (0, True,  False) # o-o o
    ]

    scatter_artists = []
    line_artists = []
    for dot_y, c01, c12 in kernel_terms:
        scatter_artists.append(ax.scatter(
            dot_x, [dot_y] * 3, s=8, color='black', zorder=3, clip_on=False
        ))
        if c01:
            line, = ax.plot(
                [dot_x[0], dot_x[1]], [dot_y, dot_y],
                color='black', lw=0.5, clip_on=False
            )
            line_artists.append((line, 0, 1, dot_y))
        if c12:
            line, = ax.plot(
                [dot_x[1], dot_x[2]], [dot_y, dot_y],
                color='black', lw=0.5, clip_on=False
            )
            line_artists.append((line, 1, 2, dot_y))

    box = Rectangle(
        (dot_x[1] - 0.30, 0.75), dot_x[2] - dot_x[1] + 0.55, 1.5,
        fill=False,
        edgecolor='black',
        linewidth=0.5,
        linestyle=':',
        zorder=1,
        clip_on=False
    )
    ax.add_patch(box)
    time_arrow = FancyArrowPatch(
        (dot_x[-1] + 0.25, 4.0),
        (dot_x[0] - 0.25, 4.0),
        arrowstyle='->',
        mutation_scale=6,
        linewidth=0.5,
        color='black'
    )
    ax.add_patch(time_arrow)

    time_label = ax.text(dot_x.mean(), 4 + 1 / 6, 'Time', fontsize=8, ha='center', va='bottom')
    k_label = ax.text((dot_x[1] + dot_x[2]) / 2, 1.50, r'$K^{(1)}$', fontsize=8, ha='center', va='center')
    bottom_labels = [
        ax.text((dot_x[0] + dot_x[1]) / 2, -1/3, r'$U^{(2)}$', fontsize=8, ha='center', va='top'),
        ax.text(dot_x[2], -1/3, r'$K^{(0)}$', fontsize=8, ha='center', va='top'),
    ]

    ax.set_xlim(-0.15, 3.55)
    ax.set_ylim(-0.45, 4.35)
    ax.set_aspect('auto')
    ax.axis('off')
    return {
        'dot_x': dot_x,
        'dot_y': np.array([3.0, 2.0, 1.0, 0.0]),
        'scatter_artists': scatter_artists,
        'line_artists': line_artists,
        'box': box,
        'time_arrow': time_arrow,
        'time_label': time_label,
        'k_label': k_label,
        'bottom_labels': bottom_labels
    }


def _update_compact_diagram_xcoords(diagram_artists: dict, dot_x: np.ndarray) -> None:
    """Update compact diagram x-coordinates while keeping text style fixed."""
    diagram_artists['dot_x'] = dot_x
    dot_y = diagram_artists.get('dot_y', np.array([3.0, 2.0, 1.0, 0.0]))
    old_dot_y_to_idx = {3: 0, 2: 1, 1: 2, 0: 3}
    for scatter, row_y in zip(diagram_artists['scatter_artists'], dot_y):
        scatter.set_offsets(np.column_stack([dot_x, np.full(3, row_y)]))
    for line, left_idx, right_idx, old_row_y in diagram_artists['line_artists']:
        row_y = dot_y[old_dot_y_to_idx[old_row_y]]
        line.set_data([dot_x[left_idx], dot_x[right_idx]], [row_y, row_y])

    box = diagram_artists['box']
    box_x_pad = 0.28
    box.set_x(dot_x[1] - box_x_pad)
    box.set_width(dot_x[2] - dot_x[1] + 2 * box_x_pad)
    k_label_y = diagram_artists['k_label'].get_position()[1]
    diagram_artists['k_label'].set_position(((dot_x[1] + dot_x[2]) / 2, k_label_y))
    time_y = diagram_artists['time_label'].get_position()[1]
    diagram_artists['time_label'].set_position((dot_x.mean(), time_y))
    bottom_label_y0 = diagram_artists['bottom_labels'][0].get_position()[1]
    diagram_artists['bottom_labels'][0].set_position(((dot_x[0] + dot_x[1]) / 2, bottom_label_y0))


def _update_compact_diagram_ycoords(diagram_artists: dict, dot_y: np.ndarray) -> None:
    """Update compact diagram y-coordinates with equal row spacing."""
    diagram_artists['dot_y'] = dot_y
    dot_x = diagram_artists['dot_x']
    old_dot_y_to_idx = {3: 0, 2: 1, 1: 2, 0: 3}
    for scatter, row_y in zip(diagram_artists['scatter_artists'], dot_y):
        scatter.set_offsets(np.column_stack([dot_x, np.full(3, row_y)]))
    for line, left_idx, right_idx, old_row_y in diagram_artists['line_artists']:
        row_y = dot_y[old_dot_y_to_idx[old_row_y]]
        line.set_data([dot_x[left_idx], dot_x[right_idx]], [row_y, row_y])

    row_spacing = dot_y[0] - dot_y[1]
    box = diagram_artists['box']
    box_y_pad = 0.40 * row_spacing
    box.set_y(dot_y[2] - box_y_pad)
    box.set_height(dot_y[1] - dot_y[2] + 2 * box_y_pad)
    diagram_artists['k_label'].set_position((
        (dot_x[1] + dot_x[2]) / 2,
        (dot_y[1] + dot_y[2]) / 2
    ))

    arrow_y = dot_y[0] + row_spacing
    time_x = diagram_artists['time_label'].get_position()[0]
    diagram_artists['time_label'].set_position((time_x, arrow_y + 1 / 6))
    diagram_artists['time_arrow'].set_positions(
        (dot_x[-1] + 0.25, arrow_y),
        (dot_x[0] - 0.25, arrow_y)
    )


def _match_compact_diagram_arrow_to_time_label(fig, ax: Axes, diagram_artists: dict) -> None:
    """Make the top diagram arrow the rendered width of the Time label."""
    renderer = fig.canvas.get_renderer()
    time_bbox = diagram_artists['time_label'].get_window_extent(renderer)
    x0_data = ax.transData.inverted().transform((time_bbox.x0, time_bbox.y0))[0]
    x1_data = ax.transData.inverted().transform((time_bbox.x1, time_bbox.y1))[0]
    time_x, time_y = diagram_artists['time_label'].get_position()
    arrow_y = time_y - 1 / 6
    half_width = 0.68 * (x1_data - x0_data)
    diagram_artists['time_arrow'].set_positions(
        (time_x + half_width, arrow_y),
        (time_x - half_width, arrow_y)
    )


def _data_dx_from_fig_dx(fig, ax: Axes, dx_fig: float) -> float:
    """Convert a horizontal figure-coordinate offset to data coordinates."""
    inv_fig = fig.transFigure.inverted()
    x0_fig = inv_fig.transform(ax.transData.transform((0, 0)))[0]
    x1_fig = inv_fig.transform(ax.transData.transform((1, 0)))[0]
    return dx_fig / (x1_fig - x0_fig)


def _align_axes_text_right_to_fig_x(fig, ax: Axes, text, target_x: float) -> None:
    """Align the rendered right edge of an axes text object to a figure x."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    bbox = text.get_window_extent(renderer).transformed(inv_fig)
    text_x, text_y = text.get_position()
    text.set_position((
        text_x + _data_dx_from_fig_dx(fig, ax, target_x - bbox.x1),
        text_y
    ))


def _legend_unique(ax: Axes, **kwargs) -> None:
    """Draw a legend with only the first handle for each label."""
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label.startswith('_') or label in unique_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, **kwargs)


if __name__ == "__main__":
    try:
        plt.style.use('rotskoff')
        plt.rcParams.update({'text.usetex': True,
                            'font.family': 'CMU',
                            'text.latex.preamble': r'\usepackage{amsfonts,amssymb}'})
    except: pass

    ### DATALOADING AND ESTIMATION ERRORS ###
    student_CI_ppf_one_sided = -student.ppf(0.5 - CI_PCT / 2.0, N_REPS - 1).item()
    easy_summary = _load_toy_summary('easy')
    hard_summary = _load_toy_summary('hard')

    trace = easy_summary['trace']
    nmacro = easy_summary['nmacro']
    Us_true = easy_summary['Us_true']
    Ks_diagrammatic = easy_summary['Ks_diagrammatic']
    Ks_analytic = easy_summary['Ks_analytic']
    Gs_true = easy_summary['Gs_true']
    plot_lags = easy_summary['plot_lags']
    err_U_means = easy_summary['err_U_means']
    err_U_stderrs = easy_summary['err_U_stderrs']
    err_G_means = easy_summary['err_G_means']
    err_G_stderrs = easy_summary['err_G_stderrs']
    err_K_means = easy_summary['err_K_means']
    err_K_stderrs = easy_summary['err_K_stderrs']

    
    ### SCHEMATIC ###
    fig = plt.figure(figsize=(6, 7))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])

    top_column_wspace = 0.3
    middle_column_wspace = 0.6
    bottom_column_wspace = 0.6
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.25, 2.05], wspace=top_column_wspace)
    gs_middle = gs[1].subgridspec(1, 3, width_ratios=[10, 9, 10], wspace=middle_column_wspace)
    gs_bottom = gs[3].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=bottom_column_wspace)

    ax_top_left = fig.add_subplot(gs_top[0, 0])
    ax_top_right = fig.add_subplot(gs_top[0, 1])
    ax_middle_left = fig.add_subplot(gs_middle[0, 0])
    ax_middle_center = fig.add_subplot(gs_middle[0, 1])
    ax_middle_right = fig.add_subplot(gs_middle[0, 2], sharey=ax_middle_left)
    ax_center = fig.add_subplot(gs[2])
    ax_bottom_left = fig.add_subplot(gs_bottom[0, 0])
    ax_bottom_center = fig.add_subplot(gs_bottom[0, 1], sharey=ax_bottom_left)
    ax_bottom_right = fig.add_subplot(gs_bottom[0, 2], sharey=ax_bottom_left)

    axes = [
        ax_top_left, ax_top_right,
        ax_middle_left, ax_middle_center, ax_middle_right,
        ax_center,
        ax_bottom_left, ax_bottom_center, ax_bottom_right
    ]
    for a in axes:
        a.tick_params(axis='both', which='both', labelsize=8)
        for axis in ['right', 'top', 'left', 'bottom']:
            a.spines[axis].set_linewidth(0.6)
    fig.subplots_adjust(hspace=0.5)

    ### panel (a) ###
    def _draw_harpoon(
            ax: Axes, start: np.ndarray, end: np.ndarray, barb_side: np.ndarray
        ) -> None:
        """Draw harpoon."""
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        barb_side = np.array(barb_side, dtype=float)

        direction = end - start
        direction /= np.linalg.norm(direction)
        barb_side /= np.linalg.norm(barb_side)

        barb_end = end - direction * 0.08 + barb_side * 0.05
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', lw=0.5)
        ax.plot([end[0], barb_end[0]], [end[1], barb_end[1]], color='black', lw=0.5)

    side_length = 1.80
    circle_radius = 0.30
    state_pos = {
        'M1': np.array([side_length, side_length]),
        'M2': np.array([0.0, side_length]),
        'M3': np.array([0.0, 0.0]),
        'M4': np.array([side_length, 0.0]),
    }
    state_labels = {
        'M1': r'$\mu_1$',
        'M2': r'$\mu_2$',
        'M3': r'$\mu_3$',
        'M4': r'$\mu_4$',
    }
    cycle_state_labels = {
        'M1': r'$\mu_3$',
        'M2': r'$\mu_4$',
        'M3': r'$\mu_1$',
        'M4': r'$\mu_2$',
    }

    outer_offset = 0.10
    outer_trim = np.sqrt(circle_radius ** 2 - outer_offset ** 2)
    outer_label_offset = circle_radius - outer_offset
    right_outer_label_offset = outer_label_offset + 0.05
    left_outer_label_offset = outer_label_offset + 0.1
    outer_harpoons = [
        ('M1', 'M2', (0.0, 1.0), outer_label_offset, '1'),
        ('M2', 'M3', (-1.0, 0.0), left_outer_label_offset, r'$1/4$'),
        ('M3', 'M4', (0.0, -1.0), outer_label_offset, r'$1/4$'),
        ('M4', 'M1', (1.0, 0.0), right_outer_label_offset, r'$1/4$'),
    ]
    top_outer_rate_label = None
    left_outer_rate_label = None
    bottom_outer_rate_label = None
    for state_i, state_j, outside, label_offset, label in outer_harpoons:
        start = state_pos[state_i]
        end = state_pos[state_j]
        outside = np.array(outside)
        direction = end - start
        direction /= np.linalg.norm(direction)
        harpoon_start = start + outer_offset * outside + outer_trim * direction
        harpoon_end = end + outer_offset * outside - outer_trim * direction
        _draw_harpoon(ax_top_left, harpoon_start, harpoon_end, outside)
        rate_label = ax_top_left.text(
            *((harpoon_start + harpoon_end) / 2 + label_offset * outside),
            label,
            fontsize=8,
            ha='center',
            va='center'
        )
        if state_i == 'M1' and state_j == 'M2':
            top_outer_rate_label = rate_label
        if state_i == 'M2' and state_j == 'M3':
            left_outer_rate_label = rate_label
        if state_i == 'M3' and state_j == 'M4':
            bottom_outer_rate_label = rate_label

    inner_offset = 0.09
    inner_trim = np.sqrt(circle_radius ** 2 - inner_offset ** 2)
    inner_label_offset = circle_radius - inner_offset + 0.05
    inner_label_offset_close = inner_label_offset - 0.03
    inner_label_offset_far = inner_label_offset + 0.03
    inner_harpoons = [
        ('M1', 'M4', (-1.0, 0.0), inner_label_offset_far, r'$1/8$'),
        ('M4', 'M3', (0.0, 1.0), inner_label_offset_close, r'$1/8$'),
        ('M3', 'M2', (1.0, 0.0), inner_label_offset_far, r'$1/8$'),
        ('M2', 'M1', (0.0, -1.0), inner_label_offset_close, r'$1/2$'),
    ]
    for state_i, state_j, inside, label_offset, label in inner_harpoons:
        start = state_pos[state_i]
        end = state_pos[state_j]
        inside = np.array(inside)
        direction = end - start
        direction /= np.linalg.norm(direction)
        harpoon_start = start + inner_offset * inside + inner_trim * direction
        harpoon_end = end + inner_offset * inside - inner_trim * direction
        _draw_harpoon(ax_top_left, harpoon_start, harpoon_end, inside)
        ax_top_left.text(
            *((harpoon_start + harpoon_end) / 2 + label_offset * inside),
            label,
            fontsize=8,
            ha='center',
            va='center'
        )

    box_x_pad = 0.06
    box_y_pad = 0.22
    ax_top_left.add_patch(Rectangle(
        (-circle_radius - box_x_pad, side_length - circle_radius - box_y_pad),
        side_length + 2 * (circle_radius + box_x_pad),
        2 * (circle_radius + box_y_pad),
        fill=False,
        edgecolor='black',
        linewidth=0.5,
        linestyle=':',
        zorder=1
    ))

    for state, pos in state_pos.items():
        ax_top_left.add_patch(Circle(
            pos,
            radius=circle_radius,
            facecolor='white',
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        ))
        ax_top_left.text(
            *pos,
            cycle_state_labels[state],
            fontsize=8,
            ha='center',
            va='center',
            zorder=4
        )

    x_plot_margin = circle_radius + box_x_pad + 0.02
    y_plot_margin = circle_radius + box_y_pad + 0.02
    ax_top_left.set_xlim(-x_plot_margin, side_length + x_plot_margin)
    ax_top_left.set_ylim(-y_plot_margin, side_length + y_plot_margin)
    ax_top_left.set_aspect('equal', adjustable='box')
    ax_top_left.set_anchor('W')
    ax_top_left.axis('off')

    ### panel (b) ###
    trace_window = np.arange(0, 101)
    ax_top_right.plot(
        trace_window,
        np.asarray(trace)[trace_window],
        c='k'
    )
    ax_top_right.set_xlim(trace_window[0], trace_window[-1])
    ax_top_right.set_xlabel('Time', fontsize=8)
    ax_top_right.set_xticks(np.arange(0, 101, 25))
    ax_top_right.set_yticks(np.arange(nmacro))
    ax_top_right.yaxis.tick_right()
    ax_top_right.set_yticklabels([
        state_labels['M1'], state_labels['M2'], r'$\{\mu_3,\mu_4\}$'
    ])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    if top_outer_rate_label is not None and bottom_outer_rate_label is not None:
        top_rate_bbox = top_outer_rate_label.get_window_extent(renderer).transformed(inv_fig)
        bottom_rate_bbox = bottom_outer_rate_label.get_window_extent(renderer).transformed(inv_fig)
        top_macro_bbox = max(
            (
                ticklabel.get_window_extent(renderer).transformed(inv_fig)
                for ticklabel in ax_top_right.get_yticklabels()
                if ticklabel.get_visible()
            ),
            key=lambda bbox: (bbox.y0 + bbox.y1) / 2
        )
        xlabel_bbox = ax_top_right.xaxis.label.get_window_extent(renderer).transformed(inv_fig)
        top_rate_y = (top_rate_bbox.y0 + top_rate_bbox.y1) / 2
        bottom_rate_y = (bottom_rate_bbox.y0 + bottom_rate_bbox.y1) / 2
        top_macro_y = (top_macro_bbox.y0 + top_macro_bbox.y1) / 2
        xlabel_y = (xlabel_bbox.y0 + xlabel_bbox.y1) / 2
        top_left_pos = ax_top_left.get_position()
        rate_span = top_rate_y - bottom_rate_y
        target_span = top_macro_y - xlabel_y
        if rate_span > 0 and target_span > 0:
            top_rate_rel = (top_rate_y - top_left_pos.y0) / top_left_pos.height
            bottom_rate_rel = (bottom_rate_y - top_left_pos.y0) / top_left_pos.height
            new_height = target_span / (top_rate_rel - bottom_rate_rel)
            scale = new_height / top_left_pos.height
            new_width = top_left_pos.width * scale
            new_y0 = xlabel_y - bottom_rate_rel * new_height
            ax_top_left.set_position([
                top_left_pos.x0,
                new_y0,
                new_width,
                new_height
            ])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

    top_right_ticklabel_right = max(
        ticklabel.get_window_extent(renderer).transformed(inv_fig).x1
        for ticklabel in ax_top_right.get_yticklabels()
        if ticklabel.get_visible()
    )
    right_axes_edge = ax_bottom_right.get_position().x1
    top_right_pos = ax_top_right.get_position()
    ax_top_right.set_position([
        top_right_pos.x0,
        top_right_pos.y0,
        top_right_pos.width - (top_right_ticklabel_right - right_axes_edge),
        top_right_pos.height
    ])
    top_left_pos = ax_top_left.get_position()
    top_right_pos = ax_top_right.get_position()
    top_row_gap = top_column_wspace * (top_left_pos.width + top_right_pos.width) / 2
    new_top_right_x0 = top_left_pos.x1 + top_row_gap
    ax_top_right.set_position([
        new_top_right_x0,
        top_right_pos.y0,
        top_right_pos.x1 - new_top_right_x0,
        top_right_pos.height
    ])

    ### panels (c), (d), (e) ###
    ax_middle_left.plot(
        plot_lags,
        Us_true.reshape(Us_true.shape[0], -1).cpu(),
        c='C1',
        linewidth=0.5
    )
    # ax_middle_left.axvline(x=N_LAGS_TRUNC, color='k', linestyle='--', linewidth=0.5, label='Cutoff')
    # ax_middle_left.legend(fontsize=6)
    ax_middle_left.set_xticks(plot_lags)
    ax_middle_left.set_xlim(plot_lags[0], plot_lags[-1])
    ax_middle_left.set_ylim(0.0, 1.0)
    ax_middle_left.set_xlabel(r'Lagtime $n\tau$', fontsize=8)
    ax_middle_left.set_ylabel(r'$U_{ij}^{(n)}$', fontsize=8)

    ax_middle_center.plot(
        plot_lags[:-1],
        Ks_diagrammatic.reshape(Ks_diagrammatic.shape[0], -1).cpu(),
        c='C0',
        linestyle=':',
        linewidth=0.5,
        label='Diagrammatic'
    )
    ax_middle_center.plot(
        plot_lags[:-1],
        Ks_analytic.reshape(Ks_analytic.shape[0], -1).cpu(),
        c='C0',
        alpha=0.5,
        linewidth=0.5,
        label='Analytic'
    )
    ax_middle_center.set_yscale('log')
    # ax_middle_center.axvline(x=N_LAGS_TRUNC - 1, color='k', linestyle='--', linewidth=0.5, label='Cutoff')
    _legend_unique(ax_middle_center, fontsize=6, handlelength=1.0)
    ax_middle_center.set_xticks(plot_lags[:-1])
    ax_middle_center.set_yticks([1e-1, 1e-5, 1e-9])
    ax_middle_center.set_xlim(plot_lags[0], plot_lags[-2])
    ax_middle_center.set_xlabel(r'Delay $n\tau$', fontsize=8)
    ax_middle_center.set_ylabel(r'$K_{ij}^{(n)}$', fontsize=8)

    ax_middle_right.plot(
        plot_lags,
        Gs_true.reshape(Gs_true.shape[0], -1).cpu(),
        c='C0',
        linewidth=0.5
    )
    ax_middle_right.axvline(
        x=N_LAGS_TRUNC - 1, color='k', linestyle='--', linewidth=0.5, label='Cutoff'
    )
    ax_middle_right.legend(fontsize=6, handlelength=1.0)
    ax_middle_right.set_xticks(plot_lags)
    ax_middle_right.set_xlim(plot_lags[0], plot_lags[-1])
    ax_middle_right.set_xlabel(r'Lagtime $n\tau$', fontsize=8)
    ax_middle_right.set_ylabel(r'$G_{ij}^{(n)}$', fontsize=8)

    ### figure realignment ###
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    if left_outer_rate_label is not None:
        left_rate_bbox = left_outer_rate_label.get_window_extent(renderer).transformed(inv_fig)
        target_ticklabels = [
            ticklabel for ticklabel in ax_middle_left.get_yticklabels()
            if ticklabel.get_visible()
        ]
        if target_ticklabels:
            target_bbox = max(
                (
                    ticklabel.get_window_extent(renderer).transformed(inv_fig)
                    for ticklabel in target_ticklabels
                ),
                key=lambda bbox: (bbox.y0 + bbox.y1) / 2
            )
            left_rate_x = left_rate_bbox.x0
            target_x = target_bbox.x0
            top_left_pos = ax_top_left.get_position()
            ax_top_left.set_position([
                top_left_pos.x0 + target_x - left_rate_x,
                top_left_pos.y0,
                top_left_pos.width,
                top_left_pos.height
            ])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_right_pos_for_center = ax_top_right.get_position().frozen()
    middle_center_tick_bboxes = [
        ticklabel.get_window_extent(renderer).transformed(inv_fig)
        for ticklabel in ax_middle_center.get_yticklabels()
        if ticklabel.get_visible()
    ]
    trace_tick_bboxes = [
        ticklabel.get_window_extent(renderer).transformed(inv_fig)
        for ticklabel in ax_top_right.get_yticklabels()
        if ticklabel.get_visible()
    ]
    trace_pos = ax_top_right.get_position()
    trace_x0 = min(bbox.x0 for bbox in middle_center_tick_bboxes) + 0.006
    diagram_x0 = ax_middle_right.get_position().x0
    diagram_x1 = ax_middle_right.get_position().x1
    trace_x1 = ax_middle_center.get_position().x1
    ax_top_right.set_position([
        trace_x0,
        trace_pos.y0,
        trace_x1 - trace_x0,
        trace_pos.height
    ])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_right_ticklabel_right = max(
        ticklabel.get_window_extent(renderer).transformed(inv_fig).x1
        for ticklabel in ax_top_right.get_yticklabels()
        if ticklabel.get_visible()
    )
    right_axes_edge = ax_bottom_right.get_position().x1
    top_right_pos = ax_top_right.get_position()
    diagram_width = diagram_x1 - diagram_x0
    ax_top_right.set_xticks(np.arange(0, 101, 25))
    ax_top_diagram = fig.add_axes([
        diagram_x0,
        top_right_pos.y0,
        diagram_width,
        top_right_pos.height
    ])
    diagram_artists = _draw_compact_diagrams(ax_top_diagram)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    trace_xlabel_bbox = ax_top_right.xaxis.label.get_window_extent(renderer).transformed(inv_fig)
    trace_xlabel_y = (trace_xlabel_bbox.y0 + trace_xlabel_bbox.y1) / 2
    panel_a_y_for_diagram = (
        top_outer_rate_label.get_window_extent(renderer).transformed(inv_fig).y0
        + top_outer_rate_label.get_window_extent(renderer).transformed(inv_fig).y1
    ) / 2
    diagram_pos = ax_top_diagram.get_position()
    diagram_bottom_label_y = np.mean([
        (
            label.get_window_extent(renderer).transformed(inv_fig).y0
            + label.get_window_extent(renderer).transformed(inv_fig).y1
        ) / 2
        for label in diagram_artists['bottom_labels']
    ])
    diagram_time_label_bbox = diagram_artists['time_label'].get_window_extent(renderer).transformed(inv_fig)
    diagram_time_label_y = (diagram_time_label_bbox.y0 + diagram_time_label_bbox.y1) / 2
    bottom_label_rel_y = (diagram_bottom_label_y - diagram_pos.y0) / diagram_pos.height
    time_label_rel_y = (diagram_time_label_y - diagram_pos.y0) / diagram_pos.height
    diagram_height = (
        (panel_a_y_for_diagram - trace_xlabel_y)
        / (time_label_rel_y - bottom_label_rel_y)
    )
    diagram_y0 = trace_xlabel_y - bottom_label_rel_y * diagram_height
    ax_top_diagram.set_position([
        diagram_pos.x0,
        diagram_y0,
        diagram_pos.width,
        diagram_height
    ])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    trace_tick_bboxes = [
        ticklabel.get_window_extent(renderer).transformed(inv_fig)
        for ticklabel in ax_top_right.get_yticklabels()
        if ticklabel.get_visible()
    ]
    longest_trace_tick_bbox = max(
        trace_tick_bboxes,
        key=lambda bbox: bbox.x1 - bbox.x0
    )
    trace_pos = ax_top_right.get_position()
    macrostate_ylabel = fig.text(
        longest_trace_tick_bbox.x1 - 0.021,
        trace_pos.y0 + 0.42 * trace_pos.height,
        'Macrostate',
        fontsize=8,
        ha='center',
        va='center',
        rotation=270,
        rotation_mode='anchor'
    )

    ### panel (f) ###
    ax_center.axis('off')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    cycle_bbox = ax_top_left.get_window_extent(renderer).transformed(inv_fig)
    center_bbox = ax_center.get_window_extent(renderer).transformed(inv_fig)
    fig_width, fig_height = fig.get_size_inches()
    heptagon_center = (
        (cycle_bbox.x0 + cycle_bbox.x1) / 2,
        center_bbox.y0 + 0.22 * (center_bbox.y1 - center_bbox.y0)
    )
    harpoon_start = ax_top_left.transData.transform(
        (outer_trim, side_length + outer_offset)
    )
    harpoon_end = ax_top_left.transData.transform(
        (side_length - outer_trim, side_length + outer_offset)
    )
    harpoon_length = np.linalg.norm(harpoon_end - harpoon_start) / fig.dpi
    heptagon_radius = 0.75 * harpoon_length / (2 * np.sin(np.pi / 7))
    heptagon_angles = np.pi / 2 + 2 * np.pi * np.arange(7) / 7
    heptagon_x_offsets = heptagon_radius * np.cos(heptagon_angles) / fig_width
    heptagon_y_offsets = heptagon_radius * np.sin(heptagon_angles) / fig_height
    heptagon_label_gap = 0.018
    heptagon_label = fig.text(
        heptagon_center[0],
        heptagon_center[1],
        r'$T_{\hat{\Pi}\to\hat{\Pi}}\subsetneq\mathbb{R}^{3\times 3}$',
        fontsize=8,
        ha='center',
        va='center'
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bbox = heptagon_label.get_window_extent(renderer).transformed(inv_fig)
    label_height = label_bbox.y1 - label_bbox.y0
    middle_xlabel_bbox = ax_middle_left.xaxis.label.get_window_extent(renderer).transformed(inv_fig)
    heptagon_center = (
        heptagon_center[0] - 0.035,
        (
            middle_xlabel_bbox.y0 + ax_bottom_left.get_position().y1
            - heptagon_y_offsets.max() - heptagon_y_offsets.min()
            - heptagon_label_gap - label_height
        ) / 2
    )
    heptagon_label_y = (
        heptagon_center[1] + heptagon_y_offsets.max()
        + heptagon_label_gap + label_height / 2
    )
    heptagon_label.set_position((heptagon_center[0], heptagon_label_y))
    heptagon_vertices = np.column_stack([
        heptagon_center[0] + heptagon_x_offsets,
        heptagon_center[1] + heptagon_y_offsets
    ])
    heptagon_bottom = heptagon_vertices[:, 1].min()
    fig.patches.append(Polygon(
        heptagon_vertices,
        closed=True,
        transform=fig.transFigure,
        facecolor=plt.get_cmap('GnBu')(0.8),
        edgecolor='none',
        alpha=0.25,
        zorder=0.5
    ))

    ellipse_x0 = top_right_pos_for_center.x0
    ellipse_x1 = top_right_pos_for_center.x1
    page_right_edge = max(right_axes_edge, top_right_ticklabel_right)
    ellipse_x1 += 0.5 * (page_right_edge - ellipse_x1)
    ellipse_width = ellipse_x1 - ellipse_x0
    gradient_x = (ax_middle_right.get_position().x0 - ellipse_x0) / ellipse_width
    ellipse_x1 = ax_middle_right.get_position().x1
    ellipse_x0 = ellipse_x1 - ellipse_width
    ellipse_height = heptagon_center[1] - heptagon_bottom
    ellipse_y0 = heptagon_bottom
    ellipse_y1 = ellipse_y0 + ellipse_height
    ellipse_ax = fig.add_axes(
        [ellipse_x0, ellipse_y0, ellipse_width, ellipse_height],
        zorder=1
    )
    ellipse_ax.patch.set_alpha(0)
    ellipse_ax.axis('off')
    gradient_n = 256
    xx, yy = np.meshgrid(
        np.linspace(0, 1, gradient_n),
        np.linspace(0, 1, gradient_n)
    )
    ellipse_width_in = ellipse_width * fig_width
    ellipse_height_in = ellipse_height * fig_height
    gradient_y = 0.5
    asterisk_x = gradient_x
    asterisk_y = 0.4
    vertical_gradient_weight = 3.5
    distance_from_gradient_center = np.sqrt(
        ((xx - gradient_x) * ellipse_width_in) ** 2
        + (vertical_gradient_weight * (yy - gradient_y) * ellipse_height_in) ** 2
    )
    distance_to_right_vertex = np.sqrt(
        ((1 - gradient_x) * ellipse_width_in) ** 2
        + (vertical_gradient_weight * (0.5 - gradient_y) * ellipse_height_in) ** 2
    )
    gradient = np.clip(
        0.8 * distance_from_gradient_center / distance_to_right_vertex,
        0,
        0.8
    )

    heptagon_anchor = np.array(heptagon_center)
    ellipse_center_fig = np.array([
        (ellipse_x0 + ellipse_x1) / 2,
        (ellipse_y0 + ellipse_y1) / 2
    ])
    ellipse_center_in = np.array([
        ellipse_center_fig[0] * fig_width,
        ellipse_center_fig[1] * fig_height
    ])
    ellipse_a = ellipse_width_in / 2
    ellipse_b = ellipse_height_in / 2
    heptagon_anchor_in = np.array([
        heptagon_anchor[0] * fig_width,
        heptagon_anchor[1] * fig_height
    ])
    point_rel = heptagon_anchor_in - ellipse_center_in
    tangent_a = point_rel[0] / ellipse_a
    tangent_b = point_rel[1] / ellipse_b
    tangent_r = np.hypot(tangent_a, tangent_b)
    tangent_phi = np.arctan2(tangent_b, tangent_a)
    tangent_angles = tangent_phi + np.array([1, -1]) * np.arccos(1 / tangent_r)
    ellipse_tangent_points = np.column_stack([
        (ellipse_center_in[0] + ellipse_a * np.cos(tangent_angles)) / fig_width,
        (ellipse_center_in[1] + ellipse_b * np.sin(tangent_angles)) / fig_height
    ])
    ellipse_lower_tangent_point = ellipse_tangent_points[
        np.argmin(ellipse_tangent_points[:, 1])
    ]
    ellipse_top_point = np.array([
        ellipse_center_fig[0],
        ellipse_y1
    ])
    fig.patches.append(Polygon(
        [heptagon_anchor, ellipse_lower_tangent_point, ellipse_top_point],
        closed=True,
        transform=fig.transFigure,
        facecolor='black',
        edgecolor='none',
        alpha=0.25,
        zorder=0.7
    ))

    ellipse_clip = Ellipse(
        (0.5, 0.5),
        width=1.0,
        height=1.0,
        transform=ellipse_ax.transAxes
    )
    ellipse_image = ellipse_ax.imshow(
        gradient,
        origin='lower',
        extent=(0, 1, 0, 1),
        cmap='GnBu',
        interpolation='bilinear',
        aspect='auto',
        vmin=0,
        vmax=1
    )
    ellipse_image.set_clip_path(ellipse_clip)
    ellipse_ax.text(
        asterisk_x,
        asterisk_y,
        '*',
        color='black',
        fontsize=16,
        ha='center',
        va='center',
        transform=ellipse_ax.transAxes
    )
    eta_inf_pos = (ellipse_x1, heptagon_label_y)
    eta_inf_arrow_end = (eta_inf_pos[0], eta_inf_pos[1] - 0.012)
    fig.text(
        *eta_inf_pos,
        r'$\eta\to\infty$',
        fontsize=8,
        ha='right',
        va='center',
        color='k'
    )
    gradient_center_fig = (
        ellipse_x0 + gradient_x * ellipse_width,
        ellipse_y0 + gradient_y * ellipse_height
    )
    ellipse_right_vertex = (ellipse_x1, ellipse_y0 + 0.5 * ellipse_height)
    dot_radius = 0.004
    intermediate_dot_positions = []
    for dot_fraction in [1 / 6, 1 / 3]:
        intermediate_dot_pos = (
            gradient_center_fig[0]
            + dot_fraction * (ellipse_right_vertex[0] - gradient_center_fig[0]),
            gradient_center_fig[1]
            + dot_fraction * (ellipse_right_vertex[1] - gradient_center_fig[1])
        )
        intermediate_dot_positions.append(np.array(intermediate_dot_pos))
        fig.patches.append(Circle(
            intermediate_dot_pos,
            radius=dot_radius,
            transform=fig.transFigure,
            facecolor='none',
            edgecolor='black',
            linewidth=0.5,
            zorder=2
        ))
    dot_pos = (
        ellipse_x0 + (ellipse_x1 - gradient_center_fig[0]),
        gradient_center_fig[1]
    )
    fig.patches.append(Circle(
        dot_pos,
        radius=dot_radius,
        transform=fig.transFigure,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        zorder=2
    ))
    eta_zero_label = fig.text(
        dot_pos[0] - 0.012,
        dot_pos[1],
        r'$\eta=0$',
        fontsize=8,
        ha='right',
        va='center',
        color='k'
    )
    curve_t = np.linspace(0, 1, 100)
    eta_zero_dot_direction = np.array(eta_inf_arrow_end) - np.array(dot_pos)
    eta_zero_dot_direction /= np.linalg.norm(eta_zero_dot_direction)
    eta_zero_edge_pos = np.array(dot_pos) + dot_radius * eta_zero_dot_direction
    curve_start = eta_zero_edge_pos
    curve_end = curve_start + 0.990 * (np.array(eta_inf_arrow_end) - curve_start)
    curve_direction = curve_end - curve_start
    curve_normal = np.array([-curve_direction[1], curve_direction[0]])
    curve_normal /= np.linalg.norm(curve_normal)
    curve = (
        curve_start
        + curve_t[:, None] * curve_direction
        + 0.012 * np.sin(2 * np.pi * curve_t)[:, None] * curve_normal
    )
    curve[0] = curve_start
    curve[-1] = curve_end
    fig.lines.append(Line2D(
        curve[:, 0],
        curve[:, 1],
        color='C6',
        linewidth=0.5,
        transform=fig.transFigure,
        zorder=3.8
    ))
    curve_label_t = 0.55
    curve_label_point = (
        curve_start
        + curve_label_t * curve_direction
        + 0.012 * np.sin(2 * np.pi * curve_label_t) * curve_normal
        + 0.010 * curve_normal
        + np.array([0.0, 0.006])
    )
    fig.text(
        curve_label_point[0],
        curve_label_point[1],
        r'$-\nabla f$',
        fontsize=8,
        ha='right',
        va='center',
        color='k'
    )
    eta_arrow_tip = np.array(eta_inf_arrow_end)
    eta_arrow_direction = eta_arrow_tip - eta_zero_edge_pos
    eta_arrow_base = _add_filled_arrowhead(
        fig,
        eta_arrow_tip,
        eta_arrow_direction,
        color='C6',
        zorder=4.1
    )
    fig.lines.append(Line2D(
        [eta_zero_edge_pos[0], eta_arrow_base[0]],
        [eta_zero_edge_pos[1], eta_arrow_base[1]],
        color='C6',
        linewidth=0.5,
        linestyle=':',
        transform=fig.transFigure,
        zorder=4
    ))
    curved_connector_start = curve[int(2 / 3 * (len(curve) - 1))]
    fig.patches.append(FancyArrowPatch(
        curved_connector_start,
        intermediate_dot_positions[0],
        transform=fig.transFigure,
        arrowstyle='->',
        mutation_scale=8,
        color='k',
        linewidth=0.5,
        linestyle='-',
        connectionstyle='arc3,rad=-0.35',
        zorder=4.2,
        clip_on=False,
        shrinkA=0,
        shrinkB=2
    ))
    dot_to_eta = np.array(eta_inf_arrow_end) - eta_zero_edge_pos
    dotted_connector_t = (
        (intermediate_dot_positions[1][0] - eta_zero_edge_pos[0]) / dot_to_eta[0]
    )
    dotted_connector_start = np.array([
        intermediate_dot_positions[1][0],
        eta_zero_edge_pos[1] + dotted_connector_t * dot_to_eta[1]
    ])
    fig.patches.append(FancyArrowPatch(
        dotted_connector_start,
        intermediate_dot_positions[1],
        transform=fig.transFigure,
        arrowstyle='->',
        mutation_scale=8,
        color='k',
        linewidth=0.5,
        linestyle=':',
        zorder=4.2,
        clip_on=False,
        shrinkA=0,
        shrinkB=2
    ))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    eta_zero_bbox = eta_zero_label.get_window_extent(renderer).transformed(inv_fig)
    colorbar_center_x = (eta_zero_bbox.x0 + eta_zero_bbox.x1) / 2 - 0.1
    colorbar_title = fig.text(
        colorbar_center_x,
        heptagon_label_y,
        r'$\leftarrow$ Lower negative loglikelihood $f$',
        fontsize=8,
        ha='center',
        va='center'
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = colorbar_title.get_window_extent(renderer).transformed(inv_fig)
    colorbar_width = max(title_bbox.x1 - title_bbox.x0 + 0.012, 0.19)
    colorbar_x0 = colorbar_center_x - colorbar_width / 2
    colorbar_x1 = colorbar_center_x + colorbar_width / 2
    colorbar_y0 = heptagon_label_y - 0.023
    colorbar_height = 0.009
    colorbar_ax = fig.add_axes(
        [colorbar_x0, colorbar_y0, colorbar_x1 - colorbar_x0, colorbar_height],
        zorder=2
    )
    colorbar_ax.imshow(
        np.linspace(0, 0.8, 256)[None, :],
        cmap='GnBu',
        aspect='auto',
        vmin=0,
        vmax=1
    )
    colorbar_ax.axis('off')
    legend_handles = [
        FancyArrowPatch((0, 0), (1, 0), arrowstyle='-|>', facecolor='C6', edgecolor='C6', linewidth=0.5, linestyle='-'),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle='-|>', facecolor='C6', edgecolor='C6', linewidth=0.5, linestyle=':'),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle='->', color='k', linewidth=0.5, linestyle='-'),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle='->', color='k', linewidth=0.5, linestyle=':'),
    ]
    for handle in legend_handles[:2]:
        handle._filled_head_only = True
    legend_labels = [
        r'$D_{\mathrm{KL}}$ mirror update',
        r'$L^2$ gradient update',
        r'$D_{\mathrm{KL}}$ projection',
        r'$L^2$ projection',
    ]
    schematic_legend = fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(colorbar_center_x, colorbar_y0 - 0.006),
        ncol=2,
        fontsize=8,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.3,
        borderaxespad=0,
        handler_map={FancyArrowPatch: _HandlerArrow()}
    )
    for text in schematic_legend.get_texts()[:2]:
        text.set_color('C6')

    for i in range(len(heptagon_vertices)):
        for j in range(i + 1, len(heptagon_vertices)):
            fig.lines.append(Line2D(
                [heptagon_vertices[i, 0], heptagon_vertices[j, 0]],
                [heptagon_vertices[i, 1], heptagon_vertices[j, 1]],
                color='black',
                linewidth=0.5,
                transform=fig.transFigure,
                zorder=0.4
            ))

    ### panels (g), (h), (i) ###
    ax_bottom_left.set_xscale('log')
    ax_bottom_left.errorbar(
        N_STEPS, err_U_means[0], student_CI_ppf_one_sided * err_U_stderrs[0],
        color='C1', linewidth=0.5, linestyle='-', capsize=1, label=r'$n=1$ (0.95 CI)'
    )
    ax_bottom_left.errorbar(
        N_STEPS, err_U_means[1], student_CI_ppf_one_sided * err_U_stderrs[1],
        color='C1', linewidth=0.5, linestyle='--', capsize=1, label=r'$n=2$ (0.95 CI)'
    )
    ax_bottom_left.errorbar(
        N_STEPS, err_U_means[2], student_CI_ppf_one_sided * err_U_stderrs[2],
        color='C1', linewidth=0.5, linestyle=':', capsize=1, label=r'$n=3$ (0.95 CI)'
    )
    ax_bottom_left.set_xlim(N_STEPS[0], N_STEPS[-1])
    ax_bottom_left.set_xticks(N_STEPS)
    ax_bottom_left.set_xlabel(r'Observation time $N\tau$', fontsize=8)
    ax_bottom_left.set_ylabel(r'$\|\hat{U}^{(n)}-U^{(n)}\|_2$', fontsize=8)
    ax_bottom_left.legend(fontsize=6, handlelength=1.0)

    ax_bottom_center.set_xscale('log')
    ax_bottom_center.errorbar(
        N_STEPS, err_K_means[0], student_CI_ppf_one_sided * err_K_stderrs[0],
        color='C0', linewidth=0.5, linestyle='-', capsize=1, label=r'$n=0$ (0.95 CI)'
    )
    ax_bottom_center.errorbar(
        N_STEPS, err_K_means[1], student_CI_ppf_one_sided * err_K_stderrs[1],
        color='C0', linewidth=0.5, linestyle='--', capsize=1, label=r'$n=1$ (0.95 CI)'
    )
    ax_bottom_center.errorbar(
        N_STEPS, err_K_means[2], student_CI_ppf_one_sided * err_K_stderrs[2],
        color='C0', linewidth=0.5, linestyle=':', capsize=1, label=r'$n=2$ (0.95 CI)'
    )
    ax_bottom_center.set_xlim(N_STEPS[0], N_STEPS[-1])
    ax_bottom_center.set_ylim(0.0, 1.0)
    ax_bottom_center.set_xticks(N_STEPS)
    ax_bottom_center.set_xlabel(r'Observation time $N\tau$', fontsize=8)
    ax_bottom_center.set_ylabel(r'$\|\hat{K}^{(n)}-K^{(n)}\|_2$', fontsize=8)
    ax_bottom_center.legend(fontsize=6, handlelength=1.0)

    ax_bottom_right.set_xscale('log')
    ax_bottom_right.errorbar(
        N_STEPS, err_G_means[0], student_CI_ppf_one_sided * err_G_stderrs[0],
        color='C0', linewidth=0.5, linestyle='-', capsize=1, label=r'$n=1$ (0.95 CI)'
    )
    ax_bottom_right.errorbar(
        N_STEPS, err_G_means[1], student_CI_ppf_one_sided * err_G_stderrs[1],
        color='C0', linewidth=0.5, linestyle='--', capsize=1, label=r'$n=2$ (0.95 CI)'
    )
    ax_bottom_right.errorbar(
        N_STEPS, err_G_means[2], student_CI_ppf_one_sided * err_G_stderrs[2],
        color='C0', linewidth=0.5, linestyle=':', capsize=1, label=r'$n=3$ (0.95 CI)'
    )
    ax_bottom_right.set_xlim(N_STEPS[0], N_STEPS[-1])
    ax_bottom_right.set_xticks(N_STEPS)
    ax_bottom_right.set_xlabel(r'Observation time $N\tau$', fontsize=8)
    ax_bottom_right.set_ylabel(r'$\|\hat{G}^{(n)}-G^{(n)}\|_2$', fontsize=8)
    ax_bottom_right.legend(fontsize=6, handlelength=1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()

    def _bbox_center(bbox):
        return (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2

    def _ylabel_x(ax):
        bbox = ax.yaxis.label.get_window_extent(renderer).transformed(inv_fig)
        return _bbox_center(bbox)[0]

    for bottom_ax, middle_ax in [
        (ax_bottom_left, ax_middle_left),
        (ax_bottom_center, ax_middle_center),
        (ax_bottom_right, ax_middle_right),
    ]:
        bottom_ax.yaxis.set_label_coords(
            _ylabel_x(middle_ax),
            bottom_ax.get_position().y0,
            transform=fig.transFigure
        )
        bottom_ax.yaxis.label.set_ha('left')
        bottom_ax.yaxis.label.set_va('center')

    ### panel annotations ###
    panel_label_kwargs = dict(fontsize=8, ha='center', va='center')

    def _top_ytick_y(ax):
        tick_bboxes = [
            ticklabel.get_window_extent(renderer).transformed(inv_fig)
            for ticklabel in ax.get_yticklabels()
            if ticklabel.get_visible()
        ]
        top_bbox = max(tick_bboxes, key=lambda bbox: _bbox_center(bbox)[1])
        return _bbox_center(top_bbox)[1]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()

    diagram_row_label_x = ax_middle_right.get_position().x0
    diagram_row_labels = []
    for row_y, row_label in [
        (3, r'$U^{(3)}$'),
        (2, r'$U^{(1)}$'),
        (1, r'$U^{(1)}$'),
    ]:
        row_fig_y = inv_fig.transform(
            ax_top_diagram.transData.transform((0, row_y))
        )[1]
        diagram_row_labels.append(fig.text(
            diagram_row_label_x,
            row_fig_y,
            row_label,
            fontsize=8,
            ha='left',
            va='center'
        ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for row_label in diagram_row_labels:
        row_label_bbox = row_label.get_window_extent(renderer).transformed(inv_fig)
        row_label_x, row_label_y = row_label.get_position()
        row_label.set_position((
            row_label_x + diagram_row_label_x - row_label_bbox.x0,
            row_label_y
        ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    macrostate_ylabel_bbox = macrostate_ylabel.get_window_extent(
        renderer
    ).transformed(inv_fig)
    top_row_label_bbox = diagram_row_labels[0].get_window_extent(
        renderer
    ).transformed(inv_fig)
    target_top_row_center_y = (
        macrostate_ylabel_bbox.y1
        - 0.5 * (top_row_label_bbox.y1 - top_row_label_bbox.y0)
        - 0.008
    )
    target_top_row_y = ax_top_diagram.transData.inverted().transform(
        fig.transFigure.transform((0, target_top_row_center_y))
    )[1]
    arrow_y = diagram_artists['time_label'].get_position()[1] - 1 / 6
    row_spacing = arrow_y - target_top_row_y
    if row_spacing > 0:
        dot_y = target_top_row_y - row_spacing * np.arange(4)
        _update_compact_diagram_ycoords(diagram_artists, dot_y)
        for row_label, row_y in zip(diagram_row_labels, dot_y[:3]):
            row_fig_y = inv_fig.transform(
                ax_top_diagram.transData.transform((0, row_y))
            )[1]
            row_label.set_position((row_label.get_position()[0], row_fig_y))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    u_row_label_right = max(
        row_label.get_window_extent(renderer).transformed(inv_fig).x1
        for row_label in diagram_row_labels
    )
    dot_x = diagram_artists['dot_x'].copy()
    right_panel_edge = ax_middle_right.get_position().x1
    k0_label = diagram_artists['bottom_labels'][1]
    _align_axes_text_right_to_fig_x(fig, ax_top_diagram, k0_label, right_panel_edge)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    k0_bbox = k0_label.get_window_extent(renderer).transformed(inv_fig)
    right_dot_fig_x = (k0_bbox.x0 + k0_bbox.x1) / 2
    left_dot_fig_x = u_row_label_right + 0.020
    dot_x[0] = ax_top_diagram.transData.inverted().transform(
        fig.transFigure.transform((left_dot_fig_x, 0))
    )[0]
    dot_x[2] = ax_top_diagram.transData.inverted().transform(
        fig.transFigure.transform((right_dot_fig_x, 0))
    )[0]
    dot_x[1] = dot_x[0] + 0.52 * (dot_x[2] - dot_x[0])
    _update_compact_diagram_xcoords(diagram_artists, dot_x)
    _align_axes_text_right_to_fig_x(fig, ax_top_diagram, k0_label, right_panel_edge)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    k0_bbox = k0_label.get_window_extent(renderer).transformed(inv_fig)
    right_dot_fig_x = (k0_bbox.x0 + k0_bbox.x1) / 2
    dot_x[2] = ax_top_diagram.transData.inverted().transform(
        fig.transFigure.transform((right_dot_fig_x, 0))
    )[0]
    dot_x[1] = dot_x[0] + 0.52 * (dot_x[2] - dot_x[0])
    _update_compact_diagram_xcoords(diagram_artists, dot_x)
    dot_y = diagram_artists.get('dot_y', np.array([3.0, 2.0, 1.0, 0.0]))
    upper_u1_row_y = inv_fig.transform(
        ax_top_diagram.transData.transform((0, dot_y[1]))
    )[1]
    macrostate_ylabel.set_position((
        macrostate_ylabel.get_position()[0],
        upper_u1_row_y
    ))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    _match_compact_diagram_arrow_to_time_label(fig, ax_top_diagram, diagram_artists)

    panel_a_x = _ylabel_x(ax_middle_left)
    panel_a_y = _bbox_center(top_outer_rate_label.get_window_extent(renderer).transformed(inv_fig))[1]
    panel_b_x = _ylabel_x(ax_middle_center)
    panel_b_y = panel_a_y
    right_panel_label_x = _ylabel_x(ax_middle_right)
    panel_c_x = diagram_row_label_x
    panel_c_y = panel_a_y
    panel_d_x = panel_a_x
    panel_d_y = _top_ytick_y(ax_middle_left)
    panel_e_x = panel_b_x
    panel_e_y = panel_d_y
    panel_f_x = right_panel_label_x
    panel_f_y = panel_d_y
    panel_g_x = panel_d_x
    panel_g_y = heptagon_label_y
    panel_h_x = panel_d_x
    panel_h_y = _top_ytick_y(ax_bottom_left)
    panel_i_x = panel_e_x
    panel_i_y = _top_ytick_y(ax_bottom_center)
    panel_j_x = right_panel_label_x
    panel_j_y = _top_ytick_y(ax_bottom_right)

    fig.text(panel_a_x, panel_a_y, '(a)', **panel_label_kwargs)
    fig.text(panel_b_x, panel_b_y, '(b)', **panel_label_kwargs)
    panel_c_label = fig.text(panel_c_x, panel_c_y, '(c)', fontsize=8, ha='left', va='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_c_bbox = panel_c_label.get_window_extent(renderer).transformed(inv_fig)
    panel_c_label.set_position((
        panel_c_label.get_position()[0] + panel_c_x - panel_c_bbox.x0,
        panel_c_label.get_position()[1]
    ))
    fig.text(panel_d_x, panel_d_y, '(d)', **panel_label_kwargs)
    fig.text(panel_e_x, panel_e_y, '(e)', **panel_label_kwargs)
    fig.text(panel_f_x, panel_f_y, '(f)', **panel_label_kwargs)
    fig.text(panel_g_x, panel_g_y, '(g)', **panel_label_kwargs)
    fig.text(panel_h_x, panel_h_y, '(h)', **panel_label_kwargs)
    fig.text(panel_i_x, panel_i_y, '(i)', **panel_label_kwargs)
    fig.text(panel_j_x, panel_j_y, '(j)', **panel_label_kwargs)

    plt.savefig(RES_DIR / SCHEME_FN, bbox_inches='tight') 

    ### ADVERSARIAL EXAMPLE ###
    trace_adv = hard_summary['trace']
    nmacro_adv = hard_summary['nmacro']
    Us_true_adv = hard_summary['Us_true']
    Ks_diagrammatic_adv = hard_summary['Ks_diagrammatic']
    Ks_analytic_adv = hard_summary['Ks_analytic']
    Gs_true_adv = hard_summary['Gs_true']
    plot_lags_adv = hard_summary['plot_lags']
    err_U_means_adv = hard_summary['err_U_means']
    err_U_stderrs_adv = hard_summary['err_U_stderrs']
    err_G_means_adv = hard_summary['err_G_means']
    err_G_stderrs_adv = hard_summary['err_G_stderrs']
    err_K_means_adv = hard_summary['err_K_means']
    err_K_stderrs_adv = hard_summary['err_K_stderrs']

    fig_adv = plt.figure(figsize=(6, 5.25))
    gs_adv = fig_adv.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    gs_adv_top = gs_adv[0].subgridspec(
        1, 2, width_ratios=[1.25, 2.05], wspace=top_column_wspace
    )
    gs_adv_middle = gs_adv[1].subgridspec(
        1, 3, width_ratios=[10, 9, 10], wspace=middle_column_wspace
    )
    gs_adv_bottom = gs_adv[2].subgridspec(
        1, 3, width_ratios=[1, 1, 1], wspace=bottom_column_wspace
    )

    ax_adv_top_left = fig_adv.add_subplot(gs_adv_top[0, 0])
    ax_adv_top_right = fig_adv.add_subplot(gs_adv_top[0, 1])
    ax_adv_middle_left = fig_adv.add_subplot(gs_adv_middle[0, 0])
    ax_adv_middle_center = fig_adv.add_subplot(gs_adv_middle[0, 1])
    ax_adv_middle_right = fig_adv.add_subplot(gs_adv_middle[0, 2], sharey=ax_adv_middle_left)
    ax_adv_bottom_left = fig_adv.add_subplot(gs_adv_bottom[0, 0])
    ax_adv_bottom_center = fig_adv.add_subplot(gs_adv_bottom[0, 1], sharey=ax_adv_bottom_left)
    ax_adv_bottom_right = fig_adv.add_subplot(gs_adv_bottom[0, 2], sharey=ax_adv_bottom_left)

    axes_adv = [
        ax_adv_top_left, ax_adv_top_right,
        ax_adv_middle_left, ax_adv_middle_center, ax_adv_middle_right,
        ax_adv_bottom_left, ax_adv_bottom_center, ax_adv_bottom_right
    ]
    for a in axes_adv:
        a.tick_params(axis='both', which='both', labelsize=8)
        for axis in ['right', 'top', 'left', 'bottom']:
            a.spines[axis].set_linewidth(0.6)
    fig_adv.subplots_adjust(hspace=0.5)

    top_outer_rate_label_adv = None
    left_outer_rate_label_adv = None
    bottom_outer_rate_label_adv = None
    for state_i, state_j, outside, label_offset, _ in outer_harpoons:
        start = state_pos[state_i]
        end = state_pos[state_j]
        outside = np.array(outside)
        direction = end - start
        direction /= np.linalg.norm(direction)
        harpoon_start = start + outer_offset * outside + outer_trim * direction
        harpoon_end = end + outer_offset * outside - outer_trim * direction
        _draw_harpoon(ax_adv_top_left, harpoon_start, harpoon_end, outside)
        rate_label = ax_adv_top_left.text(
            *((harpoon_start + harpoon_end) / 2 + label_offset * outside),
            '1',
            fontsize=8,
            ha='center',
            va='center'
        )
        if state_i == 'M1' and state_j == 'M2':
            top_outer_rate_label_adv = rate_label
        if state_i == 'M2' and state_j == 'M3':
            left_outer_rate_label_adv = rate_label
        if state_i == 'M3' and state_j == 'M4':
            bottom_outer_rate_label_adv = rate_label

    for state_i, state_j, inside, label_offset, _ in inner_harpoons:
        start = state_pos[state_i]
        end = state_pos[state_j]
        inside = np.array(inside)
        direction = end - start
        direction /= np.linalg.norm(direction)
        harpoon_start = start + inner_offset * inside + inner_trim * direction
        harpoon_end = end + inner_offset * inside - inner_trim * direction
        _draw_harpoon(ax_adv_top_left, harpoon_start, harpoon_end, inside)
        ax_adv_top_left.text(
            *((harpoon_start + harpoon_end) / 2 + label_offset * inside),
            r'$1/2$',
            fontsize=8,
            ha='center',
            va='center'
        )

    ax_adv_top_left.add_patch(Rectangle(
        (-circle_radius - box_x_pad, side_length - circle_radius - box_y_pad),
        side_length + 2 * (circle_radius + box_x_pad),
        2 * (circle_radius + box_y_pad),
        fill=False,
        edgecolor='black',
        linewidth=0.5,
        linestyle=':',
        zorder=1
    ))
    for state, pos in state_pos.items():
        ax_adv_top_left.add_patch(Circle(
            pos,
            radius=circle_radius,
            facecolor='white',
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        ))
        ax_adv_top_left.text(
            *pos,
            cycle_state_labels[state],
            fontsize=8,
            ha='center',
            va='center',
            zorder=4
        )
    ax_adv_top_left.set_xlim(-x_plot_margin, side_length + x_plot_margin)
    ax_adv_top_left.set_ylim(-y_plot_margin, side_length + y_plot_margin)
    ax_adv_top_left.set_aspect('equal', adjustable='box')
    ax_adv_top_left.set_anchor('W')
    ax_adv_top_left.axis('off')

    ax_adv_top_right.plot(trace_adv, c='k')
    ax_adv_top_right.set_xlim(0, 100)
    ax_adv_top_right.set_xlabel('Time', fontsize=8)
    ax_adv_top_right.set_ylabel('Macrostate', fontsize=8)
    ax_adv_top_right.set_xticks(np.arange(11) * 10)
    ax_adv_top_right.set_yticks(np.arange(nmacro_adv))
    ax_adv_top_right.yaxis.tick_right()
    ax_adv_top_right.set_yticklabels([
        state_labels['M1'], state_labels['M2'], r'$\{\mu_3,\mu_4\}$'
    ])

    fig_adv.canvas.draw()
    renderer_adv = fig_adv.canvas.get_renderer()
    inv_fig_adv = fig_adv.transFigure.inverted()
    top_rate_bbox_adv = top_outer_rate_label_adv.get_window_extent(renderer_adv).transformed(inv_fig_adv)
    bottom_rate_bbox_adv = bottom_outer_rate_label_adv.get_window_extent(renderer_adv).transformed(inv_fig_adv)
    top_macro_bbox_adv = max(
        (
            ticklabel.get_window_extent(renderer_adv).transformed(inv_fig_adv)
            for ticklabel in ax_adv_top_right.get_yticklabels()
            if ticklabel.get_visible()
        ),
        key=lambda bbox: (bbox.y0 + bbox.y1) / 2
    )
    xlabel_bbox_adv = ax_adv_top_right.xaxis.label.get_window_extent(renderer_adv).transformed(inv_fig_adv)
    top_rate_y_adv = (top_rate_bbox_adv.y0 + top_rate_bbox_adv.y1) / 2
    bottom_rate_y_adv = (bottom_rate_bbox_adv.y0 + bottom_rate_bbox_adv.y1) / 2
    top_macro_y_adv = (top_macro_bbox_adv.y0 + top_macro_bbox_adv.y1) / 2
    xlabel_y_adv = (xlabel_bbox_adv.y0 + xlabel_bbox_adv.y1) / 2
    top_left_pos_adv = ax_adv_top_left.get_position()
    rate_span_adv = top_rate_y_adv - bottom_rate_y_adv
    target_span_adv = top_macro_y_adv - xlabel_y_adv
    if rate_span_adv > 0 and target_span_adv > 0:
        top_rate_rel_adv = (top_rate_y_adv - top_left_pos_adv.y0) / top_left_pos_adv.height
        bottom_rate_rel_adv = (bottom_rate_y_adv - top_left_pos_adv.y0) / top_left_pos_adv.height
        new_height_adv = target_span_adv / (top_rate_rel_adv - bottom_rate_rel_adv)
        scale_adv = new_height_adv / top_left_pos_adv.height
        new_width_adv = top_left_pos_adv.width * scale_adv
        new_y0_adv = xlabel_y_adv - bottom_rate_rel_adv * new_height_adv
        ax_adv_top_left.set_position([
            top_left_pos_adv.x0,
            new_y0_adv,
            new_width_adv,
            new_height_adv
        ])
        fig_adv.canvas.draw()
        renderer_adv = fig_adv.canvas.get_renderer()

    top_right_ticklabel_right_adv = max(
        ticklabel.get_window_extent(renderer_adv).transformed(inv_fig_adv).x1
        for ticklabel in ax_adv_top_right.get_yticklabels()
        if ticklabel.get_visible()
    )
    right_axes_edge_adv = ax_adv_bottom_right.get_position().x1
    top_right_pos_adv = ax_adv_top_right.get_position()
    ax_adv_top_right.set_position([
        top_right_pos_adv.x0,
        top_right_pos_adv.y0,
        top_right_pos_adv.width - (top_right_ticklabel_right_adv - right_axes_edge_adv),
        top_right_pos_adv.height
    ])
    top_left_pos_adv = ax_adv_top_left.get_position()
    top_right_pos_adv = ax_adv_top_right.get_position()
    top_row_gap_adv = top_column_wspace * (top_left_pos_adv.width + top_right_pos_adv.width) / 2
    new_top_right_x0_adv = top_left_pos_adv.x1 + top_row_gap_adv
    ax_adv_top_right.set_position([
        new_top_right_x0_adv,
        top_right_pos_adv.y0,
        top_right_pos_adv.x1 - new_top_right_x0_adv,
        top_right_pos_adv.height
    ])

    ax_adv_middle_left.plot(
        plot_lags_adv,
        Us_true_adv.reshape(Us_true_adv.shape[0], -1).cpu(),
        c='C1',
        linewidth=0.5
    )
    ax_adv_middle_left.set_xticks(plot_lags_adv)
    ax_adv_middle_left.set_xlim(plot_lags_adv[0], plot_lags_adv[-1])
    ax_adv_middle_left.set_ylim(0.0, 1.0)
    ax_adv_middle_left.set_xlabel(r'Lagtime $n\tau$', fontsize=8)
    ax_adv_middle_left.set_ylabel(r'$U_{ij}^{(n)}$', fontsize=8)

    ax_adv_middle_center.plot(
        plot_lags_adv[:-1],
        Ks_diagrammatic_adv.reshape(Ks_diagrammatic_adv.shape[0], -1).cpu(),
        c='C0',
        linestyle=':',
        linewidth=0.5,
        label='Diagrammatic'
    )
    ax_adv_middle_center.plot(
        plot_lags_adv[:-1],
        Ks_analytic_adv.reshape(Ks_analytic_adv.shape[0], -1).cpu(),
        c='C0',
        alpha=0.5,
        linewidth=0.5,
        label='Analytic'
    )
    ax_adv_middle_center.set_yscale('log')
    # ax_adv_middle_center.axvline(
    #     x=N_LAGS_TRUNC - 1, color='k', linestyle='--', linewidth=0.5, label='Cutoff'
    # )
    _legend_unique(ax_adv_middle_center, fontsize=6, handlelength=1.0)
    ax_adv_middle_center.set_xticks(plot_lags_adv[:-1])
    ax_adv_middle_center.set_yticks([1e-1, 1e-5, 1e-9])
    ax_adv_middle_center.set_xlim(plot_lags_adv[0], plot_lags_adv[-2])
    ax_adv_middle_center.set_xlabel(r'Delay $n\tau$', fontsize=8)
    ax_adv_middle_center.set_ylabel(r'$K_{ij}^{(n)}$', fontsize=8)

    ax_adv_middle_right.plot(
        plot_lags_adv,
        Gs_true_adv.reshape(Gs_true_adv.shape[0], -1).cpu(),
        c='C0',
        linewidth=0.5
    )
    ax_adv_middle_right.axvline(
        x=N_LAGS_TRUNC, color='k', linestyle='--', linewidth=0.5, label='Cutoff'
    )
    ax_adv_middle_right.legend(fontsize=6, handlelength=1.0)
    ax_adv_middle_right.set_xticks(plot_lags_adv)
    ax_adv_middle_right.set_xlim(plot_lags_adv[0], plot_lags_adv[-1])
    ax_adv_middle_right.set_xlabel(r'Lagtime $n\tau$', fontsize=8)
    ax_adv_middle_right.set_ylabel(r'$G_{ij}^{(n)}$', fontsize=8)

    fig_adv.canvas.draw()
    renderer_adv = fig_adv.canvas.get_renderer()
    inv_fig_adv = fig_adv.transFigure.inverted()
    left_rate_bbox_adv = left_outer_rate_label_adv.get_window_extent(renderer_adv).transformed(inv_fig_adv)
    target_ticklabels_adv = [
        ticklabel for ticklabel in ax_adv_middle_left.get_yticklabels()
        if ticklabel.get_visible()
    ]
    if target_ticklabels_adv:
        target_bbox_adv = max(
            (
                ticklabel.get_window_extent(renderer_adv).transformed(inv_fig_adv)
                for ticklabel in target_ticklabels_adv
            ),
            key=lambda bbox: (bbox.y0 + bbox.y1) / 2
        )
        top_left_pos_adv = ax_adv_top_left.get_position()
        ax_adv_top_left.set_position([
            top_left_pos_adv.x0 + target_bbox_adv.x0 - left_rate_bbox_adv.x0,
            top_left_pos_adv.y0,
            top_left_pos_adv.width,
            top_left_pos_adv.height
        ])

    fig_adv.canvas.draw()
    renderer_adv = fig_adv.canvas.get_renderer()
    top_right_ticklabel_right_adv = max(
        ticklabel.get_window_extent(renderer_adv).transformed(inv_fig_adv).x1
        for ticklabel in ax_adv_top_right.get_yticklabels()
        if ticklabel.get_visible()
    )
    right_axes_edge_adv = ax_adv_bottom_right.get_position().x1
    top_right_pos_adv = ax_adv_top_right.get_position()
    ax_adv_top_right.set_position([
        top_right_pos_adv.x0,
        top_right_pos_adv.y0,
        top_right_pos_adv.width - (top_right_ticklabel_right_adv - right_axes_edge_adv),
        top_right_pos_adv.height
    ])
    top_left_pos_adv = ax_adv_top_left.get_position()
    top_right_pos_adv = ax_adv_top_right.get_position()
    top_row_gap_adv = top_column_wspace * (top_left_pos_adv.width + top_right_pos_adv.width) / 2
    new_top_right_x0_adv = top_left_pos_adv.x1 + top_row_gap_adv
    ax_adv_top_right.set_position([
        new_top_right_x0_adv,
        top_right_pos_adv.y0,
        top_right_pos_adv.x1 - new_top_right_x0_adv,
        top_right_pos_adv.height
    ])

    ax_adv_bottom_left.set_xscale('log')
    ax_adv_bottom_left.errorbar(
        N_STEPS, err_U_means_adv[0], student_CI_ppf_one_sided * err_U_stderrs_adv[0],
        color='C1', linewidth=0.5, linestyle='-', capsize=1, label=r'$n=1$ (0.95 CI)'
    )
    ax_adv_bottom_left.errorbar(
        N_STEPS, err_U_means_adv[1], student_CI_ppf_one_sided * err_U_stderrs_adv[1],
        color='C1', linewidth=0.5, linestyle='--', capsize=1, label=r'$n=2$ (0.95 CI)'
    )
    ax_adv_bottom_left.errorbar(
        N_STEPS, err_U_means_adv[2], student_CI_ppf_one_sided * err_U_stderrs_adv[2],
        color='C1', linewidth=0.5, linestyle=':', capsize=1, label=r'$n=3$ (0.95 CI)'
    )
    ax_adv_bottom_left.set_xlim(N_STEPS[0], N_STEPS[-1])
    ax_adv_bottom_left.set_xticks(N_STEPS)
    ax_adv_bottom_left.set_xlabel(r'Observation time $N\tau$', fontsize=8)
    ax_adv_bottom_left.set_ylabel(r'$\|\hat{U}^{(n)}-U^{(n)}\|_2$', fontsize=8)
    ax_adv_bottom_left.legend(fontsize=6, handlelength=1.0)

    ax_adv_bottom_center.set_xscale('log')
    ax_adv_bottom_center.errorbar(
        N_STEPS, err_K_means_adv[0], student_CI_ppf_one_sided * err_K_stderrs_adv[0],
        color='C0', linewidth=0.5, linestyle='-', capsize=1, label=r'$n=0$ (0.95 CI)'
    )
    ax_adv_bottom_center.errorbar(
        N_STEPS, err_K_means_adv[1], student_CI_ppf_one_sided * err_K_stderrs_adv[1],
        color='C0', linewidth=0.5, linestyle='--', capsize=1, label=r'$n=1$ (0.95 CI)'
    )
    ax_adv_bottom_center.errorbar(
        N_STEPS, err_K_means_adv[2], student_CI_ppf_one_sided * err_K_stderrs_adv[2],
        color='C0', linewidth=0.5, linestyle=':', capsize=1, label=r'$n=2$ (0.95 CI)'
    )
    ax_adv_bottom_center.set_xlim(N_STEPS[0], N_STEPS[-1])
    ax_adv_bottom_center.set_ylim(0.0, 1.0)
    ax_adv_bottom_center.set_xticks(N_STEPS)
    ax_adv_bottom_center.set_xlabel(r'Observation time $N\tau$', fontsize=8)
    ax_adv_bottom_center.set_ylabel(r'$\|\hat{K}^{(n)}-K^{(n)}\|_2$', fontsize=8)
    ax_adv_bottom_center.legend(fontsize=6, handlelength=1.0)

    ax_adv_bottom_right.set_xscale('log')
    ax_adv_bottom_right.errorbar(
        N_STEPS, err_G_means_adv[0], student_CI_ppf_one_sided * err_G_stderrs_adv[0],
        color='C0', linewidth=0.5, linestyle='-', capsize=1, label='$n=1$\n(0.95 CI)'
    )
    ax_adv_bottom_right.errorbar(
        N_STEPS, err_G_means_adv[1], student_CI_ppf_one_sided * err_G_stderrs_adv[1],
        color='C0', linewidth=0.5, linestyle='--', capsize=1, label='$n=2$\n(0.95 CI)'
    )
    ax_adv_bottom_right.errorbar(
        N_STEPS, err_G_means_adv[2], student_CI_ppf_one_sided * err_G_stderrs_adv[2],
        color='C0', linewidth=0.5, linestyle=':', capsize=1, label='$n=3$\n(0.95 CI)'
    )
    ax_adv_bottom_right.set_xlim(N_STEPS[0], N_STEPS[-1])
    ax_adv_bottom_right.set_xticks(N_STEPS)
    ax_adv_bottom_right.set_xlabel(r'Observation time $N\tau$', fontsize=8)
    ax_adv_bottom_right.set_ylabel(r'$\|\hat{G}^{(n)}-G^{(n)}\|_2$', fontsize=8)
    # ax_adv_bottom_right.legend(fontsize=4, handlelength=1.0)

    fig_adv.canvas.draw()
    renderer_adv = fig_adv.canvas.get_renderer()
    inv_fig_adv = fig_adv.transFigure.inverted()

    def _bbox_center_adv(bbox):
        return (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2

    def _ylabel_x_adv(ax):
        bbox = ax.yaxis.label.get_window_extent(renderer_adv).transformed(inv_fig_adv)
        return _bbox_center_adv(bbox)[0]

    for bottom_ax, middle_ax in [
        (ax_adv_bottom_left, ax_adv_middle_left),
        (ax_adv_bottom_center, ax_adv_middle_center),
        (ax_adv_bottom_right, ax_adv_middle_right),
    ]:
        bottom_ax.yaxis.set_label_coords(
            _ylabel_x_adv(middle_ax),
            bottom_ax.get_position().y0,
            transform=fig_adv.transFigure
        )
        bottom_ax.yaxis.label.set_ha('left')
        bottom_ax.yaxis.label.set_va('center')

    def _top_ytick_y_adv(ax):
        tick_bboxes = [
            ticklabel.get_window_extent(renderer_adv).transformed(inv_fig_adv)
            for ticklabel in ax.get_yticklabels()
            if ticklabel.get_visible()
        ]
        top_bbox = max(tick_bboxes, key=lambda bbox: _bbox_center_adv(bbox)[1])
        return _bbox_center_adv(top_bbox)[1]

    fig_adv.canvas.draw()
    renderer_adv = fig_adv.canvas.get_renderer()
    inv_fig_adv = fig_adv.transFigure.inverted()

    panel_a_x_adv = _ylabel_x_adv(ax_adv_middle_left)
    panel_a_y_adv = _bbox_center_adv(
        top_outer_rate_label_adv.get_window_extent(renderer_adv).transformed(inv_fig_adv)
    )[1]
    panel_b_x_adv = _ylabel_x_adv(ax_adv_middle_center)
    panel_b_y_adv = panel_a_y_adv
    panel_c_x_adv = panel_a_x_adv
    panel_c_y_adv = _top_ytick_y_adv(ax_adv_middle_left)
    panel_d_x_adv = panel_b_x_adv
    panel_d_y_adv = panel_c_y_adv
    panel_e_x_adv = _ylabel_x_adv(ax_adv_middle_right)
    panel_e_y_adv = panel_d_y_adv
    panel_f_x_adv = panel_c_x_adv
    panel_f_y_adv = _top_ytick_y_adv(ax_adv_bottom_left)
    panel_g_x_adv = panel_d_x_adv
    panel_g_y_adv = _top_ytick_y_adv(ax_adv_bottom_center)
    panel_h_x_adv = panel_e_x_adv
    panel_h_y_adv = _top_ytick_y_adv(ax_adv_bottom_right)

    fig_adv.text(panel_a_x_adv, panel_a_y_adv, '(a)', **panel_label_kwargs)
    fig_adv.text(panel_b_x_adv, panel_b_y_adv, '(b)', **panel_label_kwargs)
    fig_adv.text(panel_c_x_adv, panel_c_y_adv, '(c)', **panel_label_kwargs)
    fig_adv.text(panel_d_x_adv, panel_d_y_adv, '(d)', **panel_label_kwargs)
    fig_adv.text(panel_e_x_adv, panel_e_y_adv, '(e)', **panel_label_kwargs)
    fig_adv.text(panel_f_x_adv, panel_f_y_adv, '(f)', **panel_label_kwargs)
    fig_adv.text(panel_g_x_adv, panel_g_y_adv, '(g)', **panel_label_kwargs)
    fig_adv.text(panel_h_x_adv, panel_h_y_adv, '(h)', **panel_label_kwargs)

    plt.savefig(RES_DIR / ADVERSARIAL_EXAMPLE_FN, bbox_inches='tight') 
    
