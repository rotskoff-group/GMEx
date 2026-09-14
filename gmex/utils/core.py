# gmex/utils/core.py
# General helper functions.


import math
from itertools import chain

import networkx as nx
import numpy as np
import torch


@torch.no_grad()
def coarsen_states(
    fine_states: torch.Tensor | list, groups: list
) -> torch.Tensor | list:
    """Coarse-grain series of states of a Markov jump process.

    Parameters
    ----------
    fine_states : torch.Tensor | list
        Tensor or (possibly ragged) list of states of arbitrary dimension.
    groups : list of list of int
        Groups of states to coarse-grain.

    Returns
    -------
    coarse_states: torch.Tensor | list
        Coarse-grained tensor or list with same shape as input.
    """
    if isinstance(fine_states, torch.Tensor):
        # Store original shape
        original_shape = fine_states.shape

        # Flatten the tensor for processing
        flat_fine_states = fine_states.flatten()
        coarse_states = torch.zeros_like(flat_fine_states, dtype=torch.long)

        for i, group in enumerate(groups):
            for state in group:
                coarse_states[flat_fine_states == state] = i

        # Reshape back to original shape
        return coarse_states.view(original_shape).to(fine_states.device)

    # Precompute lookup table for fast mapping
    state_to_group = {}
    for i, group in enumerate(groups):
        for state in group:
            state_to_group[int(state)] = i

    # Handle ragged lists by recursive mapping
    def _map(obj):
        if isinstance(obj, list):
            return [_map(x) for x in obj]
        return state_to_group[int(obj)]

    return _map(fine_states)


@torch.no_grad()
def get_random_init_dist(n_states: int) -> torch.Tensor:
    """Initialize distribution for simulation.

    Parameters
    ----------
    n_states : int
        Number of states

    Returns
    -------
    initial_dist : torch.Tensor
        Random initial distribution
    """
    initial_dist = torch.rand(n_states).to(torch.float64)
    initial_dist /= initial_dist.sum()
    return initial_dist


@torch.no_grad()
def allocate_dists(
    n_steps: int, n_states: int | None = None, initial_dist: torch.Tensor | None = None
) -> torch.Tensor:
    """Allocate tensor to store evolving distribution.

    Parameters
    ----------
    n_steps : int
        Number of simulation steps
    n_states : int, optional
        Number of states in distribution support,
        set to len(initial_dist) if None
    initial_dist : torch.Tensor, optional
        Initial distribution, randomized if None

    Returns
    -------
    dists : torch.Tensor, (n_steps + 1, n_states)
        Pre-allocated array to fill with evolving distribution
        dists[0] is initial_dist
    """
    if initial_dist is None:
        if n_states is None:
            raise ValueError("Either n_states or initial_dist must be provided.")
        initial_dist = get_random_init_dist(n_states)
    else:
        initial_dist = initial_dist.squeeze()
        if initial_dist.ndim > 1:
            raise ValueError("initial_dist must be one-dimensional after squeezing.")
        initial_dist = initial_dist.reshape(-1)
        inferred_n_states = initial_dist.numel()

        if n_states is None:
            n_states = inferred_n_states
        elif n_states != inferred_n_states:
            raise ValueError("initial_dist must have length n_states.")

    dists = torch.zeros(
        (n_steps + 1, n_states),
        dtype=torch.float64,
        device=initial_dist.device,
    )
    dists[0] = initial_dist
    return dists


@torch.no_grad()
def validate_groups(n_states: int, groups: list):
    """Check that groups encodes a valid coarse-graining.

    Parameters
    ----------
    n_states : int
        Number of states to coarse-grain
    groups : list of list of int
        Groups of states to coarse-grain. For example,
        [[0, 1], [2, 3, 4]] would group states 1 and 2 together and 3, 4, and 5 together.
    """
    # make sure no microstate is assigned more than once
    grouped_microstates = list(chain(*groups))
    unique, counts = np.unique(grouped_microstates, return_counts=True)
    assert np.all(counts == 1), (
        f"{unique[counts > 1]} grouped {counts[counts > 1]} times"
    )
    grouped_microstates = set(grouped_microstates)

    # make sure assigned microstates are exactly those expected
    expected_microstates = set(range(n_states))
    shared_microstates = expected_microstates.intersection(grouped_microstates)
    missing_microstates = expected_microstates - shared_microstates
    assert len(missing_microstates) == 0, (
        f"microstates {missing_microstates} not grouped"
    )
    extra_microstates = grouped_microstates - shared_microstates
    assert len(extra_microstates) == 0, f"extra microstates {extra_microstates} grouped"


@torch.no_grad()
def randomly_group(
    n_states: int,
    n_groups: int,
    rng: torch.Generator | None = None,
    equal_assignment: bool = False,
) -> list[list[int]]:
    """Randomly group n_states states into n_groups macrostates.

    Parameters
    ----------
    n_states : int
        Number of states to randomly group.
    n_groups : int
        Number of random states.
    rng : torch.Generator, optional
        Random number generator.
        Uses system entropy or global generator if not provided.
    equal_assignment : bool, optional
        If True, assign states to groups as evenly as possible.

    Returns
    -------
    groups : list of list of int
        Groups of states to coarse-grain. For example,
        [[0, 1], [2, 3, 4]] would group states 1 and 2 together and 3, 4, and 5 together.
    """
    if not 1 <= n_groups <= n_states:
        raise ValueError("n_groups must be between 1 and n_states.")

    perm = torch.randperm(n_states, generator=rng)

    if equal_assignment:
        groups = [[] for _ in range(n_groups)]
        for index, state in enumerate(perm.tolist()):
            groups[index % n_groups].append(int(state))
    else:
        interior_cuts = torch.randperm(n_states - 1, generator=rng)[: n_groups - 1] + 1
        interior_cuts, _ = torch.sort(interior_cuts)
        cuts: list[int] = [0, *interior_cuts.tolist(), n_states]
        groups = [perm[cuts[i] : cuts[i + 1]].tolist() for i in range(n_groups)]

    validate_groups(n_states, groups)
    return groups


@torch.no_grad()
def get_macrostate_indicator(groups: list) -> torch.Tensor:
    """Get macrostate indicator (Psi) Tensor from groups.

    Parameters
    ----------
    groups : list of list of int
        Groups of states to coarse-grain. For example,
        [[0, 1], [2, 3, 4]] would group states 1 and 2 together and 3, 4, and 5 together.

    Returns
    -------
    macrostate_indicator : torch.Tensor
        1 where microstate i is in macrostate j, 0 elsewhere
    """
    microstates = list(chain.from_iterable(groups))
    macrostate_indicator = torch.zeros(
        len(microstates), len(groups), dtype=torch.float64
    )
    for alpha, s in enumerate(groups):
        macrostate_indicator[s, alpha] = 1.0
    return macrostate_indicator


@torch.no_grad()
def get_p_micro_given_macro(
    macrostate_indicator: torch.Tensor, lim_dist: torch.Tensor
) -> torch.Tensor:
    """Get Tensor of stationary microstate probabilities given macrostates (Phi).

    Parameters
    ----------
    macrostate_indicator : torch.Tensor
        1 where microstate i is in macrostate j, 0 elsewhere
    lim_dist : torch.Tensor
        Limiting distribution over microstates.

    Returns
    -------
    p_micro_given_macro : torch.Tensor
        Element i, j is the probability of microstate i given macrostate j.
    """
    p_micro_given_macro = macrostate_indicator * lim_dist[:, None].to(
        macrostate_indicator
    )
    lim_dist_macro = macrostate_indicator.T @ lim_dist.to(macrostate_indicator)
    return p_micro_given_macro / lim_dist_macro


@torch.no_grad()
def connected_erdos_renyi(n: int, p: float | None = None, max_tries: int = 1000):
    """
    Rejection-sample a random connected graph on n labeled nodes.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float | None, optional
        Edge probability; percolation threshold log(n) / n by default
    max_tries : int
        Maximum number of attempts.
    """
    assert n > 1, "Number of nodes must be at least 2."
    if p is None:
        # Pick p at the percolation threshold
        p = math.log(n) / n

    for _ in range(max_tries):
        g = nx.fast_gnp_random_graph(n, p)
        if nx.is_connected(g):
            return torch.tensor(nx.to_numpy_array(g), dtype=torch.int32)

    raise RuntimeError(f"Failed to generate a connected graph after {max_tries} tries.")


@torch.no_grad()
def column_stochastic_lim_dist(L: torch.Tensor) -> torch.Tensor:
    """Estimate the stationary distribution of a column-stochastic matrix.

    Parameter
    ---------
    L : torch.Tensor
        Column-stochastic matrix of shape (M, M).

    Returns
    -------
    pi : torch.Tensor
        Stationary distribution of shape (M,).
    """
    # get left eigenvectors
    eigenvals, eigenvecs = torch.linalg.eig(L)
    # Find index of eigenvalue closest to 1
    idx = torch.argmin(torch.abs(eigenvals - 1.0))
    # Get corresponding eigenvector and normalize
    stationary = eigenvecs[:, idx].real
    return stationary / stationary.sum()


@torch.no_grad()
def check_nonnegative_square_matrix(M: torch.Tensor, name: str = "Matrix"):
    """Make sure M is a nonnegative square matrix."""
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(M.shape)}.")
    if M.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if torch.any(M < 0):
        raise ValueError(f"{name} cannot have negative entries.")


@torch.no_grad()
def check_probability_vector(
    P: torch.Tensor, n: int, name: str = "P", rtol: float = 1e-05, atol: float = 1e-08
) -> None:
    """Validate that P is a probability vector of length n."""
    if P.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(P.shape)}.")
    if P.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},), got {tuple(P.shape)}")
    if torch.any(P <= 0):
        raise ValueError(f"{name} cannot have nonnegative entries.")
    should_be_one = P.sum()
    if not torch.allclose(should_be_one, torch.ones_like(P), rtol=rtol, atol=atol):
        raise ValueError(
            f"{name} must sum to one, sums to {should_be_one.flatten().item()}."
        )


@torch.no_grad()
def normalize_probability_vector(P: torch.Tensor) -> torch.Tensor:
    """Return P normalized to sum to 1 (same dtype/device)."""
    Pt = as_float64(P)
    return Pt / Pt.sum()


@torch.no_grad()
def as_float64(x: torch.Tensor) -> torch.Tensor:
    """Return x as a torch.Tensor with dtype torch.float64."""
    if x.dtype == torch.float64:
        return x
    return x.to(torch.float64)


@torch.no_grad()
def nan_to_pos(x: torch.Tensor, min_entry: float = 1e-24) -> torch.Tensor:
    """Replace non-finite entries with positive values."""
    if torch.isfinite(x).all():
        return x
    finfo = torch.finfo(x.dtype)
    return torch.nan_to_num(x, nan=min_entry, posinf=finfo.max, neginf=min_entry)


@torch.no_grad()
def symmetrize_matrix(M: torch.Tensor, geom: bool = False) -> torch.Tensor:
    """Average a matrix with its transpose."""
    if geom:
        return as_float64((M * M.T).sqrt())
    return as_float64(0.5 * (M + M.T))


@torch.no_grad()
def column_normalize(
    C: torch.Tensor,
    fallback_col: torch.Tensor | None = None,
    min_entry: float | None = None,
) -> torch.Tensor:
    """Column-normalize square matrix.

    Parameters
    ----------
    C : (n, n) torch.Tensor
        Nonnegative transition counts or other square matrix.
    fallback_col : (n,) torch.Tensor, optional
        Strictly positive probability vector used for columns with zero total counts.
    min_entry : float, optional
        Minimum value to which to clamp counts, effectively a pseudocount.

    Returns
    -------
    U0 : (n, n) torch.Tensor
        Column normalization of C.
    """
    n = C.shape[0]
    Cf = as_float64(C).clone()
    if min_entry is not None:
        Cf = torch.clamp(Cf, min=min_entry)
    col_sums = Cf.sum(dim=0, keepdim=True)
    U0 = torch.empty_like(Cf)
    nonzero = col_sums.squeeze(0) > 0

    if torch.any(nonzero):
        U0[:, nonzero] = Cf[:, nonzero] / col_sums[:, nonzero]
    if torch.any(~nonzero):
        assert fallback_col is not None, (
            "fallback_col must be passed when some columns are empty"
        )
        if fallback_col.shape != (n,):
            raise ValueError(
                f"Expected fallback_col shape ({n},), got {tuple(fallback_col.shape)}."
            )
        U0[:, ~nonzero] = fallback_col.unsqueeze(1).expand(
            (n, int((~nonzero).sum().item()))
        )
    return U0


@torch.no_grad()
def check_column_stochastic_matrix(
    M: torch.Tensor, name: str = "M", rtol: float = 1e-05, atol: float = 1e-08
) -> None:
    """Validate that M is a nonnegative column-stochastic matrix."""
    check_nonnegative_square_matrix(M, name=name)
    col_sums = M.sum(dim=0)
    ones = torch.ones_like(col_sums)
    if not torch.allclose(col_sums, ones, rtol=rtol, atol=atol):
        err = torch.abs(col_sums - ones).max()
        raise ValueError(
            f"{name} must be column-stochastic; max column-sum residual {err.item():.2e}."
        )


@torch.no_grad()
def check_stationary_column_stochastic_matrix(
    U: torch.Tensor,
    P: torch.Tensor,
    db: bool = False,
    tol: float = 1e-08,
    name: str = "U",
) -> None:
    """Validate that U is column-stochastic, stationary wrt P,
    and, if db, reversible with respect to P."""
    check_column_stochastic_matrix(U, name=name)
    check_probability_vector(P, U.shape[0], name="P")
    Uf = as_float64(U)
    Pf = as_float64(P).to(device=Uf.device, dtype=Uf.dtype)
    if db:
        db_err = torch.abs(Uf * Pf[None, :] - Uf.T * Pf[:, None]).max()
        if db_err > tol:
            raise ValueError(
                f"{name} must satisfy detailed balance with P; residual {db_err.item():.2e}."
            )
    else:
        pi_err = torch.abs(Uf @ Pf - Pf).max()
        if pi_err > tol:
            raise ValueError(
                f"{name} must be stationary with P; residual {pi_err.item():.2e}."
            )
