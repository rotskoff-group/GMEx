# gmex/utils/stats.py
# Helper functions for validation and evaluation.


import numpy as np
import torch


def np_rng_from_L():
    """Get np rng from lightning-set global bit-generator state.
    L.seed_everything only seeds traditional np global bit generator.
    It is necessary to get np rng in a complicated way for new np.random.Generator methods.
    """
    seed = torch.randint(2**32 - 1, (1,))[0]
    return np.random.default_rng(int(seed))


def _normalize_reduce_dims(
    ndim: int, reduce_dim: int | tuple[int, ...] | None
) -> tuple[int, ...]:
    """Normalize reduction dimensions for batched distribution metrics."""
    if reduce_dim is None:
        if ndim == 1:
            return (0,)
        return tuple(range(1, ndim))

    if isinstance(reduce_dim, int):
        reduce_dims = (reduce_dim,)
    else:
        reduce_dims = tuple(reduce_dim)

    normalized_dims = tuple(dim if dim >= 0 else ndim + dim for dim in reduce_dims)
    assert len(normalized_dims) > 0, "reduce_dim must contain at least one dimension"
    assert len(set(normalized_dims)) == len(normalized_dims), (
        "reduce_dim must not contain duplicate dimensions"
    )
    assert all(0 <= dim < ndim for dim in normalized_dims), (
        "reduce_dim contains an invalid dimension"
    )
    return normalized_dims


def kl_divergence(
    dists_truth: torch.Tensor,
    dists_model: torch.Tensor,
    eps: float | None = 1e-12,
    reduce_dim: int | tuple[int, ...] | None = None,
) -> float | torch.Tensor:
    """KL divergences between distributions or batches of distributions.

    Parameters
    ----------
    dists_truth : torch.Tensor
        True distributions.
    dists_model : torch.Tensor
        Model distributions.
    eps : float, optional
        Small value to which dists_model is clamped for numerical stability.
    reduce_dim : int | tuple[int] | None, optional
        Dimensions spanning the support of each distribution. If ``None``,
        reduce over the whole tensor for 1D inputs and over all except zeroth otherwise.

    Returns
    -------
    kls : float | torch.Tensor
        KL divergences from true to model distributions.
    """
    assert dists_truth.shape == dists_model.shape, (
        "dists_truth and dists_model shape mismatch"
    )
    reduce_dims = _normalize_reduce_dims(dists_truth.ndim, reduce_dim)
    dists_truth_sum = dists_truth.sum(dim=reduce_dims)
    dists_model_sum = dists_model.sum(dim=reduce_dims)
    ones_like_dists_sum = torch.ones_like(dists_truth_sum)
    assert torch.allclose(dists_truth_sum, ones_like_dists_sum), (
        "dists_truth must sum to (approximately) 1 across reduce_dim"
    )
    assert torch.allclose(dists_model_sum, ones_like_dists_sum), (
        "dists_model must sum to (approximately) 1 across reduce_dim"
    )

    # clamp for safety
    if eps is not None:
        dists_model = dists_model.clone().clamp_min(eps)
        dists_truth_safe = dists_truth.clamp_min(eps)
        dists_truth_safe = dists_truth_safe / dists_truth_safe.sum(
            dim=reduce_dims, keepdim=True
        )
        dists_model = dists_model / dists_model.sum(dim=reduce_dims, keepdim=True)
    else:
        dists_truth_safe = dists_truth

    mask = dists_truth > 0
    log_ratio = torch.log(dists_truth_safe) - torch.log(dists_model)
    term = torch.where(
        mask,
        dists_truth * log_ratio,
        torch.zeros((), dtype=dists_truth.dtype, device=dists_truth.device),
    )
    result = term.sum(dim=reduce_dims)
    return result.item() if result.ndim == 0 else result


def tv_distance(
    dists_truth: torch.Tensor,
    dists_model: torch.Tensor,
    reduce_dim: int | tuple[int, ...] | None = None,
) -> float | torch.Tensor:
    """Total-variation distance between distributions or batches of distributions.

    Parameters
    ----------
    dists_truth : torch.Tensor
        True distributions.
    dists_model : torch.Tensor
        Model distributions.
    reduce_dim : int | tuple[int] | None, optional
        Dimensions spanning the support of each distribution. If ``None``,
        reduce over the whole tensor for 1D inputs and over all dimensions except zeroth otherwise.

    Returns
    -------
    tvs : float | torch.Tensor
        Total-variation distance between true and model distributions.
    """
    assert dists_truth.shape == dists_model.shape, (
        "dists_truth and dists_model shape mismatch"
    )
    reduce_dims = _normalize_reduce_dims(dists_truth.ndim, reduce_dim)
    dists_truth_sum = dists_truth.sum(dim=reduce_dims)
    dists_model_sum = dists_model.sum(dim=reduce_dims)
    ones_like_dists_sum = torch.ones_like(dists_truth_sum)
    assert torch.allclose(dists_truth_sum, ones_like_dists_sum), (
        "dists_truth must sum to (approximately) 1 across reduce_dim"
    )
    assert torch.allclose(dists_model_sum, ones_like_dists_sum), (
        "dists_model must sum to (approximately) 1 across reduce_dim"
    )
    result = (dists_truth - dists_model).abs().sum(dim=reduce_dims) / 2
    return result.item() if result.ndim == 0 else result


def get_bootstrap_curve_CI(
    samples: np.ndarray, pct: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get bootstrap confidence interval of a curve."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {tuple(samples.shape)}")
    if pct >= 1.0 or pct <= 0.0:
        raise ValueError("pct must be in the range (0.0, 1.0)")

    n_bootstraps = samples.shape[0]
    half_alpha = 0.5 - pct / 2.0
    idx_lower = int(n_bootstraps * half_alpha)
    if idx_lower < 1:
        raise ValueError(
            f"{n_bootstraps} bootstrap samples is not enough for a {pct} CI"
        )
    idx_upper = n_bootstraps - idx_lower - 1

    samples.sort(axis=0)
    return samples[idx_lower], samples[idx_upper]


def fpts_from_inhomogeneous_mc(
    tpms: torch.Tensor,
    start_states: int | list[int] | torch.Tensor,
    end_states: int | list[int] | torch.Tensor,
    start_dist: torch.Tensor | None = None,
    n_trajs: int = 1,
    max_steps: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor | int:
    """Simulate first-passage times for a time-inhomogeneous Markov chain.

    Parameters
    ----------
    tpms : (N, M, M) torch.Tensor
        Row-stochastic time-dependent Markov-chain generators;
        tpms[min(i, N)] used at timestep i.
    start_states : int | list[int] | torch.Tensor
        Initial states from which to sample.
    end_states : int | list[int] | torch.Tensor
        States to treat as absorbing.
    start_dist : (M,) torch.Tensor, optional
        Initial states will be sampled from start_dist[start_states] (normalized).
        Initial states are otherwise sampled uniformly from start_states.
    n_trajs : int, optional
        Number of trajectories to simulate in parallel.
    max_steps : int | None, optional
        Maximum number of steps for which to simulate trajectories.
    generator : torch.Generator | None, optional
        Generator for sampling.

    Returns
    -------
    fpts : int | torch.Tensor
        First-passage time or tensor of first-passage times from start to end states.
    """
    if tpms.ndim != 3 or tpms.shape[1] != tpms.shape[2]:
        raise ValueError("tpms must have shape (N, M, M)")
    if n_trajs < 1:
        raise ValueError("n_trajs must be >= 1")

    N, M, _ = tpms.shape
    device = tpms.device

    def _to_states(x, name: str) -> torch.Tensor:
        if isinstance(x, int):
            out = torch.tensor([x], dtype=torch.long, device=device)
        else:
            out = torch.as_tensor(x, dtype=torch.long, device=device).flatten()
        if out.numel() == 0:
            raise ValueError(f"{name} must be non-empty.")
        if ((out < 0) | (out >= M)).any():
            raise ValueError(f"All {name} must be in [0, M).")
        return torch.unique(out)

    start_states_t = _to_states(start_states, "start_states")
    end_states_t = _to_states(end_states, "end_states")

    # get initial sampling distribution over states
    if start_dist is None:
        probs = torch.zeros(M, dtype=torch.float64, device=device)
        probs[start_states_t] = 1.0
    else:
        start_dist_t = torch.as_tensor(
            start_dist, dtype=torch.float64, device=device
        ).flatten()
        if start_dist_t.numel() != M:
            raise ValueError("start_dist must have shape (M,).")
        if (start_dist_t < 0).any():
            raise ValueError("start_dist must be nonnegative.")
        probs = torch.zeros_like(start_dist_t)
        probs[start_states_t] = start_dist_t[start_states_t]

    mass = probs.sum()
    if mass <= 0:
        raise ValueError("No positive probability on the provided start_states.")
    probs = probs / mass

    # precompute CDFs once for fast sampling
    cdf = tpms.cumsum(dim=-1).contiguous()
    cdf[..., -1] = 1.0  # avoid numerical drift

    # sample initial states
    states = torch.multinomial(
        probs, num_samples=n_trajs, replacement=True, generator=generator
    ).to(torch.long)

    steps = torch.zeros(n_trajs, dtype=torch.long, device=device)

    # get masks for absorbed and not-yet-absorbed states
    end_mask = torch.zeros(M, dtype=torch.bool, device=device)
    end_mask[end_states_t] = True
    alive = ~end_mask[states]

    t = 0
    while alive.any():
        if max_steps is not None and t >= max_steps:
            raise RuntimeError(
                "max_steps reached before all trajectories hit end_states"
            )
        idx = torch.where(alive)[0]
        row_cdf = cdf[min(t, N - 1), states[idx]]
        u = torch.rand(
            (idx.numel(), 1), dtype=row_cdf.dtype, device=device, generator=generator
        )
        next_states = torch.searchsorted(row_cdf, u, right=False).squeeze(-1)

        states[idx] = next_states
        steps[idx] += 1
        alive[idx] = ~end_mask[next_states]
        t += 1

    return int(steps.item()) if n_trajs == 1 else torch.sort(steps).values


def fpts_from_traj(
    traj: torch.Tensor,
    start_states: int | list[int] | torch.Tensor,
    end_states: int | list[int] | torch.Tensor,
) -> torch.Tensor:
    """Get first-passage times from a 1D integer trajectory.

    Parameters
    ----------
    traj : torch.Tensor
        Trajectory of integer states.
    start_states : int | list[int] | torch.Tensor
        Initial states.
    end_states : int | list[int] | torch.Tensor
        States to treat as absorbing.

    Returns
    -------
    fpts : torch.Tensor
        First-passage times from start_states to end_states.
    """
    if traj.ndim != 1:
        raise ValueError(f"traj must be 1D, got shape {tuple(traj.shape)}")
    if traj.dtype != torch.long:
        traj = traj.long()

    device = traj.device
    starts = torch.as_tensor(start_states, dtype=torch.long, device=device).reshape(-1)
    ends = torch.as_tensor(end_states, dtype=torch.long, device=device).reshape(-1)

    if traj.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if starts.numel() == 0 or ends.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    is_start = torch.isin(traj, starts)
    is_end = torch.isin(traj, ends)

    # first index of each contiguous run of start states
    prev_is_start = torch.zeros_like(is_start)
    prev_is_start[1:] = is_start[:-1]
    start_mask = is_start & ~prev_is_start
    start_idx = torch.nonzero(start_mask, as_tuple=False).flatten()

    end_idx = torch.nonzero(is_end, as_tuple=False).flatten()

    if start_idx.numel() == 0 or end_idx.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    # for each start index, find the first end index >= start index
    locs = torch.searchsorted(end_idx, start_idx, right=False)
    valid = locs < end_idx.numel()
    if not torch.any(valid):
        return torch.empty((0,), dtype=torch.long, device=device)

    fpts = end_idx[locs[valid]] - start_idx[valid]
    return torch.sort(fpts.to(torch.long)).values


def get_dwell_probs(gs: torch.Tensor, max_timestep: int = 1000) -> torch.Tensor:
    """Get dwell-time densities for each state.

    Parameters
    ----------
    gs : torch.Tensor, (N, M, M)
        Potentially inhomogeneous Markov-chain transition-probability matrices.
    max_timestep : int, optional
        Maximum number of timesteps. Default 1000.

    Returns
    -------
    prob_dwell : torch.Tensor, (M, max_timestep)
        Probability of a j-timestep dwell in state i is prob_dwell[i, j].
    """
    device = gs.device

    if gs.shape[1] != gs.shape[2] or gs.ndim != 3:
        raise ValueError("gs must be 3D with same last two dims")
    if (gs < 0.0).any() or (gs > 1.0).any():
        raise ValueError("gs may contain only valid probabilities")

    prob_dwell = torch.zeros((gs.shape[1], max_timestep), device=device)
    for start_state in range(gs.shape[1]):
        survival_probs = torch.ones((max_timestep,), dtype=gs.dtype, device=device)
        survival_probs *= gs[-1, start_state, start_state]
        survival_probs[: gs.shape[0]] = gs[:max_timestep, start_state, start_state]
        prob_alive = torch.cumprod(survival_probs, 0)
        prob_dwell[start_state, 1:] = prob_alive[:-1] - prob_alive[1:]

    return prob_dwell
