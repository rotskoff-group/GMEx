# tests/test_utils_stats.py
# test gmex/utils/stats.py helpers


import numpy as np
import pytest
import torch

from gmex.utils.stats import (
    fpts_from_traj,
    get_bootstrap_curve_CI,
    kl_divergence,
    np_rng_from_L,
    tv_distance,
)


def test_fpts_from_traj_returns_sorted_first_passage_times() -> None:
    """Return sorted first-passage times from a simple trajectory."""
    traj = torch.tensor([0, 0, 1, 1, 2, 1, 3, 0, 2], dtype=torch.long)
    fpts = fpts_from_traj(traj, start_states=[0, 1], end_states=2)
    assert torch.equal(fpts, torch.tensor([1, 3, 4], dtype=torch.long))


def test_fpts_from_traj_returns_empty_when_no_hit_occurs() -> None:
    """Return an empty tensor when no end state is reached after a start."""
    traj = torch.tensor([0, 0, 1, 1, 3, 3], dtype=torch.long)
    fpts = fpts_from_traj(traj, start_states=[0, 1], end_states=2)
    assert torch.equal(fpts, torch.empty((0,), dtype=torch.long))


def test_tv_distance_returns_expected_scalar() -> None:
    """Compute the total-variation distance between two simple distributions."""
    dists_truth = torch.tensor([0.75, 0.25], dtype=torch.float64)
    dists_model = torch.tensor([0.5, 0.5], dtype=torch.float64)
    observed_tv = tv_distance(dists_truth, dists_model)
    assert np.isclose(observed_tv, 0.25)


def test_tv_distance_returns_expected_batch_values(
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Compute one TV distance per batch row by default."""
    dists_truth = torch.stack(
        (
            simple_stationary_dist,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
    )
    dists_model = torch.stack(
        (
            torch.full((4,), 0.25, dtype=torch.float64),
            torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64),
        )
    )
    observed_tv = tv_distance(dists_truth, dists_model)
    assert isinstance(observed_tv, torch.Tensor)
    expected_tv = torch.tensor([0.2, 0.5], dtype=torch.float64)
    assert torch.allclose(observed_tv, expected_tv)


def test_kl_divergence_returns_expected_scalar() -> None:
    """Compute the KL divergence between two simple distributions."""
    dists_truth = torch.tensor([0.75, 0.25], dtype=torch.float64)
    dists_model = torch.tensor([0.5, 0.5], dtype=torch.float64)
    observed_kl = kl_divergence(dists_truth, dists_model)
    expected_kl = 0.75 * np.log(1.5) + 0.25 * np.log(0.5)
    assert np.isclose(observed_kl, expected_kl)


def test_kl_divergence_handles_zero_truth_mass_with_default_eps() -> None:
    """Ignore zero-mass truth entries while keeping the finite KL value."""
    dists_truth = torch.tensor([1.0, 0.0], dtype=torch.float64)
    dists_model = torch.tensor([0.5, 0.5], dtype=torch.float64)
    observed_kl = kl_divergence(dists_truth, dists_model)
    assert np.isclose(observed_kl, np.log(2.0))


def test_get_bootstrap_curve_CI_returns_expected_bounds() -> None:
    """Return column-wise lower and upper bootstrap bounds."""
    samples = np.array(
        [
            [30, 300],
            [10, 100],
            [40, 400],
            [20, 200],
            [50, 500],
            [60, 600],
            [70, 700],
            [80, 800],
            [90, 900],
            [100, 1000],
            [110, 1100],
            [120, 1200],
            [130, 1300],
            [140, 1400],
            [150, 1500],
            [160, 1600],
            [170, 1700],
            [180, 1800],
            [190, 1900],
            [200, 2000],
            [210, 2100],
        ]
    )

    lower, upper = get_bootstrap_curve_CI(samples, 0.90)

    assert np.array_equal(lower, np.array([20, 200]))
    assert np.array_equal(upper, np.array([200, 2000]))
    assert lower.shape == (2,)
    assert upper.shape == (2,)


def test_get_bootstrap_curve_CI_rejects_non_2d_samples() -> None:
    """Reject arrays that do not represent bootstrap curves."""
    samples = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="samples must be 2D"):
        get_bootstrap_curve_CI(samples, 0.95)


def test_get_bootstrap_curve_CI_rejects_pct_outside_open_unit_interval() -> None:
    """Reject confidence levels outside the open interval (0, 1)."""
    samples = np.arange(80).reshape(40, 2)
    with pytest.raises(ValueError, match=r"pct must be in the range \(0\.0, 1\.0\)"):
        get_bootstrap_curve_CI(samples.copy(), 0.0)
    with pytest.raises(ValueError, match=r"pct must be in the range \(0\.0, 1\.0\)"):
        get_bootstrap_curve_CI(samples.copy(), 1.0)


def test_get_bootstrap_curve_CI_rejects_too_few_bootstraps() -> None:
    """Reject sample sets that are too small for the requested CI width."""
    samples = np.arange(78).reshape(39, 2)
    with pytest.raises(ValueError, match="bootstrap samples is not enough"):
        get_bootstrap_curve_CI(samples, 0.95)


def test_np_rng_from_L_returns_numpy_generator() -> None:
    """Return a numpy random-number generator."""
    rng = np_rng_from_L()
    assert isinstance(rng, np.random.Generator)


def test_np_rng_from_L_matches_torch_seed_when_reseeded() -> None:
    """Reproduce numpy-generator output after resetting the torch seed."""
    torch.manual_seed(123)
    rng_1 = np_rng_from_L()
    sample_1 = rng_1.integers(0, 100, size=5)

    torch.manual_seed(123)
    rng_2 = np_rng_from_L()
    sample_2 = rng_2.integers(0, 100, size=5)

    assert np.array_equal(sample_1, sample_2)


def test_np_rng_from_L_produces_distinct_generators_without_reseeding() -> None:
    """Advance with torch RNG state when called repeatedly without reseeding."""
    torch.manual_seed(123)
    rng_1 = np_rng_from_L()
    rng_2 = np_rng_from_L()

    sample_1 = rng_1.integers(0, 100, size=5)
    sample_2 = rng_2.integers(0, 100, size=5)

    assert not np.array_equal(sample_1, sample_2)
