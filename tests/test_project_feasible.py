# tests/test_project_feasible.py
# test gmex/project_feasible.py classes and helper


import math
from typing import Dict
import warnings

import pytest
import torch

from gmex.project_feasible import (
    SinkhornKnoppScaler,
    KnightRuizUcarScaler,
    ReversibleCommutingIProjector,
    _CommutingFluxIProjector,
    project_Us,
)
from gmex.utils.core import column_normalize, symmetrize_matrix


def test_sinkhorn_projects_sampled_flux_to_target_marginals(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Project sampled flux onto fixed row and column marginals."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)
    projector = SinkhornKnoppScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    flux, info = projector.project(count_matrices[1] + 1e-6, simple_stationary_dist)

    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(flux.sum(dim=1) - simple_stationary_dist)).item() <= 1e-10
    assert torch.max(torch.abs(flux.sum(dim=0) - simple_stationary_dist)).item() <= 1e-10


def test_sinkhorn_reports_actual_row_and_column_residuals() -> None:
    """Report the same residuals that are obtained from the returned flux."""
    matrix = torch.tensor(
        [[1.0, 100.0, 2.0], [5.0, 3.0, 4.0], [7.0, 6.0, 8.0]],
        dtype=torch.float64,
    )
    stationary = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    projector = SinkhornKnoppScaler(
        min_entry=1e-24, max_iters=1, tol=1e-20, progress=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux, info = projector.project(matrix, stationary) # intentionally nonconvergent
    row_err = torch.linalg.vector_norm(flux.sum(dim=1) - stationary, ord=1).item() / 2.0
    col_err = torch.linalg.vector_norm(flux.sum(dim=0) - stationary, ord=1).item() / 2.0

    assert info["row_err"] == row_err
    assert info["col_err"] == col_err


def test_sinkhorn_stays_stable_for_adversarial_dynamic_range() -> None:
    """Converge on a near-decomposable case near the practical edge for unit tests.
    
    @Grant: for why this is adversarial, see P.A. Knight, D. Ruiz, and B. Ucar,
    A symmetry preserving algorithm for matrix scaling,
    SIAM J. Matrix Anal. Appl. 35, 25 (2014). 
    """
    stationary = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    base = torch.tensor(
        [[1.0, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 1.0]],
        dtype=torch.float64,
    )
    scale = 15.0
    factors = torch.tensor(
        [
            [scale, 1.0 / scale, 1.0 / scale],
            [1.0 / scale, scale, 1.0 / scale],
            [1.0 / scale, 1.0 / scale, scale],
        ],
        dtype=torch.float64,
    )
    matrix = base * factors
    assert matrix.max().item() / matrix.min().item() == 2250.0

    projector = SinkhornKnoppScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    flux, info = projector.project(matrix, stationary)

    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(flux.sum(dim=1) - stationary)).item() <= 1e-10
    assert torch.max(torch.abs(flux.sum(dim=0) - stationary)).item() <= 1e-10


def test_sinkhorn_is_idempotent_on_feasible_flux(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reprojecting a feasible Sinkhorn-scaled flux should leave it unchanged."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = SinkhornKnoppScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    flux_1, info_1 = projector.project(count_matrices[1] + 1e-6, simple_stationary_dist)
    flux_2, info_2 = projector.project(flux_1, simple_stationary_dist)

    assert info_1["converged"]
    assert info_2["converged"]
    torch.testing.assert_close(flux_2, flux_1, atol=1e-10, rtol=0.0)


def test_kru_projects_sampled_flux_to_symmetric_target_marginal(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Project sampled flux onto the symmetric fixed-marginal subset."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = KnightRuizUcarScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    flux, info = projector.project(count_matrices[1] + 1e-6, simple_stationary_dist)

    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(flux - flux.T)).item() <= 1e-12
    assert torch.max(torch.abs(flux.sum(dim=1) - simple_stationary_dist)).item() <= 1e-10


def test_kru_stays_finite_for_wide_dynamic_range() -> None:
    """Remain finite and symmetric on a wide but well-behaved input."""
    stationary = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    scale = 100.0
    matrix = torch.tensor(
        [
            [1.0 / scale, scale, 1.0],
            [scale, 1.0 / scale, math.sqrt(scale)],
            [1.0, math.sqrt(scale), 1.0 / scale],
        ],
        dtype=torch.float64,
    )

    projector = KnightRuizUcarScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    flux, info = projector.project(matrix, stationary)

    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(flux - flux.T)).item() <= 1e-12
    assert torch.max(torch.abs(flux.sum(dim=1) - stationary)).item() <= 1e-10


def test_kru_is_idempotent_on_feasible_flux(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reprojecting a feasible symmetric fixed-marginal flux should be stable."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = KnightRuizUcarScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    flux_1, info_1 = projector.project(count_matrices[1] + 1e-6, simple_stationary_dist)
    flux_2, info_2 = projector.project(flux_1, simple_stationary_dist)

    assert info_1["converged"]
    assert info_2["converged"]
    torch.testing.assert_close(flux_2, flux_1, atol=1e-10, rtol=0.0)


def test_commuting_projector_is_identity_for_identity_uprev() -> None:
    """Leave the input unchanged when every flux commutes with the identity."""
    matrix = torch.tensor(
        [[0.5, 0.2, 0.4], [0.3, 0.8, 0.3], [0.4, 0.3, 0.7]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)

    projector = _CommutingFluxIProjector(
        min_entry=1e-24, max_iters=200, tol=1e-12, progress=False
    )
    flux, info = projector.project(matrix, identity)

    assert torch.equal(flux, matrix)
    assert info == {
        "converged": True,
        "iters": 0,
        "comm_err": 0.0,
        "neg_err": 0.0,
        "step_err": 0.0,
        "affine_err": 0.0,
        "constraint_rank": 0,
        "objective": 0.0,
    }


def test_commuting_projector_enforces_commutation_on_sampled_flux(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    reversible_column_stochastic_matrix: torch.Tensor,
) -> None:
    """Project sampled flux onto the commuting affine subset."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = _CommutingFluxIProjector(
        min_entry=1e-24, max_iters=500, tol=1e-10, progress=False
    )
    flux, info = projector.project(
        count_matrices[1] + 1e-6,
        reversible_column_stochastic_matrix,
    )
    commutator = (
        reversible_column_stochastic_matrix @ flux
        - flux @ reversible_column_stochastic_matrix.T
    )

    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(commutator)).item() <= 1e-10


def test_commuting_projector_is_idempotent_on_feasible_flux(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    reversible_column_stochastic_matrix: torch.Tensor,
) -> None:
    """Reprojecting an already commuting positive flux should leave it unchanged."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = _CommutingFluxIProjector(
        min_entry=1e-24, max_iters=500, tol=1e-10, progress=False
    )
    flux_1, info_1 = projector.project(
        count_matrices[1] + 1e-6,
        reversible_column_stochastic_matrix,
    )
    flux_2, info_2 = projector.project(flux_1, reversible_column_stochastic_matrix)

    assert info_1["converged"]
    assert info_2["converged"]
    torch.testing.assert_close(flux_2, flux_1, atol=1e-10, rtol=0.0)


def test_reversible_commuting_matches_kru_for_identity_uprev() -> None:
    """Reduce to KRU when the commutation constraint is trivial."""
    stationary = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    matrix = torch.tensor(
        [[0.5, 0.2, 0.4], [0.3, 0.8, 0.3], [0.4, 0.3, 0.7]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)

    kru = KnightRuizUcarScaler(min_entry=1e-24, max_iters=100000, tol=1e-12, progress=False)
    rev = ReversibleCommutingIProjector(
        min_entry=1e-24,
        max_iters=250,
        tol=1e-12,
        max_iters_symm=2500,
        max_iters_comm=250,
        progress=False,
    )
    flux_kru, _ = kru.project(matrix, stationary)
    flux_rev, info = rev.project(matrix, identity, stationary)

    assert info["converged"]
    torch.testing.assert_close(flux_rev, flux_kru, atol=3e-12, rtol=0.0)


def test_reversible_commuting_projects_to_feasible_subset(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    reversible_column_stochastic_matrix: torch.Tensor,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Project sampled flux onto the reversible commuting feasible subset."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = ReversibleCommutingIProjector(
        min_entry=1e-24,
        max_iters=50,
        tol=1e-10,
        max_iters_symm=5000,
        max_iters_comm=500,
        progress=False,
    )
    flux, info = projector.project(
        count_matrices[1] + 1e-6,
        reversible_column_stochastic_matrix,
        reversible_stationary_dist,
    )

    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(flux - flux.T)).item() <= 1e-12
    assert torch.max(torch.abs(flux.sum(dim=1) - reversible_stationary_dist)).item() <= 1e-10
    assert torch.max(
        torch.abs(
            reversible_column_stochastic_matrix @ flux
            - flux @ reversible_column_stochastic_matrix.T
        )
    ).item() <= 1e-10


def test_reversible_commuting_is_idempotent_on_feasible_flux(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
    reversible_column_stochastic_matrix: torch.Tensor,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Reprojecting an already feasible flux should leave it unchanged."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    projector = ReversibleCommutingIProjector(
        min_entry=1e-24,
        max_iters=50,
        tol=1e-10,
        max_iters_symm=5000,
        max_iters_comm=500,
        progress=False,
    )
    flux_1, info_1 = projector.project(
        count_matrices[1] + 1e-6,
        reversible_column_stochastic_matrix,
        reversible_stationary_dist,
    )
    flux_2, info_2 = projector.project(
        flux_1,
        reversible_column_stochastic_matrix,
        reversible_stationary_dist,
    )

    assert info_1["converged"]
    assert info_2["converged"]
    torch.testing.assert_close(flux_2, flux_1, atol=1e-10, rtol=0.0)


def test_project_us_matches_manual_lagwise_projection_nonreversible(
    simple_inhomogeneous_markov_count_data: Dict[str, torch.Tensor | object],
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Match an explicit per-lag Sinkhorn projection loop exactly."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)
    transitions = torch.stack(
        [column_normalize(count_matrices[i], fallback_col=simple_stationary_dist) for i in range(count_matrices.shape[0])]
    )
    wrapper_out, metrics = project_Us(
        transitions,
        simple_stationary_dist,
        tol=1e-10,
        min_entry=1e-24,
        max_iters=5000,
        reversible=False,
        verbose=False,
    )
    projector = SinkhornKnoppScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    manual_out = torch.zeros_like(transitions)
    manual_out[0] = transitions[0]
    for lag in range(1, transitions.shape[0]):
        projector._reset()
        flux = transitions[lag] * simple_stationary_dist[None, :]
        flux, _ = projector.project(flux, simple_stationary_dist)
        manual_out[lag] = flux / simple_stationary_dist[None, :]

    assert torch.equal(wrapper_out, manual_out)
    assert torch.equal(wrapper_out[0], transitions[0])
    assert set(metrics) == {1, 2}


def test_project_us_matches_manual_lagwise_projection_reversible(
    simple_inhomogeneous_markov_count_data: Dict[str, torch.Tensor | object],
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Match the reversible lagwise projection loop exactly."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)
    transitions = torch.stack(
        [column_normalize(count_matrices[i], fallback_col=simple_stationary_dist) for i in range(count_matrices.shape[0])]
    )
    wrapper_out, metrics = project_Us(
        transitions,
        simple_stationary_dist,
        tol=1e-10,
        min_entry=1e-24,
        max_iters=5000,
        reversible=True,
        verbose=False,
    )
    projector = KnightRuizUcarScaler(
        min_entry=1e-24, max_iters=5000, tol=1e-10, progress=False
    )
    manual_out = torch.zeros_like(transitions)
    manual_out[0] = transitions[0]
    for lag in range(1, transitions.shape[0]):
        projector._reset()
        flux = transitions[lag] * simple_stationary_dist[None, :]
        flux = torch.clamp(flux, min=1e-24)
        flux = symmetrize_matrix(flux, geom=True)
        flux, _ = projector.project(flux, simple_stationary_dist)
        manual_out[lag] = flux / simple_stationary_dist[None, :]

    assert torch.equal(wrapper_out, manual_out)
    assert torch.equal(wrapper_out[0], transitions[0])
    assert set(metrics) == {1, 2}
    for lag in range(1, wrapper_out.shape[0]):
        flux = wrapper_out[lag] * simple_stationary_dist[None, :]
        assert torch.max(torch.abs(flux - flux.T)).item() <= 1e-12
