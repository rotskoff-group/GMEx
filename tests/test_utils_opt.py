# tests/test_utils_opt.py
# test gmex/utils/opt.py helpers


import math
import pytest
import torch
from typing import Dict

from gmex.utils.opt import *


def test_check_count_matrices_accepts_markov_generated_counts(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
) -> None:
    """Accept count matrices generated from sampled trajectories."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)
    check_count_matrices(count_matrices)


def test_check_count_matrices_rejects_non_3d_input() -> None:
    """Reject count matrices that are not three-dimensional."""
    with pytest.raises(ValueError, match="must be 3D"):
        check_count_matrices(torch.ones((4, 4), dtype=torch.float64))


def test_check_count_matrices_rejects_nonsquare_slices() -> None:
    """Reject count matrices whose slices are not square."""
    with pytest.raises(ValueError, match="same 1st and 2nd dimension"):
        check_count_matrices(torch.ones((2, 3, 4), dtype=torch.float64))


def test_check_count_matrices_rejects_nondiagonal_lag_zero(
    simple_markov_count_data: Dict[str, torch.Tensor | object],
) -> None:
    """Reject stacks whose zeroth count matrix is not diagonal."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)
    bad = count_matrices.clone()
    bad[0, 0, 1] = 1.0

    with pytest.raises(ValueError, match="must be diagonal"):
        check_count_matrices(bad)


def test_rescale_sinkhorn_input_replaces_bad_entries_and_normalizes_mean() -> None:
    """Return a finite positive matrix with mean one after clamping and rescaling."""
    matrix = torch.tensor(
        [[float("nan"), float("inf")], [0.0, -3.0]],
        dtype=torch.float32,
    )
    scaled = rescale_sinkhorn_input(matrix, min_entry=1e-6)

    assert scaled.dtype == torch.float64
    assert torch.isfinite(scaled).all()
    assert torch.all(scaled > 0)
    assert torch.isclose(scaled.mean(), torch.tensor(1.0, dtype=torch.float64))


def test_get_log_likelihood_matches_manual_smoothed_sum() -> None:
    """Match a direct smoothed log-likelihood calculation."""
    counts = torch.tensor([[3, 0], [1, 2]], dtype=torch.int64)
    transitions = torch.tensor([[0.8, 0.0], [0.2, 1.0]], dtype=torch.float32)
    min_entry = 1e-6

    observed = get_log_likelihood(counts, transitions, min_entry=min_entry)
    expected = torch.sum(
        counts.to(dtype=transitions.dtype) * torch.log(torch.clamp(transitions, min=min_entry))
    )
    assert torch.allclose(observed, expected)


def test_generalized_kl_divergence_is_zero_when_inputs_match() -> None:
    """Return zero divergence when both inputs are equal."""
    matrix = torch.tensor([[0.2, 0.3], [0.4, 0.5]], dtype=torch.float64)
    observed = generalized_kl_divergence(matrix, matrix)
    assert observed.item() == pytest.approx(0.0)


def test_generalized_kl_divergence_returns_elementwise_values() -> None:
    """Return the elementwise generalized KL divergence when requested."""
    p = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=torch.float64)
    q = torch.tensor([[0.25, 2.0], [1.0, 8.0]], dtype=torch.float64)
    observed = generalized_kl_divergence(p, q, reduction="none")
    expected = p * (torch.log(p) - torch.log(q)) - p + q
    assert torch.allclose(observed, expected)


def test_generalized_kl_divergence_rejects_bad_reduction() -> None:
    """Reject unsupported reduction modes."""
    with pytest.raises(ValueError, match="reduction must be 'sum' or 'none'"):
        generalized_kl_divergence(
            torch.ones((2, 2), dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.float64),
            reduction="mean",
        )


def test_solve_levenberg_step_matches_exact_unregularized_solution() -> None:
    """Solve the Newton system exactly when the shifted matrix is already positive definite."""
    grad = torch.tensor([4.0, 9.0], dtype=torch.float64)
    hess = -torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    step, rho = solve_levenberg_step(grad, hess, tol=1e-12)

    expected = torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert rho == pytest.approx(0.0)
    assert torch.allclose(step, expected)


def test_solve_levenberg_step_regularizes_indefinite_system() -> None:
    """Apply a positive ridge shift and still return a finite step."""
    grad = torch.tensor([2.0, -3.0], dtype=torch.float64)
    hess = torch.tensor([[1.0, 0.0], [0.0, -1e-16]], dtype=torch.float64)
    tol = 1e-12
    step, rho = solve_levenberg_step(grad, hess, tol=tol)
    reg_matrix = -0.5 * (hess + hess.T) + rho * torch.eye(2, dtype=torch.float64)

    assert rho > 0.0
    assert torch.isfinite(step).all()
    assert torch.min(torch.linalg.eigvalsh(reg_matrix)).item() >= 0.5 * tol


def test_build_commuting_constraints_flux_matches_direct_commutator() -> None:
    """Encode the linear commutator map in row-major vector form."""
    stationary = torch.tensor([0.3, 0.7], dtype=torch.float64)
    transition = 0.6 * torch.eye(2, dtype=torch.float64) + 0.4 * (
        stationary[:, None] * torch.ones((1, 2), dtype=torch.float64)
    )
    flux = torch.tensor([[1.2, 0.4], [0.7, 2.0]], dtype=torch.float64)
    A, b = build_commuting_constraints_flux(transition)
    observed = A @ flux.reshape(-1)
    expected = (transition @ flux - flux @ transition.T).reshape(-1)

    assert torch.allclose(observed, expected)
    assert torch.equal(b, torch.zeros_like(b))


def test_orthonormalize_affine_constraints_preserves_consistent_system() -> None:
    """Return orthonormal rows and preserve a nearly rank-deficient consistent system."""
    A = torch.diag(torch.tensor([1.0, 5e-12, 5e-14], dtype=torch.float64))
    x_exact = torch.tensor([1.0, 0.4, 0.0], dtype=torch.float64)
    b = A @ x_exact
    Q, c, rank = orthonormalize_affine_constraints(A, b, tol=1e-13)

    assert rank == 2
    assert Q.shape == (2, 3)
    assert c.shape == (2,)
    assert torch.allclose(Q @ Q.T, torch.eye(rank, dtype=torch.float64))
    assert torch.allclose(Q @ x_exact, c, atol=1e-12, rtol=1e-12)


def test_orthonormalize_affine_constraints_rejects_inconsistent_system() -> None:
    """Reject affine systems that are inconsistent within tolerance."""
    A = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="inconsistent within numerical tolerance"):
        orthonormalize_affine_constraints(A, b, tol=1e-12)


def test_orthonormalize_affine_constraints_returns_empty_linear_system() -> None:
    """Return an empty orthonormal system for zero-row linear constraints."""
    Q, c, rank = orthonormalize_affine_constraints(
        torch.zeros((0, 3), dtype=torch.float64),
        torch.zeros((0,), dtype=torch.float64),
        linear=True,
    )
    assert Q.shape == (0, 3)
    assert c.shape == (0,)
    assert rank == 0


def test_kl_affine_primal_from_dual_clamps_extreme_logs() -> None:
    """Clamp primal log-values to the requested finite range."""
    sqrt3 = math.sqrt(3.0)
    log_k = torch.log(torch.tensor([1e-200, 1.0, 1e200], dtype=torch.float64))
    Q = torch.tensor([[1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3]], dtype=torch.float64)
    dual = torch.tensor([-1e6], dtype=torch.float64)
    log_min = math.log(1e-300)
    log_max = math.log(torch.finfo(torch.float64).max)
    log_x, x = kl_affine_primal_from_dual(log_k, Q, dual, log_min, log_max)

    assert torch.isfinite(log_x).all()
    assert torch.isfinite(x).all()
    assert torch.all(log_x >= log_min)
    assert torch.all(log_x <= log_max)
    assert torch.all(x > 0)


def test_kl_affine_dual_backtracking_accepts_real_descent_step() -> None:
    """Accept a descent step that lowers the dual objective."""
    sqrt2 = math.sqrt(2.0)
    k = torch.tensor([0.4, 1.6], dtype=torch.float64)
    Q = torch.tensor([[1.0 / sqrt2, 1.0 / sqrt2]], dtype=torch.float64)
    c = torch.tensor([1.0 / sqrt2], dtype=torch.float64)
    dual0 = torch.zeros((1,), dtype=torch.float64)
    log_min = math.log(1e-24)
    log_max = math.log(torch.finfo(torch.float64).max)
    log_k = torch.log(k)

    _, x0 = kl_affine_primal_from_dual(log_k, Q, dual0, log_min, log_max)
    resid = Q @ x0 - c
    hess = (Q * x0.unsqueeze(0)) @ Q.T
    step, _ = solve_levenberg_step(resid, -hess, tol=1e-12)
    dual_obj0 = float((c @ dual0 + x0.sum()).item())
    slope = float((-torch.dot(resid, step)).item())
    dual1, dual_obj1, alpha, accepted = kl_affine_dual_backtracking(
        log_k,
        Q,
        c,
        dual0,
        step,
        dual_obj0,
        slope,
        log_min,
        log_max,
        max_iters_ls=10,
    )
    assert accepted
    assert alpha > 0.0
    assert dual_obj1 < dual_obj0
    assert not torch.equal(dual1, dual0)


def test_kl_affine_dual_backtracking_returns_original_iterate_when_no_steps_allowed() -> None:
    """Return the original iterate unchanged when line search is disabled."""
    dual = torch.tensor([0.0], dtype=torch.float64)
    dual_obj = 1.5
    dual_new, dual_obj_new, alpha, accepted = kl_affine_dual_backtracking(
        log_k=torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)),
        Q=torch.tensor([[1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]], dtype=torch.float64),
        c=torch.tensor([1.0 / math.sqrt(2.0)], dtype=torch.float64),
        dual=dual,
        step=torch.tensor([1.0], dtype=torch.float64),
        dual_obj=dual_obj,
        slope=-1.0,
        log_min=math.log(1e-24),
        log_max=math.log(torch.finfo(torch.float64).max),
        max_iters_ls=0,
    )
    assert not accepted
    assert alpha == pytest.approx(0.0)
    assert dual_obj_new == pytest.approx(dual_obj)
    assert torch.equal(dual_new, dual)


def test_project_affine_kl_orthonormal_is_stable_for_large_dynamic_range() -> None:
    """Project onto a sum constraint without losing positivity or proportionality."""
    sqrt3 = math.sqrt(3.0)
    Q = torch.tensor([[1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3]], dtype=torch.float64)
    c = torch.tensor([3.0 / sqrt3], dtype=torch.float64)
    k = torch.tensor([1e-12, 1.0, 1e12], dtype=torch.float64)
    x_proj, info, _ = project_affine_kl_orthonormal(
        k,
        Q,
        c,
        tol=1e-10,
        max_iters_proj=200,
        max_iters_ls=25,
        min_entry=1e-24,
    )
    ratios = x_proj / k

    assert info["converged"]
    assert torch.all(x_proj > 0)
    assert x_proj.sum().item() == pytest.approx(3.0, abs=1e-10)
    assert torch.allclose(
        ratios,
        torch.full_like(ratios, ratios.mean()),
        rtol=1e-10,
        atol=1e-24,
    )


def test_project_affine_kl_orthonormal_projects_markov_generated_counts(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_markov_count_data: Dict[str, torch.Tensor | object],
) -> None:
    """Project a sampled count matrix onto the commuting affine subset."""
    count_matrices = simple_markov_count_data["count_matrices"]
    assert isinstance(count_matrices, torch.Tensor)

    K = count_matrices[1] + 1e-6
    A, b = build_commuting_constraints_flux(simple_column_stochastic_matrix)
    Q, c, rank = orthonormalize_affine_constraints(A, b, tol=1e-12, linear=True)
    x_proj, info, _ = project_affine_kl_orthonormal(
        K.reshape(-1),
        Q,
        c,
        tol=1e-10,
        max_iters_proj=200,
        max_iters_ls=25,
        min_entry=1e-24,
    )
    flux = x_proj.reshape_as(K)
    commutator = simple_column_stochastic_matrix @ flux - flux @ simple_column_stochastic_matrix.T

    assert rank == Q.shape[0]
    assert info["converged"]
    assert torch.isfinite(flux).all()
    assert torch.all(flux > 0)
    assert torch.max(torch.abs(commutator)).item() <= 1e-10


def test_project_affine_kl_orthonormal_rejects_non_1d_k() -> None:
    """Reject reference vectors that are not one-dimensional."""
    with pytest.raises(ValueError, match="k must be 1D"):
        project_affine_kl_orthonormal(
            torch.ones((2, 2), dtype=torch.float64),
            torch.zeros((0, 4), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
        )


def test_project_affine_kl_orthonormal_rejects_non_1d_c() -> None:
    """Reject affine right-hand sides that are not one-dimensional."""
    with pytest.raises(ValueError, match="c must be 1D"):
        project_affine_kl_orthonormal(
            torch.ones((4,), dtype=torch.float64),
            torch.zeros((1, 4), dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
        )


def test_project_affine_kl_orthonormal_rejects_non_1d_dual0() -> None:
    """Reject dual warm starts that are not one-dimensional."""
    with pytest.raises(ValueError, match="dual0 must be 1D"):
        project_affine_kl_orthonormal(
            torch.ones((1,), dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.float64),
            torch.ones((1,), dtype=torch.float64),
            dual0=torch.zeros((1, 1), dtype=torch.float64),
        )
