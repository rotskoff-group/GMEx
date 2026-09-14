# tests/test_max_likelihood.py
# test gmex/max_likelihood.py classes and helpers


import warnings
from typing import TypedDict

import numpy as np
import pytest
import torch
from deeptime.markov import TransitionCountEstimator, msm

from gmex.max_likelihood import (
    DeeptimeReversibleU,
    MirrorDescentG,
    MirrorDescentU,
    get_Gs_mle,
    get_reversible_Us_deeptime,
    get_Us_mle,
)
from gmex.utils.core import check_stationary_column_stochastic_matrix
from tests.fixture_types import InhomogeneousMarkovCountData, MarkovCountData


def test_reversible_u_satisfies_constraints(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Get maximum-likelihood estimate detailed-balance wrt specified dist."""
    count_matrices = simple_markov_count_data["count_matrices"]
    mle = MirrorDescentU(
        count_matrices[1], simple_stationary_dist, reversible=True, tol=1e-12
    )
    U, _ = mle.fit()
    check_stationary_column_stochastic_matrix(
        U, simple_stationary_dist, db=True, tol=1e-12
    )


def test_nonreversible_u_satisfies_constraints(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Get maximum-likelihood estimate stationary wrt specified dist."""
    count_matrices = simple_markov_count_data["count_matrices"]
    mle = MirrorDescentU(
        count_matrices[1], simple_stationary_dist, reversible=False, tol=1e-12
    )
    U, _ = mle.fit()
    check_stationary_column_stochastic_matrix(
        U, simple_stationary_dist, db=False, tol=1e-12
    )


def test_reversible_g_satisfies_constraints(
    reversible_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Get maximum-likelihood estimate commuting with U_prev and detailed-balance wrt specified dist."""
    count_matrices = reversible_inhomogeneous_markov_count_data["count_matrices"]
    transition_matrices = reversible_inhomogeneous_markov_count_data["transition_stack"]
    mle = MirrorDescentG(
        count_matrices[2],
        transition_matrices[1],
        reversible_stationary_dist,
        reversible=True,
        tol=1e-12,
    )
    G, _ = mle.fit()
    check_stationary_column_stochastic_matrix(
        G, reversible_stationary_dist, db=True, tol=1e-12
    )
    assert (
        G @ transition_matrices[1] - transition_matrices[1] @ G
    ).abs().max() <= 1e-12


def test_nonreversible_g_satisfies_constraints(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Get maximum-likelihood estimate stationary wrt specified dist."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    transition_matrices = simple_inhomogeneous_markov_count_data["transition_stack"]
    mle = MirrorDescentG(
        count_matrices[2],
        transition_matrices[1],
        simple_stationary_dist,
        reversible=False,
        tol=1e-12,
    )
    G, _ = mle.fit()
    check_stationary_column_stochastic_matrix(
        G, simple_stationary_dist, db=False, tol=1e-12
    )


def test_reversible_u_converges_asymptotically(
    reversible_column_stochastic_matrix: torch.Tensor,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Get true reversible transition matrix in asymptotic limit."""
    count_matrix = (
        reversible_column_stochastic_matrix * reversible_stationary_dist[None, :] * 1e6
    )
    mle = MirrorDescentU(count_matrix, reversible_stationary_dist, reversible=True)
    U, _ = mle.fit()
    assert torch.allclose(reversible_column_stochastic_matrix, U)


def test_nonreversible_u_converges_asymptotically(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Get true nonreversible transition matrix in asymptotic limit."""
    count_matrix = (
        simple_column_stochastic_matrix * simple_stationary_dist[None, :] * 1e6
    )
    mle = MirrorDescentU(count_matrix, simple_stationary_dist, reversible=False)
    U, _ = mle.fit()
    assert torch.allclose(simple_column_stochastic_matrix, U)


def test_reversible_g_converges_asymptotically(
    reversible_inhomogeneous_column_stochastic_stack: torch.Tensor,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Get true reversible propagator in asymptotic limit."""
    U_prev = reversible_inhomogeneous_column_stochastic_stack[1].clone()
    G_true = reversible_inhomogeneous_column_stochastic_stack[2]
    U_true = G_true @ U_prev
    count_matrix = U_true * reversible_stationary_dist[None, :] * 1e6
    mle = MirrorDescentG(
        count_matrix, U_prev, reversible_stationary_dist, reversible=True
    )
    G_est, _ = mle.fit()
    assert torch.allclose(G_true, G_est)


def test_nonreversible_g_converges_asymptotically(
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Get true nonreversible propagator in asymptotic limit."""
    U_prev = simple_inhomogeneous_column_stochastic_stack[1].clone()
    G_true = simple_inhomogeneous_column_stochastic_stack[2]
    U_true = G_true @ U_prev
    count_matrix = U_true * simple_stationary_dist[None, :] * 1e6
    mle = MirrorDescentG(count_matrix, U_prev, simple_stationary_dist, reversible=False)
    G_est, _ = mle.fit()
    assert torch.allclose(G_true, G_est)


def test_reversible_u_and_g_agree_when_uprev_is_eye(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reversible MirrorDescentU and MirrorDescentG get the same thing at first lag."""
    transition_matrices = simple_inhomogeneous_markov_count_data["transition_stack"]
    U0 = torch.eye(
        transition_matrices.shape[1],
        dtype=transition_matrices.dtype,
        device=transition_matrices.device,
    )
    count_matrix = simple_inhomogeneous_markov_count_data["count_matrices"][1]
    mleu = MirrorDescentU(count_matrix, simple_stationary_dist, reversible=True)
    U1, _ = mleu.fit(line_search=False)
    mleg = MirrorDescentG(count_matrix, U0, simple_stationary_dist, reversible=True)
    G1, _ = mleg.fit(line_search=False)
    assert torch.allclose(U1, G1)


def test_nonreversible_u_and_g_agree_when_uprev_is_eye(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Nonreversible MirrorDescentU and MirrorDescentG get the same thing at first lag."""
    transition_matrices = simple_inhomogeneous_markov_count_data["transition_stack"]
    U0 = torch.eye(
        transition_matrices.shape[1],
        dtype=transition_matrices.dtype,
        device=transition_matrices.device,
    )
    count_matrix = simple_inhomogeneous_markov_count_data["count_matrices"][1]
    mleu = MirrorDescentU(count_matrix, simple_stationary_dist, reversible=False)
    U1, _ = mleu.fit(line_search=False)
    mleg = MirrorDescentG(count_matrix, U0, simple_stationary_dist, reversible=False)
    G1, _ = mleg.fit(line_search=False)
    assert torch.allclose(U1, G1)


def test_reversible_u_agrees_with_deeptime(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reversible MirrorDescentU gets Prinz-Trendelkamp-Schroer estimate."""
    count_matrices = simple_markov_count_data["count_matrices"]
    mle = MirrorDescentU(
        count_matrices[1], simple_stationary_dist, reversible=True, tol=1e-12
    )
    U, _ = mle.fit(line_search=False)

    records = simple_markov_count_data["dataset"].data
    counts_estimator_deeptime = TransitionCountEstimator(
        lagtime=1, count_mode="sliding"
    )
    counts_deeptime = counts_estimator_deeptime.fit(np.array(records)).fetch_model()
    msm_estimator_deeptime = msm.MaximumLikelihoodMSM(
        reversible=True,
        maxerr=1e-24,
        transition_matrix_tolerance=1e-24,
        stationary_distribution_constraint=simple_stationary_dist.cpu().numpy(),
    )
    msm_deeptime = msm_estimator_deeptime.fit(counts_deeptime).fetch_model()
    U_deeptime = torch.Tensor(msm_deeptime.transition_matrix).to(U).T

    assert torch.allclose(U, U_deeptime)


def test_reversible_g_agrees_with_deeptime_when_uprev_is_eye(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reversible MirrorDescentG gets Prinz-Trendelkamp-Schroer estimate at first lag."""
    count_matrices = simple_markov_count_data["count_matrices"]
    transition_matrix = simple_markov_count_data["transition_matrix"]
    U_prev = torch.eye(
        transition_matrix.shape[1],
        dtype=transition_matrix.dtype,
        device=transition_matrix.device,
    )
    mle = MirrorDescentG(
        count_matrices[1], U_prev, simple_stationary_dist, reversible=True, tol=1e-12
    )
    G1, _ = mle.fit(line_search=False)

    records = simple_markov_count_data["dataset"].data
    counts_estimator_deeptime = TransitionCountEstimator(
        lagtime=1, count_mode="sliding"
    )
    counts_deeptime = counts_estimator_deeptime.fit(np.array(records)).fetch_model()
    msm_estimator_deeptime = msm.MaximumLikelihoodMSM(
        reversible=True,
        maxerr=1e-24,
        transition_matrix_tolerance=1e-24,
        stationary_distribution_constraint=simple_stationary_dist.cpu().numpy(),
    )
    msm_deeptime = msm_estimator_deeptime.fit(counts_deeptime).fetch_model()
    U_deeptime = torch.Tensor(msm_deeptime.transition_matrix).to(G1).T

    assert torch.allclose(G1, U_deeptime)


def test_deeptime_reversible_u_matches_manual_transposed_count_fit(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """DeeptimeReversibleU matches explicit transposed-count Deeptime usage."""
    count_matrices = simple_markov_count_data["count_matrices"]
    U_wrapper, info = DeeptimeReversibleU(
        count_matrices[1], simple_stationary_dist
    ).fit()
    msm_estimator_deeptime = msm.MaximumLikelihoodMSM(
        reversible=True,
        stationary_distribution_constraint=simple_stationary_dist.cpu().numpy(),
        sparse=False,
        allow_disconnected=False,
        maxiter=1000000,
        maxerr=1e-24,
        connectivity_threshold=0,
        transition_matrix_tolerance=1e-24,
        lagtime=None,
        use_lcc=False,
    )
    msm_deeptime = msm_estimator_deeptime.fit(
        count_matrices[1].T.cpu().numpy()
    ).fetch_model()
    U_manual = torch.Tensor(msm_deeptime.transition_matrix).to(U_wrapper).T
    torch.testing.assert_close(U_wrapper, U_manual, atol=1e-12, rtol=0.0)
    assert info == {"method": "deeptime"}


def test_get_us_mle_matches_manual_lagwise_fit_nonreversible(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Match an explicit per-lag nonreversible MirrorDescentU loop exactly."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    wrapper_out, metrics = get_Us_mle(
        count_matrices, simple_stationary_dist, reversible=False, verbose=False
    )

    manual_out = torch.zeros_like(wrapper_out)
    manual_out[0] = torch.eye(
        count_matrices.shape[1], dtype=wrapper_out.dtype, device=wrapper_out.device
    )
    for lag in range(1, count_matrices.shape[0]):
        estimator = MirrorDescentU(
            count_matrices[lag], simple_stationary_dist, reversible=False
        )
        manual_out[lag], _ = estimator.fit(verbose=False)

    assert torch.equal(wrapper_out, manual_out)
    assert torch.equal(wrapper_out[0], manual_out[0])
    assert set(metrics) == {1, 2}


def test_get_us_mle_matches_manual_lagwise_fit_reversible(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Match an explicit per-lag reversible MirrorDescentU loop exactly."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    wrapper_out, metrics = get_Us_mle(
        count_matrices, simple_stationary_dist, reversible=True, verbose=False
    )

    manual_out = torch.zeros_like(wrapper_out)
    manual_out[0] = torch.eye(
        count_matrices.shape[1], dtype=wrapper_out.dtype, device=wrapper_out.device
    )
    for lag in range(1, count_matrices.shape[0]):
        estimator = MirrorDescentU(
            count_matrices[lag], simple_stationary_dist, reversible=True
        )
        manual_out[lag], _ = estimator.fit(verbose=False)

    assert torch.equal(wrapper_out, manual_out)
    assert torch.equal(wrapper_out[0], manual_out[0])
    assert set(metrics) == {1, 2}
    for lag in range(1, wrapper_out.shape[0]):
        flux = wrapper_out[lag] * simple_stationary_dist[None, :]
        assert torch.max(torch.abs(flux - flux.T)).item() <= 1e-12


def test_get_reversible_us_deeptime_matches_manual_lagwise_fit(
    reversible_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Match an explicit lagwise reversible DeeptimeReversibleU loop exactly."""
    count_matrices = reversible_inhomogeneous_markov_count_data["count_matrices"]
    wrapper_out, metrics = get_reversible_Us_deeptime(
        count_matrices, reversible_stationary_dist, verbose=False
    )
    manual_out = torch.zeros_like(wrapper_out)
    manual_out[0] = torch.eye(
        count_matrices.shape[1], dtype=wrapper_out.dtype, device=wrapper_out.device
    )
    for lag in range(1, count_matrices.shape[0]):
        estimator = DeeptimeReversibleU(count_matrices[lag], reversible_stationary_dist)
        manual_out[lag], _ = estimator.fit()
    torch.testing.assert_close(wrapper_out, manual_out, atol=1e-12, rtol=0.0)
    assert torch.equal(wrapper_out[0], manual_out[0])
    assert set(metrics) == {1, 2}
    assert all(info == {"method": "deeptime"} for info in metrics.values())


def test_get_gs_mle_matches_manual_lagwise_fit_nonreversible(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Match an explicit lagwise nonreversible MirrorDescentG loop exactly."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    transition_matrices = simple_inhomogeneous_markov_count_data["transition_stack"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper_out, metrics = get_Gs_mle(
            count_matrices, simple_stationary_dist, reversible=False, verbose=False
        )
        manual_out = torch.zeros_like(wrapper_out)
        manual_out[0] = torch.eye(
            transition_matrices.shape[1],
            dtype=wrapper_out.dtype,
            device=wrapper_out.device,
        )
        U_prev = manual_out[0].clone()
        for lag in range(1, count_matrices.shape[0]):
            estimator = MirrorDescentG(
                count_matrices[lag], U_prev, simple_stationary_dist, reversible=False
            )
            manual_out[lag], _ = estimator.fit(
                verbose=False, G0=None if lag == 1 else manual_out[lag - 1]
            )
            U_prev = manual_out[lag] @ U_prev

    torch.testing.assert_close(wrapper_out, manual_out, atol=1e-12, rtol=0.0)
    assert torch.equal(wrapper_out[0], manual_out[0])
    assert set(metrics) == {1, 2}


def test_get_gs_mle_reversible_sets_second_lag_to_first(
    reversible_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Set the second reversible propagator equal to the first."""
    count_matrices = reversible_inhomogeneous_markov_count_data["count_matrices"]
    Gs, metrics = get_Gs_mle(
        count_matrices, reversible_stationary_dist, reversible=True, verbose=False
    )

    torch.testing.assert_close(Gs[1], Gs[2], atol=1e-12, rtol=0.0)
    assert set(metrics) == {1, 2}


def test_get_gs_mle_reversible_rejects_precomputed_mismatched_first_two_lags(
    reversible_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Reject reversible precomputed propagators whose first two lags differ."""
    count_matrices = reversible_inhomogeneous_markov_count_data["count_matrices"]
    precomputed_gs = reversible_inhomogeneous_markov_count_data["transition_stack"]

    with pytest.raises(
        ValueError, match=r"Gs_precomputed\[1\] must equal Gs_precomputed\[2\]"
    ):
        get_Gs_mle(
            count_matrices,
            reversible_stationary_dist,
            reversible=True,
            precomputed=(precomputed_gs, {1: {}, 2: {}}),
            verbose=False,
        )


def test_get_gs_mle_resumes_from_precomputed_checkpoint(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reuse precomputed propagators and accept string lag keys in metadata."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_out, full_metrics = get_Gs_mle(
            count_matrices, simple_stationary_dist, reversible=False, verbose=False
        )
        resumed_out, resumed_metrics = get_Gs_mle(
            count_matrices,
            simple_stationary_dist,
            reversible=False,
            precomputed=(full_out[:2], {"1": full_metrics[1]}),
            verbose=False,
        )

    torch.testing.assert_close(resumed_out, full_out, atol=1e-12, rtol=0.0)
    assert set(resumed_metrics) == {1, 2}
    assert resumed_metrics[1] == full_metrics[1]


def test_mirror_descent_u_rejects_zero_counts(
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reject an all-zero count matrix at construction time."""
    with pytest.raises(ValueError, match="Counts cannot total 0"):
        MirrorDescentU(torch.zeros((4, 4), dtype=torch.float64), simple_stationary_dist)


def test_mirror_descent_u_rejects_nonpositive_min_entry(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reject a nonpositive lower clamp value."""
    count_matrices = simple_markov_count_data["count_matrices"]
    with pytest.raises(ValueError, match="min_entry must be positive"):
        MirrorDescentU(count_matrices[1], simple_stationary_dist, min_entry=0.0)


class _FitControlOverrides(TypedDict, total=False):
    """Optional fit controls used to exercise input validation."""

    max_iters: int
    max_iters_ls: int
    ls_update: float


@pytest.mark.parametrize(
    ("fit_kwargs", "match"),
    [
        ({"max_iters": 0}, "max_iters must be at least 1"),
        ({"max_iters_ls": 0}, "max_iters_ls must be at least 1"),
        ({"ls_update": 0.0}, "ls_update must be in"),
        ({"ls_update": 1.5}, "ls_update must be in"),
    ],
)
def test_mirror_descent_u_fit_rejects_invalid_iteration_controls(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
    fit_kwargs: _FitControlOverrides,
    match: str,
) -> None:
    """Validate shared fit controls through MirrorDescentU."""
    count_matrices = simple_markov_count_data["count_matrices"]
    estimator = MirrorDescentU(count_matrices[1], simple_stationary_dist)
    with pytest.raises(ValueError, match=match):
        estimator.fit(verbose=False, **fit_kwargs)


def test_mirror_descent_g_rejects_shape_mismatch(
    simple_markov_count_data: MarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reject a previous transition matrix with the wrong shape."""
    count_matrices = simple_markov_count_data["count_matrices"]
    bad_u_prev = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="same shape"):
        MirrorDescentG(count_matrices[1], bad_u_prev, simple_stationary_dist)


def test_mirror_descent_g_fit_rejects_invalid_g0_shape(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Reject an initial propagator with the wrong shape."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    transition_matrices = simple_inhomogeneous_markov_count_data["transition_stack"]
    estimator = MirrorDescentG(
        count_matrices[2],
        transition_matrices[1],
        simple_stationary_dist,
    )
    with pytest.raises(ValueError, match="Expected G0 shape"):
        estimator.fit(G0=torch.eye(3, dtype=torch.float64), verbose=False)


def test_mirror_descent_g_warns_and_repairs_invalid_uprev(
    simple_inhomogeneous_markov_count_data: InhomogeneousMarkovCountData,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Warn and repair a slightly invalid previous transition matrix."""
    count_matrices = simple_inhomogeneous_markov_count_data["count_matrices"]
    transition_matrices = simple_inhomogeneous_markov_count_data["transition_stack"]
    bad_u_prev = transition_matrices[1].clone()
    bad_u_prev[0, 0] += 0.02

    estimator = MirrorDescentG(
        count_matrices[2],
        bad_u_prev,
        simple_stationary_dist,
        tol=1e-6,
    )
    with pytest.warns(UserWarning, match="Attempting matrix-scaling fix"):
        G, _ = estimator.fit(
            max_iters=100,
            max_iters_proj=100,
            verbose=False,
        )

    check_stationary_column_stochastic_matrix(
        G, simple_stationary_dist, db=False, tol=1e-6
    )
