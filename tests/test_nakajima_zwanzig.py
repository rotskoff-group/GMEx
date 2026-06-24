# tests/test_nakajima_zwanzig.py
# test gmex/nakajima_zwanzig.py classes and helpers


import pytest
import torch
from torch.linalg import matrix_exp, matrix_power

from gmex.markov_process import MarkovChain, MarkovJumpProcess
from gmex.nakajima_zwanzig import (
    convolve_sum,
    DiscreteTimeGroundTruthNZGME,
    extract_Us_dtnzgme,
    get_Ks_from_Us,
    TransferMatrixNZGME
)


def _build_example_Us(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> torch.Tensor:
    """Build the shared 4-slice transfer-matrix stack used in NZ tests."""
    stack = simple_inhomogeneous_column_stochastic_stack
    eye = torch.eye(
        simple_column_stochastic_matrix.shape[0],
        dtype=torch.float64,
        device=simple_column_stochastic_matrix.device
    )
    return torch.stack((eye, simple_column_stochastic_matrix, stack[1], stack[2]))


def _build_ground_truth_Us(
    model: DiscreteTimeGroundTruthNZGME,
    L: torch.Tensor,
    n_steps: int
) -> torch.Tensor:
    """Build transfer matrices from analytical coarse-graining."""
    return torch.stack([model.get_U_dtgme(L, m) for m in range(n_steps + 1)])


def test_get_Ks_from_Us_matches_base_case_and_left_recursion(
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Match the base case and first left-recursion step exactly."""
    Us = simple_inhomogeneous_column_stochastic_stack
    Ks = get_Ks_from_Us(Us, verbose=False)

    assert Ks.shape == (Us.shape[0] - 1, Us.shape[1], Us.shape[2])
    assert torch.allclose(Ks[0], Us[1], atol=1e-12, rtol=0.0)
    assert torch.allclose(Ks[1], Us[2] - Us[1] @ Ks[0], atol=1e-12, rtol=0.0)


def test_get_Ks_from_Us_left_and_right_recursions_agree_for_noncommuting_case(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Left and right recursions agree even in a noncommuting example."""
    stack = simple_inhomogeneous_column_stochastic_stack
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    left = Us[3] - (Us[1] @ Ks[1] + Us[2] @ Ks[0])
    right = Us[3] - (Ks[1] @ Us[1] + Ks[0] @ Us[2])

    assert not torch.allclose(
        stack[1] @ stack[2],
        stack[2] @ stack[1],
        atol=1e-12,
        rtol=0.0
    )
    assert not torch.allclose(
        Us[1] @ Ks[1],
        Ks[1] @ Us[1],
        atol=1e-12,
        rtol=0.0
    )
    assert torch.allclose(Ks[2], left, atol=1e-12, rtol=0.0)
    assert torch.allclose(Ks[2], right, atol=1e-12, rtol=0.0)
    assert torch.allclose(left, right, atol=1e-12, rtol=0.0)


def test_get_Ks_from_Us_vanishes_for_matrix_powers(
    simple_column_stochastic_matrix: torch.Tensor
) -> None:
    """Kernels at nonzero delays vanish for a homogeneous Markov stack."""
    U1 = simple_column_stochastic_matrix
    Us = torch.stack([matrix_power(U1, i) for i in range(5)])
    Ks = get_Ks_from_Us(Us, verbose=False)

    assert torch.allclose(Ks[0], U1, atol=1e-12, rtol=0.0)
    assert torch.allclose(Ks[1:], torch.zeros_like(Ks[1:]), atol=1e-12, rtol=0.0)


def test_transfer_matrix_nzgme_get_Ks_matches_helper(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """TransferMatrixNZGME should delegate kernel extraction to the helper."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    model = TransferMatrixNZGME(Us)
    Ks = get_Ks_from_Us(Us, verbose=False)

    assert model.n_macrostates == Us.shape[1]
    assert model.device == Us.device
    assert torch.equal(model.Us, Us)
    assert torch.allclose(model.get_Ks(verbose=False), Ks, atol=1e-12, rtol=0.0)


def test_transfer_matrix_nzgme_integrate_matches_transfer_matrix_evolution(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Integrating with supplied kernels should reproduce the transfer matrices."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    initial_dist = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    model = TransferMatrixNZGME(Us)

    returned_Ks, dists = model.integrate(
        Us.shape[0] - 1,
        kernels=Ks,
        initial_dist=initial_dist,
        verbose=False
    )
    expected_dists = torch.stack([Us[t] @ initial_dist for t in range(Us.shape[0])])

    assert torch.allclose(returned_Ks, Ks, atol=1e-12, rtol=0.0)
    assert torch.allclose(dists, expected_dists, atol=1e-12, rtol=0.0)


def test_transfer_matrix_nzgme_integrate_computes_kernels_when_omitted(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Integrating without kernels should compute the same kernels internally."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    initial_dist = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    model = TransferMatrixNZGME(Us)

    returned_Ks, dists = model.integrate(
        Us.shape[0] - 1,
        initial_dist=initial_dist,
        verbose=False
    )
    expected_dists = torch.stack([Us[t] @ initial_dist for t in range(Us.shape[0])])

    assert torch.allclose(returned_Ks, Ks, atol=1e-12, rtol=0.0)
    assert torch.allclose(dists, expected_dists, atol=1e-12, rtol=0.0)


def test_discrete_time_ground_truth_nzgme_initialization(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_groups: list[list[int]],
    simple_macrostate_indicator: torch.Tensor
) -> None:
    """Ground-truth DT-NZGME should store the expected coarse-graining state."""
    process = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    model = DiscreteTimeGroundTruthNZGME(process, simple_groups)
    Us = _build_ground_truth_Us(model, model.L, n_steps=4)

    assert model.n_macrostates == len(simple_groups)
    assert model.dt == 0.5
    assert model.device == process.device
    assert model.groups == simple_groups
    assert torch.equal(model.L, process.L)
    assert torch.equal(model.macrostate_indicator, simple_macrostate_indicator)
    assert torch.allclose(
        Us.sum(dim=1),
        torch.ones_like(Us.sum(dim=1)),
        atol=1e-12,
        rtol=0.0
    )


def test_discrete_time_ground_truth_nzgme_get_U_dtgme_matches_manual_computation(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_groups: list[list[int]],
    simple_macrostate_indicator: torch.Tensor
) -> None:
    """Ground-truth transfer matrices should match manual coarse-graining."""
    process = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    model = DiscreteTimeGroundTruthNZGME(process, simple_groups)
    pi = process.stationary_distribution()
    p_micro_given_macro = simple_macrostate_indicator * pi[:, None]
    p_macro = simple_macrostate_indicator.T @ pi
    p_micro_given_macro = p_micro_given_macro / p_macro
    identity = torch.eye(model.n_macrostates, dtype=torch.float64)

    for m in [0, 1, 3]:
        observed = model.get_U_dtgme(model.L, m)
        expected = simple_macrostate_indicator.T @ matrix_power(model.L, m) @ p_micro_given_macro
        assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)

    assert torch.allclose(model.get_U_dtgme(model.L, 0), identity, atol=1e-12, rtol=0.0)


def test_discrete_time_ground_truth_nzgme_kernels_match_transfer_matrix_nzgme(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_groups: list[list[int]]
) -> None:
    """Kernels from exact coarse-graining should match transfer-matrix kernels."""
    process = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    model = DiscreteTimeGroundTruthNZGME(process, simple_groups)
    n_steps = 4
    Us = _build_ground_truth_Us(model, model.L, n_steps)
    transfer_kernels = TransferMatrixNZGME(Us).get_Ks(verbose=False)
    ground_truth_kernels = torch.stack([
        model.get_K_dtgme(model.L, m) for m in range(n_steps)
    ])

    assert torch.allclose(
        transfer_kernels,
        ground_truth_kernels,
        atol=1e-12,
        rtol=0.0
    )


def test_discrete_time_ground_truth_nzgme_integrate_matches_transfer_matrix_nzgme(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_groups: list[list[int]]
) -> None:
    """Ground-truth integration should match integration from exact transfer matrices."""
    process = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    model = DiscreteTimeGroundTruthNZGME(process, simple_groups)
    n_steps = 4
    initial_dist = torch.tensor([0.4, 0.6], dtype=torch.float64)
    L = matrix_power(model.L, 2)
    Us = _build_ground_truth_Us(model, L, n_steps)
    transfer_model = TransferMatrixNZGME(Us)
    transfer_kernels = transfer_model.get_Ks(verbose=False)

    ground_truth_kernels, ground_truth_dists = model.integrate(
        deltat=1.0,
        n_steps=n_steps,
        initial_dist=initial_dist,
        verbose=False
    )
    returned_transfer_kernels, transfer_dists = transfer_model.integrate(
        n_steps,
        kernels=transfer_kernels,
        initial_dist=initial_dist,
        verbose=False
    )

    assert torch.allclose(
        ground_truth_kernels,
        returned_transfer_kernels,
        atol=1e-12,
        rtol=0.0
    )
    assert torch.allclose(ground_truth_dists, transfer_dists, atol=1e-12, rtol=0.0)


def test_discrete_time_ground_truth_nzgme_markov_jump_process_matches_transfer_matrix_nzgme(
    simple_cycle_rates: torch.Tensor,
    simple_groups: list[list[int]]
) -> None:
    """MJP-based ground-truth kernels and integrals should match transfer matrices."""
    process = MarkovJumpProcess(device="cpu")
    process.parameterize_from_rates(simple_cycle_rates.to(torch.float64))
    model = DiscreteTimeGroundTruthNZGME(process, simple_groups)
    deltat = 0.25
    n_steps = 4
    initial_dist = torch.tensor([0.4, 0.6], dtype=torch.float64)
    L = matrix_exp(model.trm * deltat)
    L = L / L.sum(dim=0, keepdim=True)
    Us = _build_ground_truth_Us(model, L, n_steps)
    transfer_model = TransferMatrixNZGME(Us)
    transfer_kernels = transfer_model.get_Ks(verbose=False)
    direct_kernels = torch.stack([
        model.get_K_dtgme(L, m) for m in range(n_steps)
    ])

    ground_truth_kernels, ground_truth_dists = model.integrate(
        deltat=deltat,
        n_steps=n_steps,
        initial_dist=initial_dist,
        verbose=False
    )
    returned_transfer_kernels, transfer_dists = transfer_model.integrate(
        n_steps,
        kernels=transfer_kernels,
        initial_dist=initial_dist,
        verbose=False
    )

    assert torch.allclose(transfer_kernels, direct_kernels, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        ground_truth_kernels,
        returned_transfer_kernels,
        atol=1e-12,
        rtol=0.0
    )
    assert torch.allclose(ground_truth_dists, transfer_dists, atol=1e-12, rtol=0.0)


def test_discrete_time_ground_truth_nzgme_rejects_invalid_inputs(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor,
    simple_groups: list[list[int]]
) -> None:
    """Reject unsupported processes, invalid groups, timesteps, and kernels."""
    process = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    model = DiscreteTimeGroundTruthNZGME(process, simple_groups)
    inhomogeneous_process = MarkovChain(
        simple_inhomogeneous_column_stochastic_stack,
        dt=0.5,
        device="cpu"
    )

    with pytest.raises(ValueError, match="inhomogeneous Markov chains"):
        DiscreteTimeGroundTruthNZGME(inhomogeneous_process, simple_groups)
    with pytest.raises(AssertionError, match="not grouped"):
        DiscreteTimeGroundTruthNZGME(process, [[0, 1], [2]])
    with pytest.raises(ValueError, match="integer multiple"):
        model.integrate(0.75, n_steps=2, verbose=False)
    with pytest.raises(AssertionError, match="kernels must be 3D"):
        model.integrate(
            0.5,
            n_steps=2,
            kernels=torch.zeros((2, 2), dtype=torch.float64),
            verbose=False
        )
    with pytest.raises(AssertionError, match="kernels must have square slices"):
        model.integrate(
            0.5,
            n_steps=2,
            kernels=torch.zeros((2, 2, 3), dtype=torch.float64),
            verbose=False
        )


def test_transfer_matrix_nzgme_rejects_invalid_shapes(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Reject invalid transfer-matrix and kernel shapes."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    model = TransferMatrixNZGME(Us)

    with pytest.raises(AssertionError, match="ps must have same dimensions 1 and 2"):
        TransferMatrixNZGME(torch.zeros((3, 4, 5), dtype=torch.float64))
    with pytest.raises(AssertionError, match="kernels must be 3D"):
        model.integrate(Us.shape[0] - 1, kernels=torch.zeros((3, 4), dtype=torch.float64), verbose=False)
    with pytest.raises(AssertionError, match="kernels must have square slices"):
        model.integrate(Us.shape[0] - 1, kernels=torch.zeros((3, 4, 3), dtype=torch.float64), verbose=False)


def test_convolve_sum_matches_manual_sum_with_full_available_history(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Convolution should match the explicit sum when enough kernels are present."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    dists = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.2, 0.1, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25]
        ],
        dtype=torch.float64
    )
    observed = convolve_sum(1, Ks, dists)
    expected = Ks[0] @ dists[1] + Ks[1] @ dists[0]

    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_convolve_sum_matches_manual_sum_when_delay_exceeds_kernel_horizon(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Convolution should use only the available kernels at long delay."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    dists = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.2, 0.1, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25]
        ],
        dtype=torch.float64
    )
    observed = convolve_sum(3, Ks[:2], dists)
    expected = Ks[0] @ dists[3] + Ks[1] @ dists[2]

    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_convolve_sum_rejects_invalid_shapes() -> None:
    """Reject nonsquare kernels and mismatched distribution support."""
    with pytest.raises(AssertionError, match="kernel at each timestep must be square"):
        convolve_sum(
            1,
            torch.zeros((2, 4, 3), dtype=torch.float64),
            torch.zeros((3, 4), dtype=torch.float64)
        )
    with pytest.raises(
        AssertionError,
        match="dist support cardinality must equal second dimension of kernels"
    ):
        convolve_sum(
            1,
            torch.zeros((2, 4, 4), dtype=torch.float64),
            torch.zeros((3, 3), dtype=torch.float64)
        )


def test_extract_Us_dtnzgme_recovers_full_transfer_matrix_stack(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor
) -> None:
    """Extracted transfer matrices should reproduce the full input stack."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    extracted = extract_Us_dtnzgme(Ks)

    assert torch.allclose(extracted, Us, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("max_delay", [0, 1])
def test_extract_Us_dtnzgme_recovers_prefix_for_truncated_memory(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor,
    max_delay: int
) -> None:
    """Truncated memory should still recover the exact initial prefix."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack,
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    extracted = extract_Us_dtnzgme(Ks, max_delay=max_delay)

    assert torch.allclose(
        extracted[:max_delay + 2],
        Us[:max_delay + 2],
        atol=1e-12,
        rtol=0.0,
    )


def test_extract_Us_dtnzgme_with_zero_delay_matches_markovian_powers(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor,
) -> None:
    """Zero-delay extraction should reduce to powers of the Markovian kernel."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack,
    )
    Ks = get_Ks_from_Us(Us, verbose=False)
    extracted = extract_Us_dtnzgme(Ks, max_delay=0)
    expected = torch.stack([matrix_power(Us[1], i) for i in range(Us.shape[0])])

    assert torch.allclose(extracted, expected, atol=1e-12, rtol=0.0)


def test_extract_Us_dtnzgme_rejects_negative_max_delay(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor,
) -> None:
    """Negative max_delay should be rejected explicitly."""
    Us = _build_example_Us(
        simple_column_stochastic_matrix,
        simple_inhomogeneous_column_stochastic_stack,
    )
    Ks = get_Ks_from_Us(Us, verbose=False)

    with pytest.raises(ValueError, match="max_delay"):
        extract_Us_dtnzgme(Ks, max_delay=-1)
