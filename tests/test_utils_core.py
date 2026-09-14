# tests/test_core_utils.py
# test gmex.utils.core helpers


import networkx as nx
import pytest
import torch

from gmex.utils.core import *


def test_coarsen_states():
    """Test trajectory coarse graining.

    Parameters
    ----------
    device_for_testing : str
        Device to run tests on (e.g., 'cpu', 'cuda').
    """
    fine_states = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    groups = [[0, 1, 4], [2, 3]]
    coarse_states = coarsen_states(fine_states, groups)
    assert isinstance(coarse_states, torch.Tensor)
    expected_coarse_states = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    assert torch.equal(coarse_states, expected_coarse_states)


def test_get_random_init_dist_returns_normalized_distribution() -> None:
    """Return a nonnegative float64 distribution that sums to one."""
    initial_dist = get_random_init_dist(4)
    assert initial_dist.shape == (4,)
    assert initial_dist.dtype == torch.float64
    assert torch.all(initial_dist >= 0)
    assert torch.isclose(initial_dist.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_allocate_dists_uses_provided_initial_dist() -> None:
    """Allocate a distribution history and store the initial distribution in row zero."""
    initial_dist = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    dists = allocate_dists(n_steps=3, n_states=4, initial_dist=initial_dist)
    assert dists.shape == (4, 4)
    assert dists.dtype == torch.float64
    assert torch.allclose(dists[0], initial_dist)
    assert torch.allclose(dists[1:], torch.zeros((3, 4), dtype=torch.float64))


def test_allocate_dists_requires_state_information() -> None:
    """Raise when neither the number of states nor an initial distribution is provided."""
    with pytest.raises(
        ValueError, match="Either n_states or initial_dist must be provided"
    ):
        allocate_dists(n_steps=2)


def test_allocate_dists_rejects_multidimensional_initial_dist() -> None:
    """Reject an initial distribution that remains multidimensional after squeezing."""
    initial_dist = torch.full((2, 2), 0.25, dtype=torch.float64)

    with pytest.raises(
        ValueError, match="initial_dist must be one-dimensional after squeezing"
    ):
        allocate_dists(n_steps=2, initial_dist=initial_dist)


def test_validate_groups_accepts_complete_partition(
    simple_groups: list[list[int]],
) -> None:
    """Accept a valid partition of four microstates."""
    validate_groups(4, simple_groups)


def test_validate_groups_rejects_duplicate_state() -> None:
    """Reject groupings that assign a microstate more than once."""
    with pytest.raises(AssertionError, match="grouped"):
        validate_groups(4, [[0, 1], [1, 2], [3]])


def test_validate_groups_rejects_missing_state() -> None:
    """Reject groupings that fail to cover all expected microstates."""
    with pytest.raises(AssertionError, match="not grouped"):
        validate_groups(4, [[0, 1], [2]])


def test_randomly_group_returns_valid_partition() -> None:
    """Return groups that cover each state exactly once."""
    groups = randomly_group(4, 2)
    flattened = sorted(state for group in groups for state in group)
    assert len(groups) == 2
    assert flattened == [0, 1, 2, 3]
    assert all(len(group) >= 1 for group in groups)


def test_randomly_group_equal_assignment_balances_group_sizes() -> None:
    """Assign states as evenly as possible when equal assignment is requested."""
    groups = randomly_group(4, 3, equal_assignment=True)
    group_sizes = sorted(len(group) for group in groups)
    assert sorted(state for group in groups for state in group) == [0, 1, 2, 3]
    assert group_sizes[-1] - group_sizes[0] <= 1


def test_get_macrostate_indicator_matches_shared_fixture(
    simple_groups: list[list[int]],
    simple_macrostate_indicator: torch.Tensor,
) -> None:
    """Build the expected 4x2 macrostate indicator matrix."""
    indicator = get_macrostate_indicator(simple_groups)
    assert torch.equal(indicator, simple_macrostate_indicator)


def test_get_p_micro_given_macro_returns_expected_conditionals(
    simple_macrostate_indicator: torch.Tensor,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Compute conditional microstate probabilities within each macrostate."""
    expected = torch.tensor(
        [
            [1.0 / 3.0, 0.0],
            [2.0 / 3.0, 0.0],
            [0.0, 3.0 / 7.0],
            [0.0, 4.0 / 7.0],
        ],
        dtype=torch.float64,
    )
    p_micro_given_macro = get_p_micro_given_macro(
        simple_macrostate_indicator, simple_stationary_dist
    )
    assert torch.allclose(p_micro_given_macro, expected)


def test_connected_erdos_renyi_returns_connected_adjacency_matrix() -> None:
    """Return a connected undirected adjacency matrix."""
    adjacency = connected_erdos_renyi(4, p=1.0)
    assert adjacency.shape == (4, 4)
    assert torch.equal(adjacency, adjacency.T)
    assert torch.equal(torch.diag(adjacency), torch.zeros(4, dtype=adjacency.dtype))
    assert nx.is_connected(nx.from_numpy_array(adjacency.numpy()))


def test_lim_dist_from_stochastic_recovers_shared_stationary_distribution(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Recover the stationary distribution of the shared 4x4 matrix."""
    stationary = column_stochastic_lim_dist(simple_column_stochastic_matrix)
    assert torch.allclose(stationary, simple_stationary_dist)


def test_check_nonnegative_square_matrix_accepts_valid_tensor() -> None:
    """Accept a nonnegative 4x4 tensor."""
    matrix = torch.ones((4, 4), dtype=torch.float64)
    check_nonnegative_square_matrix(matrix)


def test_check_nonnegative_square_matrix_rejects_non_2d_tensor() -> None:
    """Reject tensors that are not two-dimensional."""
    with pytest.raises(ValueError, match="must be 2D"):
        check_nonnegative_square_matrix(torch.ones(4, dtype=torch.float64))


def test_check_nonnegative_square_matrix_rejects_empty_tensor() -> None:
    """Reject empty tensors."""
    with pytest.raises(ValueError, match="must be non-empty"):
        check_nonnegative_square_matrix(torch.empty((0, 0), dtype=torch.float64))


def test_check_nonnegative_square_matrix_rejects_negative_entries() -> None:
    """Reject tensors with negative entries."""
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[0, 1] = -1.0
    with pytest.raises(ValueError, match="negative entries"):
        check_nonnegative_square_matrix(matrix)


def test_check_probability_vector_accepts_valid_distribution() -> None:
    """Accept a strictly positive probability vector of the expected length."""
    check_probability_vector(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64), 4)


def test_check_probability_vector_rejects_wrong_rank() -> None:
    """Reject probability inputs that are not one-dimensional."""
    with pytest.raises(ValueError, match="must be 1D"):
        check_probability_vector(torch.ones((2, 2), dtype=torch.float64), 4)


def test_check_probability_vector_rejects_wrong_length() -> None:
    """Reject probability inputs with the wrong length."""
    with pytest.raises(ValueError, match="must have shape"):
        check_probability_vector(torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64), 4)


def test_check_probability_vector_rejects_nonpositive_entries() -> None:
    """Reject probability inputs with zero or negative entries."""
    with pytest.raises(ValueError, match="cannot have .* entries"):
        check_probability_vector(
            torch.tensor([0.0, 0.2, 0.3, 0.5], dtype=torch.float64), 4
        )


def test_check_probability_vector_rejects_sum_not_one() -> None:
    """Reject probability inputs that do not sum to one."""
    with pytest.raises(ValueError, match="must sum to one"):
        check_probability_vector(
            torch.tensor([0.1, 0.2, 0.3, 0.3], dtype=torch.float64), 4
        )


def test_normalize_probability_vector_returns_float64_normalized_values() -> None:
    """Normalize a positive vector to a float64 probability vector."""
    normalized = normalize_probability_vector(
        torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    )
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    assert normalized.dtype == torch.float64
    assert torch.allclose(normalized, expected)
    assert torch.isclose(normalized.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_as_float64_converts_non_float64_tensor() -> None:
    """Convert tensors to float64 when needed."""
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    converted = as_float64(tensor)
    assert converted.dtype == torch.float64
    assert torch.allclose(converted, tensor.to(torch.float64))


def test_as_float64_returns_same_object_for_float64_tensor() -> None:
    """Return the original tensor when it is already float64."""
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert as_float64(tensor) is tensor


def test_nan_to_pos_returns_finite_input_unchanged() -> None:
    """Leave finite tensors unchanged."""
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert torch.equal(nan_to_pos(tensor), tensor)


def test_nan_to_pos_replaces_nonfinite_entries() -> None:
    """Replace nan and infinite values with positive finite values."""
    tensor = torch.tensor(
        [float("nan"), float("inf"), float("-inf")], dtype=torch.float64
    )
    replaced = nan_to_pos(tensor, min_entry=1e-12)
    expected = torch.tensor(
        [1e-12, torch.finfo(torch.float64).max, 1e-12], dtype=torch.float64
    )
    assert torch.equal(replaced, expected)


def test_symmetrize_matrix_returns_arithmetic_average() -> None:
    """Average a matrix with its transpose in the default mode."""
    matrix = torch.tensor([[1.0, 4.0], [2.0, 3.0]], dtype=torch.float32)
    expected = torch.tensor([[1.0, 3.0], [3.0, 3.0]], dtype=torch.float64)
    assert torch.equal(symmetrize_matrix(matrix), expected)


def test_symmetrize_matrix_returns_geometric_average() -> None:
    """Combine a matrix with its transpose using elementwise geometric means."""
    matrix = torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float32)
    expected = torch.tensor([[1.0, 6.0], [6.0, 16.0]], dtype=torch.float64)
    assert torch.equal(symmetrize_matrix(matrix, geom=True), expected)


def test_column_normalize_normalizes_nonzero_columns_and_uses_fallback() -> None:
    """Normalize nonzero columns and fill empty columns from the fallback vector."""
    counts = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    fallback = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    expected = torch.tensor(
        [
            [0.25, 0.1, 0.5, 0.1],
            [0.25, 0.2, 0.0, 0.2],
            [0.5, 0.3, 0.5, 0.3],
            [0.0, 0.4, 0.0, 0.4],
        ],
        dtype=torch.float64,
    )

    normalized = column_normalize(counts, fallback_col=fallback)

    assert normalized.dtype == torch.float64
    assert torch.allclose(normalized, expected)


def test_column_normalize_with_min_entry_produces_positive_matrix() -> None:
    """Clamp entries before normalization when a minimum entry is provided."""
    counts = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    normalized = column_normalize(counts, min_entry=1e-3)

    assert normalized.dtype == torch.float64
    assert torch.all(normalized > 0)
    assert torch.allclose(normalized.sum(dim=0), torch.ones(4, dtype=torch.float64))


def test_column_normalize_requires_fallback_for_empty_columns() -> None:
    """Raise when empty columns are present and no fallback vector is provided."""
    counts = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(AssertionError, match="fallback_col must be passed"):
        column_normalize(counts)


def test_column_normalize_rejects_wrong_fallback_shape() -> None:
    """Raise when the fallback vector has the wrong shape."""
    counts = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    fallback = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="Expected fallback_col shape"):
        column_normalize(counts, fallback_col=fallback)


def test_check_column_stochastic_matrix_accepts_valid_matrix(
    simple_column_stochastic_matrix: torch.Tensor,
) -> None:
    """Accept a valid column-stochastic matrix."""
    check_column_stochastic_matrix(simple_column_stochastic_matrix)


def test_check_column_stochastic_matrix_rejects_negative_entry(
    simple_column_stochastic_matrix: torch.Tensor,
) -> None:
    """Reject matrices with negative entries."""
    invalid = simple_column_stochastic_matrix.clone()
    invalid[0, 0] = -0.1

    with pytest.raises(ValueError, match="negative entries"):
        check_column_stochastic_matrix(invalid)


def test_check_column_stochastic_matrix_rejects_bad_column_sum(
    simple_column_stochastic_matrix: torch.Tensor,
) -> None:
    """Reject matrices whose columns do not sum to one."""
    invalid = simple_column_stochastic_matrix.clone()
    invalid[0, 0] += 0.1

    with pytest.raises(ValueError, match="must be column-stochastic"):
        check_column_stochastic_matrix(invalid)


def test_check_stationary_column_stochastic_matrix_accepts_stationary_pair(
    simple_column_stochastic_matrix: torch.Tensor,
    simple_stationary_dist: torch.Tensor,
) -> None:
    """Accept a column-stochastic matrix with its stationary distribution."""
    check_stationary_column_stochastic_matrix(
        simple_column_stochastic_matrix, simple_stationary_dist
    )


def test_check_stationary_column_stochastic_matrix_rejects_nonstationary_vector(
    simple_column_stochastic_matrix: torch.Tensor,
) -> None:
    """Reject probability vectors that are not stationary for the matrix."""
    nonstationary = torch.full((4,), 0.25, dtype=torch.float64)

    with pytest.raises(ValueError, match="must be stationary with P"):
        check_stationary_column_stochastic_matrix(
            simple_column_stochastic_matrix, nonstationary
        )


def test_check_stationary_column_stochastic_matrix_accepts_reversible_fixture(
    reversible_column_stochastic_matrix: torch.Tensor,
    reversible_stationary_dist: torch.Tensor,
) -> None:
    """Accept a reversible column-stochastic matrix in detailed-balance mode."""
    check_stationary_column_stochastic_matrix(
        reversible_column_stochastic_matrix, reversible_stationary_dist, db=True
    )


def test_check_stationary_column_stochastic_matrix_rejects_nonreversible_matrix() -> (
    None
):
    """Reject a stationary matrix that does not satisfy detailed balance."""
    nonreversible = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    stationary = torch.full((4,), 0.25, dtype=torch.float64)

    with pytest.raises(ValueError, match="must satisfy detailed balance with P"):
        check_stationary_column_stochastic_matrix(nonreversible, stationary, db=True)
