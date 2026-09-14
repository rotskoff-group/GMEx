# tests/test_utils_datasets.py
# test gmex/utils/datasets.py classes and helpers

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from deeptime.markov import TransitionCountEstimator

from gmex.markov_process import MarkovChain
from gmex.utils import (
    GROUPS_FN,
    METADATA_FN,
    STATE_LIST_FN,
    STATE_TENSOR_FN,
    TIMES_FN,
    MultiSeqDataset,
    SingleSeqDataset,
    StateSeqDataset,
    coarsen_states,
    fixed_timestep_resample,
    get_data_dir,
    get_results_dir,
    get_split,
    load_times_macrostates,
    save_metadata,
)


@pytest.fixture
def single_list_data() -> list[int]:
    """Return a small single-sequence dataset with both transitions and repeats."""
    return [0, 1, 1, 0]


@pytest.fixture
def multi_list_data() -> list[list[int]]:
    """Return a small multi-sequence dataset with one short record to be filtered."""
    return [[0, 1, 1, 0], [1, 0], [1, 1, 0, 0]]


@pytest.fixture
def deterministic_two_state_chain(device_for_testing: str) -> MarkovChain:
    """Return a deterministic two-state alternating Markov chain."""
    prop = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    return MarkovChain(prop, dt=0.5, device=device_for_testing)


def _assert_metadata(
    dataset,
    n_states: int,
    n_records: int,
    max_rec_len: int,
    n_obs: int,
    n_obs_by_state: list[int],
    lim_dist: list[float],
    resampled: bool,
    n_contexts: int,
    n_transitionless_contexts: int,
    frac_transitionless: float,
    frac_transitions_lost: float,
) -> None:
    """Assert the shared metadata fields for a dataset instance."""
    metadata = dataset.metadata
    assert metadata["n_states"] == n_states
    assert metadata["n_records"] == n_records
    assert metadata["max_rec_len"] == max_rec_len
    assert metadata["n_obs"] == n_obs
    assert metadata["n_obs_by_state"] == n_obs_by_state
    assert metadata["lim_dist"] == pytest.approx(lim_dist)
    assert metadata["resampled"] is resampled
    assert metadata["n_contexts"] == n_contexts
    assert metadata["n_transitionless_contexts"] == n_transitionless_contexts
    assert metadata["frac_transitionless"] == pytest.approx(frac_transitionless)
    assert metadata["frac_transitions_lost"] == pytest.approx(frac_transitions_lost)


def _deeptime_dtrajs(dataset) -> list:
    """Return a deeptime-compatible list of discrete trajectories."""
    if isinstance(dataset, SingleSeqDataset):
        return [dataset.data.cpu().numpy()]
    return [rec.cpu().numpy() for rec in dataset.data]


def _assert_lag_zero_is_state_counts(dataset, count_matrices: torch.Tensor) -> None:
    """Assert that the zeroth count matrix is diagonal with state observation counts."""
    expected = torch.diag(torch.tensor(dataset.n_obs_by_state, dtype=torch.float64))
    assert torch.equal(count_matrices[0], expected)


def _assert_positive_lags_match_deeptime(
    dataset,
    sample_interval: int,
    max_lag: int | None = None,
) -> torch.Tensor:
    """Assert that positive-lag count matrices agree with deeptime after transpose."""
    count_matrices = dataset.get_count_matrices(
        sample_interval=sample_interval,
        max_lag=max_lag,
        verbose=False,
    )
    assert count_matrices.dtype == torch.float64
    _assert_lag_zero_is_state_counts(dataset, count_matrices)

    dtrajs = _deeptime_dtrajs(dataset)
    for lag_idx in range(1, count_matrices.shape[0]):
        lag = lag_idx * sample_interval
        counts_deeptime = (
            TransitionCountEstimator(
                lagtime=lag,
                count_mode="sliding",
            )
            .fit(dtrajs)
            .fetch_model()
        )
        assert torch.equal(
            count_matrices[lag_idx].T,
            torch.tensor(counts_deeptime.count_matrix, dtype=torch.float64),
        )

    return count_matrices


def _assert_total_counts_match_per_lag(*count_stacks: torch.Tensor) -> None:
    """Assert that each lag has the same total counts across several count-matrix stacks."""
    reference = [int(count_stack.sum().item()) for count_stack in count_stacks[0]]
    for count_stack in count_stacks[1:]:
        assert [int(count_mat.sum().item()) for count_mat in count_stack] == reference


def test_state_seq_dataset_rejects_nonpositive_context_length() -> None:
    """StateSeqDataset should reject nonpositive context lengths."""
    with pytest.raises(ValueError, match="context_length must be a positive integer"):
        StateSeqDataset(timestep=1.0, context_length=0, data=[0, 1, 0])


def test_state_seq_dataset_requires_exactly_one_input_source() -> None:
    """StateSeqDataset should require either data or both times and states."""
    with pytest.raises(
        ValueError, match="Either data xor both times and states must be passed"
    ):
        StateSeqDataset(timestep=1.0, context_length=1)


def test_state_seq_dataset_rejects_mixed_input_sources() -> None:
    """StateSeqDataset should reject passing data together with times and states."""
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    states = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(
        ValueError, match="If data is passed, times and states must be None"
    ):
        StateSeqDataset(
            timestep=1.0,
            context_length=1,
            times=times,
            states=states,
            data=[0, 1],
        )


@pytest.mark.parametrize(
    ("times", "states", "expected_message"),
    [
        (
            [0.0, 1.0],
            torch.tensor([0, 1], dtype=torch.long),
            "times must be a torch.Tensor",
        ),
        (
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            [0, 1],
            "states must be a torch.Tensor",
        ),
    ],
)
def test_state_seq_dataset_rejects_nontensor_times_or_states(
    times,
    states,
    expected_message: str,
) -> None:
    """StateSeqDataset should require tensor inputs on the times/states path."""
    with pytest.raises(ValueError, match=expected_message):
        StateSeqDataset(timestep=1.0, context_length=1, times=times, states=states)


def test_state_seq_dataset_rejects_mismatched_times_and_states_shapes() -> None:
    """StateSeqDataset should reject tensor inputs with different shapes."""
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    states = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="states and times must have same shape"):
        StateSeqDataset(timestep=1.0, context_length=1, times=times, states=states)


def test_single_seq_dataset_list_path_initializes_and_tracks_metadata(
    single_list_data: list[int],
) -> None:
    """SingleSeqDataset should slice and summarize a list-backed record correctly."""
    dataset = SingleSeqDataset(timestep=1.0, context_length=2, data=single_list_data)

    assert len(dataset) == 2
    x0, y0 = dataset[0]
    x1, y1 = dataset[1]
    assert torch.equal(x0, torch.tensor([0, 1]))
    assert torch.equal(y0, torch.tensor([1, 1]))
    assert torch.equal(x1, torch.tensor([1, 1]))
    assert torch.equal(y1, torch.tensor([1, 0]))

    _assert_metadata(
        dataset,
        n_states=2,
        n_records=1,
        max_rec_len=4,
        n_obs=4,
        n_obs_by_state=[2, 2],
        lim_dist=[0.5, 0.5],
        resampled=False,
        n_contexts=2,
        n_transitionless_contexts=1,
        frac_transitionless=0.5,
        frac_transitions_lost=0.0,
    )


def test_single_seq_dataset_list_path_rejects_too_short_record() -> None:
    """SingleSeqDataset should reject records with no available training context."""
    with pytest.raises(ValueError, match="shorter than context length 4"):
        SingleSeqDataset(timestep=1.0, context_length=4, data=[0, 1, 1, 0])


def test_single_seq_dataset_list_path_rejects_missing_state_labels() -> None:
    """SingleSeqDataset should reject observed states with gaps in the labels."""
    with pytest.raises(ValueError, match=r"Missing states: \[1\]"):
        SingleSeqDataset(timestep=1.0, context_length=1, data=[0, 2, 2])


def test_single_seq_dataset_tensor_path_initializes_from_exact_dt_chain(
    deterministic_two_state_chain: MarkovChain,
) -> None:
    """SingleSeqDataset should preserve a tensor trajectory already sampled at the target dt."""
    times, states = deterministic_two_state_chain.sample(
        n_steps=4, n_trajs=1, initial_state=0
    )
    dataset = SingleSeqDataset(
        timestep=0.5,
        context_length=2,
        times=times.cpu(),
        states=states.cpu(),
    )

    assert len(dataset) == 3
    x0, y0 = dataset[0]
    assert torch.equal(x0, torch.tensor([0, 1]))
    assert torch.equal(y0, torch.tensor([1, 0]))

    _assert_metadata(
        dataset,
        n_states=2,
        n_records=1,
        max_rec_len=5,
        n_obs=5,
        n_obs_by_state=[3, 2],
        lim_dist=[0.6, 0.4],
        resampled=True,
        n_contexts=3,
        n_transitionless_contexts=0,
        frac_transitionless=0.0,
        frac_transitions_lost=0.0,
    )


def test_single_seq_dataset_tensor_path_resamples_irregular_observations() -> None:
    """SingleSeqDataset should resample irregular times into a fixed-step sequence."""
    times = torch.tensor([0.0, 0.2, 0.5, 1.0], dtype=torch.float64)
    states = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = SingleSeqDataset(
        timestep=0.25, context_length=2, times=times, states=states
    )

    assert torch.equal(dataset.data, torch.tensor([0, 1, 0, 0, 1]))
    assert dataset.n_transitions_raw == 3
    assert dataset.n_transitions_resampled == 3
    assert dataset.metadata["resampled"] is True


def test_multi_seq_dataset_list_path_filters_short_records_and_indexes_across_records(
    multi_list_data: list[list[int]],
) -> None:
    """MultiSeqDataset should drop short records and index through the surviving records."""
    dataset = MultiSeqDataset(timestep=1.0, context_length=2, data=multi_list_data)

    assert len(dataset.data) == 2
    assert len(dataset) == 4
    x0, y0 = dataset[0]
    x1, y1 = dataset[1]
    x2, y2 = dataset[2]
    x3, y3 = dataset[3]
    assert torch.equal(x0, torch.tensor([0, 1]))
    assert torch.equal(y0, torch.tensor([1, 1]))
    assert torch.equal(x1, torch.tensor([1, 1]))
    assert torch.equal(y1, torch.tensor([1, 0]))
    assert torch.equal(x2, torch.tensor([1, 1]))
    assert torch.equal(y2, torch.tensor([1, 0]))
    assert torch.equal(x3, torch.tensor([1, 0]))
    assert torch.equal(y3, torch.tensor([0, 0]))

    _assert_metadata(
        dataset,
        n_states=2,
        n_records=2,
        max_rec_len=4,
        n_obs=8,
        n_obs_by_state=[4, 4],
        lim_dist=[0.5, 0.5],
        resampled=False,
        n_contexts=4,
        n_transitionless_contexts=2,
        frac_transitionless=0.5,
        frac_transitions_lost=0.0,
    )


def test_multi_seq_dataset_list_path_rejects_when_no_record_survives() -> None:
    """MultiSeqDataset should fail cleanly when all list-backed records are too short."""
    with pytest.raises(ValueError, match="No records longer than context_length 2"):
        MultiSeqDataset(timestep=1.0, context_length=2, data=[[0, 1], [1, 0]])


def test_multi_seq_dataset_tensor_path_initializes_from_exact_dt_chain(
    deterministic_two_state_chain: MarkovChain,
) -> None:
    """MultiSeqDataset should preserve exact-dt tensor trajectories after resampling."""
    times_0, states_0 = deterministic_two_state_chain.sample(
        n_steps=4, n_trajs=1, initial_state=0
    )
    times_1, states_1 = deterministic_two_state_chain.sample(
        n_steps=4, n_trajs=1, initial_state=1
    )
    dataset = MultiSeqDataset(
        timestep=0.5,
        context_length=2,
        times=torch.stack((times_0.cpu(), times_1.cpu())),
        states=torch.stack((states_0.cpu(), states_1.cpu())),
    )

    assert len(dataset.data) == 2
    assert len(dataset) == 6
    x3, y3 = dataset[3]
    assert torch.equal(x3, torch.tensor([1, 0]))
    assert torch.equal(y3, torch.tensor([0, 1]))

    _assert_metadata(
        dataset,
        n_states=2,
        n_records=2,
        max_rec_len=5,
        n_obs=10,
        n_obs_by_state=[5, 5],
        lim_dist=[0.5, 0.5],
        resampled=True,
        n_contexts=6,
        n_transitionless_contexts=0,
        frac_transitionless=0.0,
        frac_transitions_lost=0.0,
    )


def test_multi_seq_dataset_tensor_path_filters_out_short_resampled_records() -> None:
    """MultiSeqDataset should keep only tensor-backed records that remain long enough after resampling."""
    times = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.1, 0.2],
        ],
        dtype=torch.float64,
    )
    states = torch.tensor(
        [
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=torch.long,
    )

    dataset = MultiSeqDataset(
        timestep=0.5, context_length=2, times=times, states=states
    )

    assert len(dataset.data) == 1
    assert len(dataset) == 1
    assert dataset.n_transitions_raw == 2
    assert dataset.n_transitions_resampled == 2
    _assert_metadata(
        dataset,
        n_states=2,
        n_records=1,
        max_rec_len=3,
        n_obs=3,
        n_obs_by_state=[2, 1],
        lim_dist=[2.0 / 3.0, 1.0 / 3.0],
        resampled=True,
        n_contexts=1,
        n_transitionless_contexts=0,
        frac_transitionless=0.0,
        frac_transitions_lost=0.0,
    )


def test_get_split_returns_single_dataset_for_train_frac_one(
    single_list_data: list[int],
) -> None:
    """get_split should return a single dataset when train_frac is 1."""
    dataset = get_split(
        timestep=1.0, context_length=2, data=single_list_data, train_frac=1.0
    )

    assert isinstance(dataset, SingleSeqDataset)
    assert dataset.context_length == 2
    assert len(dataset) == 2


def test_get_split_splits_flat_list_data_into_single_seq_datasets() -> None:
    """get_split should dispatch flat list data to SingleSeqDataset."""
    train_data, valid_data = get_split(
        timestep=1.0,
        context_length=2,
        data=[0, 1, 0, 1, 0, 1],
        train_frac=0.5,
    )

    assert isinstance(train_data, SingleSeqDataset)
    assert isinstance(valid_data, SingleSeqDataset)
    assert train_data.context_length == 2
    assert valid_data.context_length == 2
    assert len(train_data) == 1
    assert len(valid_data) == 1


def test_get_split_splits_nested_list_data_into_multi_seq_datasets() -> None:
    """get_split should dispatch nested list data to MultiSeqDataset."""
    train_data, valid_data = get_split(
        timestep=1.0,
        context_length=2,
        data=[
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        train_frac=0.5,
    )

    assert isinstance(train_data, MultiSeqDataset)
    assert isinstance(valid_data, MultiSeqDataset)
    assert train_data.context_length == 2
    assert valid_data.context_length == 2
    assert len(train_data) == 4
    assert len(valid_data) == 4


def test_get_split_splits_tensor_data_by_dim() -> None:
    """get_split should dispatch 1D tensors to SingleSeqDataset and 2D tensors to MultiSeqDataset."""
    times_1d = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    states_1d = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    train_1d, valid_1d = get_split(
        timestep=0.5,
        context_length=2,
        times=times_1d,
        states=states_1d,
        train_frac=0.5,
    )

    times_2d = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.5, 1.0, 1.5],
        ],
        dtype=torch.float64,
    )
    states_2d = torch.tensor(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        dtype=torch.long,
    )
    train_2d, valid_2d = get_split(
        timestep=0.5,
        context_length=2,
        times=times_2d,
        states=states_2d,
        train_frac=0.5,
    )

    assert isinstance(train_1d, SingleSeqDataset)
    assert isinstance(valid_1d, SingleSeqDataset)
    assert isinstance(train_2d, MultiSeqDataset)
    assert isinstance(valid_2d, MultiSeqDataset)
    assert train_1d.context_length == 2
    assert valid_2d.context_length == 2


@pytest.mark.parametrize("train_frac", [0.0, -0.1, 1.1])
def test_get_split_rejects_invalid_train_frac(train_frac: float) -> None:
    """get_split should reject fractions outside (0, 1]."""
    with pytest.raises(ValueError, match=r"train_frac must be in range \(0, 1\]"):
        get_split(
            timestep=1.0, context_length=2, data=[0, 1, 0, 1], train_frac=train_frac
        )


def test_get_split_requires_input_source() -> None:
    """get_split should require either data or both times and states."""
    with pytest.raises(
        ValueError, match="Either data xor both times and states must be provided"
    ):
        get_split(timestep=1.0, context_length=1)


def test_fixed_timestep_resample_preserves_exact_dt_series() -> None:
    """fixed_timestep_resample should leave an exact-dt trajectory unchanged."""
    times = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    states = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    new_times, new_states, n_transitions = fixed_timestep_resample(
        times, states, timestep=0.5
    )

    assert torch.equal(new_times, times)
    assert new_times.dtype == torch.float64
    assert torch.equal(new_states, states)
    assert n_transitions == (2, 2)


def test_fixed_timestep_resample_interpolates_piecewise_constant_states() -> None:
    """fixed_timestep_resample should use previous-state interpolation between jump times."""
    times = torch.tensor([0.0, 0.2, 0.5, 1.0], dtype=torch.float64)
    states = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    new_times, new_states, n_transitions = fixed_timestep_resample(
        times, states, timestep=0.25
    )

    assert torch.allclose(
        new_times, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    )
    assert new_times.dtype == torch.float64
    assert torch.equal(new_states, torch.tensor([0, 1, 0, 0, 1]))
    assert n_transitions == (3, 3)


def test_fixed_timestep_resample_uses_true_fixed_spacing_when_final_time_is_not_multiple() -> (
    None
):
    """fixed_timestep_resample should still use the requested timestep when the final time is off-grid."""
    times = torch.tensor([0.0, 0.4, 1.1], dtype=torch.float64)
    states = torch.tensor([0, 1, 1], dtype=torch.long)
    new_times, new_states = fixed_timestep_resample(
        times,
        states,
        timestep=0.5,
        return_transition_cts=False,
    )

    assert torch.allclose(new_times, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
    assert new_times.dtype == torch.float64
    assert torch.equal(new_states, torch.tensor([0, 1, 1]))


def test_fixed_timestep_resample_keeps_final_on_grid_sample_despite_float_rounding() -> (
    None
):
    """fixed_timestep_resample should not drop a final sample that lies on the timestep grid."""
    times = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    states = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    new_times, new_states, n_transitions = fixed_timestep_resample(
        times,
        states,
        timestep=0.1,
    )

    assert torch.allclose(
        new_times, torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    )
    assert new_times.dtype == torch.float64
    assert torch.equal(new_states, torch.tensor([0, 1, 0, 1]))
    assert n_transitions == (3, 3)


def test_fixed_timestep_resample_rejects_mismatched_lengths() -> None:
    """fixed_timestep_resample should reject different numbers of times and states."""
    with pytest.raises(
        AssertionError, match="states and times must have the same length"
    ):
        fixed_timestep_resample(
            times=torch.tensor([0.0, 0.5], dtype=torch.float64),
            states=torch.tensor([0], dtype=torch.long),
            timestep=0.5,
        )


def test_single_seq_dataset_get_count_matrices_matches_expected_literal_counts() -> (
    None
):
    """SingleSeqDataset should produce the expected literal count matrices."""
    dataset = SingleSeqDataset(timestep=1.0, context_length=1, data=[0, 1, 1, 0])
    observed = dataset.get_count_matrices(verbose=False)
    expected = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(observed, expected)


def test_multi_seq_dataset_get_count_matrices_matches_expected_literal_counts() -> None:
    """MultiSeqDataset should produce the expected literal count matrices."""
    dataset = MultiSeqDataset(
        timestep=1.0,
        context_length=1,
        data=[[0, 1, 1, 0], [1, 1, 0, 0]],
    )
    observed = dataset.get_count_matrices(verbose=False)
    expected = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0]],
            [[1.0, 2.0], [1.0, 2.0]],
            [[0.0, 3.0], [1.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(observed, expected)


def test_single_seq_dataset_get_count_matrices_matches_deeptime(
    simple_column_stochastic_matrix: torch.Tensor,
    device_for_testing: str,
) -> None:
    """SingleSeqDataset positive-lag count matrices should match deeptime after transpose."""
    chain = MarkovChain(
        simple_column_stochastic_matrix, dt=0.5, device=device_for_testing
    )
    times, states = chain.sample(n_steps=32, n_trajs=1, initial_state=0)
    dataset = SingleSeqDataset(
        timestep=0.5,
        context_length=1,
        times=times.cpu(),
        states=states.cpu(),
    )
    _assert_positive_lags_match_deeptime(dataset, sample_interval=2, max_lag=4)


def test_multi_seq_dataset_get_count_matrices_matches_deeptime(
    simple_column_stochastic_matrix: torch.Tensor,
    device_for_testing: str,
) -> None:
    """MultiSeqDataset positive-lag count matrices should match deeptime after transpose."""
    chain = MarkovChain(
        simple_column_stochastic_matrix, dt=0.5, device=device_for_testing
    )
    times, states = chain.sample(n_steps=16, n_trajs=4, initial_state=0)
    dataset = MultiSeqDataset(
        timestep=0.5,
        context_length=1,
        times=times.cpu(),
        states=states.cpu(),
    )
    _assert_positive_lags_match_deeptime(dataset, sample_interval=2, max_lag=4)


def test_get_count_matrices_respects_max_lag(single_list_data: list[int]) -> None:
    """get_count_matrices should return lag 0 through the requested maximum lag."""
    dataset = SingleSeqDataset(timestep=1.0, context_length=1, data=single_list_data)
    observed = dataset.get_count_matrices(sample_interval=1, max_lag=1, verbose=False)
    assert observed.shape == (2, 2, 2)
    _assert_lag_zero_is_state_counts(dataset, observed)


def test_get_count_matrices_returns_only_lag_zero_when_sample_interval_exceeds_record_length() -> (
    None
):
    """get_count_matrices should return only the zeroth matrix when no positive lag fits."""
    dataset = SingleSeqDataset(timestep=1.0, context_length=1, data=[0, 1, 1, 0])
    observed = dataset.get_count_matrices(sample_interval=10, verbose=False)
    assert observed.shape == (1, 2, 2)
    _assert_lag_zero_is_state_counts(dataset, observed)


def test_multi_seq_dataset_get_count_matrices_bootstrap_preserves_total_counts_by_lag() -> (
    None
):
    """Bootstrap count matrices should preserve the total counts at each lag."""
    dataset = MultiSeqDataset(
        timestep=1.0,
        context_length=1,
        data=[
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
        ],
    )
    counts_base = dataset.get_count_matrices(
        sample_interval=1, max_lag=2, bootstrap=False, verbose=False
    )
    counts_boot_1 = dataset.get_count_matrices(
        sample_interval=1, max_lag=2, bootstrap=True, verbose=False
    )
    counts_boot_2 = dataset.get_count_matrices(
        sample_interval=1, max_lag=2, bootstrap=True, verbose=False
    )
    _assert_total_counts_match_per_lag(counts_base, counts_boot_1, counts_boot_2)


def test_multi_seq_dataset_get_count_matrices_bootstrap_changes_counts_without_reseed() -> (
    None
):
    """Repeated bootstrap calls should redistribute counts without reseeding."""
    dataset = MultiSeqDataset(
        timestep=1.0,
        context_length=1,
        data=[
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
        ],
    )
    counts_base = dataset.get_count_matrices(
        sample_interval=1, max_lag=2, bootstrap=False, verbose=False
    )
    counts_boot_1 = dataset.get_count_matrices(
        sample_interval=1, max_lag=2, bootstrap=True, verbose=False
    )
    counts_boot_2 = dataset.get_count_matrices(
        sample_interval=1, max_lag=2, bootstrap=True, verbose=False
    )
    assert not torch.equal(counts_boot_1, counts_base)
    assert not torch.equal(counts_boot_2, counts_base)
    assert not torch.equal(counts_boot_1, counts_boot_2)


def test_get_data_dir_uses_configured_env_path(test_data_dir: Path) -> None:
    """Return the pytest-configured data directory when the env var is set.

    Parameters
    ----------
    test_data_dir : pathlib.Path
        Session-scoped temporary data directory configured by pytest.
    """
    assert get_data_dir() == test_data_dir


def test_get_results_dir_uses_configured_env_path(test_results_dir: Path) -> None:
    """Return the pytest-configured results directory when the env var is set.

    Parameters
    ----------
    test_results_dir : pathlib.Path
        Session-scoped temporary results directory configured by pytest.
    """
    assert get_results_dir() == test_results_dir


def test_get_data_dir_falls_back_to_cwd_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve the fallback data directory relative to a temporary working directory.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Helper for temporary environment and working-directory changes.
    tmp_path : pathlib.Path
        Test-specific temporary directory.
    """
    monkeypatch.delenv("GMEX_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert get_data_dir() == (tmp_path / "data").resolve()


def test_get_results_dir_falls_back_to_cwd_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve the fallback results directory relative to a temporary working directory.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Helper for temporary environment and working-directory changes.
    tmp_path : pathlib.Path
        Test-specific temporary directory.
    """
    monkeypatch.delenv("GMEX_RESULTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert get_results_dir() == (tmp_path / "results").resolve()


def test_save_metadata_writes_expected_contents() -> None:
    """Write metadata in a temporary directory and verify the saved contents."""
    argvars = {
        "alpha": 1,
        "beta": "value",
        "_run": "ignored",
        "_section": "ignored",
        "config": "ignored",
        "cmd": "ignored",
    }
    with TemporaryDirectory() as tmpdir:
        outfile = Path(tmpdir) / METADATA_FN
        save_metadata(argvars.copy(), outfile)
        with open(outfile, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata == {"alpha": 1, "beta": "value"}


def test_load_times_macrostates_reads_state_list_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_column_stochastic_matrix: torch.Tensor,
    simple_groups: list[list[int]],
) -> None:
    """Load a grouped list-backed dataset from a temporary data directory."""
    monkeypatch.setenv("GMEX_DATA_DIR", str(tmp_path))
    dataset_dir = tmp_path / "list_dataset"
    dataset_dir.mkdir()

    sim = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    times_0, microstates_0 = sim.sample(n_steps=4, n_trajs=1, initial_state=0)
    times_1, microstates_1 = sim.sample(n_steps=4, n_trajs=1, initial_state=3)
    del times_0, times_1
    microstates = torch.stack((microstates_0, microstates_1))

    with open(dataset_dir / METADATA_FN, "w", encoding="utf-8") as f:
        json.dump({"dt": 0.5}, f)
    with open(dataset_dir / GROUPS_FN, "w", encoding="utf-8") as f:
        json.dump(simple_groups, f)
    with open(dataset_dir / STATE_LIST_FN, "w", encoding="utf-8") as f:
        json.dump(microstates.tolist(), f)

    expected_data = coarsen_states(microstates.tolist(), simple_groups)

    dt, times, macrostates, data, n_macrostates = load_times_macrostates(
        "list_dataset", device=torch.device("cpu")
    )

    assert dt == 0.5
    assert times is None
    assert macrostates is None
    assert data == expected_data
    assert n_macrostates == len(simple_groups)


def test_load_times_macrostates_reads_state_tensor_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_column_stochastic_matrix: torch.Tensor,
    simple_groups: list[list[int]],
) -> None:
    """Load a grouped tensor-backed dataset from a temporary data directory."""
    monkeypatch.setenv("GMEX_DATA_DIR", str(tmp_path))
    dataset_dir = tmp_path / "tensor_dataset"
    dataset_dir.mkdir()

    sim = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    times_0, microstates_0 = sim.sample(n_steps=4, n_trajs=1, initial_state=0)
    times_1, microstates_1 = sim.sample(n_steps=4, n_trajs=1, initial_state=3)
    times = torch.stack((times_0, times_1))
    microstates = torch.stack((microstates_0, microstates_1))

    with open(dataset_dir / METADATA_FN, "w", encoding="utf-8") as f:
        json.dump({"dt": 0.5}, f)
    with open(dataset_dir / GROUPS_FN, "w", encoding="utf-8") as f:
        json.dump(simple_groups, f)
    torch.save(times, dataset_dir / TIMES_FN)
    torch.save(microstates, dataset_dir / STATE_TENSOR_FN)

    expected_macrostates = coarsen_states(microstates, simple_groups)
    assert isinstance(expected_macrostates, torch.Tensor)
    dt, observed_times, macrostates, data, n_macrostates = load_times_macrostates(
        "tensor_dataset", device=torch.device("cpu")
    )

    assert dt == 0.5
    assert torch.equal(observed_times, times)
    assert torch.equal(macrostates, expected_macrostates)
    assert data is None
    assert n_macrostates == len(simple_groups)
