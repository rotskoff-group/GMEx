# tests/conftest.py
# pytest parameters


import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gmex.markov_process import MarkovChain
from gmex.utils.datasets import MultiSeqDataset

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".pytest_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def pytest_addoption(parser) -> None:
    """Register custom pytest command-line options.

    Parameters
    ----------
    parser : pytest.Parser
        Parser used to register command-line options.
    """
    parser.addoption(
        "--seed", action="store", default="42", help="Random seed for tests."
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on (e.g., 'cpu', 'cuda').",
    )


def pytest_configure(config) -> None:
    """Configure deterministic test behavior.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration object.
    """
    seed = int(config.getoption("--seed"))
    device = config.getoption("--device")
    print(f"using seed = {seed}, device = {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session", autouse=True)
def configure_test_directories(tmp_path_factory: pytest.TempPathFactory):
    """Set test-only data and results directories for the pytest session.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for creating session-scoped temporary directories.

    Yields
    ------
    dict[str, pathlib.Path]
        Mapping containing resolved temporary data and results directories.
    """
    base_dir = tmp_path_factory.mktemp("gmex-test-dirs")
    data_dir = (base_dir / "data").resolve()
    results_dir = (base_dir / "results").resolve()
    data_dir.mkdir()
    results_dir.mkdir()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("GMEX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("GMEX_RESULTS_DIR", str(results_dir))

    yield {"data_dir": data_dir, "results_dir": results_dir}

    monkeypatch.undo()


@pytest.fixture(scope="session")
def test_data_dir(configure_test_directories: dict[str, Path]) -> Path:
    """Return the configured session-scoped test data directory.

    Parameters
    ----------
    configure_test_directories : dict[str, pathlib.Path]
        Mapping of configured session directory paths.

    Returns
    -------
    pathlib.Path
        Resolved temporary data directory for tests.
    """
    return configure_test_directories["data_dir"]


@pytest.fixture(scope="session")
def test_results_dir(configure_test_directories: dict[str, Path]) -> Path:
    """Return the configured session-scoped test results directory.

    Parameters
    ----------
    configure_test_directories : dict[str, pathlib.Path]
        Mapping of configured session directory paths.

    Returns
    -------
    pathlib.Path
        Resolved temporary results directory for tests.
    """
    return configure_test_directories["results_dir"]


@pytest.fixture(scope="session")
def simple_cycle_rates() -> torch.Tensor:
    """Return a 4x4 transition-rate matrix for a simple cycle.

    Returns
    -------
    torch.Tensor
        Matrix of transition rates.
    """
    rates = torch.zeros((4, 4))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edges:
        rates[i, j] = 2.0
        rates[j, i] = 1.0
    return rates


@pytest.fixture(scope="session")
def simple_stationary_dist() -> torch.Tensor:
    """Return a simple stationary distribution over four states."""
    return torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)


@pytest.fixture(scope="session")
def simple_column_stochastic_matrix(
    simple_stationary_dist: torch.Tensor,
) -> torch.Tensor:
    """Return a non-detailed-balance full-rank 4x4 column-stochastic matrix."""
    del simple_stationary_dist
    return torch.tensor(
        [
            [0.55, 0.075, 0.05, 0.0375],
            [0.05, 0.60, 0.35 / 3.0, 0.10],
            [0.15, 0.125, 0.65, 0.1625],
            [0.25, 0.20, 0.55 / 3, 0.70],
        ],
        dtype=torch.float64,
    )


@pytest.fixture(scope="session")
def simple_inhomogeneous_column_stochastic_stack(
    simple_stationary_dist: torch.Tensor,
) -> torch.Tensor:
    """Return a 3x4x4 inhomogeneous stack with nonreversible stationary slices."""
    del simple_stationary_dist
    eye = torch.eye(4, dtype=torch.float64)
    prop_1 = torch.tensor(
        [
            [0.64, 0.065, 0.11 / 3.0, 0.03],
            [0.03, 0.680, 0.29 / 3.0, 0.08],
            [0.13, 0.095, 2.16 / 3.0, 0.13],
            [0.20, 0.160, 0.44 / 3.0, 0.76],
        ],
        dtype=torch.float64,
    )
    prop_2 = torch.tensor(
        [
            [0.37, 0.10, 0.22 / 3.0, 0.0525],
            [0.08, 0.44, 0.48 / 3.0, 0.1400],
            [0.20, 0.18, 1.53 / 3.0, 0.2275],
            [0.35, 0.28, 0.77 / 3.0, 0.5800],
        ],
        dtype=torch.float64,
    )
    return torch.stack((eye, prop_1, prop_2))


@pytest.fixture(scope="session")
def reversible_stationary_dist() -> torch.Tensor:
    """Return a simple reversible stationary distribution over four states."""
    return torch.full((4,), 0.25, dtype=torch.float64)


@pytest.fixture(scope="session")
def reversible_column_stochastic_matrix(
    reversible_stationary_dist: torch.Tensor,
) -> torch.Tensor:
    """Return a reversible full-rank 4x4 column-stochastic matrix."""
    return 0.5 * torch.eye(4, dtype=torch.float64) + 0.5 * (
        reversible_stationary_dist[:, None] * torch.ones((1, 4), dtype=torch.float64)
    )


@pytest.fixture(scope="session")
def reversible_inhomogeneous_column_stochastic_stack(
    reversible_stationary_dist: torch.Tensor,
) -> torch.Tensor:
    """Return a 3x4x4 inhomogeneous stack with commuting reversible slices.
    Note that this isn't actually fully compatible with reversibility;
    that would require prop_1 == prop_2.
    It is not used as an expected output for the stack wrapper.
    """
    n_states = reversible_stationary_dist.numel()
    eye = torch.eye(n_states, dtype=torch.float64)
    ones = torch.ones((n_states, n_states), dtype=torch.float64)
    off_diag = ones - eye

    prop_1 = 0.70 * eye + 0.10 * off_diag
    prop_2 = 0.55 * eye + 0.15 * off_diag

    return torch.stack((eye, prop_1, prop_2))


@pytest.fixture(scope="session")
def simple_groups() -> list[list[int]]:
    """Return a simple grouping of four microstates into two macrostates."""
    return [[0, 1], [2, 3]]


@pytest.fixture(scope="session")
def simple_macrostate_indicator(simple_groups: list[list[int]]) -> torch.Tensor:
    """Return the macrostate-indicator matrix for the shared 4-state grouping."""
    indicator = torch.zeros((4, 2), dtype=torch.float64)
    indicator[0, 0] = 1.0
    indicator[1, 0] = 1.0
    indicator[2, 1] = 1.0
    indicator[3, 1] = 1.0
    return indicator


@pytest.fixture(scope="session")
def device_for_testing(request):
    """Return the device selected for the test session.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    str
        Device to run tests on, either 'cpu' or 'cuda'.
    """
    return request.config.getoption("--device")


@pytest.fixture
def simple_markov_count_data(
    simple_column_stochastic_matrix: torch.Tensor,
) -> dict[str, torch.Tensor | MultiSeqDataset]:
    """Return simple count matrices generated from sampled Markov-chain trajectories."""
    process = MarkovChain(simple_column_stochastic_matrix, dt=0.5, device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        _, states = process.sample(n_steps=20, n_trajs=6, initial_state=0)

    dataset = MultiSeqDataset(
        timestep=0.5,
        context_length=1,
        data=states.cpu().tolist(),
    )
    count_matrices = dataset.get_count_matrices(
        sample_interval=1,
        max_lag=1,
        verbose=False,
    )
    return {
        "transition_matrix": simple_column_stochastic_matrix.clone(),
        "dataset": dataset,
        "count_matrices": count_matrices,
    }


@pytest.fixture
def simple_inhomogeneous_markov_count_data(
    simple_inhomogeneous_column_stochastic_stack: torch.Tensor,
) -> dict[str, torch.Tensor | MultiSeqDataset]:
    """Return multi-lag count matrices from sampled inhomogeneous trajectories."""
    process = MarkovChain(
        simple_inhomogeneous_column_stochastic_stack,
        dt=0.5,
        device="cpu",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        _, states = process.sample(n_steps=40, n_trajs=12, initial_state=0)

    dataset = MultiSeqDataset(
        timestep=0.5,
        context_length=1,
        data=states.cpu().tolist(),
    )
    count_matrices = dataset.get_count_matrices(
        sample_interval=1,
        max_lag=2,
        verbose=False,
    )
    return {
        "transition_stack": simple_inhomogeneous_column_stochastic_stack.clone(),
        "dataset": dataset,
        "count_matrices": count_matrices,
    }


@pytest.fixture
def reversible_inhomogeneous_markov_count_data(
    reversible_inhomogeneous_column_stochastic_stack: torch.Tensor,
) -> dict[str, torch.Tensor | MultiSeqDataset]:
    """Return multi-lag count matrices from sampled reversible inhomogeneous trajectories."""
    process = MarkovChain(
        reversible_inhomogeneous_column_stochastic_stack,
        dt=0.5,
        device="cpu",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        _, states = process.sample(n_steps=40, n_trajs=12, initial_state=0)

    dataset = MultiSeqDataset(
        timestep=0.5,
        context_length=1,
        data=states.cpu().tolist(),
    )
    count_matrices = dataset.get_count_matrices(
        sample_interval=1,
        max_lag=2,
        verbose=False,
    )
    return {
        "transition_stack": reversible_inhomogeneous_column_stochastic_stack.clone(),
        "dataset": dataset,
        "count_matrices": count_matrices,
    }
