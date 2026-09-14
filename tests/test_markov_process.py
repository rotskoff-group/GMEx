# tests/test_markov_process.py
# test gmex.markov_process classes and helpers

import numpy as np
import pytest
import torch
from scipy import linalg, stats

from gmex.markov_process import (
    GillespieMJP,
    MarkovChain,
    MarkovJumpProcess,
    MarkovProcess,
)


class DummyMarkovProcess(MarkovProcess):
    """Small concrete MarkovProcess used to test base-class helpers."""

    def __init__(self, stationary_dist: torch.Tensor, device: str = "cpu"):
        super().__init__(device=device)
        self._stationary_dist = stationary_dist.to(self.device)
        self.n_states = stationary_dist.numel()
        self.stationary_distribution_called = False

    def sample(
        self,
        n_steps: int,
        n_trajs: int = 1,
        initial_state: int | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("sample is not needed for this test")

    def stationary_distribution(self) -> torch.Tensor:
        self.stationary_distribution_called = True
        return self._stationary_dist


@torch.no_grad()
def test_get_initial_states_samples_stationary_distribution():
    """_get_initial_states should sample from stationary_distribution by default."""
    sim = DummyMarkovProcess(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))

    initial_states = sim._get_initial_states(n_trajs=5)

    assert initial_states.shape == (5,)
    assert initial_states.device.type == sim.device
    assert torch.equal(
        initial_states, torch.ones(5, dtype=torch.long, device=sim.device)
    )
    assert sim.stationary_distribution_called


@torch.no_grad()
def test_get_initial_states_uses_explicit_initial_state():
    """_get_initial_states should not call stationary_distribution when state is given."""
    sim = DummyMarkovProcess(
        torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    )

    initial_states = sim._get_initial_states(n_trajs=5, initial_state=2)

    assert torch.equal(
        initial_states, torch.full((5,), 2, dtype=torch.long, device=sim.device)
    )
    assert not sim.stationary_distribution_called


torch.no_grad()


def test_markov_jump(device_for_testing, simple_cycle_rates):
    """
    Test parameterization of the MarkovJumpProcess class.

    Parameters
    ----------
    device_for_testing : str
        Device to run tests on (e.g., 'cpu', 'cuda').
    simple_cycle_rates : torch.Tensor
        4x4 matrix of transition rates for a simple cycle.
    """
    sim = MarkovJumpProcess(device=device_for_testing)
    sim.parameterize_from_rates(simple_cycle_rates)

    # total rate = 3.0 (rate to 1 = 2.0, rate to 3 = 1.0)
    # probability of transition to 1 = 2/3
    # probability of transition to 3 = 1/3

    # test 1: verify that transition-rate matrix is correctly computed
    observed_trm = sim.transition_rate_matrix().cpu().numpy()
    expected_trm = linalg.toeplitz([-3, 2, 0, 1], [-3, 1, 0, 2])
    trm_close = np.isclose(observed_trm, expected_trm)
    assert np.all(trm_close), "Master-equation transition-rate matrix incorrect"

    # test 2: verify that stationary entropy-production rate is correctly computed
    observed_epr = sim.stationary_epr()  # should be one bit/time
    assert np.isclose(observed_epr, np.log(2)), "Entropy-production rate incorrect"

    # test 3: verify that (in)fluxes are correctly computed
    observed_influxes = sim.stationary_influxes().cpu().numpy()
    expected_influxes = linalg.toeplitz([0.0, 0.5, 0.0, 0.25], [0.0, 0.25, 0.0, 0.5])
    influxes_close = np.isclose(observed_influxes, expected_influxes)
    assert np.all(influxes_close), "Influxes incorrect"

    # test 4: verify that barriers are correctly computed
    expected_barriers = linalg.toeplitz(
        [0.0, -np.log(2) / 2, float("inf"), -np.log(2) / 2]
    )
    observed_barriers = sim.barriers.cpu().numpy()
    barriers_close = np.isclose(expected_barriers, observed_barriers)
    assert np.all(barriers_close), "Barriers incorrect"

    # test 5: verify that forces are correctly computed
    expected_forces = linalg.toeplitz(
        [0.0, -np.log(2), 0.0, np.log(2)], [0.0, np.log(2), 0.0, -np.log(2)]
    )
    observed_forces = sim.forces.cpu().numpy()
    forces_close = np.isclose(expected_forces, observed_forces)
    assert np.all(forces_close), "Forces incorrect"

    # test 6: verify that rates can be recovered from energies
    sim_from_energies = MarkovJumpProcess()
    sim_from_energies.parameterize_from_energies(sim.wells, sim.barriers, sim.forces)
    observed_rates = sim_from_energies.rates.cpu().numpy()
    rates_close = np.isclose(simple_cycle_rates, observed_rates)
    assert np.all(rates_close), "Rates not recovered from energies"

    # test 7: verify exact distribution propagation
    initial_dist = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=sim.device
    )
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, device=sim.device)

    observed_dists = sim.generate(times, initial_dist=initial_dist)
    trm = sim.transition_rate_matrix()
    expected_dists = torch.stack(
        [torch.linalg.matrix_exp(t * trm) @ initial_dist for t in times]
    )

    assert observed_dists.shape == (len(times), sim.n_states), (
        "Generated distributions have wrong shape"
    )
    assert torch.allclose(observed_dists[0], initial_dist), (
        "Distribution at t=0 should equal initial distribution"
    )
    assert torch.allclose(
        observed_dists.sum(dim=1),
        torch.ones(len(times), dtype=torch.float64, device=sim.device),
        rtol=1e-8,
        atol=1e-8,
    ), "Generated distributions must remain normalized"
    assert torch.all(observed_dists >= -1e-12), (
        "Generated distributions must remain nonnegative"
    )
    assert torch.allclose(
        observed_dists,
        expected_dists,
        rtol=1e-8,
        atol=1e-8,
    ), "Generated distributions do not match exact matrix-exponential propagation"

    # test 8: reject sampling call
    with pytest.raises(NotImplementedError, match="Use a child class"):
        sim.sample(n_steps=1)


@torch.no_grad()
def test_homogeneous_markov_chain(
    device_for_testing,
    simple_column_stochastic_matrix,
    simple_stationary_dist,
):
    """
    Test parameterization and main methods of the homogeneous MarkovChain class.

    Parameters
    ----------
    device_for_testing : str
        Device to run tests on (e.g., 'cpu', 'cuda').
    simple_column_stochastic_matrix : torch.Tensor
        4x4 homogeneous transition-probability matrix.
    simple_stationary_dist : torch.Tensor
        Stationary distribution of the shared 4-state test matrix.
    """
    sim = MarkovChain(
        simple_column_stochastic_matrix, dt=0.5, device=device_for_testing
    )

    # test 1: verify homogeneous Markov-chain parameterization
    expected_prop = simple_column_stochastic_matrix.to(sim.device).to(torch.float64)
    assert sim.n_states == 4, "Number of states incorrect"
    assert sim.dt == 0.5, "Timestep size incorrect"
    assert sim.Gs is None, (
        "Homogeneous Markov chain should not store time-dependent propagators"
    )
    assert isinstance(sim.L, torch.Tensor), "Expected torch.Tensor sim.L"
    assert torch.allclose(sim.L, expected_prop), "Stored propagator incorrect"

    # test 2: verify stationary distribution
    observed_statdist = sim.stationary_distribution()
    expected_statdist = simple_stationary_dist.to(sim.device).to(torch.float64)
    assert torch.allclose(observed_statdist, expected_statdist), (
        "Stationary distribution incorrect"
    )

    # test 3: verify stationary flux
    observed_flux = sim.get_flux()
    expected_flux = sim.L @ torch.diag(expected_statdist)
    assert torch.allclose(observed_flux, expected_flux), "Stationary flux incorrect"

    # test 4: verify second-order local entropy production rate
    observed_epr2 = sim.get_2nd_local_epr()
    expected_epr2 = (
        torch.nan_to_num(
            expected_flux * torch.log(expected_flux / expected_flux.T),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).sum()
        / sim.dt
    )
    assert torch.isclose(observed_epr2, expected_epr2), (
        "Second-order local entropy production rate incorrect"
    )

    # test 5: verify explicit propagation
    initial_dist = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=sim.device
    )
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, device=sim.device)
    observed_dists = sim.propagate(times, initial_dist=initial_dist)
    expected_dists = torch.stack(
        [
            initial_dist,
            sim.L @ initial_dist,
            sim.L @ (sim.L @ initial_dist),
        ]
    )
    assert torch.allclose(observed_dists, expected_dists), (
        "Propagated distributions incorrect"
    )

    # test 6: verify simulation output
    observed_times, observed_states = sim.sample(n_steps=3, n_trajs=2, initial_state=1)
    expected_times = torch.tile(
        torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64, device=sim.device),
        (2, 1),
    )
    assert observed_times.shape == (2, 4), "Sampled times have wrong shape"
    assert observed_states.shape == (2, 4), "Sampled states have wrong shape"
    assert torch.allclose(observed_times, expected_times), "Sampled times incorrect"
    assert torch.equal(
        observed_states[:, 0], torch.full((2,), 1, dtype=torch.long, device=sim.device)
    ), "Initial sampled states incorrect"
    assert torch.all((observed_states >= 0) & (observed_states < sim.n_states)), (
        "Sampled states out of range"
    )

    # test 7: verify deterministic first-passage times
    observed_fpts = sim.first_passage_times(1, 1, n_trajs=3)
    assert isinstance(observed_fpts, torch.Tensor), "Expcted torch.Tensor observed_fpts"
    assert torch.equal(
        observed_fpts, torch.zeros(3, dtype=torch.long, device=sim.device)
    ), "First-passage times incorrect"

    # test 8: verify dwell-time probabilities
    observed_dwell = sim.dwell_probabilities(max_timestep=4)
    stay_probs = torch.diag(sim.L)
    expected_dwell = torch.zeros(
        (sim.n_states, 4), dtype=torch.float64, device=sim.device
    )
    expected_dwell[:, 1:] = (
        stay_probs.unsqueeze(1)
        ** torch.arange(3, device=sim.device, dtype=torch.float64)
    ) * (1.0 - stay_probs).unsqueeze(1)
    assert torch.allclose(observed_dwell, expected_dwell), (
        "Dwell-time probabilities incorrect"
    )


@torch.no_grad()
def test_inhomogeneous_markov_chain(
    device_for_testing,
    simple_inhomogeneous_column_stochastic_stack,
    simple_stationary_dist,
):
    """
    Test parameterization and main methods of the inhomogeneous MarkovChain class.

    Parameters
    ----------
    device_for_testing : str
        Device to run tests on (e.g., 'cpu', 'cuda').
    simple_inhomogeneous_column_stochastic_stack : torch.Tensor
        3x4x4 inhomogeneous propagator stack with identity zeroth slice.
    simple_stationary_dist : torch.Tensor
        Stationary distribution shared by the nontrivial propagators.
    """
    sim = MarkovChain(
        simple_inhomogeneous_column_stochastic_stack, dt=0.5, device=device_for_testing
    )

    # test 1: verify inhomogeneous Markov-chain parameterization
    expected_props = simple_inhomogeneous_column_stochastic_stack.to(sim.device).to(
        torch.float64
    )
    assert sim.n_states == 4, "Number of states incorrect"
    assert sim.dt == 0.5, "Timestep size incorrect"
    assert sim.L is None, (
        "Inhomogeneous Markov chain should not store a single propagator"
    )
    assert isinstance(sim.Gs, torch.Tensor), "Expected torch.Tensor sim.Gs"
    assert torch.allclose(sim.Gs, expected_props), "Stored propagator stack incorrect"
    assert sim._get_Gs_index(0) == 1, "Step-0 propagator index incorrect"
    assert sim._get_Gs_index(1) == 2, "Step-1 propagator index incorrect"
    assert sim._get_Gs_index(5) == 2, "Late-time propagator index incorrect"
    assert torch.allclose(sim._get_step_propagator(0), expected_props[1]), (
        "Step-0 propagator incorrect"
    )
    assert torch.allclose(sim._get_step_propagator(1), expected_props[2]), (
        "Step-1 propagator incorrect"
    )
    assert torch.allclose(sim._get_final_propagator(), expected_props[2]), (
        "Final propagator incorrect"
    )

    # test 2: verify stationary distribution of long-time propagators
    expected_statdist = simple_stationary_dist.to(sim.device).to(torch.float64)
    observed_statdist = sim.stationary_distribution()
    assert torch.allclose(observed_statdist, expected_statdist), (
        "Stationary distribution incorrect"
    )
    assert torch.allclose(expected_props[1] @ expected_statdist, expected_statdist), (
        "First nontrivial propagator has wrong stationary distribution"
    )
    assert torch.allclose(expected_props[2] @ expected_statdist, expected_statdist), (
        "Second nontrivial propagator has wrong stationary distribution"
    )

    # test 3: verify stationary flux after the zeroth identity propagator
    observed_flux = sim.get_flux()
    expected_flux = torch.stack(
        [
            expected_props[1] @ torch.diag(expected_statdist),
            expected_props[2] @ torch.diag(expected_statdist),
        ]
    )
    assert observed_flux.shape == (2, sim.n_states, sim.n_states), (
        "Stationary flux has wrong shape"
    )
    assert torch.allclose(observed_flux, expected_flux), "Stationary flux incorrect"

    # test 4: verify second-order local entropy production rates after the zeroth propagator
    observed_epr2 = sim.get_2nd_local_epr()
    expected_epr2 = (
        torch.nan_to_num(
            expected_flux * torch.log(expected_flux / expected_flux.transpose(-1, -2)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).sum(dim=(-1, -2))
        / sim.dt
    )
    assert torch.allclose(observed_epr2, expected_epr2), (
        "Second-order local entropy production rates incorrect"
    )

    # test 5: verify explicit propagation
    initial_dist = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=sim.device
    )
    times = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64, device=sim.device)
    observed_dists = sim.propagate(times, initial_dist=initial_dist)
    expected_dists = torch.stack(
        [
            initial_dist,
            expected_props[1] @ initial_dist,
            expected_props[2] @ (expected_props[1] @ initial_dist),
            expected_props[2]
            @ (expected_props[2] @ (expected_props[1] @ initial_dist)),
        ]
    )
    assert torch.allclose(observed_dists, expected_dists), (
        "Propagated distributions incorrect"
    )

    # test 6: verify simulation output
    observed_times, observed_states = sim.sample(n_steps=3, n_trajs=2, initial_state=1)
    expected_times = torch.tile(
        torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64, device=sim.device),
        (2, 1),
    )
    assert observed_times.shape == (2, 4), "Sampled times have wrong shape"
    assert observed_states.shape == (2, 4), "Sampled states have wrong shape"
    assert torch.allclose(observed_times, expected_times), "Sampled times incorrect"
    assert torch.equal(
        observed_states[:, 0], torch.full((2,), 1, dtype=torch.long, device=sim.device)
    ), "Initial sampled states incorrect"
    assert torch.all((observed_states >= 0) & (observed_states < sim.n_states)), (
        "Sampled states out of range"
    )

    # test 7: verify deterministic first-passage times
    observed_fpts = sim.first_passage_times(1, 1, n_trajs=3)
    assert isinstance(observed_fpts, torch.Tensor), (
        "Expected torch.Tensor observed_fpts"
    )
    assert torch.equal(
        observed_fpts, torch.zeros(3, dtype=torch.long, device=sim.device)
    ), "First-passage times incorrect"

    # test 8: verify dwell-time probabilities
    observed_dwell = sim.dwell_probabilities(max_timestep=4)
    stay_probs_1 = torch.diag(expected_props[1])
    stay_probs_2 = torch.diag(expected_props[2])
    expected_dwell = torch.zeros(
        (sim.n_states, 4), dtype=torch.float64, device=sim.device
    )
    expected_dwell[:, 1] = 1.0 - stay_probs_1
    expected_dwell[:, 2] = stay_probs_1 * (1.0 - stay_probs_2)
    expected_dwell[:, 3] = stay_probs_1 * stay_probs_2 * (1.0 - stay_probs_2)
    assert torch.allclose(observed_dwell, expected_dwell), (
        "Dwell-time probabilities incorrect"
    )


@torch.no_grad()
def test_get_3rd_local_epr_matches_manual_computation(
    device_for_testing,
    simple_inhomogeneous_column_stochastic_stack,
):
    """Third-order local EPR should match the explicit path-probability sum."""
    sim = MarkovChain(
        simple_inhomogeneous_column_stochastic_stack, dt=0.5, device=device_for_testing
    )
    assert isinstance(sim.Gs, torch.Tensor), "Expected torch.Tensor sim.Gs"
    G1 = sim.Gs[1]
    G2 = sim.Gs[2]
    pi = sim.stationary_distribution()
    expected_epr3 = torch.tensor(0.0, dtype=torch.float64, device=sim.device)

    for i in range(sim.n_states):
        for j in range(sim.n_states):
            for k in range(sim.n_states):
                forward = G2[i, j] * G1[j, k] * pi[k]
                reverse = G2[k, j] * G1[j, i] * pi[i]
                expected_epr3 += forward * torch.log(forward / reverse)
    expected_epr3 = expected_epr3 / (2.0 * sim.dt)

    observed_epr3 = sim.get_3rd_local_epr()
    assert torch.isclose(observed_epr3, expected_epr3), (
        "Third-order local entropy production rate incorrect"
    )


@torch.no_grad()
def test_get_3rd_local_epr_identical_inhomogeneous_propagators_is_zero(
    device_for_testing,
    reversible_column_stochastic_matrix,
):
    """Third-order local EPR should vanish when identical propagators are reversible."""
    prop = reversible_column_stochastic_matrix.to(torch.float64)
    stack = torch.stack((torch.eye(prop.shape[0], dtype=torch.float64), prop, prop))
    sim = MarkovChain(stack, dt=0.5, device=device_for_testing)

    observed_epr3 = sim.get_3rd_local_epr()
    expected_epr3 = torch.tensor(0.0, dtype=torch.float64, device=sim.device)

    assert torch.isclose(observed_epr3, expected_epr3, atol=1e-12), (
        "Third-order local entropy production rate should vanish for identical reversible propagators"
    )


@torch.no_grad()
def test_get_3rd_local_epr_homogeneous_is_zero(
    device_for_testing,
    simple_column_stochastic_matrix,
):
    """Third-order local EPR should vanish for homogeneous Markov chains."""
    sim = MarkovChain(
        simple_column_stochastic_matrix, dt=0.5, device=device_for_testing
    )

    assert sim.get_3rd_local_epr() == 0.0, (
        "Third-order local entropy production rate should vanish for homogeneous chains"
    )


torch.no_grad()


def test_gillespie(device_for_testing, simple_cycle_rates):
    """
    Test statistical properties of the Gillespie algorithm implementation.

    Parameters
    ----------
    device_for_testing : str
        Device to run tests on (e.g., 'cpu', 'cuda').
    simple_cycle_rates : torch.Tensor
        4x4 matrix of transition rates for a simple cycle.
    """
    sim = GillespieMJP(device=device_for_testing)
    sim.parameterize_from_rates(simple_cycle_rates)

    # total rate = 3.0 (rate to 1 = 2.0, rate to 3 = 1.0)
    # probability of transition to 1 = 2/3
    # probability of transition to 3 = 1/3

    # run a long simulation to gather statistics
    n_steps = 100000
    times, states = sim.sample(n_steps, initial_state=0)

    # convert to numpy for statistical tests
    times_np = times.cpu().numpy()
    states_np = states.cpu().numpy()

    # Test 1: waiting time distribution
    state_0_indices = np.where(states_np[:-1] == 0)[0]
    waiting_times_0 = times_np[state_0_indices + 1] - times_np[state_0_indices]

    # known total rate for state 0 is 1.0 + 2.0 = 3.0
    total_rate_0 = 3.0

    # test exponential distribution of waiting times
    _, p_value = stats.kstest(
        waiting_times_0,
        "expon",
        args=(0, 1 / total_rate_0),  # loc=0, scale=1/rate
    )
    print("p_value", p_value)
    assert p_value > 0.05, f"Waiting time distribution test failed with p-value {
        p_value
    }"

    # test 2: transition probabilities
    state_0_transitions = states_np[state_0_indices + 1]
    unique, counts = np.unique(state_0_transitions, return_counts=True)
    transition_dict = dict(zip(unique, counts))

    # calculate empirical probabilities
    total_transitions = sum(counts)

    # known theoretical probabilities
    theoretical_prob_to_1 = 2 / 3
    theoretical_prob_to_3 = 1 / 3

    # Chi-square test for transition probabilities
    observed_transition_probs = np.array(
        [transition_dict.get(1, 0), transition_dict.get(3, 0)]
    )
    expected_transition_probs = (
        np.array([theoretical_prob_to_1, theoretical_prob_to_3]) * total_transitions
    )

    _, p_transitions = stats.chisquare(
        observed_transition_probs, expected_transition_probs
    )
    assert p_transitions > 0.05, f"Transition probability test failed with p-value {
        p_transitions
    }"

    # test 3: mean waiting time
    empirical_mean = np.mean(waiting_times_0)
    true_mean = 1 / total_rate_0  # Should be 2 / 3
    relative_error = abs(empirical_mean - true_mean) / true_mean
    print(empirical_mean, true_mean, relative_error)
    assert relative_error < 0.1, f"Mean waiting time error too large: {relative_error}"

    # test 4: variance of waiting times
    empirical_var = np.var(waiting_times_0)
    true_var = 1 / (total_rate_0**2)  # Should be 4 / 9
    relative_error_var = abs(empirical_var - true_var) / true_var
    assert relative_error_var < 0.2, f"Waiting time variance error too large: {
        relative_error_var
    }"

    # test 5: verify all states are accessible
    visited_states = set(states_np)
    assert len(visited_states) == sim.n_states, "Not all states were visited"

    # test 6: Chi-square test for stationary distribution
    state_durations = np.zeros(sim.n_states)
    for i in range(len(times_np) - 1):
        duration = times_np[i + 1] - times_np[i]
        state_durations[states_np[i]] += duration

    observed_statdist = state_durations / np.sum(state_durations)
    expected_statdist = sim.stationary_distribution().cpu().numpy()

    _, p_statdist = stats.chisquare(observed_statdist, expected_statdist)
    assert p_statdist > 0.05, f"Distribution equality test failed with p-value {
        p_statdist
    }"
