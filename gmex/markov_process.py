# gmex/markov_process.py
# contains classes for simulating Markov jump processes


from abc import ABC, abstractmethod
from warnings import warn

import torch
from torch.linalg import matrix_exp
from tqdm import tqdm

from .utils import (
    check_column_stochastic_matrix,
    column_stochastic_lim_dist,
    get_random_init_dist,
)


class MarkovProcess(ABC):
    """Abstract base class for Markov processes.

    Parameters
    ----------
    device : str or torch.device, optional
        If using GPU 'cuda'; otherwise 'cpu'.

    Attributes
    ----------
    n_states : int
        Number of states. Assigned by subclasses.

    Methods
    -------
    __init__(device: str | torch.device | None = None)
        Initialize Markov process.
    sample() -> tuple[torch.Tensor, torch.Tensor]
        Simulate Markov process. Defined by subclasses.
    stationary_distribution() -> torch.Tensor
        Compute stationary distribution. Defined by subclasses.
    """

    n_states: int

    @torch.no_grad()
    def __init__(self, device: str | torch.device | None = None):
        self.device = (
            device
            if device
            # set device to GPU if available
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._parameterized = False  # MJP is not initially parameterized

    def _set_parameterized(self):
        """Set the parameterization state to True."""
        if self._parameterized:  # warn if MJP already parameterized
            warn("Parameters overwritten", UserWarning)
        self._parameterized = True  # set parameterization state to True

    @abstractmethod
    def sample(
        self,
        n_steps: int,
        n_trajs: int = 1,
        initial_state: int | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulates Markov process."""

    @abstractmethod
    def stationary_distribution(self) -> torch.Tensor:
        """Compute stationary distribution."""

    @torch.no_grad()
    def _get_initial_states(
        self, n_trajs: int, initial_state: int | None = None
    ) -> torch.Tensor:
        """Get initial states for simulation.

        Parameters
        ----------
        n_trajs : int
            Number of parallel trajectories to simulate.
        initial_state : int, optional
            Initial state of the system for all trajectories.

        Returns
        -------
        initial_states : (n_trajs,) torch.Tensor
            Initial states for each trajectory.
        """
        if initial_state is None:
            initial_states = torch.multinomial(
                self.stationary_distribution(), n_trajs, replacement=True
            ).to(self.device)
        else:
            initial_states = torch.full(
                (n_trajs,), initial_state, dtype=torch.long, device=self.device
            )
        return initial_states


class MarkovChain(MarkovProcess):
    """(Discrete-time) Markov chain (MC).

    Parameters
    ----------
    prop : torch.Tensor
        Transition-probability matrix or stack of matrices.
        If 2D, element i, j is probability of one-step transition from j to i.
        If 3D, prop[0] must be identity and prop[min(i + 1, N - 1)] is used
        at timestep i.
    dt : float
        Timestep size.
    device : str or torch.device, optional
        If using GPU 'cuda'; otherwise 'cpu'.

    Attributes
    ----------
    n_states : int
        Number of states.

    Methods
    -------
    __init__(device: str | torch.device | None = None) -> None
        Initialize discrete-time Markov chain.
    sample(n_steps: int, n_trajs: int = 1, initial_state: int | None = None, verbose: bool = False) -> torch.Tensor
        Sample n_trajs Markov-chain trajectories for n_steps steps.
    stationary_distribution() -> torch.Tensor
        Compute stationary distribution of the propagator used at long times.
    propagate(times: torch.Tensor, initial_dist: torch.Tensor | None = None) -> torch.Tensor
        Explicitly propagate a distribution.
    first_passage_times() -> torch.Tensor | int
        Sample first-passage times to a set of end states.
    dwell_probabilities(max_timestep: int = 1000) -> torch.Tensor
        Compute dwell-time probabilities for each state.

    Reference
    ---------
    [1] A. A. Markov, Rasprostranenie zakona bol'shikh chisel na velichiny,
        Izvestiya Fiziko-Matematicheskogo Obshchestva pri Kazanskom Universitete 2, 135 (1906).
    """

    @torch.no_grad()
    def __init__(
        self, prop: torch.Tensor, dt: float, device: str | torch.device | None = None
    ) -> None:
        super().__init__(device)
        if prop.ndim == 2:
            if prop.shape[0] != prop.shape[1]:
                raise ValueError("prop must be square.")
            check_column_stochastic_matrix(prop, name="prop")
            self.L = (
                (prop / prop.sum(dim=0, keepdim=True)).to(self.device).to(torch.float64)
            )
            self.Gs = None
        elif prop.ndim == 3:
            if prop.shape[0] < 1:
                raise ValueError("prop must contain at least one transition matrix.")
            if prop.shape[1] != prop.shape[2]:
                raise ValueError("prop must have square slices.")
            for tpm in prop:
                check_column_stochastic_matrix(tpm, name="tpm (slice of prop)")
            self.L = None
            self.Gs = (
                (prop / prop.sum(dim=1, keepdim=True)).to(self.device).to(torch.float64)
            )
            eye = torch.eye(self.Gs.shape[1], dtype=self.Gs.dtype, device=self.device)
            if not torch.allclose(self.Gs[0], eye, rtol=1e-8, atol=1e-8):
                raise ValueError(
                    "prop[0] must be (approximately) identity when prop is 3D."
                )
        else:
            raise ValueError(
                "prop must be an (M, M) matrix or an (N, M, M) stack of matrices."
            )

        self.dt = float(dt)
        self.n_states = int(prop.shape[-1])
        self._set_parameterized()

    @torch.no_grad()
    def _get_initial_states(
        self, n_trajs: int, initial_state: int | None = None
    ) -> torch.Tensor:
        """Get initial states for simulation."""
        if initial_state is None and self.Gs is not None:
            warn(
                "Sampling from stationary distribution of final TPM for inhomogeneous Markov chain.",
                UserWarning,
            )
        return super()._get_initial_states(n_trajs, initial_state)

    @torch.no_grad()
    def _get_Gs_index(self, step: int) -> int:
        """Get index of transition matrix used at a given step."""
        if self.Gs is None:
            return 0
        return min(step + 1, self.Gs.shape[0] - 1)

    @torch.no_grad()
    def _get_cumsum_probs(self) -> torch.Tensor:
        """Get cumulative transition probabilities for sampling using searchsorted."""
        if self.Gs is None:
            assert isinstance(self.L, torch.Tensor)
            cumsum_probs = torch.cumsum(self.L, dim=0).T
            cumsum_probs[:, -1] = 1.0
        else:
            assert isinstance(self.Gs, torch.Tensor)
            cumsum_probs = torch.cumsum(self.Gs, dim=1).transpose(1, 2)
            cumsum_probs[:, :, -1] = 1.0
        return cumsum_probs

    @torch.no_grad()
    def _get_step_propagator(self, step: int) -> torch.Tensor:
        """Get propagator used at a given step."""
        if self.Gs is None:
            assert isinstance(self.L, torch.Tensor)
            return self.L
        return self.Gs[self._get_Gs_index(step)]

    @torch.no_grad()
    def _get_final_propagator(self) -> torch.Tensor:
        """Get propagator repeated at long times."""
        if self.Gs is None:
            assert isinstance(self.L, torch.Tensor)
            return self.L
        return self.Gs[-1]

    @torch.no_grad()
    def _to_states(
        self, states: int | list[int] | torch.Tensor, name: str
    ) -> torch.Tensor:
        """Convert a state specification to a unique tensor of valid states."""
        if isinstance(states, int):
            states = torch.tensor([states], dtype=torch.long, device=self.device)
        else:
            states = torch.as_tensor(
                states, dtype=torch.long, device=self.device
            ).flatten()

        if states.numel() == 0:
            raise ValueError(f"{name} must be non-empty.")
        if ((states < 0) | (states >= self.n_states)).any():
            raise ValueError(f"All {name} must be in [0, n_states).")
        return torch.unique(states)

    @torch.no_grad()
    def stationary_distribution(self) -> torch.Tensor:
        """Compute stationary distribution.

        Returns
        -------
        pi : (n_states,) torch.Tensor
            Stationary distribution of the propagator used at long times.
            For inhomogeneous chains, this is the stationary distribution of
            the final propagator in the stack.
        """
        return column_stochastic_lim_dist(self._get_final_propagator())

    @torch.no_grad()
    def get_flux(self) -> torch.Tensor:
        """Get stationary flux matrices of propagators.

        Returns
        -------
        flux : torch.Tensor
            Stationary flux matrix of the homogeneous propagator,
            or stationary flux matrices of the inhomogeneous propagators
            after the zeroth identity slice.
        """
        if self.Gs is None:
            pi = self.stationary_distribution()
            return self.L @ torch.diag(pi)
        flux = torch.zeros_like(self.Gs[1:])
        for i in range(1, self.Gs.shape[0]):
            pi = column_stochastic_lim_dist(self.Gs[i])
            flux[i - 1] = self.Gs[i] @ torch.diag(pi)
        return flux

    @torch.no_grad()
    def propagate(
        self, times: torch.Tensor, initial_dist: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Explicitly propagate a distribution through the Markov chain.

        Parameters
        ----------
        times : torch.Tensor
            1D tensor of times at which to compute the distribution. Each time
            must be a nonnegative integer multiple of dt.
        initial_dist : (n_states,) torch.Tensor, optional
            Initial distribution. If None, a random distribution is used.

        Returns
        -------
        dists : (len(times), n_states) torch.Tensor
            Distribution at each requested time.
        """
        assert self._parameterized, (
            "Markov chain must be parameterized before propagation"
        )
        if times.ndim != 1:
            raise ValueError("times must be 1D.")
        if times.numel() == 0:
            raise ValueError("times must be non-empty.")

        times = times.to(self.device).to(torch.float64)
        if (times < 0).any():
            raise ValueError("times must be nonnegative.")

        steps = torch.round(times / self.dt).to(torch.long)
        if not torch.allclose(
            steps.to(torch.float64) * self.dt, times, rtol=1e-8, atol=1e-8
        ):
            raise ValueError("times must be integer multiples of dt.")
        if (steps[1:] < steps[:-1]).any():
            raise ValueError("times must be nondecreasing.")

        if initial_dist is None:
            initial_dist = get_random_init_dist(self.n_states).to(self.device)
        initial_dist = initial_dist.to(self.device).to(torch.float64).flatten()
        if initial_dist.numel() != self.n_states:
            raise ValueError("initial_dist must have shape (n_states,).")
        if (initial_dist < 0).any():
            raise ValueError("initial_dist must be nonnegative.")
        mass = initial_dist.sum()
        if not torch.isclose(mass, torch.ones_like(mass)):
            raise ValueError(f"initial_dist must sum to 1 but sums to {mass.item()}")

        dists = torch.empty(
            (times.shape[0], self.n_states), dtype=torch.float64, device=self.device
        )
        dist = initial_dist
        current_step = 0
        for idx, target_step in enumerate(steps.tolist()):
            while current_step < target_step:
                dist = self._get_step_propagator(current_step) @ dist
                current_step += 1
            dists[idx] = dist

        return dists

    @torch.no_grad()
    def sample(
        self,
        n_steps: int,
        n_trajs: int = 1,
        initial_state: int | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate n_trajs Markov-chain trajectories for n_steps steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to simulate.
        n_trajs : int, optional
            Number of parallel trajectories to simulate.
        initial_state : int, optional
            Initial state of the system for all trajectories.

        Returns
        -------
        times : (n_trajectories, n_steps + 1) torch.Tensor
            Times including initial time zero for each trajectory.
        states : (n_trajectories, n_steps + 1) torch.Tensor
            States at each time step including initial state at time zero for each trajectory.
        """
        assert self._parameterized, (
            "Markov chain must be parameterized before simulation"
        )

        times = torch.tile(
            torch.arange(n_steps + 1, dtype=torch.float64, device=self.device)
            * self.dt,
            (n_trajs, 1),
        )
        states = torch.zeros(n_trajs, n_steps + 1, dtype=torch.long, device=self.device)
        states[:, 0] = self._get_initial_states(n_trajs, initial_state)

        random_choices = torch.rand(n_trajs, n_steps, device=self.device)
        cumsum_probs = self._get_cumsum_probs()
        for step in tqdm(range(n_steps), desc="Timesteps", disable=not verbose):
            if self.Gs is None:
                row_cdf = cumsum_probs[states[:, step], :]
            else:
                row_cdf = cumsum_probs[self._get_Gs_index(step), states[:, step], :]
            states[:, step + 1] = torch.searchsorted(
                row_cdf.contiguous(), random_choices[:, step].contiguous().unsqueeze(1)
            ).squeeze(1)

        if n_trajs == 1:
            return times.squeeze(), states.squeeze()
        return times, states

    @torch.no_grad()
    def first_passage_times(
        self,
        start_states: int | list[int] | torch.Tensor,
        end_states: int | list[int] | torch.Tensor,
        start_dist: torch.Tensor | None = None,
        n_trajs: int = 1,
        max_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | int:
        """Simulate first-passage times to a set of end states.

        Parameters
        ----------
        start_states : int | list[int] | torch.Tensor
            States from which to initialize trajectories.
        end_states : int | list[int] | torch.Tensor
            States treated as absorbing targets.
        start_dist : (n_states,) torch.Tensor, optional
            Initial distribution used to sample starting states. Only mass on
            start_states is retained and renormalized.
        n_trajs : int, optional
            Number of trajectories to simulate in parallel.
        max_steps : int | None, optional
            Maximum number of steps to simulate.
        generator : torch.Generator | None, optional
            Random-number generator used for sampling.

        Returns
        -------
        fpts : int | torch.Tensor
            First-passage time or sorted first-passage times in units of
            discrete timesteps.
        """
        if n_trajs < 1:
            raise ValueError("n_trajs must be >= 1")

        start_states = self._to_states(start_states, "start_states")
        end_states = self._to_states(end_states, "end_states")

        if start_dist is None:
            probs = torch.zeros(self.n_states, dtype=torch.float64, device=self.device)
            probs[start_states] = 1.0
        else:
            start_dist = torch.as_tensor(
                start_dist, dtype=torch.float64, device=self.device
            ).flatten()
            if start_dist.numel() != self.n_states:
                raise ValueError("start_dist must have shape (n_states,).")
            if (start_dist < 0).any():
                raise ValueError("start_dist must be nonnegative.")
            probs = torch.zeros_like(start_dist)
            probs[start_states] = start_dist[start_states]

        mass = probs.sum()
        if mass <= 0:
            raise ValueError("No positive probability on the provided start_states.")
        probs = probs / mass

        cumsum_probs = self._get_cumsum_probs()
        states = torch.multinomial(
            probs, num_samples=n_trajs, replacement=True, generator=generator
        ).to(torch.long)
        steps = torch.zeros(n_trajs, dtype=torch.long, device=self.device)

        end_mask = torch.zeros(self.n_states, dtype=torch.bool, device=self.device)
        end_mask[end_states] = True
        alive = ~end_mask[states]

        step = 0
        while alive.any():
            if max_steps is not None and step >= max_steps:
                raise RuntimeError(
                    "max_steps reached before all trajectories hit end_states"
                )

            idx = torch.where(alive)[0]
            if self.Gs is None:
                row_cdf = cumsum_probs[states[idx], :]
            else:
                row_cdf = cumsum_probs[self._get_Gs_index(step), states[idx], :]
            random_choices = torch.rand(
                (idx.numel(), 1),
                dtype=row_cdf.dtype,
                device=self.device,
                generator=generator,
            )
            next_states = torch.searchsorted(
                row_cdf.contiguous(), random_choices, right=False
            ).squeeze(-1)

            states[idx] = next_states
            steps[idx] += 1
            alive[idx] = ~end_mask[next_states]
            step += 1

        return int(steps.item()) if n_trajs == 1 else torch.sort(steps).values

    @torch.no_grad()
    def dwell_probabilities(self, max_timestep: int = 1000) -> torch.Tensor:
        """Get dwell-time probabilities for each state.

        Parameters
        ----------
        max_timestep : int, optional
            Maximum number of dwell times for which to compute probabilities.

        Returns
        -------
        prob_dwell : (n_states, max_timestep) torch.Tensor
            Dwell-time probabilities.
            The probability of an exact j-step dwell in state i is prob_dwell[i, j].
        """
        if max_timestep < 1:
            raise ValueError("max_timestep must be >= 1")

        dtype = self._get_final_propagator().dtype
        prob_dwell = torch.zeros(
            (self.n_states, max_timestep), dtype=dtype, device=self.device
        )
        if max_timestep == 1:
            return prob_dwell

        if self.Gs is None:
            assert isinstance(self.L, torch.Tensor)
            stay_probs = torch.diag(self.L).unsqueeze(1).repeat(1, max_timestep - 1)
        else:
            tpm_indices = torch.arange(1, max_timestep, device=self.device)
            tpm_indices = torch.clamp(tpm_indices, max=self.Gs.shape[0] - 1)
            stay_probs = torch.diagonal(self.Gs[tpm_indices], dim1=1, dim2=2).T

        prob_alive = torch.ones(
            (self.n_states, max_timestep), dtype=dtype, device=self.device
        )
        prob_alive[:, 1:] = torch.cumprod(stay_probs, dim=1)
        prob_dwell[:, 1:] = prob_alive[:, :-1] - prob_alive[:, 1:]

        return prob_dwell

    @torch.no_grad()
    def get_2nd_local_epr(self) -> torch.Tensor:
        """Get stationary second-order local entropy production rate(s) of propagator(s).

        Returns
        -------
        epr2 : torch.Tensor
            Stationary second-order local entropy production rate of the homogeneous propagator
            or of the inhomogeneous propagators after the zeroth identity slice.
        """
        flux = self.get_flux()
        if torch.any(
            ((flux == 0) & (flux.transpose(-1, -2) != 0))
            | ((flux != 0) & (flux.transpose(-1, -2) == 0))
        ):
            raise ValueError(
                "Propagators must be microscopically reversible for physical EPR."
            )
        flux[flux == 0] = float("nan")
        epr_els = flux * torch.log(flux / flux.transpose(-1, -2))
        return epr_els.nansum(dim=(-1, -2)) / self.dt

    @torch.no_grad()
    def get_3rd_local_epr(self) -> torch.Tensor:
        """Get stationary third-order local entropy production rate(s) of propagator(s).

        Returns
        -------
        epr3 : torch.Tensor
            Stationary third-order local entropy production rate
            of the inhomogeneous propagators after the zeroth identity slice.
        """
        if self.Gs is None or len(self.Gs) < 3:
            return torch.tensor(
                0.0
            )  # 3rd-order EPR vanishes for homogeneous Markov chain
        D = torch.diag(self.stationary_distribution())
        path_probs = self.Gs[2][:, :, None] * (self.Gs[1] @ D)[None, :, :]

        path_probs_kji = path_probs.transpose(0, 2)
        if torch.any((path_probs_kji == 0) & (path_probs != 0)):
            raise ValueError(
                "Propagators must be microscopically reversible for physical EPR."
            )
        path_probs_kji[path_probs_kji == 0] = float("nan")

        epr_els = path_probs * torch.log(path_probs / path_probs_kji)
        return epr_els.nansum() / (2.0 * self.dt)


class MarkovJumpProcess(MarkovProcess):
    """Markov jump process (MJP).
    Following the parameterization in [1], off-diagonals obey
    rates[i, j] = exp(wells[j] - barriers[i, j] + forces[i, j] / 2).

    Attributes
    ----------
    n_states : int
        Number of states.
    rates : torch.Tensor
        Off-diagonal elements i, j are rates of i -> j transition, diagonal elements are 0.
        Not the transition-rate matrix, where off-diagonal elements i, j are rates of j -> i transition.
        User-specified when passed to parameterize_from_rates.
    wells : torch.Tensor
        Element i is depth of potential well at vertex i.
        User-specified when passed to parameterize_from_energies with barriers and forces.
    barriers : torch.Tensor
        Element i, j is energy barrier between vertices i and j.
        User-specified when passed to parameterize_from_energies with wells and forces.
    forces : torch.Tensor
        Element i, j is energy due to force driving from vertex i to j.
        User-specified when passed to parameterize_from_energies with wells and barriers.

    Methods
    -------
    __init__(device: str | torch.device | None = None)
        Initialize Markov jump process.
    parameterize_from_rates(rates: torch.Tensor)
        Compute nonunique parameters from rates using Eq. (6) from [1].
    parameterize_from_energies(wells torch.Tensor, barriers torch.Tensor, forces torch.Tensor)
        Compute unique parameters from energies using Eq. (6) from [1].
    transition_rate_matrix() -> torch.Tensor
        Compute master-equation transition-rate matrix from rate matrix.
    stationary_distribution() -> torch.Tensor
        Compute stationary distribution.
    stationary_influxes() -> torch.Tensor
        Compute stationary influxes of the Markov jump process (MJP).
    stationary_epr() -> float
        Compute the stationary entropy-production rate (EPR) using Eq. 7.6 from [2].
    generate(times: torch.Tensor, initial_dist: torch.Tensor | None = None) -> torch.Tensor
        Explicitly propagate fine-grained distribution using transition-rate matrix.

    References
    ----------
    [1] J.A. Owen, T.R. Gingrich, and J.M. Horowitz,
        Universal thermodynamic bounds on nonequilibrium response with biochemical applications,
        Phys. Rev. X 10, 011066 (2020).
    [2] M. Esposito and C. Van den Broeck,
        Three faces of the second law. I. Master equation formulation,
        Phys. Rev. E 82, 011143 (2010).
    [3] P. Espanol and F. Vazquez,
        Coarse graining from coarse-grained descriptions,
        Phil. Trans. R. Soc. A 360, 383 (2002).
    """

    @torch.no_grad()
    def __init__(self, device: str | torch.device | None = None):
        super().__init__(device=device)

    def sample(
        self,
        n_steps: int,
        n_trajs: int = 1,
        initial_state: int | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulates Markov jump process."""
        raise NotImplementedError("Use a child class to sample trajectories.")

    @torch.no_grad()
    def parameterize_from_rates(self, rates: torch.Tensor):
        """Compute parameters using Eq. (6) from [1].
        As parameters are nonunique, set wells to 0.
        Overwrites existing attributes.

        Parameters
        ----------
        rates : torch.Tensor
            Off-diagonal elements i, j are rates of i -> j transition.
            Note that this is different from the indexing of the transition-rate matrix.
        """
        assert rates.ndim == 2 and rates.shape[0] == rates.shape[1], (
            "rates must be square"
        )
        assert torch.all(rates >= 0), "rates must be nonnegative"
        assert torch.all(torch.diag(rates) == 0), "rates must have 0 diagonal"

        self.n_states = rates.shape[0]
        self.rates = rates.to(self.device)
        log_rates = torch.log(self.rates)

        self.wells = torch.zeros(
            self.n_states, device=self.device
        )  # arbitrarily set wells to 0
        self.barriers = -0.5 * (log_rates + log_rates.T)
        self.barriers.fill_diagonal_(0.0)  # barrier heights are zero on the diagonal
        self.forces = log_rates - log_rates.T

        # where rates are 0 log_rates are -inf
        # in torch, -inf-(-inf) == nan
        # to fix nan propagation set these to 0 in forces
        self.forces = torch.nan_to_num(
            self.forces, nan=0.0, posinf=float("inf"), neginf=-float("inf")
        )

        self._set_parameterized()  # set parameterization state to True

    @torch.no_grad()
    def parameterize_from_energies(
        self, wells: torch.Tensor, barriers: torch.Tensor, forces: torch.Tensor
    ):
        """Compute unique parameters using Eq. (6) from [1].
        Overwrites existing attributes.

        Parameters
        ----------
        wells : torch.Tensor
            Element i is depth of potential well at vertex i.
        barriers : torch.Tensor
            Element i, j is energy barrier between vertices i and j.
        forces : torch.Tensor
            Element i, j is energy due to force driving from vertex i to j.
        """
        assert wells.shape[0] == barriers.shape[0] == forces.shape[0], (
            "parameters must have same shape along first axis"
        )
        assert torch.all(barriers == barriers.T), "barriers must be symmetric"
        assert torch.all(torch.diag(barriers) == 0), "barriers must have 0 diagonal"
        assert torch.all(forces == -forces.T), "forces must be antisymmetric"

        self.n_states = wells.shape[0]
        self.wells = wells.to(self.device)
        self.barriers = barriers.to(self.device)
        self.forces = forces.to(self.device)

        self.rates = torch.exp(
            torch.tile(self.wells, (self.n_states, 1))
            - self.barriers
            + self.forces / 2.0
        )
        self.rates.fill_diagonal_(0.0)  # set diagonal rates to 0
        self._set_parameterized()  # set parameterization state to True

    @torch.no_grad()
    def transition_rate_matrix(self) -> torch.Tensor:
        """Compute master equation transition-rate matrix from rate matrix.
        This is the matrix W in dp/dt = Wp.
        """
        transposed_rates = self.rates.T.to(torch.float64)
        return transposed_rates - torch.diag(transposed_rates.sum(dim=0))

    @torch.no_grad()
    def stationary_distribution(self) -> torch.Tensor:
        """Compute stationary distribution of the Markov jump process (MJP)."""
        # get left eigenvectors
        trm = self.transition_rate_matrix()
        eigenvals, eigenvecs = torch.linalg.eig(trm)

        # Find index of eigenvalue closest to 0
        idx = torch.argmin(torch.abs(eigenvals))
        # Get corresponding eigenvector and normalize
        stationary = eigenvecs[:, idx].real
        return stationary / stationary.sum()

    @torch.no_grad()
    def stationary_influxes(self) -> torch.Tensor:
        """Compute stationary influxes of the Markov jump process (MJP).
        Element i, j is flux from j to i.
        """
        trm = self.transition_rate_matrix()
        p_star = self.stationary_distribution()
        influxes = trm * p_star.unsqueeze(0)  # i, j entry is trm[i, j] * p_star[j]
        influxes.fill_diagonal_(0.0)  # diagonals fluxes are 0
        return influxes

    @torch.no_grad()
    def stationary_epr(self) -> float:
        """Compute the stationary entropy-production rate (EPR) using Eq. 7.6 from [2]."""
        influxes = self.stationary_influxes()
        influxes[influxes == 0] = float("nan")
        epr_summands_doubled = (influxes - influxes.T) * torch.log(
            influxes / influxes.T
        )
        return epr_summands_doubled.nansum().item() / 2.0

    @torch.no_grad()
    def generate(
        self, times: torch.Tensor, initial_dist: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Explicitly propagate fine-grained distribution using transition-rate matrix.

        Parameters
        ----------
        times : torch.Tensor
            Times at which to compute the distribution.
        initial_distribution : int, optional
            Initial distribution.

        Returns
        -------
        dists : (len(times), n_states) torch.Tensor
            Fine-grained distribution at each timestep.
        """
        assert self._parameterized, "MJP must be parameterized before propagation"

        if initial_dist is None:
            initial_dist = get_random_init_dist(self.n_states).to(self.device)
        mass = initial_dist.sum()
        assert torch.isclose(mass, torch.ones_like(mass)), (
            f"initial_dist must sum to 1 but sums to {mass.item()}"
        )

        # reshape times to (len(times), 1, 1)
        trm = self.transition_rate_matrix().to(self.device)
        times = times.reshape(times.shape[0], 1, 1).to(self.device).to(torch.float64)
        propagators = matrix_exp(times * trm)

        return torch.matmul(
            propagators,
            initial_dist.reshape(self.n_states, 1).to(self.device).to(torch.float64),
        ).squeeze(-1)


class GillespieMJP(MarkovJumpProcess):
    """Continuous-time Markov jump process (MJP) with Gillespie algorithm.

    Parameters
    ----------
    device : str or torch.device, optional
        If using GPU 'cuda'; otherwise 'cpu'.

    Methods
    -------
    __init__(device: str | torch.device | None = None)
        Initialize Gillespie MJP.
    simulate(n_steps: int, n_trajectories: int = 1, initial_state: int | None = None, verbose: bool = False)
        Simulate a Markov jump process (MJP) for n_steps steps
        using the Gillespie algorithm [1, 2].

    References
    ----------
    [1] J. Doob, Topics in the theory of Markoff chains,
        Trans. Am. Math. Soc. 52, 37 (1942).
    [2] G. Gillespie, Exact stochastic simulation of coupled chemical reactions,
        J. Phys. Chem. 81, 2340 (1977).
    """

    @torch.no_grad()
    def __init__(self, device: str | torch.device | None = None):
        super().__init__(device=device)

    @torch.no_grad()
    def sample(
        self,
        n_steps: int,
        n_trajs: int = 1,
        initial_state: int | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n_trajs Markov jump process (MJP) trajectories for n_steps steps using the Gillespie algorithm.

        Parameters
        ----------
        n_steps : int
            Number of steps to simulate.
        n_trajs : int, optional
            Number of parallel trajectories to simulate.
        initial_state : int, optional
            Initial state of the system for all trajectories.

        Returns
        -------
        times : (n_trajectories, n_steps + 1) torch.Tensor
            Times including initial time zero for each trajectory.
        states : (n_trajectories, n_steps + 1) torch.Tensor
            States at each time step including initial state at time zero for each trajectory.
        """
        assert self._parameterized, "MJP must be parameterized before simulation"

        states = torch.zeros(n_trajs, n_steps + 1, dtype=torch.long, device=self.device)
        states[:, 0] = self._get_initial_states(n_trajs, initial_state)

        times = torch.zeros(
            n_trajs, n_steps + 1, dtype=torch.float64, device=self.device
        )
        current_times = torch.zeros(n_trajs, dtype=torch.float64, device=self.device)

        # Generate all random numbers at once
        random_times = torch.rand(
            n_trajs, n_steps, dtype=torch.float64, device=self.device
        )
        random_choices = torch.rand(n_trajs, n_steps, device=self.device)

        for step in tqdm(range(n_steps), desc="Timesteps", disable=not verbose):
            current_states = states[:, step]
            current_rates = self.rates[current_states, :]  # (n_trajectories, n_states)
            total_rates = current_rates.sum(dim=1)  # (n_trajectories,)

            # Handle trajectories with positive rates
            valid_mask = total_rates > 0

            if valid_mask.any():
                # Calculate time increments
                dt = (
                    -torch.log(random_times[valid_mask, step]) / total_rates[valid_mask]
                )
                current_times[valid_mask] += dt

                # Find next states
                cumsum_rates = torch.cumsum(
                    current_rates[valid_mask], dim=1
                ) / total_rates[valid_mask].unsqueeze(1)
                states[valid_mask, step + 1] = torch.searchsorted(
                    cumsum_rates, random_choices[valid_mask, step].unsqueeze(1)
                ).squeeze(1)

            # Copy current state for trajectories with zero rates
            states[~valid_mask, step + 1] = states[~valid_mask, step]
            times[:, step + 1] = current_times

        if n_trajs == 1:
            return times.squeeze(), states.squeeze()
        return times, states
