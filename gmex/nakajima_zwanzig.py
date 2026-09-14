# gmex/core/generalized_master.py
# contains classes for generalized master equations


import torch
from torch.linalg import matrix_exp, matrix_power
from tqdm import tqdm

from .markov_process import MarkovChain, MarkovJumpProcess, MarkovProcess
from .utils import (
    allocate_dists,
    check_column_stochastic_matrix,
    get_macrostate_indicator,
    get_p_micro_given_macro,
    validate_groups,
)


class NakajimaZwanzigGME:
    """Nakajima-Zwanzig generalized master equation."""

    @torch.no_grad()
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def _integrate_nzgmedt(
        kernels: torch.Tensor,
        n_steps: int,
        dt: float | None = None,
        initial_dist: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Integrate NZGME using convolution sum (or integral).

        Parameters
        ----------
        kernels : torch.Tensor
            Kernels to convolve.
        n_steps : int
            Number of steps to integrate for.
        n_macrostates : int
            Number of macrostates in support.
        dt : float, optional
            Integration timestep for continuous-time case.
        dists : torch.Tensor
            Buffer in which to save dists.
        initial_dist : torch.Tensor, optional
            Initial distribution over macrostates.
            Randomized if not provided.
        verbose : bool, optional
            Whether to show progress bar.

        Returns
        -------
        dists : torch.Tensor
            Integrated distributions.
        """
        # get buffer in which to save integrated distributions
        dists = allocate_dists(
            n_steps, kernels.shape[-1], initial_dist=initial_dist
        ).to(kernels.device)

        for i in tqdm(
            range(n_steps),
            desc="Timesteps integrated",
            disable=not verbose,
            leave=False,
        ):
            memory = convolve_sum(i, kernels, dists)
            if dt is not None:  # continuous-time case
                dists[i + 1] = dists[i] + dt**2 * memory
            else:  # discrete-time case
                dists[i + 1] = memory

        return dists


class GroundTruthNZGME(NakajimaZwanzigGME):
    """NZGME parameterized from analytical coarse-graining.

    Parameters
    ----------
    mjp : MarkovJumpProcess
        Continuous-time Markov jump process to coarse-grain.
    groups : list of list of int
        Groups of states to coarse-grain. For example,
        [[0, 1], [2, 3, 4]] would group states 1 and 2 together and 3, 4, and 5 together.
    """

    @torch.no_grad()
    def __init__(self, process: MarkovProcess, groups: list):
        self.process = process
        self.device = self.process.device
        validate_groups(self.process.n_states, groups)
        self.groups = groups
        self.n_macrostates = len(self.groups)

        # macrostate_indicator[s, alpha] = 1 if state s is in macrostate alpha
        self.macrostate_indicator = get_macrostate_indicator(self.groups).to(
            self.device
        )

        # compute the macrostate stationary distribution
        self.stationary_distribution_micro = self.process.stationary_distribution()
        self.stationary_distribution_macro = (
            self.macrostate_indicator.T @ self.stationary_distribution_micro
        )
        # stationary probabilities of microstates given macrostates, (n_states, n_macrostates)
        self.p_micro_given_macro = get_p_micro_given_macro(
            self.macrostate_indicator, self.stationary_distribution_micro
        )

        # compute the relevant projector P, (n_states, n_states)
        self.mz_projector_P = self.macrostate_indicator @ self.p_micro_given_macro.T
        # compute the irrelevant projector Q, (n_states, n_states)
        self.mz_projector_Q = (
            torch.eye(self.process.n_states).to(self.device) - self.mz_projector_P
        )


class DiscreteTimeGroundTruthNZGME(GroundTruthNZGME):
    """Discrete-time NZGME with ground-truth parameters.

    Parameters
    ----------
    mjp : MarkovJumpProcess
        Continuous-time Markov jump process to coarse-grain.
    groups : list of list of int
        Groups of states to coarse-grain. For example,
        [[0, 1], [2, 3, 4]] would group states 1 and 2 together and 3, 4, and 5 together.

    Methods
    -------
    get_u_dtgme(deltat: float, m: int) -> torch.Tensor
        Compute the m-timestep discrete-time GME transfer matrix with timestep deltat.
    get_k_dtgme(deltat: float, m: int) -> torch.Tensor
        Compute the discrete-time GME kernel at a delay of m timesteps deltat.
    integrate(deltat: float, n_steps: int, initial_dist: torch.Tensor | None = None) -> torch.Tensor
        Exactly integrate ground-truth discrete-time GME.
    """

    @torch.no_grad()
    def __init__(self, process: MarkovProcess, groups: list):
        super().__init__(process, groups)
        if isinstance(self.process, MarkovChain):
            self.dt = self.process.dt
            if self.process.L is None:
                raise ValueError(
                    "DiscreteTimeGroundTruthNZGME does not support inhomogeneous Markov chains. "
                    "Convert propagators to transfer matrices and pass to TransferMatrixNZGME."
                )
            self.L = self.process.L
        elif isinstance(self.process, MarkovJumpProcess):
            self.trm = self.process.transition_rate_matrix()
        else:
            raise TypeError("process must be MarkovChain or MarkovJumpProcess")

    @torch.no_grad()
    def get_U_dtgme(self, L: torch.Tensor, m: int) -> torch.Tensor:
        """Compute the discrete-time GME transfer matrix.

        Parameters
        ----------
        L : torch.Tensor
            Transition probability matrix.
        m : int
            Timestep for which to calculate transfer matrix.

        Returns
        -------
        u_dtgme : torch.Tensor
            Discrete-time GME transfer matrix at time m * deltat
        """
        return (
            self.macrostate_indicator.T @ matrix_power(L, m) @ self.p_micro_given_macro
        )

    @torch.no_grad()
    def get_K_dtgme(self, L: torch.Tensor, m: int) -> torch.Tensor:
        """Compute the discrete-time GME memory kernel.

        Parameters
        ----------
        L : torch.Tensor
            Transition probability matrix.
        m : int
            Delay index for which to calculate memory kernel.

        Returns
        -------
        k_dtgme : torch.Tensor
            Discrete-time GME memory kernel at delay m * deltat
        """
        return (
            self.macrostate_indicator.T
            @ L
            @ matrix_power(self.mz_projector_Q.T @ L, m)
            @ self.p_micro_given_macro
        )

    @torch.no_grad()
    def integrate(
        self,
        deltat: float,
        n_steps: int,
        kernels: torch.Tensor | None = None,
        initial_dist: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exactly integrate ground-truth discrete-time GME.

        Parameters
        ----------
        deltat : float
            Timestep size for MarkovJumpProcess parameterization.
            If MarkovChain is used for parameterization, this must match its dt.
        n_steps : int
            Number of steps to integrate.
        kernels : torch.Tensor, optional
            Precomputed memory kernels.
        initial_dist : int, optional
            Initial distribution.
            Randomized if not provided.
        verbose : bool, optional
            Whether to show progress bar.

        Returns
        -------
        kernels : (n_steps, n_macrostates, n_macrostates) torch.Tensor
            Kernels used to compute distributions.
        dists : (n_steps + 1, n_macrostates) torch.Tensor
            Coarse-grained distribution at each timestep.
        """
        if isinstance(self.process, MarkovChain):
            if not torch.isclose(
                torch.tensor(deltat % self.process.dt), torch.tensor(0.0)
            ):
                raise ValueError("deltat must be an integer multiple of MarkovChain dt")
            L = matrix_power(self.L, round(deltat / self.process.dt))
        elif isinstance(self.process, MarkovJumpProcess):
            L = matrix_exp(self.trm * deltat)
        else:
            raise TypeError("process must be MarkovChain or MarkovJumpProcess")
        # there are stochasticity checks in MarkovChain and MarkovJumpProcess
        L = L / L.sum(dim=0, keepdim=True)  # but we still ensure stochasticity

        if kernels is not None:
            assert kernels.ndim == 3, "kernels must be 3D"
            assert kernels.shape[1] == kernels.shape[2], (
                "kernels must have square slices"
            )
            kernels = kernels.to(device=self.device, dtype=torch.float64)
        else:
            kernels = torch.zeros(
                n_steps, self.n_macrostates, self.n_macrostates, dtype=torch.float64
            ).to(self.device)
            kernel_prefactor = self.macrostate_indicator.T @ L
            irrelevant_L = self.mz_projector_Q.T @ L
            for m in tqdm(
                range(n_steps),
                desc="Kernels computed",
                disable=not verbose,
                leave=False,
            ):
                kernels[m] = kernel_prefactor @ self.p_micro_given_macro
                kernel_prefactor = kernel_prefactor @ irrelevant_L

        dists = super()._integrate_nzgmedt(
            kernels, n_steps, initial_dist=initial_dist, verbose=verbose
        )

        return kernels, dists


class TransferMatrixNZGME(NakajimaZwanzigGME):
    """Discrete-time NZGME with kernels from transfer matrices.

    Parameters
    ----------
    Us : torch.Tensor
        Transfer matrices.

    Attribute
    ---------
    device : torch.device
        Computing device.
    n_states : int
        Number of macrostates.

    Methods
    -------
    get_Ks(n_delays: int) -> torch.Tensor
        Get memory kernels up to n_delays delays.
    integrate(n_steps: int, initial_dist: torch.Tensor | None = None) -> torch.Tensor
        Integrate discrete-time GME up to n_steps steps.
    """

    @torch.no_grad()
    def __init__(self, Us: torch.Tensor):
        assert Us.shape[1] == Us.shape[2], "ps must have same dimensions 1 and 2"
        _, self.n_macrostates, _ = Us.shape
        self.Us = Us
        self.device = Us.device

    @torch.no_grad()
    def get_Ks(self, verbose: bool = True) -> torch.Tensor:
        """Estimate discrete-time memory kernels up to n_delays delays.

        Parameter
        ---------
        verbose : bool, optional
            Whether to show progress bar.

        Returns
        -------
        Ks : torch.Tensor
            Discrete-time memory kernels up to n_delays delays.
        """
        ks = get_Ks_from_Us(self.Us, verbose=verbose)
        return ks

    @torch.no_grad()
    def integrate(
        self,
        n_steps: int,
        kernels: torch.Tensor | None = None,
        initial_dist: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate discrete-time GME up to n_steps steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to integrate.
        kernels : torch.Tensor, optional
            Precomputed memory kernels.
        initial_dist : torch.Tensor
            Initial distribution.
            Randomized if not provided.
        verbose : bool, optional
            Whether to show progress bar.

        Returns
        -------
        kernels : (n_steps, n_macrostates, n_macrostates) torch.Tensor
            Kernels used to compute distributions.
        dists : (n_steps + 1, n_macrostates) torch.Tensor
            Coarse-grained distribution at each timestep.
        """
        if kernels is not None:
            assert kernels.ndim == 3, "kernels must be 3D"
            assert kernels.shape[1] == kernels.shape[2], (
                "kernels must have square slices"
            )
            kernels = kernels.to(device=self.device, dtype=torch.float64)
        else:
            kernels = self.get_Ks(verbose=verbose).to(torch.float64)
        dists = super()._integrate_nzgmedt(
            kernels, n_steps, initial_dist=initial_dist, verbose=verbose
        )
        return kernels, dists


### HELPERS ###


@torch.no_grad()
def convolve_sum(
    max_delay: int, kernels: torch.Tensor, dists: torch.Tensor
) -> torch.Tensor:
    """Convolve kernels with past distributions up to time i and sum.

    Parameters
    ----------
    max_delay : int
        Maximum delay timesteps to include in convolution
    kernels : torch.Tensor
        Kernels precomputed for delays
    dists : torch.Tensor
        Distributions over time, filled up to slice i inclusive

    Returns
    -------
    summed_convolve : torch.Tensor
        Derivative term due to memory (continuous) or next step (discrete).
    """
    assert kernels.shape[1] == kernels.shape[2], (
        "kernel at each timestep must be square"
    )
    assert kernels.shape[1] == dists.shape[1], (
        "dist support cardinality must equal second dimension of kernels"
    )

    if kernels.shape[0] <= max_delay:  # case with max delay more than available kernels
        return (
            (
                kernels
                @ dists[max_delay + 1 - kernels.shape[0] : max_delay + 1]
                .flip(0)
                .unsqueeze(-1)
            )
            .squeeze(-1)
            .sum(0)
        )
    else:  # case with kernels computed for all delays
        return (
            (kernels[: max_delay + 1] @ dists[: max_delay + 1].flip(0).unsqueeze(-1))
            .squeeze(-1)
            .sum(0)
        )


@torch.no_grad()
def get_Ks_from_Us(Us: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    """Get discrete-time memory kernels from transfer matrices.

    Parameters
    ----------
    Us : torch.Tensor
        Transfer matrices.
    verbose : bool, optional
        Whether to display progress bar.

    Returns
    -------
    Ks : torch.Tensor
        Discrete-time memory kernels computed from ps.
    """
    timesteps, n_macrostates, _ = Us.shape
    for lag in range(timesteps):
        check_column_stochastic_matrix(Us[lag], name=f"U^{ ({lag}) }")
    Us_safe = Us / Us.sum(dim=1, keepdim=True)
    Ks = torch.zeros(
        (timesteps - 1, n_macrostates, n_macrostates), device=Us_safe.device
    ).to(Us_safe.dtype)

    for i in tqdm(
        range(timesteps - 1), desc="Kernels computed", disable=not verbose, leave=False
    ):
        if i == 0:  # base case
            Ks[i] = Us_safe[i + 1]
        else:  # recursion
            Ks[i] = Us_safe[i + 1] - torch.bmm(Us_safe[1 : i + 1], Ks[:i].flip(0)).sum(
                dim=0
            )

    return Ks


@torch.no_grad()
def extract_Us_dtnzgme(Ks: torch.Tensor, max_delay: int | None = None) -> torch.Tensor:
    """Get DT-NZ-GME-estimated transfer matrices by maximum delay.

    Parameters
    ----------
    Ks : torch.Tensor
        Discrete-time memory kernels.
    max_delay : int
        Maximum delay to integrate with.

    Returns
    -------
    integrals : torch.Tensor
        Transfer matrices estimated by DT-NZ-GME with memory truncated to max_delay.
    """
    if max_delay is not None and max_delay < 0:
        raise ValueError(f"max_delay must be nonnegative, got {max_delay}")
    if Ks.ndim != 3 or Ks.shape[1] != Ks.shape[2]:
        raise ValueError(
            f"Ks must be 3D and have square slices, got shape {tuple(Ks.shape)}."
        )

    integrals = torch.zeros(
        (Ks.shape[0] + 1, Ks.shape[1], Ks.shape[2]), device=Ks.device, dtype=Ks.dtype
    )
    for i in range(Ks.shape[1]):
        initial_dist = torch.zeros(Ks.shape[1], device=Ks.device, dtype=Ks.dtype)
        initial_dist[i] = 1.0
        # integrate using only delays up to max_delay N, i.e., \[\rho_{n+1} = \sum_{m=0}^{N} K_m\rho_{n-m}\]
        cutoff = max_delay if max_delay is None else max_delay + 1
        integrals[:, :, i] = NakajimaZwanzigGME._integrate_nzgmedt(
            Ks[:cutoff],
            Ks.shape[0],
            initial_dist=initial_dist,
            verbose=False,
        )
    return integrals


### UNUSED (and untested) ###
class EulerNZGME(GroundTruthNZGME):
    """Nakajima-Zwanzig equation with ground-truth parameters.

    Parameters
    ----------
    mjp : MarkovJumpProcess
        Continuous-time Markov jump process to coarse-grain.
    groups : list of list of int
        Groups of states to coarse-grain. For example,
        [[0, 1], [2, 3, 4]] would group states 1 and 2 together and 3, 4, and 5 together.

    Attributes
    ----------
    v_ctgme : torch.Tensor
        Markovian operator of continuous-time GME

    Methods
    -------
    get_k_ctgme(t: float) -> torch.Tensor:
        Compute the Nakajima-Zwanzig memory kernel.
    integrate(dt: float, n_steps: int, initial_dist: torch.Tensor | None = None) -> torch.Tensor:
        Euler-integrate analytical Nakajima-Zwanzig equation.
    """

    @torch.no_grad()
    def __init__(self, mjp: MarkovJumpProcess, groups: list):
        super().__init__(mjp, groups)
        self.trm = mjp.transition_rate_matrix()
        # Markovian operator of continuous-time GME, (n_macrostates, n_macrostates)
        self.v_ctgme = self.macrostate_indicator.T @ self.trm @ self.p_micro_given_macro

    @torch.no_grad()
    def get_k_ctgme(self, t: float) -> torch.Tensor:
        """Compute the Nakajima-Zwanzig memory kernel.

        Parameters
        ----------
        t : float
            Delay at which to compute memory kernel.

        Returns
        -------
        k_ctgme : torch.Tensor
            Nakajima-Zwanzig memory kernel at delay t.
        """
        QT_trm = self.mz_projector_Q.T @ self.trm
        return (
            self.macrostate_indicator.T
            @ self.trm
            @ QT_trm
            @ matrix_exp(QT_trm * t)
            @ self.p_micro_given_macro
        )

    @torch.no_grad()
    def integrate(
        self,
        dt: float,
        n_steps: int,
        kernels: torch.Tensor | None = None,
        initial_dist: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler-integrate ground-truth Nakajima-Zwanzig equation.

        Parameters
        ----------
        dt : float
            Timestep size.
        n_steps : int
            Number of steps to integrate.
        kernels : torch.Tensor, optional
            Precomputed memory kernels.
        initial_dist : int, optional
            Initial distribution.
            Randomized if not provided.
        verbose : bool, optional
            Whether to show progress bar.

        Returns
        -------
        kernels : (n_steps, n_macrostates, n_macrostates) torch.Tensor
            Kernels used to compute distributions.
        dists : (n_steps + 1, n_macrostates) torch.Tensor
            Coarse-grained distribution at each timestep.
        """
        if kernels is not None:
            assert kernels.ndim == 3, "kernels must be 3D"
            assert kernels.shape[1] == kernels.shape[2], (
                "kernels must have square slices"
            )
            kernels = kernels.to(device=self.device, dtype=torch.float64)
        else:
            kernels = torch.zeros(
                n_steps, self.n_macrostates, self.n_macrostates, dtype=torch.float64
            ).to(self.device)
            for i in tqdm(range(n_steps), desc="Kernels computed", leave=False):
                kernels[i] = self.get_k_ctgme(i * dt)
            kernels[0] += self.v_ctgme / dt

        dists = super()._integrate_nzgmedt(
            kernels,
            n_steps,
            dt=dt,
            initial_dist=initial_dist,
            verbose=verbose,
        )

        return kernels, dists
