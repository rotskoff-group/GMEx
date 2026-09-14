# gmex/max_likelihood.py
# Mirror ascent/descent for estimating maximum-likelihood transition probability matrices.


import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from time import sleep

import torch
from deeptime.markov.msm import MaximumLikelihoodMSM
from tqdm.auto import tqdm

from .project_feasible import *
from .utils.core import *
from .utils.opt import *
from .utils.stats import kl_divergence, tv_distance

### ESTIMATOR CLASSES ###


class DeeptimeReversibleU:
    """Wrapper for estimating reversible transition matrices with stationary constraint using Deeptime.

    Parameters
    ----------
    C : (n, n) torch.Tensor
        Observed transition counts with rows as destinations and columns as sources.
    P : (n,) torch.Tensor
        Required stationary distribution.
    tol : float, optional
        Convergence tolerance.
        We use a stricter default as this seems necessary for agreement with mirror descent.
    """

    def __init__(self, C: torch.Tensor, P: torch.Tensor, tol: float = 1e-24) -> None:
        check_nonnegative_square_matrix(C, name="C")
        if C.sum() == 0:
            raise ValueError("Counts cannot total 0.")
        self.P = normalize_probability_vector(P)
        self.C = C
        self.reversible = False
        self.tol = float(tol)

    @torch.no_grad()
    def fit(
        self, max_iters: int = 1000000, sparse: bool = False
    ) -> tuple[torch.Tensor, dict[str, str]]:
        """Fit maximum-likelihood transition matrix using Deeptime.

        Parameters
        ----------
        max_iters : int, optional
            Maximum number of fixed-point iterations.
        sparse : bool, optional
            Whether to use sparse matrix algebra.

        Returns
        -------
        U : torch.Tensor
            Estimated transition matrix.
        info : dict
            {'method': 'deeptime'}
        """
        msm = MaximumLikelihoodMSM(
            reversible=True,
            stationary_distribution_constraint=self.P.cpu().numpy(),
            sparse=sparse,
            allow_disconnected=False,
            maxiter=max_iters,
            maxerr=self.tol,
            connectivity_threshold=0,
            transition_matrix_tolerance=self.tol,
            lagtime=None,
            use_lcc=False,
        )

        try:
            msm = msm.fit(self.C.T.cpu().numpy()).fetch_model()
            U = torch.Tensor(msm.transition_matrix).T
        except (RuntimeError, TypeError, ValueError):
            raise RuntimeError(
                "Deeptime failed. Try increasing max_iters or setting sparse to False."
            )

        info = {"method": "deeptime"}
        return U, info


class MirrorDescentStochasticMatrix[ProjectorT: FluxIProjector](ABC):
    """Base class for estimating transition-probability matrices using mirror descent.

    Parameters
    ----------
    C : (n, n) torch.Tensor
        Observed transition counts with rows as destinations and columns as sources.
    P : (n,) torch.Tensor
        Required stationary distribution.
    tol : float, optional
        Convergence tolerance.
    min_entry: float, optional
        Smallest value to treat as positive.
    reversible : bool, optional
        Whether to require reversibility.
    """

    def __init__(
        self,
        C: torch.Tensor,
        P: torch.Tensor,
        tol: float = 1e-12,
        min_entry: float = 1e-24,
        reversible: bool = False,
    ):
        check_nonnegative_square_matrix(C, name="C")
        if C.sum() == 0:
            raise ValueError("Counts cannot total 0.")
        self.C = C
        self.tol = tol
        if min_entry <= 0.0:
            raise ValueError(f"min_entry must be positive, got {min_entry}.")
        self.min_entry = float(min_entry)
        self.n_states = C.shape[0]
        self.P = normalize_probability_vector(P)
        self.reversible = reversible

    @torch.no_grad()
    def _get_weighted_counts(self) -> torch.Tensor:
        """Get counts rescaled to average 1.0.
        Improves numerical stability of mirror ascent.
        """
        Cw = as_float64(self.C).clone()
        if self.reversible:
            Cw = Cw + Cw.T
        return Cw / Cw.sum()

    @abstractmethod
    def _initialize_estimates(self, projector: ProjectorT) -> dict:
        """Initialize estimates."""
        raise NotImplementedError

    @abstractmethod
    def _get_gradients(self, estimates: dict) -> torch.Tensor:
        """Estimate gradients of objectives with regard to parameters."""
        raise NotImplementedError

    @abstractmethod
    def _get_objectives(self, estimates: dict) -> dict[str, float]:
        """Evaluate objective-function values."""
        raise NotImplementedError

    @abstractmethod
    def _propose_estimates(
        self,
        estimates: dict,
        grad: torch.Tensor,
        eta_try: float,
        grad_clip: float,
        projector: ProjectorT,
    ) -> dict:
        """Propose updated estimates."""
        raise NotImplementedError

    @abstractmethod
    def _accept_proposal(
        self,
        grad: torch.Tensor,
        armijo: float,
        estimates_before: dict,
        estimates_try: dict,
        objectives_before: dict,
        objectives_try: dict,
    ) -> bool:
        """Decide whether to accept proposal.

        Returns
        -------
        accepted : bool
            Whether to accept the proposed estimates.
        """
        raise NotImplementedError

    @abstractmethod
    def _accept_convergence(
        self,
        estimates_before: dict,
        estimates: dict,
        objectives_before: dict,
        objectives: dict,
    ) -> tuple[bool, float, float]:
        """Decide whether optimization has converged."""
        raise NotImplementedError

    @torch.no_grad()
    def _get_postfix(
        self,
        eta_try: float,
        delta_est: float,
        delta_obj: float,
        estimates: dict,
        objectives: dict,
    ) -> dict[str, float | int]:
        """Get tqdm postfix."""
        return {
            "eta_try": eta_try,
            "delta_est": delta_est,
            "delta_obj": delta_obj,
            "obj": objectives["obj"],
            "proj_iters": estimates["proj_info"]["iters"],
        }

    @torch.no_grad()
    def _fit(
        self,
        projector: ProjectorT,
        eta: float = 1.0,
        grad_clip: float = 1e3,
        line_search: bool = True,
        max_iters: int = 10000,
        max_iters_ls: int = 50,
        ls_update: float = 0.5,
        armijo: float = 1e-9,
        verbose: bool = True,
        log_every: int = 1,
        postfix_callback: Callable[[dict], None] | None = None,
    ) -> tuple[dict, dict, dict]:
        """Fit maximum-likelihood estimate using mirror ascent with line search.

        Parameters
        ----------
        eta : float, optional
            Mirror-ascent base learning rate.
        grad_clip : float, optional
            Gradient-clipping threshold.
        line_search : bool, optional
            Whether to line search likelihood maximization.
        max_iters : int, optional
            Maximum mirror-ascent iterations.
        max_iters_ls : int, optional
            Maximum line-search iterations per mirror-ascent iteration.
            Ignored if line_search is False.
        ls_update : float, optional
            Factor by which to shrink eta during line search.
            Ignored if line_search is False.
        armijo : float, optional
            Armijo line-search improvement coefficient.
            Ignored if line-search is False.
        verbose : bool, optional
            Whether to display mirror-ascent progress bar.
        log_every : int, optional
            Interval at which to update progress-bar metrics in mirror-ascent iterations.
        postfix_callback : Callable[[dict], None] | None, optional
            Callback for per-iteration progress metrics.
            If provided, receives the same postfix dictionary used by the fit-level progress bar.

        Returns
        -------
        estimates : dict
            Estimated model parameters.
        objectives : dict
            Objective-function values.
        info : dict
            Fit metadata.
        """
        estimates = self._initialize_estimates(projector)
        objectives = self._get_objectives(estimates)

        if max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {max_iters}")
        iter_range = tqdm(
            range(max_iters), desc="MLE iterations", leave=True, disable=not verbose
        )
        if verbose and tqdm is not None:
            iter_range = tqdm(iter_range, desc="MLE iterations", leave=True)

        if max_iters_ls < 1:
            raise ValueError(f"max_iters_ls must be at least 1, got {max_iters_ls}")
        if ls_update <= 0.0 or ls_update > 1.0:
            raise ValueError(f"ls_update must be in (0, 1], got {ls_update:.2e}")
        eta_shrink = float(ls_update)
        converged = False
        ls_failed = False
        iters = 0

        for it in iter_range:
            iters += 1
            eta_try = float(eta)
            estimates_before = dict(estimates)
            objectives_before = dict(objectives)
            grad = self._get_gradients(estimates_before)
            accepted = False
            estimates_try = None
            objectives_try = None

            for _ in range(max_iters_ls):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimates_try = self._propose_estimates(
                        estimates_before, grad, eta_try, grad_clip, projector
                    )
                    objectives_try = self._get_objectives(estimates_try)
                if line_search:
                    accepted = self._accept_proposal(
                        grad,
                        armijo,
                        estimates_before,
                        estimates_try,
                        objectives_before,
                        objectives_try,
                    )
                else:
                    accepted = True

                if accepted:  # if line search is accepted or if line_search is False
                    estimates = estimates_try
                    objectives = objectives_try
                    break  # and break line-search loop
                eta_try = (
                    eta_try * eta_shrink
                )  # if line-search iter is not accepted, eta decreases for line-search next iter

            # emit any warnings from final line-search iteration
            for w in list(caught):
                warnings.warn(str(w.message), category=w.category, stacklevel=2)

            # always check for convergence upon exiting inner loop
            assert estimates_try is not None
            assert objectives_try is not None
            converged, delta_est, delta_obj = self._accept_convergence(
                estimates_before,
                estimates_try,  # latest attempt, accepted or not
                objectives_before,
                objectives_try,  # latest attempt, accepted or not
            )  # check for convergence

            if it % log_every == 0:
                postfix = self._get_postfix(
                    eta_try, delta_est, delta_obj, estimates, objectives
                )
                if verbose and tqdm is not None:
                    iter_range.set_postfix(postfix)
                if postfix_callback is not None:
                    postfix_callback(postfix)

            if converged:
                break  # break outer loop
            elif not accepted and not converged:
                ls_failed = True
                break  # break outer loop

        info = {
            "converged": converged,
            "iters": int(iters),
            "eta_final": float(eta_try),
            "ls_failed": ls_failed,
        }
        if not info["converged"]:
            if info["ls_failed"]:
                warnings.warn(
                    f"Optimization terminated after outer iteration exceeded {max_iters_ls} line searches: eta_try = {eta_try:.2e}."
                )
            else:
                warnings.warn(
                    f"Convergence failed: objective increase or iterate change in last iteration greater than tolerance {self.tol}."
                )

        return estimates, objectives, info


class MirrorDescentU(MirrorDescentStochasticMatrix):
    """Maximum-likelihood estimator of transition matrix with given limiting distribution.

    Parameters
    ----------
    C : (n, n) torch.Tensor
        Observed transition counts with rows as destinations and columns as sources.
    P : (n,) torch.Tensor
        Required stationary distribution.
    tol : float, optional
        Numerical tolerance.
    min_entry: float, optional
        Smallest value to treat as positive.
    reversible : bool, optional
        Whether to require reversibility.

    Method
    ------
    fit(**kwargs) -> tuple[torch.Tensor, dict]
        Fit maximum-likelihood estimate with specified optimization configruations.
    """

    def __init__(
        self,
        C: torch.Tensor,
        P: torch.Tensor,
        tol: float = 1e-12,
        min_entry: float = 1e-24,
        reversible: bool = False,
    ):
        super().__init__(C, P, tol=tol, min_entry=min_entry, reversible=reversible)

    @torch.no_grad()
    def _initialize_estimates(
        self, projector: SinkhornKnoppScaler | KnightRuizUcarScaler
    ) -> dict:
        """Initialize estimates."""
        Cw = super()._get_weighted_counts()
        C0 = as_float64(self.C).clone()
        if self.reversible:
            C0 = C0 + C0.T
        U0 = column_normalize(C0, fallback_col=self.P)

        F0 = torch.clamp(U0 * self.P[None, :], min=self.min_entry)
        if self.reversible:
            F0 = symmetrize_matrix(F0, geom=True)
        F0, proj_info = projector.project(F0, self.P)
        return {"F": F0, "Cw": Cw, "proj_info": proj_info}

    @torch.no_grad()
    def _get_gradients(self, estimates: dict) -> torch.Tensor:
        """Estimate gradient from fluxes and weighted counts."""
        grad = estimates["Cw"] / torch.clamp(estimates["F"], min=self.min_entry)
        grad = grad - torch.mean(grad)  # subtract average for numerical stability
        return grad

    @torch.no_grad()
    def _get_objectives(self, estimates: dict) -> dict[str, float]:
        """Evaluate objective-function values."""
        return {
            "obj": float(
                get_log_likelihood(
                    estimates["Cw"], estimates["F"], min_entry=self.min_entry
                )
                .detach()
                .cpu()
                .item()
            )
        }

    @torch.no_grad()
    def _propose_estimates(
        self,
        estimates: dict,
        grad: torch.Tensor,
        eta_try: float,
        grad_clip: float,
        projector: SinkhornKnoppScaler | KnightRuizUcarScaler,
    ) -> dict:
        """Propose updated estimates, generally outside of feasible set."""
        grad_max_amp = grad.abs().max()
        if grad_max_amp > grad_clip:
            grad = grad * grad_clip / grad_max_amp
        exp_try = torch.exp(eta_try * grad)

        F_try = torch.clamp(estimates["F"], min=self.min_entry) * exp_try
        if self.reversible:
            F_try = torch.clamp(F_try, min=self.min_entry)
            F_try = symmetrize_matrix(F_try, geom=True)

        F_try, proj_info_try = projector.project(F_try, self.P)
        return {"F": F_try, "Cw": estimates["Cw"], "proj_info": proj_info_try}

    @torch.no_grad()
    def _accept_proposal(
        self,
        grad: torch.Tensor,
        armijo: float,
        estimates_before: dict,
        estimates_try: dict,
        objectives_before: dict,
        objectives_try: dict,
    ) -> bool:
        """Decide whether to accept proposal using Armijo line search.

        Returns
        -------
        accepted : bool
            Whether to accept the proposed estimates.
        """
        if not estimates_try["proj_info"]["converged"]:
            return False  # always reject when projection fails
        inprod = torch.sum(grad * (estimates_try["F"] - estimates_before["F"]))
        accept = objectives_try["obj"] >= objectives_before["obj"] + armijo * inprod
        return bool(accept)

    @torch.no_grad()
    def _accept_convergence(
        self,
        estimates_before: dict,
        estimates: dict,
        objectives_before: dict,
        objectives: dict,
    ) -> tuple[bool, float, float]:
        """Decide whether optimization has converged."""
        delta_est = float(
            torch.abs(estimates["F"] - estimates_before["F"])
            .max()
            .detach()
            .cpu()
            .item()
        )
        delta_obj = objectives["obj"] - objectives_before["obj"]
        return (
            delta_est <= self.tol
            and abs(delta_obj) <= self.tol * max(1.0, abs(objectives_before["obj"])),
            delta_est,
            delta_obj,
        )

    @torch.no_grad()
    def fit(
        self,
        eta: float = 1.0,
        grad_clip: float = 1e3,
        line_search: bool = True,
        max_iters: int = 10000,
        max_iters_proj: int = 10000,
        max_iters_ls: int = 50,
        ls_update: float = 0.5,
        armijo: float = 1e-9,
        verbose: bool = True,
        log_every: int = 1,
        postfix_callback: Callable[[dict], None] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Fit maximum-likelihood estimate with specified optimization configurations.

        Parameters
        ----------
        max_iters_proj : int, optional
            Maximum Sinkhorn iterations per mirror-ascent iteration.
        See TPMMirrorOptimizer documentation for details on other parameters.

        Returns
        -------
        U : torch.Tensor
            Maximum likelihood of estimate of transition matrix.
        info : dict
            Fit metadata.
        """
        if self.reversible:
            projector = KnightRuizUcarScaler(
                self.min_entry, max_iters_proj, self.tol, progress=False
            )
        else:
            projector = SinkhornKnoppScaler(
                self.min_entry, max_iters_proj, self.tol, progress=False
            )

        estimates, objectives, info = super()._fit(
            eta=eta,
            grad_clip=grad_clip,
            line_search=line_search,
            max_iters=max_iters,
            max_iters_ls=max_iters_ls,
            ls_update=ls_update,
            armijo=armijo,
            verbose=verbose,
            log_every=log_every,
            projector=projector,
            postfix_callback=postfix_callback,
        )

        F = estimates["F"]
        proj_info = estimates["proj_info"]
        U = F / self.P[None, :]

        info["col_sum_err"] = (
            (U.sum(dim=0) - 1.0).abs().sum().detach().cpu().item()
        )  # final deviation from stochasticity
        info["lim_dist_err"] = tv_distance(
            U @ self.P, self.P
        )  # final error in limiting distribution
        info["llh_final"] = float(
            get_log_likelihood(self.C, U, min_entry=self.min_entry)
            .detach()
            .cpu()
            .item()
        )  # final log-likelihood
        info["obj_final"] = float(objectives["obj"])  # final objective-function value
        info["row_proj_err"] = float(
            proj_info["row_err"]
        )  # final error in flux row marginals

        kld_final = kl_divergence(
            column_normalize(self.C, fallback_col=self.P),
            U,
            eps=self.min_entry,
            reduce_dim=0,
        )
        assert isinstance(kld_final, torch.Tensor)
        info["kld_final"] = float(
            kld_final.sum().detach().cpu().item()
        )  # final KL divergence from naive transfer matrix

        info["lim_sigma"] = float(
            torch.nan_to_num(F * torch.log(F / F.T), nan=0.0, posinf=0.0, neginf=0.0)
            .sum()
            .detach()
            .cpu()
            .item()
        )  # final estimated stationary entropy production, also ligma

        if not self.reversible:
            info["col_proj_err"] = float(
                proj_info["col_err"]
            )  # final error in flux column marginals

        return U, info


class MirrorDescentG(MirrorDescentStochasticMatrix):
    """Maximum-likelihood estimator of TCL-GME-DT propagator.
    Always enforces limiting distribution and previous-lag transition matrix;
    reversible case enforces detailed balance and commutation with previous-lag transition matrix.

    Parameters
    ----------
    C : (n, n) torch.Tensor
        Observed transition counts with rows as destinations and columns as sources.
    U_prev : (n, n) torch.Tensor
        Column-stochastic previous-lag transition matrix.
    P : (n,) torch.Tensor
        Required stationary distribution.
    tol : float, optional
        Convergence tolerance.
    min_entry: float, optional
        Smallest value to treat as positive.
    reversible : bool, optional
        Whether to require reversibility

    Method
    ------
    fit(**kwargs) -> tuple[torch.Tensor, dict]
        Fit maximum-likelihood estimate with specified optimization configurations.
    """

    def __init__(
        self,
        C: torch.Tensor,
        U_prev: torch.Tensor,
        P: torch.Tensor,
        tol: float = 1e-12,
        min_entry: float = 1e-24,
        reversible: bool = False,
    ):
        if C.shape != U_prev.shape:
            raise ValueError(
                f"C and U_prev must have the same shape, got {tuple(C.shape)} and {tuple(U_prev.shape)}."
            )
        self.U_prev = as_float64(U_prev)
        self._G0 = None  # initializing as None for safety
        super().__init__(C, P, tol=tol, min_entry=min_entry, reversible=reversible)

    @torch.no_grad()
    def _initialize_estimates(
        self, projector: SinkhornKnoppScaler | ReversibleCommutingIProjector
    ) -> dict:
        """Initialize estimates."""
        try:
            check_stationary_column_stochastic_matrix(
                self.U_prev, self.P, db=self.reversible, name="U_prev"
            )
        except ValueError as err:
            warnings.warn(
                f"Attempting matrix-scaling fix, but U_prev failed check: {err}",
                stacklevel=2,
            )

        UD_prev = self.U_prev * self.P[None, :]
        if isinstance(projector, ReversibleCommutingIProjector):
            UD_prev, _ = projector._symm.project(UD_prev, self.P)
            projector._symm._reset()
        else:
            UD_prev, _ = projector.project(UD_prev, self.P)
            projector._reset()
        self.U_prev = UD_prev / self.P[None, :]
        self._U_prev_full_rank = (
            torch.linalg.matrix_rank(self.U_prev, atol=self.tol) == self.n_states
        )

        Cw = self._get_weighted_counts()

        if self._G0 is None:
            C0 = as_float64(self.C).clone()
            if self.reversible:
                C0 = C0 + C0.T

            U0 = column_normalize(C0, fallback_col=self.P)
            if self._U_prev_full_rank:
                self._G0 = torch.linalg.lstsq(self.U_prev.T, U0.T).solution.T
            else:
                self._G0 = torch.linalg.lstsq(
                    self.U_prev.cpu().T, U0.cpu().T, driver="gelss"
                ).solution.T
                self._G0 = self._G0.to(self.U_prev.device)

        GD0 = torch.clamp(self._G0 * self.P[None, :], min=self.min_entry)
        if isinstance(projector, ReversibleCommutingIProjector):
            GD0, proj_info = projector.project(GD0, self.U_prev, self.P)
        else:
            GD0, proj_info = projector.project(GD0, self.P)
        self._G0 = GD0 / self.P[None, :]

        return {"G": self._G0, "GD": GD0, "Cw": Cw, "proj_info": proj_info}

    @torch.no_grad()
    def _get_gradients(self, estimates: dict) -> torch.Tensor:
        """Estimate gradient from current parameters."""
        # we need the gradient of loglikelihood(C; UD = G UD_prev) wrt GD
        UD = torch.clamp(
            (estimates["G"] @ self.U_prev) * self.P[None, :], min=self.min_entry
        )
        UD_prev = self.U_prev * self.P[None, :]
        grad = (estimates["Cw"] / UD) @ UD_prev.T  # gradient wrt G
        grad = grad / self.P[None, :]  # gradient wrt GD
        return grad - grad.mean()

    @torch.no_grad()
    def _get_objectives(self, estimates: dict) -> dict[str, float]:
        """Evaluate objective-function values."""
        return {
            "obj": float(
                get_log_likelihood(
                    estimates["Cw"],
                    (estimates["G"] @ self.U_prev) * self.P[None, :],
                    min_entry=self.min_entry,
                )
                .detach()
                .cpu()
                .item()
            )
        }

    @torch.no_grad()
    def _propose_estimates(
        self,
        estimates: dict,
        grad: torch.Tensor,
        eta_try: float,
        grad_clip: float,
        projector: SinkhornKnoppScaler | ReversibleCommutingIProjector,
    ) -> dict:
        """Propose updated estimates, generally outside of feasible set."""
        grad_max_amp = grad.abs().max()
        if grad_max_amp > grad_clip:
            grad = grad * grad_clip / grad_max_amp
        exp_try = torch.exp(eta_try * grad)

        GD_try = torch.clamp(estimates["GD"], min=self.min_entry) * exp_try
        if isinstance(projector, ReversibleCommutingIProjector):
            GD_try, proj_info_try = projector.project(GD_try, self.U_prev, self.P)
        else:
            GD_try, proj_info_try = projector.project(GD_try, self.P)

        G_try = GD_try / self.P[None, :]
        return {
            "G": G_try,
            "GD": GD_try,
            "Cw": estimates["Cw"],
            "proj_info": proj_info_try,
        }

    @torch.no_grad()
    def _accept_proposal(
        self,
        grad: torch.Tensor,
        armijo: float,
        estimates_before: dict,
        estimates_try: dict,
        objectives_before: dict,
        objectives_try: dict,
    ) -> bool:
        """Decide whether to accept proposal using Armijo line search.

        Returns
        -------
        accepted : bool
            Whether to accept the proposed estimates.
        """
        if not estimates_try["proj_info"]["converged"]:
            return False  # always reject when projection fails
        inprod = torch.sum(grad * (estimates_try["GD"] - estimates_before["GD"]))
        accept = objectives_try["obj"] >= objectives_before["obj"] + armijo * inprod
        return bool(accept)

    @torch.no_grad()
    def _accept_convergence(
        self,
        estimates_before: dict,
        estimates: dict,
        objectives_before: dict,
        objectives: dict,
    ) -> tuple[bool, float, float]:
        """Decide whether optimization has converged."""
        delta_est = float(
            torch.abs(estimates["GD"] - estimates_before["GD"])
            .max()
            .detach()
            .cpu()
            .item()
        )
        delta_obj = objectives["obj"] - objectives_before["obj"]
        if self._U_prev_full_rank:  # optimizer is unique
            return (
                delta_est <= self.tol
                and abs(delta_obj)
                <= self.tol * max(1.0, abs(objectives_before["obj"])),
                delta_est,
                delta_obj,
            )
        return (
            delta_obj
            <= self.tol
            * max(  # optimizer is nonunique
                1.0, abs(objectives_before["obj"])
            ),
            delta_est,
            delta_obj,
        )

    @torch.no_grad()
    def fit(
        self,
        eta: float = 1.0,
        grad_clip: float = 1e3,
        line_search: bool = False,
        max_iters: int = 10000,
        max_iters_proj: int = 10000,
        max_iters_ls: int = 50,
        max_iters_proj_symm: int = 2500,
        max_iters_proj_comm: int = 250,
        max_iters_proj_ls: int = 25,
        ls_update: float = 0.5,
        armijo: float = 1e-9,
        armijo_proj: float = 1e-4,
        verbose: bool = True,
        log_every: int = 1,
        G0: torch.Tensor | None = None,
        postfix_callback: Callable[[dict], None] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Fit maximum-likelihood estimate with specified optimization configurations.

        Parameters
        ----------
        G0 : torch.Tensor | None, optional
            Initial estimate of G. If not passed, initializes by pseudoinversion.
        max_iters_proj : int, optional
            Maximum projection outer iterations per mirror step.
        max_iters_ls : int, optional
            Maximum line-search iterations per mirror step.
        max_iters_proj_symm : int, optional
            Maximum Knight-Ruiz-Ucar iterations per projection outer iteration.
            Ignored if not reversible.
        max_iters_proj_comm : int, optional
            Maximum commutation-projector Newton steps per projection outer iteration.
            Ignored if not reversible.
        max_iters_proj_ls : int, optional
            Maximum line-search iterations per commutation-projector Newton step.
            Ignored if not reversible.
        See MirrorDescentStochasticMatrix documentation for details on other parameters.

        Returns
        -------
        G : torch.Tensor
            Nested maximum-likelihood estimate of TCL propagator.
        info : dict
            Fit metadata.
        """
        if G0 is not None:
            if type(G0) != torch.Tensor:
                raise ValueError(
                    f"Expected G0 to be type torch.Tensor, got type {type(G0)}"
                )
            if (
                G0.ndim != 2
                or G0.shape[0] != G0.shape[1]
                or G0.shape[0] != self.n_states
            ):
                raise ValueError(
                    f"Expected G0 shape ({self.n_states}, {self.n_states}), got shape {tuple(G0.shape)}"
                )
        self._G0 = G0

        if self.reversible:
            projector = ReversibleCommutingIProjector(
                self.min_entry,
                max_iters_proj,
                self.tol,
                max_iters_symm=max_iters_proj_symm,
                max_iters_comm=max_iters_proj_comm,
                max_iters_ls=max_iters_proj_ls,
                armijo=armijo_proj,
                progress=False,
            )
        else:
            projector = SinkhornKnoppScaler(
                self.min_entry, max_iters_proj, self.tol, progress=False
            )

        estimates, objectives, info = super()._fit(
            eta=eta,
            grad_clip=grad_clip,
            line_search=line_search,
            max_iters=max_iters,
            max_iters_ls=max_iters_ls,
            ls_update=ls_update,
            armijo=armijo,
            verbose=verbose,
            log_every=log_every,
            postfix_callback=postfix_callback,
            projector=projector,
        )

        G = estimates["G"]
        proj_info = estimates["proj_info"]
        U = G @ self.U_prev
        GD = estimates["GD"]
        UD = U * self.P[None, :]

        info["col_sum_err"] = (
            (U.sum(dim=0) - 1.0).abs().sum().detach().cpu().item()
        )  # final deviation from stochasticity
        info["lim_dist_err"] = tv_distance(
            U @ self.P, self.P
        )  # final error in limiting distribution
        info["llh_final"] = float(
            get_log_likelihood(self.C, U, min_entry=self.min_entry)
            .detach()
            .cpu()
            .item()
        )  # final log-likelihood
        info["obj_final"] = float(objectives["obj"])  # final objective-function value
        info["row_proj_err"] = float(
            proj_info["row_err"]
        )  # final error in flux row marginals

        assert self._G0 is not None
        kld_final = kl_divergence(self._G0, G, eps=self.min_entry, reduce_dim=0)
        assert isinstance(kld_final, torch.Tensor)
        info["kld_final"] = float(
            kld_final.sum().detach().cpu().item()
        )  # final KL divergence from naive transition matrix

        info["lim_sigma_U"] = float(
            torch.nan_to_num(UD * torch.log(UD / UD.T), nan=0.0, posinf=0.0, neginf=0.0)
            .sum()
            .detach()
            .cpu()
            .item()
        )  # final estimated stationary EPR of U
        info["lim_sigma_G"] = float(
            torch.nan_to_num(GD * torch.log(GD / GD.T), nan=0.0, posinf=0.0, neginf=0.0)
            .sum()
            .detach()
            .cpu()
            .item()
        )  # final estimated stationary EPR of G
        if self.reversible:
            info["commutator_norm"] = float(
                torch.linalg.matrix_norm(
                    self.U_prev @ G - G @ self.U_prev, ord=torch.inf
                )
                .detach()
                .cpu()
                .item()
            )
        else:
            info["col_proj_err"] = float(
                proj_info["col_err"]
            )  # final error in flux column marginals

        return G, info


### CONVENIENCE WRAPPERS ###


def get_reversible_Us_deeptime(
    Cs: torch.Tensor,
    lim_dist: torch.Tensor,
    tol: float = 1e-24,
    max_iters: int = 1000000,
    sparse: bool = False,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict[int, dict[str, str]]]:
    """Estimate maximum-likelihood reversible transition matrices from a stack of count matrices.
    Uses Prinz-Trendelkamp-Schroer estimator implemented in Deeptime instead of mirror descent.

    Parameters
    ----------
    Cs : (lags, n, n) torch.Tensor
        Stack of count matrices (row is destination and column is source);
        first count matrix must be diagonal and correspond to zero lag.
    lim_dist : (n,) torch.Tensor
        Required stationary distribution.
    tol : float, optional
        Convergence tolerance.
        We use a stricter default as this seems necessary for agreement with mirror descent.
    max_iters : int, optional
        Maximum number of fixed-point iterations.
    sparse : bool, optional
        Whether to use sparse matrix algebra.
    verbose : bool, optional
        Whether to display progress bar.

    Returns
    -------
    Us : (lags, n, n) torch.Tensor
        Maximum-likelihood reversible transition matrices (column-stochastic).
    metrics : dict
        Each key is an integer lag and each value is {'method': 'deeptime'}.
    """
    check_count_matrices(Cs)
    lim_dist = lim_dist.to(torch.float64)
    lim_dist = lim_dist / lim_dist.sum()

    Us = torch.zeros(Cs.shape, dtype=torch.float64, device=Cs.device)
    Us[0] = torch.eye(Us.shape[1], dtype=torch.float64, device=Us.device)
    metrics = {}

    lag_iter = tqdm(
        range(1, Cs.shape[0]), desc="Lags", leave=False, disable=not verbose
    )
    for lag in lag_iter:
        estimator = DeeptimeReversibleU(Cs[lag], lim_dist, tol=tol)
        U, info = estimator.fit(max_iters=max_iters, sparse=sparse)
        Us[lag] = U
        metrics[lag] = info
    return Us, metrics


def get_Us_mle(
    Cs: torch.Tensor,
    lim_dist: torch.Tensor,
    tol: float = 1e-12,
    min_entry: float = 1e-24,
    eta: float = 1.0,
    grad_clip: float = 1e3,
    line_search: bool = True,
    max_iters: int = 10000,
    max_iters_proj: int = 10000,
    max_iters_ls: int = 50,
    ls_update: float = 0.5,
    armijo: float = 1e-9,
    reversible: bool = False,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Estimate maximum-likelihood transition matrices from stack of count matrices.

    Parameters
    ----------
    Cs : (lags, n, n) torch.Tensor
        Stack of count matrices (row is destination and column is source);
        first count matrix must be diagonal and correspond to zero lag.
    lim_dist : (n,) torch.Tensor
        Required stationary distribution.
    reversible : bool, optional
        Whether transition matrices must be reversible.
    verbose : bool, optional
        Whether to display progress bar.

    Returns
    -------
    Us : (lags, n, n) torch.Tensor
        Maximum-likelihood transition matrices (column-stochastic).
    metrics : dict
        Fit metadata.
    """
    check_count_matrices(Cs)
    lim_dist = lim_dist.to(torch.float64)
    lim_dist = lim_dist / lim_dist.sum()

    Us = torch.zeros(Cs.shape, dtype=torch.float64, device=Cs.device)
    Us[0] = torch.eye(Us.shape[1], dtype=torch.float64, device=Us.device)
    metrics = {}

    lag_iter = tqdm(
        range(1, Cs.shape[0]), desc="Lags", leave=False, disable=not verbose
    )
    for lag in lag_iter:

        def _update_outer_postfix(postfix: dict, lag: int = lag) -> None:
            lag_iter.set_postfix({"lag": lag, **postfix})

        estimator = MirrorDescentU(
            Cs[lag], lim_dist, tol=tol, min_entry=min_entry, reversible=reversible
        )
        U, info = estimator.fit(
            eta=eta,
            grad_clip=grad_clip,
            line_search=line_search,
            max_iters=max_iters,
            max_iters_proj=max_iters_proj,
            max_iters_ls=max_iters_ls,
            ls_update=ls_update,
            armijo=armijo,
            verbose=False,
            postfix_callback=_update_outer_postfix,
        )
        Us[lag] = U
        metrics[lag] = info
    return Us, metrics


def get_Gs_mle(
    Cs: torch.Tensor,
    lim_dist: torch.Tensor,
    tol: float = 1e-12,
    min_entry: float = 1e-24,
    eta: float = 1.0,
    grad_clip: float = 1e3,
    line_search: bool = False,
    max_iters: int = 10000,
    max_iters_proj: int = 10000,
    max_iters_ls: int = 50,
    max_iters_proj_symm: int = 2500,
    max_iters_proj_comm: int = 250,
    max_iters_proj_ls: int = 25,
    ls_update: float = 0.5,
    armijo: float = 1e-9,
    armijo_proj: float = 1e-4,
    reversible: bool = False,
    precomputed: tuple[torch.Tensor, dict] | None = None,
    verbose: bool = True,
    postfix_callback: Callable[[dict], None] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Estimate maximum-likelihood TCL GME propagators from stack of count matrices.

    Parameters
    ----------
    Cs : (lags, n, n) torch.Tensor
        Stack of count matrices (row is destination and column is source);
        first count matrix must be diagonal and correspond to zero lag.
    lim_dist : (n,) torch.Tensor
        Required stationary distribution.
    reversible : bool, optional
        Whether implied propagators must be consistent with reversibility.
    precomputed : tuple[torch.Tensor, dict], optional
        TCL GME propagators and metadata computed for early lags.
        Must have lower zeroth dimension than Cs.
        This is used for checkpointing as the estimators are nested.
    verbose : bool, optional
        Whether to display progress bar.
    postfix_callback : Callable[[dict], None] | None, optional
        Callback for per-iteration progress metrics.
        If provided, receives the same lag-prefixed postfix dictionary used by the lag progress bar.

    Returns
    -------
    Gs : (lags, n, n) torch.Tensor
        Maximum-likelihood TCL GME propagators (column-stochastic).
    metrics : dict
        Fit metadata.
    """
    check_count_matrices(Cs)
    lim_dist = lim_dist.to(torch.float64)
    lim_dist = lim_dist / lim_dist.sum()

    Gs = torch.zeros(Cs.shape, dtype=torch.float64, device=Cs.device)
    Us = torch.zeros(Cs.shape, dtype=torch.float64, device=Cs.device)
    Gs[0] = torch.eye(Us.shape[1], dtype=torch.float64, device=Us.device)
    Us[0] = torch.eye(Us.shape[1], dtype=torch.float64, device=Us.device)
    metrics = {}

    start_lag = 1
    if precomputed is not None:
        Gs_precomputed, info_precomputed = precomputed
        if Gs_precomputed.shape[1:] != Cs.shape[1:]:
            raise ValueError(
                "Precomputed propagators must have same 1st and 2nd dimension as Cs."
            )
        start_lag = min(Gs_precomputed.shape[0], Cs.shape[0])
        if (reversible and Gs_precomputed.shape[0] > 2) and not torch.allclose(
            Gs_precomputed[1], Gs_precomputed[2], atol=tol
        ):
            raise ValueError(
                "Gs_precomputed[1] must equal Gs_precomputed[2] in reversible case."
            )

    lag_iter = tqdm(
        range(1, Cs.shape[0]), desc="Lags", leave=False, disable=not verbose
    )
    for lag in lag_iter:
        if lag < start_lag:
            sleep(0.1)  # tqdm bugs out if this is too fast
            Gs[lag] = Gs_precomputed[lag]
            Us[lag] = Gs[lag] @ Us[lag - 1]
            metrics[lag] = (
                info_precomputed[lag]
                if lag in info_precomputed
                else info_precomputed[str(lag)]
            )
        else:
            if reversible and lag == 2:
                G = Gs[1]
                info = metrics[1]
            else:

                def _update_outer_postfix(postfix: dict, lag: int = lag) -> None:
                    progress = {"lag": lag, **postfix}
                    lag_iter.set_postfix(progress)
                    if postfix_callback is not None:
                        postfix_callback(progress)

                estimator = MirrorDescentG(
                    Cs[lag],
                    Us[lag - 1],
                    lim_dist,
                    tol=tol,
                    min_entry=min_entry,
                    reversible=reversible,
                )
                G, info = estimator.fit(
                    eta=eta,
                    grad_clip=grad_clip,
                    line_search=line_search,
                    max_iters=max_iters,
                    max_iters_proj=max_iters_proj,
                    max_iters_ls=max_iters_ls,
                    max_iters_proj_symm=max_iters_proj_symm,
                    max_iters_proj_comm=max_iters_proj_comm,
                    max_iters_proj_ls=max_iters_proj_ls,
                    ls_update=ls_update,
                    armijo=armijo,
                    armijo_proj=armijo_proj,
                    verbose=False,
                    G0=None
                    if lag == 1
                    else Gs[
                        lag - 1
                    ],  # initialize with previous G at lags other than first
                    postfix_callback=_update_outer_postfix,
                )
            Gs[lag] = G
            Us[lag] = G @ Us[lag - 1]
            metrics[lag] = info
    return Gs, metrics
