# gmex/project_feasible.py
# Classes and functions for projecting flux matrices.


import warnings
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .utils.core import *
from .utils.opt import *
from .utils.types import (
    CommutingProjectionInfo,
    MarginalProjectionInfo,
    ReversibleCommutingProjectionInfo,
    SinkhornKnoppInfo,
)


class FluxIProjector(ABC):
    """Parent class for flux I-projectors."""

    def __init__(
        self, min_entry: float, max_iters: int, tol: float, progress: bool
    ) -> None:
        self.min_entry = float(min_entry)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.progress = bool(progress)

    @abstractmethod
    def _reset(self) -> None:
        """Forget cached projection state."""


class SinkhornKnoppScaler(FluxIProjector):
    """Sinkhorn-Knopp scaler for I-projection onto feasible subset, i.e.,
    square matrices with positive entries and specified row and column marginals.

    Parameters
    ----------
    min_entry : float
        Minimum permitted matrix entry.
    max_iters : int
        Maximum number of scaling iterations.
    tol : float
        Tolerated deviation from target marginal.
    progress : bool
        Whether to display progress bar.

    Method
    ------
    def project(
        self, K: torch.Tensor, P: torch.Tensor, check_every: int = 1
    ) -> tuple[torch.Tensor, SinkhornKnoppInfo]:
        Sinkhorn-scale matrix K to fixed row- and column- marginal P.

    Reference
    ---------
    P. Knopp and R. Sinkhorn,
    Concerning nonnegative matrices and doubly stochastic matrices,
    Pacific J. Math 21, 343 (1967).
    """

    def __init__(
        self, min_entry: float, max_iters: int, tol: float, progress: bool = False
    ) -> None:
        super().__init__(min_entry, max_iters, tol, progress)
        self._r: torch.Tensor | None = None
        self._c: torch.Tensor | None = None

    def _reset(self) -> None:
        """Forget cached scalings."""
        self._r = None
        self._c = None

    @torch.no_grad()
    def project(
        self, K: torch.Tensor, P: torch.Tensor, check_every: int = 1
    ) -> tuple[torch.Tensor, SinkhornKnoppInfo]:
        """Sinkhorn-scale matrix K to fixed row- and column- marginal P.

        Parameters
        ----------
        K : (n, n) torch.Tensor
            Matrix with nonnegative entries.
        P : (n,) torch.Tensor
            Positive target row- and column- marginal.
        check_every : int
            Frequency in iterations at which to compute residuals.

        Returns
        -------
        F : (n, n)
            K information-projected onto feasible subset with target marginals.
        info : SinkhornKnoppInfo
            Diagnostics including residuals and iterations.
        """
        check_nonnegative_square_matrix(K, name="K")
        n = K.shape[0]
        check_probability_vector(P, n)
        Pf = normalize_probability_vector(P)

        if (
            self._r is None
            or self._c is None
            or self._r.shape != (n,)
            or self._c.shape != (n,)
        ):
            self._r = torch.ones((n,), device=K.device, dtype=K.dtype)
            self._c = torch.ones((n,), device=K.device, dtype=K.dtype)
        r = self._r
        c = self._c

        Kp = rescale_sinkhorn_input(K, min_entry=self.min_entry)
        row_err = float("inf")
        col_err = float("inf")

        iter_range = range(self.max_iters)
        if self.progress and tqdm is not None:
            iter_range = tqdm(iter_range, desc="Scaling iterations", leave=False)
        for it in iter_range:  # iteratively update scalings
            r = Pf / torch.clamp(Kp @ c, min=self.min_entry)  # r <- P / (K c)
            c = Pf / torch.clamp(Kp.T @ r, min=self.min_entry)  # c <- P / (K^T r)
            if (it % check_every) == 0 or it == self.max_iters - 1:
                F = r[:, None] * Kp * c[None, :]
                # row_err = tv_distance(F.sum(dim=1), Pf)
                # col_err = tv_distance(F.sum(dim=0), Pf)
                row_err = (
                    torch.linalg.vector_norm(F.sum(dim=1) - Pf, ord=1).item() / 2.0
                )
                col_err = (
                    torch.linalg.vector_norm(F.sum(dim=0) - Pf, ord=1).item() / 2.0
                )
                if max(row_err, col_err) <= self.tol:  # break upon convergence
                    self._r, self._c = r, c  # cache scalings
                    return F, {
                        "converged": True,
                        "iters": it + 1,
                        "row_err": row_err,
                        "col_err": col_err,
                    }

        F = r[:, None] * Kp * c[None, :]
        self._r, self._c = r, c  # cache scalings
        warnings.warn(
            f"IPF not converged: row error {row_err:.2e} or column error {col_err:.2e} greater than tolerance {self.tol}. Increase max_iters."
        )
        return F, {
            "converged": False,
            "iters": self.max_iters,
            "row_err": row_err,
            "col_err": col_err,
        }


class KnightRuizUcarScaler(FluxIProjector):
    """Sinkhorn-style scaling for I-projection onto detailed-balance fluxes, i.e.,
    symmetric matrices with positive entries and specified row and column marginal.

    Parameters
    ----------
    min_entry : float
        Minimum permitted matrix entry.
    max_iters : int
        Maximum number of scaling iterations.
    tol : float
        Tolerated deviation from target marginal.
    progress : bool
        Whether to display progress bar.

    Method
    ------
    def project(
        self, K: torch.Tensor, P: torch.Tensor, check_every: int = 1
    ) -> tuple[torch.Tensor, MarginalProjectionInfo]:
        Scale matrix K to impose symmetry and fixed row and column marginal P.

    Reference
    ---------
    P.A. Knight, D. Ruiz, and B. Ucar,
    A symmetry preserving algorithm for matrix scaling,
    SIAM J. Matrix Anal. Appl. 35, 25 (2014).
    """

    def __init__(
        self, min_entry: float, max_iters: int, tol: float, progress: bool = False
    ) -> None:
        super().__init__(min_entry, max_iters, tol, progress)
        self._d: torch.Tensor | None = None

    def _reset(self) -> None:
        """Forget cached scaling."""
        self._d = None

    @torch.no_grad()
    def project(
        self, K: torch.Tensor, P: torch.Tensor, check_every: int = 1
    ) -> tuple[torch.Tensor, MarginalProjectionInfo]:
        """Scale matrix K to impose symmetry and row- and column- marginal P.

        Parameters
        ----------
        K : (n, n) torch.Tensor
            Matrix with nonnegative entries.
        P : (n,) torch.Tensor
            Positive target row- and column- marginal.
        check_every : int
            Frequency in iterations at which to compute residuals.

        Returns
        -------
        F : (n, n)
            K information-projected onto feasible subset with target marginals.
        info : MarginalProjectionInfo
            Diagnostics including residuals and iterations.
        """
        check_nonnegative_square_matrix(K, "K")
        n = K.shape[0]
        check_probability_vector(P, n)
        Pf = normalize_probability_vector(P)

        if self._d is None or self._d.shape != (n,):
            self._d = torch.ones((n,), device=K.device, dtype=K.dtype)
        d = self._d

        Ks = torch.clamp(K, min=self.min_entry)
        Ks = symmetrize_matrix(Ks, geom=True)
        Ks = rescale_sinkhorn_input(Ks, min_entry=self.min_entry)
        row_err = float("inf")

        iter_range = range(self.max_iters)
        if self.progress and tqdm is not None:
            iter_range = tqdm(iter_range, desc="Scaling iterations", leave=False)
        for it in iter_range:
            d = d * torch.sqrt(Pf / torch.clamp(d * (Ks @ d), min=self.min_entry))
            if (it % check_every) == 0 or it == self.max_iters - 1:
                F = d[:, None] * Ks * d[None, :]
                # row_err = tv_distance(F.sum(dim=1), Pf)
                row_err = (
                    torch.linalg.vector_norm(F.sum(dim=1) - Pf, ord=1).item() / 2.0
                )
                if row_err <= self.tol:
                    self._d = d  # cache scaling
                    return F, {"converged": True, "iters": it + 1, "row_err": row_err}

        F = d[:, None] * Ks * d[None, :]
        self._d = d  # cache scaling
        warnings.warn(
            f"IPF not converged: row error {row_err:.2e} greater than tolerance {self.tol}. Increase max_iters."
        )
        return F, {"converged": False, "iters": self.max_iters, "row_err": row_err}


class _CommutingFluxIProjector(FluxIProjector):
    """I-projector onto the affine flux-commuting subset.
    This projects a square matrix F onto the set of positive matrices satisfying
        U_prev F = F U_prev^T,
    where U_prev is column-stochastic.

    The projection is computed in generalized KL divergence by solving the dual
    problem associated with the affine constraints in orthonormalized form.

    Parameters
    ----------
    min_entry : float
        Minimum permitted matrix entry.
    max_iters : int
        Maximum damped-Newton iterations in the dual.
    tol : float
        Tolerated maximum absolute commutator residual.
    max_iters_ls : int, optional
        Maximum line-search iterations.
    armijo : float, optional
        Inner-product prefactor for Armijo line search.
    progress : bool, optional
        Whether to display progress bar.
    """

    def __init__(
        self,
        min_entry: float,
        max_iters: int,
        tol: float,
        max_iters_ls: int = 25,
        armijo: float = 1e-4,
        progress: bool = False,
    ) -> None:
        super().__init__(min_entry, max_iters, tol, progress)
        self.max_iters_ls = int(max_iters_ls)
        self.armijo = float(armijo)
        self._Q: torch.Tensor | None = None
        self._c: torch.Tensor | None = None
        self._rank: int | None = None
        self._U_prev: torch.Tensor | None = None
        self._dual: torch.Tensor | None = None

    def _reset(self) -> None:
        """Forget cached affine basis and dual warm start."""
        self._Q = None
        self._c = None
        self._rank = None
        self._U_prev = None
        self._dual = None

    def _cache_prepared(self, U_prev: torch.Tensor) -> bool:
        """Check if cache has been prepared."""
        Uf = as_float64(U_prev).detach().cpu()
        return (
            self._Q is not None
            and self._c is not None
            and self._rank is not None
            and self._U_prev is not None
            and torch.equal(self._U_prev, Uf)
        )

    def _prepare_cache(self, U_prev: torch.Tensor) -> None:
        """Build or reuse the cached orthonormal affine constraint system."""
        Uf = as_float64(U_prev).detach().cpu()
        if not self._cache_prepared(U_prev):
            A, b = build_commuting_constraints_flux(Uf)
            Q, c, rank = orthonormalize_affine_constraints(
                A, b, tol=self.tol, linear=True
            )

            self._Q = Q
            self._c = c
            self._rank = int(rank)
            self._U_prev = Uf.clone()
            self._dual = None

    @staticmethod
    def _get_residuals(F: torch.Tensor, U_prev: torch.Tensor) -> dict[str, float]:
        """Compute feasibility residuals for diagnostics."""
        return {
            "comm_err": float(
                torch.abs(U_prev @ F - F @ U_prev.T).max().detach().cpu().item()
            ),
            "neg_err": float(torch.clamp(-F.min(), min=0.0).detach().cpu().item()),
        }

    @torch.no_grad()
    def project(
        self, K: torch.Tensor, U_prev: torch.Tensor, check_every: int = 1
    ) -> tuple[torch.Tensor, CommutingProjectionInfo]:
        """Project K onto the flux-commuting affine subset in KL divergence.

        Parameters
        ----------
        K : (n, n) torch.Tensor
            Nonnegative input matrix.
        U_prev : (n, n) torch.Tensor
            Fixed column-stochastic matrix.
        check_every : int
            Kept for call-signature compatibility;
            unused because the inner dual solver checks convergence every iteration.

        Returns
        -------
        F_proj : (n, n) torch.Tensor
            KL projection of K onto the commuting affine subset.
        info : CommutingProjectionInfo
            Diagnostics including residuals, iterations, and affine rank.
        """
        if K.ndim != 2 or K.shape[0] != K.shape[1] or type(K) != torch.Tensor:
            raise ValueError(
                f"K must be a square tensor, got a {tuple(K.shape)} {type(K)}."
            )
        if not torch.isfinite(K).all():
            raise ValueError("K must have only finite entries.")
        if U_prev.shape != K.shape:
            raise ValueError(
                f"U_prev must have shape {tuple(K.shape)}, got {tuple(U_prev.shape)}."
            )
        if not torch.isfinite(U_prev).all():
            raise ValueError("U_prev must have only finite entries.")

        check_column_stochastic_matrix(U_prev, name="U_prev")
        Kf = as_float64(K).detach().cpu()
        Uf = as_float64(U_prev).detach().cpu()
        Kp = torch.clamp(nan_to_pos(Kf, min_entry=self.min_entry), min=self.min_entry)

        self._prepare_cache(Uf)
        assert self._Q is not None and self._c is not None and self._rank is not None

        x_proj, dual_info, dual = project_affine_kl_orthonormal(
            Kp.reshape(-1),
            self._Q,
            self._c,
            tol=self.tol,
            max_iters_proj=self.max_iters,
            max_iters_ls=self.max_iters_ls,
            armijo=self.armijo,
            min_entry=self.min_entry,
            dual0=self._dual,
        )
        if dual_info["converged"]:
            self._dual = dual
        else:
            self._dual = None

        F_proj = x_proj.reshape_as(Kp)
        residuals = self._get_residuals(F_proj, Uf)
        comm_err = residuals["comm_err"]
        neg_err = residuals["neg_err"]
        step_err = float(dual_info["step_err"])
        converged = bool(dual_info["converged"]) and max(comm_err, neg_err) <= self.tol

        info: CommutingProjectionInfo = {
            "converged": converged,
            "iters": int(dual_info["iters"]),
            "comm_err": comm_err,
            "neg_err": neg_err,
            "step_err": step_err,
            "affine_err": float(dual_info["affine_err"]),
            "constraint_rank": self._rank,
            "objective": float(
                generalized_kl_divergence(F_proj, Kp, eps=self.min_entry)
                .detach()
                .cpu()
                .item()
            ),
        }

        if not info["converged"]:
            warnings.warn(
                "KL projection onto commuting affine subset not converged: "
                f"commutator error {comm_err:.2e}, "
                f"affine residual {info['affine_err']:.2e}, or "
                f"step error {step_err:.2e} greater than tolerance {self.tol}. "
                "Increase max_iters."
            )

        out_dtype = K.dtype if K.is_floating_point() else torch.float64
        return F_proj.to(device=K.device, dtype=out_dtype), info


class ReversibleCommutingIProjector(FluxIProjector):
    """Alternating I-projector onto the reversible commuting flux subset.
    This projects a square matrix F onto the set of positive matrices satisfying
        F = F^T,
        F 1 = P,
        U_prev F = F U_prev^T,
    where U_prev is assumed to be column-stochastic and reversible with respect to P.

    The projector alternates between
        (1) I-projection onto the symmetric fixed-marginal subset,
            computed with Knight-Ruiz-Ucar scaling; and
        (2) I-projection onto the commuting affine subset,
            computed by solving the dual of the affine KL projection problem.

    Parameters
    ----------
    min_entry : float
        Minimum permitted matrix entry.
    max_iters : int
        Maximum number of projection cycles.
    tol : float
        Tolerated maximum absolute residual for feasibility.
    max_iters_symm : int, optional
        Maximum Knight-Ruiz-Ucar matrix-scaling iterations.
    max_iters_comm : int, optional
        Maximum Newton iterations for projection onto commuting subset.
    max_iters_ls : int, optional
        Maximum line-search iterations.
    armijo : float, optional
        Inner-product prefactor for Armijo line search.
    progress : bool, optional
        Whether to display progress bar.

    Method
    ------
    project(
        K: torch.Tensor,
        U_prev: torch.Tensor,
        P: torch.Tensor,
        check_every: int = 1,
    ) -> tuple[torch.Tensor, ReversibleCommutingProjectionInfo]:
        Projects K onto feasible subset.

    Reference
    ---------
    L.M. Bregman,
    The relaxation method of ﬁnding the common point of convex sets and
    its application to the solution of problems in convex programming,
    U.S.S.R. Comput. Math. Math. Phys. 7, 200 (1967).
    """

    def __init__(
        self,
        min_entry: float,
        max_iters: int,
        tol: float,
        max_iters_symm: int = 2500,
        max_iters_comm: int = 250,
        max_iters_ls: int = 25,
        armijo: float = 1e-4,
        progress: bool = False,
    ) -> None:
        super().__init__(min_entry, max_iters, tol, progress)
        self.max_iters_ls = int(max_iters_ls)
        self.armijo = float(armijo)
        self._symm = KnightRuizUcarScaler(
            min_entry, max_iters_symm, tol, progress=False
        )
        self._comm = _CommutingFluxIProjector(
            min_entry,
            max_iters_comm,
            tol,
            max_iters_ls=self.max_iters_ls,
            armijo=self.armijo,
            progress=False,
        )
        self._P: torch.Tensor | None = None

    def _reset(self) -> None:
        """Forget cached inner-projector state."""
        self._symm._reset()
        self._comm._reset()
        self._P = None

    def _prepare_cache(self, P: torch.Tensor) -> None:
        """Reset symmetric scaling cache if the marginal changes."""
        Pf = as_float64(P).detach().cpu()
        if self._P is None or not torch.equal(self._P, Pf):
            self._symm._reset()
            self._P = Pf.clone()

    @staticmethod
    def _get_residuals(
        F: torch.Tensor, U_prev: torch.Tensor, P: torch.Tensor
    ) -> dict[str, float]:
        """Compute feasibility residuals for diagnostics."""
        return {
            "symm_err": float(torch.abs(F - F.T).max().detach().cpu().item()),
            "row_err": float(torch.abs(F.sum(dim=1) - P).max().detach().cpu().item()),
            "comm_err": float(
                torch.abs(U_prev @ F - F @ U_prev.T).max().detach().cpu().item()
            ),
            "neg_err": float(torch.clamp(-F.min(), min=0.0).detach().cpu().item()),
        }

    @torch.no_grad()
    def project(
        self,
        K: torch.Tensor,
        U_prev: torch.Tensor,
        P: torch.Tensor,
        check_every: int = 1,
    ) -> tuple[torch.Tensor, ReversibleCommutingProjectionInfo]:
        """Project K onto the reversible commuting flux subset in KL divergence.

        Parameters
        ----------
        K : (n, n) torch.Tensor
            Nonnegative input matrix.
        U_prev : (n, n) torch.Tensor
            Fixed column-stochastic matrix satisfying U_prev diag(P) = diag(P) U_prev^T.
        P : (n,) torch.Tensor
            Strictly positive stationary distribution.
        check_every : int
            Frequency in alternating sweeps at which to compute residuals.

        Returns
        -------
        F_proj : (n, n) torch.Tensor
            KL projection of K onto the feasible subset.
        info : ReversibleCommutingProjectionInfo
            Diagnostics including residuals, iterations, and inner-projector metadata.
        """
        if K.ndim != 2 or K.shape[0] != K.shape[1] or type(K) != torch.Tensor:
            raise ValueError(
                f"K must be a square tensor, got a {tuple(K.shape)} {type(K)}."
            )
        if not torch.isfinite(K).all():
            raise ValueError("K must have only finite entries.")
        if U_prev.shape != K.shape:
            raise ValueError(
                f"U_prev must have shape {tuple(K.shape)}, got {tuple(U_prev.shape)}."
            )
        if not torch.isfinite(U_prev).all():
            raise ValueError("U_prev must have only finite entries.")
        if not torch.isfinite(P).all():
            raise ValueError("P must have only finite entries.")

        n = K.shape[0]
        check_probability_vector(P, n)
        check_stationary_column_stochastic_matrix(U_prev, P, db=True, name="U_prev")
        Kf = as_float64(K).detach().cpu()
        Uf = as_float64(U_prev).detach().cpu()
        Pf = normalize_probability_vector(P).detach().cpu()
        F_proj = torch.clamp(
            nan_to_pos(Kf, min_entry=self.min_entry), min=self.min_entry
        )

        check_every = max(int(check_every), 1)
        converged = False
        iter_range = range(self.max_iters)
        if self.progress and tqdm is not None:
            iter_range = tqdm(iter_range, desc="Cyclic KL projections", leave=False)

        self._prepare_cache(Pf)
        comm_info: CommutingProjectionInfo | None = None
        symm_err = row_err = comm_err = neg_err = step_err = float("inf")

        # we always start and end with KRU for stability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F_proj, symm_info = self._symm.project(F_proj, Pf)

        for it in iter_range:
            F_last = F_proj.clone()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                F_proj, comm_info = self._comm.project(F_proj, Uf)
                F_proj, symm_info = self._symm.project(F_proj, Pf)

            if (it % check_every) == 0 or it == self.max_iters - 1:
                residuals = self._get_residuals(F_proj, Uf, Pf)
                symm_err = residuals["symm_err"]
                row_err = residuals["row_err"]
                comm_err = residuals["comm_err"]
                neg_err = residuals["neg_err"]
                step_err = float(torch.abs(F_proj - F_last).max().detach().cpu().item())
                if max(symm_err, row_err, comm_err, neg_err) <= self.tol:
                    converged = True
                    break

        info: ReversibleCommutingProjectionInfo = {
            "converged": converged,
            "iters": it + 1,
            "symm_err": symm_err,
            "row_err": row_err,
            "comm_err": comm_err,
            "neg_err": neg_err,
            "step_err": step_err,
            "symm_iters": int(symm_info["iters"]),
            "symm_row_err": float(symm_info["row_err"]),
            "comm_iters": comm_info["iters"] if comm_info is not None else 0,
            "comm_affine_err": comm_info["affine_err"]
            if comm_info is not None
            else float("inf"),
            "constraint_rank": comm_info["constraint_rank"]
            if comm_info is not None
            else 0,
            "objective": float(
                generalized_kl_divergence(
                    F_proj,
                    torch.clamp(
                        nan_to_pos(Kf, min_entry=self.min_entry), min=self.min_entry
                    ),
                    eps=self.min_entry,
                )
                .detach()
                .cpu()
                .item()
            ),
        }

        if not info["converged"]:
            warnings.warn(
                "Cyclic KL projection not converged: "
                f"symmetry error {symm_err:.2e}, "
                f"row-marginal error {row_err:.2e}, "
                f"commutator error {comm_err:.2e} greater than tolerance {self.tol}. "
                "Increase max_iters."
            )

        out_dtype = K.dtype if K.is_floating_point() else torch.float64
        return F_proj.to(device=K.device, dtype=out_dtype), info


### WRAPPER FUNCTIONS ###


def project_Us(
    Us: torch.Tensor,
    lim_dist: torch.Tensor,
    tol: float = 1e-12,
    min_entry: float = 1e-24,
    max_iters: int = 1000,
    reversible: bool = False,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict[int, SinkhornKnoppInfo | MarginalProjectionInfo]]:
    """Project transition matrices onto feasible set using Sinkhorn scaling.

    Parameters
    ----------
    Us : (lags, n, n) torch.Tensor
        Maximum-likelihood transition matrices (column-stochastic).
    lim_dist : (n,) torch.Tensor
        Required stationary distribution.
    tol : float, optional
        Convergence tolerance.
    min_entry : float, optional
        Smallest value to treat as positive.
    max_iters : int, optional
        Maximum number of Sinkhorn iterations.
    reversible : bool, optional
        Whether transition matrices must be reversible.
    verbose : bool, optional
        Whether to display progress bar.

    Returns
    -------
    Us_proj : (lags, n, n) torch.Tensor
        Transfer operators projected onto feasible set.
    metrics : dict[int, SinkhornKnoppInfo | MarginalProjectionInfo]
        Projection metadata.
    """
    if reversible:
        projector = KnightRuizUcarScaler(min_entry, max_iters, tol, progress=False)
    else:
        projector = SinkhornKnoppScaler(min_entry, max_iters, tol, progress=False)

    Us_proj = torch.zeros_like(Us)
    Us_proj[0] = Us[0]
    metrics: dict[int, SinkhornKnoppInfo | MarginalProjectionInfo] = {}
    for lag in tqdm(
        range(1, Us.shape[0]),
        desc="Projecting transition matrices",
        disable=not verbose,
        leave=False,
    ):
        projector._reset()
        F = Us[lag] * lim_dist[None, :]
        if reversible:
            F = torch.clamp(F, min=min_entry)
            F = symmetrize_matrix(F, geom=True)
        F, info = projector.project(F, lim_dist)
        Us_proj[lag] = F / lim_dist[None, :]
        metrics[lag] = info

    return Us_proj, metrics
