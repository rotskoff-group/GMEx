"""Metadata types shared by flux projectors and optimization routines."""

from typing import TypedDict


class ProjectionInfo(TypedDict):
    """Convergence status and iteration count of a projection."""

    converged: bool
    iters: int


class MarginalProjectionInfo(ProjectionInfo):
    """Projection diagnostics including the flux row-marginal error."""

    row_err: float


class SinkhornKnoppInfo(MarginalProjectionInfo):
    """Sinkhorn-Knopp diagnostics for both flux marginals."""

    col_err: float


class AffineProjectionInfo(ProjectionInfo):
    """Diagnostics from affine KL dual solver."""

    affine_err: float
    step_err: float
    dual_obj: float
    ridge: float


class CommutingProjectionInfo(ProjectionInfo):
    """Diagnostics from projection onto commuting-flux constraints."""

    comm_err: float
    neg_err: float
    step_err: float
    affine_err: float
    constraint_rank: int
    objective: float


class ReversibleCommutingProjectionInfo(MarginalProjectionInfo):
    """Diagnostics from cyclic symmetric and commuting flux projections."""

    symm_err: float
    comm_err: float
    neg_err: float
    step_err: float
    symm_iters: int
    symm_row_err: float
    comm_iters: int
    comm_affine_err: float
    constraint_rank: int
    objective: float
