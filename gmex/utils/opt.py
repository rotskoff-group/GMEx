# gmex/utils/opt.py
# Helper functions for convex optimization.


from warnings import warn

import torch

from .core import *
from .types import AffineProjectionInfo


@torch.no_grad()
def check_count_matrices(Cs: torch.Tensor) -> None:
    """Validate skip bigrams."""
    if Cs.ndim != 3:
        raise ValueError(f"Count matrices Cs must be 3D, got shape {tuple(Cs.shape)}.")
    if Cs.shape[1] != Cs.shape[2]:
        raise ValueError(
            f"Count matrices must have same 1st and 2nd dimension, got shape {tuple(Cs.shape)}."
        )
    if not torch.allclose(Cs[0], torch.diag(torch.diag(Cs[0]))):
        raise ValueError("First count matrix must be diagonal.")


@torch.no_grad()
def rescale_sinkhorn_input(K: torch.Tensor, min_entry: float = 1e-24) -> torch.Tensor:
    """Rescale Sinkhorn input K to have mean 1.0.
    This will not affect output as the diagonal matrices will absorb the scaling.
    """
    Kp = nan_to_pos(as_float64(K), min_entry=min_entry)
    Kp = torch.clamp(Kp, min=min_entry)
    return Kp / torch.clamp(Kp.mean(), min=min_entry)


@torch.no_grad()
def get_log_likelihood(
    C: torch.Tensor, U: torch.Tensor, min_entry: float | None = None
) -> torch.Tensor:
    """Compute the (smoothed) log-likelihood sum_{ij} C_{ij} log U_{ij}.

    Parameters
    ----------
    C : (n, n) torch.Tensor
        Nonnegative count matrix.
    U : (n, n) torch.Tensor
        Nonnegative flux or transition matrix.
    min_prob : float, optional
        Clamp U to this value from below to avoid log(0).

    Returns
    -------
    ll : torch.Tensor
        Log likelihood.
    """
    Cf = as_float64(C).to(device=U.device, dtype=U.dtype)
    Uc = torch.clamp(U, min=min_entry)
    return torch.sum(Cf * torch.log(Uc))


@torch.no_grad()
def generalized_kl_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12, reduction: str = "sum"
) -> torch.Tensor:
    """Generalized (unnormalized) KL / I-divergence.

    For nonnegative matrices p and q, the generalized KL is:
        gKL(p||q) = sum_{ij} p_ij * log(p_ij / q_ij) - p_ij + q_ij.

    This is the Bregman divergence for the (negative) entropy generator phi(x)=sum x log x.
    It is well-defined without requiring sum p = sum q.

    Parameters
    ----------
    p, q : torch.Tensor
        Nonnegative tensors of the same shape.
    eps : float
        Lower clamp for the log-ratio to avoid log(0).
    reduction : {'sum', 'none'}
        Whether to return the summed divergence or the elementwise divergence.

    Returns
    -------
    torch.Tensor
        Scalar divergence if reduction='sum', else an array of the same shape as p.
    """
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    div = p_safe * (torch.log(p_safe) - torch.log(q_safe)) - p_safe + q_safe
    if reduction == "sum":
        return div.sum()
    if reduction == "none":
        return div
    raise ValueError("reduction must be 'sum' or 'none'")


@torch.no_grad()
def solve_levenberg_step(
    grad: torch.Tensor, hess: torch.Tensor, tol: float = 1e-12
) -> tuple[torch.Tensor, float]:
    """Solve a Levenberg-regularized Newton system for concave maximization.
    This solves (rho I - H) p = g where H is a symmetric Hessian of a concave objective;
    rho is the smallest nonnegative scalar making -H + rho I positive definite to tolerance.

    Parameters
    ----------
    grad : (m,) torch.Tensor
        Gradient vector g.
    hess : (m, m) torch.Tensor
        Hessian matrix H.
    tol : float, optional
        Target lower bound for the minimum eigenvalue of rho I - H.

    Returns
    -------
    step : (m,) torch.Tensor
        Levenberg-regularized Newton direction.
    rho : float
        Diagonal shift applied to -H.

    Reference
    ---------
    K. Levenberg, A method of solution for certain non-linear problems in least squares,
    Quart. Appl. Math. 2, 164 (1944).
    """
    gf = as_float64(grad)
    K = -symmetrize_matrix(hess)

    if gf.ndim != 1:
        raise ValueError(f"grad must be 1D, got shape {tuple(gf.shape)}.")
    if K.shape != (gf.numel(), gf.numel()):
        raise ValueError(
            f"hess must have shape {(gf.numel(), gf.numel())}, got {tuple(K.shape)}."
        )
    if gf.numel() == 0:
        return gf.clone(), 0.0

    min_eigval = torch.linalg.eigvalsh(K).min()
    ridge_min = max(tol, torch.finfo(K.dtype).eps)
    rho_t = torch.clamp(
        torch.tensor(ridge_min, dtype=K.dtype, device=K.device) - min_eigval, min=0.0
    )
    K_reg = K + rho_t * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)

    try:  # this is faster but fails for ill-conditioned K_reg
        step = torch.linalg.solve(K_reg, gf[:, None]).squeeze(1)
    except RuntimeError:
        step = torch.linalg.lstsq(K_reg, gf[:, None]).solution.squeeze(1)

    return step, float(rho_t.detach().cpu().item())


@torch.no_grad()
def build_commuting_constraints_flux(
    U: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct A x = b equivalent to U F = F U^T where
        x[i * n + j] = F[i, j],
    i.e., using row-major convention.

    Parameters
    ----------
    U : (n, n) torch.Tensor
        Column-stochastic matrix.

    Returns
    -------
    A : (n * n, n * n) torch.Tensor
        Dense constraint matrix.
    b : (n * n,) torch.Tensor
        Right-hand side, identically zero.
    """
    check_column_stochastic_matrix(U, name="U")
    Uf = as_float64(U)
    n = Uf.shape[0]

    A = torch.zeros((n * n, n * n), dtype=Uf.dtype, device=Uf.device)
    b = torch.zeros((n * n,), dtype=Uf.dtype, device=Uf.device)

    row = 0
    for i in range(n):
        for j in range(n):
            A[row, j::n] += Uf[i, :]  # (U F)_{ij}
            A[row, i * n : (i + 1) * n] -= Uf[j, :]  # -(F U^T)_{ij}
            row += 1

    return A, b


@torch.no_grad()
def orthonormalize_affine_constraints(
    A: torch.Tensor, b: torch.Tensor, tol: float = 1e-12, linear: bool = False
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compress A x = b to an equivalent orthonormal system Q x = c.
    If A has rank r, this returns Q with shape (r, n_features).
    Q has orthonormal rows such that A x = b  <=>  Q x = c.

    Parameters
    ----------
    A : (m, d) torch.Tensor
        Constraint matrix.
    b : (m,) torch.Tensor
        Constraint right-hand side.
    tol : float, optional
        Tolerance used to determine the numerical rank and consistency.
    linear : bool, optional
        Whether to treat b as 0.
        Avoids potentially unstable computations for nonlinear affine case if True.

    Returns
    -------
    Q : (r, d) torch.Tensor
        Matrix with orthonormal rows.
    c : (r,) torch.Tensor
        Right-hand side for the orthonormal system.
    rank : int
        Numerical rank r.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {tuple(A.shape)}.")
    if b.ndim != 1:
        raise ValueError(f"b must be 1D, got shape {tuple(b.shape)}.")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b must have shape ({A.shape[0]},), got {tuple(b.shape)}.")

    Af = as_float64(A)
    bf = as_float64(b).to(device=Af.device, dtype=Af.dtype)
    m, d = Af.shape
    if m == 0:
        return (
            torch.zeros((0, d), dtype=Af.dtype, device=Af.device),
            torch.zeros((0,), dtype=Af.dtype, device=Af.device),
            0,
        )

    if linear and not bf.abs().max() <= tol:
        warn(
            "Setting linear to True while b is nonzero ignores b. "
            f"Largest element of b has magnitude {bf.abs().max():.2e} > tol."
        )

    U, S, Vh = torch.linalg.svd(Af, full_matrices=False)

    eps = torch.finfo(Af.dtype).eps
    s0 = float(S[0].detach().cpu().item()) if S.numel() else 0.0
    rank_tol = max(float(tol), eps * max(m, d)) * max(1.0, s0)
    rank = int((S > rank_tol).sum().item()) if S.numel() else 0

    Q = Vh[:rank, :]
    if linear:
        c = torch.zeros((rank,), dtype=Af.dtype, device=Af.device)
    else:
        Ur = U[:, :rank]
        resid = bf - Ur @ (Ur.T @ bf)
        inconsistency = torch.linalg.vector_norm(resid)
        rhs_scale = max(1.0, float(torch.linalg.vector_norm(bf).detach().cpu().item()))
        if float(inconsistency.detach().cpu().item()) > rank_tol * rhs_scale:
            raise ValueError(
                "Linear equality constraints are inconsistent within numerical tolerance."
            )
        c = (Ur.T @ bf) / S[:rank]

    if rank == 0:
        return (
            torch.zeros((0, d), dtype=Af.dtype, device=Af.device),
            torch.zeros((0,), dtype=Af.dtype, device=Af.device),
            0,
        )

    return Q, c, rank


@torch.no_grad()
def kl_affine_primal_from_dual(
    log_k: torch.Tensor,
    Q: torch.Tensor,
    dual: torch.Tensor,
    log_min: float,
    log_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log_x and x for the affine-KL dual iterate."""
    log_x = torch.clamp(log_k - Q.T @ dual, min=log_min, max=log_max)
    x = torch.exp(log_x)
    return log_x, x


@torch.no_grad()
def kl_affine_dual_backtracking(
    log_k: torch.Tensor,
    Q: torch.Tensor,
    c: torch.Tensor,
    dual: torch.Tensor,
    step: torch.Tensor,
    dual_obj: float,
    slope: float,
    log_min: float,
    log_max: float,
    ls_update: float = 0.5,
    armijo: float = 1e-4,
    max_iters_ls: int = 25,
) -> tuple[torch.Tensor, float, float, bool]:
    """Armijo backtracking for the affine-KL dual objective."""
    eta = 1.0
    for _ in range(max_iters_ls):
        dual_try = dual + eta * step
        _, x_try = kl_affine_primal_from_dual(log_k, Q, dual_try, log_min, log_max)
        dual_obj_try = (c @ dual_try + x_try.sum()).detach().cpu().item()
        if dual_obj_try <= dual_obj + armijo * eta * slope:
            return dual_try, dual_obj_try, eta, True
        eta *= ls_update
    return dual, dual_obj, 0.0, False


@torch.no_grad()
def project_affine_kl_orthonormal(
    k: torch.Tensor,
    Q: torch.Tensor,
    c: torch.Tensor,
    tol: float = 1e-12,
    max_iters_proj: int = 1000,
    max_iters_ls: int = 25,
    armijo: float = 1e-4,
    min_entry: float = 1e-24,
    dual0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, AffineProjectionInfo, torch.Tensor]:
    """KL-project k onto the affine set {x : Q x = c} for orthonormal-row Q.

    This solves
        min_x  D_{gKL}(x || k)
        s.t.   Q x = c
    where Q has orthonormal rows.

    The KKT conditions imply that the primal optimizer has the exponential form
        x(lambda) = k * exp(-Q^T lambda),
    where Q x(lambda) = c solves the dual optimality condition.

    Parameters
    ----------
    k : (d,) torch.Tensor
        Strictly positive reference vector.
    Q : (r, d) torch.Tensor
        Matrix with orthonormal rows.
    c : (r,) torch.Tensor
        Right-hand side for the affine constraints.
    tol : float, optional
        Tolerated maximum absolute affine residual.
    max_iters : int, optional
        Maximum damped-Newton iterations in the dual.
    min_entry : float, optional
        Smallest value to treat as positive.
    armijo : float, optional
        Inner-product prefactor for Armijo line search.
    dual0 : (r,) torch.Tensor | None, optional
        Optional warm start for the dual variables.

    Returns
    -------
    x_proj : (d,) torch.Tensor
        KL projection of k onto the affine set.
    info : AffineProjectionInfo
        Diagnostics including residuals and iterations.
    dual : (r,) torch.Tensor
        Final dual iterate, useful as a warm start.
    """
    if k.ndim != 1:
        raise ValueError(f"k must be 1D, got shape {tuple(k.shape)}.")
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2D, got shape {tuple(Q.shape)}.")
    if c.ndim != 1:
        raise ValueError(f"c must be 1D, got shape {tuple(c.shape)}.")

    kf = as_float64(k).clone()
    Qf = as_float64(Q)
    cf = as_float64(c).to(device=Qf.device, dtype=Qf.dtype)

    if Qf.shape[0] != cf.shape[0]:
        raise ValueError(
            f"Q and c have incompatible shapes {tuple(Qf.shape)} and {tuple(cf.shape)}."
        )
    if Qf.shape[1] != kf.numel():
        raise ValueError(
            f"Q has incompatible width {Qf.shape[1]} for k with {kf.numel()} entries."
        )
    if not torch.isfinite(kf).all():
        raise ValueError("k must have only finite entries.")

    min_entry = float(min_entry)
    if min_entry <= 0.0:
        raise ValueError(f"min_entry must be positive, got {min_entry}.")

    kf = torch.clamp(nan_to_pos(kf, min_entry=min_entry), min=min_entry)

    if Qf.shape[0] == 0:
        info: AffineProjectionInfo = {
            "converged": True,
            "iters": 0,
            "affine_err": 0.0,
            "step_err": 0.0,
            "dual_obj": float(kf.sum().detach().cpu().item()),
            "ridge": 0.0,
        }
        dual = torch.zeros((0,), dtype=Qf.dtype, device=Qf.device)
        return kf.clone(), info, dual

    if dual0 is None:
        dual = torch.zeros((Qf.shape[0],), dtype=Qf.dtype, device=Qf.device)
    else:
        if dual0.ndim != 1:
            raise ValueError(f"dual0 must be 1D, got shape {tuple(dual0.shape)}")
        dual = as_float64(dual0).to(device=Qf.device, dtype=Qf.dtype)
        if dual.shape != (Qf.shape[0],):
            raise ValueError(
                f"dual0 must have shape ({Qf.shape[0]},), got {tuple(dual.shape)}."
            )

    finfo = torch.finfo(kf.dtype)
    log_min = float(
        torch.log(torch.tensor(min_entry, dtype=kf.dtype, device=kf.device))
        .detach()
        .cpu()
        .item()
    )
    log_max = float(
        torch.log(torch.tensor(finfo.max, dtype=kf.dtype, device=kf.device))
        .detach()
        .cpu()
        .item()
    )
    log_k = torch.log(kf)

    converged = False
    step_err = float("inf")
    affine_err = float("inf")
    dual_obj = float("inf")
    ridge = 0.0  # this is the Levenberg regularization coefficient often called rho
    n_iters = 0

    for it in range(max_iters_proj):
        n_iters = it + 1

        _, x = kl_affine_primal_from_dual(log_k, Qf, dual, log_min, log_max)
        resid = Qf @ x - cf
        affine_err = float(torch.abs(resid).max().detach().cpu().item())
        dual_obj = (cf @ dual + x.sum()).detach().cpu().item()

        if affine_err <= tol:
            converged = True
            step_err = 0.0
            ridge = 0.0
            break

        H = symmetrize_matrix((Qf * x.unsqueeze(0)) @ Qf.T)
        step, ridge = solve_levenberg_step(resid, -H, tol=tol)
        step_err = float(torch.abs(step).max().detach().cpu().item())

        slope_t = -torch.dot(resid, step)  # = grad^T step for dual minimization
        if (
            (not torch.isfinite(step).all())
            or (not torch.isfinite(slope_t))
            or float(slope_t.detach().cpu().item()) >= 0.0
        ):
            break
        slope = float(slope_t.detach().cpu().item())

        dual, dual_obj, alpha, accepted = kl_affine_dual_backtracking(
            log_k=log_k,
            Q=Qf,
            c=cf,
            dual=dual,
            step=step,
            dual_obj=dual_obj,
            slope=slope,
            log_min=log_min,
            log_max=log_max,
            armijo=armijo,
            max_iters_ls=max_iters_ls,
        )
        step_err = alpha * step_err

        if not accepted:
            break

    _, x_proj = kl_affine_primal_from_dual(log_k, Qf, dual, log_min, log_max)
    resid = Qf @ x_proj - cf
    affine_err = float(torch.abs(resid).max().detach().cpu().item())
    if affine_err <= tol:
        converged = True

    info: AffineProjectionInfo = {
        "converged": converged,
        "iters": n_iters,
        "affine_err": affine_err,
        "step_err": step_err,
        "dual_obj": dual_obj,
        "ridge": ridge,
    }

    return x_proj, info, dual
