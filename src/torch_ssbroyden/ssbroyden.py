"""
Self-scaled Broyden (SSBroyden) optimizers for PyTorch, as torch.optim.Optimizer subclasses.

  * SSBroyden  : dense n x n inverse-Hessian approximation (reference; small n only)
  * LSSBroyden : limited-memory version, O(mn) memory, O(mn + m^3) per iteration

The interface follows torch.optim.LBFGS:

    opt = LSSBroyden(model.parameters(), history_size=50, max_iter=20,
                     line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    for epoch in range(N):
        loss = opt.step(closure)      # runs up to max_iter iterations

Update (inverse form, Kiyani et al., CMAME 446 (2025) 118308, Eq. (8)):
    H+ = (1/tau) [ H - H y y^T H / (y^T H y) + phi (y^T H y) v v^T ] + s s^T / (y^T s)
    v  = s / (y^T s) - H y / (y^T H y)
(tau, phi) follow Sec. 1.3 of that paper (see ss_params). update="bfgs" gives
(L-)BFGS; with update="bfgs", h0="latest", curvature_guard="absolute" LSSBroyden
reproduces torch.optim.LBFGS.

curvature_guard: an update is skipped when s^T y is too small. "relative" (default)
uses s^T y <= eps |s||y|; "absolute" uses torch.optim.LBFGS's s^T y <= 1e-10, which
skips most updates once the loss is small and stalls high-accuracy PINN training.

Limited-memory representation (compact form, computed in Gram space):
    H_k = c * gamma * I + P A P^T,   P = [s_1..s_m, y_1..y_m]  (n x 2m)
After each update, A (2m x 2m) and c are rebuilt from the Gram matrix G = P^T P and the
stored scalars (tau_i, phi_i). H v costs two skinny mat-vecs (P^T v, P w); under
parameter sharding the only communication is an all-reduce of the 2m-vector P^T v.

b_k = s^T B s / s^T y is computed as -t s^T g / s^T y (Eq. (5)), with t the step length
actually taken along d = -H g.

Only one parameter group is supported (as for LBFGS). float64 is strongly recommended.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.optim import Optimizer

try:  # in-place BLAS rank-1 update (falls back to NumPy)
    from scipy.linalg.blas import dger as _dger

    def _ger(alpha, u, B):
        return _dger(alpha, u, u, a=B, overwrite_a=1)
except ImportError:  # pragma: no cover
    def _ger(alpha, u, B):
        B += alpha * np.outer(u, u)
        return B
from ._line_search import _strong_wolfe  # copy of the helper used by torch.optim.LBFGS (BSD-3-Clause)

__all__ = ["SSBroyden", "LSSBroyden", "DenseSSBroyden", "ss_params"]


# ---------------------------------------------------------------------------
# Update parameters (Kiyani et al. 2025, Sec. 1.3)
# ---------------------------------------------------------------------------
def ss_params(b: float, h: float, n: int, update: str = "ssbroyden",
              tau_beta: float = 0.0) -> tuple[float, float]:
    """Return (tau, phi) for the inverse update.

    b = s^T B s / s^T y,  h = y^T H y / s^T y,  n = number of parameters (N in the paper).
    tau_beta (experimental, SSBroyden only): tau uses rho_beta = min(1, kappa(a)^beta / b)
    with kappa(a) = (1+a)(1-sqrt(a/(1+a))) in (1/2, 1].  beta = 0 is the paper's rule
    (rho^+); beta = 1 replaces rho^+ by rho^- (stronger scaling).
    """
    if update == "bfgs":
        return 1.0, 1.0
    rho_p = min(1.0, 1.0 / b)
    if update == "ssbfgs":
        return rho_p, 1.0
    if update != "ssbroyden":
        raise ValueError(update)
    a = b * h - 1.0  # >= 0 by Cauchy-Schwarz
    if a <= 1e-12:  # H exact along y: theta = 0
        return rho_p, 1.0
    c = math.sqrt(a / (1.0 + a))
    rho_m = min(1.0, h * (1.0 - c))
    theta_m = (rho_m - 1.0) / a
    theta_p = 1.0 / rho_m
    theta = max(theta_m, min(theta_p, (1.0 - b) / b))
    if tau_beta:
        rho_p = min(1.0, ((1.0 + a) * (1.0 - c)) ** tau_beta / b)
    sigma = 1.0 + a * theta  # >= rho_m > 0
    sig = abs(sigma) ** (1.0 / (1 - n)) if n > 1 else 1.0  # sigma^(1-N)
    if theta <= 0:
        tau = min(rho_p * sig, sigma)
    else:
        tau = rho_p * min(sig, 1.0 / theta)
    phi = (1.0 - theta) / (1.0 + a * theta)
    return tau, phi


# ---------------------------------------------------------------------------
# Common driver
# ---------------------------------------------------------------------------
class _SSBroydenBase(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search_fn: str | None = "strong_wolfe",
        update: str = "ssbroyden",
        line_search_tolerance: float = 1e-9,
        curvature_guard: str = "relative",
        tau_beta: float = 0.0,
        **extra,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if update not in ("ssbroyden", "ssbfgs", "bfgs"):
            raise ValueError("update must be 'ssbroyden', 'ssbfgs' or 'bfgs'")
        if curvature_guard not in ("relative", "absolute"):
            raise ValueError("curvature_guard must be 'relative' or 'absolute'")
        if line_search_fn not in (None, "strong_wolfe"):
            raise ValueError("only 'strong_wolfe' or None is supported")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change, line_search_fn=line_search_fn,
            update=update, line_search_tolerance=line_search_tolerance,
            curvature_guard=curvature_guard, tau_beta=tau_beta, **extra,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(f"{type(self).__name__} doesn't support per-parameter options")
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    # flat helpers (same conventions as torch.optim.LBFGS) ----------------
    def _numel(self) -> int:
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat_grad(self) -> torch.Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                views.append(p.new_zeros(p.numel()))
            elif p.grad.is_sparse:
                views.append(p.grad.to_dense().view(-1))
            else:
                views.append(p.grad.view(-1))
        return torch.cat(views, 0)

    @torch.no_grad()
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    @torch.no_grad()
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    # mini-batch support ------------------------------------------------------
    def new_batch(self):
        """Call after the closure has switched to a new batch of data.

        The curvature memory is kept, but the pair that would connect the last step on the
        old batch with the gradient on the new batch is not formed (its gradient
        difference would mix two different objectives).  See torch_ssbroyden.SegmentTrainer.
        """
        state = self.state[self._params[0]]
        state["skip_next_update"] = True

    # to be implemented by subclasses --------------------------------------
    def _apply_H(self, state, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _push(self, state, s, y, Hy, tau, phi):
        raise NotImplementedError

    def _reset(self, state):
        raise NotImplementedError

    # curvature-pair update (called at the start of an iteration) ---------
    def _update_memory(self, state, group, flat_grad):
        d, t, g_prev = state["d"], state["t"], state["prev_flat_grad"]
        s = d.mul(t)
        y = flat_grad.sub(g_prev)
        sy = float(y.dot(s))
        if group["curvature_guard"] == "absolute":
            skip = sy <= 1e-10  # torch.optim.LBFGS rule
        else:
            # scale-invariant: the absolute 1e-10 rule skips most updates once the
            # loss is small (~1e-7 for a PINN), which freezes H and stalls training
            skip = sy <= torch.finfo(s.dtype).eps * float(s.norm()) * float(y.norm())
        if skip:
            state["n_skipped"] = state.get("n_skipped", 0) + 1
            return
        Hy = self._apply_H(state, y)
        h = float(y.dot(Hy)) / sy
        b = -t * float(s.dot(g_prev)) / sy  # = s^T B s / s^T y since B d = -g_prev
        tau, phi = ss_params(b, h, self._numel(), group["update"], group.get("tau_beta", 0.0))
        self._push(state, s, y, Hy, tau, phi)

    # one call = up to max_iter iterations ---------------------------------
    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        if closure is None:
            raise RuntimeError(f"{type(self).__name__} requires a closure")
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        lr = float(group["lr"])
        max_iter, max_eval = group["max_iter"], group["max_eval"]
        tolerance_grad, tolerance_change = group["tolerance_grad"], group["tolerance_change"]
        line_search_fn = group["line_search_fn"]

        # global state lives on the first parameter (as in LBFGS) so that
        # state_dict()/load_state_dict() work
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)
        if "initialized" not in state:
            self._reset(state)
            state["initialized"] = True

        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1
        flat_grad = self._gather_flat_grad()
        if flat_grad.abs().max() <= tolerance_grad:
            return orig_loss

        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1

            # direction (a restart after a stall starts again from the gradient)
            restart = state.pop("restart", False)
            if state["n_iter"] == 1 or restart:
                d = flat_grad.neg()
            else:
                if not state.pop("skip_next_update", False):
                    self._update_memory(state, group, flat_grad)
                d = self._apply_H(state, flat_grad).neg()
            state["prev_flat_grad"] = flat_grad.clone(memory_format=torch.contiguous_format)
            state["prev_loss"] = loss

            # step length
            t = (min(1.0, 1.0 / float(flat_grad.abs().sum())) * lr
                 if (state["n_iter"] == 1 or restart) else lr)
            gtd = float(flat_grad.dot(d))
            if gtd >= 0 and not restart and float(flat_grad.abs().max()) > 0:
                # H is no longer positive definite along g (round-off or a negative Broyden
                # parameter).  Without a restart the next call would return here again with the
                # same H and the run would stop for good; discard the memory instead.
                self._reset(state)
                state["n_restarts"] = state.get("n_restarts", 0) + 1
                state["restart"] = True
                continue
            if gtd > -tolerance_change:
                break

            ls_func_evals = 0
            if line_search_fn == "strong_wolfe":
                x_init = self._clone_param()

                # torch's _strong_wolfe compares losses with '<' and '>', which are False for NaN,
                # so a trial point with a NaN loss can be accepted.  Replace a non-finite trial
                # loss by a large finite value (with the starting gradient), so that the search
                # backs off; if it still ends on such a point (or makes no step after meeting one), stay
                # put and discard the memory.
                f_bad = abs(loss) * 1e8 + 1e8
                g0 = flat_grad
                n_bad = state.get("n_nonfinite", 0)

                def obj_func(x, t_, d_):
                    f_, g_ = self._directional_evaluate(closure, x, t_, d_)
                    if not (math.isfinite(f_) and bool(torch.isfinite(g_).all())):
                        state["n_nonfinite"] = state.get("n_nonfinite", 0) + 1
                        return f_bad, g0.clone()
                    return f_, g_

                loss0 = loss
                loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                    obj_func, x_init, t, d, loss, flat_grad, gtd,
                    tolerance_change=group["line_search_tolerance"],
                    max_ls=max(max_eval - current_evals, 1),
                )
                if loss == f_bad or (t == 0 and state.get("n_nonfinite", 0) > n_bad):
                    loss, flat_grad = loss0, g0
                    current_evals += ls_func_evals
                    state["func_evals"] += ls_func_evals
                    self._reset(state)
                    state["n_restarts"] = state.get("n_restarts", 0) + 1
                    state["restart"] = True
                    if current_evals >= max_eval:
                        break
                    continue
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                self._add_grad(t, d)
                opt_cond = False
                if n_iter != max_iter:
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1
            state["d"], state["t"] = d, t
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            if n_iter == max_iter or current_evals >= max_eval or opt_cond:
                break
            if d.mul(t).abs().max() <= tolerance_change:
                break
            if abs(loss - state["prev_loss"]) < tolerance_change:
                break

        return orig_loss


# ---------------------------------------------------------------------------
# Dense reference implementation
# ---------------------------------------------------------------------------
class SSBroyden(_SSBroydenBase):
    """Dense SSBroyden (n x n inverse Hessian). For reference / small models only.

    Args: as torch.optim.LBFGS (lr, max_iter, max_eval, tolerance_grad,
    tolerance_change, line_search_fn; default line_search_fn="strong_wolfe"), plus
    update: "ssbroyden" | "ssbfgs" | "bfgs".
    """

    def _reset(self, state):
        p0 = self._params[0]
        state["H"] = torch.eye(self._numel(), dtype=p0.dtype, device=p0.device)

    def _apply_H(self, state, v):
        return state["H"] @ v

    def _push(self, state, s, y, Hy, tau, phi):
        H = state["H"]
        yHy, sy = float(y.dot(Hy)), float(s.dot(y))
        v = s / sy - Hy / yHy
        # in-place rank-1 updates (no n x n temporaries); H stays symmetric
        H.addr_(Hy, Hy, alpha=-1.0 / yHy)
        H.addr_(v, v, alpha=phi * yHy)
        H.div_(tau)
        H.addr_(s, s, alpha=1.0 / sy)


DenseSSBroyden = SSBroyden  # backward-compatible name


# ---------------------------------------------------------------------------
# Limited-memory implementation (Gram-space compact form)
# ---------------------------------------------------------------------------
class LSSBroyden(_SSBroydenBase):
    """Limited-memory SSBroyden.

    Args: as torch.optim.LBFGS (lr, max_iter, max_eval, tolerance_grad,
    tolerance_change, history_size, line_search_fn; default
    line_search_fn="strong_wolfe"), plus
      update : "ssbroyden" (default) | "ssbfgs" | "bfgs"
      h0     : "latest" (default; gamma = s^T y / y^T y of the newest pair, as LBFGS),
               "bb1" (experimental; gamma = s^T s / s^T y of the newest pair),
               or a positive number (experimental; fixed gamma)
               | "identity" (H0 = I; reproduces SSBroyden while nothing is dropped)
    """

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-7,
                 tolerance_change=1e-9, history_size: int = 20,
                 line_search_fn="strong_wolfe", update="ssbroyden", h0: str | float = "latest",
                 line_search_tolerance: float = 1e-9, curvature_guard: str = "relative",
                 tau_beta: float = 0.0):
        numeric = isinstance(h0, (int, float)) and not isinstance(h0, bool)
        if not ((numeric and h0 > 0) or (not numeric and h0 in ("latest", "identity", "bb1"))):
            raise ValueError("h0 must be 'latest', 'identity', 'bb1' or a positive number")
        super().__init__(params, lr=lr, max_iter=max_iter, max_eval=max_eval,
                         tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                         line_search_fn=line_search_fn, update=update,
                         line_search_tolerance=line_search_tolerance,
                         curvature_guard=curvature_guard, tau_beta=tau_beta,
                         history_size=history_size, h0=h0)

    # state layout (all on the first parameter's state):
    #   S, Y   : (m, n) ring buffers (param device/dtype)
    #   gram   : (2m, 2m) Gram of [S; Y] in slot layout
    #   order  : list of slots, oldest -> newest
    #   taus, phis : lists (per slot)
    #   gamma, c   : floats;  A : (2m, 2m) compact matrix in slot layout
    def _reset(self, state):
        m = self.param_groups[0]["history_size"]
        p0, n = self._params[0], self._numel()
        state["S"] = p0.new_zeros(m, n)
        state["Y"] = p0.new_zeros(m, n)
        state["gram"] = torch.zeros(2 * m, 2 * m, dtype=torch.float64)
        state["A"] = torch.zeros(2 * m, 2 * m, dtype=torch.float64)
        state["order"] = []
        state["taus"] = [1.0] * m
        state["phis"] = [1.0] * m
        h0 = self.param_groups[0]["h0"]
        state["gamma"] = float(h0) if isinstance(h0, (int, float)) else 1.0
        state["c"] = 1.0

    def _apply_H(self, state, v):
        if not state["order"]:
            return v.clone()
        S, Y = state["S"], state["Y"]
        PTv = torch.cat([S @ v, Y @ v])  # <- all_reduce here under sharding
        A = state["A"].to(device=v.device, dtype=v.dtype)
        w = A @ PTv
        m = S.shape[0]
        return (state["c"] * state["gamma"]) * v + w[:m] @ S + w[m:] @ Y

    def _push(self, state, s, y, Hy, tau, phi):
        group = self.param_groups[0]
        S, Y, order = state["S"], state["Y"], state["order"]
        m = S.shape[0]
        slot = len(order) if len(order) < m else order.pop(0)
        S[slot].copy_(s)
        Y[slot].copy_(y)
        gram = state["gram"].to("cpu", torch.float64)
        rs = torch.cat([S @ s, Y @ s]).to("cpu", torch.float64)
        ry = torch.cat([S @ y, Y @ y]).to("cpu", torch.float64)
        gram[slot, :], gram[:, slot] = rs, rs
        gram[m + slot, :], gram[:, m + slot] = ry, ry
        state["gram"] = gram
        order.append(slot)
        state["taus"][slot] = tau
        state["phis"][slot] = max(phi, 0.0)  # keep H positive definite under truncation
        if group["h0"] == "latest":
            state["gamma"] = float(s.dot(y)) / float(y.dot(y))
        elif group["h0"] == "bb1":         # experimental: s^T s / s^T y (>= s^T y / y^T y)
            state["gamma"] = float(s.dot(s)) / float(s.dot(y))
        self._rebuild(state)

    @staticmethod
    def _rebuild(state):
        """Replay the window of updates in Gram space: O(m^3), CPU float64.

        Each update  A <- (A - u u^T/yu + phi yu w w^T)/tau + e e^T/sy,  w = e/sy - u/yu,
        is expanded as one symmetric rank-1 term in u plus a row/column correction at the
        slot of s_i, and the division by tau is carried in a scalar.  This needs one mat-vec
        and one rank-1 update per stored pair (NumPy/BLAS), about 4x faster than the direct
        form for m = 100.
        """
        m = state["S"].shape[0]
        G = state["gram"].to("cpu", torch.float64).numpy()
        gam = float(state["gamma"])
        taus, phis = state["taus"], state["phis"]
        B = np.zeros((2 * m, 2 * m), order="F")
        sc, c = 1.0, 1.0                     # A = sc * B
        for slot in state["order"]:
            tau, phi = float(taus[slot]), float(phis[slot])
            Gy = G[:, m + slot]
            u = sc * (B @ Gy)
            u[m + slot] += c * gam           # u = H_i y_i in basis P
            yu = float(Gy @ u)
            sy = float(G[slot, m + slot])
            if not (yu > 0.0 and sy > 0.0):  # underflow at (near-)exact convergence: skip the pair
                continue
            B = _ger((phi - 1.0) / yu / sc, u, B)
            k = phi / sy / sc
            B[slot, :] -= k * u
            B[:, slot] -= k * u
            sy2 = sy ** 2
            B[slot, slot] += (phi * yu / sy2 if sy2 > 0.0 else phi * (yu / sy) / sy) / sc
            sc /= tau
            c /= tau
            B[slot, slot] += 1.0 / sy / sc
        A = sc * B
        state["A"], state["c"] = torch.from_numpy(0.5 * (A + A.T)), c
