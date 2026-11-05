"""Point samplers for segment-wise training (torch_ssbroyden.SegmentTrainer).

A *sampler* draws the collocation (training) points of the next segment.  The segment method
keeps every set of points for many iterations, so any rule that changes the points between
segments can be used with a quasi-Newton optimizer: plain resampling (mini-batches), or the
residual-based adaptive methods of the PINN literature.

  * `Uniform`  : a fresh sample from ``sample_fn`` (default of SegmentTrainer).
  * `RAD`      : residual-based adaptive distribution (Wu et al. 2023): resample all points
                 from candidates with probability  p(x) ~ |r(x)|^k / mean|r|^k + c.
  * `RARD`     : residual-based adaptive refinement with distribution (Wu et al. 2023):
                 keep the current points and add new ones drawn from the same density
                 (k = 2, c = 0 by default).  ``greedy=True`` adds the points with the largest
                 residuals instead (RAR, Lu et al. 2021).

Batches may be a tensor of points (N, d) or a dict / tuple / list of such tensors (e.g.
``{"pde": X_int, "bc": X_bc}``).  ``residual_fn(batch)`` must return per-point residuals with
the same structure (a tensor (N,) or (N, k) per adapted component).  ``keys`` selects the
components that are adapted (dict keys or tuple indices; default: all); the other components
are resampled uniformly.

Any callable ``sampler(n, generator, trainer) -> batch`` can be passed to SegmentTrainer, so
other strategies (importance sampling, curriculum, a data loader) need no change in torch_ssbroyden.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

__all__ = ["Uniform", "RAD", "RARD", "batch_size"]


# ---------------------------------------------------------------------------------------
# helpers for tensor / dict / tuple batches
# ---------------------------------------------------------------------------------------
def _items(batch):
    if isinstance(batch, torch.Tensor):
        return [(None, batch)]
    if isinstance(batch, dict):
        return list(batch.items())
    if isinstance(batch, (tuple, list)):
        return list(enumerate(batch))
    raise TypeError(f"unsupported batch type {type(batch).__name__}")


def _rebuild(batch, new: dict):
    if isinstance(batch, torch.Tensor):
        return new[None]
    if isinstance(batch, dict):
        return {k: new[k] for k in batch}
    return type(batch)(new[i] for i in range(len(batch)))


def batch_size(batch) -> int:
    """Number of points in a batch (sum over its components)."""
    return sum(int(x.shape[0]) for _, x in _items(batch))


def _magnitude(r: torch.Tensor) -> torch.Tensor:
    r = r.detach()
    return r.abs() if r.dim() == 1 else r.reshape(r.shape[0], -1).norm(dim=1)


def _density(r: torch.Tensor, k: float, c: float) -> torch.Tensor:
    e = _magnitude(r).double() ** k
    w = e / e.mean().clamp_min(1e-300) + c
    return torch.nan_to_num(w, nan=0.0, posinf=0.0).clamp_min(0.0) + 1e-300


def _residuals(residual_fn, batch):
    with torch.enable_grad():               # PINN residuals differentiate w.r.t. inputs
        r = residual_fn(batch)
    return dict(_items(r)) if not isinstance(r, torch.Tensor) else {None: r}


# ---------------------------------------------------------------------------------------
class Uniform:
    """Fresh points from ``sample_fn(n, generator)`` for every segment."""
    respects_n = True

    def __init__(self, sample_fn: Callable[[int, torch.Generator], Any]):
        self.sample_fn = sample_fn
        self.last_cost = 0

    def __call__(self, n, generator, trainer=None):
        self.last_cost = 0
        return self.sample_fn(n, generator)


class RAD:
    """Residual-based adaptive distribution: p(x) ~ |r(x)|^k / mean|r|^k + c.

    For every segment, ``factor * n`` candidates are drawn uniformly and the points of the
    adapted components are chosen among them without replacement.  ``k = 1, c = 1`` is the
    default recommended by Wu et al. (2023); ``c -> inf`` recovers uniform sampling.
    ``last_cost`` is the number of candidate points whose residual was evaluated (forward
    and input derivatives only, no parameter gradient).
    """
    respects_n = True

    def __init__(self, sample_fn, residual_fn, k: float = 1.0, c: float = 1.0,
                 factor: int = 10, keys: Sequence | None = None):
        self.sample_fn, self.residual_fn = sample_fn, residual_fn
        self.k, self.c, self.factor, self.keys = k, c, factor, keys
        self.last_cost = 0

    def __call__(self, n, generator, trainer=None):
        template = self.sample_fn(n, generator)
        cand = self.sample_fn(self.factor * n, generator)
        r = _residuals(self.residual_fn, cand)
        new = dict(_items(template))
        cand_items = dict(_items(cand))
        self.last_cost = 0
        for key, x in _items(template):
            if (self.keys is not None and key not in self.keys) or key not in r:
                continue
            w = _density(r[key], self.k, self.c)
            m = min(int(x.shape[0]), int(w.numel()))
            idx = torch.multinomial(w, m, replacement=False, generator=generator)
            new[key] = cand_items[key][idx.to(cand_items[key].device)]
            self.last_cost += int(cand_items[key].shape[0])
        return _rebuild(template, new)


class RARD:
    """Residual-based adaptive refinement: keep the points and add new ones.

    The first call returns ``sample_fn(n)``.  Every later call adds, to each adapted
    component, ``n_add`` points (an int, or a fraction of the current size if < 1) chosen from
    ``factor * n_add`` uniform candidates, with probability ~ |r|^k / mean|r|^k + c
    (RAR-D; default k = 2, c = 0) or, if ``greedy``, the largest residuals (RAR).  The batch
    grows, so SegmentTrainer does not double ``n`` for this sampler.  ``n_max`` caps the size
    of each adapted component.
    """
    respects_n = False

    def __init__(self, sample_fn, residual_fn, n_add: float = 0.1, k: float = 2.0,
                 c: float = 0.0, factor: int = 10, greedy: bool = False,
                 keys: Sequence | None = None, n_max: int | None = None):
        self.sample_fn, self.residual_fn = sample_fn, residual_fn
        self.n_add, self.k, self.c, self.factor = n_add, k, c, factor
        self.greedy, self.keys, self.n_max = greedy, keys, n_max
        self.batch = None
        self.last_cost = 0

    def __call__(self, n, generator, trainer=None):
        self.last_cost = 0
        if self.batch is None:
            self.batch = self.sample_fn(n, generator)
            return self.batch
        cur = dict(_items(self.batch))
        adds = {}
        for key, x in cur.items():
            if self.keys is not None and key not in self.keys:
                continue
            m = int(self.n_add if self.n_add >= 1 else round(self.n_add * x.shape[0]))
            if self.n_max is not None:
                m = min(m, self.n_max - int(x.shape[0]))
            if m > 0:
                adds[key] = m
        if adds:
            n_cand = self.factor * max(adds.values())
            cand = self.sample_fn(n_cand, generator)
            cand_items = dict(_items(cand))
            r = _residuals(self.residual_fn, cand)
            for key, m in adds.items():
                if key not in r:
                    continue
                pool = cand_items[key]
                m = min(m, int(pool.shape[0]))
                if self.greedy:
                    idx = torch.topk(_magnitude(r[key]), m).indices
                else:
                    idx = torch.multinomial(_density(r[key], self.k, self.c), m,
                                            replacement=False, generator=generator)
                cur[key] = torch.cat([cur[key], pool[idx.to(pool.device)]])
                self.last_cost += int(pool.shape[0])
        self.batch = _rebuild(self.batch, cur)
        return self.batch
