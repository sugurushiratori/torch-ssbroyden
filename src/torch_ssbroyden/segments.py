"""Segment-wise mini-batch training for the quasi-Newton optimizers of torch_ssbroyden.

Quasi-Newton methods need a consistent objective over many iterations: resampling the
collocation points at every iteration stalls them, even with large batches.  The
`SegmentTrainer` keeps each batch for a *segment* of iterations, keeps the curvature memory
across segments, and decides the segment length and the batch size automatically from an
independent probe batch (Shiratori, "L-SSBroyden", Appendix C; ``theta`` is the threshold
beta there):

  * a segment ends when the probe loss has not improved by more than ``rel_tol`` over the
    last ``patience`` probe evaluations (one every ``probe_every`` iterations), but not
    before ``m_min`` iterations;
  * at the end of a segment the batch size is doubled (up to ``n_max``) if the
    generalisation gap measured on the probe, (f_probe - f_batch) / f_batch, exceeds
    ``theta``.

Example (PINN)::

    from torch_ssbroyden import LSSBroyden, SegmentTrainer

    def sample(n, generator):                 # returns any object your loss understands
        return draw_collocation_points(n, generator=generator)

    def loss_fn(batch):                       # loss on a batch (no backward here)
        return pinn_loss(model, batch)

    opt = LSSBroyden(model.parameters(), history_size=20, max_iter=10)
    trainer = SegmentTrainer(opt, loss_fn, sample, n0=4096, n_max=262144)
    trainer.run(max_evals=20000)

``n0`` should not be smaller than the number of points at which one evaluation saturates
the device (on a GPU the time per evaluation is nearly constant below that size).

The points of a segment come from a *sampler* (``torch_ssbroyden.sampling``).  The default draws fresh
uniform points (mini-batches); residual-based adaptive sampling works the same way::

    from torch_ssbroyden.sampling import RAD, RARD
    sampler = RAD(sample, residual_fn, k=1.0, c=1.0, keys=["pde"])   # resample by residual
    sampler = RARD(sample, residual_fn, n_add=0.1, keys=["pde"])      # add points (RAR-D)
    trainer = SegmentTrainer(opt, loss_fn, sample, n0=2000, sampler=sampler, theta=None)

Segment length is counted in calls of the optimizer: each call makes at most ``probe_every``
iterations (fewer if the per-call evaluation cap ``max_eval`` is reached, which happens for
methods that need many line-search evaluations), and ``it_in_seg`` adds ``probe_every`` per
call, so ``m_min`` and ``segment_iters`` are nominal iteration counts.

Fixed schedules: ``segment_iters=M`` ends every segment after M/probe_every calls (no
stall test), ``theta=None`` keeps the batch size (no doubling), ``n_max=None`` = ``n0``.
The optimizer can be any optimizer of torch_ssbroyden with ``new_batch()`` (LSSBroyden, SSBroyden).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .sampling import Uniform, batch_size

__all__ = ["SegmentTrainer"]


@dataclass
class SegmentRecord:
    iterations: int
    batch_size: int
    f_batch: float
    f_probe: float

    @property
    def gap(self) -> float:
        return (self.f_probe - self.f_batch) / max(abs(self.f_batch), 1e-300)


@dataclass
class SegmentTrainer:
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[Any], torch.Tensor]
    sample_fn: Callable[[int, torch.Generator], Any]
    n0: int
    n_max: int | None = None
    m_min: int = 200
    theta: float | None = 0.5
    probe_every: int = 10
    patience: int = 3
    rel_tol: float = 0.01
    probe_fraction: float = 0.5
    generator: torch.Generator | None = None
    sampler: Callable | None = None          # sampler(n, generator, trainer) -> batch
    segment_iters: int | None = None         # fixed segment length (disables the stall test)
    probe: bool = True                       # False: no probe batch (needs segment_iters)
    segments: list = field(default_factory=list)

    def __post_init__(self):
        if self.n_max is None:
            self.n_max = self.n0
        if self.generator is None:
            self.generator = torch.Generator().manual_seed(0)
        if self.sampler is None:
            self.sampler = Uniform(self.sample_fn)
        if not self.probe and self.segment_iters is None:
            raise ValueError("probe=False needs a fixed segment_iters")
        self.n = self.n0
        self.evals = 0                    # loss+gradient evaluations (closure calls)
        self.point_evals = 0              # sum of batch sizes over those evaluations
        self.sampler_point_evals = 0      # residual evaluations spent by adaptive samplers
        for g in self.optimizer.param_groups:
            g["max_iter"] = self.probe_every
            g["max_eval"] = self.probe_every * 5 // 4 + 5

    # ------------------------------------------------------------------------------------
    def _closure(self):
        self.evals += 1
        self.point_evals += self.batch_n
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.batch)
        loss.backward()
        return loss

    def _n_iter(self) -> int:
        st = self.optimizer.state.get(self.optimizer.param_groups[0]["params"][0], {})
        return int(st.get("n_iter", 0))

    def _batch_loss(self) -> float:
        with torch.enable_grad():
            return float(self.loss_fn(self.batch).detach())

    def _probe_loss(self) -> float:
        if not self.probe:
            return float("nan")
        with torch.enable_grad():         # PINN losses differentiate w.r.t. the inputs
            return float(self.loss_fn(self.probe_batch).detach())

    def _new_segment(self, first=False):
        self.batch = self.sampler(self.n, self.generator, self)
        self.batch_n = batch_size(self.batch)
        self.sampler_point_evals += getattr(self.sampler, "last_cost", 0)
        if self.probe:                    # the probe is always uniform (population loss);
            # sample_fn takes the same size argument as for the batch (self.n), not the total
            # number of points of the batch
            self.probe_batch = self.sample_fn(max(1, int(self.n * self.probe_fraction)),
                                              self.generator)
        if not first:
            self.optimizer.new_batch()
        self.it_in_seg, self.hist = 0, []

    # ------------------------------------------------------------------------------------
    def run(self, max_evals: int | None = None, max_point_evals: int | None = None,
            max_segments: int | None = None, callback: Callable | None = None):
        """Train until a budget is exhausted.  ``callback(trainer)`` runs after every
        probe evaluation (e.g. for logging); return True from it to stop."""
        if not hasattr(self, "batch"):
            self._new_segment(first=True)

        def budget_left():
            return ((max_evals is None or self.evals < max_evals)
                    and (max_point_evals is None or self.point_evals < max_point_evals)
                    and (max_segments is None or len(self.segments) < max_segments))

        while budget_left():
            it_before = self._n_iter()
            self.optimizer.step(self._closure)
            self.it_in_seg += self.probe_every
            self.hist.append(self._probe_loss())
            if callback is not None and callback(self):
                break
            stalled = (len(self.hist) > self.patience and
                       min(self.hist[-self.patience:])
                       > (1 - self.rel_tol) * min(self.hist[:-self.patience]))
            no_progress = self._n_iter() == it_before   # the optimizer made no iteration
            if self.segment_iters is not None:
                end = self.it_in_seg >= self.segment_iters
            else:
                end = self.it_in_seg >= self.m_min and stalled
            if end or no_progress:
                self.f_batch = self._batch_loss()   # batch loss at the end of the segment
                rec = SegmentRecord(self.it_in_seg, self.batch_n, self.f_batch, self.hist[-1])
                self.segments.append(rec)
                grow = (self.theta is not None and getattr(self.sampler, "respects_n", True)
                        and math.isfinite(rec.gap) and rec.gap > self.theta)
                if grow and self.n < self.n_max:
                    self.n = min(2 * self.n, self.n_max)
                self._new_segment()
        return self
