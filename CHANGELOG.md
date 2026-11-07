# Changelog

## 0.4.1
- `tolerance_grad` and `tolerance_change` default to 0 in `LSSBroyden` and `SSBroyden` (were 1e-7 and
  1e-9 as in `torch.optim.LBFGS`). These absolute tolerances depend on the scale of the loss and
  can stop a run early; pass the old values explicitly to restore the previous behaviour.
- `ss_params`: evaluate 1 - sqrt(a/(1+a)) as 1/((1+a)(1+sqrt(a/(1+a)))) for a >= 1e6. The
  direct form loses accuracy for large a and is exactly 0 for a >~ 1e16, which raised
  `ZeroDivisionError`. Results for a < 1e6 are bit-identical to 0.4.0.

## 0.4.0
- Renamed from `ssopt` to `torch-ssbroyden` (import `torch_ssbroyden`). PyTorch only.
- Restart from the gradient when H is no longer a descent direction (instead of stopping);
  counted in `state["n_restarts"]`.
- The strong Wolfe line search rejects non-finite trial points (counted in
  `state["n_nonfinite"]`); degenerate pairs at round-off are skipped.
- The strong Wolfe line search is a vendored copy of PyTorch's (BSD-3-Clause) instead of an
  import of the private `torch.optim.lbfgs._strong_wolfe`.

## 0.3.0
- Segment-wise training with changing points: `SegmentTrainer` takes a pluggable
  `sampler`, a fixed `segment_iters`, `theta=None` and `probe=False`.
- Samplers in `.sampling`: `Uniform`, `RAD`, `RARD` (`greedy=True` gives RAR).
- Dense `SSBroyden` supports `new_batch()` and can be used with `SegmentTrainer`.

## 0.2.1
- `LSSBroyden(h0=...)`: `"bb1"` or a fixed positive gamma in addition to `"latest"` and
  `"identity"`.
- `numpy` dependency; `[fast]` extra (scipy BLAS rank-1 updates in the rebuild).

## 0.2.0
- `SegmentTrainer` for mini-batch training (automatic segment length and batch size).
- `new_batch()` hook on the optimizers.

## 0.1.x
- `SSBroyden`, `LSSBroyden` (compact Gram-space rebuild, relative curvature guard).
- Faster rebuild in `LSSBroyden` (rank-1 plus row/column form).
