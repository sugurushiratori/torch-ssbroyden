# torch-ssbroyden — self-scaled Broyden optimizers for PyTorch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23998783.svg)](https://doi.org/10.5281/zenodo.23998783)

`torch-ssbroyden` provides limited-memory and dense self-scaled Broyden (SSBroyden)
quasi-Newton optimizers as `torch.optim.Optimizer` subclasses with the interface of
`torch.optim.LBFGS`, and a helper for mini-batch training with resampled collocation points.
It was written for physics-informed neural networks (PINNs) but works for any smooth loss.

(日本語の説明は [README_ja.md](README_ja.md) にあります。)

| Class | Description |
|---|---|
| `torch_ssbroyden.LSSBroyden` | Limited-memory self-scaled Broyden method, O(mn) memory. `update="bfgs"` gives L-BFGS. |
| `torch_ssbroyden.SSBroyden` | Dense SSBroyden (n×n inverse Hessian; small models only). |
| `torch_ssbroyden.SegmentTrainer` | Segment-wise mini-batch training with automatic segment length and batch size. |

The SSBroyden update and its scaling parameters follow Kiyani et al., *Comput. Methods Appl.
Mech. Engrg.* 446 (2025) 118308, Sec. 1.3. The limited-memory
version stores the scaling parameters (τ, φ) of every pair and rebuilds the compact
representation H = cγI + PAPᵀ in the 2m-dimensional Gram space.

## Installation

```bash
pip install "torch-ssbroyden @ git+https://github.com/Shiratori-Lab/torch-ssbroyden.git"          # latest
pip install "torch-ssbroyden @ git+https://github.com/Shiratori-Lab/torch-ssbroyden.git@v0.4.1"   # a release

# or from a clone
git clone https://github.com/Shiratori-Lab/torch-ssbroyden.git
pip install -e torch-ssbroyden              # PyTorch only
pip install -e "torch-ssbroyden[fast]"      # with scipy (faster L-SSBroyden rebuild)
pip install -e "torch-ssbroyden[test]"      # with the test dependencies
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1. The package is imported as `torch_ssbroyden`.

## Usage

### L-SSBroyden (same interface as `torch.optim.LBFGS`)

```python
import torch
from torch_ssbroyden import LSSBroyden
torch.set_default_dtype(torch.float64)        # recommended for high accuracy

opt = LSSBroyden(model.parameters(), history_size=100, max_iter=20)

def closure():
    opt.zero_grad()
    loss = loss_fn()
    loss.backward()
    return loss

for epoch in range(n_epochs):
    opt.step(closure)                          # up to max_iter iterations per call
```

| Argument | Default | Meaning |
|---|---|---|
| `history_size` | 20 | number m of stored pairs (s, y) |
| `update` | `"ssbroyden"` | `"ssbroyden"`, `"ssbfgs"` or `"bfgs"` |
| `h0` | `"latest"` | initial matrix of the window: γI with γ = sᵀy/yᵀy of the newest pair, `"identity"` (γ = 1), `"bb1"` (γ = sᵀs/sᵀy of the newest pair) or a positive number (fixed γ) |
| `curvature_guard` | `"relative"` | skip an update if sᵀy ≤ ε‖s‖‖y‖; `"absolute"` uses sᵀy ≤ 1e-10 as `torch.optim.LBFGS` |
| `line_search_fn` | `"strong_wolfe"` | or `None` for a fixed step `lr` |
| `tau_beta` | 0.0 | experimental variant of the τ rule; 0 is the published rule |
| others | | `lr`, `max_iter`, `max_eval`, `tolerance_grad`, `tolerance_change` as in `torch.optim.LBFGS`, but the two tolerances default to 0 (absolute tolerances depend on the scale of the loss; stop by `max_iter` / `max_eval`) |

### Mini-batch training with resampled points (`SegmentTrainer`)

Resampling the collocation points at every iteration stalls quasi-Newton methods. The
`SegmentTrainer` keeps each batch for a segment of iterations, keeps the curvature memory
across segments, and sets the segment length and the batch size from an independent probe
batch.

```python
from torch_ssbroyden import LSSBroyden, SegmentTrainer

def sample(n, generator):        # a batch of n points, in any form loss_fn accepts
    ...

def loss_fn(batch):              # loss on a batch (do not call backward)
    ...

opt = LSSBroyden(model.parameters(), history_size=20)
trainer = SegmentTrainer(opt, loss_fn, sample, n0=4096, n_max=262144)
trainer.run(max_evals=20000)     # or max_point_evals=... (points × evaluations)
```

| Argument | Default | Meaning |
|---|---|---|
| `n0` | required | initial batch size; on a GPU, there is no gain below the size at which one evaluation saturates the device |
| `n_max` | `n0` | maximum batch size (set by memory) |
| `m_min` | 200 | minimum segment length (iterations) |
| `theta` | 0.5 | double the batch when the probe gap (f_probe − f_batch)/f_batch exceeds `theta` |
| `probe_every`, `patience`, `rel_tol` | 10, 3, 0.01 | probe every 10 iterations; end the segment when the probe loss improved by less than 1 % over the last 3 probes |
| `segment_iters` | `None` | fixed segment length (disables the stall test) |
| `sampler` | `Uniform(sample)` | how the points of the next segment are chosen (below) |
| `probe` | `True` | `False`: no probe batch (requires `segment_iters`) |

Both `LSSBroyden` and the dense `SSBroyden` work with `SegmentTrainer`.
In a custom training loop, call `opt.new_batch()` right after switching to a new batch.
The memory is kept, but no curvature pair is formed across the two batches.

See `examples/poisson_minibatch.py`.

### Adaptive residual points (RAD, RAR-D, RAR)

The segment method also makes residual-based adaptive sampling usable with quasi-Newton
methods: the points are changed only between segments.

```python
from torch_ssbroyden.sampling import RAD, RARD

def sample(n, generator):        # e.g. {"pde": X_int (n, d), "bc": X_bc (n_bc, d)}
    ...

def residual_fn(batch):          # per-point residuals of the adapted components
    return {"pde": pde_residual(model, batch["pde"])}

sampler = RAD(sample, residual_fn, k=1.0, c=1.0, keys=["pde"])        # resample by residual
sampler = RARD(sample, residual_fn, n_add=0.1, keys=["pde"])          # add 10 % per segment
sampler = RARD(sample, residual_fn, n_add=0.1, greedy=True, keys=["pde"])   # RAR
trainer = SegmentTrainer(opt, loss_fn, sample, n0=2000, sampler=sampler,
                         segment_iters=200, theta=None, probe=False)
```

`RAD` draws `factor * n` uniform candidates (default 10) and keeps `n` of them with
probability ∝ |r|^k / mean|r|^k + c (Wu et al., 2023). `RARD` keeps the current points and
adds new ones from the same density (k = 2, c = 0) or, with `greedy=True`, the largest
residuals (Lu et al., 2021). Any callable `sampler(n, generator, trainer) -> batch` can be
passed instead. `trainer.sampler_point_evals` counts the candidate points whose residual was
evaluated.

## Notes

- Use float64 for high-accuracy PINN training.
- `torch.optim.LBFGS` skips updates when sᵀy ≤ 1e-10, which stalls PINN training once the
  loss is around 1e-7. Use `LSSBroyden(update="bfgs")` for an L-BFGS with a scale-invariant test.
- Keep the line search (`line_search_fn="strong_wolfe"`). With a fixed step, SSBroyden can
  diverge even on the full point set, and changing the batch at every iteration makes it worse.
- With mini-batches or adaptive points, change the points only between segments
  (`SegmentTrainer`, or `opt.new_batch()` in your own loop): keep the memory, do not form a
  curvature pair across two batches, and make a segment long enough for the optimizer to
  converge on its batch (10²–10³ iterations). In our tests this reached the accuracy of a
  fixed point set, while a new batch at every iteration was 1.5–3 orders of magnitude worse.
- The optimizers use a copy of PyTorch's strong Wolfe line search
  (`torch_ssbroyden/_line_search.py`, BSD-3-Clause), so they do not depend on a private
  PyTorch function.

## Tests

```bash
pytest
```

## License and citation

Apache License 2.0 (the vendored line search is BSD-3-Clause, see the file header).

If you use this package, please cite the software (DOI [10.5281/zenodo.23998783](https://doi.org/10.5281/zenodo.23998783)), the
accompanying note and the SSBroyden paper (see [CITATION.cff](CITATION.cff)):

- S. Shiratori, *torch-ssbroyden: self-scaled Broyden quasi-Newton optimizers for PyTorch*,
  Zenodo, 2026. https://doi.org/10.5281/zenodo.23998783
- S. Shiratori, *L-SSBroyden: limited-memory and mini-batch self-scaled Broyden training of
  physics-informed neural networks*, arXiv, 2026.
- E. Kiyani et al., *Comput. Methods Appl. Mech. Engrg.* 446 (2025) 118308.
