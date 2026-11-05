"""2D Poisson PINN trained with L-SSBroyden on resampled collocation points.

    python examples/poisson_minibatch.py --budget 5000

-Δu = 2π² sin(πx) sin(πy) on (0,1)², u = 0 on ∂Ω.  MLP 2-40-40-40-1, float64.
Compares the segment-wise mini-batch training (SegmentTrainer, automatic segment length
and batch size) with full-batch L-SSBroyden on a fixed set of 2000+400 points, at the same
cost measured in point-gradients (budget in units of 2400-point gradients).
"""
import argparse
import time

import torch

from torch_ssbroyden import LSSBroyden, SegmentTrainer

torch.set_default_dtype(torch.float64)
pi = torch.pi


def make_net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(2, 40), torch.nn.Tanh(),
                               torch.nn.Linear(40, 40), torch.nn.Tanh(),
                               torch.nn.Linear(40, 40), torch.nn.Tanh(),
                               torch.nn.Linear(40, 1))


def sample(n, gen):
    """n interior points and n/4 boundary points."""
    xi = torch.rand(n, 2, generator=gen)
    nb = max(n // 4, 1)
    t = torch.rand(nb, generator=gen)
    side = torch.randint(0, 4, (nb,), generator=gen)
    xb = torch.stack([torch.where(side == 0, 0.0, torch.where(side == 1, 1.0, t)),
                      torch.where(side == 2, 0.0, torch.where(side == 3, 1.0, t))], 1)
    return xi, xb


def pinn_loss(net, batch):
    xi, xb = batch
    xi = xi.clone().requires_grad_(True)
    u = net(xi)
    gu = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
    lap = sum(torch.autograd.grad(gu[:, j].sum(), xi, create_graph=True)[0][:, j] for j in range(2))
    f = 2 * pi ** 2 * torch.sin(pi * xi[:, 0]) * torch.sin(pi * xi[:, 1])
    return ((-lap - f) ** 2).mean() + (net(xb) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=5000, help="in 2400-point gradients")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    val = sample(4000, torch.Generator().manual_seed(12345))
    budget_points = int(a.budget * 2000)          # interior points (boundary = 1/4 extra)

    # mini-batch: segments with automatic length and batch size
    net = make_net(a.seed)
    opt = LSSBroyden(net.parameters(), history_size=20)
    tr = SegmentTrainer(opt, lambda b: pinn_loss(net, b), sample, n0=256, n_max=16000,
                        generator=torch.Generator().manual_seed(1000 + a.seed))
    t0 = time.perf_counter()
    tr.run(max_point_evals=budget_points)
    with torch.enable_grad():
        v = float(pinn_loss(net, val).detach())
    print(f"segments : val loss {v:.2e}  segments {len(tr.segments)}  final batch {tr.n}  "
          f"time {time.perf_counter() - t0:.0f}s")

    # full batch on a fixed set
    net = make_net(a.seed)
    data = sample(2000, torch.Generator().manual_seed(1000 + a.seed))
    opt = LSSBroyden(net.parameters(), history_size=20, max_iter=50, max_eval=70)
    evals = 0

    def closure():
        nonlocal evals
        evals += 1
        opt.zero_grad()
        loss = pinn_loss(net, data)
        loss.backward()
        return loss

    t0 = time.perf_counter()
    while evals * 2000 < budget_points:
        before = evals
        opt.step(closure)
        if evals == before:
            break
    with torch.enable_grad():
        v = float(pinn_loss(net, val).detach())
    print(f"fixed set: val loss {v:.2e}  time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
