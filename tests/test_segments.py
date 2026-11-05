import math

import torch

from torch_ssbroyden import LSSBroyden, SegmentTrainer

torch.set_default_dtype(torch.float64)


def _mlp(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1, 20), torch.nn.Tanh(),
                               torch.nn.Linear(20, 20), torch.nn.Tanh(),
                               torch.nn.Linear(20, 1))


def test_new_batch_skips_one_pair():
    net = _mlp()
    x = torch.linspace(-1, 1, 64)[:, None]
    opt = LSSBroyden(net.parameters(), history_size=50, max_iter=5)

    def closure():
        opt.zero_grad()
        loss = ((net(x) - torch.sin(3 * x)) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    st = opt.state[opt._params[0]]
    n0 = len(st["order"]) + st.get("n_skipped", 0)
    for g in opt.param_groups:
        g["max_iter"] = 1
    opt.new_batch()
    opt.step(closure)                      # its single iteration must not form a pair
    assert len(st["order"]) + st.get("n_skipped", 0) == n0
    opt.step(closure)                      # the next one does
    assert len(st["order"]) + st.get("n_skipped", 0) == n0 + 1


def test_segment_trainer_regression():
    net = _mlp()

    def sample(n, gen):
        return 2 * torch.rand(n, 1, generator=gen) - 1

    def loss_fn(xb):
        return ((net(xb) - torch.sin(3 * xb)) ** 2).mean()

    xv = torch.linspace(-1, 1, 1001)[:, None]
    f0 = float(loss_fn(xv))
    opt = LSSBroyden(net.parameters(), history_size=20)
    tr = SegmentTrainer(opt, loss_fn, sample, n0=16, n_max=512, m_min=50)
    tr.run(max_evals=3000)
    f1 = float(loss_fn(xv))
    assert math.isfinite(f1) and f1 < 1e-4 * f0
    assert len(tr.segments) >= 2
    assert tr.n > 16                       # 16 points overfit, so the batch must grow


# --- samplers (RAD / RAR-D) and the dense optimizer ------------------------------------
from torch_ssbroyden import SSBroyden  # noqa: E402
from torch_ssbroyden.sampling import RAD, RARD, batch_size  # noqa: E402


def _peak_problem(net):
    """Fit a sharp peak on [-1, 1] (dict batch: "int" adapted, "end" fixed)."""
    target = lambda x: torch.exp(-((x - 0.3) / 0.05) ** 2)  # noqa: E731

    def sample(n, gen):
        return {"int": 2 * torch.rand(n, 1, generator=gen) - 1,
                "end": torch.tensor([[-1.0], [1.0]])}

    def residual(b):
        return {"int": (net(b["int"]) - target(b["int"]))[:, 0]}

    def loss_fn(b):
        return ((net(b["int"]) - target(b["int"])) ** 2).mean() + (net(b["end"]) ** 2).mean()

    return sample, residual, loss_fn, target


def test_rad_concentrates_points():
    net = _mlp()
    sample, residual, _, _ = _peak_problem(net)
    with torch.no_grad():                   # make the residual large only near the peak
        for p in net.parameters():
            p.zero_()
    g = torch.Generator().manual_seed(0)
    b = RAD(sample, residual, k=2.0, c=0.0, keys=["int"])(400, g)
    assert b["int"].shape == (400, 1) and b["end"].shape == (2, 1)
    frac = float(((b["int"] - 0.3).abs() < 0.15).double().mean())
    assert frac > 0.6                       # uniform would give 0.15


def test_rard_grows_and_trains():
    net = _mlp()
    sample, residual, loss_fn, target = _peak_problem(net)
    sampler = RARD(sample, residual, n_add=32, keys=["int"])
    opt = LSSBroyden(net.parameters(), history_size=20)
    tr = SegmentTrainer(opt, loss_fn, sample, n0=64, sampler=sampler, theta=None,
                        segment_iters=100, probe=False)
    tr.run(max_evals=1500)
    assert batch_size(tr.batch) > 64 + 2
    xv = torch.linspace(-1, 1, 2001)[:, None]
    err = float(((net(xv) - target(xv)) ** 2).mean())
    assert math.isfinite(err) and err < 1e-3


def test_dense_ssbroyden_segments():
    net = _mlp()

    def sample(n, gen):
        return 2 * torch.rand(n, 1, generator=gen) - 1

    def loss_fn(xb):
        return ((net(xb) - torch.sin(3 * xb)) ** 2).mean()

    xv = torch.linspace(-1, 1, 1001)[:, None]
    f0 = float(loss_fn(xv))
    opt = SSBroyden(net.parameters())
    tr = SegmentTrainer(opt, loss_fn, sample, n0=64, n_max=512, m_min=50)
    tr.run(max_evals=2000)
    assert float(loss_fn(xv)) < 1e-4 * f0 and len(tr.segments) >= 2
