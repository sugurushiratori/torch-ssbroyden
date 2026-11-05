"""Tests for ssbroyden.py (run: python test_equivalence.py, or pytest).

1. LSSBroyden(history_size >= #iters, h0="identity") == SSBroyden (dense), iterate by iterate
2. LSSBroyden(update="bfgs", h0="latest") == torch.optim.LBFGS(line_search_fn="strong_wolfe")
3. state_dict() / load_state_dict() round trip continues the run exactly
"""
import copy

import torch

from torch_ssbroyden import LSSBroyden, SSBroyden

torch.set_default_dtype(torch.float64)
TOL = dict(tolerance_grad=0.0, tolerance_change=0.0)


def rosenbrock(n=10):
    x = torch.nn.Parameter(torch.full((n,), -1.2))
    return [x], lambda: (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


def mlp():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.Tanh(), torch.nn.Linear(10, 1))
    X = torch.linspace(-1, 1, 64)[:, None]
    Y = torch.sin(3 * X)
    return list(net.parameters()), lambda: ((net(X) - Y) ** 2).mean()


def run(make_opt, problem, iters, chunk=None):
    params, f = problem()
    opt = make_opt(params)

    def closure():
        opt.zero_grad()
        loss = f()
        loss.backward()
        return loss

    for _ in range(iters // (chunk or iters)):
        opt.step(closure)
    return torch.cat([p.detach().reshape(-1) for p in params]), f().item()


def rel(a, b):
    return ((a - b).norm() / a.norm()).item()


def test_lssbroyden_matches_dense():
    K = 40
    for update in ["bfgs", "ssbfgs", "ssbroyden"]:
        for name, prob in [("rosenbrock", rosenbrock), ("mlp", mlp)]:
            xd, fd = run(lambda p: SSBroyden(p, max_iter=K, max_eval=10 * K, update=update, **TOL), prob, K)
            xl, fl = run(lambda p: LSSBroyden(p, max_iter=K, max_eval=10 * K, update=update,
                                              history_size=K + 5, h0="identity", **TOL), prob, K)
            e = rel(xd, xl)
            print(f"[dense vs limited | {update:9s} | {name:10s}] rel.diff={e:.1e}  f={fd:.3e}/{fl:.3e}")
            assert e < 1e-5  # round-off growth along the (ill-conditioned) Rosenbrock path


def test_bfgs_matches_torch_lbfgs():
    K, m = 60, 10
    for name, prob in [("rosenbrock", rosenbrock), ("mlp", mlp)]:
        xt, ft = run(lambda p: torch.optim.LBFGS(p, max_iter=K, max_eval=10 * K, history_size=m,
                                                 line_search_fn="strong_wolfe", **TOL), prob, K)
        xl, fl = run(lambda p: LSSBroyden(p, max_iter=K, max_eval=10 * K, history_size=m,
                                          update="bfgs", h0="latest", curvature_guard="absolute", **TOL), prob, K)
        e = rel(xt, xl)
        print(f"[torch.optim.LBFGS vs LSSBroyden(bfgs) | {name:10s}] rel.diff={e:.1e}  f={ft:.3e}/{fl:.3e}")
        assert e < 1e-5  # two-loop recursion vs compact form: round-off only


def test_state_dict_roundtrip():
    params, f = mlp()
    opt = LSSBroyden(params, max_iter=10, history_size=5, **TOL)

    def closure_for(o, fn):
        def c():
            o.zero_grad(); l = fn(); l.backward(); return l
        return c

    opt.step(closure_for(opt, f))
    sd = copy.deepcopy(opt.state_dict())
    snap = [p.detach().clone() for p in params]
    opt.step(closure_for(opt, f))  # uninterrupted
    x_ref = torch.cat([p.detach().reshape(-1) for p in params])

    params2, f2 = mlp()
    with torch.no_grad():
        for p, s in zip(params2, snap):
            p.copy_(s)
    opt2 = LSSBroyden(params2, max_iter=10, history_size=5, **TOL)
    opt2.load_state_dict(sd)
    opt2.step(closure_for(opt2, f2))
    x_new = torch.cat([p.detach().reshape(-1) for p in params2])
    e = rel(x_ref, x_new)
    print(f"[state_dict round trip] rel.diff={e:.1e}")
    assert e < 1e-12


if __name__ == "__main__":
    test_lssbroyden_matches_dense()
    test_bfgs_matches_torch_lbfgs()
    test_state_dict_roundtrip()
    print("OK")


def test_h0_options():
    """h0="identity" equals the fixed value 1.0; every option decreases the MLP loss."""
    K = 60
    x1, _ = run(lambda p: LSSBroyden(p, max_iter=K, history_size=10, h0="identity", **TOL), mlp, K)
    x2, _ = run(lambda p: LSSBroyden(p, max_iter=K, history_size=10, h0=1.0, **TOL), mlp, K)
    assert rel(x1, x2) < 1e-12
    f0 = mlp()[1]().item()
    for h0 in ["latest", "identity", "bb1", 0.1]:
        _, f = run(lambda p: LSSBroyden(p, max_iter=K, history_size=10, h0=h0, **TOL), mlp, K)
        print(f"[h0={h0}] f={f:.3e}")
        assert f < 0.1 * f0
    try:
        LSSBroyden(mlp()[0], h0=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative h0 must raise")


def test_nonfinite_trial_points_are_rejected():
    """NaN losses at some trial points: the strong Wolfe search must back off, not accept them."""
    import torch
    from torch_ssbroyden import LSSBroyden, SSBroyden
    A = torch.diag(torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64))
    for cls in (LSSBroyden, SSBroyden):
        x = torch.nn.Parameter(torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64))
        opt = cls([x], lr=1.0, max_iter=20, tolerance_grad=0.0, tolerance_change=0.0)
        calls = [0]

        def closure():
            opt.zero_grad()
            calls[0] += 1
            f = 0.5 * x @ A @ x + (x ** 4).sum()
            if calls[0] in (4, 5, 6):
                f = f * float("nan")
            f.backward()
            return f

        opt.step(closure)
        assert torch.isfinite(x).all()
        assert float(x.detach().abs().max()) < 1e-4
        assert opt.state[x]["n_nonfinite"] >= 2
