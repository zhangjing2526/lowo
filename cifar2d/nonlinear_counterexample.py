"""Nonlinearity counterexample for the absorption fold.

The fold  W(R z0) = (WR) z0 = W' z0  is the associativity of two linear
maps.  Insert a pointwise nonlinearity between them and it fails:

    W sigma(R z0)  =?=  W' sigma(z0)   for all z0.

Rotations do not commute with a pointwise nonlinearity, so in general no
fixed W' exists.  This script measures how badly it fails, using the real
Q and the real mixer W taken from an audited dense-mixer site of the
trained A5 checkpoints (2D / CIFAR-10 probe), and reports

  * relative residual of the least-squares-OPTIMAL W'  (best possible W',
    unrestricted -- so the residual is a lower bound on the failure),
  * the linear control (sigma removed): residual at float round-off,
  * the Q = I control (nonlinearity present, no rotation): residual 0,
    showing the failure comes from non-commutation, not from the fit.

Inputs z0: (a) i.i.d. Gaussian, the same protocol the paper uses for the
per-site patch check; (b) real activations at that site on CIFAR-10 test
images (Haar coefficients of the true layer input).

Usage:
  python nonlinear_counterexample.py --arm A5 --all-seeds
  python nonlinear_counterexample.py --arm A5 --seed 123 --samples 20000
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lowo2d import HaarWavelet2D  # noqa: E402
from resnet_lowo import build_model  # noqa: E402

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
HAAR_THETA = 0.7853981633974483  # pi/4


def q_of(wave: HaarWavelet2D) -> torch.Tensor:
    """Learned SO(4) factor Q (identity for a fixed theta-only bank)."""
    if hasattr(wave, "skew"):
        a = wave.skew.detach()
        return torch.matrix_exp(a - a.transpose(0, 1))
    return torch.eye(4, device=wave.theta.device, dtype=wave.theta.dtype)


def haar_kernel(wave: HaarWavelet2D) -> torch.Tensor:
    """K_haar(theta) in R^{4x4}, without the learned Q."""
    theta = wave.theta.detach()
    c, s = torch.cos(theta), torch.sin(theta)
    f = torch.stack(
        [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-2
    )
    return torch.einsum("ah,bw->abhw", f[0], f[1]).reshape(4, 4)


def apply_r(z: torch.Tensor, q: torch.Tensor, c: int) -> torch.Tensor:
    """z -> (Q (x) I_C) z in subband-major order (index s*C + c)."""
    b = z.shape[0]
    zz = z.reshape(b, 4, c)
    return torch.einsum("st,btc->bsc", q, zz).reshape(b, 4 * c)


def best_linear_fit(b: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Least-squares-optimal W' minimizing ||W' B - Y||_F, and relative residual."""
    # Fit in float64: the linear control must sit at round-off, not at the
    # float32 noise floor of the normal equations.
    bd, yd = b.double(), y.double()
    gram = bd.t() @ bd
    gram = gram + 1e-12 * torch.eye(gram.shape[0], dtype=gram.dtype)
    w = torch.linalg.solve(gram, bd.t() @ yd).t()  # (m, d)
    res = (w @ bd.t()).t() - yd
    rel = (res.norm() / yd.norm().clamp_min(1e-30)).item()
    return w.float(), rel


def load_test(data_dir: str, limit: int | None = None) -> torch.Tensor:
    path = os.path.join(data_dir, "cifar-10-batches-py", "test_batch")
    if not os.path.exists(path):
        raise SystemExit(f"missing CIFAR-10 test batch: {path}")
    with open(path, "rb") as fh:
        batch = pickle.load(fh, encoding="latin1")
    x = torch.tensor(batch["data"], dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x[:limit] if limit else x


def real_activations(model, site, x: torch.Tensor, cap: int = 4096) -> torch.Tensor:
    """z0 = Haar coefficients (Q = I) of the TRUE layer input at `site`.

    A pre-hook records what the site actually receives; every spatial
    position of every image is one sample, so the result is (B*H*W, 4C).
    """
    fixed = copy.deepcopy(site.wave)
    if hasattr(fixed, "skew"):
        fixed.skew.data.zero_()
    fixed.eval()

    captured = []
    handle = site.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach()))
    with torch.no_grad():
        for i in range(0, min(len(x), cap), 256):
            model(x[i : i + 256])
    handle.remove()

    zs = []
    with torch.no_grad():
        for inp in captured:
            z = fixed.dwt(inp)  # (b, 4C, H, W)
            zs.append(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]))
    return torch.cat(zs, 0)


def run_one(ckpt_path: str, data_dir: str, samples: int, cap_real: int) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    arm, seed = ck.get("arm"), ck.get("seed")
    model = build_model(arm)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    sites = [m for m in model.modules() if m.__class__.__name__ == "HaarDownsample2D"]
    if not sites:
        raise SystemExit(f"no HaarDownsample2D sites in {ckpt_path}")

    print(f"\n=== {os.path.basename(ckpt_path)}  arm={arm} seed={seed} ===")
    print(f"    sites: {len(sites)}")

    # pick the dense-mixer (groups=1) site with the largest drift -- the ones
    # the audit declares removable (rho = 0), i.e. the fold DOES hold there
    # in the linear case.
    cand = []
    for idx, s in enumerate(sites):
        if s.mix.groups != 1:
            continue
        q = q_of(s.wave)
        cand.append((float((q - torch.eye(4)).norm()), idx, s))
    if not cand:
        raise SystemExit("no groups=1 site found")
    cand.sort(key=lambda t: -t[0])
    drift, idx, site = cand[0]

    q = q_of(site.wave)
    w = site.mix.weight.detach().flatten(1)  # (out, 4C)
    c = site.mix.in_channels // 4
    print(f"    site #{idx}: C={c}, out={w.shape[0]}, ||Q-I||_F={drift:.4f}")

    # ---- inputs -------------------------------------------------------
    d = 4 * c
    g = torch.Generator().manual_seed(0)
    zg = torch.randn(samples, d, generator=g)
    try:
        xt = load_test(data_dir, limit=cap_real)
        zr = real_activations(model, site, xt, cap=cap_real)
        if zr.shape[0] > samples:
            perm = torch.randperm(zr.shape[0], generator=g)[:samples]
            zr = zr[perm]
    except SystemExit:
        zr = None

    for tag, z0 in (("gaussian", zg), ("real-act", zr)):
        if z0 is None:
            continue
        n = z0.shape[0]
        # everything below is float64 so the linear control sits at round-off
        wd, qd, z0d = w.double(), q.double(), z0.double()
        rq = apply_r(z0d, qd, c)
        rq0 = apply_r(z0d, torch.eye(4, dtype=z0d.dtype), c)  # Q = I control

        # (1) linear control: sigma removed
        y = (wd @ rq.t()).t()
        _, rel_lin = best_linear_fit(z0d, y)

        # (2) ReLU inserted between R and W
        yr = (wd @ torch.relu(rq).t()).t()
        _, rel_nl = best_linear_fit(torch.relu(z0d), yr)

        # (3) Q = I control: nonlinearity present, no rotation
        y0 = (wd @ torch.relu(rq0).t()).t()
        _, rel_id = best_linear_fit(torch.relu(z0d), y0)

        print(
            f"    [{tag:<9}] n={n:6d}  "
            f"linear {rel_lin:.3e}   ReLU(Q) {rel_nl:.4f}   "
            f"ReLU(Q=I) {rel_id:.3e}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A5")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--cap-real", type=int, default=4096)
    ap.add_argument(
        "--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    )
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    seeds = (123, 456, 789) if args.all_seeds else (args.seed,)
    for sd in seeds:
        p = os.path.join(root, f"{args.arm.lower()}_s{sd}", "best.pt")
        if not os.path.exists(p):
            print(f"missing {p}")
            continue
        run_one(p, args.data_dir, args.samples, args.cap_real)


if __name__ == "__main__":
    main()
