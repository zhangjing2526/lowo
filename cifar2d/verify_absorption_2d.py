#!/usr/bin/env python3
"""Checkpoint-only 2D absorption audit (CPU, no CIFAR).

    python verify_absorption_2d.py runs/a6_s123/best.pt
    python verify_absorption_2d.py --synthetic

Same identity as the 3D script: dwt_Q(x) = (Q ⊗ I_C) dwt_haar(x).
A dense (groups=1) mixer absorbs Q; groups=4 leaves a residual ρ.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

DT = torch.float64


def haar_matrix(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    f = torch.stack(
        [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-2
    )
    return torch.einsum("ah,bw->abhw", f[0], f[1]).reshape(4, 4)


K_HAAR = haar_matrix(torch.full((2,), math.pi / 4, dtype=DT))


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in obj.items() if torch.is_tensor(v)}
    if not sd:
        raise SystemExit("no tensors in checkpoint")
    return sd


def q_of(sd: dict, wave: str) -> torch.Tensor:
    if f"{wave}.skew" in sd:
        a = sd[f"{wave}.skew"].to(DT)
        return torch.matrix_exp(a - a.T)
    if f"{wave}.theta" in sd:
        return haar_matrix(sd[f"{wave}.theta"].to(DT)) @ K_HAAR.T
    raise KeyError(wave)


def full_weight(w: torch.Tensor, groups: int) -> torch.Tensor:
    w = w.to(DT).reshape(w.shape[0], w.shape[1])
    if groups == 1:
        return w
    cout, cin_g = w.shape
    full = torch.zeros(cout, groups * cin_g, dtype=DT)
    cout_g = cout // groups
    for g in range(groups):
        full[g * cout_g : (g + 1) * cout_g, g * cin_g : (g + 1) * cin_g] = w[
            g * cout_g : (g + 1) * cout_g
        ]
    return full


def kron_q(q: torch.Tensor, c: int) -> torch.Tensor:
    return torch.kron(q, torch.eye(c, dtype=DT))


def rho_of(w_full: torch.Tensor, q: torch.Tensor, c: int, groups: int) -> float:
    folded = w_full @ kron_q(q, c)
    if groups == 1:
        return 0.0
    cout, cin = folded.shape
    cin_g = cin // groups
    cout_g = cout // groups
    block = torch.zeros_like(folded)
    for g in range(groups):
        block[g * cout_g : (g + 1) * cout_g, g * cin_g : (g + 1) * cin_g] = folded[
            g * cout_g : (g + 1) * cout_g, g * cin_g : (g + 1) * cin_g
        ]
    den = folded.norm().item()
    return ((folded - block).norm().item() / den) if den else 0.0


def fold_err(w_full: torch.Tensor, q: torch.Tensor, c: int) -> float:
    x = torch.randn(2, c, 8, 8, dtype=DT)
    k_q = (q @ K_HAAR).reshape(4, 1, 2, 2)
    k_h = K_HAAR.reshape(4, 1, 2, 2)

    def dwt(z, k):
        b, ch = z.shape[:2]
        y = F.conv2d(z.reshape(b * ch, 1, *z.shape[2:]), k, stride=2)
        y = y.reshape(b, ch, 4, *y.shape[2:]).permute(0, 2, 1, 3, 4)
        return y.reshape(b, 4 * ch, *y.shape[3:])

    def mix(z, w):
        return torch.einsum("oi,bihw->bohw", w, z)

    y_q = mix(dwt(x, k_q), w_full)
    y_h = mix(dwt(x, k_h), w_full @ kron_q(q, c))
    return (y_q - y_h).abs().max().item()


def discover(sd: dict) -> list[dict]:
    sites = []
    seen = set()
    for key in sd:
        if not (key.endswith("wave.skew") or key.endswith("wave.theta")):
            continue
        wave = key.rsplit(".", 1)[0]
        if wave in seen:
            continue
        seen.add(wave)
        if wave == "wave":
            prefix, mix_w = "", "mix.weight"
        else:
            prefix = wave[: -len(".wave")]
            mix_w = f"{prefix}.mix.weight"
        if mix_w not in sd:
            continue
        w = sd[mix_w]
        # Conv2d(4C, Cout, groups=G) stores (Cout, 4C/G, 1, 1).
        # G=4 → (Cout, C); G=1 → (Cout, 4C) which is wider than Cout here.
        if w.shape[1] > w.shape[0]:
            groups, c = 1, int(w.shape[1] // 4)
        else:
            groups, c = 4, int(w.shape[1])
        sites.append(dict(name=prefix, wave=wave, mix=mix_w, groups=groups, c=c))
    return sites


def audit(sd: dict) -> None:
    sites = discover(sd)
    if not sites:
        raise SystemExit("no HaarDownsample2D sites in this checkpoint (B0 has none)")
    print(f"{'site':<42} grp  ||Q-I||    rho    fold err")
    print("-" * 72)
    n_live = 0
    max_err = 0.0
    for s in sites:
        q = q_of(sd, s["wave"])
        drift = (q - torch.eye(4, dtype=DT)).norm().item()
        w_full = full_weight(sd[s["mix"]], s["groups"])
        rho = rho_of(w_full, q, s["c"], s["groups"])
        err = fold_err(w_full, q, s["c"])
        max_err = max(max_err, err)
        if rho > 1e-12:
            n_live += 1
        print(
            f"{(s['name'] or 'haar'):<42} {s['groups']:>3}  {drift:7.4f}  {rho:6.3f}  {err:.2e}"
        )
    print(
        f"\nsites {len(sites)}   rho>0 {n_live}/{len(sites)}   "
        f"max fold err {max_err:.2e}"
    )


def synthetic() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lowo2d import HaarDownsample2D

    print("A6-like  groups=4  (expect rho>0)")
    m6 = HaarDownsample2D(16, 32, learnable=True, mode="so4", grouped_mix=True)
    with torch.no_grad():
        m6.wave.skew.normal_(0, 0.4)
    audit(m6.state_dict())
    print("\nA5-like  groups=1  (expect rho=0, drift>0)")
    m5 = HaarDownsample2D(16, 32, learnable=True, mode="so4", grouped_mix=False)
    with torch.no_grad():
        m5.wave.skew.copy_(m6.wave.skew)
    audit(m5.state_dict())
    print("\nA2-like  Q=I  (expect rho=0, drift=0)")
    m2 = HaarDownsample2D(16, 32, learnable=False, mode="so4", grouped_mix=True)
    audit(m2.state_dict())
    from lowo2d import HWDOfficial2D

    print("\nHWD official mixer + learned Q  (expect rho=0, drift>0)")
    hq = HWDOfficial2D(16, 32, learnable=True)
    with torch.no_grad():
        hq.wave.skew.copy_(m6.wave.skew)
    audit(hq.state_dict())
    print("\nHWD official  Q=I  (expect rho=0, drift=0)")
    audit(HWDOfficial2D(16, 32, learnable=False).state_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic or args.ckpt is None:
        print("synthetic groups=4 site")
        synthetic()
        return
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"missing {args.ckpt}")
    audit(load_state_dict(args.ckpt))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
