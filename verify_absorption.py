"""Verify the absorption identity behind LOWO's local-identifiability claim.

The 3D DWT in ``lowo.py`` emits subband-major channels (index ``s*C + c``), and
its analysis kernel is ``K = Q @ K_haar``.  Therefore the learned orthogonal
factor acts on the fixed-Haar subbands as a Kronecker factor::

    dwt_Q(x) = (Q kron I_C) @ dwt_haar(x)

Consequences, both checked here against real checkpoints:

  * A *dense* subband mixer (groups=1) absorbs Q exactly.  Folding
    ``W' = W @ (Q kron I_C)`` yields a network whose wavelet is bit-exact Haar
    and whose function is unchanged, so Q is locally absorbed at that site --
    however far it has drifted from the identity.
  * A *per-subband* mixer (groups=8) does not.  ``W @ (Q kron I_C)`` carries
    energy outside the block-diagonal sparsity pattern, and that off-block
    fraction rho is the distance to the set of realisable groups=8 mixers.
    rho > 0 means Q is load-bearing relative to that mixer class, not that Q
    can be uniquely recovered from the network.

The decoder's ``HaarUpsample`` applies Q after its dense expansion, so there the
fold is ``W' = (Q^T kron I_Cout) @ W`` and the bias transforms as well.

``WaveletConv3d`` is a sandwich ``idwt(W dwt(x))`` with the *same* Q on both
sides.  With ``R = Q kron I_C`` the Haar-coordinate mixer is
``W' = R^T @ W @ R``, ``b' = R^T @ b`` --- not the one-sided ``W @ R`` that
applies to a lone analysis mixer.

Absorption is assessed layer-locally, which is what matters here: the operator
immediately following each mixer is a channel-wise normalisation, so a
cross-channel linear factor cannot be commuted any further downstream.

Runs on CPU from a checkpoint alone: no data, no GPU, no model construction.
With ``--synthetic``, random Q and mixers are enough to exercise the identity
when no checkpoint is available.

    python verify_absorption.py <ckpt.pt> [--in-chans 4] [--tol 1e-10]
    python verify_absorption.py --synthetic
"""

from __future__ import annotations

import argparse
import math
import re

import torch
import torch.nn.functional as F

DT = torch.float64


# --------------------------------------------------------------------------- #
# checkpoint plumbing
# --------------------------------------------------------------------------- #

def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise SystemExit(f"unrecognised checkpoint layout: {type(obj)}")
    sd = {re.sub(r"^(module\.)+", "", k): v
          for k, v in obj.items() if torch.is_tensor(v)}
    if not sd:
        raise SystemExit("checkpoint contains no tensors")
    return sd


# --------------------------------------------------------------------------- #
# the wavelet, transcribed from lowo.py so the layout claim is actually tested
# --------------------------------------------------------------------------- #

def haar_matrix(theta: torch.Tensor) -> torch.Tensor:
    """The 8x8 separable analysis matrix for per-axis angles ``theta`` (3,)."""
    c, s = torch.cos(theta), torch.sin(theta)
    f = torch.stack([torch.stack([c, s], dim=-1),
                     torch.stack([-s, c], dim=-1)], dim=-2)  # (3, 2, 2)
    return torch.einsum("ad,bh,cw->abcdhw", f[0], f[1], f[2]).reshape(8, 8)


K_HAAR = haar_matrix(torch.full((3,), math.pi / 4, dtype=DT))


def orthogonal_factor(sd: dict[str, torch.Tensor], wave: str) -> torch.Tensor:
    """Q such that this wavelet's analysis matrix equals ``Q @ K_haar``."""
    if f"{wave}.skew" in sd:
        a = sd[f"{wave}.skew"].to(DT)
        return torch.matrix_exp(a - a.T)
    if f"{wave}.theta" in sd:
        return haar_matrix(sd[f"{wave}.theta"].to(DT)) @ K_HAAR.T
    raise KeyError(wave)


def dwt(x: torch.Tensor, k8: torch.Tensor) -> torch.Tensor:
    """(B, C, D, H, W) -> (B, 8C, D/2, H/2, W/2), subband-major."""
    b, c = x.shape[:2]
    k = k8.reshape(8, 1, 2, 2, 2)
    y = F.conv3d(x.reshape(b * c, 1, *x.shape[2:]), k, stride=2)
    y = y.reshape(b, c, 8, *y.shape[2:]).permute(0, 2, 1, 3, 4, 5)
    return y.reshape(b, 8 * c, *y.shape[3:])


def idwt(x: torch.Tensor, k8: torch.Tensor) -> torch.Tensor:
    """(B, 8C, d, h, w) -> (B, C, 2d, 2h, 2w)."""
    b, c = x.shape[0], x.shape[1] // 8
    k = k8.reshape(8, 1, 2, 2, 2)
    y = x.reshape(b, 8, c, *x.shape[2:]).permute(0, 2, 1, 3, 4, 5)
    y = F.conv_transpose3d(y.reshape(b * c, 8, *y.shape[3:]), k, stride=2)
    return y.reshape(b, c, *y.shape[2:])


def apply1x1(z: torch.Tensor, w: torch.Tensor,
             b: torch.Tensor | None) -> torch.Tensor:
    y = torch.einsum("oi,bidhw->bodhw", w, z)
    return y if b is None else y + b.reshape(1, -1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# site discovery
#
# Mixers are the reliable anchor: every LOWO site has one, whereas the wavelet
# may be shared (``wave``) or per-channel (``waves.N``).  Encoder stage widths
# cannot be read off a mixer weight alone -- [96, 48] is groups=8 on a 384-wide
# subband input, and groups=1 on a 48-wide one -- so they are chained forward
# from the number of input modalities.
# --------------------------------------------------------------------------- #

DOWN_RE = re.compile(
    r"(.*downsample_layers\.(\d+)(?:\.\d+)?)\.(mix|mixes\.\d+)\.weight$")


def _site(path, role, kind, sd, wkey, wave, c_in, groups) -> dict:
    return {"path": path, "role": role, "kind": kind, "wave": wave,
            "w": sd[wkey], "b": sd.get(wkey[:-len("weight")] + "bias"),
            "c_in": c_in, "groups": groups}


def discover(sd: dict[str, torch.Tensor], in_chans: int) -> list[dict]:
    sites: list[dict] = []
    claimed: set[str] = set()

    # ---- encoder downsampling, chained so per-channel stems don't break it --
    stages: dict[int, list[tuple[str, str, str]]] = {}
    for key in sd:
        m = DOWN_RE.match(key)
        if m:
            stages.setdefault(int(m.group(2)), []).append(
                (m.group(1), m.group(3), key))

    width = in_chans
    for idx in sorted(stages):
        entries = sorted(stages[idx], key=lambda e: e[1])
        prefix = entries[0][0]
        out_total = sum(sd[k].shape[0] for _, _, k in entries)
        shared = f"{prefix}.wave"
        if len(entries) == 1 and entries[0][1] == "mix":
            _, _, key = entries[0]
            groups, rem = divmod(8 * width, sd[key].shape[1])
            sites.append(_site(prefix, "downsample" if not rem else "downsample?",
                               "analysis", sd, key, shared, width, groups))
            claimed.add(shared)
        else:
            # per_channel: each modality gets a dense 8 -> out_c mixer
            for _, sub, key in entries:
                n = sub.split(".")[1]
                per = f"{prefix}.waves.{n}"
                wave = per if f"{per}.skew" in sd or f"{per}.theta" in sd else shared
                sites.append(_site(f"{prefix}.{sub}", "downsample/pc", "analysis",
                                   sd, key, wave, 1, 1))
                claimed.add(wave)
        width = out_total

    # ---- everything else, anchored on the remaining wavelet instances -------
    for wave in sorted({m.group(1) for m in
                        (re.match(r"(.*\.waves?(?:\.\d+)?)\.(?:skew|theta)$", k)
                         for k in sd) if m}):
        if wave in claimed:
            continue
        parent = re.sub(r"\.waves?(?:\.\d+)?$", "", wave)
        if f"{parent}.mix.weight" in sd:
            # WaveletConv3d: Conv3d(8C, 8C, 1, groups=g) -> weight [8C, 8C/g]
            w = sd[f"{parent}.mix.weight"]
            c = w.shape[0] // 8
            groups, rem = divmod(8 * c, w.shape[1])
            sites.append(_site(parent, "waveletconv" if not rem else "waveletconv?",
                               "waveletconv", sd, f"{parent}.mix.weight", wave,
                               c, groups))
        elif f"{parent}.expand.weight" in sd:
            # HaarUpsample: Conv3d(in, 8*Cout, 1), always dense
            w = sd[f"{parent}.expand.weight"]
            sites.append(_site(parent, "upsample", "synthesis", sd,
                               f"{parent}.expand.weight", wave, w.shape[0] // 8, 1))
        else:
            sites.append({"path": parent, "role": "orphan", "kind": "orphan",
                          "wave": wave, "w": None, "b": None,
                          "c_in": 0, "groups": 0})
    return sites


# --------------------------------------------------------------------------- #
# the two checks
# --------------------------------------------------------------------------- #

def dense_form(w: torch.Tensor, groups: int, c_in: int) -> torch.Tensor:
    """Block-diagonal expansion of a grouped 1x1 mixer to its [out, 8*c_in] form."""
    out_ch = w.shape[0]
    if groups == 1:
        return w.reshape(out_ch, 8 * c_in)
    opg, ipg = out_ch // groups, (8 * c_in) // groups
    dense = torch.zeros(out_ch, 8 * c_in, dtype=w.dtype)
    flat = w.reshape(out_ch, ipg)
    for g in range(groups):
        dense[g * opg:(g + 1) * opg, g * ipg:(g + 1) * ipg] = \
            flat[g * opg:(g + 1) * opg]
    return dense


def group_mask(out_ch: int, in_ch: int, groups: int) -> torch.Tensor:
    mask = torch.zeros(out_ch, in_ch, dtype=torch.bool)
    opg, ipg = out_ch // groups, in_ch // groups
    for g in range(groups):
        mask[g * opg:(g + 1) * opg, g * ipg:(g + 1) * ipg] = True
    return mask


def check_analysis(site: dict, q: torch.Tensor, seed: int) -> dict:
    """Fold Q into an analysis-side mixer and measure what the fold costs."""
    c_in, out_ch, groups = site["c_in"], site["w"].shape[0], site["groups"]
    w = dense_form(site["w"].to(DT), groups, c_in)
    bias = site["b"].to(DT) if site["b"] is not None else None
    folded = w @ torch.kron(q, torch.eye(c_in, dtype=DT))

    mask = group_mask(out_ch, 8 * c_in, groups)
    off = (folded[~mask].norm() / folded.norm()).item() if groups > 1 else 0.0

    # learned wavelet + stored mixer   vs   Haar wavelet + folded mixer
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, c_in, 8, 8, 8, generator=g, dtype=DT)
    ref = apply1x1(dwt(x, q @ K_HAAR), w, bias)
    fold = apply1x1(dwt(x, K_HAAR), folded, bias)
    # ... and what a groups=g implementation could actually store
    proj = apply1x1(dwt(x, K_HAAR), folded * mask, bias)

    scale = ref.abs().max().clamp_min(1e-30)
    return {"off_block": off,
            "err_fold": ((ref - fold).abs().max() / scale).item(),
            "err_proj": ((ref - proj).abs().max() / scale).item()}


def check_synthesis(site: dict, q: torch.Tensor, seed: int) -> dict:
    """Fold Q into HaarUpsample's dense expansion (Q acts after the 1x1)."""
    c_out, in_ch = site["c_in"], site["w"].shape[1]
    w = site["w"].to(DT).reshape(8 * c_out, in_ch)
    bias = site["b"].to(DT) if site["b"] is not None else None
    kron = torch.kron(q.T.contiguous(), torch.eye(c_out, dtype=DT))

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, in_ch, 4, 4, 4, generator=g, dtype=DT)
    ref = idwt(apply1x1(x, w, bias), q @ K_HAAR)
    fold = idwt(apply1x1(x, kron @ w,
                         kron @ bias if bias is not None else None), K_HAAR)

    scale = ref.abs().max().clamp_min(1e-30)
    return {"off_block": 0.0,
            "err_fold": ((ref - fold).abs().max() / scale).item(),
            "err_proj": float("nan")}


def check_waveletconv(site: dict, q: torch.Tensor, seed: int) -> dict:
    """Fold Q through DWT and IDWT: W' = R^T W R, b' = R^T b, R = Q ⊗ I_C."""
    c_in, out_ch, groups = site["c_in"], site["w"].shape[0], site["groups"]
    w = dense_form(site["w"].to(DT), groups, c_in)
    bias = site["b"].to(DT) if site["b"] is not None else None
    R = torch.kron(q, torch.eye(c_in, dtype=DT))
    folded = R.T @ w @ R
    folded_b = (R.T @ bias) if bias is not None else None

    mask = group_mask(out_ch, 8 * c_in, groups)
    off = (folded[~mask].norm() / folded.norm()).item() if groups > 1 else 0.0

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, c_in, 8, 8, 8, generator=g, dtype=DT)
    k_q = q @ K_HAAR
    ref = idwt(apply1x1(dwt(x, k_q), w, bias), k_q)
    fold = idwt(apply1x1(dwt(x, K_HAAR), folded, folded_b), K_HAAR)
    proj = idwt(apply1x1(dwt(x, K_HAAR), folded * mask, folded_b), K_HAAR)

    scale = ref.abs().max().clamp_min(1e-30)
    return {"off_block": off,
            "err_fold": ((ref - fold).abs().max() / scale).item(),
            "err_proj": ((ref - proj).abs().max() / scale).item()}


# --------------------------------------------------------------------------- #

def synthetic_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """Random Q and mixers covering the two mixer regimes the paper contrasts.

    One dense (groups=1) downsample, one grouped (groups=8) WaveletConv, and
    one dense HaarUpsample.  Enough to show that the fold is exact in float64
    and that only groups=8 leaks energy off the block-diagonal.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    sd: dict[str, torch.Tensor] = {}
    # downsample, groups=1, C_in=4 -> 48
    sd["vit.downsample_layers.0.0.wave.skew"] = rand(8, 8)
    sd["vit.downsample_layers.0.0.mix.weight"] = rand(48, 32, 1, 1, 1)
    sd["vit.downsample_layers.0.0.mix.bias"] = rand(48)
    # WaveletConv3d, groups=8, C=48 (DWT and IDWT share Q; fold is R^T W R)
    sd["vit.gscs.0.proj.wave.skew"] = rand(8, 8)
    sd["vit.gscs.0.proj.mix.weight"] = rand(384, 48, 1, 1, 1)
    sd["vit.gscs.0.proj.mix.bias"] = rand(384)
    # HaarUpsample, dense expansion 96 -> 8*48
    sd["decoder5.wave.skew"] = rand(8, 8)
    sd["decoder5.expand.weight"] = rand(384, 96, 1, 1, 1)
    sd["decoder5.expand.bias"] = rand(384)
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=None,
                    help="LOWO checkpoint (.pt). Omit when using --synthetic.")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on random Q,W (CPU, no file). Does not reproduce "
                         "the paper's 7.8e-15 figure; use a real checkpoint "
                         "for that.")
    ap.add_argument("--in-chans", type=int, default=4,
                    help="input modalities feeding downsample_layers.0")
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="relative error below which the fold counts as exact")
    args = ap.parse_args()

    if args.synthetic:
        sd = synthetic_state_dict()
        ckpt_label = "--synthetic (random Q, W)"
    elif args.ckpt:
        sd = load_state_dict(args.ckpt)
        ckpt_label = args.ckpt
    else:
        ap.error("pass a checkpoint path, or --synthetic")

    sites = discover(sd, args.in_chans)
    if not sites:
        raise SystemExit("no HaarWavelet3D instances found in this checkpoint")

    print(f"checkpoint: {ckpt_label}")
    print(f"wavelet sites: {len(sites)}\n")

    head = (f"{'site':<48}{'role':<15}{'mixer':<14}{'grp':>4}"
            f"{'||Q-I||':>10}{'off-block':>11}{'fold err':>11}{'proj err':>11}")
    print(head)
    print("-" * len(head))

    absorbable, identifiable, broken = [], [], []
    fold_errors: list[float] = []
    for i, site in enumerate(sites):
        if site["kind"] == "orphan" or site["groups"] == 0:
            print(f"{site['path']:<48}{'(unresolved)':<15}")
            broken.append(site["path"])
            continue
        q = orthogonal_factor(sd, site["wave"])
        dev = (q - torch.eye(8, dtype=DT)).norm().item()
        if site["kind"] == "synthesis":
            res = check_synthesis(site, q, seed=1234 + i)
        elif site["kind"] == "waveletconv":
            res = check_waveletconv(site, q, seed=1234 + i)
        else:
            res = check_analysis(site, q, seed=1234 + i)

        shape = "x".join(str(d) for d in site["w"].shape[:2])
        print(f"{site['path']:<48}{site['role']:<15}{shape:<14}{site['groups']:>4}"
              f"{dev:>10.4f}{res['off_block']:>11.4f}"
              f"{res['err_fold']:>11.2e}{res['err_proj']:>11.2e}")

        fold_errors.append(res["err_fold"])
        if res["err_fold"] > args.tol:
            broken.append(site["path"])
        elif site["groups"] == 1 or res["off_block"] < 1e-12:
            absorbable.append((site["path"], dev))
        else:
            identifiable.append((site["path"], dev, res["off_block"]))

    print()
    if fold_errors:
        print(f"max fold error: {max(fold_errors):.1e}")
    print(f"inert (ρ≈0): {len(absorbable)}/{len(sites)}")
    print(f"absorbed     (fold exact, ρ≈0)              : {len(absorbable)}")
    print(f"load-bearing (escapes the grouped mixer)    : {len(identifiable)}")
    if broken:
        print(f"UNRESOLVED / fold failed                    : {len(broken)}")
        for p in broken:
            print(f"    {p}")

    if absorbable:
        worst = max(dev for _, dev in absorbable)
        print(f"\nAt the {len(absorbable)} absorbed sites the fold reproduces the "
              f"output to float64 round-off\neven though Q drifts up to "
              f"||Q-I||_F = {worst:.4f}: those degrees of freedom are locally "
              f"absorbed.")
    if identifiable:
        print("\nLoad-bearing sites, by how much of the folded operator escapes "
              "the groups pattern:")
        for path, dev, off in sorted(identifiable, key=lambda t: -t[2]):
            print(f"    {off:6.1%}  ||Q-I||_F={dev:7.4f}  {path}")


if __name__ == "__main__":
    main()
