"""P4: grouped-sandwich projection cost on the REAL 3D network.

Why this exists
---------------
The audit says a groups=8 sandwich site is *not* absorbable: after folding the
learned SO(8) factor into the neighbouring mixer, part of the folded operator
lands outside the 8-block pattern that a groups=8 convolution can store.  That
statement is about the whole weight space.  It does not say how much this costs
on the actual task, and a reviewer will ask exactly that.

So we build, at ONE chosen sandwich site, two concrete replacements and measure
what each one costs.  Nothing is retrained.

  1. exact fold  -- W' = R^T W R, b' = R^T b, R = Q (x) I_C.
     Learned bank -> fixed Haar, mixer upgraded to a DENSE 1x1 holding W'.
     This is the correctness control: it must reproduce the original output up
     to float round-off.

  2. projected fold -- What = Pi_G8(W'), i.e. keep only the eight diagonal
     blocks of W' and zero everything else.  Bias stays b' (the bias is not
     group-limited in this architecture).  In Frobenius norm over kernels this
     is the closest groups=8 mixer to W'.  It is NOT guaranteed to be the best
     approximation on the task: that caveat is stated in the paper.

Then three versions are compared (original, exact, projected) on identical
tensors:

  * local error, using the activation that the site ACTUALLY receives in the
    original network (captured with a forward hook), reported in Frobenius
    norm:  e_local = ||Y_alt - Y_orig||_F / ||Y_orig||_F

  * task metrics, running all three full networks over a small set of BraTS
    cases: TC / WT / ET Dice against the real label, plus their mean.

Every variant starts from the SAME original checkpoint and replaces exactly ONE
site.  Variants are never chained, otherwise the effect cannot be attributed.

Sampling rule
-------------
Sites are chosen by rho BEFORE any Dice is looked at, one each from the low,
middle and high end of the sandwich rho distribution.  This avoids picking
examples from the answer.

The rho ranking is ALWAYS recomputed from the checkpoint that is being run.
Never carry a ranking over from another run or from ``_a6_audit_out.txt``:
those rankeds are only valid for the exact checkpoint, site naming and
conjugation convention that produced them.

Checkpoint identity
-------------------
Nothing here decides which paper arm a checkpoint is.  ``describe_ckpt``
records structural facts (mixer shape and groups, per-site wavelet mode and
how far Q drifted from Haar) together with a SHA-256 of the checkpoint.  Those
facts pin the object down, but the arm label stays a human decision: grouped
downsampling is used by several arms, so a shape alone cannot name one.

USAGE (remote 8xA100 host, lowo repo root)

  # step 1: rank every sandwich site by rho, recomputed from THIS ckpt.
  #         No data needed. Save the file, it is the audit trail of the pick.
  python scripts/fold_sandwich_projection.py \\
      --config <cfg.yaml> --ckpt <best.pt> --rank-only

  # step 2: pilot on the validation split first (10 cases). Explore, do NOT
  #         pick sites from these numbers.
  python scripts/fold_sandwich_projection.py \\
      --config <cfg.yaml> --ckpt <best.pt> \\
      --data_dir ./data/fullres/train --test_list splits/val.txt \\
      --n_cases 10 --purpose pilot --out sandwich_projection_pilot.json

  # step 3: freeze the protocol, then run the held-out split with a purpose tag
  python scripts/fold_sandwich_projection.py \\
      --config <cfg.yaml> --ckpt <best.pt> \\
      --data_dir ./data/fullres/train --test_list splits/test_250.txt \\
      --n_cases 10 --purpose final --out sandwich_projection_test.json

  # sites are chosen from step 1; to pin them explicitly use
  #   --sites vit.gscs.1.proj2,vit.gscs.2.proj2,vit.gscs.0.proj2

  # CPU self-check with random weights, no checkpoint and no data:
  python scripts/fold_sandwich_projection.py --selftest

Notes on precision
------------------
Local errors are accumulated in float32 by default, which is what deployment
uses.  On the random self-test net the exact fold lands near 1e-6.  That number
describes THIS net only: it depends on the weight scale, the input, the backend,
and on whether TF32 is enabled, so it is not a floor or a bound for the real
network, and it says nothing at all about task Dice.  Pass --fp64-local to
recompute the local step in float64, where the exact fold should fall to
~1e-15 and the projected value should be unchanged, confirming the gap is
algebraic rather than numerical.

In float64 we must make sure both sides use the SAME kernel definition and no
stale float32 cache survives: check_wavelet_basis() verifies
K_original = Q @ K_new at evaluation dtype before any number is trusted.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from fold_all_sites_check import (  # noqa: E402
    dice,
    find_repo,
    haar8,
    kernel_matrix,
    reset_to_haar,
)
from fold_audit_supplement import region_dices  # noqa: E402

DT = torch.float64


# --------------------------------------------------------------------------- #
# small tensor algebra (same convention as verify_absorption.py)
# --------------------------------------------------------------------------- #
def group_mask(out_ch: int, in_ch: int, groups: int) -> torch.Tensor:
    """Boolean (out_ch, in_ch) mask of the block pattern a grouped 1x1 can store."""
    mask = torch.zeros(out_ch, in_ch, dtype=torch.bool)
    opg, ipg = out_ch // groups, in_ch // groups
    for g in range(groups):
        mask[g * opg:(g + 1) * opg, g * ipg:(g + 1) * ipg] = True
    return mask


def dense_form(w: torch.Tensor, groups: int, c_in: int) -> torch.Tensor:
    """Expand a grouped 1x1 mixer to its (out, groups*c_in) dense matrix."""
    out_ch = w.shape[0]
    if groups == 1:
        return w.reshape(out_ch, groups * c_in)
    opg, ipg = out_ch // groups, (groups * c_in) // groups
    dense = torch.zeros(out_ch, groups * c_in, dtype=w.dtype)
    flat = w.reshape(out_ch, ipg)
    for g in range(groups):
        dense[g * opg:(g + 1) * opg, g * ipg:(g + 1) * ipg] = \
            flat[g * opg:(g + 1) * opg]
    return dense


def q_of_wave(wave) -> torch.Tensor:
    """Q such that this wavelet's analysis matrix equals Q @ K_haar.

    Same convention as verify_absorption.orthogonal_factor.  For theta mode the
    learned rotation lives inside theta, so Q must be recovered as
    K(theta) @ K_haar^T rather than taking K(theta) itself.
    """
    with torch.no_grad():
        w = copy.deepcopy(wave)          # never mutate the caller's module
        if getattr(w, "mode", "") == "so8" and hasattr(w, "skew"):
            s = w.skew.detach().cpu().to(DT)
            return torch.matrix_exp(s - s.T)
        return (kernel_matrix(w).cpu() @ haar8().T.cpu())


def _q_and_mask(mod: nn.Module) -> tuple:
    """(Q, c_in, W_dense_cpu, mask) for one WaveletConv3d站点."""
    Q = q_of_wave(mod.wave)
    c_in = mod.mix.in_channels // 8
    W = dense_form(mod.mix.weight.detach().cpu().to(DT),
                   mod.mix.groups, c_in)
    out_ch = mod.mix.out_channels
    mask = group_mask(out_ch, 8 * c_in, mod.mix.groups)
    return Q, c_in, W, mask


def _kernel64(wave) -> torch.Tensor:
    """The (8,8) analysis matrix recomputed cleanly in float64 (CPU)."""
    w = copy.deepcopy(wave).double().cpu()
    w.__dict__.pop("_k_cache", None)
    return w._kernel().reshape(8, 8)


def _q64(wave) -> torch.Tensor:
    w = copy.deepcopy(wave).double().cpu()
    if getattr(w, "mode", "") == "so8" and hasattr(w, "skew"):
        return torch.matrix_exp(w.skew - w.skew.T).cpu()
    return (_kernel64(wave) @ haar8().T).cpu()


def check_wavelet_basis(orig_mod, new_mod, Q: torch.Tensor) -> dict:
    """Verify K_original = Q @ K_new, at two precision levels.

    The whole fold rests on one identity: the learned kernel must equal Q
    applied to the frozen-Haar kernel that replaces it.  If either side is
    computed from a different definition -- a stale ``_k_cache`` frozen before
    the reset, a reset that only took effect on one side -- the exact fold stops
    being an identity and everything downstream is meaningless.

    The identity is checked twice, and the gap between the two rows is the
    useful part:

      basis_rel_err64   both kernels recomputed in float64.  Anything larger
                        than ~1e-10 is a genuine disagreement about the
                        definition and means the numbers are wrong.  This is
                        the authoritative check: Q here is the float64 expm
                        result, so K_original must be compared against the
                        float64 recomputation, NOT against a float32 _kernel().
      basis_rel_err     at the model's own dtype, comparing the float32
                        _kernel() against the float64 Q.  On sites with a
                        large learned drift (q_drift ~ 3) the float32
                        matrix_exp alone contributes ~1e-4 noise, so this row
                        is reported as information only.  It is NOT a failure
                        gate: a real bank/reset mismatch shows up as an O(1)
                        disagreement in basis_rel_err64, which is the only
                        signal we halt on.
    """
    K_orig = kernel_matrix(orig_mod.wave).cpu()
    K_new = kernel_matrix(new_mod.wave).cpu()
    rel = float((K_orig - Q @ K_new).norm() / K_orig.norm().clamp_min(1e-30))

    Ko64, Kn64 = _kernel64(orig_mod.wave), _kernel64(new_mod.wave)
    Ko64, Kn64 = Ko64.detach(), Kn64.detach()
    Q64 = _q64(orig_mod.wave).detach()
    rel64 = float((Ko64 - Q64 @ Kn64).norm() / Ko64.norm().clamp_min(1e-30))

    return {"basis_rel_err": rel,
            "basis_rel_err64": rel64,
            "orig_has_kernel_cache": hasattr(orig_mod.wave, "_k_cache"),
            "new_has_kernel_cache": hasattr(new_mod.wave, "_k_cache"),
            "new_skew_is_zero": bool(
                not hasattr(new_mod.wave, "skew")
                or float(new_mod.wave.skew.detach().abs().max()) == 0.0)}


# --------------------------------------------------------------------------- #
# checkpoint identity (description only, never an arm verdict)
# --------------------------------------------------------------------------- #
def sha256_of(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_param(mod: nn.Module, name: str) -> bool:
    attr = getattr(mod, name, None)
    return isinstance(attr, nn.Parameter)


def describe_ckpt(model, ckpt_path: str, cfg_path: str, cfg: dict) -> dict:
    """Facts that pin down which object we ran, without naming a paper arm.

    The arm label is deliberately NOT inferred here.  Several arms share
    grouped downsampling, so mixer shape can establish the grouping pattern but
    cannot establish the arm.  That mapping stays a human decision; what this
    records is enough evidence to make it, plus a hash so the decision can be
    re-checked later.
    """
    from model_segmamba.lowo import WaveletConv3d

    sites = []
    for name, mod in model.named_modules():
        if not isinstance(mod, WaveletConv3d):
            continue
        wave = mod.wave
        Q = q_of_wave(wave)
        sites.append(dict(
            site=name,
            mix_weight_shape=list(mod.mix.weight.shape),   # always 5-D
            mix_bias_shape=(list(mod.mix.bias.shape)
                            if mod.mix.bias is not None else None),
            mix_groups=int(mod.mix.groups),
            in_channels=int(mod.mix.in_channels),
            out_channels=int(mod.mix.out_channels),
            wave_mode=getattr(wave, "mode", "unknown"),
            theta_is_param=_is_param(wave, "theta"),
            skew_is_param=_is_param(wave, "skew"),
            q_drift_from_identity=float(
                (Q - torch.eye(8, dtype=Q.dtype, device=Q.device)).norm()),
            has_kernel_cache=hasattr(wave, "_k_cache"),
        ))

    # Shape of the stem downsample mixer, recorded only so a human can tell a
    # dense from a grouped downsampling arm later.  It is not used by any check.
    stem = None
    for key in ("vit.downsample_layers.0.0", "downsample_layers.0.0"):
        try:
            mm = get_module(model, key)
        except (AttributeError, IndexError, KeyError):
            continue
        if hasattr(mm, "mix"):
            stem = dict(site=key,
                        mix_weight_shape=list(mm.mix.weight.shape),
                        mix_groups=int(mm.mix.groups))
            break

    cfg_fields = {k: cfg.get(k) for k in
                  ("model", "wavelet_mode", "learnable_wavelet", "grouped_mix",
                   "per_channel", "wavelet_boundary_weight")
                  if k in cfg}
    try:
        st = os.stat(ckpt_path)
        ckpt_info = dict(path=ckpt_path, sha256=sha256_of(ckpt_path),
                         size_bytes=st.st_size,
                         mtime=_dt.datetime.fromtimestamp(
                             st.st_mtime).isoformat(timespec="seconds"))
    except OSError as exc:
        ckpt_info = dict(path=ckpt_path, sha256=None, error=repr(exc))

    # The config next to the checkpoint is what actually produced it; prefer it.
    sibling = os.path.join(os.path.dirname(ckpt_path), "..", "config.yaml")
    sibling = os.path.normpath(sibling)
    return dict(ckpt=ckpt_info,
                config_path=cfg_path,
                sibling_config=(sibling if os.path.exists(sibling) else None),
                config_fields=cfg_fields,
                stem=stem,
                sites=sites)


# --------------------------------------------------------------------------- #
# site discovery and rho ranking
# --------------------------------------------------------------------------- #
def get_module(model, path: str) -> nn.Module:
    """Look up a dotted path such as ``vit.gscs.0.proj``; empty path is the root."""
    m = model
    if not path:
        return m
    for part in path.split("."):
        m = getattr(m, part) if not part.isdigit() else m[int(part)]
    return m


def set_module(model, path: str, new: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = get_module(model, parent_path) if parent_path else model
    if leaf.isdigit():
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)


def list_sandwich(model) -> list:
    """All WaveletConv3d sites with their off-block rho of the FOLDED operator.

    rho off-block is measured on W' = R^T W R, which is the quantity the
    projection step has to approximate.  Sites are returned sorted by rho so a
    low / middle / high triple can be picked before any task metric is seen.
    """
    from model_segmamba.lowo import WaveletConv3d

    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, WaveletConv3d):
            continue
        Q, c_in, W, mask = _q_and_mask(mod)
        R = torch.kron(Q, torch.eye(c_in, dtype=DT))
        Wp = R.T @ W @ R
        off = Wp[~mask.to(Wp.device)].norm().item()
        tot = Wp.norm().item()
        rows.append(dict(site=name, groups=int(mod.mix.groups),
                         c_in=int(c_in), out=int(mod.mix.out_channels),
                         rho_off_block=off / max(tot, 1e-30),
                         q_drift=float((Q - torch.eye(8, dtype=DT, device=Q.device)).norm())))
    rows.sort(key=lambda r: r["rho_off_block"])
    return rows


def precompute_wavelet_kernels(model) -> int:
    """Eagerly cache every learnable Haar wavelet's fused kernel, once, in fp32.

    The deployed A6 checkpoint has learnable=True on every HaarWavelet3D, so
    `_get_kernel` falls through to `_kernel()` on each forward and runs
    `torch.matrix_exp` inside `run_test_eval.tta_predict`'s bf16 autocast
    region.  On some activation batches cusolver's bf16 path degenerates
    (looks like a hard hang).  One eager evaluation up front, outside any
    autocast context, populates `_k_cache`; `_get_kernel` then short-circuits
    via `hasattr(self, "_k_cache")` and matrix_exp never runs in the hot path.
    """
    from model_segmamba.lowo import HaarWavelet3D

    n = 0
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, HaarWavelet3D):
                k = mod._kernel().detach()
                mod.register_buffer("_k_cache", k)
                n += 1
    return n


# --------------------------------------------------------------------------- #
# build the two replacements (each from a fresh copy of the original)
# --------------------------------------------------------------------------- #
def build_variant(model, site: str, kind: str, device, dtype) -> nn.Module:
    """Deepcopy `model` and replace exactly ONE sandwich site.

    kind="exact"  dense 1x1 holding W' = R^T W R
    kind="proj"   groups=8 1x1 holding Pi_G8(W')
    In both cases the learned bank is frozen at fixed Haar and the bias is b'.
    """
    assert kind in ("exact", "proj")
    var = copy.deepcopy(model)
    mod = get_module(var, site)

    Q, c_in, W, mask = _q_and_mask(mod)
    b = (mod.mix.bias.detach().cpu().to(DT)
             if mod.mix.bias is not None else None)
    R = torch.kron(Q, torch.eye(c_in, dtype=DT))
    Wp = R.T @ W @ R
    bp = R.T @ b if b is not None else None

    out_ch = int(mod.mix.out_channels)
    in_ch = 8 * c_in
    packed = None
    with torch.no_grad():
        if kind == "exact":
            new_mix = nn.Conv3d(in_ch, out_ch, kernel_size=1,
                                groups=1, bias=bp is not None)
            new_mix.weight.copy_(Wp.reshape(out_ch, in_ch, 1, 1, 1))
        else:
            new_mix = nn.Conv3d(in_ch, out_ch, kernel_size=1,
                                groups=8, bias=bp is not None)
            m = mask.to(Wp.device)
            packed = torch.zeros(out_ch, in_ch // 8, dtype=DT)
            opg, ipg = out_ch // 8, in_ch // 8
            for g in range(8):
                packed[g * opg:(g + 1) * opg] = \
                    (Wp * m)[g * opg:(g + 1) * opg, g * ipg:(g + 1) * ipg]
            new_mix.weight.copy_(packed.reshape(out_ch, in_ch // 8, 1, 1, 1))
        if bp is not None:
            new_mix.bias.copy_(bp)
        new_mix = new_mix.to(device=device, dtype=dtype)
        mod.mix = new_mix
        reset_to_haar(mod.wave)

    # Keep the float64 folded kernel so the local check can rebuild the layer
    # at full precision.  A model can only store W' at its own dtype, so the
    # fp32 --fp64-local path would otherwise inherit that rounding and mask the
    # algebraic identity we are trying to verify.
    mod._fold64 = dict(kind=kind, Wp=Wp, packed=packed, bp=bp,
                       out_ch=out_ch, in_ch=in_ch)
    return var


# --------------------------------------------------------------------------- #
# local error on the real activation
# --------------------------------------------------------------------------- #
def _rebuild64(mod: nn.Module) -> nn.Module:
    """float64 copy, re-injecting the un-rounded folded kernel when available.

    The deployed layer necessarily stores the folded kernel at model dtype, so
    calling .double() on it keeps that rounding.  build_variant therefore keeps
    the float64 tensor as ``_fold64`` and we re-inject it here, which lets the
    exact fold actually reach machine precision.
    """
    m = copy.deepcopy(mod).double()
    # A cached kernel was materialized at the old dtype before we promoted the
    # module, so it would keep the float32 rounding.  Drop it and let both the
    # original side and the replacement recompute from their own theta / skew,
    # otherwise the two sides disagree in the last digits and the float64 check
    # silently stops measuring what we think it measures.
    m.wave.__dict__.pop("_k_cache", None)
    plan = getattr(m, "_fold64", None)
    if plan is None:
        return m  # the original site: no plan, nothing to re-inject
    with torch.no_grad():
        w = plan["Wp"] if plan["kind"] == "exact" else plan["packed"]
        cols = plan["in_ch"] if plan["kind"] == "exact" else plan["in_ch"] // 8
        m.mix.weight.copy_(w.reshape(plan["out_ch"], cols, 1, 1, 1))
        if plan["bp"] is not None:
            m.mix.bias.copy_(plan["bp"])
    return m


def _prepare(mods, use_fp64: bool) -> list:
    """Optionally promote the three modules to float64, once per site."""
    if use_fp64:
        return [_rebuild64(m) for m in mods]
    return list(mods)


def _module_dtype(mods) -> torch.dtype:
    p = next(mods[0].parameters(), None)
    return p.dtype if p is not None else torch.float32


def _accumulate(mods, xv: torch.Tensor) -> dict:
    """Feed one window to all three; return squared norms, in float64."""
    with torch.no_grad():
        # Call forward() directly, not __call__(), on purpose.  The original
        # module carries the SiteRecorder forward hook, and `m(xv)` would
        # re-enter that hook recursively until RecursionError.  forward()
        # bypasses the __call__ hook machinery entirely.
        ys = [m.forward(xv) for m in mods]
        ref, other = ys[0], ys[1:]
        # Promote BEFORE taking the norm: norm() at the model dtype then
        # squaring loses ~1e-7 relative on large volumes, which would be
        # indistinguishable from a real (small) difference later.
        scale = float(ref.double().norm().clamp_min(1e-300))
        diff = [float((o - ref).double().norm()) for o in other]
        return dict(sq_ref=scale * scale, sq_exact=diff[0] ** 2,
                    sq_proj=diff[1] ** 2,
                    rel_exact=diff[0] / max(scale, 1e-300),
                    rel_proj=diff[1] / max(scale, 1e-300))


def assert_ready(model, site: str, tag: str) -> None:
    """Guard the conditions the comparison silently depends on."""
    mod = get_module(model, site)
    if mod.training:
        raise SystemExit(f"[{tag}] {site} is not in eval() mode")
    skew = getattr(mod.wave, "skew", None)
    if tag != "original" and skew is not None:
        if float(skew.detach().abs().max()) != 0.0:
            raise SystemExit(
                f"[{tag}] {site} still carries a learned skew, so its bank was "
                f"not reset to Haar and the fold identity does not apply")


class SiteRecorder:
    """Forward hook comparing the three blocks on the real activation.

    The hook is registered on the WaveletConv3d BLOCK, so ``inp[0]`` is what
    that whole region receives in the original network, before its own analysis
    bank runs.  The identical tensor is handed to all three blocks, each of
    which then performs its own complete analysis, mixing and synthesis.  The
    comparison is therefore made at the same output boundary, and nothing has
    to be assumed about what the intermediate coefficients mean.  Capturing the
    post-analysis coefficients instead and feeding them to the replacements
    would compare different objects, since the replacements start from a Haar
    bank.

    With sliding-window inference plus mirror TTA this fires once per window
    per mirror.      Errors are pooled by accumulating squared norms in float64.  Doing the
    accumulation in float32 is not a rounding detail here: summed squares over
    a whole volume grow past what float32 can resolve, and on a small three
    window probe that choice alone moves the reported ratio by ~1e-8.
    """

    def __init__(self, site, orig_mod, exact_mod, proj_mod, use_fp64):
        self.site = site
        self.use_fp64 = use_fp64
        self.mods = _prepare([orig_mod, exact_mod, proj_mod], use_fp64)
        self.dtype = torch.float64 if use_fp64 else _module_dtype(self.mods)
        Q, _, _, _ = _q_and_mask(orig_mod)
        self.basis = check_wavelet_basis(self.mods[0], self.mods[1], Q)
        self.n_windows = 0
        self._clear()

    def _clear(self) -> None:
        self.sq_ref = self.sq_exact = self.sq_proj = 0.0
        self.worst_exact = self.worst_proj = 0.0
        self.n_windows = 0

    def pooled(self) -> dict:
        """Relative feature-norm error pooled over every window seen since the last reset."""
        ref = max(self.sq_ref, 1e-300)
        return {"e_local_exact": float(np.sqrt(self.sq_exact / ref)),
                "e_local_proj": float(np.sqrt(self.sq_proj / ref)),
                "e_local_exact_worst_window": self.worst_exact,
                "e_local_proj_worst_window": self.worst_proj,
                "n_windows": int(self.n_windows)}

    def reset_case(self):
        """Hand back what this case accumulated and start the next one."""
        prev = self.pooled()
        self._clear()
        return prev

    def __call__(self, module, inp, out):
        xv = inp[0].detach().to(self.dtype)
        r = _accumulate(self.mods, xv)
        self.sq_ref += r["sq_ref"]
        self.sq_exact += r["sq_exact"]
        self.sq_proj += r["sq_proj"]
        self.worst_exact = max(self.worst_exact, r["rel_exact"])
        self.worst_proj = max(self.worst_proj, r["rel_proj"])
        self.n_windows += 1
        return None


def local_triple(orig_mod, exact_mod, proj_mod, x, use_fp64: bool) -> dict:
    """Single-sample convenience wrapper used by the self-test."""
    mods = _prepare([orig_mod, exact_mod, proj_mod], use_fp64)
    dtype = torch.float64 if use_fp64 else _module_dtype(mods)
    r = _accumulate(mods, x.detach().to(dtype))
    ref = max(r["sq_ref"], 1e-300)
    return {"e_local_exact": float(np.sqrt(r["sq_exact"] / ref)),
            "e_local_proj": float(np.sqrt(r["sq_proj"] / ref))}


# --------------------------------------------------------------------------- #
# whole-network dice for three models on one case
# --------------------------------------------------------------------------- #
def predict(model, img, inferer, R, use_tta: bool, et_thr: int) -> np.ndarray:
    """Run the repo's own evaluation path so the protocol cannot drift.

    Re-implementing TTA here would open the door to a silent protocol change.
    The reported per-arm Dice come from run_test_eval's window + TTA + ET->NCR
    triple; reproducing fewer than all three would make these numbers look like
    measurements of the same thing while actually being something else.
    """
    pred = R.tta_predict(img, model, inferer, use_tta)
    if et_thr:
        pred = R.apply_et_to_ncr(pred, et_thr)
    return pred


def evaluate_case(img, gt, models, inferer, R, use_tta: bool,
                  et_thr: int) -> dict:
    out = {}
    for tag, m in models.items():
        pred = predict(m, img, inferer, R, use_tta, et_thr)
        out[tag] = region_dices(gt, pred)
    base = out["original"]
    paired = {}
    for tag in ("exact", "proj"):
        for reg in ("TC", "WT", "ET"):
            paired[f"ddice_{reg}_{tag}"] = (out[tag][f"dice_{reg}"]
                                            - base[f"dice_{reg}"])
    out["paired_vs_original"] = paired
    return out


# --------------------------------------------------------------------------- #
# CPU self-test: no checkpoint, no data
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    repo_root = find_repo("")
    sys.path.insert(0, repo_root)
    from model_segmamba.lowo import WaveletConv3d

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.sand = WaveletConv3d(6, learnable=True, mode="so8")

        def forward(self, x):
            return self.sand(x)

    torch.manual_seed(0)
    net = Net().eval()
    with torch.no_grad():
        net.sand.wave.skew.copy_(torch.randn_like(net.sand.wave.skew) * 0.7)

    rows = list_sandwich(net)
    assert rows, "no sandwich site found"
    r = rows[0]
    print(f"[selftest] site={r['site']} groups={r['groups']} "
          f"c_in={r['c_in']} rho={r['rho_off_block']:.4f}")

    dev, dtype = torch.device("cpu"), torch.float32
    net = net.to(dtype)
    site = r["site"]
    exact = build_variant(net, site, "exact", dev, dtype)
    proj = build_variant(net, site, "proj", dev, dtype)

    x = torch.randn(1, 6, 16, 16, 16)
    om = get_module(net, site)
    em = get_module(exact, site)
    pm = get_module(proj, site)

    # STEP 1: the two sides must agree on what the Haar basis IS, otherwise the
    # exact fold has nothing to be exact about.
    assert_ready(net, site, "original")
    assert_ready(exact, site, "exact")
    assert_ready(proj, site, "proj")
    Q0, _, _, _ = _q_and_mask(om)
    chk = check_wavelet_basis(om, em, Q0)
    print(f"[selftest] K_orig = Q @ K_new: rel64 {chk['basis_rel_err64']:.3e}, "
          f"rel@fp32 {chk['basis_rel_err']:.3e}, "
          f"skew zeroed={chk['new_skew_is_zero']}")
    assert chk["basis_rel_err64"] < 1e-10, "replacement uses a different basis"
    assert chk["new_skew_is_zero"], "replacement bank was not reset to Haar"

    # STEP 2: pooled local error is a relative feature-norm error, not a mean of
    # per-window errors. Feeding the same window three times must give the single-window
    # value, which catches the two definitions being confused.
    x = torch.randn(1, 6, 16, 16, 16)
    rec = SiteRecorder(site, om, em, pm, False)
    for _ in range(3):
        rec(None, (x,), None)
    pooled = rec.pooled()
    single = local_triple(om, em, pm, x, False)
    print(f"[selftest] pooled over 3 identical windows: "
          f"exact={pooled['e_local_exact']:.3e} proj={pooled['e_local_proj']:.3e}"
          f"  single exact={single['e_local_exact']:.3e}")
    assert abs(pooled["e_local_exact"] - single["e_local_exact"]) < 1e-12
    assert pooled["n_windows"] == 3

    for tag, u64 in (("fp32", False), ("fp64", True)):
        res = local_triple(om, em, pm, x, u64)
        print(f"[selftest] local {tag}: exact={res['e_local_exact']:.3e} "
              f"proj={res['e_local_proj']:.3e}")

    # exact must be a true control, proj must be visibly worse
    res = local_triple(om, em, pm, x, True)
    assert res["e_local_exact"] < 1e-12, "exact fold is not an identity"
    assert res["e_local_proj"] > 1e-3, "projection shows no cost at all"

    gt = np.zeros((8, 8, 8), dtype=np.int8)
    gt[0, 0, 0], gt[1, 1, 1], gt[2, 2, 2] = 1, 2, 3
    assert min(region_dices(gt, gt).values()) == 1.0
    print("[selftest] OK")


# --------------------------------------------------------------------------- #
def pick_sites(rows, which: str) -> list:
    """Choose low / middle / high rho sites before seeing any task metric."""
    if which:
        names = [w.strip() for w in which.split(",") if w.strip()]
        known = {r["site"]: r for r in rows}
        missing = [n for n in names if n not in known]
        if missing:
            raise SystemExit(f"unknown site(s): {missing}\n"
                             f"available: {[r['site'] for r in rows]}")
        return [known[n] for n in names]
    if not rows:
        return []
    n = len(rows)
    idx = sorted({0, n // 2, n - 1})
    return [rows[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--data_dir", default="./data/fullres/train")
    ap.add_argument("--test_list", default="splits/test_250.txt")
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--sites", default="", help="comma-separated; default low/mid/high rho")
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--roi", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--fp64-local", action="store_true")
    ap.add_argument("--force_ethr", type=int, default=50,
                    help="ET->NCR total-volume threshold used by every arm; "
                         "set 0 to disable. Keeping it fixed is what makes "
                         "these Dice comparable with the reported ones.")
    ap.add_argument("--purpose", choices=("pilot", "final"), default="pilot",
                    help="pilot runs are exploratory: do not use them to pick "
                         "sites and then re-report the same cases as evidence")
    ap.add_argument("--out", default="sandwich_projection.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    repo_root = find_repo(args.repo)
    sys.path.insert(0, repo_root)
    import yaml
    import run_test_eval as R
    from monai.inferers import SlidingWindowInferer
    from light_training.dataloading.dataset import MedicalDataset

    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = R.build_model_from_config(cfg, device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(R.filte_state_dict(sd), strict=True)
    model.eval()
    n_kernels = precompute_wavelet_kernels(model)
    print(f"[ckpt] precomputed {n_kernels} learnable wavelet kernels outside autocast")

    provenance = describe_ckpt(model, args.ckpt, args.config, cfg)
    print(f"[ckpt] {args.ckpt}")
    print(f"[ckpt] sha256 {provenance['ckpt']['sha256']}")
    print(f"[ckpt] config fields {provenance['config_fields']}")
    print(f"[ckpt] stem {provenance['stem']}")

    rows = list_sandwich(model)
    print(f"[rank] {len(rows)} sandwich sites, rho recomputed from THIS ckpt")
    for r in rows:
        print(f"   {r['site']:<40} grp={r['groups']} "
              f"c_in={r['c_in']:<4} rho={r['rho_off_block']:.4f} "
              f"qdrift={r['q_drift']:.4f}")
    if args.rank_only:
        payload = dict(checkpoint=provenance["ckpt"],
                       config_path=args.config,
                       config_fields=provenance["config_fields"],
                       ranking=rows, sites=provenance["sites"])
        with open("sandwich_rho_ranking.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("[rank] wrote sandwich_rho_ranking.json "
              "(valid for this checkpoint only)")
        return

    targets = pick_sites(rows, args.sites)
    print(f"[targets] {[t['site'] for t in targets]} "
          f"(rho {[round(t['rho_off_block'], 4) for t in targets]})")

    inferer = SlidingWindowInferer(roi_size=[args.roi] * 3,
                                   sw_batch_size=args.sw_batch,
                                   overlap=args.overlap, mode="gaussian")
    names = [l.strip() for l in open(args.test_list, encoding="utf-8")
             if l.strip() and not l.startswith("#")][:args.n_cases]
    ds = MedicalDataset([os.path.join(args.data_dir, n + ".npz") for n in names])

    dtype = next(model.parameters()).dtype
    settings = dict(
        purpose=args.purpose,
        device=str(device), model_dtype=str(dtype),
        torch=torch.__version__, python=platform.python_version(),
        host=platform.node(),
        overlap=args.overlap, sw_batch_size=args.sw_batch,
        roi=args.roi, inferer_mode="gaussian",
        tta=not args.no_tta,
        mirror_passes=len(R.MIRROR_COMBOS) if not args.no_tta else 1,
        autocast="bfloat16 via run_test_eval.tta_predict",
        et_to_ncr_threshold=args.force_ethr,
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        fp64_local=args.fp64_local,
        data_dir=args.data_dir, test_list=args.test_list,
        case_ids=list(names),
        local_pooling=("sqrt(sum ||delta||_F^2 / sum ||Y_orig||_F^2) over all "
                       "sliding windows" +
                       (" and mirrors" if not args.no_tta else "")),
    )
    print(f"[settings] {json.dumps(settings, indent=2)}")
    results = []
    for t in targets:
        site = t["site"]
        # each variant is rebuilt from the ORIGINAL checkpoint, one site only
        model_exact = build_variant(model, site, "exact", device, dtype)
        model_proj = build_variant(model, site, "proj", device, dtype)
        model_exact.eval()
        model_proj.eval()
        models = {"original": model, "exact": model_exact, "proj": model_proj}
        for tag, m in models.items():
            assert_ready(m, site, tag)

        om = get_module(model, site)
        em = get_module(model_exact, site)
        pm = get_module(model_proj, site)
        rec = SiteRecorder(site, om, em, pm, args.fp64_local)
        brel = rec.basis["basis_rel_err"]
        brel64 = rec.basis["basis_rel_err64"]
        print(f"[{site}] K_orig = Q @ K_new: rel64 {brel64:.3e}, "
              f"rel@model dtype {brel:.3e} "
              f"(orig_cache={rec.basis['orig_has_kernel_cache']}, "
              f"new_cache={rec.basis['new_has_kernel_cache']}, "
              f"skew_zeroed={rec.basis['new_skew_is_zero']})")
        if brel64 > 1e-10:
            raise SystemExit(
                f"[{site}] the two sides do not share one kernel definition "
                f"(rel in float64 {brel64:.3e}). The bank reset did not take, "
                f"or a stale kernel cache survived it, so the exact fold is "
                f"not an identity and nothing below would be meaningful.")
        if brel > 1e-2:
            raise SystemExit(
                f"[{site}] K_orig and Q @ K_new differ by {brel:.3e} at model "
                f"dtype while agreeing at float64. This is far above the "
                f"~1e-4 float32 matrix_exp noise we expect on high-drift "
                f"sites, so the float32 bank itself looks abnormal, check "
                f"the model first.")

        handle = om.register_forward_hook(rec)
        per_case = []
        for i in range(len(ds)):
            item = ds[i]
            img = torch.from_numpy(
                np.ascontiguousarray(item["data"])).float()[None].to(device)
            gt = np.asarray(item["seg"][0]).astype(np.int8)
            gt[gt < 0] = 0
            rec.reset_case()
            entry = evaluate_case(img, gt, models, inferer, R,
                                  not args.no_tta, args.force_ethr)
            loc = rec.pooled()
            cid = item["properties"]["name"]
            per_case.append(dict(case=cid, rho_off_block=t["rho_off_block"],
                                 dice=entry, local=loc))
            dd = entry["paired_vs_original"]
            print(f"   {site} | {cid}: WT {entry['original']['dice_WT']:.4f} "
                  f"(exact {dd['ddice_WT_exact']:+.4f}, "
                  f"proj {dd['ddice_WT_proj']:+.4f}) | "
                  f"e_exact={loc['e_local_exact']:.2e} "
                  f"e_proj={loc['e_local_proj']:.2e} "
                  f"over {loc['n_windows']} windows")

        handle.remove()
        results.append(dict(site=site, rho_off_block=t["rho_off_block"],
                            q_drift=t["q_drift"], basis=rec.basis,
                            cases=per_case))

        del model_exact, model_proj, models
        torch.cuda.empty_cache()

    def _mean_dice(res, tag, region):
        return float(np.mean([c["dice"][tag][f"dice_{region}"]
                              for c in res["cases"]]))

    def _paired(res, tag, region):
        return float(np.mean([c["dice"]["paired_vs_original"][
            f"ddice_{region}_{tag}"] for c in res["cases"]]))

    def _loc(res, key):
        v = [c["local"][key] for c in res["cases"]]
        return float(np.mean(v)), float(np.min(v)), float(np.max(v))

    def _paired_abs(res, tag, region):
        return float(np.mean([abs(c["dice"]["paired_vs_original"][
            f"ddice_{region}_{tag}"]) for c in res["cases"]]))

    def _paired_worst(res, tag, region):
        return float(max([abs(c["dice"]["paired_vs_original"][
            f"ddice_{region}_{tag}"]) for c in res["cases"]]))

    def _paired_worst_case(res, tag, region):
        worst = max(res["cases"], key=lambda c: abs(
            c["dice"]["paired_vs_original"][f"ddice_{region}_{tag}"]))
        return worst["case"]

    def _paired_drop(res, tag, region):
        """Min signed delta (worst, or smallest gain if all positive) and its case."""
        c = min(res["cases"], key=lambda x:
                x["dice"]["paired_vs_original"][f"ddice_{region}_{tag}"])
        return (float(c["dice"]["paired_vs_original"][f"ddice_{region}_{tag}"]),
                c["case"])

    def _case_overall(c, tag):
        """Per-case overall delta = mean over the three regions."""
        return float(np.mean([c["dice"]["paired_vs_original"][
            f"ddice_{reg}_{tag}"] for reg in ("TC", "WT", "ET")]))

    summary = []
    for res in results:
        row = dict(site=res["site"],
                   rho_off_block=res["rho_off_block"],
                   q_drift=res["q_drift"])
        for key in ("e_local_exact", "e_local_proj"):
            m, lo, hi = _loc(res, key)
            row[f"{key}_mean"], row[f"{key}_min"], row[f"{key}_max"] = m, lo, hi
        for tag in ("original", "exact", "proj"):
            for reg in ("WT", "TC", "ET"):
                row[f"dice_{reg}_{tag}"] = _mean_dice(res, tag, reg)
            row[f"dice_mean_{tag}"] = float(np.mean(
                [_mean_dice(res, tag, r) for r in ("WT", "TC", "ET")]))
        for tag in ("exact", "proj"):
            for reg in ("WT", "TC", "ET"):
                row[f"ddice_{reg}_{tag}"] = _paired(res, tag, reg)
                row[f"ddice_abs_mean_{reg}_{tag}"] = _paired_abs(res, tag, reg)
                row[f"ddice_abs_worst_{reg}_{tag}"] = _paired_worst(res, tag, reg)
                row[f"ddice_abs_worst_case_{reg}_{tag}"] = _paired_worst_case(
                    res, tag, reg)
                row[f"ddice_drop_{reg}_{tag}"], row[f"ddice_drop_case_{reg}_{tag}"] = \
                    _paired_drop(res, tag, reg)
            # case-based overall (per-case mean over regions, then aggregate)
            per_case = [_case_overall(c, tag) for c in res["cases"]]
            row[f"ddice_mean_{tag}"] = float(np.mean(per_case))
            row[f"ddice_abs_mean_{tag}"] = float(np.mean(
                [abs(v) for v in per_case]))
            worst = max(range(len(per_case)), key=lambda i: abs(per_case[i]))
            drop = min(range(len(per_case)), key=lambda i: per_case[i])
            row[f"ddice_abs_worst_{tag}"] = float(abs(per_case[worst]))
            row[f"ddice_abs_worst_case_{tag}"] = res["cases"][worst]["case"]
            row[f"ddice_drop_{tag}"] = float(per_case[drop])
            row[f"ddice_drop_case_{tag}"] = res["cases"][drop]["case"]
        summary.append(row)

    print("\n== summary (ddice = per-case mean over 3 regions; abs = |ddice|; "
          "worst = max|ddice|; drop = min ddice (worst if negative, smallest gain if all positive) ==")
    print(f"{'site':<28}{'rho':>7}{'e_exact':>10}{'e_proj':>9}"
          f"{'Dice_o':>8}{'Dice_e':>8}{'Dice_p':>8}"
          f"{'ddice':>9}{'abs':>9}{'worst':>9}{'drop':>9}")
    for r in summary:
        print(f"{r['site']:<28}{r['rho_off_block']:>7.3f}"
              f"{r['e_local_exact_mean']:>10.2e}"
              f"{r['e_local_proj_mean']:>9.2e}"
              f"{r['dice_mean_original']:>8.4f}"
              f"{r['dice_mean_exact']:>8.4f}"
              f"{r['dice_mean_proj']:>8.4f}"
              f"{r['ddice_mean_proj']:>+9.4f}"
              f"{r['ddice_abs_mean_proj']:>9.4f}"
              f"{r['ddice_abs_worst_proj']:>9.4f}"
              f"{r['ddice_drop_proj']:>+9.4f}")

    payload = dict(
        purpose=args.purpose,
        caveat=("pilot run, exploratory: do not select sites from these "
                "numbers and then re-report the same cases as confirmation"
                if args.purpose == "pilot" else
                "final run on the held-out list named in settings.test_list"),
        checkpoint=provenance["ckpt"],
        config_path=args.config,
        config_fields=provenance["config_fields"],
        observed_structure=dict(stem=provenance["stem"],
                                sites=provenance["sites"]),
        settings=settings,
        ranking=rows, summary=summary, detail=results)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
