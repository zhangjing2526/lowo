# LOWO — Learnable Orthogonal Wavelet Operator

**[ICASSP 2027 submission]** Structural orthogonal wavelet resampling for 3D convolutions, and the
identifiability of its learned basis.

This repository contains the models, training/inference recipes and the exact
checkpoint-level audit that supports the claims of the paper
*"LOWO: Structurally Orthogonal Wavelet Resampling for 3D Convolutions and the
Identifiability of Its Learned Basis"*.

```text
┌─────────────────────────────┐
│  K(θ, A) = Q(A) · K_haar(θ) │   Q(A) = exp(A − Aᵀ) ∈ SO(8), 28 DOF
└─────────────────────────────┘
         │ stride-2 conv³D (DWT fused)          (IDWT = same kernel, transposed)
         ▼
  subband-major 8C feature map ──► 1×1 subband mixer (groups=1 → absorbs Q)
```

---

## What is in this repository

| Path | Contents |
|---|---|
| `model_segmamba/lowo.py` | **The LOWO operator itself** — `HaarWavelet3D`, `HaarDownsample`, `HaarUpsample`, `WaveletConv3d`. Backbone-agnostic (only `torch`). |
| `model_segmamba/haar_blocks.py` | UNet decoder blocks built on LOWO (`WaveletGSC`, `HaarUpBlock`, `MlpChannel`). |
| `model_segmamba/segmamba_haar.py` | SegMamba + LOWO (encoder/decoder wiring). |
| `model_segmamba/segmamba_attn.py` | SegAttn + LOWO (Swin-attention sequence mixer). |
| `model_segmamba/segmamba.py` | Original SegMamba backbone (Mamba SSM mixer) — **B0** & the low-precision reference. |
| `configs/` | One YAML per paper arm (B0, S1, A1, A2, A3, A5, A6, A7, A8, C1, C2). |
| `scripts/` | Training / inference / evaluation / metric scripts. |
| `verify_absorption.py` | **§3.2 exact audit** — folds the learned Q into the subband mixer and reproduces each trained network to float64 round-off *from the checkpoint alone* (CPU, no data, no GPU). |
| `analysis/absorption_sites.csv` | Per-site audit records for the nine trained checkpoints (drift ‖Q−I‖_F, escape fraction ρ). |

## Paper arms and their configs

| ID | Configuration | Wavelet | Subband mixer | Config file |
|---|---|---|---|---|
| B0 | SegMamba, learned strided conv (baseline) | — | — | `configs/B0_segmamba.yaml` |
| S1 | SwinUNETR baseline | — | — | `configs/S1_swin_unetr.yaml` |
| A1 | fixed Haar | fixed (θ=π/4, Q=I) | groups=1 | `configs/A1_fixed.yaml` |
| A2 | **fixed Haar (best)** | fixed | groups=8 | `configs/A2_fixed_g8.yaml` |
| A3 | theta | learnable θ (3 DOF) | groups=1 | `configs/A3_theta.yaml` |
| A5 | so8 | learnable Q (28 DOF) | groups=1 | `configs/A5_so8.yaml` |
| A6 | so8 | learnable Q | groups=8 | `configs/A6_so8_live.yaml` |
| A7 | so8 + wavelet-boundary loss | learnable Q | groups=1 | `configs/A7_so8_wb.yaml` |
| A8 | so8, per-channel mixer+wavelet | learnable Q (per-channel) | per-channel | `configs/A8_per_channel.yaml` |
| C1 | SegAttn + Haar | learnable Q | groups=1 | `configs/C1_segattn_haar.yaml` |
| C2 | SegAttn + routing | learnable Q | groups=1 + router | `configs/C2_segattn_routed.yaml` |

## Quick start

### Requirements

- Python ≥ 3.9, PyTorch ≥ 2.0
- `monai` (networks/blocks/inferers) — install via `pip install monai`
- `mamba-ssm` + `causal-conv1d` for the Mamba backbone (`B0`, all `segmamba*` arms)
- `batchgenerators` for preprocessing

### Verify the absorption identity (CPU, no data — the §3.2 audit)

```bash
# against a released checkpoint
python verify_absorption.py logs/segmamba_haar_so8_dicece_ds/model/best_model_0.9317.pt
```

This does **not** load the network, does **not** need a GPU or BraTS data. It reads
the state dict of any trained LOWO checkpoint, folds each learned basis Q into the
adjacent subband mixer, and compares the folded network against the original to
machine precision. You should see:

```text
wavelet sites: 16
max fold error: 7.3e-15
inert (ρ≈0): 8/16
```

### Reproduce a paper arm (example: A6, so8 + groups=8)

```bash
# 1. data -> ./data/fullres/train (BraTS2021, labels remapped 4→3)
python scripts/rename_brats2021_data.py
python scripts/preprocessing_brats2021.py

# 2. train (500 epochs, DiceCE + deep supervision)
python train_brats2021.py --config configs/A6_so8_live.yaml --tag segmamba_haar_so8_live

# 3. inference (sliding window, overlap 0.7, 8-way mirror TTA)
python predict.py --ckpt logs/segmamba_haar_so8_live/.../best_model_*.pt

# 4. metrics (Dice / HD95, BraTS convention)
python compute_metrics.py --pred_name segmamba_haar_so8_live
```

All arms share the identical evaluation protocol (single split, same 250-case test
manifest, same post-processing), so per-arm numbers in the paper are directly
comparable.

## Repository layout rationale

- **`model_segmamba/lowo.py` is deliberately dependency-free** (torch only), so the
  operator can be dropped into any CNN / attention / SSM backbone — the paper's
  "drop-in replacement" claim is embodied in the file layout.
- **`verify_absorption.py` is deliberately data-free**: identifiability is an
  architectural property, verified exactly per checkpoint, not from aggregate
  metrics.
- Training framework (`light_training/`) is included verbatim so the exact
  optimizer / LR schedule / deep-supervision helper used in the paper is
  reproducible.

## Reproducing Table 1 (main results)

| Command | Variant |
|---|---|
| `python train_brats2021.py --config configs/B0_segmamba.yaml` | B0 |
| `python train_swin_unetr_brats2021.py --config configs/S1_swin_unetr.yaml` | S1 |
| `python train_brats2021.py --config configs/A1_fixed.yaml` | A1 |
| `python train_brats2021.py --config configs/A2_fixed_g8.yaml` | A2 |
| `python train_brats2021.py --config configs/A3_theta.yaml` | A3 |
| `python train_brats2021.py --config configs/A5_so8.yaml` | A5 |
| `python train_brats2021.py --config configs/A6_so8_live.yaml` | A6 |
| `python train_brats2021.py --config configs/A7_so8_wb.yaml` | A7 |
| `python train_brats2021.py --config configs/A8_per_channel.yaml` | A8 |
| `python train_brats2021_attn.py --config configs/C1_segattn_haar.yaml` | C1 |
| `python train_brats2021_attn.py --config configs/C2_segattn_routed.yaml` | C2 |

## License

Code released for research reproducibility. BraTS2021 data follows its own
usage terms (see https://www.synapse.org/#!Synapse:syn27046444).