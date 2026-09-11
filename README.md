# When Does a Learned Wavelet Basis Matter?

**An Absorption Audit for 3D Resampling** — [ICASSP 2027 submission]

Code for the Learned Orthogonal Wavelet Operator (LOWO) and the
checkpoint-only absorption audit.

Paper: *When Does a Learned Wavelet Basis Matter? An Absorption Audit for 3D Resampling*

Code: [https://github.com/zhangjing2526/lowo](https://github.com/zhangjing2526/lowo)

A learned orthogonal wavelet sitting upstream of a linear mixer is not
identified by its coefficients: any rotation \(Q\) can be folded into the
next layer.  Folding the learned rotation into the adjacent mixer
reproduces ten trained networks to machine precision (**163 sites**, max
fold error \(7.8\times10^{-15}\)).  The per-site escape fraction \(\rho\)
is zero at every shared mixer, including after a large rotation
(\(\rho=0\) at **91 of 163** sites).  On 250 internal held-out BraTS2021
volumes, fixed Haar with a per-subband mixer (A2, the 3D instantiation of
[HWD](https://arxiv.org/abs/2304.08366)) showed no detectable difference
from a learned-convolution backbone at \(12\%\) fewer parameters.  Making
the rotation learnable where it is locally non-absorbable does not
improve Dice across three seeds (fixed-minus-learned: \(+0.79\),
\(+0.84\), \(-0.85\)).

```text
┌───────────────────────────────────────────┐
│  K(θ, A) = Q(A) · K_haar(θ)               │
│  Q(A) = exp(A − Aᵀ) ∈ SO(8)               │
│  A is 8×8 (64 scalars); Q has 28 eff. DOF │
└───────────────────────────────────────────┘
         │ stride-2 conv³D (DWT fused)          (IDWT = same kernel, transposed)
         ▼
  subband-major 8C feature map ──► 1×1 subband mixer
       groups=1 absorbs Q                 groups=8 can make Q load-bearing
```

---

## What is in this repository

| Path | Contents |
|---|---|
| `model_segmamba/lowo.py` | **The LOWO operator itself** — `HaarWavelet3D`, `HaarDownsample`, `HaarUpsample`, `WaveletConv3d`. Backbone-agnostic (only `torch`). |
| `model_segmamba/haar_blocks.py` | UNet decoder blocks built on LOWO (`WaveletGSC`, `HaarUpBlock`, `MlpChannel`). |
| `model_segmamba/segmamba_haar.py` | SegMamba + LOWO (encoder/decoder wiring). |
| `model_segmamba/segmamba_attn.py` | SegAttn + LOWO (Swin-attention sequence mixer). C1/C2. |
| `model_segmamba/segmamba.py` | Original SegMamba backbone (Mamba SSM mixer) — **B0** & the low-precision reference. |
| `configs/` | One YAML per paper arm. IDs match the paper; filenames encode the factors (wavelet × mixer). |
| `verify_absorption.py` | **Exact audit** — analysis mixers fold as \(W'=W(Q\otimes I)\), decoder as \(Q^\top W\), WaveletConv as \(R^\top WR\). Reproduces each trained map to float64 round-off from the checkpoint alone (CPU, no data, no GPU). |
| `checkpoints/a6_so8_g8_audit.pt` | Wavelet `Q` and adjacent mixer tensors from the A6 checkpoint (not the full 226 MB training weight). |
| `analysis/absorption_sites.csv` | Per-site audit records for the ten trained checkpoints (163 sites: A1–A8, C1, C2; drift \(\|Q-I\|_F\), escape fraction \(\rho\), fold error). One checkpoint per arm; A4 is seed 123. |
| `analysis/a4_sites.csv` | A4 sites across three seeds. Only seed 123 is merged into the 163-site table. |
| `run_test_eval.py` | Held-out Dice / HD95 protocol (8-way TTA, overlap 0.7, `--force_ethr 50`). Needs BraTS data and a full training checkpoint. Defaults to `splits/test_250.txt`. |
| `splits/test_250.txt` | Frozen 250-case **internal held-out** list used in the paper (not the official BraTS test set). |
| `splits/val_125.txt` | Frozen 125-case validation list used for checkpoint selection. |
| `requirements.txt` | Python dependencies. |

A positive \(\rho\) means \(Q\) is **load-bearing relative to that mixer
class** (locally non-absorbable).  It is not unique recovery of \(Q\) from
the network: sign, permutation, and downstream layers remain.

`splits/test_250.txt` and `splits/val_125.txt` are an **internal held-out
subset** of the BraTS2021 training pool, not the official BraTS test set.
Training and eval load these lists automatically so the partition does not
depend on `glob` order. The training complement is every remaining case
under the npz directory.

## Paper arms and their configs

Paper IDs are unchanged. Filenames encode the ablation factors so A1–A6 read as a grid, not a flat list.

`wavelet_mode: so8` + `learnable_wavelet: false` is exactly standard Haar (`Q = I`, `θ = π/4`).

### Baselines

| ID | Backbone | Config |
|---|---|---|
| B0 | SegMamba, learned stride-2 conv | `configs/B0_segmamba.yaml` |
| S1 | SwinUNETR (unweighted CE, no deep supervision) | `configs/S1_swin_unetr.yaml` |

### Main ablation (SegMamba + LOWO)

| | mixer `groups=1` | mixer `groups=8` |
|---|---|---|
| fixed Haar (`Q = I`) | **A1** `configs/A1_fixed_g1.yaml` | **A2** `configs/A2_fixed_g8.yaml` (recommended 3D HWD rung) |
| learnable θ (3 DOF) | **A3** `configs/A3_theta_g1.yaml` | **A4** `configs/A4_theta_g8.yaml` |
| learnable \(Q\in SO(8)\) (28 effective DOF) | **A5** `configs/A5_so8_g1.yaml` | **A6** `configs/A6_so8_g8.yaml` |

A2 and A6 differ only in whether \(A\) is trainable.  Both use `groups=8`
on the resampling path, so \(Q\) is locally non-absorbable; learning it
does not reliably change Dice.

### Probes (one extra factor on top of a main arm)

| ID | Added on | Extra factor | Config |
|---|---|---|---|
| A7 | A5 | wavelet-boundary loss (`weight=0.1`) | `configs/A7_so8_g1_wbloss.yaml` |
| A8 | A5 | per-channel wavelet + mixer | `configs/A8_so8_per_channel.yaml` |
| C1 | A5 wavelet | SegAttn backbone (window attention) | `configs/C1_segattn_so8_g1.yaml` |
| C2 | C1 | routing probe | `configs/C2_segattn_so8_g1_routed.yaml` |

## Quick start

```bash
pip install -r requirements.txt
```

- Python ≥ 3.9, PyTorch ≥ 2.0
- `monai` for networks / sliding-window inference
- `mamba-ssm` + `causal-conv1d` only for the Mamba backbone (`B0`, A1–A8). C1/C2 and `verify_absorption.py` do not need them.

### Verify the absorption identity (CPU, no data — the paper's exact audit)

```bash
# 1. no download: random Q, W  (proves the identity; numbers will not match the paper)
python verify_absorption.py --synthetic

# 2. the paper's A6 tensors (in-repo, ~few MB) — A6's own max fold is 7.3e-15
python verify_absorption.py checkpoints/a6_so8_g8_audit.pt

# 3. paper-wide counts from the tracked CSV (163 sites, 91 with ρ=0, max 7.8e-15)
python -c "import csv; r=list(csv.DictReader(open('analysis/absorption_sites.csv'))); print(len(r), sum(float(x['rho'])<1e-12 for x in r), max(float(x['fold_err']) for x in r))"
```

Neither of the first two commands loads the network, needs a GPU, or touches BraTS. The script
reads wavelet `Q` and the adjacent 1×1 mixer, folds `Q` into the mixer, and
compares the folded operator against the original to machine precision.

On the A6 audit file you should see:

```text
wavelet sites: 16
max fold error: 7.3e-15
inert (ρ≈0): 4/16
```

(A6 uses `groups=8` on the encoder mixers, so only the four decoder upsamples are absorbed. A5 with `groups=1` is 8/16 absorbed.)

The paper-wide figure is larger because A4 is in the 163-site table:
`vit.gscs.2.proj2` there is \(7.8\times10^{-15}\).  Command 3 prints
`163 91 7.847...e-15`.

A full training checkpoint (`best_model_0.9317.pt`, ~226 MB) is **not** required
for this audit. Host that file on a GitHub Release / Zenodo / Hugging Face
dataset if you want to reproduce Dice; do not put it on a region-locked drive.

### Reproduce a paper arm (example: A6, learnable Q + groups=8)

```bash
# 1. data -> ./data/fullres/train (BraTS2021, labels remapped 4→3)
python scripts/rename_brats2021_data.py
python scripts/preprocessing_brats2021.py

# 2. train (500 epochs, DiceCE + deep supervision)
python train_brats2021.py --config configs/A6_so8_g8.yaml --tag a6_so8_g8

# 3. inference + metrics (sliding window, overlap 0.7, 8-way mirror TTA,
#    ET→NCR volume threshold fixed at 50 — the protocol behind Table 1)
python run_test_eval.py --config configs/A6_so8_g8.yaml \
    --ckpt logs/segmamba_haar_so8_dicece_ds/a6_so8_g8_s123/model/best_model_*.pt \
    --force_ethr 50 --stage test
```

C1 / C2 use the same trainer (`model: segattn_haar` in the YAML) and do not
import `mamba_ssm`:

```bash
python train_brats2021.py --config configs/C1_segattn_so8_g1.yaml
python train_brats2021_attn.py   # equivalent, attention-only script
```

All arms share the identical **evaluation** protocol (same internal held-out
250-case list, overlap 0.7, 8-way TTA, ET→NCR threshold 50). S1 was trained
with unweighted CE and no deep supervision; every other paper arm uses
weighted CE+Tversky and deep supervision. Dice reproduction still needs
BraTS2021 and a full training checkpoint; the absorption audit does not.

## Repository layout rationale

- **`model_segmamba/lowo.py` is deliberately dependency-free** (torch only), so the
  operator can be dropped into any CNN / attention / SSM backbone — the paper's
  "drop-in replacement" claim is embodied in the file layout.
- **`verify_absorption.py` is deliberately data-free**: local absorbability is an
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
| `python train_brats2021.py --config configs/A1_fixed_g1.yaml` | A1 |
| `python train_brats2021.py --config configs/A2_fixed_g8.yaml` | A2 |
| `python train_brats2021.py --config configs/A3_theta_g1.yaml` | A3 |
| `python train_brats2021.py --config configs/A4_theta_g8.yaml` | A4 |
| `python train_brats2021.py --config configs/A5_so8_g1.yaml` | A5 |
| `python train_brats2021.py --config configs/A6_so8_g8.yaml` | A6 |
| `python train_brats2021.py --config configs/A7_so8_g1_wbloss.yaml` | A7 |
| `python train_brats2021.py --config configs/A8_so8_per_channel.yaml` | A8 |
| `python train_brats2021.py --config configs/C1_segattn_so8_g1.yaml` | C1 |
| `python train_brats2021.py --config configs/C2_segattn_so8_g1_routed.yaml` | C2 |

Eval (same protocol for every arm; `splits/test_250.txt` is the default):

```bash
python run_test_eval.py --config configs/<arm>.yaml \
    --ckpt <full_training_weight.pt> \
    --force_ethr 50 --stage test
```

## License

Code released for research reproducibility. BraTS2021 data follows its own
usage terms (see https://www.synapse.org/#!Synapse:syn27046444).
