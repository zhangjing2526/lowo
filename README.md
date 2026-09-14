# When Is a Learned Orthogonal Transform Non-Removable?

**A Checkpoint-Only Absorption Audit for 3D Resampling** — [ICASSP 2027 submission]

Code for the **absorption audit** and the **Learned Orthogonal Wavelet Operator
(LOWO)** used as its controlled testbed.

Paper: *When Is a Learned Orthogonal Transform Non-Removable? A Checkpoint-Only
Audit for 3D Resampling*

Code: [https://github.com/zhangjing2526/lowo](https://github.com/zhangjing2526/lowo)

A learned orthogonal transform can sit far from its initialization
(\(\|Q-I\|_F=3.53\) of a \(5.66\) maximum) and reshape every analysis filter,
yet be removed *exactly*: the adjacent \(1{\times}1\) mixer absorbs it.  We
separate three routinely conflated questions — **structural validity** (is the
operator orthogonal / PR?), **functional necessity** (does the trained map still
require the learned factor, or does the adjacent mixer absorb it?), and **task
utility** (does a non-absorbable factor improve held-out performance?) — and
contribute a **checkpoint-only absorption audit**: with no data and no
retraining, the learned factor is folded into the adjacent mixer, giving a
per-site escape fraction \(\rho\) against the mixer class, where \(\rho=0\) is
an exact equivalence.

Over **163 sites in ten trained 3D checkpoints** the fold reproduces every
network to float64 round-off (\(5.9{\times}10^{-15}\)): 59 non-trivial rotations
are exactly removable, every dense mixer gives \(\rho=0\), and only a
*per-subband* mixer lets a generic rotation become non-removable.  Closing that
loophole is necessary for attributing a learned basis but **not sufficient** for
a gain (three seeds: \(+0.79\), \(+0.85\), \(-0.85\) Dice points).  LOWO is our
controlled testbed; **fixed Haar with a per-subband mixer (grouped 3D Haar control) is a
*necessary attribution control* before any claim that a learned factor is
used** — not a recommended architecture.

```text
┌───────────────────────────────────────────────────────────┐
│  K(θ, A) = Q(A) · K_haar(θ)                               │
│  Q(A) = exp(A − Aᵀ) ∈ SO(8)                               │
│  A is 8×8 (64 scalars); Q has 28 effective DOF           │
│  θ = (π/4, π/4, π/4)  →  bit-identical fixed Haar (Q=I)  │
└───────────────────────────────────────────────────────────┘
         │ stride-2 conv³D (DWT fused)          (IDWT = same kernel, transposed)
         ▼
  subband-major 8C feature map ──► 1×1 subband mixer
       groups=1 absorbs Q     groups=8 makes Q locally non-removable under the specified adjacent mixer class
```

---

## What is in this repository

| Path | Contents |
|---|---|
| `model_segmamba/lowo.py` | **The LOWO operator itself** — `HaarWavelet3D`, `HaarDownsample`, `HaarUpsample`, `WaveletConv3d`. Backbone-agnostic (only `torch`). Modes: fixed Haar (`Q=I`), separable-angle `theta` (3 DOF), `so8` (\(Q=\exp(A-A^\top)\in SO(8)\), 28 DOF), and `per_channel`. |
| `model_segmamba/haar_blocks.py` | UNet decoder blocks built on LOWO (`WaveletGSC`, `HaarUpBlock`, `MlpChannel`). |
| `model_segmamba/segmamba_haar.py` | SegMamba + LOWO (encoder/decoder wiring). |
| `model_segmamba/segmamba_attn.py` | SegAttn + LOWO (window-attention sequence mixer). C1/C2. |
| `model_segmamba/segmamba.py` | Original SegMamba backbone (Mamba SSM mixer) — **B0** & the low-precision reference. |
| `configs/` | One YAML per paper arm. IDs match the paper; filenames encode the factors (wavelet × mixer). |
| `verify_absorption.py` | **Exact audit** — analysis mixers fold as \(W'=W(Q\otimes I)\), decoder as \(Q^\top W\), WaveletConv as \(R^\top WR\). Reproduces each trained map to float64 round-off from the checkpoint alone (CPU, no data, no GPU). |
| `checkpoints/a6_so8_g8_audit.pt` | Wavelet `Q` and adjacent mixer tensors from the A6 checkpoint (not the full 226 MB training weight). |
| `analysis/absorption_sites.csv` | Per-site audit records for the ten trained checkpoints (163 sites: A1–A8, C1, C2; drift \(\|Q-I\|_F\), escape fraction \(\rho\), fold error). One checkpoint per arm; A4 is seed 123. |
| `analysis/a4_sites.csv` | A4 sites across three seeds. Only seed 123 is merged into the 163-site table. |
| `run_test_eval.py` | Held-out Dice / HD95 protocol (8-way TTA, overlap 0.7, `--force_ethr 50`). Defaults to the frozen `splits/test_250.txt`. Needs BraTS data and a full training checkpoint. |
| `splits/test_250.txt` | Frozen 250-case **internal held-out** list used in the paper (not the official BraTS test set). |
| `splits/val_125.txt` | Frozen 125-case validation list used for checkpoint selection. |
| `cifar2d/` | 2D probe: \(SO(4)\) Haar downsample on CIFAR ResNet-18. Analysis fold only (`mix(dwt(x))`). Scripts, no weights. |
| `figs/` | The three paper figures (PDF) **plus their editable sources**: the absorption fold (`fig1_absorption.pdf`, source `scripts/fig1_method_overview.py` → `figs/fig1_method_overview.svg`), the audit over 163 sites (`drift_vs_rho.pdf`, source `scripts/make_rho_scatter.py` / `scripts/replot_rho_scatter.py`), and the A2/A6 paired difference (`a2_a6_paired_diff.pdf`, source `scripts/make_a2_a6_paired_diff.py`). |
| `requirements.txt` | Python dependencies. |

A positive \(\rho\) means \(Q\) is **locally non-removable under the specified adjacent mixer
class** (locally non-absorbable by the adjacent mixer).  It is not unique recovery of \(Q\) from
the network: sign, permutation, and downstream layers remain.

`splits/test_250.txt` and `splits/val_125.txt` are an **internal held-out
subset** of the BraTS2021 training pool, not the official BraTS test set.
Training and eval load these lists automatically so the partition does not
depend on `glob` order. The training complement is every remaining case
under the npz directory.

## Three questions, three answers

Work on learned filter banks routinely conflates three questions:

1. **Structural validity** — is the operator orthogonal and perfect-reconstruction (PR)? *Follows from the construction (Section "LOWO: A Controlled Testbed").*
2. **Functional necessity** — does the trained map still require the learned factor, or can the adjacent mixer absorb it exactly? *Decided by the mixer class, not by the coefficients (the audit).*
3. **Task utility** — does a non-absorbable factor improve held-out performance? *Does not follow from (ii): three seeds give no stable Dice direction (Section "Task utility").*

LOWO is the **vehicle**, not the contribution: a learned orthogonal
wavelet-initialized operator that manufactures both verdicts inside one network
family, so the audit can be observed exactly, from a checkpoint, without a
second dataset.

## Paper arms and their configs

`wavelet_mode: so8` + `learnable_wavelet: false` is exactly standard Haar
(\(Q = I\), \(\theta = \pi/4\)).  Two knobs manufacture the audit's two
verdicts: whether \(Q\) is frozen at \(I\) (the 0-DOF rung) or trainable
(28-DOF), and whether the adjacent mixer is dense (`groups=1`) or per-subband
(`groups=8`).

### Baselines

| ID | Backbone | Config |
|---|---|---|
| B0 | SegMamba, learned stride-2 conv (67.42 M params) | `configs/B0_segmamba.yaml` |
| S1 | SwinUNETR (unweighted CE, no deep supervision) | `configs/S1_swin_unetr.yaml` |

### Main ablation (SegMamba + LOWO)

| | mixer `groups=1` | mixer `groups=8` |
|---|---|---|
| fixed Haar (`Q = I`) | **A1** `configs/A1_fixed_g1.yaml` | **A2** `configs/A2_fixed_g8.yaml` (grouped 3D Haar control on BraTS) |
| learnable θ (3 DOF) | **A3** `configs/A3_theta_g1.yaml` | **A4** `configs/A4_theta_g8.yaml` |
| learnable \(Q\in SO(8)\) (28 effective DOF) | **A5** `configs/A5_so8_g1.yaml` | **A6** `configs/A6_so8_g8.yaml` |

A2 and A6 differ only in whether \(A\) is trainable.  `groups=8` makes a generic
learned \(Q\) locally non-absorbable; A2 remains the \(Q=I\) control, and learning
\(Q\) shows no consistent Dice gain on BraTS.  LOWO replaces, without changing the
backbone, the stem, stride-2 downsampling, transposed upsampling, and an
equal-channel \(3{\times}3{\times}3\) convolution: **16 filter-bank sites per
checkpoint**.

### Probes (one extra factor on top of a main arm)

| ID | Added on | Extra factor | Config |
|---|---|---|---|
| A7 | A5 | wavelet-boundary loss (`weight=0.1`) | `configs/A7_so8_g1_wbloss.yaml` |
| A8 | A5 | per-channel wavelet + mixer | `configs/A8_so8_per_channel.yaml` |
| C1 | A5 wavelet | SegAttn backbone (window attention) | `configs/C1_segattn_so8_g1.yaml` |
| C2 | C1 | routing probe | `configs/C2_segattn_so8_g1_routed.yaml` |

The audit cohort is every configuration we trained: A1–A8 plus C1 (a
window-attention backbone with a dense downsample mixer) and C2 (C1 with a
routing probe) — **ten checkpoints, 163 sites**.

## Main results (internal held-out BraTS2021, 250 volumes, three seeds)

| Config. | Par. (M) | 123 | 456 | 789 | mean\(\pm\)std |
|---|---|---|---|---|---|
| B0 learned stride conv | 67.42 | 0.8987 | 0.8828 | 0.8928 | \(0.8914{\pm}0.0080\) |
| A2 fixed Haar, \(Q=I\) | 59.25 | 0.9017 | 0.9016 | 0.8936 | \(0.8990{\pm}0.0046\) |
| A6 \(SO(8)\), \(Q\) learned | 59.25 | 0.8938 | 0.8931 | 0.9021 | \(0.8963{\pm}0.0050\) |
| A2\(-\)A6 | — | \(+0.79\) | \(+0.85\) | \(-0.85\) | \(+0.26{\pm}0.96\) |

A2 uses **12% fewer parameters** than B0 (59.25 M vs 67.42 M), but the saving
belongs to the full A2 replacement (the Haar resampler accounts for 7.49 M and
the grouping for 0.68 M); we do not attribute the reduction to fixed Haar alone.
Over three seeds B0 averages \(0.8914{\pm}0.0080\): A2 is a smaller rung of
comparable accuracy rather than a demonstrated win, and we do not call the two
compatible (no pre-specified non-inferiority margin).  The sign reverses once,
so we report **no stable Dice direction** from the 28 degrees of freedom per
site under this protocol.  What survives is weaker and more useful: once the
mixer already makes a generic \(Q\) locally non-removable under the specified adjacent mixer class, allocating \(SO(8)\) capacity
there yields no stable held-out gain.  **Non-removability is necessary for
attributing a learned basis and not sufficient for a performance claim.**

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

# 2. the paper's A6 tensors (in-repo, ~few MB) — A6's own max fold is 4.3e-15
python verify_absorption.py checkpoints/a6_so8_g8_audit.pt

# 3. paper-wide counts from the tracked CSV (163 sites, 91 with ρ=0, max 5.9e-15)
python -c "import csv; r=list(csv.DictReader(open('analysis/absorption_sites.csv'))); print(len(r), sum(float(x['rho'])<1e-12 for x in r), max(float(x['fold_err']) for x in r))"
```

Neither of the first two commands loads the network, needs a GPU, or touches
BraTS. The script reads wavelet `Q` and the adjacent 1×1 mixer, folds `Q` into
the mixer, and compares the folded operator against the original to machine
precision.

On the A6 audit file you should see:

```text
wavelet sites: 16
max fold error: 4.3e-15
inert (ρ≈0): 4/16
```

(A6 uses `groups=8` on the encoder mixers, so only the four decoder upsamples
are absorbed. A5 with `groups=1` is 8/16 absorbed.)

The paper-wide figure is the maximum over all 163 sites in
`analysis/absorption_sites.csv`.  Command 3 prints `163 91 5.92...e-15`.

A full training checkpoint (`best_model_0.9317.pt`, ~226 MB) is **not** required
for this audit. Host that file on a GitHub Release / Zenodo / Hugging Face
dataset if you want to reproduce Dice; do not put it on a region-locked drive.

### Reproduce a paper arm (example: A6, learnable Q + groups=8)

```bash
# 1. data -> ./data/fullres/train (BraTS2021, labels remapped 4→3)
python scripts/rename_brats2021_data.py
python scripts/preprocessing_brats2021.py

# 2. train (500 epochs, weighted CE+Tversky + deep supervision, poly-SGD)
python train_brats2021.py --config configs/A6_so8_g8.yaml --tag a6_so8_g8

# 3. inference + metrics (sliding window, overlap 0.7, 8-way mirror TTA,
#    ET→NCR volume threshold fixed at 50 — the protocol behind the table)
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
weighted CE+Tversky and deep supervision. `run_test_eval.py` defaults to the
shipped frozen lists (`splits/test_250.txt`, `splits/val_125.txt`); pass
`--test_list ""` only if you deliberately want to re-derive the split (which is
not portable across hosts). Dice reproduction still needs BraTS2021 and a full
training checkpoint; the absorption audit does not.

## Repository layout rationale

- **`model_segmamba/lowo.py` is deliberately dependency-free** (torch only), so
  the operator can be dropped into any CNN / attention / SSM backbone — the
  paper's "drop-in replacement" claim is embodied in the file layout.
- **`verify_absorption.py` is deliberately data-free**: local absorbability is an
  architectural property, verified exactly per checkpoint, not from aggregate
  metrics.  `cifar2d/verify_absorption_2d.py` is the same identity on analysis
  sites only (\(SO(4)\), fold \(W'=WR\)).
- Training framework (`light_training/`) is included verbatim so the exact
  optimizer / LR schedule / deep-supervision helper used in the paper is
  reproducible.

## Reproducing the main table

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

## CIFAR-10 2D probe (`cifar2d/`)

ResNet-18, no stem DWT.  Haar sites are layer2/3/4 × (main `conv1` + skip) =
**6 analysis sites**.  Fold is \(W'=WR\) (`groups=1` absorbs; `groups=4` can
make \(Q\) locally non-removable under the specified adjacent mixer class).  There is no sandwich `idwt∘W∘dwt` in this probe.

| Arm | Downsample | Paper analogue |
|---|---|---|
| B0 | stride-2 conv | learned resample |
| A2 | fixed Haar, `groups=4` | 2D HWD |
| A5 | learnable \(Q\in SO(4)\), `groups=1` | absorbable |
| A6 | learnable \(Q\in SO(4)\), `groups=4` | locally non-removable under the specified adjacent mixer class, 6 DOF |

`wave.skew` is in a **wd=0** group.  Report official-test top-1 at the best
validation epoch (5k hold-out from the 50k train split).  Three seeds:
A2 \(94.44\) vs B0 \(95.14\); A6 \(94.64\) does not beat A2.  The audit
transfers; A2\(\approx\)B0 does not.

```bash
cd cifar2d
python train_cifar.py --selftest
python train_cifar.py --arm A2 --seed 123 --device cuda:0
python verify_absorption_2d.py --synthetic          # no checkpoint needed
python verify_absorption_2d.py runs/a6_s123/best.pt  # after training
python mcnemar.py --all --runs runs
```

Needs `torchvision`.  Do not commit `cifar2d/data/` or `cifar2d/runs/`.

## License

Code released for research reproducibility. BraTS2021 data follows its own
usage terms (see https://www.synapse.org/#!Synapse:syn27046444).
