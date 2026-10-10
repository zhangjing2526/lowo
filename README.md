# When Is a Learned Orthogonal Transform Non-Removable?

**A Checkpoint-Only Absorption Audit for 3D Resampling** — [ICASSP 2027 submission]

Code for the **absorption audit** and the **Learned Orthogonal Wavelet Operator
(LOWO)** used as its controlled testbed.

Paper: *When Is a Learned Orthogonal Transform Non-Removable? A Checkpoint-Only
Audit for 3D Resampling*

Code: [https://github.com/zhangjing2526/lowo](https://github.com/zhangjing2526/lowo)

A learned orthogonal transform can move far from its initialization yet remain
exactly absorbable by an adjacent dense mixer. This repository accompanies a
**checkpoint-only, transform-level absorption audit**: it asks whether folding
the learned factor preserves the adjacent mixer's prescribed connectivity.

We separate **structural validity** (orthogonality and perfect reconstruction),
**adjacent-mixer removability** (exact folding within the allowed mixer class),
and **task utility** (held-out performance). Across **10 checkpoints and 163
transform locations**, original and unrestricted merged blocks agree to a
maximum relative output error of **5.9×10⁻¹⁵ in FP64**. This is a per-block
implementation check, not a whole-network error bound or evidence that every
fold preserves grouped connectivity. The audit finds 32 frozen identity
factors and 59 learned dense-mixer factors with zero escape, and 72 learned
grouped-mixer factors with positive escape.

On BraTS2021, fixed-Haar A2 and separable-learnable A4 use **12.1% fewer
parameters** than B0 and achieve higher Dice in all three runs; A4 has the
highest mean Dice (**0.9015 ± 0.0007**). Learning full rotations in A6 gives
no consistent additional gain over A2. LOWO is the controlled testbed for
these structural and performance comparisons.

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
       groups=1 absorbs Q     groups=8 can obstruct absorption (see criterion below)
```

---

## What is in this repository

| Path | Contents |
|---|---|
| `model_segmamba/lowo.py` | **The LOWO operator itself** — `HaarWavelet3D`, `HaarDownsample`, `HaarUpsample`, `WaveletConv3d`. Backbone-agnostic (only `torch`). Modes: fixed Haar (`Q=I`), separable-angle `theta` (3 DOF), `so8` ($Q=\exp(A-A^\top)\in SO(8)$, 28 DOF), and `per_channel`. |
| `model_segmamba/haar_blocks.py` | UNet decoder blocks built on LOWO (`WaveletGSC`, `HaarUpBlock`, `MlpChannel`). |
| `model_segmamba/segmamba_haar.py` | SegMamba + LOWO (encoder/decoder wiring). |
| `model_segmamba/segmamba_attn.py` | SegAttn + LOWO (window-attention sequence mixer). C1/C2. |
| `model_segmamba/segmamba.py` | Original SegMamba backbone (Mamba SSM mixer) — **B0** & the low-precision reference. |
| `configs/` | One YAML per paper arm. IDs match the paper; filenames encode the factors (wavelet × mixer). |
| `verify_absorption.py` | **Exact audit** — analysis mixers fold as $W'=W(Q\otimes I)$, decoder as $R^\top W$, WaveletConv as $R^\top WR$. Checks each audited block against its unrestricted fold to float64 round-off from the checkpoint alone (CPU, no data, no GPU). |
| `checkpoints/a6_so8_g8_audit.pt` | Wavelet `Q` and adjacent mixer tensors from the A6 checkpoint (not the full 226 MB training weight). |
| `analysis/absorption_sites.csv` | Per-site audit records for the ten trained checkpoints (163 sites: A1–A8, C1, C2; drift $\|Q-I\|_F$, escape fraction $\rho$, fold error). One checkpoint per arm; A4 is seed 123. |
| `analysis/a4_sites.csv` | A4 sites across three seeds. Only seed 123 is merged into the 163-site table. |
| `run_test_eval.py` | Held-out Dice / HD95 protocol (8-way TTA, overlap 0.7, `--force_ethr 50`). Defaults to the frozen `splits/test_250.txt`. Needs BraTS data and a full training checkpoint. |
| `splits/test_250.txt` | Frozen 250-case **internal held-out** list used in the paper (not the official BraTS test set). |
| `splits/val_125.txt` | Frozen 125-case validation list used for checkpoint selection. |
| `cifar2d/` | 2D probe: $SO(4)$ Haar downsample on CIFAR ResNet-18. Analysis fold only (`mix(dwt(x))`). Scripts, no weights. |
| `scripts/fold_sandwich_projection.py` (+ `fold_all_sites_check.py`, `fold_audit_supplement.py`, `repro_s32_sandwich_dice.py`) | **§3.2 grouped-sandwich Dice audit** — from the A6 checkpoint, pick low/mid/high-ρ sandwich sites and build exact-reparam vs forced block-diagonal variants; reports mean Dice change (<4×10⁻⁶) and worst-case drift (0.0038→0.344) on the frozen 250 cases. |
| `oft_absorption_audit/` | External-operator OFT merge audit: a self-trained OFT adapter on a public DINOv2 ViT-Large backbone. 12 scripts + report JSONs; detailed experiment notes in `oft_absorption_audit/README.md`. |
| `figs/` | The two main-paper figures and an additional paired-difference figure (PDF) **plus their editable sources**: the absorption fold (`fig1_absorption.pdf`, source `scripts/fig1_method_overview.py` → `figs/fig1_method_overview.svg`), the audit over 163 sites (`drift_vs_rho.pdf`, source `scripts/make_rho_scatter.py` / `scripts/replot_rho_scatter.py`), and the A2/A6 paired difference (`a2_a6_paired_diff.pdf`, source `scripts/make_a2_a6_paired_diff.py`). |
| `requirements.txt` | Python dependencies. |

A positive $\rho$ means $Q$ is **locally non-removable under the specified adjacent mixer
class** (locally non-absorbable by the adjacent mixer).  It is not unique recovery of $Q$ from
the network: sign, permutation, and downstream layers remain.

`splits/test_250.txt` and `splits/val_125.txt` contain **internal held-out
subsets** of the BraTS2021 training pool, not the official BraTS test set.
The paper used **875 training / 125 validation / 250 test cases**, with split
seed 42; one of 1251 cases was excluded by split-size rounding.

Evaluation uses the shipped held-out lists. The published training loader
still derives its split from filesystem order and seed 42; a fixed seed alone
does not guarantee the same partition on another host. The original 875-case
training list and excluded case ID are not included in this release, so exact
training-split reproduction requires the original file order or saved lists.

The training entry points also pass a `dump_split_dir` argument that the
published loader does not yet accept. The training commands below identify the
intended configurations; end-to-end retraining requires resolving this interface
mismatch and checking the split. The checkpoint-only audit does not use that
loader, and held-out evaluation uses its own frozen-list override.

## Three questions, three answers

Work on learned filter banks routinely conflates three questions:

1. **Structural validity** — is the operator orthogonal and perfect-reconstruction (PR)? *Follows from the orthogonal construction, with even spatial sizes and no padding or cropping.*
2. **Adjacent-mixer removability** — can the prescribed adjacent mixer absorb the learned factor exactly? *Determined from the folded checkpoint weights and allowed connectivity.*
3. **Task utility** — does a non-absorbable factor improve held-out performance? *Does not follow from (ii): A2 versus A6 gives no stable Dice direction.*

LOWO is the **vehicle**, not the contribution: a learned orthogonal
wavelet-initialized operator that manufactures both verdicts inside one network
family, so the audit can be observed exactly, from a checkpoint, without a
second dataset.

## Absorption criterion (paper §2.2–2.3)

For shared channel-wise rotations, $R=Q\otimes I_C$. With an unrestricted bias,
the exact folds are:

| Arrangement | Folded weight | Folded bias |
|---|---|---|
| Analysis | $W'=WR$ | $b'=b$ |
| Synthesis | $W'=R^\top W$ | $b'=R^\top b$ |
| Sandwich | $W'=R^\top WR$ | $b'=R^\top b$ |

Let $P_g$ keep weights allowed by the original grouping and zero all others:

$$
\rho=\frac{\|W'-P_g(W')\|_F}{\|W'\|_F},
\qquad \rho=0\text{ when }W'=0.
$$

**Proposition 1:** for an invertible transform, linear adjacency and unrestricted
biases, absorption without changing the prescribed grouping is possible if and
only if $\rho=0$. The reported audit uses FP64 and a numerical zero threshold
of $10^{-12}$. An unrestricted dense fold remains an algebraic identity when
$\rho>0$, but may change the connectivity.

**Proposition 2:** for a shared sandwich rotation and
$W=\operatorname{diag}(W_0,\ldots,W_7)$, the fold stays block diagonal if and
only if each column of $Q$ has nonzero entries only on identical blocks $W_i$.
Identical blocks permit arbitrary shared rotations; pairwise-distinct blocks
permit only signed permutations compatible with $Q\in SO(8)$. Intermediate
cases permit rotations within identical-block classes and permutations.
Channel-specific rotations are outside Proposition 2; Proposition 1 still
applies. Thus `groups=8` does not make every possible rotation non-removable.

A positive $\rho$ is a local obstruction under the specified mixer class,
not unique recovery of a basis or a guarantee of a performance gain.

## Paper arms and their configs

`wavelet_mode: so8` + `learnable_wavelet: false` is exactly standard Haar
($Q = I$, $\theta = \pi/4$).  Two knobs manufacture the audit's two
verdicts: whether $Q$ is frozen at $I$ (the 0-DOF rung) or trainable
(28-DOF), and whether the adjacent mixer is dense (`groups=1`) or per-subband
(`groups=8`).

### Baseline and additional reference

| ID | Backbone | Config |
|---|---|---|
| B0 | SegMamba, learned stride-2 conv (67.42 M params) | `configs/B0_segmamba.yaml` |
| S1 (additional; outside Table 1) | SwinUNETR (unweighted CE, no deep supervision) | `configs/S1_swin_unetr.yaml` |

### Main ablation (SegMamba + LOWO)

| | mixer `groups=1` | mixer `groups=8` |
|---|---|---|
| fixed Haar (`Q = I`) | **A1** `configs/A1_fixed_g1.yaml` | **A2** `configs/A2_fixed_g8.yaml` (grouped 3D Haar control on BraTS) |
| learnable θ (3 DOF) | **A3** `configs/A3_theta_g1.yaml` | **A4** `configs/A4_theta_g8.yaml` |
| learnable $Q\in SO(8)$ (28 effective DOF) | **A5** `configs/A5_so8_g1.yaml` | **A6** `configs/A6_so8_g8.yaml` |

A2 and A6 differ only in whether $A$ is trainable.  `groups=8` makes a generic
learned $Q$ locally non-absorbable; A2 remains the $Q=I$ control, and learning
$Q$ shows no consistent Dice gain on BraTS.  LOWO replaces, without changing the
backbone, the stem, stride-2 downsampling, transposed upsampling, and an
equal-channel $3{\times}3{\times}3$ convolution: **16 filter-bank sites per standard checkpoint**: 4 analysis, 4 synthesis,
and 8 sandwich sites. Analysis grouping is the factor varied in the table;
synthesis mixers are dense and sandwich mixers are per-subband grouped.

### Probes (one extra factor on top of a main arm)

| ID | Added on | Extra factor | Config |
|---|---|---|---|
| A7 | A5 | wavelet-boundary loss (`weight=0.1`) | `configs/A7_so8_g1_wbloss.yaml` |
| A8 | A5 | per-channel wavelet + mixer | `configs/A8_so8_per_channel.yaml` |
| C1 | A5 wavelet | SegAttn backbone (window attention) | `configs/C1_segattn_so8_g1.yaml` |
| C2 | C1 | routing probe | `configs/C2_segattn_so8_g1_routed.yaml` |

The structural audit cohort is A1–A8 plus C1 (a
window-attention backbone with a dense downsample mixer) and C2 (C1 with a
routing probe) — **ten checkpoints, 163 sites**. A8 has 19 locations because of its
modality-specific stem factors; the other nine checkpoints have 16 each.
A7, A8, C1 and C2 are structural probes outside the segmentation table.

## Main results (paper Table 1)

Mean Dice over TC, WT and ET on the fixed 250-case internal held-out split:

| ID | Configuration | 123 | 456 | 789 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| B0 | Strided convolution | 0.8987 | 0.8828 | 0.8928 | 0.8914 ± 0.0080 |
| A1 | Fixed Haar, g=1 | 0.8846 | 0.8951 | 0.8992 | 0.8930 ± 0.0075 |
| A2 | Fixed Haar, g=8 | 0.9017 | 0.9016 | 0.8936 | 0.8990 ± 0.0046 |
| A3 | Separable θ, g=1 | 0.8802 | 0.9012 | 0.8917 | 0.8911 ± 0.0105 |
| A4 | Separable θ, g=8 | 0.9007 | 0.9020 | 0.9018 | 0.9015 ± 0.0007 |
| A5 | SO(8), g=1 | 0.8944 | 0.8852 | 0.8955 | 0.8917 ± 0.0057 |
| A6 | SO(8), g=8 | 0.8938 | 0.8931 | 0.9021 | 0.8964 ± 0.0050 |
| A2−A6 | Percentage points | +0.79 | +0.84 | -0.85 | +0.26 ± 0.96 |

Statistics and paired differences are calculated from the **unrounded seed
metrics** in the original experiment records, then rounded for display. For example, A6 averages 0.8963553229, and the seed-456 A2−A6
difference is 0.8447175 percentage points. The existing additional paired-difference
plot was generated from rounded seed values; the table above uses the original
precision and matches the paper.

A2, A4 and A6 exceed B0 in mean Dice by **0.76, 1.01 and 0.50 points**,
respectively; grouped-versus-dense differences are **0.60 (A2−A1), 1.04
(A4−A3) and 0.47 (A6−A5) points**. A2 and A4 exceed B0 in all three runs,
and **A4 has the highest mean Dice, 0.9015 ± 0.0007**. These are observed
results across three runs, not a claim of statistical significance.

Grouped variants have **59.25 M parameters versus B0's 67.42 M (12.1% fewer)**,
but measured inference latency is not lower. The parameter saving belongs to
the full replacement: 7.49 M from the Haar resampler and 0.68 M from grouping,
rather than fixed Haar alone.

A2 and A6 isolate full-rotation learning. At the audited A6 checkpoint, 12/16
locations are non-removable under their prescribed mixers, yet A2−A6 changes
sign across seeds. **Structural non-removability does not imply a consistent
task gain.**

## Quick start

```bash
pip install -r requirements.txt
```

- Python ≥ 3.9, PyTorch ≥ 2.0
- `monai` for networks / sliding-window inference
- `mamba-ssm` + `causal-conv1d` only for the Mamba backbone (`B0`, A1–A8). C1/C2 and `verify_absorption.py` do not need them.

### Verify the absorption identity (CPU, no data — the paper's exact audit)

```bash
# 1. no download: random Q, W (numerically checks the identity)
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

(A6 has 4 grouped analysis sites, 8 grouped sandwich sites and 4 dense
synthesis sites: 4/16 have zero escape. A5 has dense analysis and synthesis
mixers but grouped sandwich mixers: 8/16 have zero escape.)

The paper-wide figure is the maximum over all 163 sites in
`analysis/absorption_sites.csv`.  Command 3 prints `163 91 5.92...e-15`.

A full training checkpoint (`best_model_0.9317.pt`, ~226 MB) is **not** required
for this audit. Full training weights are not included and no download URL
is supplied in this release. Dice reproduction needs the full matching
checkpoint and BraTS2021; the small audit-only tensor file cannot produce
segmentation predictions.

### Reproduce a paper arm (example: A6, learnable Q + groups=8)

```bash
# 1. raw data -> ./data/fullres/train (labels remapped 4→3)
python scripts/rename_brats2021_data.py --data-dir /path/to/BraTS2021
python scripts/preprocessing_brats2021.py --data-dir /path/to/BraTS2021

# 2. train (500 epochs, weighted CE+Tversky + deep supervision, poly-SGD)
python train_brats2021.py --config configs/A6_so8_g8.yaml --tag a6_so8_g8 --seed 123

# 3. inference + metrics (sliding window, overlap 0.7, 8-way mirror TTA,
#    ET→NCR volume threshold fixed at 50 — the protocol behind the table)
python run_test_eval.py --config configs/A6_so8_g8.yaml \
    --ckpt logs/segmamba_haar_so8_dicece_ds/a6_so8_g8_s123/model/best_model_*.pt \
    --out_dir ./analysis_results/a6_s123 --force_ethr 50 --stage test
```

C1 / C2 use the same trainer (`model: segattn_haar` in the YAML) and do not
import `mamba_ssm`:

```bash
python train_brats2021.py --config configs/C1_segattn_so8_g1.yaml
# The unified trainer selects the attention backbone from this config.
```

All arms share the identical **evaluation** protocol (same internal held-out
250-case list, overlap 0.7, 8-way TTA, ET→NCR threshold 50). S1 was trained
with unweighted CE and no deep supervision; every other paper arm uses
weighted CE+Tversky and deep supervision. `run_test_eval.py` defaults to the
shipped frozen lists (`splits/test_250.txt`, `splits/val_125.txt`); pass
`--test_list ""` only if you deliberately want to re-derive the split (which is
not portable across hosts). Dice reproduction still needs BraTS2021 and a full
training checkpoint; the absorption audit does not.

To repeat a main arm with three seeds (Bash):

```bash
for seed in 123 456 789; do
  python train_brats2021.py --config configs/A6_so8_g8.yaml --tag a6_so8_g8 --seed "$seed"
done
```

Provide one explicit checkpoint path to evaluation if a wildcard matches more
than one file.

## Repository layout rationale

- **`model_segmamba/lowo.py` is deliberately dependency-free** (torch only), so
  the operator can be dropped into any CNN / attention / SSM backbone — the
  paper's "drop-in replacement" claim is embodied in the file layout.
- **`verify_absorption.py` is deliberately data-free**: local absorbability is an
  architectural property, verified exactly per checkpoint, not from aggregate
  metrics.  `cifar2d/verify_absorption_2d.py` is the same identity on analysis
  sites only ($SO(4)$, fold $W'=WR$).
- Training framework (`light_training/`) includes the
  optimizer, LR schedule and deep-supervision helper used in the paper.
  The training interface and split limitations are described above.

## Training commands by arm

Only **B0 and A1–A6** appear in paper Table 1. S1 is an additional baseline;
A7, A8, C1 and C2 are structural probes. Add `--seed 123`, `--seed 456` or
`--seed 789` to the unified training commands for the desired run.

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
    --out_dir ./analysis_results/<run> --force_ethr 50 --stage test
```

## Whole-network finite-precision check (paper §3.2)

Folding an absorbable transform in a single A5 checkpoint without retraining
changes TC / WT / ET Dice by **1.82×10⁻⁴ / 8.08×10⁻⁵ / 7.48×10⁻⁵** on
average in absolute value under the reported inference settings. The maximum
absolute TC Dice change is **1.14×10⁻³**, and the minimum voxel-wise label
agreement is **99.9981%**. These end-to-end observations are separate from the
FP64 per-block identity check. The experiment helpers are
`scripts/fold_all_sites_check.py` and `scripts/fold_audit_supplement.py`.

## Grouped-sandwich Dice audit (§3.2)

The last paragraph of §3.2 screens unsafe structural simplification before
deployment. From the A6 checkpoint it ranks grouped-sandwich sites by escape
fraction ρ, picks three with low / medium / high ρ (0.166, 0.352, 0.593), and on
each site builds two variants from the **same** weights:

- **exact reparameterization** — fold `RᵀWR` into a dense 1×1 mixer; an exact
  identity, so predictions are unchanged up to FP round-off;
- **forced block-diagonal projection** — `Π_G8`, keep only the 8 diagonal blocks
  of the folded mixer, without tying their weights; a *forced* simplification that the audit flags unsafe
  when ρ > 0.

The reported experiment evaluates all three networks (original, exact-reparam,
forced block-diagonal) on the frozen 250 cases and compares them per case.
The original experiment report gives the following mean and worst-case
single-case Dice drifts. That full report is not included in this release.

| site (ρ = escape fraction) | exact reparam — mean Dice Δ | exact reparam — worst-case drift | forced block-diagonal — mean Dice Δ | forced block-diagonal — worst-case drift |
|---|---:|---:|---:|---:|
| `vit.gscs.1.proj2` (ρ=0.166) | −2.94×10⁻⁶ | 2.57×10⁻⁴ | +0.0000 | 0.0038 |
| `vit.gscs.1.proj` (ρ=0.352) | −2.71×10⁻⁶ | 2.13×10⁻⁴ | −0.0001 | 0.0076 |
| `vit.gscs.0.proj2` (ρ=0.593) | −3.87×10⁻⁶ | 4.14×10⁻⁴ | −0.0006 | 0.3441 |

Exact reparameterization keeps mean Dice change **< 4×10⁻⁶** in all three cases
(an identity up to FP round-off), while the worst-case single-case Dice drift of
forced block-diagonal projection grows from **0.0038** (low ρ) to **0.344** (high ρ).
Thus ρ is a practical pre-deployment screen for unsafe structural simplification.

| Script | Role |
|---|---|
| `scripts/repro_s32_sandwich_dice.py` | Portable entry point. Defaults to `configs/A6_so8_g8.yaml` + `splits/test_250.txt`; forwards the rest to the audit. |
| `scripts/fold_sandwich_projection.py` | The audit itself: ρ ranking, variant building, per-case Dice. |
| `scripts/fold_all_sites_check.py` | Helpers: `find_repo()`, `region_dices`, `haar8`, dense fold. |
| `scripts/fold_audit_supplement.py` | Helpers: GT Dice, TTA ablation, strict-conjugate (block-diagonal) audit. |

```bash
# CPU self-check (no ckpt / data / GPU): numerically verifies the fold identity
python scripts/repro_s32_sandwich_dice.py --selftest

# Full 250-case reproduction (needs the full A6 ckpt + BraTS data):
python scripts/repro_s32_sandwich_dice.py \
    --ckpt /path/to/full_A6_checkpoint.pt \
    --data_dir ./data/fullres/train --n_cases 250 --purpose final \
    --out sandwich_projection_final.json
```

Pass `--n_cases 250` explicitly: the published script defaults to 10 cases,
and `--purpose final` only labels the run; it does not select all cases.
The script uses `cuda:0` by default and does not require eight GPUs.

The exact-reparam identity is float64 ≈ 3.7e-16 / fp32 ≈ 1e-6 on CPU; the Dice
comparison needs the full training checkpoint and BraTS data (not in this repo,
see "Reproduce a paper arm" above).

## CIFAR-10 2D probe (`cifar2d/`)

ResNet-18, no stem DWT.  Haar sites are layer2/3/4 × (main `conv1` + skip) =
**6 analysis sites**.  Fold is $W'=WR$ (`groups=1` absorbs; `groups=4` can
make $Q$ locally non-removable under the specified adjacent mixer class).  There is no sandwich `idwt∘W∘dwt` in this probe.

| Arm | Downsample | Paper analogue |
|---|---|---|
| B0 | stride-2 conv | learned resample |
| A2 | fixed Haar, `groups=4` | 2D HWD |
| A5 | learnable $Q\in SO(4)$, `groups=1` | absorbable |
| A6 | learnable $Q\in SO(4)$, `groups=4` | locally non-removable under the specified adjacent mixer class, 6 DOF |

The paper's structural cross-check covers six analysis sites and three seeds:
dense mixers give $\rho=0$, while grouped mixers give escape fractions of
**56–84%**.

The following accuracy comparison is an additional experiment outside the
current paper. `wave.skew` is in a **wd=0** group.  Report official-test top-1 at the best
validation epoch (5k hold-out from the 50k train split).  Three seeds:
A2 94.44%, B0 95.14%, A6 94.64%. A6 is numerically 0.20 percentage points
above A2; these means alone do not establish a statistically reliable gain.
The structural audit transfers; the BraTS performance ordering need not.

```bash
cd cifar2d
python train_cifar.py --selftest
python train_cifar.py --arm A2 --seed 123 --device cuda:0
python verify_absorption_2d.py --synthetic          # no checkpoint needed
python verify_absorption_2d.py runs/a6_s123/best.pt  # after training
python mcnemar.py --all --runs runs
```

### Intervening nonlinearity (§3.4)

The §3.4 "Intervening nonlinearity" check inserts a ReLU between the learned
transform `R` and the adjacent dense mixer `W` (groups=1) at an audited site,
then fits a replacement dense mixer `W'` by regularized least squares and
compares the original `W·ReLU(R z0)` against the fitted linear block
`W'·ReLU(z0)`. On real CIFAR-10 activations the relative output error is
**11.6%** (seed 123), versus **< 8.1×10⁻¹⁴** for the linear (no-ReLU) and
identity (`Q=I`) controls — i.e. the failure is the non-commutation of the
learned rotation with the pointwise nonlinearity, not the fit. This is why the
audit assumes a linear adjacency between the folded operators (Proposition 1).

```bash
cd cifar2d
python nonlinear_counterexample.py --arm A5 --all-seeds --data-dir data
# The historical output log is not included in this release.
```

Needs a trained 2D A5 run (`runs/a5_s<seed>/best.pt`) and CIFAR-10 under
`cifar2d/data/` (not committed).

Needs `torchvision`.  Do not commit `cifar2d/data/` or `cifar2d/runs/`.

## Additional experiment: OFT precision probe (outside the current paper)

This experiment is retained as an additional implementation check and is not
part of the current paper's §3.4 results.

OFT (orthogonal fine-tuning, Qiu et al.) merges a learned orthogonal adapter
into the adjacent linear layer by an exact identity in real arithmetic. We
reproduce this merge on an external operator: a self-trained OFT adapter on a
public DINOv2 ViT-Large backbone, as an independent implementation check rather
than a method claim. Its non-identity rotation factors are confirmed loaded and
active, and the merge is reproduced through an independent branch. Under a
controlled global comparison, end-to-end classification-logits differences are
$3.85\times10^{-3}$ (TF32), $1.71\times10^{-5}$ (plain FP32), and
$3.36\times10^{-14}$ (FP64), consistent with floating-point reassociation, not
accuracy loss. This shows the audit distinguishes algebraic removability from
finite-precision nonequivalence; it does not validate the grouped-mixer or
sandwich criterion, and the adapter's training provenance is only partially
documented.

```bash
cd oft_absorption_audit
python nonidentity_recovery.py   --checkpoint-dir outputs/oft_adapter_checkpoint
python precision_ablation.py     --checkpoint-dir outputs/oft_adapter_checkpoint
python fixed_input_layer_replay.py --checkpoint-dir outputs/oft_adapter_checkpoint
```

Needs PyTorch and a DINOv2 OFT adapter checkpoint (not shipped). The full
experiment log — the earlier $0.117$ discrepancy trace, identity-control
checks, layerwise drift, and the floating-point bound — lives in
`oft_absorption_audit/README.md`; the controlled results are summarized below.

### Results (controlled audit; dinov2-large + self-trained CIFAR-10 OFT adapter, 96 OFT layers)

End-to-end classification-logit differences after `merge_and_unload()`
(precision ablation, `precision_ablation_report.json`):

| configuration | max logit Δ | MSE | interpretation |
|---|---:|---:|---|
| TF32 enabled | 3.85×10⁻³ | 1.16×10⁻⁶ | TF32 reassociation |
| FP32 (TF32 off) | 1.71×10⁻⁵ | 1.32×10⁻¹¹ | FP32 reassociation |
| FP64 | 3.36×10⁻¹⁴ | 3.09×10⁻²⁹ | double-precision magnitude |

The TF32-on error is ≈225× the TF32-off error, and the FP64 residual lands at
double-precision magnitude (not machine epsilon) — consistent with floating-point
reassociation, not accuracy loss. Verdict: `FINITE_PRECISION`.

Non-identity recovery (independent reload; `nonidentity_recovery_report.json`):
confirms the adapter is a real, non-trivial rotation and the merge is exact.

| check | result |
|---|---|
| layers audited / OK | 96 / 96 |
| param-file max diff on reload | 0.0 (bit-exact) |
| adapter active (max Δ) | yes, 0.385 |
| ‖R−I‖_F range | 0.295 – 0.495 (non-trivial, non-identity) |
| FP32 local absorb max | 2.19×10⁻⁵ (within Higham FP bound, ratio 0.011) |
| FP32+TF32 local absorb max | 5.10×10⁻³ (exceeds bound, ratio 2.12 → TF32 flagged) |
| FP64 local absorb max | 2.84×10⁻¹⁴ (double-precision magnitude) |
| W₀ block condition number max | 1.25×10⁷ |

Identity control (R=I sanity, `identity_control_report.json`): forcing R=I,
`model(x) ≈ base(x)` holds exactly, and merge differs by max **4.17×10⁻⁶**
(MSE 1.31×10⁻¹³, 1.875% of elements >1e-6); verdict **PASS** — the merge adds
only FP32 reassociation noise, no implementation bug. Fixed-input single-layer
replay (`fixed_input_replay.json`): the captured-input replay matches the
hook-captured output exactly (max 0.0), separating local absorb error from
upstream-input drift.

## License

Repository license: [MIT](LICENSE). Retain the notices and licenses attached
to included upstream components. BraTS2021 data is not distributed here and
follows its own [usage terms](https://www.synapse.org/#!Synapse:syn27046444).
