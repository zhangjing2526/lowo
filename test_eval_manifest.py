"""Manifest of the 11 Haar/LOWO/SegAttn checkpoints needing test-set eval.

Each entry:
  name      : output subdir under <data_root>/analysis_results/<name>/
  config    : config YAML (relative to repo root) that defines the architecture
  ckpt_rel  : checkpoint dir RELATIVE TO --data_root (best_model_*.pt auto-picked)
              e.g. "logs/segmamba_brats2021" -> <data_root>/logs/segmamba_brats2021

All ckpt_rel are relative to a single --data_root so the SAME manifest works on
Windows (--data_root "G:/material/天枢/mamba") and Ubuntu (--data_root /mount/path).
On Ubuntu every variant (incl. segmamba_haar_fixed_g8) lives under ./logs; on
Windows segmamba_haar_fixed_g8 sits at the data-root top level instead -- eval
runs on Ubuntu, so the Ubuntu (nested-under-logs) path is authoritative.

SAFETY: run `python run_test_eval.py --manifest test_eval_manifest.py --index N --dry_run`
on the Ubuntu box first -- it does a STRICT load and will flag any config<->ckpt
mismatch (e.g. wrong grouped_mix / use_routing) before you spend GPU time.

#0 and #1 are OLDER BraTS2021-protocol runs (CE-only, pre-LOWO refactor).
Dry-run shows their checkpoints carry legacy keys (`*.wave.skew`,
`ds_out*.conv.conv.*`) that the CURRENT model code + monai version no longer
produce -> strict load fails (architecture drift). They are superseded by the
new-protocol runs below (#6 `segmamba_haar_theta` == base learnable Haar,
#2 `segmamba_haar_so8` == so8), so dropping them loses no design point. If you
truly need #0/#1 numbers, rebuild the model with the matching legacy definition
(old Haar code w/ `skew` + old monai `UnetOutBlock`). The harness now SKIPs them
cleanly (writes SKIP_arch_mismatch.txt) and continues with the other variants.
"""
MANIFEST = [
    # --- Haar / LOWO 8-arm ablation (validation set) ---
    {"name": "segmamba_haar_brats2021",   "config": "configs/segmamba_haar.yaml",
     "ckpt_rel": "logs/segmamba_haar_brats2021"},
    {"name": "segmamba_haar_so8_brats2021", "config": "configs/segmamba_haar.yaml",
     "ckpt_rel": "logs/segmamba_haar_brats2021_so8"},
    {"name": "segmamba_haar_so8",          "config": "configs/exp2_segmamba_haar.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar"},
    {"name": "segmamba_haar_fixed",        "config": "configs/exp6_segmamba_haar_fixed.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed"},
    {"name": "segmamba_haar_perch",        "config": "configs/exp8_segmamba_haar_perchannel.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_perch"},
    {"name": "segmamba_haar_so8live",      "config": "configs/exp6c_segmamba_haar_so8_live.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_so8live"},
    {"name": "segmamba_haar_theta",        "config": "configs/exp7_segmamba_haar_theta.yaml",
     "ckpt_rel": "logs/segmamba_haar_dicece_ds/segmamba_haar_theta"},
    {"name": "segmamba_haar_wb",           "config": "configs/exp4_segmamba_haar_wb.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_wb"},
    # --- SegAttn (window-attention mixer)对照 ---
    {"name": "segattn_haar",               "config": "configs/segattn_haar.yaml",
     "ckpt_rel": "logs/segattn_haar_so8_dicece_ds/segattn_haar"},
    {"name": "segattn_haar_routed",        "config": "configs/exp3_segattn_haar_routed.yaml",
     "ckpt_rel": "logs/segattn_haar_so8_dicece_ds/segattn_haar_routed"},
    # --- clean attribution arm (fixed Haar + groups=8 mixer) ---
    # On Ubuntu the ckpt lives INSIDE ./logs (sibling of #2/#3/#4/#5/#7), not at
    # the data-root top level. Use this nested path so data_root=repo resolves it;
    # on Windows the same experiment is at G:/material/天枢/mamba/segmamba_haar_fixed_g8
    # (top level) -- eval runs on Ubuntu, so the Ubuntu path is authoritative.
    {"name": "segmamba_haar_fixed_g8",     "config": "configs/exp6d_segmamba_haar_fixed_g8.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed_g8"},
    # --- baselines (re-run under the SAME unified protocol for a comparable table) ---
    {"name": "segmamba_baseline_dicece",   "config": "configs/segmamba_baseline.yaml",
     "ckpt_rel": "logs/segmamba_brats2021_dicece"},
    {"name": "segmamba_baseline",          "config": "configs/segmamba_baseline.yaml",
     "ckpt_rel": "logs/segmamba_brats2021"},
    # --- external baseline #2: MONAI SwinUNETR (no wavelet / LOWO involved) ---
    # Constructed with the exact kwargs of 3_train_swin_unetr_brats2021.py:110-124;
    # SwinUNETR's state_dict is sensitive to feature_size/depths/num_heads/norm_name.
    # ckpt: logs/swin_unetr_brats2021/model/best_model_0.9277.pt (val mean 0.9277).
    # Heavy net -- the config sets eval_sw_batch: 1 to avoid OOM.
    {"name": "swin_unetr_baseline",        "config": "configs/swin_unetr_baseline.yaml",
     "ckpt_rel": "logs/swin_unetr_brats2021"},
    # --- seed runs for the main-spine contrast A2 vs A6 ---
    # Trainer.py:342 calls set_determinism(42+local_rank) inside train(), which
    # OVERWRITES the [run] seed set before model construction. So across seeds
    # only WEIGHT INITIALIZATION differs (123/456/789); training aug stays 42.
    # The original (no-suffix) run uses set_determinism(123) at script top ->
    # init seed 123. Therefore:
    #   - s123  has init seed 123 == original  -> SAME-init replicate (measures
    #     DataLoaderMultiProcess(seeds=None) noise, NOT a 2nd seed)
    #   - s456  has init seed 456 != original  -> TRUE 2nd seed (different init)
    #   - s789  not finished yet (empty model dirs in the tree)
    # Test 250 is identical across all runs (verified: s123 stdout test list
    # starts at BraTS2021_00011 == frozen list; same-host glob reproduces it).
    # -- s123 (same-init replicate, secondary): measures DataLoader noise on delta
    # A2 s123 = fixed Haar + groups=8 : val best 0.9239 (orig val 0.9285 -> test 0.9017)
    {"name": "segmamba_haar_fixed_g8_s123", "config": "configs/exp6d_segmamba_haar_fixed_g8.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed_g8_s123_s123"},
    # A6 s123 = so8 + groups=8 : val best 0.9244 (orig val 0.9317 -> test 0.8938; main spine +0.79)
    {"name": "segmamba_haar_so8live_s123", "config": "configs/exp6c_segmamba_haar_so8_live.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_so8live_s123_s123"},
    # -- s456 (TRUE 2nd seed, init 456, PRIMARY judgment of cross-init reproduction)
    # A2 s456 = fixed Haar + groups=8 : val best 0.9222
    {"name": "segmamba_haar_fixed_g8_s456", "config": "configs/exp6d_segmamba_haar_fixed_g8.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed_g8_s456_s456"},
    # A6 s456 = so8 + groups=8 : val best 0.9206
    {"name": "segmamba_haar_so8live_s456", "config": "configs/exp6c_segmamba_haar_so8_live.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_so8live_s456_s456"},
    # -- s456 (TRUE 2nd seed) for the remaining 2x2 cells + baseline + theta/perch
    # (2026-09-08 补齐: 2x2 四格 A1/A2/A5/A6 全双种子; B0/A3/A8 作陪衬行双种子)
    # B0 s456 = SegMamba baseline (dicece+ds) : best val 0.9019
    {"name": "segmamba_baseline_s456", "config": "configs/exp1_segmamba.yaml",
     "ckpt_rel": "logs/segmamba_dicece_ds/segmamba_s456_s456"},
    # A1 s456 = fixed Haar, groups=1 : best val 0.9244 (resume 后; 首次 0.9084)
    {"name": "segmamba_haar_fixed_s456", "config": "configs/exp6_segmamba_haar_fixed.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed_s456_s456"},
    # A3 s456 = theta (3-DOF separable), groups=1 : best val 0.9238
    {"name": "segmamba_haar_theta_s456", "config": "configs/exp7_segmamba_haar_theta.yaml",
     "ckpt_rel": "logs/segmamba_haar_dicece_ds/segmamba_haar_theta_s456_s456"},
    # A5 s456 = so8, groups=1 : best val 0.9182 (resume 后; 首次 0.9056)
    {"name": "segmamba_haar_so8_s456", "config": "configs/exp2_segmamba_haar.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_s456_s456"},
    # A5 s789 = so8, groups=1 (TRUE 3rd seed, init 789) : training DONE 2026-09-12
    #   found at logs/segmamba_haar_so8_dicece_ds/segmamba_haar_s789_s789
    #   NOTE: dir tag uses "s789" (not "so8") -- typo in tag name; architecture = A5.
    #   find_best_ckpt globs **/best_model_*.pt so the ./model subdir is picked up.
    {"name": "segmamba_haar_so8_s789", "config": "configs/exp2_segmamba_haar.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_s789_s789"},
    # A8 s456 = so8 per-channel mixer : best val 0.9168
    {"name": "segmamba_haar_perch_s456", "config": "configs/exp8_segmamba_haar_perchannel.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_perch_s456_s456"},
]
