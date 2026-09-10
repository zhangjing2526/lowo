"""Paper-arm eval manifest (IDs match the paper / configs/).

Checkpoints are not in this repository (full weights are ~200 MB each).
Point ``--data_root`` at a directory whose layout matches ``ckpt_rel``, or
pass ``--ckpt`` to ``run_test_eval.py`` directly.

    python run_test_eval.py --config configs/A6_so8_g8.yaml \
        --ckpt /path/to/best_model_0.9317.pt \
        --force_ethr 50 --test_list splits/test_250.txt --stage test
"""
MANIFEST = [
    {"name": "B0", "config": "configs/B0_segmamba.yaml",
     "ckpt_rel": "logs/segmamba_dicece_ds/segmamba"},
    {"name": "S1", "config": "configs/S1_swin_unetr.yaml",
     "ckpt_rel": "logs/unetr_dicece/swin_unetr"},
    {"name": "A1", "config": "configs/A1_fixed_g1.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed"},
    {"name": "A2", "config": "configs/A2_fixed_g8.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed_g8"},
    {"name": "A3", "config": "configs/A3_theta_g1.yaml",
     "ckpt_rel": "logs/segmamba_haar_dicece_ds/segmamba_haar_theta"},
    {"name": "A5", "config": "configs/A5_so8_g1.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar"},
    {"name": "A6", "config": "configs/A6_so8_g8.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_so8live"},
    {"name": "A7", "config": "configs/A7_so8_g1_wbloss.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_wb"},
    {"name": "A8", "config": "configs/A8_so8_per_channel.yaml",
     "ckpt_rel": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_perch"},
    {"name": "C1", "config": "configs/C1_segattn_so8_g1.yaml",
     "ckpt_rel": "logs/segattn_haar_so8_dicece_ds/segattn_haar"},
    {"name": "C2", "config": "configs/C2_segattn_so8_g1_routed.yaml",
     "ckpt_rel": "logs/segattn_haar_so8_dicece_ds/segattn_haar_routed"},
]
