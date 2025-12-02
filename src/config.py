# src/config.py

EXP_CONFIGS = {
    "exp0": {
        "name": "Exp0_FrozenEffB0_Early_Ridge",
        "backbone": "efficientnet_b0",
        "img_size": 256,
        "aug": "lite",
        "ridge_alpha": 1.0,
        "n_splits": 5,
        "batch_size": 32,
        "seed": 42,
    },
    "exp1": {
        "name": "Exp1_TabOnly_LGBM",
        "n_splits": 5,
        "seed": 42,
    },
    "exp2": {
        "name": "Exp2_EffNetB0_256_Lite",
        "backbone": "efficientnet_b0",
        "img_size": 256,
        "aug": "lite",
        "loss": "mse",
        "n_splits": 5,
        "epochs": 10,
        "batch_size": 32,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "patience": 5,
        "seed": 42,
    },
    "exp3": {
        "name": "Exp3_SwinT_384_Strong",
        "backbone": "swin_tiny_patch4_window7_224",
        "img_size": 384,
        "aug": "strong",
        "loss": "bce",  # y/100 + BCEWithLogitsLoss
        "n_splits": 5,
        "epochs": 10,
        "batch_size": 32,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "patience": 5,
        "seed": 42,
    },
   
}
