# src/config.py

COMMON_CONFIG = {
    "n_splits": 5,
    "seed": 42,
    "epochs": 10,
    "patience": 5,
    "lr": 2e-5,
    "weight_decay": 0.01,
}

EXP_CONFIGS = {
    "exp0": {
        **COMMON_CONFIG,
        "name": "Exp0_FrozenEffB0_Early_Ridge",
        "backbone": "efficientnet_b0",
        "img_size": 256,
        "aug": "lite",
        "ridge_alpha": 1.0,
        "batch_size": 32,
    },
    "exp1": {
        **COMMON_CONFIG,
        "name": "Exp1_TabOnly_LGBM",
        "batch_size": 32, # not required for tab-only
    },
    "exp2": {
        **COMMON_CONFIG,
        "name": "Exp2_EffNetB0_256_Lite",
        "backbone": "efficientnet_b0",
        "img_size": 256,
        "aug": "lite",
        "loss": "mse",
        "batch_size": 32,
    },
    "exp3": {
        **COMMON_CONFIG,
        "name": "Exp3_SwinT_384_Strong",
        "backbone": "swin_large_patch4_window12_384",
        "img_size": 384,
        "aug": "strong",
        "loss": "bce",  # y/100 + BCEWithLogitsLoss
        "batch_size": 12,
    },
    "exp4": {
        **COMMON_CONFIG,
        "name": "Exp4_EffB1_TabMLP_Early_MLPHead",
        "backbone": "efficientnet_b1",
        "img_size": 256,
        "aug": "lite",
        "head_type": "mlp",
        "loss": "mse",
        "batch_size": 32
    },
    "exp5": {
        **COMMON_CONFIG,
        "name": "Exp5_SwinT_384_Strong_TabMLP_Early_MLPHead",
        "backbone": "swin_large_patch4_window12_384",
        "img_size": 384,
        "aug": "strong",
        "head_type": "mlp",
        "loss": "bce",
        "batch_size": 12,
    },
     "exp6": {
        **COMMON_CONFIG,
        "name": "Exp6_EffB1_256_Lite",
        "backbone": "efficientnet_b1",
        "img_size": 256,
        "aug": "lite",
        "loss": "mse",
        "batch_size": 32,
    },
    "exp8": {
        **COMMON_CONFIG,
        "name": "Exp8_EffB4_384_Strong",
        "backbone": "efficientnet_b4",
        "img_size": 384,
        "aug": "strong",
        "loss": "mse",
        "batch_size": 32,
    },
    "exp12_film": {
        **COMMON_CONFIG,
        "name": "exp12_film_fusion",
        "backbone": "efficientnet_b1",
        "img_size": 256,
        "aug": "lite",
        "head_type": "mlp",
        "loss": "mse",
        "batch_size": 32
    },
     "exp13_cross_attn": {
        **COMMON_CONFIG,
        "name": "Exp13_SWIN_cross_attention",
        "backbone": "swin_large_patch4_window12_384",
        "img_size": 384,
        "aug": "strong",
        "head_type": "mlp",
        "loss": "bce", # or mse
        "batch_size": 12,
    },
    "exp14_gated_fusion": {
        **COMMON_CONFIG,
        "name": "exp14_gated_fusion",
        "backbone": "efficientnet_b1",
        "img_size": 256,
        "aug": "lite",
        "head_type": "mlp",
        "loss": "mse",
        "batch_size": 32
    },
   

}
