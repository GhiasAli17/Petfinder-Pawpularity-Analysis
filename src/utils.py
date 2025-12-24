# src/utils.py
import os, random, json
import numpy as np
import torch
from lightgbm import LGBMRegressor  # or XGBRegressor, etc
import pandas as pd


import warnings

from sklearn.metrics import root_mean_squared_error

from xgboost import XGBRegressor
from cuml.svm import SVR as cuSVR  


warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_SEED = 42

def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)



def late_fusion_from_oof(
    oof_df,
    img_col="oof_pred",
    tab_col="gbdt_oof",
    y_col="ytrue",
    fold_col="fold",
    mode="weighted",     # "simple" or "weighted"
    n_grid=101,
):
    """
    Perform late fusion directly from a merged OOF DataFrame.

    oof_df must contain columns:
      - y_col   : ground-truth
      - img_col : image-model OOF predictions
      - tab_col : tabular-model OOF predictions
      - fold_col: fold index (for fold-wise RMSE)
    """
    df = oof_df.copy()

    y_true   = df[y_col].to_numpy()
    pred_img = df[img_col].to_numpy()
    pred_tab = df[tab_col].to_numpy()

   
    if mode == "simple":
        blend = 0.5 * pred_img + 0.5 * pred_tab
        best_rmse = root_mean_squared_error(y_true, blend)
        best_w = 0.5
    elif mode == "weighted":
        best_rmse = np.inf
        best_w = None
        for w in np.linspace(0.0, 1.0, n_grid):
            blend_tmp = w * pred_img + (1.0 - w) * pred_tab
            rmse_tmp = root_mean_squared_error(y_true, blend_tmp)
            if rmse_tmp < best_rmse:
                best_rmse = rmse_tmp
                best_w = w
        blend = best_w * pred_img + (1.0 - best_w) * pred_tab
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df["final_pred"] = blend
    score = best_rmse

    info = {
        "mode": mode,
        "weight_a": best_w if mode == "weighted" else 0.5,
        "weight_b": (1.0 - best_w) if mode == "weighted" else 0.5,
    }

    # ---- fold-wise RMSE ----
    if fold_col in df.columns:
        fold_rmse = {}
        for f, sub in df.groupby(fold_col):
            fold_rmse[int(f)] = root_mean_squared_error(
                sub[y_col].to_numpy(),
                sub["final_pred"].to_numpy(),
            )
        vals = np.array(list(fold_rmse.values()))
        info["fold_rmse"] = fold_rmse
        info["fold_rmse_mean"] = float(vals.mean())
        info["fold_rmse_std"] = float(vals.std())

    return df, score, info

def build_meta_oof(
    base_oof_specs,
    out_path=None,
    id_col="Id",
    y_col="ytrue",
    fold_col="fold",
):
    """
    Build a meta-level OOF DataFrame from multiple base OOF files.

    base_oof_specs: list of dicts, each like
        {"name": "exp1", "path": ".../oof_detail.csv", "pred_col": "oof_pred"}
    Returns:
        meta_df with columns: Id, ytrue, fold, p_<name>...
    """
    meta = None
    for spec in base_oof_specs:
        name = spec["name"]
        path = spec["path"]
        pred_col = spec.get("pred_col", "oof_pred")

        df = pd.read_csv(path)
        df = df[[id_col, y_col, fold_col, pred_col]].rename(
            columns={pred_col: f"p_{name}"}
        )

        if meta is None:
            meta = df
        else:
            meta = meta.merge(df, on=[id_col, y_col, fold_col], how="inner")

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        meta.to_csv(out_path, index=False)

    return meta


def train_meta_learner_oof(
    meta_df,
    model_name="lgbm",     
    feature_prefix="oof_",
    y_col="ytrue",
    fold_col="fold",
    model_params=None,
):
    """
    Train a 2nd-level meta-learner on OOF Predictions.

    """
    feature_cols = [c for c in meta_df.columns if c.startswith(feature_prefix)]
    X = meta_df[feature_cols].values
    y = meta_df[y_col].values
    folds = meta_df[fold_col].values

    # ----- choose model + default params -----
    if model_name == "lgbm":
        if model_params is None:
            model_params = dict(
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
                force_col_wise=True,
            )
        model = LGBMRegressor(**model_params)

    elif model_name == "xgb":
        if model_params is None:
            model_params = dict(
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                verbosity=0,
            )
        model = XGBRegressor(**model_params)

    elif model_name == "svr_gpu":
            model_params = dict(
                C=20.0,
                epsilon=0.1,
                kernel="rbf",
            )
            model = cuSVR(**model_params)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # ---- train on full data ----
    model.fit(X, y)

    # ---- evaluate ----
    pred = model.predict(X)
    pred = pred.ravel()
    train_rmse = root_mean_squared_error(y, pred)

    return model, train_rmse, feature_cols
