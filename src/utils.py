# src/utils.py
import gc
import os, random, json
import numpy as np
import torch
from lightgbm import LGBMRegressor  # or XGBRegressor, etc
import pandas as pd


import warnings

from sklearn.metrics import root_mean_squared_error

from xgboost import XGBRegressor
from cuml.svm import SVR as cuSVR  
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.models import VisionAuxNet
from src.data import build_transforms, ImageOnlyDataset


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


def build_binary_aux_cfg(df, binary_aux_tasks, target_col="Pawpularity"):
    """
    Automatically compute:
      - pos_weight: majority_count / minority_count for each binary task
      - flip_targets: tasks where class 0 is the minority
                      (i.e. we flip label so the rare class becomes 1 for BCE)

    Returns:
        pos_weight_dict   : {task: float}
        flip_targets_list : [task, ...]
    """
    pos_weight_dict   = {}
    flip_targets_list = []

    for task in binary_aux_tasks:
        counts = df[task].value_counts()   # counts for 0 and 1
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)

        print(f"  {task}: class_0={n0}, class_1={n1}", end="")

        if n1 == 0 or n0 == 0:
            print(f"   WARNING: only one class present, skipping pos_weight")
            pos_weight_dict[task] = 1.0
            continue

        if n1 < n0:
            # class 1 is minority . standard BCE, pos_weight = n0 / n1
            pos_weight_dict[task] = n0 / n1
            print(f"   minority=class_1, pos_weight={pos_weight_dict[task]:.2f}")
        else:
            # class 0 is minority . flip targets so class 0 becomes 1
            pos_weight_dict[task] = n1 / n0
            flip_targets_list.append(task)
            print(f"   minority=class_0, flip target, pos_weight={pos_weight_dict[task]:.2f}")

    return pos_weight_dict, flip_targets_list

def extract_aux_oof_to_csv(out_dir, df, img_folder, cfg, force=False, device="cuda", target_col="Pawpularity"):
    """
    Create oof_detail_aux.csv with per-sample aux probabilities and labels.
    """
    aux_path = os.path.join(out_dir, "oof_detail_aux.csv")

    if os.path.exists(aux_path) and not force:
        print(f"Loading existing: {aux_path}")
        return pd.read_csv(aux_path)
    

    binary_aux_tasks = cfg["binary_aux_tasks"]
    flip_targets = set(cfg.get("binary_aux_flip_targets", []))

    kf = KFold(
        n_splits=cfg["n_splits"],
        shuffle=True,
        random_state=cfg["seed"]
    )

    rows = []
    val_tf = build_transforms(cfg["img_size"], cfg["aug"], train=False)

    for fold, (_, val_idx) in enumerate(kf.split(df), start=1):
        val_df = df.iloc[val_idx].reset_index(drop=True)
        ckpt_path = os.path.join(out_dir, f"model_fold{fold}.pt")

        if not os.path.exists(ckpt_path):
            print(f"fold {fold}: checkpoint missing, skipped")
            continue

        model = VisionAuxNet(
            backbone_name=cfg["backbone"],
            img_size=cfg["img_size"],
            aux_tasks=[],
            binary_aux_tasks=binary_aux_tasks,
            head_type=cfg.get("head_type", "linear"),
            pretrained=False,
            use_saliency=False,
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        val_ds = ImageOnlyDataset(
            val_df,
            img_folder,
            val_tf,
            aux_tasks=[],
            binary_aux_tasks=binary_aux_tasks
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        local_ptr = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, y, aux = batch
                imgs = imgs.to(device)
                preds = model(imgs)

                bs = imgs.size(0)
                batch_ids = val_df.iloc[local_ptr:local_ptr + bs]["Id"].tolist()
                batch_y   = val_df.iloc[local_ptr:local_ptr + bs][target_col].values
                local_ptr += bs

                task_probs = {}
                task_preds = {}
                task_trues = {}

                for task in binary_aux_tasks:
                    raw_prob = torch.sigmoid(preds[task]).cpu().numpy().reshape(-1)
                    true_lab = aux[task].numpy().reshape(-1)

                    # Convert probabilities back to the original label semantics.
                    # Some tasks were trained with flipped targets to make the minority
                    # class the positive class for weighted BCE training.
                    # and this if condition will not be called for standard BCE as flip_targets will not contain that aux task.
                    prob_orig = (1.0 - raw_prob) if task in flip_targets else raw_prob
                    pred_bin = (prob_orig >= 0.5).astype(int)

                    task_probs[task] = prob_orig
                    task_preds[task] = pred_bin
                    task_trues[task] = true_lab

                for i in range(bs):
                    row = {
                        "Id": batch_ids[i],
                        "fold": fold,
                        "ytrue": float(batch_y[i]),
                    }
                    for task in binary_aux_tasks:
                        row[f"{task}_true"] = int(task_trues[task][i])
                        row[f"{task}_prob"] = float(task_probs[task][i])
                        row[f"{task}_pred_05"] = int(task_preds[task][i])
                    rows.append(row)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"fold {fold}: extracted {len(val_df)} samples")

    aux_df = pd.DataFrame(rows)
    aux_df.to_csv(aux_path, index=False)
    print("\nSaved:", aux_path)
    return aux_df

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
