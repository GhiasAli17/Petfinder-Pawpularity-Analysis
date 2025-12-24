from torch.utils.data import DataLoader
from src.data import build_transforms, ImageTabDataset
from src.models import EarlyFusionNet
from sklearn.metrics import root_mean_squared_error
import torch.nn as nn
import torch
import numpy as np
import os
import gc

import pandas as pd
import joblib
import gc
import os
from src.data import build_transforms, ImageOnlyDataset
from src.models import  build_vision_backbone


def infer_test_ensemble(test_df, img_folder, cfg, out_dir, device, tab_cols, workers=8):
    backbone_name = cfg["backbone"]
    img_size = cfg["img_size"]
    aug_type = cfg["aug"]
    head_type = cfg["head_type"]
    loss_name = cfg["loss"]

    scale_target = (loss_name == "bce")

    test_tf = build_transforms(img_size, aug_type, train=False)
    test_ds = ImageTabDataset(test_df, img_folder, tab_cols, test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    all_fold_preds = []
    fold_rmses = []

    y_true = test_df["Pawpularity"].values  # 

    for fold in range(1, cfg["n_splits"] + 1):

        model = EarlyFusionNet(
            backbone_name=backbone_name,
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            head_type=head_type,
            pretrained=False,  # weights will be loaded
        ).to(device)

        state_path = os.path.join(out_dir, f"model_fold{fold}.pt")
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for imgs, tabs, _ in test_loader:
                imgs = imgs.to(device)
                tabs = tabs.to(device)
                preds = model(imgs, tabs)
                if scale_target:
                    probs = torch.sigmoid(preds).cpu().numpy().squeeze()
                    out = probs * 100.0
                else:
                    out = preds.cpu().numpy().squeeze()
                fold_preds.append(out)

        fold_preds = np.concatenate(fold_preds)
        all_fold_preds.append(fold_preds)

        # RMSE for this fold on test_df
        rmse_fold = root_mean_squared_error(y_true, fold_preds)
        fold_rmses.append(rmse_fold)
        print(f"Fold {fold} Model RMSE on test set: {rmse_fold:.4f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # shape: (n_folds, n_test)
    all_fold_preds = np.stack(all_fold_preds, axis=0)
    test_pred_mean = all_fold_preds.mean(axis=0)

    # Ensemble RMSE
    rmse_ens = root_mean_squared_error(y_true, test_pred_mean)
    print(f"\nEnsemble mean RMSE on test set: {rmse_ens:.4f}")

    return test_pred_mean, all_fold_preds, fold_rmses, rmse_ens



# #2nd stage meta learner inference

def infer_meta_ensemble(
    test_df,
    img_folder,
    cfg,
    out_dir,
    device,
    tab_cols,
    workers=8,
    scale_target=True,
    metaStage_model="meta_model_lgbm.pkl"
):
    n_splits = cfg["n_splits"]

    # 1. SWIN inference fold-wise
    swin_preds_folds = []

    test_tf = build_transforms(cfg["img_size"], cfg["aug"], train=False)
    test_ds = ImageOnlyDataset(test_df, img_folder, test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    for fold in range(1, n_splits + 1):
        model = build_vision_backbone(
            cfg["backbone"], cfg["img_size"], mode="regression"
        ).to(device)

        swin_path = os.path.join(out_dir, f"model_fold{fold}.pt")
        state = torch.load(swin_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        fold_out = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                if scale_target:
                    probs = torch.sigmoid(preds).cpu().numpy().squeeze()
                    out = probs * 100.0
                else:
                    out = preds.cpu().numpy().squeeze()
                fold_out.append(out)

        swin_preds_folds.append(np.concatenate(fold_out))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    swin_preds_folds = np.stack(swin_preds_folds, axis=0)   # (n_splits, n_test)
    swin_mean = swin_preds_folds.mean(axis=0)               # (n_test,)

    print("Swin inference done.")

    # 2. GBDT inference fold-wise
    gbdt_preds_folds = []

    for fold in range(1, n_splits + 1):
        gbdt_path = os.path.join(out_dir, f"gbdt_fold{fold}.pkl")
        model_gbdt = joblib.load(gbdt_path)
        preds = model_gbdt.predict(test_df[tab_cols])
        gbdt_preds_folds.append(preds)

    gbdt_preds_folds = np.stack(gbdt_preds_folds, axis=0)   # (n_splits, n_test)
    gbdt_mean = gbdt_preds_folds.mean(axis=0)               # (n_test,)
    print("GBDT inference done.")

    # 3. build meta input df used for meta model
    meta_test_df = pd.DataFrame({
        "Id": test_df["Id"],
        "oof_pred_swin": swin_mean,
        "oof_pred_gbdt": gbdt_mean,
    })

    # 4. load saved meta learner
    meta_saved = joblib.load(os.path.join(out_dir, metaStage_model))
    meta_model = meta_saved["model"]
    meta_features = meta_saved["features"]  # ["oof_pred_swin", "oof_pred_gbdt"]

    # 5. Final predictions from 2nd stage
    final_pred = meta_model.predict(meta_test_df[meta_features].values)
    rmse = root_mean_squared_error(test_df["Pawpularity"].values, final_pred)

    # 6. Build detailed output dataframe 
    out_df = pd.DataFrame({
        "Id": test_df["Id"].values,
        "ytrue": test_df["Pawpularity"].values,
    })

    # per-fold preds for both models
    for f in range(n_splits):  
        out_df[f"pred_swin_fold{f+1}"] = swin_preds_folds[f]
        out_df[f"pred_gbdt_fold{f+1}"] = gbdt_preds_folds[f]

    # averages across folds
    out_df["pred_swin_mean"] = swin_mean
    out_df["pred_gbdt_mean"] = gbdt_mean

    # 2nd-stage prediction
    out_df["pred_2ndStage_gbdt"] = final_pred


    return final_pred, meta_test_df, rmse, out_df
