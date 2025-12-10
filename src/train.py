# src/train_utils.py
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
import os, gc
import copy

import pandas as pd

from torch.utils.data import DataLoader



from src.data import build_transforms, ImageOnlyDataset, ImageTabDataset
from src.models import build_vision_backbone, EarlyFusionNet



def train_one_epoch_image(model, loader, optimizer, criterion, device,scaler, scale_target):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for imgs, y in loader:
        imgs = imgs.to(device)
        y = y.to(device).float().unsqueeze(1)
        if scale_target:
            y = y / 100.0

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            preds = model(imgs)
            loss = criterion(preds, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
    return total_loss / n_samples


def validate_image(model, loader, device, scale_target):
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            if scale_target:
                probs = torch.sigmoid(preds).cpu().numpy().squeeze()
                out = probs * 100.0
            else:
                out = preds.cpu().numpy().squeeze()
            val_preds.append(out)
            val_targets.append(y.numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    rmse = root_mean_squared_error(val_targets, val_preds)
    return rmse, val_preds, val_targets


def train_one_epoch_fusion(model, loader, optimizer, criterion, device,scaler, scale_target):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for imgs, tabs, y in loader:
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        y = y.to(device).float().unsqueeze(1)
        if scale_target:
            y = y / 100.0

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            preds = model(imgs, tabs)
            loss = criterion(preds, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
    return total_loss / n_samples


def validate_fusion(model, loader, device, scale_target):
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, tabs, y in loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            preds = model(imgs, tabs)
            if scale_target:
                probs = torch.sigmoid(preds).cpu().numpy().squeeze()
                out = probs * 100.0
            else:
                out = preds.cpu().numpy().squeeze()
            val_preds.append(out)
            val_targets.append(y.numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    rmse = root_mean_squared_error(val_targets, val_preds)
    return rmse, val_preds, val_targets


def extract_image_tab_features(loader, backbone, device):
    img_feats_list, tab_feats_list, y_list = [], [], []
    backbone.eval()
    with torch.no_grad():
        for imgs, tabs, y in loader:
            imgs = imgs.to(device)
            img_feats = backbone(imgs).cpu().numpy()
            tab_feats = tabs.numpy()
            img_feats_list.append(img_feats)
            tab_feats_list.append(tab_feats)
            y_list.append(y.numpy())
    img_feats = np.concatenate(img_feats_list)
    tab_feats = np.concatenate(tab_feats_list)
    y = np.concatenate(y_list)
    return img_feats, tab_feats, y


def run_single_fold(
    fold,
    train_df,
    val_df,
    img_folder,
    cfg,
    out_dir,
    device,
    mode,               # "image" or "fusion"
    tab_cols=None,
    workers=8,
    pin_memory=True,
    persistent_workers=False,
):
    """
    Run ONE fold and return (best_rmse, val_preds, val_ids, val_targets).

    mode="image":  ImageOnlyDataset + build_vision_backbone + train_one_epoch_image / validate_image
    mode="fusion": ImageTabDataset  + EarlyFusionNet      + train_one_epoch_fusion / validate_fusion
    """
    backbone_name = cfg["backbone"]
    img_size = cfg["img_size"]
    aug_type = cfg["aug"]
    loss_name = cfg["loss"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    patience = cfg["patience"]

    train_tf = build_transforms(img_size, aug_type, train=True)
    val_tf   = build_transforms(img_size, aug_type, train=False)

    if mode == "image":
        train_ds = ImageOnlyDataset(train_df, img_folder, train_tf)
        val_ds   = ImageOnlyDataset(val_df,   img_folder, val_tf)
    elif mode == "fusion":
        assert tab_cols is not None
        train_ds = ImageTabDataset(train_df, img_folder, tab_cols, train_tf)
        val_ds   = ImageTabDataset(val_df,   img_folder, tab_cols, val_tf)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )

    if mode == "image":
        model = build_vision_backbone(
            backbone_name, img_size, mode="regression"
        ).to(device)
    else:  # fusion
        model = EarlyFusionNet(
            backbone_name=backbone_name,
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            head_type=cfg["head_type"],
            pretrained=True,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    if loss_name == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
        scale_target = True
    else:
        criterion = torch.nn.MSELoss()
        scale_target = False

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = torch.amp.GradScaler("cuda")

    best_rmse = 1e10
    best_state = None
    epochs_no_improve = 0
    train_losses, val_rmses = [], []
    epoch_logs = []

    for epoch in range(epochs):
        if mode == "image":
            avg_train_loss = train_one_epoch_image(
                model, train_loader, optimizer, criterion, device, scaler, scale_target
            )
            rmse, val_preds, val_targets = validate_image(
                model, val_loader, device, scale_target
            )
        else:
            avg_train_loss = train_one_epoch_fusion(
                model, train_loader, optimizer, criterion, device, scaler, scale_target
            )
            rmse, val_preds, val_targets = validate_fusion(
                model, val_loader, device, scale_target
            )

        scheduler.step()
        train_losses.append(avg_train_loss)
        val_rmses.append(rmse)

        # print(
        #     f"Epoch {epoch+1}/{epochs} | Fold {fold} "
        #     f"TrainLoss: {avg_train_loss:.4f} ValRMSE: {rmse:.4f}"
        # )
        display_train = (
            f"RMSE={np.sqrt(avg_train_loss):.4f}"
            if loss_name == "mse"
            else f"Loss={avg_train_loss:.4f}"
        )
        print(
            f"Epoch {epoch+1}/{epochs} | Fold {fold} | "
            f"Train[{loss_name.upper()}]: {display_train} "
            f"| ValRMSE: {rmse:.4f}"
        )
        epoch_logs.append({
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_rmse": float(rmse),
        })

        if rmse < best_rmse:
            best_rmse = rmse
            # best_state = model.state_dict() shallow copy and it will always save the last epoch, instead of best model 
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # logs per fold
    pd.DataFrame(epoch_logs).to_csv(
        os.path.join(out_dir, f"epoch_logs_fold{fold}.csv"), index=False
    )
    hist_df = pd.DataFrame({
        "epoch": range(1, len(train_losses)+1),
        "train_loss": train_losses,
        "train_rmse": np.sqrt(train_losses) if loss_name=="mse" else np.nan,
        "val_rmse": val_rmses,
    })
    hist_df.to_csv(os.path.join(out_dir, f"history_fold{fold}.csv"),
                   index=False)

    # best weights & preds
    model.load_state_dict(best_state)
    if mode == "image":
        _, val_preds, val_targets = validate_image(
            model, val_loader, device, scale_target
        )
    else:
        _, val_preds, val_targets = validate_fusion(
            model, val_loader, device, scale_target
        )

    torch.save(best_state,
               os.path.join(out_dir, f"model_fold{fold}.pt"))

    val_ids = val_df["Id"].values
    del model, optimizer, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    gc.collect()

    return best_rmse, np.array(val_preds), np.array(val_targets), val_ids
