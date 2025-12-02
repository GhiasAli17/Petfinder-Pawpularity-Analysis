# src/train_utils.py
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error

def train_one_epoch_image(model, loader, optimizer, criterion, device, scale_target):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for imgs, y in loader:
        imgs = imgs.to(device)
        y = y.to(device).float().unsqueeze(1)
        if scale_target:
            y = y / 100.0

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

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


def train_one_epoch_fusion(model, loader, optimizer, criterion, device, scale_target):
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
        preds = model(imgs, tabs)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

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
