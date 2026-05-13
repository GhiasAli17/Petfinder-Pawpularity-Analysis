# src/train_utils.py
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
import os, gc
import copy

import pandas as pd

from torch.utils.data import DataLoader



from src.data import build_transforms, ImageOnlyDataset, ImageTabDataset
from src.models import  build_vision_backbone, FeatureConcatFusionNet, VisionAuxNet
from src.film import FiLMExternalModulation, FiLMInternalModulation, FiLMExternalSingle, FiLMInternalResModulation
from src.cross_attention import  SWINCrossAttention
from src.gate_fusion import EfficientNetGatedFusion



def train_one_epoch_image(model, loader, optimizer, criterion, device,scaler, scale_target,aux_loss_weights=None, freeze_backbone=False,BN_running_state_frozen=False):
    aux_loss_weights = aux_loss_weights or {}
    model.train()
    # if freeze_backbone and BN_running_state_frozen:
    #     for m in model.modules():
    #         if isinstance(m, torch.nn.BatchNorm2d):
    #             m.eval()

    total_loss = 0.0
    total_brisque_loss = 0.0      # track brisque aux loss separately
    total_paw_loss = 0.0         # track main pawpularity loss separately
    total_vis_loss = 0.0          # track visibility aux loss separately if there is
    n_samples = 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, y = batch
            aux = {}
        else:
            imgs, y, aux = batch
        imgs = imgs.to(device)
        y = y.to(device).float().unsqueeze(1)
        if scale_target:
            y = y / 100.0

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):

            preds = model(imgs)  
            # loss = criterion(preds, y)

            # baseline: model outputs a tensor
            if not isinstance(preds, dict):
                loss = criterion(preds, y)
                total_paw_loss += loss.item() * imgs.size(0)
            else:
                # VisionAuxNet: preds is a dict
                main_pred = preds["main"]
                loss = criterion(main_pred, y)
                total_paw_loss += loss.item() * imgs.size(0)


                # Add auxiliary losses if available in this batch
                if "brisque" in aux:
                    brisque_t = aux["brisque"].to(device).float().unsqueeze(1)
                    brisque_loss = torch.nn.functional.mse_loss(
                        preds["brisque"], brisque_t
                    )
                    loss = loss + aux_loss_weights.get("brisque", 1.0) * brisque_loss
                    total_brisque_loss += brisque_loss.item() * imgs.size(0)  # track total brisque loss

                if "visibility_ratio" in aux:
                    vis_t = aux["visibility_ratio"].to(device).float().unsqueeze(1)
                    vis_loss = torch.nn.functional.mse_loss(
                        preds["visibility_ratio"], vis_t
                    )
                    loss = loss + aux_loss_weights.get("visibility_ratio", 1.0) * vis_loss
                    total_vis_loss += vis_loss.item() * imgs.size(0)  # track total visibility loss


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)

    logs = {"loss": total_loss / n_samples, "loss_paw": total_paw_loss / n_samples}
    if total_brisque_loss > 0:
        logs["loss_brisque"] = total_brisque_loss / n_samples
    if total_vis_loss > 0:
        logs["loss_visibility_ratio"] = total_vis_loss / n_samples
    return logs    
 

def validate_image(model, loader, device, scale_target):
    model.eval()
    val_preds, val_targets = [], []
    aux_pred_store   = {}  # {task: [arrays]}
    aux_target_store = {}  # {task: [arrays]}

    with torch.no_grad():
        for batch in loader: 
            if len(batch) == 2:# unpack batch — 2 items if no aux tasks, >2 if aux tasks present
                imgs, y = batch
                aux = {}  # no aux label
            else:
                imgs, y, aux = batch # aux is a dict: {"brisque": tensor, ...}

            imgs = imgs.to(device)
            preds = model(imgs)

            # --- baseline model: output is a plain tensor ---
            if not isinstance(preds, dict):
                main_pred = preds
            # --- VisionAuxNet: output is a dict {"main", "brisque", ...} ---
            else:
                main_pred = preds["main"]
                # generic: collect any aux task present in both preds and aux
                for task in preds:
                    if task == "main":
                        continue
                    if task in aux:
                        if task not in aux_pred_store:
                            aux_pred_store[task]   = []
                            aux_target_store[task] = []
                        aux_pred_store[task].append(preds[task].cpu().numpy())
                        t = aux[task]
                        if not isinstance(t, np.ndarray):
                            t = t.numpy()
                        aux_target_store[task].append(t)

            if scale_target:
                out = torch.sigmoid(main_pred).cpu().numpy().squeeze() * 100.0
            else:
                out = main_pred.cpu().numpy().squeeze()

            val_preds.append(out)
            val_targets.append(y.numpy())

    val_preds  = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    rmse = root_mean_squared_error(val_targets, val_preds)

    # generic: compute MSE for every collected aux task
    aux_metrics = {}
    for task in aux_pred_store:
        p = np.concatenate(aux_pred_store[task]).reshape(-1)
        t = np.concatenate(aux_target_store[task]).reshape(-1)
        aux_metrics[f"{task}_val_mse"] = float(np.mean((p - t) ** 2))

    return rmse, val_preds, val_targets, aux_metrics

def train_one_epoch_fusion(model, loader, optimizer, criterion, device,scaler, scale_target, freeze_backbone=False,BN_running_state_frozen=False):
    model.train()
    # if freeze_backbone and BN_running_state_frozen:
    #     for name, m in model.named_modules():
    #         if isinstance(m, torch.nn.BatchNorm2d):
    #             m.eval()

    total_loss = 0.0
    n_samples = 0
   
    
    for batch_idx, (imgs, tabs, y) in enumerate(loader):
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
    tab_encoder_capacity="small", # for tab encoder MLP, "small" is 2-layer MLP, "big" is deeper/wider 5-layer MLP
    ##film related parameter
    film_start_idx=5,                   # choose based on EfficientNet stage
    apply_to_all_after=False,            # True: apply FiLM to all blocks >= idx, False: only at idx
    use_bn_affine=False,  
    freeze_backbone=False,
    BN_running_state_frozen=False, # If True, also freeze BN running stats (mean/var) in addition to affine params
    identity_init=False, # film start as identity instead of random initialization
     # cross-attention specific
    cross_attn_num_heads=8,
    cross_attn_query_mode="tab_queries_image",  # or "image_queries_tab"
    cross_attn_head_hidden=256,
    cross_attn_dropout=0.1,
    num_cross_attn_blocks=1,
    use_global_image_feature=True,
):
    """
    Run ONE fold and return (best_rmse, val_preds, val_ids, val_targets).

    mode="image":  ImageOnlyDataset + build_vision_backbone + train_one_epoch_image / validate_image
    mode="fusion": ImageTabDataset  + FeatureConcatFusionNet      + train_one_epoch_fusion / validate_fusion
    """
    aux_tasks = cfg.get("aux_tasks", [])
    aux_loss_weights = cfg.get("aux_loss_weights", {})
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
        train_ds = ImageOnlyDataset(train_df, img_folder, train_tf, aux_tasks=aux_tasks)
        val_ds   = ImageOnlyDataset(val_df,   img_folder, val_tf, aux_tasks=aux_tasks)
    elif mode == "fusion" or "film" in mode or mode == "gated_effb1" or "cross" in mode:
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


    if mode == "image": #it means only image model without metadata and without fusion
        if len(aux_tasks) == 0:
             model = build_vision_backbone(
                backbone_name, img_size, mode="regression"
            ).to(device)
        else:
            model = VisionAuxNet(
                backbone_name=backbone_name,
                img_size=img_size,
                aux_tasks=aux_tasks,
                pretrained=True,
            ).to(device)
        # model = build_vision_backbone(
        #     backbone_name, img_size, mode="regression"
        # ).to(device)
    elif mode == "film_fusion": # it shows FiLM modulating internal blocks of Image Backbone
        model = FiLMInternalModulation(
            backbone_name=backbone_name,        # e.g., "tf_efficientnet_b1"
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            tab_hidden=64,
            film_start_idx=film_start_idx,                   # choose based on EffB1 block count
            apply_to_all_after=apply_to_all_after,            # True: all blocks >= idx, False: only at idx
            head_hidden=256,
            pretrained=True,
            use_bn_affine=use_bn_affine,                # False => BN only normalizes, FiLM affine
            freeze_backbone=freeze_backbone,  # False => finetune EfficientNet-B1,True -> Freezen backbone entirely 
            tab_encoder_capacity=tab_encoder_capacity, # "small" or "big" for tab encoder MLP
            identity_init=identity_init # to init film from identity function and then learn accroding
        ).to(device)
    elif mode == "film_stack_effb1" or mode == "film_stack_effb4": # it shows the FiLM applied after feature extraction from the image backbone and FiLM extra Blocks are added 
        model = FiLMExternalModulation(
            backbone_name=backbone_name,
            tab_input_dim=len(tab_cols),
            tab_hidden=64,
            num_film_blocks=4,
            # head_hidden=256,
            pretrained_backbone=True,
            freeze_backbone=freeze_backbone,  # False => finetune EfficientNet-B1
            tab_encoder_capacity=tab_encoder_capacity, # "small" or "big" for tab encoder MLP
            identity_init=identity_init

        ).to(device) 
    elif mode == "film_external_single":
        model = FiLMExternalSingle(
            backbone_name=backbone_name,
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            tab_hidden=64,
            pretrained_backbone=True,
            freeze_backbone=freeze_backbone,
            tab_encoder_capacity=tab_encoder_capacity,  # 
            identity_init=identity_init
        ).to(device)   
    elif mode == "film_internal_res":
        model = FiLMInternalResModulation(
            backbone_name=backbone_name,
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            tab_hidden=64,
            film_start_idx=film_start_idx,
            num_film_blocks=4,
            head_hidden=256,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            tab_encoder_capacity=tab_encoder_capacity,
            identity_init=identity_init,
        ).to(device)     
    elif mode == "cross_attn_swin":
        model = SWINCrossAttention(
            backbone_name=backbone_name,
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            tab_hidden=64,
            num_heads=cross_attn_num_heads,
            head_hidden=cross_attn_head_hidden,
            dropout=cross_attn_dropout,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            tab_encoder_capacity=tab_encoder_capacity,
            query_mode=cross_attn_query_mode,
            num_cross_attn_blocks = num_cross_attn_blocks,
            use_global_image_feature=use_global_image_feature,
        ).to(device)    
    elif mode == "gated_effb1":
        model = EfficientNetGatedFusion(
            backbone_name=backbone_name,
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            tab_hidden=64,
            head_hidden=256,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            tab_encoder_capacity=tab_encoder_capacity,
        ).to(device)                 
    else:  # fusion concat based without FiLM
         model = FeatureConcatFusionNet(
            backbone_name=backbone_name,
            tab_encoder_capacity=tab_encoder_capacity, # "small" or "big" for tab encoder MLP
            img_size=img_size,
            tab_input_dim=len(tab_cols),
            head_type=cfg["head_type"],
            pretrained=True,
            freeze_backbone=freeze_backbone,
        ).to(device)


    
    # To Only train parameters that still require gradients
    if freeze_backbone:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay
    )

    if loss_name == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
        scale_target = True  
    elif loss_name == "mse":
        criterion = torch.nn.MSELoss()
        scale_target = False
    else:
        raise ValueError("incorrect loss passed")    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    
    best_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_paw_losses , train_total_losses,val_rmses = [], [],[]
    epoch_logs = []

    for epoch in range(epochs):
        aux_metrics = {}  # default, overwritten in image mode
        train_out = {}     # default; overwritten in image mode per epoch
        if mode == "image": 
            train_out = train_one_epoch_image(
                model, train_loader, optimizer, criterion, device, scaler, scale_target,
                aux_loss_weights=aux_loss_weights,
                freeze_backbone=freeze_backbone,
                BN_running_state_frozen=BN_running_state_frozen,
            )
            avg_train_loss = train_out["loss"] if isinstance(train_out, dict) else train_out #total
            avg_paw_loss   = train_out.get("loss_paw", avg_train_loss) #pawpularity only
            rmse, val_preds, val_targets, aux_metrics = validate_image(
                model, val_loader, device, scale_target,
            )
        else:
            avg_train_loss = train_one_epoch_fusion(
                model, train_loader, optimizer, criterion, device, scaler, scale_target, freeze_backbone,BN_running_state_frozen
            )
            avg_paw_loss = avg_train_loss # fusion has no aux, total = paw
            rmse, val_preds, val_targets = validate_fusion(
                model, val_loader, device, scale_target
            )

        scheduler.step()
        train_total_losses.append(avg_train_loss)
        if mode == "image" and isinstance(train_out, dict):
            train_paw_losses.append(train_out.get("loss_paw", avg_train_loss))
        else:
            train_paw_losses.append(avg_train_loss)   # fusion mode: total = paw
        val_rmses.append(rmse)

         
        if len(aux_tasks) > 0:
            display_train = (
            f"PawRMSE={np.sqrt(avg_paw_loss):.4f} TotalLoss={avg_train_loss:.4f}"
            if loss_name == "mse"
            else f"PawLoss={avg_paw_loss:.4f} TotalLoss={avg_train_loss:.4f}"
        )
        else:
            display_train = (
                f"RMSE={np.sqrt(avg_train_loss):.4f}"
                if loss_name == "mse"
                else f"Loss={avg_train_loss:.4f}"
            )
       

        aux_str = ""
        if mode == "image":
            for k, v in aux_metrics.items():
                aux_str += f" | {k}: {v:.4f}"
        print(
            f"Epoch {epoch+1}/{epochs} | Fold {fold} | "
            f"Train[{loss_name.upper()}]: {display_train} "
            f"| ValRMSE: {rmse:.4f}{aux_str}"
        )

        log_row = {
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),#total
            "train_paw_loss": float(avg_paw_loss),     # paw-only
            "val_rmse": float(rmse),
        }

        if mode == "image":
            # generic: log all train aux losses
            if isinstance(train_out, dict):
                for k, v in train_out.items():
                    if k not in ("loss", "loss_paw"):
                        log_row[k] = float(v)
            # generic: log all val aux metrics
            for k, v in aux_metrics.items():
                log_row[k] = float(v)
        epoch_logs.append(log_row)

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
    # hist_df = pd.DataFrame({
    #     "epoch": range(1, len(train_losses)+1),
    #     "train_loss": train_losses,
    #     "train_rmse": np.sqrt(train_losses) if loss_name in ("mse") else np.nan,
    #     "val_rmse": val_rmses,
    # })
    hist_df = pd.DataFrame({
        "epoch":           range(1, len(train_paw_losses) + 1),
        "train_loss":      train_total_losses,    # total (paw + weighted aux)
        "train_paw_loss":  train_paw_losses,      # paw-only
        "train_rmse":      (np.sqrt(train_paw_losses)
                            if loss_name == "mse"
                            else [float("nan")] * len(train_paw_losses)),
        "val_rmse":        val_rmses,
    })
    hist_df.to_csv(os.path.join(out_dir, f"history_fold{fold}.csv"),
                   index=False)

    # best weights & preds
    model.load_state_dict(best_state)
    if mode == "image":
        # _, val_preds, val_targets = validate_image(
        #     model, val_loader, device, scale_target
        # )
        _, val_preds, val_targets, _ = validate_image(
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


def debug_bn_states(model, label=""):
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            rm = m.running_mean.detach().cpu().numpy()
            rv = m.running_var.detach().cpu().numpy()
            print(f"{label} | {name:30s} | "
                f"mean_range=[{rm.min():.6f}, {rm.max():.6f}] | "
                f"var_range=[{rv.min():.6f}, {rv.max():.6f}] | "
                f"mean_std={rm.std():.6f} | var_std={rv.std():.6f}")
