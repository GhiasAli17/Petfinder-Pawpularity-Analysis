# src/train_utils.py
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, f1_score, precision_score, recall_score

import os, gc
import copy

import pandas as pd

from torch.utils.data import DataLoader
from src.losses import saliency_loss


from src.data import build_transforms, ImageOnlyDataset, ImageTabDataset
from src.models import  VisionRegNet, FeatureConcatFusionNet, VisionAuxNet
from src.film import FiLMExternalModulation, FiLMInternalModulation, FiLMExternalSingle, FiLMInternalResModulation
from src.cross_attention import  SWINCrossAttention
from src.gate_fusion import EfficientNetGatedFusion



def train_one_epoch_image(model, loader, optimizer, criterion, device,scaler, 
                          scale_target,aux_loss_weights=None, 
                          binary_aux_tasks=None,
                          freeze_backbone=False,BN_running_state_frozen=False,
                          binary_aux_loss_type="bce",
                            binary_aux_pos_weight=None,
                            binary_aux_flip_targets=None,):
    
    cfg_binary_aux_tasks = binary_aux_tasks or []
    aux_loss_weights = aux_loss_weights or {}
    binary_aux_pos_weight = binary_aux_pos_weight or {}
    binary_aux_flip_targets = set(binary_aux_flip_targets or [])
    model.train()


    total_loss = 0.0
    total_brisque_loss = 0.0      # track brisque aux loss separately
    total_paw_loss = 0.0         # track main pawpularity loss separately
    total_vis_loss = 0.0          # track visibility aux loss separately if there is
    total_sal_loss    = 0.0
    total_binary_losses = {t: 0.0 for t in cfg_binary_aux_tasks}
    total_binary_correct = {t: 0.0 for t in cfg_binary_aux_tasks}  # 
    total_binary_counts  = {t: 0   for t in cfg_binary_aux_tasks}  # 

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
                
                for task in cfg_binary_aux_tasks:
                    if task in aux and task in preds:
                        label = aux[task].to(device).float().unsqueeze(1)
                        logits = preds[task]

                        # flip target , e.g. Face where original class 0 is minority
                        if task in binary_aux_flip_targets:
                            label_for_loss = 1.0 - label
                        else:
                            label_for_loss = label

                        if binary_aux_loss_type == "weighted_bce":
                            pos_w_val = binary_aux_pos_weight.get(task, 1.0)
                            pos_weight = torch.tensor(
                                [pos_w_val],
                                device=logits.device,
                                dtype=logits.dtype,
                            )
                            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                                logits,
                                label_for_loss,
                                pos_weight=pos_weight
                            )
                        else:    
                            bce   = torch.nn.functional.binary_cross_entropy_with_logits(
                                preds[task], label_for_loss
                            )
                        loss = loss + aux_loss_weights.get(task, 1.0) * bce
                        total_binary_losses[task] += bce.item() * imgs.size(0)    

                        #  track train accuracy for metadata 
                        with torch.no_grad():
                            pred_bin = (torch.sigmoid(preds[task]) > 0.5).float()
                            total_binary_correct[task] += (pred_bin == label).sum().item()
                            total_binary_counts[task]  += label.size(0)
                        

        sal_weight = aux_loss_weights.get("saliency", 0.0)
        if sal_weight > 0.0 and "spatial" in preds and "pet_bbox" in aux:
            sal = saliency_loss(
                preds["spatial"].float(),
                aux["pet_bbox"].to(device),
                img_size=384,
            )
            loss = loss + sal_weight * sal
            total_sal_loss += sal.item() * imgs.size(0)            


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
    if total_sal_loss > 0:                                    
        logs["loss_saliency"] = total_sal_loss / n_samples   

    for task, total in total_binary_losses.items():
        if total > 0:
            logs[f"loss_{task}"] = total / n_samples    
    # train accuracy per binary task 
    for task in cfg_binary_aux_tasks:
        if total_binary_counts[task] > 0:
            logs[f"train_acc_{task}"] = (
                total_binary_correct[task] / total_binary_counts[task]
            )
    # 
    return logs    
 

def validate_image(model, loader, device, scale_target,binary_aux_flip_targets=None):
    model.eval()
    val_preds, val_targets = [], []
    aux_pred_store   = {}  # {task: [arrays]}
    aux_target_store = {}  # {task: [arrays]}
    binary_aux_flip_targets = set(binary_aux_flip_targets or [])

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
                    if task in ("main", "spatial"):
                        continue
                    if task in aux:
                        if task not in aux_pred_store:
                            aux_pred_store[task]   = []
                            aux_target_store[task] = []
                        aux_pred_store[task].append(preds[task].cpu().numpy())
                        t = aux[task]
                        if not isinstance(t, np.ndarray):
                            t = t.numpy()
                        if task in binary_aux_flip_targets:
                            t = 1.0 - t
                        
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

        # to check if this is a binary task (all targets are 0 or 1)
        is_binary = np.all((t == 0) | (t == 1))
        if is_binary:
        # for binary tasks: report accuracy only, not MSE
            # p_binary = (1 / (1 + np.exp(-p))) > 0.5   # sigmoid threshold
            # aux_metrics[f"{task}_val_acc"] = float(np.mean(p_binary == t))
            p_clipped = np.clip(1 / (1 + np.exp(-p)), 1e-7, 1 - 1e-7)
            val_bce = -np.mean(t * np.log(p_clipped) + (1 - t) * np.log(1 - p_clipped))
            aux_metrics[f"{task}_val_bce"] = float(val_bce)

            #  val accuracy 
            p_binary = (1 / (1 + np.exp(-p))) > 0.5
            aux_metrics[f"{task}_val_acc"] = float(np.mean(p_binary == t))
            

            # to track F1, precision, recall for binary aux tasks
            
            aux_metrics[f"{task}_val_f1"]   = float(f1_score(t, p_binary, zero_division=0))
            aux_metrics[f"{task}_val_prec"] = float(precision_score(t, p_binary, zero_division=0))
            aux_metrics[f"{task}_val_rec"]  = float(recall_score(t, p_binary, zero_division=0))
        else:
            # for regression tasks: report MSE only
            aux_metrics[f"{task}_val_mse"] = float(np.mean((p - t) ** 2))



    return rmse, val_preds, val_targets, aux_metrics

def train_one_epoch_fusion(model, loader, optimizer, criterion, device,scaler, scale_target, freeze_backbone=False,BN_running_state_frozen=False):
    model.train()

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
    binary_aux_tasks = cfg.get("binary_aux_tasks", [])
    aux_loss_weights = cfg.get("aux_loss_weights", {})
    binary_aux_loss_type = cfg.get("binary_aux_loss_type", "bce")
    binary_aux_pos_weight = cfg.get("binary_aux_pos_weight", {})
    binary_aux_flip_targets = cfg.get("binary_aux_flip_targets", [])


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
        train_ds = ImageOnlyDataset(train_df, img_folder, train_tf, aux_tasks=aux_tasks,binary_aux_tasks=binary_aux_tasks)
        val_ds   = ImageOnlyDataset(val_df,   img_folder, val_tf, aux_tasks=aux_tasks,binary_aux_tasks=binary_aux_tasks)
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


    if mode == "image": #image model without metadata and without fusion
        use_saliency = cfg.get("use_saliency", False)
        if len(aux_tasks) == 0 and not use_saliency and len(binary_aux_tasks) == 0:
            model = VisionRegNet(
            backbone_name=backbone_name,
            img_size=img_size,
            head_type=cfg.get("head_type", "linear"),
            head_hidden=256,
            pretrained=True,
        ).to(device)
        else:
            model = VisionAuxNet(
                backbone_name=backbone_name,
                img_size=img_size,
                aux_tasks=aux_tasks,
                binary_aux_tasks=binary_aux_tasks,
                pretrained=True,
                head_type=cfg.get("head_type", "linear"),
                head_hidden=256,
                use_saliency=cfg.get("use_saliency", False)
            ).to(device)
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
    elif loss_name == "focal_mse":
        from src.losses import focal_mse_loss
        focal_gamma = cfg.get("focal_gamma", 1.0)
        criterion = lambda pred, target: focal_mse_loss(pred, target, gamma=focal_gamma)
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
                binary_aux_tasks=binary_aux_tasks,
                binary_aux_loss_type=binary_aux_loss_type,
                binary_aux_pos_weight=binary_aux_pos_weight,
                binary_aux_flip_targets=binary_aux_flip_targets,
            )
            avg_train_loss = train_out["loss"] if isinstance(train_out, dict) else train_out #total
            avg_paw_loss   = train_out.get("loss_paw", avg_train_loss) #pawpularity only
            rmse, val_preds, val_targets, aux_metrics = validate_image(
                model, val_loader, device, scale_target,
                binary_aux_flip_targets=binary_aux_flip_targets,
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

         
        use_saliency = cfg.get("use_saliency", False)
        if len(aux_tasks) > 0 or use_saliency or len(binary_aux_tasks) > 0:
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
        # if mode == "image":
        #     for k, v in aux_metrics.items():
        #         aux_str += f" | {k}: {v:.4f}"
        aux_str = ""
        if mode == "image":
            # training aux losses
            if isinstance(train_out, dict):
                for k, v in train_out.items():
                    if k not in ("loss", "loss_paw"):
                        aux_str += f" | train_{k}: {v:.4f}"
            # validation aux metrics
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
        _, val_preds, val_targets, _ = validate_image(
            model, val_loader, device, scale_target,
            binary_aux_flip_targets=binary_aux_flip_targets
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



