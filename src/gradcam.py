#src/gradcam.py
"""
GradCAM for Swin Transformer (timm) with visualization and comparison utilities.

"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import KFold

from src.models import VisionAuxNet, VisionRegNet, FeatureConcatFusionNet
from src.cross_attention import SWINCrossAttention
from src.data import build_transforms



# Shape helpers,


def _factorize_grid(n: int):
    """
    Return (H, W) closest to square with H * W == n.
    Works for any n (square or not).
        144 -> (12, 12)   108 -> (9, 12)   12 -> (3, 4)
    """
    h = int(np.sqrt(n))
    while h >= 1:
        if n % h == 0:
            return h, n // h
        h -= 1
    return 1, n


def _get_swin_output_grid(backbone):
    """
     to read final spatial (H, W) from timm SwinTransformer attributes.
    Returns (H, W) or None.
    """
    if hasattr(backbone, "layers"):
        last = backbone.layers[-1]
        for attr in ("grid", "input_resolution"):
            g = getattr(last, attr, None)
            if isinstance(g, (tuple, list)) and len(g) == 2:
                return tuple(int(x) for x in g)

    # fallback from patch_embed and PatchMerging stages
    if hasattr(backbone, "patch_embed"):
        pe = backbone.patch_embed
        if hasattr(pe, "grid_size"):
            H0, W0 = pe.grid_size
        elif hasattr(pe, "img_size") and hasattr(pe, "patch_size"):
            img = pe.img_size
            ps = pe.patch_size
            H0 = (img[0] if isinstance(img, (tuple, list)) else img) // (
                ps[0] if isinstance(ps, (tuple, list)) else ps
            )
            W0 = (img[1] if isinstance(img, (tuple, list)) else img) // (
                ps[1] if isinstance(ps, (tuple, list)) else ps
            )
        else:
            return None

        if hasattr(backbone, "layers"):
            for stage in backbone.layers:
                if getattr(stage, "downsample", None) is not None:
                    H0 //= 2
                    W0 //= 2
        return H0, W0

    return None



# GradCAM for image-only models


class GradCAMSwin:
    """
    GradCAM for Swin Transformer (timm).

    Parameters
    ----------
    model        : nn.Module
    target_layer : nn.Module
    device       : torch.device
    grid_size    : (H, W) optional override, e.g. (12, 12)
    """

    def __init__(self, model, target_layer, device, grid_size=None):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self._grid_size = grid_size

        self._gradients = None
        self._activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Forward hook stores activations.
        Backward hook stores gradients.
        """
        def fwd_hook(module, inp, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = (
                grad_out[0].detach()
                if (grad_out and grad_out[0] is not None)
                else None
            )

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, img_tensor, target_head: str = "main") -> np.ndarray:
        """
        img_tensor  : (1, 3, H, W)
        target_head : used when model returns dict, e.g. VisionAuxNet
        """
        self.model.eval()
        self.model.zero_grad()
        self._gradients = None
        self._activations = None

        img_tensor = img_tensor.to(self.device)

        out = self.model(img_tensor)
        score = out[target_head] if isinstance(out, dict) else out
        score.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "GradCAM: hooks did not capture gradients/activations."
            )

        grads = self._gradients
        acts = self._activations

        if acts.dim() == 4:
            cam = self._cam_from_bhwc(grads, acts)
        elif acts.dim() == 3:
            cam = self._cam_from_bnc(grads, acts)
        elif acts.dim() == 2:
            weights = grads.mean(dim=0, keepdim=True)
            flat = torch.relu((weights * acts).sum(dim=-1))
            cam = flat.cpu().numpy().reshape(1, 1)
        else:
            raise RuntimeError(f"GradCAM: unexpected activation shape {tuple(acts.shape)}")

        c_min, c_max = float(cam.min()), float(cam.max())
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam.astype(np.float32)

    def _cam_from_bhwc(self, grads, acts):
        # (B, H, W, C)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=-1)
        cam = torch.relu(cam).squeeze(0)
        return cam.cpu().numpy()

    def _cam_from_bnc(self, grads, acts):
        # (B, N, C)
        weights = grads.mean(dim=1, keepdim=True)
        cam = (weights * acts).sum(dim=-1)
        cam = torch.relu(cam).squeeze(0)
        flat = cam.cpu().numpy()

        n = flat.shape[0]
        h, w = self._resolve_grid(n)

        if h * w != n:
            sq = int(np.sqrt(n))
            flat = flat[: sq * sq]
            h = w = sq

        return flat.reshape(h, w)

    def _resolve_grid(self, n_tokens: int):
        if self._grid_size is not None:
            H, W = self._grid_size
            if H * W == n_tokens:
                return H, W

        backbone = getattr(self.model, "backbone", self.model)
        hw = _get_swin_output_grid(backbone)
        if hw is not None and hw[0] * hw[1] == n_tokens:
            return hw

        return _factorize_grid(n_tokens)



# GradCAM for fusion models


class GradCAMSwinFusion:
    """
    GradCAM for models that require forward(img, tab),
    e.g. FeatureConcatFusionNet and SWINCrossAttention.
    """

    def __init__(self, model, target_layer, device, grid_size=None):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self._grid_size = grid_size

        self._gradients = None
        self._activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = (
                grad_out[0].detach()
                if (grad_out and grad_out[0] is not None)
                else None
            )

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, img_tensor, tab_tensor, target_head="main"):
        self.model.eval()
        self.model.zero_grad()
        self._gradients = None
        self._activations = None

        img_tensor = img_tensor.to(self.device)
        tab_tensor = tab_tensor.to(self.device)

        out = self.model(img_tensor, tab_tensor)
        score = out[target_head] if isinstance(out, dict) else out
        score.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM fusion: hooks did not capture gradients/activations.")

        grads = self._gradients
        acts = self._activations

        if acts.dim() == 4:
            weights = grads.mean(dim=(1, 2), keepdim=True)
            cam = (weights * acts).sum(dim=-1)
            cam = torch.relu(cam).squeeze(0).cpu().numpy()
        elif acts.dim() == 3:
            weights = grads.mean(dim=1, keepdim=True)
            cam = (weights * acts).sum(dim=-1)
            cam = torch.relu(cam).squeeze(0).cpu().numpy()

            n = cam.shape[0]
            if self._grid_size is not None and self._grid_size[0] * self._grid_size[1] == n:
                h, w = self._grid_size
            else:
                h, w = _factorize_grid(n)
            cam = cam.reshape(h, w)
        else:
            raise RuntimeError(f"Unexpected activation shape: {tuple(acts.shape)}")

        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam.astype(np.float32)



# Visualisation helpers


def cam_to_heatmap(cam: np.ndarray, img_pil: Image.Image, alpha: float = 0.5):
    """
    Overlay CAM on original PIL image.

    Returns
    -------
    blended     : PIL Image
    cam_resized : (H_img, W_img) uint8
    """
    W, H = img_pil.size

    cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    cam_rsz = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(cam_rsz, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = np.array(img_pil.convert("RGB"))
    blended = (alpha * heatmap + (1 - alpha) * img_np).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended), cam_rsz


def compute_cam_mask_overlap(
    cam_resized: np.ndarray,
    bbox,
    img_w: int,
    img_h: int,
    cam_threshold: float = 0.5,
) -> float:
    """
    IoU between high-activation CAM region and bbox.
    """
    threshold = float(cam_resized.max()) * cam_threshold
    cam_bin = (cam_resized.astype(np.float32) >= threshold)

    x1 = int(max(0, float(bbox[0])))
    y1 = int(max(0, float(bbox[1])))
    x2 = int(min(img_w, float(bbox[2])))
    y2 = int(min(img_h, float(bbox[3])))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_bin = np.zeros((img_h, img_w), dtype=np.float32)
    bbox_bin[y1:y2, x1:x2] = 1.0

    intersection = float((cam_bin * bbox_bin).sum())
    union = float(np.clip(cam_bin + bbox_bin, 0, 1).sum())
    return intersection / union if union > 0 else 0.0

def make_face_proxy_bbox(
    bbox,
    img_w: int,
    img_h: int,
    x_frac=(0.25, 0.75),
    y_frac=(0.0, 0.45),
):
    """
    Heuristic face-region proxy inside the YOLO pet bbox.

    Assumes the pet face is roughly in the upper-center of the YOLO pet box.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    fx1 = int(max(0, min(img_w, x1 + x_frac[0] * bw)))
    fx2 = int(max(0, min(img_w, x1 + x_frac[1] * bw)))
    fy1 = int(max(0, min(img_h, y1 + y_frac[0] * bh)))
    fy2 = int(max(0, min(img_h, y1 + y_frac[1] * bh)))
    return (fx1, fy1, fx2, fy2)

# Model loading helpers


def load_baseline(fold, device, backbone, img_size, baseline_dir):
    model = VisionRegNet(
        backbone_name=backbone,
        img_size=img_size,
        head_type="mlp",
        pretrained=False,
    ).to(device)

    ckpt = os.path.join(baseline_dir, f"model_fold{fold}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def load_metadata_aux_model(fold, device, aux_name, aux_model_config, backbone, img_size):
    cfg = aux_model_config[aux_name]

    model = VisionAuxNet(
        backbone_name=backbone,
        img_size=img_size,
        aux_tasks=[],
        binary_aux_tasks=[cfg["binary_name"]],
        head_type="mlp",
        pretrained=False,
        use_saliency=False,
    ).to(device)

    ckpt = os.path.join(cfg["dir"], f"model_fold{fold}.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg["binary_name"], cfg["is_bce"]

def load_pseudo_aux_or_saliency_model(
    fold,
    device,
    model_name,
    pseudo_aux_config,
    backbone,
    img_size,
):
    cfg = pseudo_aux_config[model_name]

    aux_tasks = []
    use_saliency = bool(cfg.get("use_saliency", False))

    if cfg.get("aux_name") is not None:
        aux_tasks = [cfg["aux_name"]]

    model = VisionAuxNet(
        backbone_name=backbone,
        img_size=img_size,
        aux_tasks=aux_tasks,
        binary_aux_tasks=[],
        head_type=cfg.get("head_type", "mlp"),
        pretrained=False,
        use_saliency=use_saliency,
    ).to(device)

    ckpt = os.path.join(cfg["dir"], f"model_fold{fold}.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg.get("aux_name"), False

def load_concat_model(fold, device, tab_input_dim, fusion_model_config, backbone, img_size):
    cfg = fusion_model_config["concat"]

    model = FeatureConcatFusionNet(
        backbone_name=backbone,
        img_size=img_size,
        tab_input_dim=tab_input_dim,
        tab_hidden=cfg["tab_hidden"],
        fusion_hidden=cfg["fusion_hidden"],
        head_type=cfg["head_type"],
        pretrained=False,
        freeze_backbone=cfg["freeze_backbone"],
        tab_encoder_capacity=cfg["tab_encoder_capacity"],
    ).to(device)

    ckpt = os.path.join(cfg["dir"], f"model_fold{fold}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    return model


def load_crossattn_model(fold, device, tab_input_dim, fusion_model_config, backbone, img_size):
    cfg = fusion_model_config["crossattn"]

    model = SWINCrossAttention(
        backbone_name=backbone,
        img_size=img_size,
        tab_input_dim=tab_input_dim,
        tab_hidden=cfg["tab_hidden"],
        num_heads=cfg["num_heads"],
        head_hidden=cfg["head_hidden"],
        dropout=cfg["dropout"],
        pretrained=False,
        freeze_backbone=cfg["freeze_backbone"],
        tab_encoder_capacity=cfg["tab_encoder_capacity"],
        query_mode=cfg["query_mode"],
        num_cross_attn_blocks=cfg["num_cross_attn_blocks"],
        use_global_image_feature=cfg["use_global_image_feature"],
    ).to(device)

    ckpt = os.path.join(cfg["dir"], f"model_fold{fold}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    return model


def load_named_model(
    fold,
    device,
    model_name,
    backbone,
    img_size,
    baseline_dir,
    aux_model_config,
    fusion_model_config,
    tab_input_dim=None,
    pseudo_aux_config=None,
):
    """
    Returns: model, aux_key, aux_is_bce, needs_tabular
    """
    if model_name == "main":
        return load_baseline(fold, device, backbone, img_size, baseline_dir), None, False, False

    if model_name in aux_model_config:
        model, aux_key, is_bce = load_metadata_aux_model(
            fold, device, model_name, aux_model_config, backbone, img_size
        )
        return model, aux_key, is_bce, False

    if model_name == "concat":
        if tab_input_dim is None:
            raise ValueError("tab_input_dim must be provided for concat model")
        model = load_concat_model(
            fold, device, tab_input_dim, fusion_model_config, backbone, img_size
        )
        return model, None, False, True

    if pseudo_aux_config is not None and model_name in pseudo_aux_config:
        model, aux_key, is_bce = load_pseudo_aux_or_saliency_model(
            fold, device, model_name, pseudo_aux_config, backbone, img_size
        )
        return model, aux_key, is_bce, False


    if model_name == "crossattn":
        if tab_input_dim is None:
            raise ValueError("tab_input_dim must be provided for crossattn model")
        model = load_crossattn_model(
            fold, device, tab_input_dim, fusion_model_config, backbone, img_size
        )
        return model, None, False, True

    raise ValueError(f"Unknown model_name: {model_name}")



# Target layer helpers


def get_target_layer_baseline(model):
    if hasattr(model, "model"):
        return model.model.layers[-1].blocks[-1].norm1
    elif hasattr(model, "backbone"):
        return model.backbone.layers[-1].blocks[-1].norm1
    raise AttributeError("Could not find Swin backbone")


def get_target_layer_aux(model):
    return model.backbone.layers[-1].blocks[-1].norm1


def get_target_layer_concat(model):
    return model.img_model.layers[-1].blocks[-1].norm1


def get_target_layer_crossattn(model):
    return model.backbone.layers[-1].blocks[-1].norm1


def get_target_layer(model, model_name, aux_model_config, pseudo_aux_config=None):
    if model_name == "main":
        return get_target_layer_baseline(model)
    if model_name in aux_model_config:
        return get_target_layer_aux(model)
    if pseudo_aux_config is not None and model_name in pseudo_aux_config:
        return get_target_layer_aux(model)
    if model_name == "concat":
        return get_target_layer_concat(model)
    if model_name == "crossattn":
        return get_target_layer_crossattn(model)
    raise ValueError(f"Unknown model_name: {model_name}")



# Preprocess / fold / tabular helpers


def preprocess(img_path, img_size, aug_type="strong"):
    # reuse the shared transform from src.data
    val_tf = build_transforms(img_size, aug_type, train=False)
    img = Image.open(img_path).convert("RGB")
    return val_tf(img).unsqueeze(0), img


def get_fold_val(df, fold=1, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _, val_idx = list(kf.split(df))[fold - 1]
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return val_df


# def select_sample_ids(val_df, n_high=2, n_low=2):
#     high_ids = val_df.nlargest(n_high, "Pawpularity")["Id"].tolist()
#     low_ids = val_df.nsmallest(n_low, "Pawpularity")["Id"].tolist()
#     return high_ids + low_ids
def select_sample_ids(
    val_df,
    n_high=2,
    n_low=2,
    n_mid=0,
    mode="pawpularity",
    oof_df=None,
):
    """
    Select image IDs from the validation fold.

    mode="pawpularity":
        n_high  -> highest Pawpularity images
        n_low   -> lowest  Pawpularity images
        n_mid   -> closest to median Pawpularity images

    mode="error":
        Requires oof_df (loaded from oof_detail.csv).
        n_high  -> highest abs_err images (hardest samples)
        n_low   -> lowest  abs_err images (easiest samples)
        n_mid   is ignored in this mode.

    oof_detail.csv columns used: Id, ytrue, oof_pred, abs_err
    """
    if mode == "pawpularity":
        ids = []

        if n_high > 0:
            ids += val_df.nlargest(n_high, "Pawpularity")["Id"].tolist()

        if n_low > 0:
            ids += val_df.nsmallest(n_low, "Pawpularity")["Id"].tolist()

        if n_mid > 0:
            # pick images whose Pawpularity is closest to the median
            median_val = val_df["Pawpularity"].median()
            mid_df = (
                val_df
                .assign(_dist=(val_df["Pawpularity"] - median_val).abs())
                .nsmallest(n_mid, "_dist")
            )
            ids += mid_df["Id"].tolist()

        return ids

    elif mode == "error":
        if oof_df is None:
            raise ValueError(
                "oof_df must be provided when mode='error'. "
                "Load oof_detail.csv and pass it as oof_df."
            )

        # restrict to only the held-out IDs of this fold
        val_ids = set(val_df["Id"].tolist())
        oof_val = oof_df[oof_df["Id"].isin(val_ids)].copy()

        # compute abs_err if not already present in the csv
        if "abs_err" not in oof_val.columns:
            oof_val["abs_err"] = (oof_val["ytrue"] - oof_val["oof_pred"]).abs()

        oof_val = oof_val.sort_values("abs_err").reset_index(drop=True)

        ids = []
        if n_low > 0:
            # lowest error = model was most confident and correct
            ids += oof_val.head(n_low)["Id"].tolist()
        if n_high > 0:
            # highest error = model struggled most
            ids += oof_val.tail(n_high)["Id"].tolist()

        return ids

    else:
        raise ValueError(f"Unknown mode='{mode}'. Use 'pawpularity' or 'error'.")


def make_tab_tensor(row, tab_cols):
    vals = row[tab_cols].values.astype(np.float32)
    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)



# Prediction helpers


def predict_scores(
    model,
    img_tensor,
    device,
    aux_key=None,
    aux_is_bce=False,
    tab_tensor=None,
    needs_tabular=False,
):
    model.eval()
    with torch.no_grad():
        if needs_tabular:
            out = model(img_tensor.to(device), tab_tensor.to(device))
        else:
            out = model(img_tensor.to(device))

        if isinstance(out, dict):
            main_pred = out["main"]
            main_val = float(main_pred.squeeze().cpu().item())

            aux_val = None
            if aux_key is not None and aux_key in out:
                aux_pred = out[aux_key]
                if aux_is_bce:
                    aux_pred = torch.sigmoid(aux_pred)
                aux_val = float(aux_pred.squeeze().cpu().item())

            return main_val, aux_val

        return float(out.squeeze().cpu().item()), None


def fmt_or_na(x, fmt="{:.3f}"):
    if x is None:
        return "N/A"
    try:
        return fmt.format(x)
    except Exception:
        return "N/A"



# Build / remove CAM helpers


def build_cam_objects(models, device, aux_model_config, grid_size=(12, 12),pseudo_aux_config=None):
    cams = {}
    for name, model in models.items():
        target_layer = get_target_layer(model, name, aux_model_config, pseudo_aux_config)
        if name in ["concat", "crossattn"]:
            cams[name] = GradCAMSwinFusion(model, target_layer, device, grid_size=grid_size)
        else:
            cams[name] = GradCAMSwin(model, target_layer, device, grid_size=grid_size)
    return cams


def remove_cam_hooks(cams):
    for cam in cams.values():
        cam.remove_hooks()



# Main plotting function


def plot_gradcam_simple(
    df,
    img_folder,
    fold,
    model_names,
    save_path,
    device,
    backbone,
    img_size,
    baseline_dir,
    aux_model_config,
    fusion_model_config,
    display_names,
    n_high=2,
    n_low=2,
    n_mid=0,                      #  number of mid-pawpularity images
    selection_mode="pawpularity",  #  "pawpularity" or "error"
    oof_df=None, 
    tab_cols=None,
    grid_size=(12, 12),
    aug_type="strong",
    pseudo_aux_config=None,
):
    """
    For each selected image, show:
      Original | CAM for model 1 | CAM for model 2 | ...

    Grad-CAM is computed w.r.t. main Pawpularity head.
    """
    val_df = get_fold_val(df, fold=fold)
    # sample_ids = select_sample_ids(val_df, n_high=n_high, n_low=n_low)
    sample_ids = select_sample_ids(
        val_df,
        n_high=n_high,
        n_low=n_low,
        n_mid=n_mid,
        mode=selection_mode,
        oof_df=oof_df,
    )

    tab_input_dim = len(tab_cols) if tab_cols is not None else None

    models = {}
    aux_keys = {}
    aux_is_bce = {}
    needs_tabular = {}

    for name in model_names:
        model, aux_key, is_bce, need_tab = load_named_model(
            fold=fold,
            device=device,
            model_name=name,
            backbone=backbone,
            img_size=img_size,
            baseline_dir=baseline_dir,
            aux_model_config=aux_model_config,
            fusion_model_config=fusion_model_config,
            tab_input_dim=tab_input_dim,
            pseudo_aux_config=pseudo_aux_config
        )
        models[name] = model
        aux_keys[name] = aux_key
        aux_is_bce[name] = is_bce
        needs_tabular[name] = need_tab

    cams = build_cam_objects(
        models=models,
        device=device,
        aux_model_config=aux_model_config,
        grid_size=grid_size,
        pseudo_aux_config=pseudo_aux_config
    )

    n_cols = 1 + len(model_names)
    fig, axes = plt.subplots(len(sample_ids), n_cols, figsize=(4 * n_cols, 4 * len(sample_ids)))

    if len(sample_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_id in enumerate(sample_ids):
        row = df[df["Id"] == img_id].iloc[0]
        path = os.path.join(img_folder, img_id + ".jpg")

        img_tensor, img_pil = preprocess(path, img_size=img_size, aug_type=aug_type)
        true_paw = float(row["Pawpularity"])

        tab_tensor = None
        if tab_cols is not None:
            tab_tensor = make_tab_tensor(row, tab_cols)

        # original image
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"Original\nTrue Paw={true_paw:.0f}", fontsize=8)
        axes[i, 0].axis("off")

        # grad-cam per model
        for j, name in enumerate(model_names, start=1):
            model = models[name]
            cam_obj = cams[name]

            main_paw, aux_val = predict_scores(
                model=model,
                img_tensor=img_tensor,
                device=device,
                aux_key=aux_keys[name],
                aux_is_bce=aux_is_bce[name],
                tab_tensor=tab_tensor,
                needs_tabular=needs_tabular[name],
            )

            if needs_tabular[name]:
                cam_map = cam_obj(
                    img_tensor.to(device),
                    tab_tensor.to(device),
                    target_head="main",
                )
            else:
                cam_map = cam_obj(
                    img_tensor.to(device),
                    target_head="main",
                )

            overlay, _ = cam_to_heatmap(cam_map, img_pil)

            axes[i, j].imshow(overlay)

            if aux_keys[name] is None:
                title = f"{display_names[name]}\nPred Paw={main_paw:.1f}"
            else:
                aux_fmt = "{:.3f}" if aux_is_bce[name] else "{:.1f}"
                title = (
                    f"{display_names[name]}\n"
                    f"Pred Paw={main_paw:.1f} | Aux={fmt_or_na(aux_val, aux_fmt)}"
                )

            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    remove_cam_hooks(cams)

from scipy.stats import pearsonr, spearmanr  # add near top of file
def run_gradcam_quantitative_analysis(
df,
img_folder,
folds,
model_names,
device,
backbone,
img_size,
baseline_dir,
aux_model_config,
fusion_model_config,
tab_cols=None,
grid_size=(12, 12),
aug_type="strong",
pseudo_aux_config=None,
yolo_conf_threshold=0.3,
):
    """
    Compute quantitative GradCAM stats for each model across folds.

    Returns:
        img_level_df  : row per (fold, Id, model)
        summary_df    : mean overlaps per model
        corr_df       : correlation between overlap and abs error per model
    """
    img_rows = []

    for fold in folds:
        val_df = get_fold_val(df, fold=fold)
        # need YOLO bbox + conf to compute overlaps
        if "yolo_conf" not in val_df.columns:
            raise ValueError("val_df must contain 'yolo_conf' and bbox columns.")

        val_detected = val_df[val_df["yolo_conf"] > yolo_conf_threshold].reset_index(drop=True)
        if len(val_detected) == 0:
            continue

        tab_input_dim = len(tab_cols) if tab_cols is not None else None

        # load models for this fold
        models = {}
        aux_keys = {}
        aux_is_bce = {}
        needs_tabular = {}

        for name in model_names:
            model, aux_key, is_bce, need_tab = load_named_model(
                fold=fold,
                device=device,
                model_name=name,
                backbone=backbone,
                img_size=img_size,
                baseline_dir=baseline_dir,
                aux_model_config=aux_model_config,
                fusion_model_config=fusion_model_config,
                tab_input_dim=tab_input_dim,
                pseudo_aux_config=pseudo_aux_config,
            )
            models[name] = model
            aux_keys[name] = aux_key
            aux_is_bce[name] = is_bce
            needs_tabular[name] = need_tab

        cams = build_cam_objects(
            models=models,
            device=device,
            aux_model_config=aux_model_config,
            grid_size=grid_size,
            pseudo_aux_config=pseudo_aux_config,
        )

        for _, row in val_detected.iterrows():
            img_id = row["Id"]
            path = os.path.join(img_folder, img_id + ".jpg")

            img_tensor, img_pil = preprocess(path, img_size=img_size, aug_type=aug_type)
            W, H = img_pil.size

            bbox = (
                row["bbox_x1"],
                row["bbox_y1"],
                row["bbox_x2"],
                row["bbox_y2"],
            )
            face_bbox = make_face_proxy_bbox(bbox, W, H)

            true_paw = float(row["Pawpularity"])

            tab_tensor = None
            if tab_cols is not None:
                tab_tensor = make_tab_tensor(row, tab_cols)

            for name in model_names:
                model = models[name]
                cam_obj = cams[name]

                main_paw, _ = predict_scores(
                    model=model,
                    img_tensor=img_tensor,
                    device=device,
                    aux_key=aux_keys[name],
                    aux_is_bce=aux_is_bce[name],
                    tab_tensor=tab_tensor,
                    needs_tabular=needs_tabular[name],
                )

                if needs_tabular[name]:
                    cam_map = cam_obj(
                        img_tensor.to(device),
                        tab_tensor.to(device),
                        target_head="main",
                    )
                else:
                    cam_map = cam_obj(
                        img_tensor.to(device),
                        target_head="main",
                    )

                _, cam_rsz = cam_to_heatmap(cam_map, img_pil)

                pet_overlap = compute_cam_mask_overlap(cam_rsz, bbox, W, H)
                face_overlap = compute_cam_mask_overlap(cam_rsz, face_bbox, W, H)

                img_rows.append({
                    "fold": fold,
                    "Id": img_id,
                    "model_name": name,
                    "true_paw": true_paw,
                    "pred_paw": main_paw,
                    "abs_error": abs(true_paw - main_paw),
                    "cam_pet_overlap": pet_overlap,
                    "cam_face_overlap": face_overlap,
                    "yolo_conf": float(row["yolo_conf"]),
                })

        remove_cam_hooks(cams)

    import pandas as pd
    img_df = pd.DataFrame(img_rows)
    if img_df.empty:
        return img_df, img_df, img_df

    # summary per model
    summary_df = (
        img_df
        .groupby("model_name", as_index=False)[["cam_pet_overlap", "cam_face_overlap"]]
        .mean()
        .rename(columns={
            "cam_pet_overlap": "mean_cam_pet_overlap",
            "cam_face_overlap": "mean_cam_face_overlap",
        })
    )

    # correlations per model
    corr_rows = []
    for name, g in img_df.groupby("model_name"):
        if g["cam_pet_overlap"].nunique() > 1 and g["abs_error"].nunique() > 1:
            pr, _ = pearsonr(g["cam_pet_overlap"], g["abs_error"])
            sr, _ = spearmanr(g["cam_pet_overlap"], g["abs_error"])
        else:
            pr, sr = float("nan"), float("nan")

        if g["cam_face_overlap"].nunique() > 1 and g["abs_error"].nunique() > 1:
            pr_face, _ = pearsonr(g["cam_face_overlap"], g["abs_error"])
            sr_face, _ = spearmanr(g["cam_face_overlap"], g["abs_error"])
        else:
            pr_face, sr_face = float("nan"), float("nan")

        corr_rows.append({
            "model_name": name,
            "pearson_cam_pet_vs_abs_err": pr,
            "spearman_cam_pet_vs_abs_err": sr,
            "pearson_cam_face_vs_abs_err": pr_face,
            "spearman_cam_face_vs_abs_err": sr_face,
        })

    corr_df = pd.DataFrame(corr_rows)
    return img_df, summary_df, corr_df