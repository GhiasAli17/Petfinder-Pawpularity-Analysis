# src/saliency.py
"""
GradCAM for Swin Transformer (timm).

"""

import numpy as np
import torch
import cv2
from PIL import Image


#
# Shape helpers
#

def _factorize_grid(n: int):
    """
    Return (H, W) closest to square with H * W == n.
    Works for any n (square or not).
        144 -> (12, 12)   108 -> (9, 12)   12 -> (3, 4)
    """

    # start from sqrt(n) to get the most square-like split
    # search downward until we find h that divides n evenly
    # e.g. n=144: h=12 -> 144%12==0 -> return (12,12)
    # e.g. n=108: h=10 fails, h=9 -> 108%9==0 -> return (9,12)
    h = int(np.sqrt(n))
    while h >= 1:
        if n % h == 0:
            return h, n // h
        h -= 1
    return 1, n #single row (never happens for normal Swin), just a fallback


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

    # fallback: derive from patch_embed and count PatchMerging downsamples
    # patch_embed gives initial H0, W0 (e.g. 96x96 for 384px / patch_size=4)
    # each PatchMerging stage halves H and W -> divide by 2 per stage
    if hasattr(backbone, "patch_embed"):
        pe = backbone.patch_embed
        if hasattr(pe, "grid_size"):
            H0, W0 = pe.grid_size
        elif hasattr(pe, "img_size") and hasattr(pe, "patch_size"):
            img = pe.img_size
            ps  = pe.patch_size
            H0  = (img[0] if isinstance(img, (tuple, list)) else img) // (
                   ps[0]  if isinstance(ps,  (tuple, list)) else ps)
            W0  = (img[1] if isinstance(img, (tuple, list)) else img) // (
                   ps[1]  if isinstance(ps,  (tuple, list)) else ps)
        else:
            return None
        # count PatchMerging layers (each halves spatial resolution)
        if hasattr(backbone, "layers"):
            for stage in backbone.layers:
                if getattr(stage, "downsample", None) is not None:
                    H0 //= 2
                    W0 //= 2
        return H0, W0

    return None


#
# GradCAM
#

class GradCAMSwin:
    """
    GradCAM for Swin Transformer (timm), 
    Parameters
    ----------
    model        : nn.Module
    target_layer : nn.Module  (the layer to hook)
    device       : torch.device
    grid_size    : (H, W) override, e.g. (12, 12) for swin_large_384.
                   Used only when the auto-detection disagrees with the
                   actual token count; leave None to rely on auto-detect.
    """

    def __init__(self, model, target_layer, device, grid_size=None):
        self.model        = model
        self.target_layer = target_layer
        self.device       = device
        self._grid_size   = grid_size

        self._gradients   = None
        self._activations = None
        self._hooks: list = []
        self._register_hooks()

    #  hooks

    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer.
        Forward hook captures activations during forward pass.
        Backward hook captures gradients during backward pass.
        Both are needed for GradCAM computation.
        """
        def fwd_hook(module, inp, output):
            # store activations from forward pass
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            # store gradients from backward pass
            # grad_out[0] is the gradient w.r.t. the layer output
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

    #  CAM computation

    def __call__(self, img_tensor, target_head: str = "main") -> np.ndarray:
        """
        Parameters
        ----------
        img_tensor  : (1, 3, H, W) tensor 
        target_head : dict key to use if the model returns a dict

        Returns
        -------
        cam : (H_patch, W_patch) float32 ndarray in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()
        self._gradients = None
        self._activations = None

        img_tensor = img_tensor.to(self.device)

        # forward
        out   = self.model(img_tensor)
        score = out[target_head] if isinstance(out, dict) else out
        score.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "GradCAM: hooks did not capture gradients/activations. "
                "Ensure target_layer lies on the backward path."
            )

        grads = self._gradients    
        acts  = self._activations  # same shape as grads

     

        ndim = acts.dim()

        if ndim == 4:
            # (B, H, W, C) — spatial grid is already 2-D; pool over H and W
            cam = self._cam_from_bhwc(grads, acts)   # returns (H, W) numpy

        elif ndim == 3:
            # (B, N, C) — flat token sequence; reshape to (H, W) after pooling
            cam = self._cam_from_bnc(grads, acts)    # returns (H, W) numpy

        elif ndim == 2:
            # (B, C) — already globally pooled; trivial 1×1 map
            weights = grads.mean(dim=0, keepdim=True)
            flat    = torch.relu((weights * acts).sum(dim=-1))
            cam     = flat.cpu().numpy().reshape(1, 1)

        else:
            raise RuntimeError(
                f"GradCAM: unexpected activation shape {tuple(acts.shape)}"
            )

        #  normalise to [0, 1] 
        c_min, c_max = float(cam.min()), float(cam.max())
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam.astype(np.float32)   # (H_patch, W_patch)

    #   helpers

    def _cam_from_bhwc(self, grads: torch.Tensor, acts: torch.Tensor) -> np.ndarray:
        """
        (B, H, W, C) 
        Pool gradients over both spatial dims (H and W), then weight the
        activation map.  Result is (H, W); no reshape needed.
        """
        # 1. mean over spatial dims 1 and 2  ->  (B, 1, 1, C)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        # 2. weighted channel sum  ->  (B, H, W)
        cam = (weights * acts).sum(dim=-1)
        # step 3: ReLU (only regions that positively influenced the score)
        cam = torch.relu(cam).squeeze(0)   # (H, W)
        return cam.cpu().numpy()

    def _cam_from_bnc(self, grads: torch.Tensor, acts: torch.Tensor) -> np.ndarray:
        """
        (B, N, C) format — used by older timm Swin.
        Pool gradients over the token dim, weight activations, then reshape
        the flat result to (H, W).
        """
        weights = grads.mean(dim=1, keepdim=True)   # (B, 1, C)
        cam     = (weights * acts).sum(dim=-1)        # (B, N)
        cam     = torch.relu(cam).squeeze(0)          # (N,)
        flat    = cam.cpu().numpy()                   # (N,)

        n       = flat.shape[0]
        h, w    = self._resolve_grid(n)

        # safety net: if grid still doesn't tile n, fall back to square crop
        if h * w != n:
            sq   = int(np.sqrt(n))
            flat = flat[: sq * sq]
            h = w = sq

        return flat.reshape(h, w)

    def _resolve_grid(self, n_tokens: int):
        """Return (H, W) with H*W == n_tokens."""
        # 1. user-supplied override
        if self._grid_size is not None:
            H, W = self._grid_size
            if H * W == n_tokens:
                return H, W

        # 2. derive from model attributes (timm layer metadata)
        backbone = getattr(self.model, "backbone", self.model)
        hw = _get_swin_output_grid(backbone)
        if hw is not None and hw[0] * hw[1] == n_tokens:
            return hw

        # 3. find the factorisation closest to square
        return _factorize_grid(n_tokens)


#
# Visualisation helpers
#

def cam_to_heatmap(cam: np.ndarray, img_pil: Image.Image, alpha: float = 0.5):
    """
    Overlay a (H_patch, W_patch) CAM on the original PIL image.

    Returns
    -------
    blended     : PIL Image
    cam_resized : (H_img, W_img) uint8 — pass directly to
                  compute_cam_mask_overlap
    """
    W, H    = img_pil.size

   # scale CAM to uint8 for colormap application
    cam_u8  = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    
    # resize from patch grid (12×12) to full image resolution using bilinear interp

    cam_rsz = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(cam_rsz, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np  = np.array(img_pil.convert("RGB"))
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
    IoU between the high-activation CAM region and the YOLO bounding box.

    IoU = intersection / union
    - 0.0 means model attention and pet bbox do not overlap at all
    - 1.0 means model attention perfectly matches pet bbox

    Parameters
    ----------
    cam_resized   : (H_img, W_img) uint8 [0, 255], already at image resolution
    bbox          : (x1, y1, x2, y2) pixel coords
    img_w, img_h  : image width and height
    cam_threshold : fraction of max activation used as binary threshold
                    0.5 means top 50% of activation is treated as "attending here"

    Returns
    -------
    iou : float in [0, 1]
    """

    # 1: threshold CAM into binary mask
    # top 50% of activation values -> 1 (model is looking here)
    # bottom 50%                   -> 0 (model is not looking here)
    threshold = float(cam_resized.max()) * cam_threshold
    cam_bin   = (cam_resized.astype(np.float32) >= threshold)

    #  2: convert bbox to binary mask
    # clip to image bounds, then fill 1.0 inside the pet bbox
    x1 = int(max(0,     float(bbox[0])))
    y1 = int(max(0,     float(bbox[1])))
    x2 = int(min(img_w, float(bbox[2])))
    y2 = int(min(img_h, float(bbox[3])))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_bin              = np.zeros((img_h, img_w), dtype=np.float32)
    bbox_bin[y1:y2, x1:x2] = 1.0

    # 3: compute IoU
    # intersection = pixels where BOTH cam and bbox are 1
    # union        = pixels where EITHER cam or bbox is 1
    intersection = float((cam_bin * bbox_bin).sum())
    union        = float(np.clip(cam_bin + bbox_bin, 0, 1).sum())
    return intersection / union if union > 0 else 0.0