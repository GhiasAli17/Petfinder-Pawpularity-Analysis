import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models import VisionAuxNet
from src.data import build_transforms, ImageOnlyDataset
from src.gradcam import GradCAMSwin, cam_to_heatmap, compute_cam_mask_overlap



# OOF Auxiliary Predictions


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



# Label-Specific Thresholds & Uncertainty Bands
# (feeds pipeline outputs 3 — aux probabilities, and 7 — reliability table)


def sweep_thresholds(y_true, y_prob):
    """
    For one auxiliary label, sweep decision thresholds in [0.01, 0.99]
    and compute F1, precision, and recall at each threshold.

    Returns:
        thresholds: array of thresholds
        f1s:        F1 scores at each threshold
        precs:      precision scores at each threshold
        recs:       recall scores at each threshold
        best_idx:   index of the threshold with maximum F1
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        # Convert probabilities to 0/1 predictions at threshold t
        pred = (y_prob >= t).astype(int)
        # Compute metrics at this threshold
        f1s.append(f1_score(y_true, pred, zero_division=0))
        precs.append(precision_score(y_true, pred, zero_division=0))
        recs.append(recall_score(y_true, pred, zero_division=0))

    f1s = np.array(f1s)
    precs = np.array(precs)
    recs = np.array(recs)
    # Index of threshold giving the highest F1
    best_idx = int(f1s.argmax())
    return thresholds, f1s, precs, recs, best_idx


def uncertainty_band(thresholds, f1s, best_idx, pct=0.90):
    """
    Define an 'uncertain' threshold band around the F1-optimal point.

    Idea:
      - Let peak_F1 = F1 at best_idx.
      - Any threshold whose F1 >= pct * peak_F1 is considered
        part of a flat/near-optimal region.
      - We expand left and right from best_idx while F1 stays above
        this cutoff and return the min/max thresholds in that region.

    Returns:
        uncertain_low:  left boundary of the near-peak region
        uncertain_high: right boundary of the near-peak region
    """
    peak = f1s[best_idx]
    if peak <= 0:
        # If the model is useless, treat the whole [min, max] as uncertain
        return float(thresholds[0]), float(thresholds[-1])

    cutoff = pct * peak  # e.g. 90% of peak F1
    lo, hi = best_idx, best_idx
    # Move left while F1 remains above cutoff
    while lo > 0 and f1s[lo - 1] >= cutoff:
        lo -= 1
    # Move right while F1 remains above cutoff
    while hi < len(f1s) - 1 and f1s[hi + 1] >= cutoff:
        hi += 1

    return float(thresholds[lo]), float(thresholds[hi])



# Reliability & Display Policy
# (pipeline output 7 — reliability table)


def assign_reliability(peak_f1: float):
    """
    Decide reliability / display policy using only the model's F1.
    """
    if peak_f1 >= 0.75:
        return "High", "show"
    elif peak_f1 >= 0.60:
        return "Medium", "supplementary"
    else:
        return "Low", "exclude"



# Model Loading


def load_model(fold, cfg, out_dir, device):
    ckpt = os.path.join(out_dir, f"model_fold{fold}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    model = VisionAuxNet(
        backbone_name=cfg["backbone"],
        img_size=cfg["img_size"],
        aux_tasks=[],
        binary_aux_tasks=cfg["binary_aux_tasks"],
        head_type=cfg.get("head_type", "linear"),
        pretrained=False,
        use_saliency=False,
    ).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    return model



# Single Image Inference
# ( 1 — input image, 2 — score, 3 — aux probs, 5 — GradCAM)


def preprocess(image_path, cfg):
    tf = build_transforms(cfg["img_size"], cfg["aug"], train=False)
    img_pil = Image.open(image_path).convert("RGB")
    tensor = tf(img_pil).unsqueeze(0)
    return tensor, img_pil


def predict_pawpularity(img_tensor, allFolds_models, device, fold=0):
    # Run the main pawpularity head of a chosen fold on one preprocessed image
    # Returns a scalar pawpularity score clipped to [0, 100]
    x = img_tensor.to(device)
    with torch.no_grad():
        out = allFolds_models[fold](x)
        pred = out["main"].squeeze().clamp(0.0, 100.0)
    return float(pred.cpu().item())


def predict_aux(img_tensor, allFolds_models, cfg, aux_tasks, device):
    # Run the auxiliary binary heads (Eyes/Face/Occlusion/Blur) on one image
    # Returns a dict {task: probability in [0,1]} in the original label meaning,
    # applying flipping for tasks listed in binary_aux_flip_targets (only for weighted BCE loss model, not used for standard BCE loss model)
    probs = {}
    flip_targets = set(cfg.get("binary_aux_flip_targets", []))
    x = img_tensor.to(device)

    with torch.no_grad():
        out = allFolds_models[0](x)
        for task in aux_tasks:
            raw = float(torch.sigmoid(out[task]).squeeze().cpu().item())
            probs[task] = (1.0 - raw) if task in flip_targets else raw
    return probs  # e.g. {"Eyes": 0.85, "Face": 0.92, "Occlusion": 0.10, "Blur": 0.05}


def get_gradcam(img_tensor, img_pil, allFolds_models, device, swin_grid):
    # Compute Grad-CAM for the main pawpularity head on a Swin backbone.
    # Uses the last block's norm layer as target, then:
    #   - cam: raw Grad-CAM map (model grid resolution)
    #   - cam_rsz: CAM resized to image size
    #   - overlay: CAM heatmap overlaid on the original image
    model = allFolds_models[0]
    target_layer = model.backbone.layers[-1].blocks[-1].norm1
    cam_obj = GradCAMSwin(model, target_layer, device, grid_size=swin_grid)
    cam = cam_obj(img_tensor.to(device), target_head="main")
    overlay, cam_rsz = cam_to_heatmap(cam, img_pil, alpha=0.5)
    cam_obj.remove_hooks()
    return cam, cam_rsz, overlay


def detect_pet_bbox(image_path):
    # Use YOLOv8n to detect the main pet (cat=15, dog=16 in COCO).
    # Returns the best bounding box (x1, y1, x2, y2) in image coordinates,
    # or None if detection fails or no cat/dog is found.
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        results = yolo(image_path, verbose=False, classes=[15, 16], conf=0.01)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            return float(x1), float(y1), float(x2), float(y2)
    except Exception as e:
        print("YOLO skipped:", e)
    return None


def get_cam_overlap(cam_rsz, bbox, img_pil):
    # Compute how much the Grad-CAM heatmap overlaps with the detected pet box.
    # Returns a scalar overlap score in [0,1] (or None if no bbox).
    if bbox is None:
        return None
    W, H = img_pil.size
    return compute_cam_mask_overlap(cam_rsz, bbox, W, H)



# Output Generation
# - confidence-aware feedback,
#  — separate user-facing and evaluation outputs)


def generate_feedback_user(paw_score, aux_probs, label_policy, aux_tasks, cam_pet_overlap=None):
    #  the *user-facing* explanation:
    #   - neutral predicted Pawpularity score
    #   - confidence-aware semantic feedback from aux heads
    #   - a short note about limitations
    lines = [f"Predicted Pawpularity score: {paw_score:.1f}"]

    semantics = []
    for task in aux_tasks:
        prob = aux_probs.get(task)
        policy = label_policy.get(task)
        if prob is None or policy is None:
            continue
        if policy["display"] == "exclude":
            continue

        thr = policy["threshold"]
        lo = policy["uncertain_low"]
        hi = policy["uncertain_high"]
        # the region where the model is uncertain near the best F1, with threshold low and high
        uncertain = prob >= lo and prob <= hi
        prefix = "[Supplementary] " if policy["display"] == "supplementary" else ""

        if task == "Face":
            if uncertain:
                msg = "The model is uncertain whether the pet's face is clearly visible."
            elif prob >= thr:
                msg = f"The pet's face appears to be clearly visible (confidence: {prob:.0%})."
            else:
                msg = f"The pet's face may not be clearly visible (confidence: {1-prob:.0%})."

        elif task == "Eyes":
            if uncertain:
                msg = "The model is uncertain whether the pet's eyes are clearly visible."
            elif prob >= thr:
                msg = f"The pet's eyes appear to be clearly visible (confidence: {prob:.0%})."
            else:
                msg = f"The pet's eyes may not be clearly visible (confidence: {1-prob:.0%})."

        elif task == "Occlusion":
            if uncertain:
                msg = f"{prefix}Occlusion status is uncertain (probability: {prob:.2f})."
            elif prob >= thr:
                msg = f"{prefix}The pet may be partially occluded (probability: {prob:.2f})."
            else:
                msg = f"{prefix}No strong occlusion signal detected (probability: {prob:.2f})."

        elif task == "Blur":
            if uncertain:
                msg = f"{prefix}Image sharpness is uncertain (probability: {prob:.2f})."
            elif prob >= thr:
                msg = f"{prefix}The image may appear blurry (probability: {prob:.2f})."
            else:
                msg = f"{prefix}The image appears sharp based on model cues (probability: {1-prob:.2f})."

        semantics.append(msg)

    if semantics:
        lines.append("Semantic observations:")
        for s in semantics:
            lines.append(f"  {s}")
    # overlap values are not included in user-facing feedback
    # if cam_pet_overlap is not None:
    #     lines.append(f"CAM–pet overlap: {cam_pet_overlap:.2f}")

    lines.append(
        "Note: These statements describe model-detected visual cues only and "
        "should be treated as reference information."
    )
    return lines


def generate_output_evaluation(paw_score, aux_probs, label_policy, aux_tasks,
                                true_paw=None, true_aux=None, cam_pet_overlap=None):
    #  the *evaluation* view (not for user-facing feedback):
    #   - true vs predicted pawpularity and absolute error
    #   - per-label true aux, predicted prob, binarized pred, and ✓/✗
    #   - CAM–pet overlap value for analysis
    lines = []
    if true_paw is not None:
        lines.append(f"True Pawpularity     : {true_paw:.1f}")
        lines.append(f"Predicted Pawpularity: {paw_score:.1f}")
        lines.append(f"Absolute Error       : {abs(true_paw - paw_score):.2f}")
    else:
        lines.append(f"Predicted Pawpularity: {paw_score:.1f}")

    lines.append("Auxiliary label comparison:")
    true_aux = true_aux or {}
    for task in aux_tasks:
        prob = aux_probs.get(task, np.nan)
        thr = label_policy[task]["threshold"]
        pred_bin = int(prob >= thr) if not np.isnan(prob) else None
        
        t = true_aux.get(task, None)
        # if true and predicted are both same then  ✓ else ✗
        if t is not None and pred_bin is not None:
            ok = "✓" if int(t) == pred_bin else "✗"
            lines.append(f"  {task:<12} true={int(t)}  prob={prob:.3f}  pred={pred_bin}  {ok}")
        else:
            lines.append(f"  {task:<12} true=N/A  prob={prob:.3f}  pred={pred_bin}")

    if cam_pet_overlap is not None:
        lines.append(f"CAM–pet overlap      : {cam_pet_overlap:.3f}")
    return lines


def visualise(img_pil, cam_overlay, user_lines, eval_lines, img_id="", save_path=None,label=None):
    # Create a 2x3 grid figure:
    #   Row 1: input image | Grad-CAM overlay | user-facing feedback
    #   Row 2: full-width evaluation / research output
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.5], height_ratios=[1.2, 0.6], hspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_pil)
    ax1.set_title(f"Input Image\n{img_id}", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cam_overlay)
    ax2.set_title("Model Attention (GradCAM)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("User Feedback", fontsize=11, fontweight="bold", color="#2c7a2c")
    ax3.text(
        0.02, 0.97, "\n\n".join(user_lines),
        transform=ax3.transAxes, fontsize=9, va="top", wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4e8", alpha=0.9)
    )

    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
  
    ax4.set_title("Evaluation Output [Analysis Only, not for user]" + label, fontsize=10, fontweight="bold", color="#888888")
    ax4.text(
        0.01, 0.90, "\n".join(eval_lines),
        transform=ax4.transAxes, fontsize=8.5, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8)
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)



# Batch Runner
# over a list of images


def run_batch(image_ids, img_folder, cfg, allFolds_models, label_policy, aux_tasks,
              swin_grid, device, true_paw_dict=None, true_aux_dict=None,
              save_dir=None, use_yolo=True, labels=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    true_paw_dict = true_paw_dict or {}
    true_aux_dict = true_aux_dict or {}
    labels = labels or {}  

    rows = []
    for img_id in image_ids:
        path = os.path.join(img_folder, img_id + ".jpg")
        if not os.path.exists(path):
            print("Skipping missing:", img_id)
            continue

        label = labels.get(img_id, "")
        print("*"*80)
        print(f"\nImgId: {img_id[0:5]}")
        

        img_tensor, img_pil = preprocess(path, cfg)                                      #  — input image
        paw_score = predict_pawpularity(img_tensor, allFolds_models, device, fold=0)     #  — predicted score
        aux_probs = predict_aux(img_tensor, allFolds_models, cfg, aux_tasks, device)     #  — aux probs
        cam, cam_rsz, cam_ov = get_gradcam(img_tensor, img_pil, allFolds_models,
                                           device, swin_grid)                            #  — GradCAM

        overlap = None
        # Optional YOLO pet box + CAM–pet overlap
        if use_yolo:
            bbox = detect_pet_bbox(path)
            if bbox is not None:
                overlap = get_cam_overlap(cam_rsz, bbox, img_pil)

        #  Look up ground-truth labels for evaluation view
        true_paw = true_paw_dict.get(img_id, None)
        true_aux = true_aux_dict.get(img_id, {})

        user_fb  = generate_feedback_user(paw_score, aux_probs,
                                          label_policy, aux_tasks, overlap)              # — confidence-aware feedback
        eval_out = generate_output_evaluation(paw_score, aux_probs,
                                              label_policy, aux_tasks,
                                              true_paw, true_aux, overlap)              #  — evaluation output

        if save_dir:
            fig_path = os.path.join(save_dir, f"{img_id}_feedback.png")
            visualise(img_pil, cam_ov, user_fb, eval_out,
                      img_id=img_id, save_path=fig_path, label=label)                                #  rendered together

        row = {
            "Id": img_id,
            "true_paw": true_paw,
            "pred_paw": round(paw_score, 2),
            "cam_pet_overlap": None if overlap is None else round(float(overlap), 3),
        }
        for task in aux_tasks:
            prob = aux_probs.get(task, np.nan)
            thr = label_policy[task]["threshold"]
            row[f"{task.lower()}_true"] = true_aux.get(task, None)
            row[f"{task.lower()}_prob"] = None if np.isnan(prob) else round(prob, 3)
            row[f"{task.lower()}_pred"] = None if np.isnan(prob) else int(prob >= thr)

        rows.append(row)

    return pd.DataFrame(rows)