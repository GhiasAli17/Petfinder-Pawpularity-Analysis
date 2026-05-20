# src/gen_pseudo_labels.py
# Generates pseudo-labels for aux-task training:
#   - BRISQUE (image quality)
#   - visibility_ratio (pet bbox area / image area)

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from piq import brisque
from ultralytics import YOLO
import pandas as pd
#1. BRISQUE score: no-reference image quality metric (lower is better)
def compute_brisque(df, img_folder):
    """Compute BRISQUE score for each image."""

    tf = transforms.Compose([
        # transforms.Resize((384, 384)),
        transforms.ToTensor(),  # [0,1]
    ])

    scores = []
    print("  Computing BRISQUE scores...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(img_folder, row["Id"] + ".jpg")
        img  = Image.open(path).convert("RGB")
        x    = tf(img).unsqueeze(0)
        with torch.no_grad():
            score = brisque(x, data_range=1.0).item()
        scores.append(score)

    df["BRISQUE"] = scores
    print(f"  BRISQUE done — mean={df['BRISQUE'].mean():.2f}, "
          f"std={df['BRISQUE'].std():.2f}, "
          f"min={df['BRISQUE'].min():.2f}, "
          f"max={df['BRISQUE'].max():.2f}")
    return df

#2. YOLO visibility ratio: area of detected pet bbox / image area
def compute_yolo(df, img_folder):
    """
    Compute pet visibility ratio and bbox from YOLO.
    Adds: visibility_ratio, yolo_conf, bbox_x1, bbox_y1, bbox_x2, bbox_y2
    """

    yolo = YOLO("yolov8n.pt")  # downloads automatically on first run

    vis_ratios, confs = [], []
    bx1s, by1s, bx2s, by2s = [], [], [], []

    print("  Computing YOLO visibility labels...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(img_folder, row["Id"] + ".jpg")
        img  = Image.open(path).convert("RGB")
        W, H = img.size

        results = yolo(path, verbose=False, classes=[15, 16])  # 15=cat 16=dog

        best_conf = 0.0
        vis_ratio = 0.0
        bx1 = by1 = bx2 = by2 = 0.0

        if (results and
            results[0].boxes is not None and
            len(results[0].boxes) > 0):

            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    x1 = max(0.0, float(x1))
                    y1 = max(0.0, float(y1))
                    x2 = min(float(W), float(x2))
                    y2 = min(float(H), float(y2))
                    bbox_area = (x2 - x1) * (y2 - y1)
                    vis_ratio = float(bbox_area / (W * H)) if W * H > 0 else 0.0
                    bx1, by1, bx2, by2 = x1, y1, x2, y2

        vis_ratios.append(vis_ratio)
        confs.append(best_conf)
        bx1s.append(bx1)
        by1s.append(by1)
        bx2s.append(bx2)
        by2s.append(by2)

    df["visibility_ratio"] = vis_ratios
    df["yolo_conf"]        = confs
    df["bbox_x1"]          = bx1s
    df["bbox_y1"]          = by1s
    df["bbox_x2"]          = bx2s
    df["bbox_y2"]          = by2s

    detected = (df["yolo_conf"] > 0.3).sum()
    print(f"  YOLO done — pet detected (conf>0.3): {detected} / {len(df)} "
          f"({detected/len(df)*100:.1f}%)")
    print(f"  visibility_ratio — mean={df['visibility_ratio'].mean():.3f}, "
          f"std={df['visibility_ratio'].std():.3f}")
    return df


def generate_pseudo_labels(
    df,
    img_folder,
    out_csv,
    run_brisque=True,
    run_yolo=True,
    force_recompute=False,
):
    """
    Generate pseudo-labels and save to out_csv.
    Skips generation if out_csv already exists (unless force_recompute=True).

    Args:
        df          : original train DataFrame (must have 'Id' column)
        img_folder  : path to image folder
        out_csv     : path to save output CSV
        run_brisque : whether to compute BRISQUE
        run_yolo    : whether to compute YOLO visibility
        force_recompute : if True, recomputes even if out_csv exists

    Returns:
        df with pseudo-label columns added
    """

    if os.path.exists(out_csv) and not force_recompute:
        print(f"Pseudo-label CSV already exists, loading: {out_csv}")
        return pd.read_csv(out_csv)

    print(f"Generating pseudo-labels for {len(df)} images...")
    df = df.copy()

    if run_brisque:
        df = compute_brisque(df, img_folder)

    if run_yolo:
        df = compute_yolo(df, img_folder)

    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Columns: {df.columns.tolist()}")

    return df