# src/plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



def plot_oof_true_pred_lines(oof_df, title_prefix=""):
    """
    Line plot of true and predicted Pawpularity over sample index.
    oof_df must have columns: 'ytrue', 'oof_pred'.
    """
    ytrue = oof_df["ytrue"].values
    ypred = oof_df["oof_pred"].values
    idx = np.arange(len(ytrue))

    plt.figure(figsize=(10, 4))
    plt.plot(idx, ytrue, label="True", linewidth=1.0)
    plt.plot(idx, ypred, label="Predicted", linewidth=1.0, alpha=0.8)
    plt.xlabel("Sample index (sorted by Id or as in OOF)")
    plt.ylabel("Pawpularity")
    title = "True vs Predicted (line plot)"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()






def show_error_images_grid(oof_errors_df, img_folder, n=12, title_prefix=""):
    """
    Show top-n rows from a top-errors DataFrame as an image grid.
    oof_errors_df must have columns: 'Id', 'ytrue', 'oof_pred'.
    """
    subset = oof_errors_df.head(n)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, row in enumerate(subset.itertuples(), start=1):
        img_id = row.Id
        ytrue = row.ytrue
        ypred = row.oof_pred
        img_path = os.path.join(img_folder, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        plt.subplot(nrows, ncols, i)
        plt.imshow(img)
        plt.axis("off")
        title = f"{img_id}\ntrue={ytrue} pred={ypred:.1f}"
        if title_prefix:
            title = f"{title_prefix}\n{title}"
        plt.title(title)
    plt.tight_layout()
    plt.show()
