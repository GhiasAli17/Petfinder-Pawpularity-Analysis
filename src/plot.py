# src/plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



def plot_oof_true_pred_lines(oof_df, title_prefix=""):
    """
    Line plot of true and predicted Pawpularity over sample index.
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






def show_images_grid(oof_errors_df, img_folder, n=12, title_prefix=""):
    """
    Show top-n rows from a top-errors DataFrame as an image grid.
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
        title = f"true={ytrue} pred={ypred:.1f}"
        if title_prefix:
            title = f"{title_prefix}\n{title}"
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_all_folds_history(out_dir, folds, title_prefix=""):
    plt.figure(figsize=(8, 5))
    for fold in folds:
        hist_path = os.path.join(out_dir, f"history_fold{fold}.csv")
        if not os.path.exists(hist_path):
            continue
        hist = pd.read_csv(hist_path)
        epochs = hist["epoch"].values
        val_rmse = hist["val_rmse"].values
        plt.plot(epochs, val_rmse, label=f"fold {fold}")

    plt.xlabel("Epoch")
    plt.ylabel("val RMSE")
    title = "Validation RMSE per fold"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_oof_by_bins(
    oof_path_a,
    oof_path_b,
    bins,
    label_a="ExpA",
    label_b="ExpB",
    title_prefix="",
):
    """
    Compare two experiments' OOF RMSE across target bins.

    oof_path_a, oof_path_b: CSVs with columns ['ytrue', 'oof_pred'].
    bins: list of bin edges for ytrue, e.g. [0,20,40,60,80,100].
    label_a, label_b: short labels for legend / column names.
    """

    # --- load and bin ---
    oof_a = pd.read_csv(oof_path_a)
    oof_b = pd.read_csv(oof_path_b)

    oof_a["bin"] = pd.cut(oof_a["ytrue"], bins)
    oof_b["bin"] = pd.cut(oof_b["ytrue"], bins)

    # --- per-bin RMSE ---
    rmse_a = (
        oof_a.groupby("bin", observed=False)[["ytrue", "oof_pred"]]
             .apply(lambda x: ((x["ytrue"] - x["oof_pred"])**2).mean()**0.5)
    )
    rmse_b = (
        oof_b.groupby("bin", observed=False)[["ytrue", "oof_pred"]]
             .apply(lambda x: ((x["ytrue"] - x["oof_pred"])**2).mean()**0.5)
    )
    
    bin_counts = (
        oof_a.groupby("bin", observed=False)["ytrue"]
             .size()
    )
    df = pd.DataFrame({
        "bin": rmse_a.index,
        "count": bin_counts.values, 
        f"{label_a}_rmse": rmse_a.values,
        f"{label_b}_rmse": rmse_b.values,
    })

   
    df["rmse_diff"] = df[f"{label_b}_rmse"] - df[f"{label_a}_rmse"]

    # human-readable winner
    def winner(row):
        a = row[f"{label_a}_rmse"]
        b = row[f"{label_b}_rmse"]
        if pd.isna(a) or pd.isna(b):
            return "no_data"
        if a < b:
            return f"{label_a}_better"
        elif a > b:
            return f"{label_b}_better"
        else:
            return "tie"

    df["winner"] = df.apply(winner, axis=1)

    # --- overall OOF RMSEs ---
    rmse_a_all = np.sqrt(((oof_a["ytrue"] - oof_a["oof_pred"])**2).mean())
    rmse_b_all = np.sqrt(((oof_b["ytrue"] - oof_b["oof_pred"])**2).mean())
    oof_gap = rmse_b_all - rmse_a_all   

    print(f"{label_a} OOF RMSE:", rmse_a_all)
    print(f"{label_b} OOF RMSE:", rmse_b_all)
    print(f"ΔRMSE ({label_b} - {label_a}):", oof_gap)

    # --- bar plot per bin ---
    plt.figure(figsize=(8, 4))
    x = np.arange(len(df))
    width = 0.35

    plt.bar(x - width/2, df[f"{label_a}_rmse"], width, label=label_a)
    plt.bar(x + width/2, df[f"{label_b}_rmse"], width, label=label_b)

    plt.xticks(x, df["bin"].astype(str), rotation=45)
    plt.xlabel("Pawpularity bin")
    plt.ylabel("RMSE")
    title = "RMSE per Pawpularity bin"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, rmse_a_all, rmse_b_all, oof_gap
