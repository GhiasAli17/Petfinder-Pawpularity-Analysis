# src/plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def plot_pseudo_label_analysis(df, pseudo_cols, target_col="Pawpularity", out_dir=None):
    """
    1. Distribution of each pseudo-label
    2. Correlation matrix heatmap
    3. Scatter of each pseudo-label vs target
    """

    n = len(pseudo_cols)
    cols_all = [target_col] + pseudo_cols

    # 1. Distributions
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 4))
    if n + 1 == 1:
        axes = [axes]
    for i, col in enumerate(cols_all):
        ax = axes[i]
        ax.hist(df[col], bins=30, edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="red", linestyle="--",
                   label=f"mean={df[col].mean():.1f}")
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Distributions", fontsize=12)
    plt.tight_layout()

    plt.show()

      # 2. Correlation matrix — all numeric columns
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[all_numeric_cols].dropna().corr()
    
    fig, ax = plt.subplots(figsize=(len(all_numeric_cols) * 0.8 + 2, 
                                    len(all_numeric_cols) * 0.8 + 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(all_numeric_cols)))
    ax.set_yticks(range(len(all_numeric_cols)))
    ax.set_xticklabels(all_numeric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_numeric_cols, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Correlation Matrix")
    plt.tight_layout()

    plt.show()

    


def plot_oof_true_pred_lines(oof_df,true_col="ytrue",pred_col="oof_pred", title_prefix=""):
    """
    Line plot of true and predicted Pawpularity over sample index.
    """
    ytrue = oof_df[true_col].values
    ypred = oof_df[pred_col].values
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


def plot_pred_distribution(df, pred_cols, y_col="ytrue", score_range=(0, 100),
                           title_prefix="", save_path=None):
    """
    Overlaid histograms of the true and predicted score distributions.
   
    """
    names = list(pred_cols)
    bins = np.linspace(score_range[0], score_range[1], 31)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4),
                             squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names):
        ax.hist(df[y_col].values, bins=bins, alpha=0.5, density=True,
                color="tab:gray", label="True")
        ax.hist(df[pred_cols[name]].values, bins=bins, alpha=0.6, density=True,
                color="tab:blue", label="Predicted")
        ax.set_xlabel("Pawpularity")
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()



def show_images_grid(oof_errors_df, img_folder, n=12, title_prefix="", pred_col="oof_pred"):
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
        ypred = getattr(row, pred_col)
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
import pandas as pd
import matplotlib.pyplot as plt


def plot_train_val_history(out_dir, folds, title_prefix=""):
    n_folds = len(folds)
    n_rows, n_cols = 2, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, fold in enumerate(folds):
        ax = axes[i]
        hist_path = os.path.join(out_dir, f"history_fold{fold}.csv")
        if not os.path.exists(hist_path):
            ax.set_visible(False)
            continue

        hist = pd.read_csv(hist_path)
        epochs = hist["epoch"].values
        train_rmse = hist["train_rmse"].values
        val_rmse = hist["val_rmse"].values

        ax.plot(epochs, train_rmse, label="Train RMSE", color="tab:blue")
        ax.plot(epochs, val_rmse, label="Val RMSE", color="tab:orange")

        # min train RMSE
        train_min_idx = np.argmin(train_rmse)
        train_min_epoch = epochs[train_min_idx]
        train_min_val = train_rmse[train_min_idx]
        ax.scatter(train_min_epoch, train_min_val, color="tab:blue", zorder=5)
        ax.annotate(
            f"{train_min_val:.2f}",
            xy=(train_min_epoch, train_min_val),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="tab:blue"
        )

        # min val RMSE
        val_min_idx = np.argmin(val_rmse)
        val_min_epoch = epochs[val_min_idx]
        val_min_val = val_rmse[val_min_idx]
        ax.scatter(val_min_epoch, val_min_val, color="tab:orange", zorder=5)
        ax.annotate(
            f"{val_min_val:.2f}",
            xy=(val_min_epoch, val_min_val),
            xytext=(5, -12), textcoords="offset points",
            fontsize=8, color="tab:orange"
        )

        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.legend(loc="upper right", fontsize=9)  # legend inside each subplot
        ax.grid(alpha=0.3)

    for j in range(n_folds, n_rows * n_cols):
        axes[j].set_visible(False)

    main_title = "Train vs Val RMSE"
    if title_prefix:
        main_title = f"{title_prefix} - {main_title}"
    fig.suptitle(main_title, fontsize=13)

    plt.tight_layout()
    plt.show()

def plot_compare_two_experiments(
    out_dir_a, out_dir_b,
    folds,
    label_a="Exp A",
    label_b="Exp B",
):
    """
    Compare two experiments side by side per fold.
    Layout: 5 rows x 2 cols
      - Left col:  Exp A
      - Right col: Exp B
    Each subplot shows Train RMSE and Val RMSE with min annotations.
    """
    n_folds = len(folds)
    n_rows, n_cols = n_folds, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_folds))

    for i, fold in enumerate(folds):
        for j, (out_dir, label) in enumerate([(out_dir_a, label_a), (out_dir_b, label_b)]):
            ax = axes[i][j]
            hist_path = os.path.join(out_dir, f"history_fold{fold}.csv")

            if not os.path.exists(hist_path):
                ax.set_visible(False)
                continue

            hist = pd.read_csv(hist_path)
            epochs     = hist["epoch"].values
            train_rmse = hist["train_rmse"].values
            val_rmse   = hist["val_rmse"].values

            ax.plot(epochs, train_rmse, label="Train RMSE", color="tab:blue")
            ax.plot(epochs, val_rmse,   label="Val RMSE",   color="tab:orange")

            # min train RMSE
            train_min_idx   = np.argmin(train_rmse)
            train_min_epoch = epochs[train_min_idx]
            train_min_val   = train_rmse[train_min_idx]
            ax.scatter(train_min_epoch, train_min_val, color="tab:blue", zorder=5)
            ax.annotate(
                f"{train_min_val:.2f}",
                xy=(train_min_epoch, train_min_val),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="tab:blue"
            )

            # min val RMSE
            val_min_idx   = np.argmin(val_rmse)
            val_min_epoch = epochs[val_min_idx]
            val_min_val   = val_rmse[val_min_idx]
            ax.scatter(val_min_epoch, val_min_val, color="tab:orange", zorder=5)
            ax.annotate(
                f"{val_min_val:.2f}",
                xy=(val_min_epoch, val_min_val),
                xytext=(5, -12), textcoords="offset points",
                fontsize=8, color="tab:orange"
            )

            ax.set_title(f"{label} - Fold {fold}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3)
    fig.text(
        0.5, 1.01,
        f"{label_a}  vs  {label_b}",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.55)

    plt.show()




def plot_train_val_per_fold(out_dir, folds, title_prefix=""):
    """
    For each fold, create one figure with:
      - train_loss (MSE) vs epoch
      - val_rmse vs epoch
    """
    for fold in folds:
        hist_path = os.path.join(out_dir, f"history_fold{fold}.csv")
        if not os.path.exists(hist_path):
            print(f"history for fold {fold} not found, skipping.")
            continue

        hist = pd.read_csv(hist_path)
        epochs = hist["epoch"].values
        train_loss = hist["train_rmse"].values
        val_rmse = hist["val_rmse"].values

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss, label="Train MSE", color="tab:blue")
        plt.plot(epochs, val_rmse, label="Val RMSE", color="tab:orange")

        plt.xlabel("Epoch")
        plt.ylabel("Loss / RMSE")
        title = f"Fold {fold} - Train vs Val"
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

def plot_foldwise_rmse(df, y_col, pred_cols, fold_col="fold", title="Fold-wise RMSE Comparison"):
    """
    Plot fold-wise RMSE for multiple prediction columns.
    """
    folds = sorted(df[fold_col].unique())
    results = {name: [] for name in pred_cols.keys()}

    # compute RMSE per fold for each experiment
    for fold in folds:
        sub = df[df[fold_col] == fold]
        y_true = sub[y_col].values
        for exp_name, col in pred_cols.items():
            rmse = np.sqrt(((y_true - sub[col].values) ** 2).mean())
            results[exp_name].append(rmse)

    # plot
    plt.figure(figsize=(8, 5))
    for exp_name, rmse_list in results.items():
        plt.plot(folds, rmse_list, marker="o", label=exp_name)

    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.xticks(folds)
    plt.tight_layout()
    plt.show()

    return results


def compare_oof_by_bins_multi(
    df,
    bins,
    y_col="ytrue",
    pred_cols=None,
    title_prefix="",
):
    """
    Compare multiple experiments' OOF RMSE across target bins.

    """

    if pred_cols is None:
        raise ValueError("You must provide pred_cols mapping {label: column_name}")

    # --- assign bins ---
    df = df.copy()
    df["bin"] = pd.cut(df[y_col], bins)

    # --- per-bin RMSEs ---
    rmse_per_bin = {}
    for label, col in pred_cols.items():
        rmse_bin = (
            df.groupby("bin", observed=False)[[y_col, col]]
              .apply(lambda x: np.sqrt(((x[y_col] - x[col])**2).mean()))
        )
        rmse_per_bin[label] = rmse_bin

    # --- counts ---
    bin_counts = df.groupby("bin", observed=False)[y_col].size()

    # --- build summary dataframe ---
    summary = pd.DataFrame({"bin": rmse_bin.index, "count": bin_counts.values})
    for label, rmse_bin in rmse_per_bin.items():
        summary[f"{label}_rmse"] = rmse_bin.values

    # --- overall RMSEs ---
    overall = {}
    for label, col in pred_cols.items():
        rmse_all = np.sqrt(((df[y_col] - df[col])**2).mean())
        overall[label] = rmse_all
        print(f"{label} OOF RMSE: {rmse_all:.4f}")

    # --- bar plot per bin ---
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.8 / len(pred_cols)  # distribute bars evenly

    for i, label in enumerate(pred_cols.keys()):
        plt.bar(x + i*width - (width*len(pred_cols))/2,
                summary[f"{label}_rmse"], width, label=label)

    plt.xticks(x, summary["bin"].astype(str), rotation=0)
    plt.xlabel("Target bin")
    plt.ylabel("RMSE")
    title = "RMSE per bin"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return summary, overall     


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


# src/plot.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_metadata_aux_curves(out_dir, binary_aux_tasks, n_folds=5):
    all_logs = []
    for fold in range(1, n_folds + 1):
        lp = os.path.join(out_dir, f"epoch_logs_fold{fold}.csv")
        if os.path.exists(lp):
            all_logs.append(pd.read_csv(lp))

    if not all_logs:
        print(f"No epoch logs found in: {out_dir}")
        return

    logs_df = pd.concat(all_logs).reset_index(drop=True)

    for aux_task in binary_aux_tasks:
        bce_col = f"{aux_task}_val_bce"
        acc_col = f"{aux_task}_val_acc"
        f1_col  = f"{aux_task}_val_f1"   

        if bce_col not in logs_df.columns:
            print(f"Column '{bce_col}' not found, skipping {aux_task}")
            continue

        
        fig, axes = plt.subplots(1, 4, figsize=(22, 4))

        for fold_i in range(1, n_folds + 1):
            fold_log = logs_df[logs_df["fold"] == fold_i]
            if fold_log.empty:
                continue

            axes[0].plot(fold_log["epoch"], fold_log["val_rmse"],
                         label=f"fold {fold_i}", alpha=0.85)
            axes[1].plot(fold_log["epoch"], fold_log[bce_col],
                         label=f"fold {fold_i}", alpha=0.85)
            if acc_col in fold_log.columns:
                axes[2].plot(fold_log["epoch"], fold_log[acc_col] * 100,
                             label=f"fold {fold_i}", alpha=0.85)

            
            if f1_col in fold_log.columns:
                axes[3].plot(fold_log["epoch"], fold_log[f1_col],
                             label=f"fold {fold_i}", alpha=0.85)
           

        axes[0].set_title("Main task — val RMSE (Pawpularity) per epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMSE")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_title(f"Aux head — val BCE ({aux_task}) per epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("BCE Loss")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        axes[2].set_title(f"Aux head — val Accuracy ({aux_task}) per epoch")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy (%)")
        if acc_col in logs_df.columns:
            min_acc = (logs_df[acc_col].min() - 0.02) * 100
            max_acc = (logs_df[acc_col].max() + 0.02) * 100
            axes[2].set_ylim(min_acc, max_acc)
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        
        axes[3].set_title(f"Aux head — val F1 ({aux_task}) per epoch")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("F1 Score")
        if f1_col in logs_df.columns:
            min_f1 = max(0, logs_df[f1_col].min() - 0.05)
            max_f1 = min(1, logs_df[f1_col].max() + 0.05)
            axes[3].set_ylim(min_f1, max_f1)
        axes[3].legend(fontsize=8)
        axes[3].grid(alpha=0.3)

        if f1_col not in logs_df.columns:
            axes[3].text(0.5, 0.5, "Not saved\n(retrain with updated code)",
                         ha="center", va="center",
                         transform=axes[3].transAxes,
                         fontsize=10, color="red")
        

        plt.suptitle(f"Aux-task learning curves — {aux_task}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"learning_curves_{aux_task}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()

def compute_aux_head_summary_table(
    out_dirs_dict,
    n_folds=5,
):
    """
    Compute per-label, per-loss-type summary of auxiliary head metrics
    at the best RMSE epoch across all folds.

    out_dirs_dict: {
        "Blur (BCE)":  "/path/to/exp17_blur_dir",
        "Blur (wBCE)": "/path/to/exp18_blur_wbce_dir",
        "Eyes (BCE)":  "/path/to/exp17_eyes_dir",
        ...
    }

    Returns: summary DataFrame
    """
    rows = []

    for label_name, out_dir in out_dirs_dict.items():
        fold_metrics = []

        for fold in range(1, n_folds + 1):
            lp = os.path.join(out_dir, f"epoch_logs_fold{fold}.csv")
            if not os.path.exists(lp):
                continue

            fold_log = pd.read_csv(lp)

            # pick the best RMSE epoch
            best_row = fold_log.loc[fold_log["val_rmse"].idxmin()]
            fold_metrics.append(best_row)

        if not fold_metrics:
            print(f"No logs found for {label_name} in {out_dir}")
            continue

        metrics_df = pd.DataFrame(fold_metrics)

        # find any metric columns that exist
        def mean_if_exists(col):
            if col in metrics_df.columns:
                return round(metrics_df[col].mean(), 3)
            return None

        # detect which aux task column prefix is present
        # e.g. "Blur_val_f1" or "Eyes_val_f1"
        f1_col   = next((c for c in metrics_df.columns if c.endswith("_val_f1")),   None)
        acc_col  = next((c for c in metrics_df.columns if c.endswith("_val_acc")),  None)
        rec_col  = next((c for c in metrics_df.columns if c.endswith("_val_rec")),  None)
        prec_col = next((c for c in metrics_df.columns if c.endswith("_val_prec")), None)
        auc_col  = next((c for c in metrics_df.columns if c.endswith("_val_auc")),  None)
        ap_col   = next((c for c in metrics_df.columns if c.endswith("_val_ap")),   None)

        rows.append({
            "Label / Loss":  label_name,
            "Val RMSE":      round(metrics_df["val_rmse"].mean(), 3),
            "Accuracy":      mean_if_exists(acc_col)  if acc_col  else None,
            "F1":            mean_if_exists(f1_col)   if f1_col   else None,
            "Recall":        mean_if_exists(rec_col)  if rec_col  else None,
            "Precision":     mean_if_exists(prec_col) if prec_col else None,
            "ROC AUC":       mean_if_exists(auc_col)  if auc_col  else None,
            "Avg Precision": mean_if_exists(ap_col)   if ap_col   else None,
        })

    summary_df = pd.DataFrame(rows)
    return summary_df        
# def plot_metadata_aux_curves(out_dir, binary_aux_tasks, n_folds=5):
#     """
#     Plot per-epoch learning curves for metadata binary aux task experiments.
#     Shows val RMSE, val BCE, and val accuracy for each aux task.

#     Parameters
#     ----------
#     out_dir          : path to experiment output folder
#     binary_aux_tasks : list of aux task names e.g. ["Face"] or ["Eyes", "Face"]
#     n_folds          : number of folds (default 5)
#     """
#     # load all fold logs
#     all_logs = []
#     for fold in range(1, n_folds + 1):
#         lp = os.path.join(out_dir, f"epoch_logs_fold{fold}.csv")
#         if os.path.exists(lp):
#             all_logs.append(pd.read_csv(lp))

#     if not all_logs:
#         print(f"No epoch logs found in: {out_dir}")
#         return

#     logs_df = pd.concat(all_logs).reset_index(drop=True)

#     # plot one figure per aux task
#     for aux_task in binary_aux_tasks:
#         bce_col = f"{aux_task}_val_bce"
#         acc_col = f"{aux_task}_val_acc"

#         if bce_col not in logs_df.columns:
#             print(f"Column '{bce_col}' not found, skipping {aux_task}")
#             continue

#         fig, axes = plt.subplots(1, 3, figsize=(17, 4))

#         for fold_i in range(1, n_folds + 1):
#             fold_log = logs_df[logs_df["fold"] == fold_i]
#             if fold_log.empty:
#                 continue

#             axes[0].plot(fold_log["epoch"], fold_log["val_rmse"],
#                          label=f"fold {fold_i}", alpha=0.85)
#             axes[1].plot(fold_log["epoch"], fold_log[bce_col],
#                          label=f"fold {fold_i}", alpha=0.85)
#             if acc_col in fold_log.columns:
#                 axes[2].plot(fold_log["epoch"], fold_log[acc_col] * 100,
#                              label=f"fold {fold_i}", alpha=0.85)

#         axes[0].set_title("Main task — val RMSE (Pawpularity) per epoch")
#         axes[0].set_xlabel("Epoch")
#         axes[0].set_ylabel("RMSE")
#         axes[0].legend(fontsize=8)
#         axes[0].grid(alpha=0.3)

#         axes[1].set_title(f"Aux head — val BCE ({aux_task}) per epoch")
#         axes[1].set_xlabel("Epoch")
#         axes[1].set_ylabel("BCE Loss")
#         axes[1].legend(fontsize=8)
#         axes[1].grid(alpha=0.3)

#         axes[2].set_title(f"Aux head — val Accuracy ({aux_task}) per epoch")
#         axes[2].set_xlabel("Epoch")
#         axes[2].set_ylabel("Accuracy (%)")
#         if acc_col in logs_df.columns:
#             min_acc = (logs_df[acc_col].min() - 0.02) * 100
#             max_acc = (logs_df[acc_col].max() + 0.02) * 100
#             axes[2].set_ylim(min_acc, max_acc)
#         axes[2].legend(fontsize=8)
#         axes[2].grid(alpha=0.3)

#         plt.suptitle(f"Aux-task learning curves — {aux_task}", fontsize=11)
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, f"learning_curves_{aux_task}.png"),
#                     dpi=150, bbox_inches="tight")
#         plt.show()


