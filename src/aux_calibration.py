"""
 Probability Calibration Diagnosis and Post-Hoc Calibration

This module provides:
1. Raw OOF calibration diagnosis:
   - calibration curve
   - Brier score
   - Expected Calibration Error (ECE)

2. Five-fold cross-fitted post-hoc calibration:
   - raw probability baseline
   - Platt scaling
   - isotonic regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss




def calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

     feedback Section 2:
    "ECE measures the discrepancy between predicted probabilities
    and observed positive rates."

    For each probability bin:
    - mean_predicted_probability = average probability in that bin
    - observed_positive_rate = mean true label in that bin
    - bin_gap = absolute difference between those values

    ECE is the sample-count-weighted average bin gap.

    Parameters
    ----------
    y_true : array-like
        Binary labels, containing only 0 and 1.
    y_prob : array-like
        Continuous predicted probabilities in [0, 1].
    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    ece : float
        Lower is better. ECE=0 is ideal under this binning scheme.
    bin_df : pd.DataFrame
        Per-bin data used for calibration curves and interpretation.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    ece = 0.0
    total_n = len(y_true)

    for bin_id in range(n_bins):
        lower = edges[bin_id]
        upper = edges[bin_id + 1]

        # Include probability 0 in the first bin only.
        if bin_id == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)

        n_in_bin = int(mask.sum())

        if n_in_bin == 0:
            rows.append({
                "bin_id": bin_id,
                "bin_lower": lower,
                "bin_upper": upper,
                "n": 0,
                "mean_probability": np.nan,
                "observed_positive_rate": np.nan,
                "absolute_gap": np.nan,
            })
            continue

        mean_probability = float(y_prob[mask].mean())
        observed_positive_rate = float(y_true[mask].mean())
        absolute_gap = abs(mean_probability - observed_positive_rate)

        ece += (n_in_bin / total_n) * absolute_gap

        rows.append({
            "bin_id": bin_id,
            "bin_lower": lower,
            "bin_upper": upper,
            "n": n_in_bin,
            "mean_probability": mean_probability,
            "observed_positive_rate": observed_positive_rate,
            "absolute_gap": absolute_gap,
        })

    return float(ece), pd.DataFrame(rows)


def probability_metrics(y_true, y_prob, n_bins=10):
    """
    Calculate calibration metrics for one probability vector.

     feedback Section 2:
    - Brier score: probability MSE against binary labels.
    - ECE: predicted-probability vs observed-rate mismatch.
    - Bin table: used to draw calibration curve.

    This function does not fit calibration. It only evaluates the
    probabilities supplied to it.

    Returns
    -------
    metrics : dict
        brier, ece, n, positive_rate.
    bin_df : pd.DataFrame
        Calibration-curve bin statistics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = float(brier_score_loss(y_true, y_prob))
    ece, bin_df = calibration_error(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
    )

    metrics = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "brier": brier,
        "ece": ece,
    }

    return metrics, bin_df


def diagnose_raw_oof_calibration(aux_df, aux_tasks, n_bins=10):
    """
    Diagnose raw OOF probabilities for all auxiliary heads.

     feedback Section 2:
    "Calibration should be evaluated separately for each label using
    calibration curve, Brier score, and ECE."


    Parameters
    ----------
    aux_df : pd.DataFrame
        Must have '<task>_true' and '<task>_prob' columns.
    aux_tasks : list[str]
        All auxiliary head names to analyse.
    n_bins : int
        Number of equal-width calibration bins.

    Returns
    -------
    summary_df : pd.DataFrame
        One raw-calibration summary row per task.
    bin_tables : dict[str, pd.DataFrame]
        Calibration-curve statistics per task.
    """
    summary_rows = []
    bin_tables = {}

    for task in aux_tasks:
        true_col = f"{task}_true"
        prob_col = f"{task}_prob"

        if true_col not in aux_df.columns or prob_col not in aux_df.columns:
            raise KeyError(
                f"Missing required columns for task '{task}': "
                f"{true_col}, {prob_col}"
            )

        y_true = aux_df[true_col].to_numpy(dtype=int)
        y_prob = aux_df[prob_col].to_numpy(dtype=float)

        metrics, bin_df = probability_metrics(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=n_bins,
        )

        bin_tables[task] = bin_df

        summary_rows.append({
            "task": task,
            "n": metrics["n"],
            "n_positive": int(y_true.sum()),
            "n_negative": int((y_true == 0).sum()),
            "positive_rate": metrics["positive_rate"],
            "raw_prob_min": float(np.min(y_prob)),
            "raw_prob_max": float(np.max(y_prob)),
            "raw_brier": metrics["brier"],
            "raw_ece": metrics["ece"],
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, bin_tables


def plot_calibration_curves(
    bin_tables,
    aux_tasks,
    title="Raw OOF calibration curves",
    n_cols=4,
    method_label="Raw probability",
):
    """
    Draw one calibration curve per auxiliary head.

     feedback Section 2:
    "Compare predicted probability ranges with observed positive rate."

    """
    n_tasks = len(aux_tasks)
    n_rows = int(np.ceil(n_tasks / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.0 * n_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for i, task in enumerate(aux_tasks):
        ax = axes[i]
        bin_df = bin_tables[task].dropna(
            subset=["mean_probability", "observed_positive_rate"]
        )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.0,
            label="Ideal",
        )

        ax.plot(
            bin_df["mean_probability"],
            bin_df["observed_positive_rate"],
            "o-",
            linewidth=1.8,
            markersize=4,
            label=method_label,
        )

        ax.set_title(task, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Observed positive rate", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    # Hide unused axes if task count is not divisible by n_cols.
    for j in range(n_tasks, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def fit_platt_calibrator(y_prob_fit, y_true_fit):
    """
    Fit Platt scaling on a calibration-fit subset.

     feedback Section 2:
    Candidate post-hoc calibration method: Platt scaling.

    Implementation:
    Logistic regression learns a smooth mapping:
        raw probability -> calibrated probability

    We use raw probabilities as the input because the current OOF CSV
    contains post-sigmoid probabilities, not original logits.
    """
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    x_fit = np.asarray(y_prob_fit, dtype=float).reshape(-1, 1)
    y_fit = np.asarray(y_true_fit, dtype=int)

    model.fit(x_fit, y_fit)

    return model


def fit_isotonic_calibrator(y_prob_fit, y_true_fit):
    """
    Fit isotonic regression on a calibration-fit subset.

     feedback Section 2:
    Candidate post-hoc calibration method: isotonic regression.

    Isotonic regression learns a flexible monotonic mapping:
        raw probability -> calibrated probability
    """
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    model.fit(
        np.asarray(y_prob_fit, dtype=float),
        np.asarray(y_true_fit, dtype=int),
    )

    return model


def apply_calibrator(y_prob, method, calibrator=None):
    """
    Transform raw probabilities using one requested method.

    Parameters
    ----------
    y_prob : array-like
        Raw probabilities.
    method : {"raw", "platt", "isotonic"}
    calibrator : fitted model or None
        Required for Platt and isotonic methods.

    Returns
    -------
    np.ndarray
        Probability vector in [0, 1].
    """
    y_prob = np.asarray(y_prob, dtype=float)

    if method == "raw":
        return y_prob

    if calibrator is None:
        raise ValueError(
            f"A fitted calibrator is required for method='{method}'."
        )

    if method == "platt":
        return calibrator.predict_proba(
            y_prob.reshape(-1, 1)
        )[:, 1]

    if method == "isotonic":
        return calibrator.predict(y_prob)

    raise ValueError(
        "method must be one of: 'raw', 'platt', 'isotonic'."
    )


def crossfit_calibration_one_task(
    aux_df,
    task,
    fold_col="fold",
    n_bins=10,
):
    """
    Generate cross-fitted calibrated probabilities for one auxiliary task.

     feedback Section 2:
    "The data used to fit the calibration method should be separated
    from the data used to evaluate calibration results."

    This uses the existing five OOF folds:
      - For fold k:
          Fit Platt and isotonic on OOF rows from all folds except k.
          Transform raw OOF probabilities in fold k only.
      - Repeat for every fold.
    every row receives:
      - raw probability;
      - Platt calibrated probability fitted without that row/fold;
      - isotonic calibrated probability fitted without that row/fold.

    Returns
    -------
    result_df : pd.DataFrame
        One row per original OOF record with:
          Id, fold, true, raw_prob, platt_prob_cf, isotonic_prob_cf
    method_metrics_df : pd.DataFrame
        Raw/Platt/Isotonic Brier and ECE over all cross-fitted records.
    bin_tables : dict[str, pd.DataFrame]
        Per-method calibration-curve bin tables.
    """
    true_col = f"{task}_true"
    prob_col = f"{task}_prob"

    y_true = aux_df[true_col].to_numpy(dtype=int)
    raw_prob = aux_df[prob_col].to_numpy(dtype=float)
    fold_values = aux_df[fold_col].to_numpy()

    unique_folds = np.sort(np.unique(fold_values))

    platt_prob_cf = np.full(len(aux_df), np.nan, dtype=float)
    isotonic_prob_cf = np.full(len(aux_df), np.nan, dtype=float)

    for heldout_fold in unique_folds:
        fit_mask = fold_values != heldout_fold
        eval_mask = fold_values == heldout_fold

        y_fit = y_true[fit_mask]
        p_fit = raw_prob[fit_mask]
        p_eval = raw_prob[eval_mask]

        # Fit calibration methods on OOF records from the other folds.
        platt_model = fit_platt_calibrator(
            y_prob_fit=p_fit,
            y_true_fit=y_fit,
        )
        isotonic_model = fit_isotonic_calibrator(
            y_prob_fit=p_fit,
            y_true_fit=y_fit,
        )

        # Predict calibrated probabilities only for held-out fold k.
        platt_prob_cf[eval_mask] = apply_calibrator(
            y_prob=p_eval,
            method="platt",
            calibrator=platt_model,
        )
        isotonic_prob_cf[eval_mask] = apply_calibrator(
            y_prob=p_eval,
            method="isotonic",
            calibrator=isotonic_model,
        )

    if np.isnan(platt_prob_cf).any() or np.isnan(isotonic_prob_cf).any():
        raise RuntimeError(
            f"Cross-fitted calibration did not fill all rows for '{task}'."
        )

    result_df = pd.DataFrame({
        "Id": aux_df["Id"].to_numpy(),
        "fold": fold_values,
        "true": y_true,
        "raw_prob": raw_prob,
        "platt_prob_cf": platt_prob_cf,
        "isotonic_prob_cf": isotonic_prob_cf,
    })

    method_probabilities = {
        "raw": raw_prob,
        "platt": platt_prob_cf,
        "isotonic": isotonic_prob_cf,
    }

    metric_rows = []
    bin_tables = {}

    for method, probability in method_probabilities.items():
        metrics, bin_df = probability_metrics(
            y_true=y_true,
            y_prob=probability,
            n_bins=n_bins,
        )

        metric_rows.append({
            "task": task,
            "method": method,
            "n": metrics["n"],
            "positive_rate": metrics["positive_rate"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
        })

        bin_tables[method] = bin_df

    method_metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=["brier", "ece"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return result_df, method_metrics_df, bin_tables


def crossfit_calibration_all_tasks(
    aux_df,
    aux_tasks,
    fold_col="fold",
    n_bins=10,
):
    """
    Run five-fold calibration cross-fitting for every auxiliary head.

     feedback Section 2:
    - Evaluate each label separately.
    - Compare raw and calibrated probabilities on held-out data.
    - Avoid fitting/evaluating calibration on the same samples.

    Returns
    -------
    final_probability_df : pd.DataFrame
        Original Id/fold plus, for every task:
          <task>_true
          <task>_prob_raw
          <task>_prob_platt_cf
          <task>_prob_isotonic_cf

    comparison_df : pd.DataFrame
        One row per task and method, containing Brier and ECE.

    bin_tables_by_task : dict
        {task: {raw: bin_df, platt: bin_df, isotonic: bin_df}}
    """
    final_probability_df = aux_df[["Id", fold_col]].copy()

    comparison_parts = []
    bin_tables_by_task = {}

    for task in aux_tasks:
        task_df, task_metrics_df, task_bin_tables = (
            crossfit_calibration_one_task(
                aux_df=aux_df,
                task=task,
                fold_col=fold_col,
                n_bins=n_bins,
            )
        )

        final_probability_df[f"{task}_true"] = task_df["true"]
        final_probability_df[f"{task}_prob_raw"] = task_df["raw_prob"]
        final_probability_df[f"{task}_prob_platt_cf"] = (
            task_df["platt_prob_cf"]
        )
        final_probability_df[f"{task}_prob_isotonic_cf"] = (
            task_df["isotonic_prob_cf"]
        )

        comparison_parts.append(task_metrics_df)
        bin_tables_by_task[task] = task_bin_tables

    comparison_df = pd.concat(
        comparison_parts,
        ignore_index=True,
    )

    return final_probability_df, comparison_df, bin_tables_by_task

def plot_calibration_comparison(
    bin_tables_by_task,
    aux_tasks,
    title="Raw vs cross-fitted calibration curves",
    n_cols=4,
):
    # Plot raw, Platt, and isotonic curves for each feedback head.
    methods = {
        "raw": "Raw",
        "platt": "Platt (cross-fitted)",
        "isotonic": "Isotonic (cross-fitted)",
    }

    n_tasks = len(aux_tasks)
    n_rows = int(np.ceil(n_tasks / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.0 * n_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for i, task in enumerate(aux_tasks):
        ax = axes[i]

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.0,
            label="Ideal",
        )

        for method, label in methods.items():
            bin_df = bin_tables_by_task[task][method].dropna(
                subset=["mean_probability", "observed_positive_rate"]
            )

            ax.plot(
                bin_df["mean_probability"],
                bin_df["observed_positive_rate"],
                "o-",
                linewidth=1.5,
                markersize=3,
                label=label,
            )

        ax.set_title(task, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Observed positive rate", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    for j in range(n_tasks, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def select_calibration_method(
    comparison_df,
    task,
    min_brier_improvement=0.0,
    min_ece_improvement=0.0,
):
    """
    Select raw, Platt, or isotonic separately for one task.

     feedback Section 2:
    "Adopt a post-hoc calibration method only if it produces a genuine
    improvement on the separate evaluation set. Otherwise retain raw."

    A calibrated method is accepted only if, relative to raw:
    - Brier score decreases ;
    - ECE decreases .

    Returns
    -------
    dict
        Selected method and its raw-vs-selected metric comparison.
    """
    task_rows = comparison_df[
        comparison_df["task"] == task
    ].copy()

    raw_row = task_rows[
        task_rows["method"] == "raw"
    ].iloc[0]

    candidate_rows = task_rows[
        task_rows["method"] != "raw"
    ].copy()

    candidate_rows["brier_improvement"] = (
        raw_row["brier"] - candidate_rows["brier"]
    )
    candidate_rows["ece_improvement"] = (
        raw_row["ece"] - candidate_rows["ece"]
    )

    valid_candidates = candidate_rows[
        (candidate_rows["brier_improvement"] > min_brier_improvement)
        & (candidate_rows["ece_improvement"] > min_ece_improvement)
    ]

    if valid_candidates.empty:
        return {
            "task": task,
            "selected_method": "raw",
            "raw_brier": float(raw_row["brier"]),
            "raw_ece": float(raw_row["ece"]),
            "selected_brier": float(raw_row["brier"]),
            "selected_ece": float(raw_row["ece"]),
            "brier_improvement": 0.0,
            "ece_improvement": 0.0,
        }

    best_row = valid_candidates.sort_values(
        by=["brier", "ece"],
        ascending=[True, True],
    ).iloc[0]

    return {
        "task": task,
        "selected_method": best_row["method"],
        "raw_brier": float(raw_row["brier"]),
        "raw_ece": float(raw_row["ece"]),
        "selected_brier": float(best_row["brier"]),
        "selected_ece": float(best_row["ece"]),
        "brier_improvement": float(best_row["brier_improvement"]),
        "ece_improvement": float(best_row["ece_improvement"]),
    }


def select_calibration_methods_all_tasks(
    comparison_df,
    aux_tasks,
    min_brier_improvement=0.0,
    min_ece_improvement=0.0,
):
    """
    Select the final probability method independently for all heads.

    This returns the completed Section 2 decision table:
      task | selected_method | raw metrics | selected metrics | improvements
    """
    decisions = [
        select_calibration_method(
            comparison_df=comparison_df,
            task=task,
            min_brier_improvement=min_brier_improvement,
            min_ece_improvement=min_ece_improvement,
        )
        for task in aux_tasks
    ]

    return pd.DataFrame(decisions)


