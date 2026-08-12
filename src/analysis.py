"""
src/strengthening.py

Reusable functions for the manuscript-strengthening tasks requested in the
supervisor's review. Used by notebooks/exp_strengthening.ipynb.

Grouped by review item:
    A.1  paired_bootstrap, paired_bootstrap_table, marginal_ci_table
    A.2  seed_summary
    B.1  backbone_capacity, config_capacity, count_params
    B.2  loss_group_table
    C.2  feature_dim_caption
"""

import gc
import os

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# Shared OOF loading
# ══════════════════════════════════════════════════════════════════════

def load_oof(path):
    """Read an oof_detail.csv and normalise the prediction column name."""
    d = pd.read_csv(path)
    pred_col = "oof_pred" if "oof_pred" in d.columns else "final_pred"
    return d[["Id", "fold", "ytrue"]].assign(pred=d[pred_col])


def merge_oof(models):
    """
    models: {display_name: path_to_oof_detail.csv}
    Returns a single frame with one prediction column per model, inner-joined
    on Id so that every model is evaluated on exactly the same samples.
    """
    names = list(models)
    merged = load_oof(models[names[0]]).rename(columns={"pred": names[0]})
    for name in names[1:]:
        tmp = load_oof(models[name])[["Id", "pred"]].rename(columns={"pred": name})
        merged = merged.merge(tmp, on="Id", how="inner")
    return merged


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ══════════════════════════════════════════════════════════════════════
# A.1  Paired bootstrap
# ══════════════════════════════════════════════════════════════════════

def paired_bootstrap(y_true, pred_a, pred_b, n_boot=5000, seed=42):
    """
    Bootstrap the DIFFERENCE in RMSE between two models on identical samples.

    The unpaired version (one independent resample per model) measures the
    variance of each model's RMSE, which is dominated by which samples happen
    to be drawn. Drawing one index set and scoring BOTH models on it cancels
    that shared term, leaving the variance of the difference -- the quantity
    that decides whether one model is reliably better than the other.
    """
    y_true = np.asarray(y_true, dtype=float)
    sq_a = (y_true - np.asarray(pred_a, dtype=float)) ** 2
    sq_b = (y_true - np.asarray(pred_b, dtype=float)) ** 2
    n = len(y_true)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)                       # ONE index set ...
        deltas[i] = (np.sqrt(sq_b[idx].mean())
                     - np.sqrt(sq_a[idx].mean()))         # ... scored on BOTH

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    rmse_a, rmse_b = float(np.sqrt(sq_a.mean())), float(np.sqrt(sq_b.mean()))
    return {
        "n_paired": n,
        "RMSE_A": rmse_a,
        "RMSE_B": rmse_b,
        "delta_RMSE": rmse_b - rmse_a,
        "CI95_lower": float(lo),
        "CI95_upper": float(hi),
        "P_delta_lt_0": float((deltas < 0).mean()),
        "excludes_zero": bool((lo < 0 and hi < 0) or (lo > 0 and hi > 0)),
        "_deltas": deltas,
    }


def paired_bootstrap_table(merged, comparisons, n_boot=5000, seed=42):
    """
    comparisons: list of (baseline_name, comparison_name) column pairs.
    Pairs whose columns are absent are skipped, so you can list the optional
    late-fusion comparison before that OOF file exists.
    """
    rows, raw = [], {}
    for a, b in comparisons:
        if a not in merged.columns or b not in merged.columns:
            print(f"  [skip] {b} vs {a} -- column missing")
            continue
        r = paired_bootstrap(merged["ytrue"], merged[a], merged[b], n_boot, seed)
        raw[f"{b} vs {a}"] = r.pop("_deltas")
        rows.append({"Baseline (A)": a, "Model (B)": b,
                     **{k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()}})
    return pd.DataFrame(rows), raw


def marginal_ci_table(merged, names, n_boot=5000, seed=42):
    """
    Reproduces the existing unpaired bootstrap (notebook section 4.5.1d) so the
    two approaches can be shown side by side. Kept deliberately faithful to the
    original: a fresh index set is drawn for each model.
    """
    y_true = merged["ytrue"].values
    n = len(y_true)
    rng = np.random.default_rng(seed)
    rows = []
    for name in names:
        y_pred = merged[name].values
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boot[i] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"Configuration": name,
                     "OOF RMSE": round(rmse(y_true, y_pred), 4),
                     "CI95 lower": round(float(lo), 4),
                     "CI95 upper": round(float(hi), 4)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# A.2  Seed sensitivity
# ══════════════════════════════════════════════════════════════════════

def seed_summary(runs, baseline=None):
    """
    runs: {config_name: {seed: run_directory}}
    Reads <run_directory>/oof_detail.csv for each and reports the pooled OOF
    RMSE per seed with mean and SD across seeds.
    """
    seeds = sorted({s for cfg in runs.values() for s in cfg})
    rows = []
    for name, by_seed in runs.items():
        vals = {}
        for s in seeds:
            rd = by_seed.get(s)
            p = os.path.join(rd, "oof_detail.csv") if rd else None
            if p and os.path.exists(p):
                d = load_oof(p)
                vals[s] = rmse(d["ytrue"], d["pred"])
            else:
                vals[s] = None
        present = [v for v in vals.values() if v is not None]
        row = {"Configuration": name}
        for s in seeds:
            row[f"Seed {s}"] = round(vals[s], 4) if vals[s] is not None else "-"
        row["Mean"] = round(float(np.mean(present)), 4) if present else "-"
        row["SD"] = round(float(np.std(present, ddof=1)), 4) if len(present) > 1 else "-"
        row["_mean"] = float(np.mean(present)) if present else None
        rows.append(row)

    if baseline is not None:
        base = next((r["_mean"] for r in rows if r["Configuration"] == baseline), None)
        for r in rows:
            r["Delta vs baseline"] = (
                round(r["_mean"] - base, 4)
                if base is not None and r["_mean"] is not None
                and r["Configuration"] != baseline else "-")
    return pd.DataFrame(rows).drop(columns=["_mean"])


def seed_verdict(seed_df, observed_gain):
    """Compare the reported improvement against the largest seed SD."""
    sds = [v for v in seed_df.get("SD", []) if isinstance(v, float)]
    if not sds:
        return "Not enough seeds to judge (need at least two per configuration)."
    max_sd = max(sds)
    if observed_gain > 2 * max_sd:
        verdict = "larger than twice the seed SD -- keep the claim as stated."
    elif observed_gain > max_sd:
        verdict = ("larger than one SD but not two -- report as a consistent but "
                   "small improvement and quote the SD alongside it.")
    else:
        verdict = ("WITHIN seed variation -- describe cross-attention as achieving the "
                   "lowest point estimate without a reliable improvement, and soften "
                   "the Abstract, Section 4.4.3 and the Conclusion.")
    return f"Largest seed SD = {max_sd:.4f}; observed gain = {observed_gain:.4f} -> {verdict}"


# ══════════════════════════════════════════════════════════════════════
# B.1 / C.2  Capacity reporting
# ══════════════════════════════════════════════════════════════════════

def count_params(module):
    """Parameter count in millions."""
    return sum(p.numel() for p in module.parameters()) / 1e6


def backbone_capacity(backbones):
    """
    backbones: list of (timm_name, img_size, batch_size_note, loss_note)
    Every value is read from the model itself, not hard-coded. Models are
    created with pretrained=False so nothing downloads; parameter counts and
    feature dimensions are unaffected, and the default pretrained TAG is still
    reported -- that tag is what your training code loaded when it called
    pretrained=True without naming one.
    """
    import timm

    rows = []
    for name, img_size, bs_note, loss_note in backbones:
        kw = {}
        if "swin" in name or "vit" in name:
            kw = {"img_size": img_size, "dynamic_img_pad": True}
        m = timm.create_model(name, pretrained=False, num_classes=0, **kw)
        cfg = getattr(m, "pretrained_cfg", {}) or {}
        rows.append({
            "Exact variant (timm)": name,
            "Feature dimension d_i": int(m.num_features),
            "Parameters (M)": round(count_params(m), 2),
            "Default pretrained tag": cfg.get("tag", "?"),
            "Pretrain source": cfg.get("hf_hub_id", cfg.get("url", "?")),
            "Input resolution": img_size,
            "Batch size": bs_note,
            "Loss": loss_note,
        })
        del m
        gc.collect()
    return pd.DataFrame(rows)


def build_config_model(model_id, eff="efficientnet_b1",
                       swin="swin_large_patch4_window12_384", n_tab=12):
    """
    Mirrors the model construction in run_single_fold() for the configurations
    that appear in Table 1. Imports are local so this module stays importable
    even if a particular src file is unavailable.
    """
    from src.models import VisionRegNet, FeatureConcatFusionNet
    from src.cross_attention import SWINCrossAttention
    from src.film import FiLMInternalModulation, FiLMExternalSingle

    reg = {
        "Swin-Image baseline (MLP head)":
            lambda: (VisionRegNet(swin, 384, head_type="mlp", pretrained=False), swin),
        "Swin-Linear (linear head)":
            lambda: (VisionRegNet(swin, 384, head_type="linear", pretrained=False), swin),
        "Swin-Concat":
            lambda: (FeatureConcatFusionNet(swin, 384, n_tab, head_type="mlp",
                                            pretrained=False), swin),
        "Swin-Cross Attention (MLP encoder)":
            lambda: (SWINCrossAttention(swin, 384, n_tab, tab_hidden=64,
                                        pretrained=False,
                                        tab_encoder_capacity="small"), swin),
        "Swin-Cross Attention (Transformer encoder)":
            lambda: (SWINCrossAttention(swin, 384, n_tab, tab_hidden=64,
                                        pretrained=False,
                                        tab_encoder_capacity="tab_transformer"), swin),
        "Eff-Img-Lin":
            lambda: (VisionRegNet(eff, 256, head_type="linear", pretrained=False), eff),
        "Eff-Img-MLP":
            lambda: (VisionRegNet(eff, 256, head_type="mlp", pretrained=False), eff),
        "Eff-Concat":
            lambda: (FeatureConcatFusionNet(eff, 256, n_tab, head_type="mlp",
                                            pretrained=False), eff),
        "Eff-FiLM-Internal mid":
            lambda: (FiLMInternalModulation(eff, 256, n_tab, tab_hidden=64,
                                            film_start_idx=5,
                                            apply_to_all_after=False,
                                            pretrained=False), eff),
        "Eff-FiLM-External single":
            lambda: (FiLMExternalSingle(eff, 256, n_tab, tab_hidden=64,
                                        pretrained_backbone=False), eff),
    }
    if model_id not in reg:
        raise KeyError(f"Unknown configuration: {model_id}")
    return reg[model_id]()


def _find_backbone(model):
    import torch.nn as nn
    for attr in ("backbone", "img_model", "model"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    return None


def config_capacity(config_ids, **kw):
    """
    Total / backbone / added parameter counts per configuration.

    The 'added' column is what answers the RQ2 form of the capacity objection:
    how much extra capacity each fusion mechanism introduces on top of the
    shared image-only baseline. Models are built one at a time and released,
    since Swin-L is roughly 800 MB in float32.
    """
    rows = []
    for cid in config_ids:
        try:
            model, bb_name = build_config_model(cid, **kw)
        except Exception as ex:
            print(f"  [fail] {cid}: {type(ex).__name__}: {str(ex)[:60]}")
            continue
        total = count_params(model)
        bb = _find_backbone(model)
        bb_p = count_params(bb) if bb is not None else float("nan")
        rows.append({
            "Configuration": cid,
            "Backbone": bb_name,
            "Total (M)": round(total, 2),
            "Backbone (M)": round(bb_p, 2),
            "Added by head/fusion (M)": round(total - bb_p, 2),
        })
        del model
        gc.collect()
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# B.2  Loss usage
# ══════════════════════════════════════════════════════════════════════

def loss_group_table(exp_configs, exp_ids=None):
    """
    Extract backbone / resolution / augmentation / loss / batch size straight
    from src.config.EXP_CONFIGS, so the Loss column in Table 1 is evidenced by
    the code rather than transcribed by hand.
    """
    ids = exp_ids or list(exp_configs)
    rows = []
    for eid in ids:
        c = exp_configs.get(eid, {})
        rows.append({
            "exp id": eid,
            "name": c.get("name", "-"),
            "backbone": c.get("backbone", "-"),
            "img_size": c.get("img_size", "-"),
            "aug": c.get("aug", "-"),
            "head_type": c.get("head_type", "linear"),
            "loss": c.get("loss", "-"),
            "batch_size": c.get("batch_size", "-"),
            "seed": c.get("seed", "-"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# C.2  Figure 2 caption
# ══════════════════════════════════════════════════════════════════════

def feature_dim_caption(backbone_df, tab_dim=64):
    """Generate the corrected Figure 2 caption sentence from measured values."""
    parts = [f"d_i = {int(r['Feature dimension d_i'])} for {r['Exact variant (timm)']}"
             for _, r in backbone_df.iterrows()]
    return (f"d_i denotes the dimension of the image backbone features and d_t "
            f"(dim = {tab_dim}) the dimension of the tabular MLP encoder output. "
            f"Because d_i is backbone-dependent, " + " and ".join(parts) + ".")