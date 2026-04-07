# src/film_analysis.py

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# HOOKS: collect gamma, beta, feature before/after FiLM
# ─────────────────────────────────────────────────────────────

class FiLMAnalysisHooks:
    """
    Registers hooks on every FiLM layer inside a FiLMInternalModulation
    or FiLMExternalModulation model.

    Captured per batch:
        gamma_stats[layer_idx]  -> list of (B, C) gamma tensors  
        beta_stats[layer_idx]   -> list of (B, C) beta  tensors  
        feat_before[layer_idx]  -> list of (B, C, H, W) tensors    [pre-FiLM]
        feat_after[layer_idx]   -> list of (B, C, H, W) tensors    [post-FiLM]
    """

    def __init__(self, model):
        self.handles = [] # list of hook handle objects so they can be removed later
        self.gamma_stats = defaultdict(list) # maps layer_idx -> list of gamma batches (B, C)
        self.beta_stats  = defaultdict(list) # maps layer_idx -> list of beta  batches (B, C)
        self.feat_before = defaultdict(list)  # maps layer_idx -> list of features before FiLM (B, C, H, W)
        self.feat_after  = defaultdict(list) # maps layer_idx -> list of features after  FiLM (B, C, H, W)
        self._register(model) # to find all FiLM modules in model and attach hooks
 
    def _register(self, model):
        """
        Works for:
          - Internal: model.film_layers (ModuleDict keyed by block index)
          - External: model.film_stack.blocks[i].film
        """
        # from src.film import FiLM  

        # internal FiLM
        if hasattr(model, "film_layers"): # internal model has a ModuleDict self.film_layers
            for key, film_module in model.film_layers.items():
                layer_idx = int(key)
                self._hook_film(film_module, layer_idx) # register hooks on this FiLM

        # external FiLM
        if hasattr(model, "film_stack") and hasattr(model.film_stack, "blocks"):
            for i, res_block in enumerate(model.film_stack.blocks): # iterate FiLMedResBlocks: i = 0..num_blocks-1
                if hasattr(res_block, "film"): # each block has a .film submodule in case of external FiLM Modluation
                    self._hook_film(res_block.film, i)

    def _hook_film(self, film_module, layer_idx):
        def pre_hook(module, args):
            # args = (x, cond) from FiLM.forward(x, cond)
            x = args[0].detach().cpu() # x is (B, C, H, W)
            self.feat_before[layer_idx].append(x) # store features **before** FiLM for this layer and batch

        def fwd_hook(module, args, output):
            x, cond = args[0], args[1]  # x: (B, C, H, W), cond: (B, cond_dim)
            with torch.no_grad():
                gamma = module.gamma_fc(cond).detach().cpu() # gamma: (B, C) predicted from cond
                beta  = module.beta_fc(cond).detach().cpu()
            self.gamma_stats[layer_idx].append(gamma) # store γ for this layer and batch
            self.beta_stats[layer_idx].append(beta)
            self.feat_after[layer_idx].append(output.detach().cpu()) # output: FiLM(x, cond) (B, C, H, W)

        h1 = film_module.register_forward_pre_hook(pre_hook) # attach pre-forward hook to module
        h2 = film_module.register_forward_hook(fwd_hook) # attach post-forward hook to module
        self.handles.extend([h1, h2]) #handles for later removal

    def remove(self):
        for h in self.handles: # iterate over all stored hook handles
            h.remove() # unregister hook from module
        self.handles.clear() # clear list

    def aggregate(self):
        # For each layer_idx k: list v = [ (B1,C), (B2,C), ... ]
        gamma  = {k: torch.cat(v, dim=0).numpy() for k, v in self.gamma_stats.items()}   # gamma[k]: (N, C) where N = total number of samples over all processed batches
        beta   = {k: torch.cat(v, dim=0).numpy() for k, v in self.beta_stats.items()}         # beta[k]: (N, C)
        before = {k: torch.cat(v, dim=0).numpy() for k, v in self.feat_before.items()} # before[k]: (N, C, H, W) before FiLM for layer k
        after  = {k: torch.cat(v, dim=0).numpy() for k, v in self.feat_after.items()}   # after[k]: (N, C, H, W)
        return gamma, beta, before, after


@torch.no_grad()
def collect_film_stats(model, val_loader, device, max_batches=None):
    """
    Run model on val_loader and collect γ, β, features before/after FiLM.

    Returns
    -------
    gamma, beta, feat_before, feat_after : dict[int -> array]
    """
    hooks = FiLMAnalysisHooks(model) # attach hooks to all FiLM modules in the model
    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs, tabs, _ = batch  # imgs: (B, 3, H, W), tabs: (B, tab_dim), _: labels (unused)
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        _ = model(imgs, tabs) # forward pass triggers hooks; output is ignored

    gamma, beta, feat_before, feat_after = hooks.aggregate() # gather all collected stats
    hooks.remove() # detach hooks from model
    return gamma, beta, feat_before, feat_after 


# ─────────────────────────────────────────────────────────────
# CASE A1: per-experiment, per-FiLM-block histograms + tables
# ─────────────────────────────────────────────────────────────

def case_A_per_block(
    gamma: dict, # dict[int -> (N, C)]
    beta: dict, # dict[int -> (N, C)]
    feat_before: dict, # dict[int -> (N, C, H, W)]
    feat_after: dict, # dict[int -> (N, C, H, W)]
    exp_name: str,
    out_dir: str,
    n_table_rows: int = 10,
):
    """
    For each FiLM block in this experiment (for internal: single index,
    for external: 4 blocks), produce:

      - Histogram of gamma values
      - Histogram of beta values
      - Histogram of features before + after FiLM (two curves on same plot)
      - Table (first n_table_rows scalars) with:
          sample_idx, channel, h, w, before, after, gamma, beta

     printing shapes of gamma, beta, feat_before, feat_after for each block.
    """
    os.makedirs(out_dir, exist_ok=True)
    block_ids = sorted(feat_before.keys()) # list of FiLM block indices present in stats

    for block_idx in block_ids:
        g = gamma[block_idx]        # (N, C)
        b = beta[block_idx]         # (N, C)
        fb = feat_before[block_idx] # (N, C, H, W)
        fa = feat_after[block_idx]  # (N, C, H, W)

        print(f"\n[Case A] Block {block_idx}  — {exp_name}")
        print(f"  gamma shape:       {g.shape}")
        print(f"  beta shape:        {b.shape}")
        print(f"  feat_before shape: {fb.shape}")
        print(f"  feat_after shape:  {fa.shape}")

        # ---- histograms: gamma and beta -------------------------
        g_flat = g.reshape(-1)  # (N*C,) all gamma scalars for this block
        b_flat = b.reshape(-1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"[Case A] γ/β Histograms — Block {block_idx} — {exp_name}",
                     fontsize=13)

        axes[0].hist(g_flat, bins=80, color="cornflowerblue",
                     alpha=0.8, edgecolor="white")
        axes[0].set_title(f"γ (mean={g_flat.mean():.4f}, std={g_flat.std():.4f})")
        axes[0].set_xlabel("γ")
        axes[0].set_ylabel("count")

        axes[1].hist(b_flat, bins=80, color="salmon",
                     alpha=0.8, edgecolor="white")
        axes[1].set_title(f"β (mean={b_flat.mean():.4f}, std={b_flat.std():.4f})")
        axes[1].set_xlabel("β")
        axes[1].set_ylabel("count")

        plt.tight_layout()
        fname = os.path.join(out_dir,
                             f"A_block{block_idx}_gamma_beta_hist_{exp_name}.png")
        # plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        # print(f"Saved  {fname}")

        # ---- histogram: features before & after in same plot ----
        fb_flat = fb.reshape(-1)
        fa_flat = fa.reshape(-1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(fb_flat, bins=80, alpha=0.5, label="before", color="steelblue",
                edgecolor="white")
        # vmin, vmax = np.percentile(fb_flat, [1, 99])
        # ax.set_xlim(vmin, vmax)
        ax.hist(fa_flat, bins=80, alpha=0.5, label="after",  color="darkorange",
                edgecolor="white")
        # vmin, vmax = np.percentile(fa_flat, [1, 99])
        # ax.set_xlim(vmin, vmax)
        ax.set_title(
            f"Features before/after FiLM — Block {block_idx}\n"
            f"before (μ={fb_flat.mean():.4f}, σ={fb_flat.std():.4f}), "
            f"after (μ={fa_flat.mean():.4f}, σ={fa_flat.std():.4f})"
        )
        ax.set_xlabel("feature value")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(out_dir,
                             f"A_block{block_idx}_features_hist_{exp_name}.png")
        # plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        # print(f"Saved  {fname}")

        # ---- small table: first n_table_rows scalar entries ----
        N, C, H, W = fb.shape
        total_scalars = N * C * H * W
        n_show = min(n_table_rows, total_scalars) #entries to show in table
        
        rows = []
        #this block converts a single flat scalar index flat_idx into
        #  its 4D coordinates (sample n, channel c, height h, width w)
        #  within the tensor of shape (N, C, H, W)
        for flat_idx in range(n_show): # flat_idx runs over a 1D indexing of all N*C*H*W scalars
            n = flat_idx // (C * H * W)  # decode which sample this scalar belongs to (0..N-1)
            rem = flat_idx %  (C * H * W)  # remaining offset inside that sample's (C*H*W) block
            c = rem // (H * W)  # decode which channel within the sample (0..C-1)
            rem = rem %  (H * W) # remaining offset inside that 
            h = rem // W  # decode row index in the spatial grid (0..H-1)
            w = rem %  W  # decode column index in the spatial grid (0..W-1)
            before_val = float(fb[n, c, h, w])
            after_val  = float(fa[n, c, h, w])
            rows.append({
                "block_idx":  int(block_idx),
                "sample_idx": int(n),  # sample index in dataset (0..N-1)
                "channel":    int(c),
                "h":          int(h),
                "w":          int(w),
                "before":     float(fb[n, c, h, w]),
                "after":      float(fa[n, c, h, w]),
                "gamma":      float(g[n, c]),
                "beta":       float(b[n, c]),
                "abs_diff":   abs(after_val - before_val)
            })

        table_df = pd.DataFrame(rows)
        display(table_df) #display() is the Jupyter/IPython y function that shows nice  table in notebooks

       # Global statistics for this block
        diff_flat = fa_flat - fb_flat
        l2 = np.linalg.norm(diff_flat)

        before_stats = {
            "min":  fb_flat.min(),
            "max":  fb_flat.max(),
            "mean": fb_flat.mean(),
            "std":  fb_flat.std(),
        }
        after_stats = {
            "min":  fa_flat.min(),
            "max":  fa_flat.max(),
            "mean": fa_flat.mean(),
            "std":  fa_flat.std(),
        }

        # Optional prints (keep or remove)
        print(f"[Case A] Block {block_idx} global stats:")
        print(f"  L2 ||after - before|| = {l2:.4f}")

        # DataFrame view
        global_rows = [
            {
                "block_idx": block_idx,
                "which": "before",
                "min":  before_stats["min"],
                "max":  before_stats["max"],
                "mean": before_stats["mean"],
                "std":  before_stats["std"],
               
            },
            {
                "block_idx": block_idx,
                "which": "after",
                "min":  after_stats["min"],
                "max":  after_stats["max"],
                "mean": after_stats["mean"],
                "std":  after_stats["std"],
                
            },
        ]

        global_df = pd.DataFrame(global_rows)
        display(global_df)



def case_A_per_channel(
    feat_before: dict,
    feat_after: dict,
    exp_name: str,
    out_dir: str,
    max_channels: int = None,
):
    """
    For each FiLM block and each channel c:
      - Histogram of features before FiLM (over all N,H,W)
      - Histogram of features after  FiLM (over all N,H,W)

    If max_channels is not None, only plot channels [0 .. max_channels-1].
    """
    os.makedirs(out_dir, exist_ok=True)
    block_ids = sorted(feat_before.keys())

    for block_idx in block_ids:
        fb = feat_before[block_idx]  # (N, C, H, W)
        fa = feat_after[block_idx]   # (N, C, H, W)
        N, C, H, W = fb.shape

        print(f"\n[Case A per-channel] Block {block_idx} — {exp_name}")
        print(f"  feat_before shape: {fb.shape}")
        print(f"  feat_after  shape: {fa.shape}")

        ch_ids = list(range(C)) # default: all channels
        if max_channels is not None:
            ch_ids = ch_ids[:max_channels] # optionally for restricting  first K channels

        for c in ch_ids:
            fb_c = fb[:, c, :, :].reshape(-1) # (N*H*W,) all scalars for channel c before FiLM
            fa_c = fa[:, c, :, :].reshape(-1) # (N*H*W,) after

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(
                fb_c, bins=80, alpha=0.5, label="before", color="steelblue",
                edgecolor="white",
            )
            ax.hist(
                fa_c, bins=80, alpha=0.5, label="after", color="darkorange",
                edgecolor="white",
            )

            ax.set_title(
                f"Block {block_idx}, Channel {c}\n"
                f"before (μ={fb_c.mean():.4f}, σ={fb_c.std():.4f}), "
                f"after (μ={fa_c.mean():.4f}, σ={fa_c.std():.4f})"
            )
            ax.set_xlabel("feature value")
            ax.set_ylabel("count")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            fname = os.path.join(
                out_dir,
                f"A_block{block_idx}_ch{c}_features_hist_{exp_name}.png",
            )
            # plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.show()
            # print(f"Saved {fname}")

def case_A_per_sample(
    feat_before: dict,
    feat_after: dict,
    exp_name: str,
    out_dir: str,
    max_samples: int = None,
):
    """
    For each FiLM block and each sample n:
      - Histogram of features before FiLM (over all C,H,W)
      - Histogram of features after  FiLM (over all C,H,W)

    If max_samples is not None, only plot samples [0 .. max_samples-1].
    """
    os.makedirs(out_dir, exist_ok=True)
    block_ids = sorted(feat_before.keys())

    for block_idx in block_ids:
        fb = feat_before[block_idx]  # (N, C, H, W)
        fa = feat_after[block_idx]   # (N, C, H, W)
        N, C, H, W = fb.shape

        print(f"\n[Case A per-sample] Block {block_idx} — {exp_name}")
        print(f"  feat_before shape: {fb.shape}")
        print(f"  feat_after  shape: {fa.shape}")

        sample_ids = list(range(N))
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]

        for n in sample_ids:
            fb_n = fb[n].reshape(-1)   # fb[n]: (C, H, W) -> (C*H*W,) all scalars for sample n before FiLM
            fa_n = fa[n].reshape(-1) # fa[n]: (C, H, W) -> (C*H*W,) after

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(
                fb_n, bins=80, alpha=0.5, label="before", color="steelblue",
                edgecolor="white",
            )
            ax.hist(
                fa_n, bins=80, alpha=0.5, label="after", color="darkorange",
                edgecolor="white",
            )

            ax.set_title(
                f"Block {block_idx}, Sample {n}\n"
                f"before (μ={fb_n.mean():.4f}, σ={fb_n.std():.4f}), "
                f"after (μ={fa_n.mean():.4f}, σ={fa_n.std():.4f})"
            )
            ax.set_xlabel("feature value")
            ax.set_ylabel("count")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            fname = os.path.join(
                out_dir,
                f"A_block{block_idx}_sample{n}_features_hist_{exp_name}.png",
            )
            # plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.show()
            # print(f"Saved {fname}")
