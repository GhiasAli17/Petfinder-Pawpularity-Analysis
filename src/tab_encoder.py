# src/tab_encoder.py
import torch.nn as nn
import numpy as np
import torch

def build_tab_encoder(
    input_dim,
    hidden_dim = 64,
    tab_encoder_capacity = "small",   # "small" or "big"
):
    """
    Small:  2-layer MLP (current default).
    Big:    5-layer deeper/wider MLP.
    """
    if tab_encoder_capacity == "small":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    elif tab_encoder_capacity == "big":
        width = hidden_dim * 2  # e.g., 128 if hidden_dim=64
        return nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width*2),
            nn.ReLU(),
            nn.Linear(width*2, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, hidden_dim),
            nn.ReLU(),
        )
    else:
        raise ValueError(f"Unknown tab encoder_type: {tab_encoder_capacity}")
    
