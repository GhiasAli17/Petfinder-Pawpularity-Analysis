# src/tab_encoder.py
import torch.nn as nn
import numpy as np
import torch

class TabularTransformerEncoder(nn.Module):
    """
    Tabular Transformer encoder.

    Each input feature gets its own linear embedding → (B, F, D).
    A standard Transformer encoder then lets features attend to each other.
    Final output is the mean-pooled token → (B, hidden_dim).

    Args:
        input_dim:   number of tabular features F
        hidden_dim:  output embedding dim D (also used as token dim)
        num_heads:   attention heads (must divide hidden_dim)
        num_layers:  number of Transformer encoder layers
        dropout:     dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Each feature scalar → D-dim token
        # One linear per feature keeps them independent at embedding stage
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(input_dim)
        ])

        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,   # (B, seq, dim)
            norm_first=True,    # pre-LN, more stable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: (B, F)  — raw tabular features
        returns: (B, hidden_dim)
        """
        B, F = x.shape

        # Embed each feature: list of (B, 1) -> (B, 1, D), concat -> (B, F, D)
        tokens = torch.stack([
            self.feature_embeddings[i](x[:, i].unsqueeze(1))   # (B, 1) -> (B, D)
            for i in range(F)
        ], dim=1)   # (B, F, D)

        # Self-attention across features
        tokens = self.transformer(tokens)   # (B, F, D)

        # Mean pool over feature tokens -> single query vector
        out = self.norm(tokens.mean(dim=1))   # (B, D)

        return out


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
    elif tab_encoder_capacity == "tab_transformer":
        return TabularTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,      # hidden_dim must be divisible by this, e,g 64
            num_layers=2,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown tab encoder_type: {tab_encoder_capacity}")
    
