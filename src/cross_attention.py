# src/cross_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.tab_encoder import build_tab_encoder


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block.

    Mode "tab_queries_image":
        Query  = tab token  (B, 1, D)
        Key    = image tokens (B, N, D)
        Value  = image tokens (B, N, D)
        -> tab attends to image, output: (B, 1, D)

    This block implements:
      tab_proj -> cross-attn -> residual + LN ->
      FFN -> residual + LN
    """

    def __init__(
        self,
        visual_dim: int,   # dim of visual tokens, e.g. 1536 for SWIN-Large
        tab_dim: int,      # dim of tab embedding from tab_enc, e.g. 64
        num_heads: int = 8,
        dropout: float = 0.1,
        query_mode: str = "tab_queries_image",
    ):
        super().__init__()
        self.query_mode = query_mode

        # Project tab to same dim as visual tokens
        # (B, tab_dim) -> (B, visual_dim)
        self.tab_proj = nn.Linear(tab_dim, visual_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # (B, seq, dim)
        )

        # Attention residual + norm
        self.norm1 = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN sublayer (Transformer-style)
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, 4 * visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * visual_dim, visual_dim),
        )
        self.norm2 = nn.LayerNorm(visual_dim)

    def forward(self, visual_tokens, tab_feat):
        """
        visual_tokens: (B, N, D)
        tab_feat:      (B, tab_dim)

        returns: (B, D) fused representation
        """

        # tab_feat: (B, tab_dim) -> (B, D) -> (B, 1, D)
        tab_token = self.tab_proj(tab_feat).unsqueeze(1)

        if self.query_mode == "tab_queries_image":
            # Q = tab token, K/V = image tokens
            attn_out, _ = self.cross_attn(
                query=tab_token,      # (B, 1, D)
                key=visual_tokens,    # (B, N, D)
                value=visual_tokens,  # (B, N, D)
            )                         # (B, 1, D)

            attn_out = attn_out.squeeze(1)          # (B, D)
            residual = tab_token.squeeze(1)         # (B, D)

            # Attention residual + norm
            x = self.norm1(residual + self.dropout(attn_out))  # (B, D)

            # FFN residual + norm
            ffn_out = self.ffn(x)                   # (B, D)
            x = self.norm2(x + self.dropout(ffn_out))  # (B, D)

        else:
            raise ValueError(f"Unknown query_mode: {self.query_mode}")

        return x  # (B, D)


class CrossAttentionFusionHead(nn.Module):
    """
    Final regression head after cross-attention fusion.
    (B, D) -> MLP -> (B, 1)
    """

    def __init__(self, visual_dim: int, head_hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, head_hidden),  # (B, D) -> (B, head_hidden)
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),           # (B, head_hidden) -> (B, 1)
        )

    def forward(self, x):
        return self.mlp(x)  # (B, 1)


class SWINCrossAttention(nn.Module):
    """
    SWIN backbone -> patch tokens (B, N, D)
    -> CrossAttentionBlock x K (stacked)
    -> residual mix with image-only embedding
    -> regression head

    Stacking rationale:
      Block 0: tab token does a broad sweep of the image patches,
               producing a coarse fused vector (B, D).
      Block 1+: that fused vector is re-used as "tab_feat" and
                attends to the image again with a more refined query.
    """

    def __init__(
        self,
        backbone_name: str,
        img_size: int,
        tab_input_dim: int,
        tab_hidden: int = 128,
        num_heads: int = 8,
        head_hidden: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        tab_encoder_capacity: str = "small",
        query_mode: str = "tab_queries_image",
        num_cross_attn_blocks: int = 1,  # number of stacked cross-attention blocks
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_pad=True,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # SWIN num_features is the token dim D
        visual_dim = self.backbone.num_features  # e.g. 1536 for swin_large

        # Tab encoder: (B, tab_input_dim) -> (B, tab_hidden)
        self.tab_enc = build_tab_encoder(
            tab_input_dim, tab_hidden, tab_encoder_capacity
        )

        # Stack num_cross_attn_blocks blocks, each with independent weights.
        # Block 0: tab_hidden -> visual_dim
        # Block 1+: visual_dim -> visual_dim (refines fused token)
        self.cross_attn_blocks = nn.ModuleList()
        for i in range(num_cross_attn_blocks):
            in_tab_dim = tab_hidden if i == 0 else visual_dim
            self.cross_attn_blocks.append(
                CrossAttentionBlock(
                    visual_dim=visual_dim,
                    tab_dim=in_tab_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    query_mode=query_mode,
                )
            )

        self.head = CrossAttentionFusionHead(
            visual_dim=visual_dim,
            head_hidden=head_hidden,
        )

    def forward(self, img, tab):
        # SWIN forward_features returns (B, H, W, D) or (B, N, D)
        visual_tokens = self.backbone.forward_features(img)

        if visual_tokens.dim() == 4:
            B, H, W, D = visual_tokens.shape
            visual_tokens = visual_tokens.reshape(B, H * W, D)  # (B, N, D)

        # image-only summary (mean over tokens)
        h_img = visual_tokens.mean(dim=1)          # (B, D)

        # tab embedding
        tab_feat = self.tab_enc(tab)               # (B, tab_hidden)

        # Stacked cross-attention blocks:
        # fused starts as the raw tab embedding and is progressively refined.
        fused = tab_feat
        for block in self.cross_attn_blocks:
            fused = block(visual_tokens, fused)    # (B, D)

        # Residual mix: image-only + fused
        fused = fused + h_img                      # (B, D)
        fused = F.layer_norm(fused, fused.shape[-1:])  # (B, D)

        out = self.head(fused)                     # (B, 1)
        return out