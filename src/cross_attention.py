# src/cross_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.tab_encoder import build_tab_encoder


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block.

    Two modes controlled by query_mode:
      "tab_queries_image":
          Query  = tab token  (B, 1, D)
          Key    = image tokens (B, N, D)
          Value  = image tokens (B, N, D)
          -> tab attends to image, output: (B, 1, D)

    """
    def __init__(
        self,
        visual_dim: int, # dim of visual tokens, e.g. 1536 for SWIN-Large, 1280 for EffB1
        tab_dim: int, # dim of tab embedding from tab_enc, e.g. 64
        num_heads: int = 8, # number of parallel attention heads
        dropout: float = 0.1, # dropout rate for attention and output
        query_mode: str = "tab_queries_image", 
    ):
        super().__init__()
        self.query_mode = query_mode

        #   project tab to same dim as visual tokens
        #  (B,64) -> (B,1536) so tab and visual are in same dim space
        #  required because Q, K, V must all have same embed_dim in MultiheadAttention
        self.tab_proj = nn.Linear(tab_dim, visual_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim, # total dim = 1536
            num_heads=num_heads, # 8 heads, each handles dim = 1536/8 = 192
            dropout=dropout, # dropout inside attention weights
            batch_first=True,   # input shape (B, seq, dim), not (seq, B, dim)
        )
        self.norm = nn.LayerNorm(visual_dim) # post-attention normalization
        self.dropout = nn.Dropout(dropout) # post-attention dropout

    def forward(self, visual_tokens, tab_feat):
        """
        visual_tokens: (B, N, D)
        tab_feat:      (B, tab_dim)
        returns:       (B, D)  fused representation
        """
         # visual_tokens: (B, N, D)   e.g. (B, 144, 1536) for SWIN
        # tab_feat:      (B, tab_dim) e.g. (B, 64)
        tab_token = self.tab_proj(tab_feat).unsqueeze(1)  # (B, 1, D)
         # tab_proj:  (B, 64)   -> (B, 1536)
        # unsqueeze: (B, 1536) -> (B, 1, 1536)  treat as sequence of length 1


        if self.query_mode == "tab_queries_image":
            # tab token queries image tokens # MODE 1: tab attends to image Q is tabular
            # K, V is image token
            attn_out, _ = self.cross_attn(
                query=tab_token,       # (B, 1, D) Q, (B, 1,   1536)
                key=visual_tokens,     # (B, N, D) K, (B, 144, 1536)
                value=visual_tokens,   # (B, N, D) V, (B, 144, 1536)
            )  # (B, 1, D)
            # internally per head:
            #   attn_weights = softmax(Q·Kᵀ / sqrt(192)) -> (B, 8, 1, 144)
            #   attn_out     = attn_weights · V           -> (B, 8, 1, 192)
            # all 8 heads concatenated                    -> (B, 1, 1536)
            out = self.norm(self.dropout(attn_out).squeeze(1))  # (B, D)  squeeze:  (B,1, 1536)    remove sequence dim LayerNorm:(B, 1536)    normalize fused representation
        else:
            raise ValueError(f"Unknown query_mode: {self.query_mode}")

        return out  # (B, D)-> (B, D=1536) fused representation


class CrossAttentionFusionHead(nn.Module):
    """
    Final regression head after cross-attention fusion.
    (B, D) -> MLP -> (B, 1)
    """
    def __init__(self, visual_dim: int, head_hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, head_hidden), # (B, 1536) -> (B, 256)
            nn.ReLU(),
            nn.Linear(head_hidden, 1), # (B, 256)  -> (B, 1)
        )

    def forward(self, x):
        return self.mlp(x)  # (B, 1) final pawpularity score

#
# SWIN Transformer + Cross-Attention
# 

class SWINCrossAttention(nn.Module):
    """
    SWIN backbone -> patch tokens (B, N, D)
    -> CrossAttentionBlock 
    -> regression head
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
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_pad=True,
        )

      

        # SWIN num_features is the token dim D
        visual_dim = self.backbone.num_features   # e.g.  1536 for swin_large

        self.tab_enc = build_tab_encoder(
            tab_input_dim, tab_hidden, tab_encoder_capacity
        ) # (B, 12) -> (B, 64) for small

        self.cross_attn = CrossAttentionBlock(
            visual_dim=visual_dim, # 1536
            tab_dim=tab_hidden, #64
            num_heads=num_heads, #8 heads, each handles dim = 1536/8 = 192 dimension
            dropout=dropout,
            query_mode=query_mode,
        )

        self.head = CrossAttentionFusionHead(
            visual_dim=visual_dim,
            head_hidden=head_hidden,
        ) # 1536 -> 256 -> 1

    def forward(self, img, tab):
       
        
        visual_tokens = self.backbone.forward_features(img)
        # swin_large_patch4_window12_384 returns (B, 12, 12, 1536)
        if visual_tokens.dim() == 4:
            B, H, W, D = visual_tokens.shape   # B, 12, 12, 1536
            visual_tokens = visual_tokens.reshape(B, H * W, D)  # (B, 144, 1536) we need to convert because head expects (B, N, D) shape where N is number of tokens, here N=H*W=144

        tab_feat = self.tab_enc(tab)                          # (B, tab_hidden)
        fused    = self.cross_attn(visual_tokens, tab_feat)   # (B, 1536)
        out      = self.head(fused)                           # (B, 1)
        return out