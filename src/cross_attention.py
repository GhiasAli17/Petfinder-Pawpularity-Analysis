import torch
import torch.nn as nn   
import torch.nn.functional as F
import timm 

class SwinImageTokenEncoder(nn.Module):
    def __init__(self, backbone_name, img_size, d_model, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,), # last stage features
            img_size=img_size,
            dynamic_img_pad=True,
        )
        self.feat_dim = self.backbone.feature_info.channels()[0] # features of last stage, e.g., 1536 for swin_large
        self.proj = nn.Linear(self.feat_dim, d_model)   # Linear projection to match transformer fusion dimension d_model


    def forward(self, x):
       # Extract last-stage Swin feature map
        feats = self.backbone(x)[0]          # (B,H,W,C)
        B, H, W, C = feats.shape
        tokens = feats.view(B, H*W, C)   # (B, N_img=H*W, C)
        return self.proj(tokens)             # (B,N_img,D)

class TabularTokenEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model), # project to d_model
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)      # (B,1,D)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff=1024):
        super().__init__()
        # Multi-head cross-attention module
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # Layer normalization after attention
        self.norm1 = nn.LayerNorm(d_model)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv):
        # Cross-attention: Q attends to K,V
        # q  : (B, N_q, D) , tab tokens
        # kv : (B, N_kv, D), img tokens
        attn_out, _ = self.attn(q, kv, kv)
        # Residual connection + normalization
        x = self.norm1(q + attn_out)
        # Feed-forward network with residual connection 
        return self.norm2(x + self.ffn(x)) # (B, N_q, D)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        backbone_name,
        img_size,
        tab_input_dim,
        tab_hidden=64,
        d_model=256,
        num_heads=4,
        num_layers=2,
        head_hidden=256,
        pretrained=True,
    ):
        super().__init__()
        # Image encoder producing a sequence of image tokens
        self.img_encoder = SwinImageTokenEncoder(
            backbone_name, img_size, d_model, pretrained
        )
        # Tabular encoder producing a single tabular token
        self.tab_encoder = TabularTokenEncoder(
            tab_input_dim, tab_hidden, d_model
        )
         # Stack of cross-attention layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, img, tab):
        # Encode image into multiple spatial tokens
        img_tokens = self.img_encoder(img) # (B, N_img, D)

        # Encode tabular data into a single token
        tab_tokens = self.tab_encoder(tab) # (B, 1, D)

         # Apply cross-attention layers 
         # Tabular token is Query (Q), image tokens are Key (K) and Value (V)
        for layer in self.cross_layers:
            tab_tokens = layer(tab_tokens, img_tokens) # (B, 1, D)

        fused = self.norm(tab_tokens).squeeze(1)
        return self.head(fused)
