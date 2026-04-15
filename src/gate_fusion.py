
# Gated Fusion 

import torch
import torch.nn as nn
import timm

class GatedFusion(nn.Module):
    """
    Gated Multimodal Unit (GMU):
    
      h_img: (B, D_img)   from backbone global pool
      h_tab: (B, D_tab)   from tab encoder
      
      gate  = sigmoid(W_g [ h_img || h_tab ])   (B, D_img)
      fused = gate * h_img + (1 - gate) * h_tab_proj
                 how much image       how much tab
    """
    def __init__(
        self,
        img_dim: int,        # e.g. 1536 for SWIN-L, 1280 for EffB1
        tab_dim: int,        # e.g. 64
        head_hidden: int = 256,
    ):
        super().__init__()

        # project tab to same dim as image so gate can mix them
        self.tab_proj = nn.Linear(tab_dim, img_dim)

        # gate network: takes concat of both → scalar per dim
        self.gate_fc = nn.Linear(img_dim + img_dim, img_dim)
        # input:  concat(h_img, h_tab_proj) = (B, 2*img_dim)
        # output: gate vector                = (B, img_dim)

        self.norm = nn.LayerNorm(img_dim)

        self.head = nn.Sequential(
            nn.Linear(img_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, h_img, h_tab):
        # h_img: (B, img_dim)
        # h_tab: (B, tab_dim)

        h_tab_proj = self.tab_proj(h_tab)          # (B, img_dim)

        # compute gate from both modalities
        gate_input = torch.cat([h_img, h_tab_proj], dim=1)  # (B, 2*img_dim)
        gate = torch.sigmoid(self.gate_fc(gate_input))       # (B, img_dim)

        # weighted mix
        fused = gate * h_img + (1 - gate) * h_tab_proj      # (B, img_dim)
        fused = self.norm(fused)

        return self.head(fused)   # (B, 1)
    

 
from src.tab_encoder import build_tab_encoder

#Efficient
class EfficientNetGatedFusion(nn.Module):
    """
    EfficientNet backbone → global average pool → (B, D)
    Tab encoder                                 → (B, tab_hidden)
    GatedFusion                                 → (B, 1)
    """
    def __init__(
        self,
        backbone_name: str,
        img_size: int,
        tab_input_dim: int,
        tab_hidden: int = 64,
        head_hidden: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        tab_encoder_capacity: str = "small",
    ):
        super().__init__()
        extra_kwargs = {}
        # flexible size for Swin 
        if "swin" in backbone_name or "vit" in backbone_name:
            extra_kwargs["img_size"] = img_size
            extra_kwargs["dynamic_img_pad"] = True


        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
             **extra_kwargs,
        )

       


        img_dim = self.backbone.num_features   # 1280 for B1

        self.tab_enc = build_tab_encoder(
            tab_input_dim, tab_hidden, tab_encoder_capacity
        )

        self.fusion = GatedFusion(
            img_dim=img_dim,
            tab_dim=tab_hidden,
            head_hidden=head_hidden,
        )

    def forward(self, img, tab):
      

        h_img = self.backbone(img)     # (B, 1280)
        h_tab = self.tab_enc(tab)       # (B, 64)
        out   = self.fusion(h_img, h_tab)  # (B, 1)
        return out    