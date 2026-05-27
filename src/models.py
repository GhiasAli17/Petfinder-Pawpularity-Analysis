# src/models.py
import torch
import torch.nn as nn
import timm

import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src.tab_encoder import build_tab_encoder



#image only without metadata 
# def build_vision_backbone(name, img_size, mode, head_type="linear", head_hidden=256):
#     """
#     img_size is used for Swin-type models that have a fixed patch embedding size. 
#     """
#     extra_kwargs = {}
#     # flexible size for Swin.
#     if "swin" in name or "vit" in name:
#         extra_kwargs["img_size"] = img_size
#         extra_kwargs["dynamic_img_pad"] = True

#     if mode == "feature":
#         model = timm.create_model(
#             name,
#             pretrained=True,
#             num_classes=0,
#             **extra_kwargs,
#         )
#     elif mode == "regression":
#         model = timm.create_model(
#             name,
#             pretrained=True,
#             num_classes=1,
#             **extra_kwargs,
#         )
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
#     return model
class VisionRegNet(nn.Module):
    def __init__(
        self,
        backbone_name,
        img_size,
        head_type="linear",
        head_hidden=256,
        pretrained=True,
    ):
        super().__init__()

        self.head_type = head_type

        extra_kwargs = {}

        if "swin" in backbone_name or "vit" in backbone_name:
            extra_kwargs["img_size"] = img_size
            extra_kwargs["dynamic_img_pad"] = True

        # EXACT original timm behavior
        if head_type == "linear":

            self.model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=1,
                **extra_kwargs,
            )

        # custom MLP head
        elif head_type == "mlp":

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                **extra_kwargs,
            )

            feat_dim = self.backbone.num_features

            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
           
            # self.head  = nn.Sequential(
            #         nn.Linear(feat_dim, 768),      
            #         nn.ReLU(),
            #         nn.Linear(768, 384),  
            #         nn.ReLU(),
            #         nn.Linear(384, head_hidden),
            #         nn.ReLU(),
            #         nn.Linear(head_hidden, 1))

        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):

        # exact timm classifier behavior
        if self.head_type == "linear":
            return self.model(x)

        # custom mlp head
        feat = self.backbone(x)
        return self.head(feat)

#image only without metadata but with aux heads, which can be one or more  aux heads
class VisionAuxNet(nn.Module):
    def __init__(self, backbone_name, img_size, aux_tasks=None, pretrained=True, head_hidden=256, 
                 head_type="linear", use_saliency=False, binary_aux_tasks=None):
        super().__init__()
        self.aux_tasks = aux_tasks or []
        self.binary_aux_tasks = binary_aux_tasks or []
        self.head_type = head_type
        self.use_saliency = use_saliency

        extra_kwargs = {}
        if "swin" in backbone_name or "vit" in backbone_name:
            extra_kwargs["img_size"] = img_size
            extra_kwargs["dynamic_img_pad"] = True

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            **extra_kwargs,
        )
        feat_dim = self.backbone.num_features
        def make_head():
            if head_type == "linear":
                return nn.Linear(feat_dim, 1)
            elif head_type == "mlp":
                return nn.Sequential(
                    nn.Linear(feat_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, 1),
                )
            
            else:
                raise ValueError(f"Unknown head_type: {head_type}")

        # self.main_head = nn.Linear(feat_dim, 1)
        self.main_head = make_head()

        self.aux_heads = nn.ModuleDict() # for multiple aux heads, e.g. brisque and visibility_ratio
        if "brisque" in self.aux_tasks:
            # self.aux_heads["brisque"] = nn.Linear(feat_dim, 1)
            self.aux_heads["brisque"] = make_head()
        if "visibility_ratio" in self.aux_tasks:
            # self.aux_heads["visibility_ratio"] = nn.Linear(feat_dim, 1)
            self.aux_heads["visibility_ratio"] = make_head()

        for task in self.binary_aux_tasks:
            self.aux_heads[task] = make_head()  # same head structure for binary aux tasks


        self._spatial_features = None
        if self.use_saliency:
            self.backbone.layers[-1].blocks[-1].register_forward_hook(
                lambda m, inp, out: setattr(self, "_spatial_features", out)
            )
    def forward(self, x):
        feat = self.backbone(x)
        out = {"main": self.main_head(feat)}
        for task, head in self.aux_heads.items():
            out[task] = head(feat)

        if self.use_saliency and self._spatial_features is not None:
            out["spatial"] = self._spatial_features    
        return out
    
#Feature concat fusion of image backbone features and tabular data    
class FeatureConcatFusionNet(nn.Module):
    """
    Vision backbone (features only) + tabular MLP encoder + fusion head.
    head_type: "linear" or "mlp".
    """
    def __init__(
        self,
        backbone_name,
        img_size,
        tab_input_dim,
        tab_hidden=64,
        fusion_hidden=256,
        head_type="mlp",
        pretrained=True,
        freeze_backbone=False,
        tab_encoder_capacity="small", # #small or big tab encoder MLP, small is default (2-layer MLP), big is deeper/wider (5-layer)
    ):
        super(FeatureConcatFusionNet, self).__init__()

        self.backbone_name = backbone_name  

        extra_kwargs = {}
        # flexible size for Swin 
        if "swin" in backbone_name or "vit" in backbone_name:
            extra_kwargs["img_size"] = img_size
            extra_kwargs["dynamic_img_pad"] = True

        self.img_model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            **extra_kwargs,
        )
        img_out_dim = self.img_model.num_features
        if freeze_backbone:
            for p in self.img_model.parameters():
                p.requires_grad = False

    
        # self.tab_enc = nn.Sequential(
        #     nn.Linear(tab_input_dim, tab_hidden),
        #     nn.ReLU(),
        #     nn.Linear(tab_hidden, tab_hidden),
        #     nn.ReLU(),
        # )
        self.tab_enc = build_tab_encoder(
        input_dim=tab_input_dim,
        hidden_dim=tab_hidden,
        tab_encoder_capacity=tab_encoder_capacity)
         
        
        tab_out_dim = tab_hidden

        fusion_in = img_out_dim + tab_out_dim
        if head_type == "linear":
            self.head = nn.Linear(fusion_in, 1)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden),
                nn.ReLU(),
                nn.Linear(fusion_hidden, 1),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, img, tab):
        img_feat = self.img_model(img)
        tab_feat = self.tab_enc(tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.head(fused)
        return out


# class TabularMLP(nn.Module):
#     def __init__(self, input_dim, hidden1=64, hidden2=32, out_dim=1):
#         super(TabularMLP, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden1),
#             nn.ReLU(),
#             nn.Linear(hidden1, hidden2),
#             nn.ReLU(),
#             nn.Linear(hidden2, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)
    

